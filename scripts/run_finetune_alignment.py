# scripts/run_finetune_alignment.py
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW # Import AdamW from torch.optim
from accelerate import Accelerator # 导入Hugging Face的分布式训练神器
import os
import argparse  # 导入 argparse 模块
from tqdm import tqdm

from clgm.models.clgm_core import CLGM, CLGMConfig
from clgm.data.datasets import PairedTSTextDataset
from clgm.data.data_collators import CausalLMCollator
from configs.config import LLM_FINETUNE_CONFIG, PATCH_SIZE

def main():
    # 0. 添加命令行参数解析器以选择GPU
    parser = argparse.ArgumentParser(description="Finetune CLGM for multimodal alignment.")
    parser.add_argument("--gpu_id", type=str, default=None, help="指定要使用的GPU ID (例如 '0', '1', 或 '0,1')")
    args = parser.parse_args()

    # 在初始化任何PyTorch或Accelerator组件之前，设置CUDA_VISIBLE_DEVICES
    # 这是控制GPU使用的标准方法
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        print(f"用户指定使用GPU: {args.gpu_id}")

    # 1. 初始化 Accelerator
    # Accelerator 会自动检测通过 CUDA_VISIBLE_DEVICES 设置的环境（单GPU, 多GPU等）
    # 并相应地配置好分布式训练所需的一切。
    accelerator = Accelerator(gradient_accumulation_steps=LLM_FINETUNE_CONFIG["gradient_accumulation_steps"])
    config = LLM_FINETUNE_CONFIG
    
    print(f"脚本启动，Accelerator检测到的设备为: {accelerator.device}")
    
    # 2. 加载分词器
    # 必须加载我们在第一步创建的、扩展了<ts_motif>等词元的分词器
    tokenizer_path = os.path.join(config["clgm_checkpoint_dir"], "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 为数据集类添加特殊词元映射，以便它能正确地找到<ts_start>等token的ID
    # 这一步也可以在分词器保存前就设置好，但在这里设置更明确
    # tokenizer.special_tokens_map = {
    #     "text_start": "<text_start>", "text_end": "<text_end>",
    #     "ts_start": "<ts_start>", "ts_end": "<ts_end>",
    #     "instruction": "<instruction>", "end_instruction": "</instruction>",
    # }

    # 3. 初始化 CLGM 模型
    # 只在主进程上打印信息，避免多进程下的信息轰炸
    if accelerator.is_local_main_process:
        print("正在初始化 CLGM 模型...")
    clgm_config = CLGMConfig()
    model = CLGM(clgm_config)
    # 调用核心初始化方法，加载VQ-VAE权重并初始化词嵌入
    model.initialize_from_pretained(
        vq_vae_checkpoint_path=config["vq_vae_checkpoint_path"],
        tokenizer=tokenizer
    )
    
    # 4. 准备数据
    if accelerator.is_local_main_process:
        print("正在加载对齐数据集...")
    dataset = PairedTSTextDataset(
        data_path=config["paired_data_path"],
        vq_vae=model.get_vq_vae(), # 将冻结的VQ-VAE传入数据集用于实时分词
        tokenizer=tokenizer,
        patch_size=PATCH_SIZE
    )
    collator = CausalLMCollator(tokenizer)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=collator, shuffle=True)
    
    # 5. 设置优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    num_training_steps = (len(dataloader) // accelerator.gradient_accumulation_steps) * config["num_epochs_alignment"]
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    # 6. 使用 accelerator.prepare() 包装所有组件
    # 这是关键步骤！Accelerator会处理所有与分布式相关的设置。
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    # 7. 训练循环
    if accelerator.is_local_main_process:
        print("开始阶段二：多模态对齐微调...")
        
    for epoch in range(config["num_epochs_alignment"]):
        model.train()
        total_loss = 0
        # tqdm只在主进程上显示，避免混乱
        progress_bar = tqdm(dataloader, desc=f"Stage 2 - Epoch {epoch+1}", disable=not accelerator.is_local_main_process)
        
        for batch in progress_bar:
            # 使用 `accelerator.accumulate` 上下文管理器来自动处理梯度累积
            with accelerator.accumulate(model):
                # 前向传播
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                
                # 反向传播。accelerator.backward 会自动处理分布式环境下的梯度同步。
                accelerator.backward(loss)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_local_main_process:
                # 更新进度条上的损失显示
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    # 8. 保存最终模型
    if accelerator.is_local_main_process:
        print("训练完成，正在保存模型...")
        # save_path = os.path.join(config["clgm_checkpoint_dir"], "stage2_aligned")
        save_path = os.path.join(config["clgm_checkpoint_dir"], "stage2_aligned2")
        # accelerator.wait_for_everyone() 确保所有进程都完成了训练
        accelerator.wait_for_everyone()
        # accelerator.unwrap_model() 获取原始的、未被包装的模型
        unwrapped_model = accelerator.unwrap_model(model)
        # 使用Hugging Face的save_pretrained方法保存模型和配置
        unwrapped_model.save_pretrained(save_path)
        # 同时保存分词器到同一目录
        tokenizer.save_pretrained(save_path)
        print(f"阶段二对齐模型已保存至: {save_path}")

if __name__ == "__main__":
    main()