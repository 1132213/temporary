# scripts/run_finetune_instructions.py
# 这个脚本与 run_finetune_alignment.py 非常相似，
# 主要区别在于加载的模型检查点和使用的数据集不同。

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW # Import AdamW from torch.optim
from accelerate import Accelerator
import os
from tqdm import tqdm

# 导入正确的、用于指令微调的数据集
from clgm.data.datasets import InstructionTuningDataset
from clgm.data.data_collators import CausalLMCollator
from clgm.models.clgm_core import CLGM, CLGMConfig
from configs.config import LLM_FINETUNE_CONFIG, PATCH_SIZE

def main():
    # 1. 初始化 Accelerator，同样支持梯度累积
    accelerator = Accelerator(gradient_accumulation_steps=LLM_FINETUNE_CONFIG["gradient_accumulation_steps"])
    config = LLM_FINETUNE_CONFIG
    
    # 2. 加载分词器和模型
    # --- 关键不同点 1: 加载第二阶段训练好的模型 ---
    # 我们从第二阶段对齐后的模型检查点继续训练
    stage2_checkpoint_path = os.path.join(config["clgm_checkpoint_dir"], "stage2_aligned")
    
    if accelerator.is_local_main_process:
        print(f"从阶段二检查点加载模型和分词器: {stage2_checkpoint_path}")
        
    tokenizer = AutoTokenizer.from_pretrained(stage2_checkpoint_path)
    # 再次设置special_tokens_map以确保数据集能正确工作
    tokenizer.special_tokens_map = {
        "text_start": "<text_start>", "text_end": "<text_end>",
        "ts_start": "<ts_start>", "ts_end": "<ts_end>",
        "instruction": "<instruction>", "end_instruction": "</instruction>",
    }
    
    # 使用 .from_pretrained 直接从文件夹加载整个CLGM模型
    model = CLGM.from_pretrained(stage2_checkpoint_path)
    # VQ-VAE 已经在 CLGM 类内部被正确处理（加载并冻结），无需再次手动加载
    
    # 3. 准备数据
    # --- 关键不同点 2: 使用指令微调数据集 ---
    if accelerator.is_local_main_process:
        print("正在加载指令微调数据集...")
    dataset = InstructionTuningDataset(
        data_path=config["instruction_data_path"],
        vq_vae=model.get_vq_vae(), # VQ-VAE仍然需要用于处理输入的时间序列
        tokenizer=tokenizer,
        patch_size=PATCH_SIZE
    )
    collator = CausalLMCollator(tokenizer) # 整理器可以复用
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=collator, shuffle=True)
    
    # 4. 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    num_training_steps = (len(dataloader) // accelerator.gradient_accumulation_steps) * config["num_epochs_instruction"]
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    # 5. 使用 Accelerator 准备
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    # 6. 训练循环
    if accelerator.is_local_main_process:
        print("开始阶段三：指令微调...")
        
    for epoch in range(config["num_epochs_instruction"]):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Stage 3 - Epoch {epoch+1}", disable=not accelerator.is_local_main_process)
        
        for batch in progress_bar:
            # 梯度累积上下文
            with accelerator.accumulate(model):
                # 前向传播。因为数据集的标签已经处理好（输入部分为-100），
                # 这里的损失将只在“输出”部分计算。
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_local_main_process:
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    # 7. 保存最终模型
    if accelerator.is_local_main_process:
        print("指令微调完成，正在保存最终模型...")
        save_path = os.path.join(config["clgm_checkpoint_dir"], "stage3_final")
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        # 保存最终的模型和分词器，准备用于推理
        unwrapped_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"阶段三最终模型已保存至: {save_path}")

if __name__ == "__main__":
    main()