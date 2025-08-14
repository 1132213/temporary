# scripts/run_finetune_alignment.py (最终版 - 支持分阶段微调)
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from accelerate import Accelerator
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from peft import get_peft_model, LoraConfig, PeftModel
except ImportError:
    print("错误：PEFT 库未安装。请运行 'pip install peft' 进行安装。")
    exit()

from clgm.models.clgm_core import CLGM, CLGMConfig
from clgm.data.datasets import PairedTSTextDataset
from clgm.data.data_collators import CausalLMCollator
from configs.config import STAGE_2A_LORA_CONFIG, STAGE_2B_FULL_FINETUNE_CONFIG, PATCH_SIZE

def main():
    parser = argparse.ArgumentParser(description="Staged finetuning for multimodal alignment.")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=['2a', '2b'],
        help="Specify the finetuning stage to run: '2a' for LoRA, '2b' for full finetuning."
    )
    args = parser.parse_args()

    # 根据选择的阶段，加载对应的配置
    if args.stage == '2a':
        config = STAGE_2A_LORA_CONFIG
        print("--- 启动 Stage 2A: LoRA 安全注入微调 ---")
    else: # args.stage == '2b'
        config = STAGE_2B_FULL_FINETUNE_CONFIG
        print("--- 启动 Stage 2B: 全量深度融合微调 ---")

    # --- 后续流程与之前类似，但会根据config执行不同逻辑 ---
    accelerator = Accelerator(gradient_accumulation_steps=config["gradient_accumulation_steps"])
    print(f"脚本启动，Accelerator检测到的设备为: {accelerator.device}")

    # 1. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])

    # 2. 初始化 CLGM 模型
    if accelerator.is_local_main_process:
        print("正在初始化 CLGM 模型...")
    clgm_config = CLGMConfig.from_pretrained(config["base_model_path"], local_files_only=True)
    model = CLGM(clgm_config)
    
    # 调用核心初始化方法 (加载VQ-VAE, 调整词嵌入)
    model.initialize_from_pretained(
        vq_vae_checkpoint_path=config["vq_vae_checkpoint_path"],
        tokenizer=tokenizer
    )

    # --- 核心逻辑分支 ---
    if args.stage == '2a':
        if accelerator.is_local_main_process:
            print("--- Stage 2A: 正在应用LoRA适配器... ---")
        lora_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            target_modules=config["lora_target_modules"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        if accelerator.is_local_main_process:
            model.print_trainable_parameters()
    else: # args.stage == '2b'
        if accelerator.is_local_main_process:
            print(f"--- Stage 2B: 正在从 '{config['lora_adapter_path']}' 加载并合并LoRA适配器... ---")
        # 加载LoRA适配器权重并合并到主模型中
        model = PeftModel.from_pretrained(model, config['lora_adapter_path'])
        model = model.merge_and_unload()
        if accelerator.is_local_main_process:
            print("--- LoRA适配器合并完成，模型已准备好进行全量微调。 ---")

    # 3. 准备数据
    dataset = PairedTSTextDataset(
        data_path=config["data_path"], vq_vae=model.get_vq_vae(), tokenizer=tokenizer, patch_size=PATCH_SIZE
    )
    collator = CausalLMCollator(tokenizer)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=collator, shuffle=True)
    
    # 4. 设置优化器和调度器
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config.get("weight_decay", 0.0))
    num_training_steps = (len(dataloader) // accelerator.gradient_accumulation_steps) * config["num_epochs"]
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.get("num_warmup_steps", 0), num_training_steps=num_training_steps
    )
    
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    
    # --- 训练循环 (逻辑基本保持不变) ---
    all_losses = []
    global_step = 0
    if accelerator.is_local_main_process:
        print(f"--- 开始 Stage {args.stage} 训练... ---")
        
    for epoch in range(config["num_epochs"]):
        model.train()
        accumulated_loss_for_log, accumulation_step_counter = 0.0, 0
        progress_bar = tqdm(dataloader, desc=f"Stage {args.stage} - Epoch {epoch+1}", disable=not accelerator.is_local_main_process)
        
        for batch in progress_bar:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accumulated_loss_for_log += loss.item()
                accumulation_step_counter += 1
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_local_main_process and accelerator.sync_gradients:
                global_step += 1
                log_steps = config.get("log_steps", 20)
                if global_step > 0 and global_step % log_steps == 0 and accumulation_step_counter > 0:
                    avg_loss = accumulated_loss_for_log / accumulation_step_counter
                    all_losses.append(avg_loss)
                    progress_bar.set_postfix({"Avg Loss": f"{avg_loss:.4f}"})
                    accumulated_loss_for_log, accumulation_step_counter = 0.0, 0
    
    # --- 保存模型 ---
    if accelerator.is_local_main_process:
        print("训练完成，正在保存模型...")
        save_path = config["output_dir"]
        os.makedirs(save_path, exist_ok=True)
        
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Stage {args.stage} 模型已保存至: {save_path}")

        # 绘制并保存Loss曲线
        if all_losses:
            plt.figure(figsize=(12, 6))
            plt.plot(all_losses)
            plt.title(f"Training Loss Curve (Stage {args.stage})")
            plt.xlabel(f"Log Points (x{config.get('log_steps', 20)} steps)")
            plt.ylabel("Average Loss per Sample")
            plt.grid(True)
            plt.savefig(os.path.join(save_path, "training_loss_curve.png"))
            print(f"Loss曲线图已保存至: {os.path.join(save_path, 'training_loss_curve.png')}")

if __name__ == "__main__":
    main()