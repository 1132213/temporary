# scripts/run_finetune_alignment.py (MODIFIED for Hyperparameter Tuning)
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from accelerate import Accelerator
import os
import argparse # 1. 导入 argparse
from tqdm import tqdm

from clgm.models.clgm_core import CLGM
from clgm.data.datasets import PairedTSTextDataset
from clgm.data.data_collators import CausalLMCollator
from configs.config import LLM_FINETUNE_CONFIG, PATCH_SIZE

def main():
    # 2. --- 定义命令行参数 ---
    parser = argparse.ArgumentParser(description="Finetune CLGM with custom hyperparameters.")
    parser.add_argument("--gpu_id", type=str, required=True, help="The CUDA device ID to use (e.g., '0', '1').")
    parser.add_argument("--learning_rate", type=float, default=LLM_FINETUNE_CONFIG["learning_rate"], help="Learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, default=LLM_FINETUNE_CONFIG["batch_size"], help="Batch size per GPU.")
    parser.add_argument("--num_epochs", type=int, default=LLM_FINETUNE_CONFIG["num_epochs_alignment"], help="Number of training epochs.")
    parser.add_argument("--exp_name", type=str, required=True, help="A unique name for this experiment run.")
    
    args = parser.parse_args()

    # 设置此进程可见的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    # 3. --- 创建唯一的实验目录 ---
    # 这会像这样创建目录: ./checkpoints/clgm/lr_2e-5_bs_4_epochs_3/
    base_checkpoint_dir = LLM_FINETUNE_CONFIG["clgm_checkpoint_dir"]
    experiment_dir = os.path.join(base_checkpoint_dir, args.exp_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"--- [Experiment: {args.exp_name} on GPU {args.gpu_id}] ---")
    print(f"Hyperparameters: LR={args.learning_rate}, BS={args.batch_size}, Epochs={args.num_epochs}")
    print(f"Checkpoints will be saved to: {experiment_dir}")

    # 1. 初始化 Accelerator
    accelerator = Accelerator(gradient_accumulation_steps=LLM_FINETUNE_CONFIG["gradient_accumulation_steps"])
    config = LLM_FINETUNE_CONFIG
    
    # 2. 加载分词器
    tokenizer_path = os.path.join(config["clgm_checkpoint_dir"], "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 3. 初始化 CLGM 模型
    if accelerator.is_local_main_process:
        print("Initializing CLGM model...")
    clgm_config = CLGMConfig()
    model = CLGM(clgm_config)
    model.initialize_from_pretained(
        vq_vae_checkpoint_path=config["vq_vae_checkpoint_path"],
        tokenizer=tokenizer
    )
    
    # 4. 准备数据
    if accelerator.is_local_main_process:
        print("Loading alignment dataset...")
    dataset = PairedTSTextDataset(
        data_path=config["paired_data_path"],
        vq_vae=model.get_vq_vae(),
        tokenizer=tokenizer,
        patch_size=PATCH_SIZE
    )
    collator = CausalLMCollator(tokenizer)
    # 4. --- 使用命令行的 batch_size ---
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=True)
    
    # 5. 设置优化器和学习率调度器
    # 4. --- 使用命令行的 learning_rate 和 num_epochs ---
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = (len(dataloader) // accelerator.gradient_accumulation_steps) * args.num_epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    # 6. 使用 accelerator.prepare()
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    # 7. 训练循环
    if accelerator.is_local_main_process:
        print("Starting Stage 2: Multimodal Alignment Finetuning...")
        
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs} on GPU {args.gpu_id}", disable=not accelerator.is_local_main_process)
        
        for batch in progress_bar:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                
                accelerator.backward(loss)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_local_main_process:
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    # 8. 保存最终模型到实验专属目录
    if accelerator.is_local_main_process:
        print("Training complete, saving model...")
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        # 5. --- 保存到唯一的实验目录 ---
        unwrapped_model.save_pretrained(experiment_dir)
        tokenizer.save_pretrained(experiment_dir)
        print(f"Stage 2 aligned model for experiment '{args.exp_name}' saved to: {experiment_dir}")

if __name__ == "__main__":
    main()