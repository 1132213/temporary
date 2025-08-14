# scripts/run_finetune_alignment.py
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from accelerate import Accelerator 
import os
import argparse 
from tqdm import tqdm
import matplotlib.pyplot as plt 

from clgm.models.clgm_core import CLGM, CLGMConfig
from clgm.data.datasets import PairedTSTextDataset
from clgm.data.data_collators import CausalLMCollator
from configs.config import LLM_FINETUNE_CONFIG, PATCH_SIZE

def main():
    parser = argparse.ArgumentParser(description="Finetune CLGM for multimodal alignment.")
    parser.add_argument("--gpu_id", type=str, default=None, help="指定要使用的GPU ID (例如 '0', '1', 或 '0,1')")
    args = parser.parse_args()

    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        print(f"用户指定使用GPU: {args.gpu_id}")

    accelerator = Accelerator(gradient_accumulation_steps=LLM_FINETUNE_CONFIG["gradient_accumulation_steps"])
    config = LLM_FINETUNE_CONFIG
    
    print(f"Accelerator检测到的设备为: {accelerator.device}")
    
    tokenizer_path = os.path.join(config["clgm_checkpoint_dir"], "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if accelerator.is_local_main_process:
        print("正在初始化 CLGM 模型...")
    clgm_config = CLGMConfig()
    model = CLGM(clgm_config)
    model.initialize_from_pretained(
        vq_vae_checkpoint_path=config["vq_vae_checkpoint_path"],
        tokenizer=tokenizer
    )
    
    if accelerator.is_local_main_process:
        print("正在加载对齐数据集...")
    dataset = PairedTSTextDataset(
        data_path=config["paired_data_path"],
        vq_vae=model.get_vq_vae(), 
        tokenizer=tokenizer,
        patch_size=PATCH_SIZE
    )
    collator = CausalLMCollator(tokenizer)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=collator, shuffle=True)
    
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0)
        )
    num_training_steps = (len(dataloader) // accelerator.gradient_accumulation_steps) * config["num_epochs_alignment"]
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.get("num_warmup_steps", 0), 
        num_training_steps=num_training_steps
    )
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    all_losses = []
    global_step = 0
    
    if accelerator.is_local_main_process:
        print("开始阶段二：多模态对齐微调...")
        
    for epoch in range(config["num_epochs_alignment"]):
        model.train()
        
        accumulated_loss_for_log = 0.0
        accumulation_step_counter = 0 

        progress_bar = tqdm(dataloader, desc=f"Stage 2 - Epoch {epoch+1}", disable=not accelerator.is_local_main_process)
        
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

            if accelerator.is_local_main_process:
                if accelerator.sync_gradients:
                    global_step += 1
                    log_steps = config.get("log_steps", 5)
                    
                    if global_step > 0 and global_step % log_steps == 0:
                        if accumulation_step_counter > 0:
                            avg_loss = accumulated_loss_for_log / (accumulation_step_counter)
                            if avg_loss < 100:
                                all_losses.append(avg_loss)
                            
                            progress_bar.set_postfix({"Avg Loss per Sample": f"{avg_loss:.4f}"})
                        
                        accumulated_loss_for_log = 0.0
                        accumulation_step_counter = 0
    
    if accelerator.is_local_main_process:
        print("训练完成，正在保存模型...")
        # save_path = os.path.join(config["clgm_checkpoint_dir"], "stage2_aligned")
        # save_path = os.path.join(config["clgm_checkpoint_dir"], "stage2_aligned2")
        save_path = os.path.join(config["clgm_checkpoint_dir"], "stage2_aligned_1.5e-5_4")
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"阶段二对齐模型已保存至: {save_path}")
        
        print("Loss plot...")
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(all_losses)), all_losses)
        plt.title("Training Loss Curve (Stage 2 Alignment)")
        plt.xlabel(f"Steps (x{log_steps})")
        plt.ylabel("Average Loss")
        plt.grid(True)
        
        loss_plot_path = os.path.join(save_path, "training_loss_curve.png")
        plt.savefig(loss_plot_path)
        print(f"Loss曲线图已保存至: {loss_plot_path}")

if __name__ == "__main__":
    main()