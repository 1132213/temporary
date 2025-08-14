# scripts/run_finetune_instructions.py
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW 
from accelerate import Accelerator
import os
from tqdm import tqdm

from clgm.data.datasets import InstructionTuningDataset
from clgm.data.data_collators import CausalLMCollator
from clgm.models.clgm_core import CLGM, CLGMConfig
from configs.config import LLM_FINETUNE_CONFIG, PATCH_SIZE
from safetensors.torch import load_file 

def main():
    accelerator = Accelerator(gradient_accumulation_steps=LLM_FINETUNE_CONFIG["gradient_accumulation_steps"])
    config = LLM_FINETUNE_CONFIG
    
    stage2_checkpoint_path = os.path.join(config["clgm_checkpoint_dir"], "stage2_aligned")
    
    if accelerator.is_local_main_process:
        print(f"从阶段二检查点加载模型和分词器: {stage2_checkpoint_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(stage2_checkpoint_path)
    
    clgm_config = CLGMConfig.from_pretrained(stage2_checkpoint_path)
    model = CLGM(clgm_config)
    
    if accelerator.is_local_main_process:
        print(f"将模型词嵌入层大小调整为: {len(tokenizer)}")

    model.llm.resize_token_embeddings(len(tokenizer))
    state_dict_path = os.path.join(stage2_checkpoint_path, 'model.safetensors') 
    if not os.path.exists(state_dict_path):
        state_dict_path = os.path.join(stage2_checkpoint_path, 'pytorch_model.bin')

    if accelerator.is_local_main_process:
        print(f"正在从 {state_dict_path} 加载权重...")

    if state_dict_path.endswith(".safetensors"):
        state_dict = load_file(state_dict_path, device="cpu")
    else:
        state_dict = torch.load(state_dict_path, map_location="cpu")

    model.load_state_dict(state_dict, strict=False)
    
    if accelerator.is_local_main_process:
        print("模型和权重加载成功！")
    
    if accelerator.is_local_main_process:
        print("正在加载指令微调数据集...")
    dataset = InstructionTuningDataset(
        data_path=config["instruction_data_path"],
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
    num_training_steps = (len(dataloader) // accelerator.gradient_accumulation_steps) * config["num_epochs_instruction"]
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.get("num_warmup_steps", 0),
        num_training_steps=num_training_steps
    )
    
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    if accelerator.is_local_main_process:
        print("开始阶段三：指令微调...")
        
    for epoch in range(config["num_epochs_instruction"]):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Stage 3 - Epoch {epoch+1}", disable=not accelerator.is_local_main_process)
        
        for batch in progress_bar:
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
    
    if accelerator.is_local_main_process:
        print("指令微调完成，正在保存最终模型...")
        save_path = os.path.join(config["clgm_checkpoint_dir"], "stage3_final")
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"阶段三最终模型已保存至: {save_path}")

if __name__ == "__main__":
    main()