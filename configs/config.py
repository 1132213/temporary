# configs/config.py
import torch
import os
# --- LLM 基础模型配置 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 然后将本地模型的相对路径与项目根目录拼接起来
LOCAL_MODEL_FOLDER = "local_models/Llama-3.2-1B-Instruct"
LLM_MODEL_NAME = os.path.join(PROJECT_ROOT, LOCAL_MODEL_FOLDER)
# LLM处理的最大序列长度
MAX_SEQ_LENGTH = 4096

# --- VQ-VAE 时间序列分词器配置 ---

# 输入时间序列的切片（patch）大小。每个patch会被编码成一个离散的token。
PATCH_SIZE = 16
# VQ-VAE的码本（codebook）大小。K=256 在重构质量和LLM处理复杂性之间取得了良好平衡。
VQ_CODEBOOK_SIZE = 256
# 每个码本向量（即“时间模体”）的维度。
VQ_EMBEDDING_DIM = 64
# VQ-VAE 编码器/解码器中的残差层数量。
VQ_NUM_LAYERS = 2
VQ_COMPRESSION_FACTOR = 4
# VQ-VAE 量化器中的 commitment loss 权重，用于稳定码本的学习。
VQ_COMMITMENT_COST = 0.25

# --- 特殊词元（Special Tokens）配置 ---
# 为LLM词汇表新增的特殊词元，用于在序列中区分不同模态（文本/时间序列）和结构（指令/内容）。
TS_MOTIF_PREFIX = "<ts_motif_" # 时间序列token的前缀
SPECIAL_TOKENS = {
    "ts_start": "<ts_start>",           # 时间序列开始
    "ts_end": "<ts_end>",             # 时间序列结束
    "text_start": "<text_start>",       # 文本开始
    "text_end": "<text_end>",         # 文本结束
    "instruction": "<instruction>",     # 指令开始
    "end_instruction": "</instruction>", # 指令结束
    "pad_token": "<pad>",             # 填充词元
}

# --- 训练配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 阶段一: VQ-VAE 预训练配置
VQ_VAE_TRAIN_CONFIG = {
    "data_dir": "./data/unsupervised_ts/",         # 无监督时间序列数据目录
    "checkpoint_dir": "./checkpoints/vq_vae/",     # VQ-VAE模型权重保存目录
    "learning_rate": 1e-4,                         # 学习率
    "batch_size": 128,                             # 批量大小
    "num_epochs": 50,                              # 训练轮数
    "log_steps": 100,                              # 每隔多少步打印一次日志
}
STAGE_2A_LORA_CONFIG = {
    "data_path": "./data/final_training_data_merge_cleaned.jsonl",
    "base_model_path": LLM_MODEL_NAME, # 从原始的LLM开始
    "vq_vae_checkpoint_path": "./checkpoints/vq_vae/best_model.pth",
    "tokenizer_path": "./checkpoints/clgm/tokenizer/",
    "output_dir": "./checkpoints/clgm/stage2a_lora_adapters/", # 只保存LoRA适配器
    
    "learning_rate": 2e-4,          # LoRA训练可以使用稍大的学习率
    "batch_size": 4,
    "num_epochs": 5,                # 5个轮次足以让LoRA适配器学会新知识
    "gradient_accumulation_steps": 16,
    "num_warmup_steps": 200,
    "weight_decay": 0.01,
    "log_steps": 20,
    
    # LoRA 专属配置
    "lora_r": 16,                   # LoRA的秩
    "lora_alpha": 32,               # LoRA的alpha值
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"], # 在这些层上应用LoRA
}

# --- Stage 2B ---
STAGE_2B_FULL_FINETUNE_CONFIG = {
    "data_path": "./data/final_training_data_merge_cleaned.jsonl",
    # 从原始LLM和训练好的LoRA适配器开始
    "base_model_path": LLM_MODEL_NAME, 
    "lora_adapter_path": STAGE_2A_LORA_CONFIG["output_dir"], # 加载上一阶段的成果
    "vq_vae_checkpoint_path": "./checkpoints/vq_vae/best_model.pth",
    "tokenizer_path": "./checkpoints/clgm/tokenizer/",
    "output_dir": "./checkpoints/clgm/stage2b_full_finetuned/", # 保存最终的、完整的模型
    
    # 核心训练参数
    "learning_rate": 2e-7,          # 使用一个极小的学习率
    "batch_size": 2,                # 可以使用更小的批量，因为所有参数都可训练，更耗显存
    "num_epochs": 2,                # 2-3个轮次的精修通常就足够了
    "gradient_accumulation_steps": 32, # 保持有效批量大小为 2*2*32=128
    "num_warmup_steps": 50,         # 一个较短的预热期
    "weight_decay": 0.0,            # 全量微调时通常不使用权重衰减
    "log_steps": 20,
}

# 阶段二 & 三: LLM 微调配置
LLM_FINETUNE_CONFIG = {
    "paired_data_path": "./data/final_training_data_merge_cleaned.jsonl",      # 阶段二（对齐）数据路径
    # "paired_data_path": "./data/qwen_generated_dataset_en/alignment_data_en.jsonl", 
    # "paired_data_path": "./data/qwen_generated_dataset_en/alignment_data_small.jsonl", 
    "instruction_data_path": "./data/sft_split_advanced/sft_train_data_en_cleaned.jsonl",# 阶段三（指令）数据路径
    "vq_vae_checkpoint_path": "./checkpoints/vq_vae/best_model.pth",      # 预训练好的VQ-VAE权重路径
    "clgm_checkpoint_dir": "./checkpoints/clgm/",                         # CLGM模型权重保存目录
    "learning_rate": 1.5e-5,                                                # LLM微调的学习率，通常较小
    "batch_size": 4,                                                      # LLM微调通常需要更小的批量以适应显存
    # "num_epochs_alignment": 200,                                            # 阶段二训练轮数
    "num_epochs_alignment": 4,                                            # 阶段二训练轮数
    "num_epochs_instruction": 5,                                          # 阶段三训练轮数
    "gradient_accumulation_steps": 16,                                     # 梯度累积步数，用于模拟更大的批量
    # "gradient_accumulation_steps": 1,                                     # 梯度累积步数，用于模拟更大的批量
    "log_steps": 5,                                                      # 日志打印频率
    "num_warmup_steps": 400,                                               # 学习率预热步数
    "weight_decay": 0.01,                                                # 权重衰减
}

# --- 推理配置 ---
INFERENCE_CONFIG = {
    "clgm_checkpoint_path": "./checkpoints/clgm/stage3_final/pytorch_model.bin", # 最终CLGM模型权重路径
    "extended_tokenizer_path": "./checkpoints/clgm/tokenizer/",                 # 扩展后的分词器路径
    "vq_vae_checkpoint_path": VQ_VAE_TRAIN_CONFIG["checkpoint_dir"] + "best_model.pth", # VQ-VAE权重路径
}