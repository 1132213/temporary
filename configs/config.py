# configs/config.py
import torch

# --- LLM 基础模型配置 ---
# 用户指定的LLM基础模型。推荐使用计算高效且能力强大的模型。
LLM_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
# LLM处理的最大序列长度
MAX_SEQ_LENGTH = 4096

# --- VQ-VAE 时间序列分词器配置 ---
# 这些参数主要参考了 TOTEM 和 "Language of Time" 论文中的成功实验配置

# 输入时间序列的切片（patch）大小。每个patch会被编码成一个离散的token。
PATCH_SIZE = 16
# VQ-VAE的码本（codebook）大小。K=256 在重构质量和LLM处理复杂性之间取得了良好平衡。
VQ_CODEBOOK_SIZE = 256
# 每个码本向量（即“时间模体”）的维度。
VQ_EMBEDDING_DIM = 64
# VQ-VAE 编码器/解码器中的残差层数量。
VQ_NUM_LAYERS = 2
# VQ-VAE 编码器中卷积层的压缩因子，这里未使用，但在模块定义中存在。
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
# 自动检测可用的计算设备（优先使用GPU）
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

# 阶段二 & 三: LLM 微调配置
LLM_FINETUNE_CONFIG = {
    "paired_data_path": "./data/paired_ts_text/alignment_data.jsonl",      # 阶段二（对齐）数据路径
    "instruction_data_path": "./data/paired_ts_text/instruction_data.jsonl",# 阶段三（指令）数据路径
    "vq_vae_checkpoint_path": "./checkpoints/vq_vae/best_model.pth",      # 预训练好的VQ-VAE权重路径
    "clgm_checkpoint_dir": "./checkpoints/clgm/",                         # CLGM模型权重保存目录
    "learning_rate": 2e-5,                                                # LLM微调的学习率，通常较小
    "batch_size": 8,                                                      # LLM微调通常需要更小的批量以适应显存
    "num_epochs_alignment": 3,                                            # 阶段二训练轮数
    "num_epochs_instruction": 3,                                          # 阶段三训练轮数
    "gradient_accumulation_steps": 4,                                     # 梯度累积步数，用于模拟更大的批量
    "log_steps": 20,                                                      # 日志打印频率
}

# --- 推理配置 ---
INFERENCE_CONFIG = {
    "clgm_checkpoint_path": "./checkpoints/clgm/stage3_final/pytorch_model.bin", # 最终CLGM模型权重路径
    "extended_tokenizer_path": "./checkpoints/clgm/tokenizer/",                 # 扩展后的分词器路径
    "vq_vae_checkpoint_path": VQ_VAE_TRAIN_CONFIG["checkpoint_dir"] + "best_model.pth", # VQ-VAE权重路径
}