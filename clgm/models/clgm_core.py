# clgm/models/clgm_core.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PretrainedConfig
from typing import Optional, Tuple

from clgm.models.vq_vae import VQVAE
from configs.config import (
    LLM_MODEL_NAME, VQ_EMBEDDING_DIM, VQ_CODEBOOK_SIZE, TS_MOTIF_PREFIX,
    VQ_NUM_LAYERS # 导入缺失的配置
)

class CLGMConfig(PretrainedConfig):
    """
    为 CLGM 模型自定义的 Hugging Face 配置类。
    这使得我们的模型可以利用 save_pretrained 和 from_pretrained 等便利功能。
    """
    model_type = "clgm"

    def __init__(self, llm_model_name=LLM_MODEL_NAME, vq_vae_config=None, **kwargs):
        self.llm_model_name = llm_model_name
        # 存储 VQ-VAE 的配置，如果未提供，则使用默认值
        self.vq_vae_config = vq_vae_config if vq_vae_config is not None else {
            "embedding_dim": VQ_EMBEDDING_DIM,
            "num_embeddings": VQ_CODEBOOK_SIZE,
            "num_residual_layers": VQ_NUM_LAYERS,
        }
        super().__init__(**kwargs)

class CLGM(PreTrainedModel):
    """
    时序-语言生成模型 (Chrono-Linguistic Generative Model, CLGM)。
    该模型将一个冻结的、预训练的 VQ-VAE（用于时间序列分词）与一个
    大型语言模型（LLM）骨干相结合，以实现多模态的推理和生成。
    """
    config_class = CLGMConfig

    def __init__(self, config: CLGMConfig):
        super().__init__(config)

        # 1. 加载冻结的 VQ-VAE 模型
        # **修正**: 从配置中安全地获取参数，而不是直接解包字典。
        self.vq_vae = VQVAE(
            embedding_dim=config.vq_vae_config.get("embedding_dim", VQ_EMBEDDING_DIM),
            num_embeddings=config.vq_vae_config.get("num_embeddings", VQ_CODEBOOK_SIZE),
            num_residual_layers=config.vq_vae_config.get("num_residual_layers", VQ_NUM_LAYERS)
        )

        # 2. 加载 LLM 骨干
        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_model_name)

        # 3. 创建一个投影层
        # 该层的作用是将 VQ-VAE 码本向量的维度 (embedding_dim) 映射到 LLM 的隐藏层维度 (hidden_size)。
        llm_hidden_size = self.llm.config.hidden_size
        self.ts_projection = nn.Linear(VQ_EMBEDDING_DIM, llm_hidden_size, bias=False)

    def initialize_from_pretained(self, vq_vae_checkpoint_path: str, tokenizer: AutoTokenizer):
        """
        从预训练的权重和扩展后的分词器来初始化模型组件。
        这个方法应该在模型实例化之后、训练开始之前调用。
        
        Args:
            vq_vae_checkpoint_path (str): 预训练的 VQ-VAE 模型权重文件路径。
            tokenizer (AutoTokenizer): 已经添加了新词元的扩展分词器。
        """
        # 加载 VQ-VAE 的权重并将其完全冻结，因为它只用作编码器
        self.vq_vae.load_state_dict(torch.load(vq_vae_checkpoint_path, map_location='cpu')['model_state_dict'])
        for param in self.vq_vae.parameters():
            param.requires_grad = False
        self.vq_vae.eval()
        print("已加载并冻结 VQ-VAE 权重。")

        # 调整 LLM 的词嵌入矩阵大小，以匹配包含了新（时间序列）词元的分词器
        self.llm.resize_token_embeddings(len(tokenizer))
        print(f"已将 LLM 词嵌入矩阵大小调整为 {len(tokenizer)}。")

        # 使用一种有原则的方式（而不是随机）来初始化新时间序列词元的嵌入向量
        self._initialize_temporal_embeddings(tokenizer)

    def _initialize_temporal_embeddings(self, tokenizer: AutoTokenizer):
        """
        使用投影层来初始化新的时间序列模体词元的嵌入向量。
        这种方法将 VQ-VAE 码本中学到的结构性知识注入到 LLM 的嵌入空间中。
        """
        with torch.no_grad():
            # 获取 LLM 的输入嵌入层和 VQ-VAE 的码本权重
            llm_embeddings = self.llm.get_input_embeddings()
            vq_codebook = self.vq_vae.vq._embedding.weight

            # **修正**: 使用一个已知的参数来安全地获取设备信息
            device = self.llm.device
            
            for i in range(VQ_CODEBOOK_SIZE):
                token_str = f"{TS_MOTIF_PREFIX}{i}>"
                token_id = tokenizer.convert_tokens_to_ids(token_str)

                if token_id == tokenizer.unk_token_id:
                    print(f"警告: 词元 '{token_str}' 在分词器中未找到。")
                    continue

                # 将 VQ-VAE 码本向量投影到 LLM 的隐藏空间
                projected_embedding = self.ts_projection(vq_codebook[i].to(device))

                # 将投影后的嵌入向量赋值给 LLM 嵌入矩阵中对应的新词元位置
                llm_embeddings.weight.data[token_id] = projected_embedding

        print("已使用投影层初始化时间序列词元的嵌入。")

    def get_vq_vae(self) -> VQVAE:
        """返回冻结的 VQ-VAE 模型实例，以便在数据集中使用。"""
        return self.vq_vae

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple:
        """
        CLGM 模型的前向传播。
        它非常简洁，直接将所有参数传递给内部的 LLM。
        真正的多模态处理“魔法”发生在数据准备阶段，即文本和时间序列token被混合在一起输入。
        """
        return self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def generate(self, *args, **kwargs):
        """
        生成任务同样完全委托给内部的 LLM。
        """
        return self.llm.generate(*args, **kwargs)