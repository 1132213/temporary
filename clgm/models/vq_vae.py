# clgm/models/vq_vae.py
import torch
import torch.nn as nn
from typing import Tuple

from clgm.models.modules import Encoder, Decoder, VectorQuantizer
from configs.config import VQ_EMBEDDING_DIM, VQ_NUM_LAYERS, VQ_COMPRESSION_FACTOR, VQ_CODEBOOK_SIZE, VQ_COMMITMENT_COST

class VQVAE(nn.Module):
    """
    完整的 VQ-VAE 模型，用于时间序列的离散化（分词）。
    它集成了编码器、矢量量化器和解码器。
    """
    def __init__(self,
                 in_channels: int = 1,                 # 输入通道数（单变量时间序列为1）
                 num_hiddens: int = 64,                # 隐藏层通道数
                 num_residual_layers: int = VQ_NUM_LAYERS,
                 num_embeddings: int = VQ_CODEBOOK_SIZE,
                 embedding_dim: int = VQ_EMBEDDING_DIM,
                 commitment_cost: float = VQ_COMMITMENT_COST,
                 compression_factor: int = VQ_COMPRESSION_FACTOR):
        super(VQVAE, self).__init__()

        # 实例化编码器
        self.encoder = Encoder(in_channels, num_hiddens, num_residual_layers, compression_factor)

        # --- 修正部分 ---
        # 增加一个 pre-quantization 卷积层。
        # 这一层的作用是确保编码器输出的特征维度（num_hiddens）能够匹配
        # 码本中嵌入向量的维度（embedding_dim）。这是一个关键步骤。
        self.pre_vq_conv = nn.Conv1d(num_hiddens, embedding_dim, kernel_size=1, stride=1)
        # --- 修正结束 ---

        # 实例化矢量量化器
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        # 实例化解码器，注意其输入维度是码本的 embedding_dim
        self.decoder = Decoder(embedding_dim, num_hiddens, out_channels=in_channels,
                               num_residual_layers=num_residual_layers,
                               compression_factor=compression_factor)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        定义 VQ-VAE 的完整前向传播过程（用于训练）。

        Args:
            x (torch.Tensor): 输入的时间序列，形状为 [Batch, Channels, Length]。

        Returns:
            一个元组，包含:
            - decoded_x (torch.Tensor): 重构后的时间序列。
            - vq_loss (torch.Tensor): 量化损失。
            - encoding_indices (torch.Tensor): 时间序列的离散token索引。
        """
        # 1. 使用编码器压缩输入时间序列
        z = self.encoder(x)

        # 2. 将编码后的特征映射到码本的嵌入维度
        z = self.pre_vq_conv(z)

        # 3. 对潜在表示进行矢量量化
        quantized, vq_loss, encoding_indices = self.vq(z)

        # 4. 使用解码器将量化后的表示重构回原始时间序列空间
        decoded_x = self.decoder(quantized)

        return decoded_x, vq_loss, encoding_indices

    @torch.no_grad() # 装饰器确保在推理过程中不计算梯度
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        将时间序列编码为一系列离散索引（tokens）。
        此方法在 LLM 微调的数据预处理阶段使用。
        """
        z = self.encoder(x)
        z = self.pre_vq_conv(z)
        # 我们只需要量化后得到的索引
        _, _, encoding_indices = self.vq(z)
        return encoding_indices

    @torch.no_grad() # 装饰器确保在推理过程中不计算梯度
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        将一系列离散索引解码回连续的时间序列。
        此方法在 LLM 的生成（推理）阶段使用。
        """
        # 使用索引从码本中直接查找对应的嵌入向量
        # indices shape: [Batch, Length]
        # quantized shape: [Batch, Length, EmbeddingDim]
        quantized = self.vq._embedding(indices)
        # 转换为解码器期望的输入格式 [Batch, EmbeddingDim, Length]
        quantized = quantized.permute(0, 2, 1)

        # 使用解码器进行重构
        decoded_x = self.decoder(quantized)
        return decoded_x