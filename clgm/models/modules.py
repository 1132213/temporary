# clgm/models/modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualStack(nn.Module):
    """
    一个由多个残差块组成的堆栈。
    每个残差块包含一个ReLU激活、一个3x1的卷积层，然后再接一个ReLU激活。
    这种结构允许网络构建得更深，同时避免梯度消失问题。
    """
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList()

        # --- 修正部分 ---
        # 原始代码中这个循环是缺失的，导致残差堆栈是空的。
        # 这里我们添加循环来真正地构建残差块。
        for _ in range(num_residual_layers):
            # 每个残差块是一个简单的序列：ReLU -> Conv1d -> ReLU
            conv_block = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(in_channels=in_channels,
                          out_channels=num_hiddens,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv1d(in_channels=num_hiddens,
                          out_channels=in_channels, # 输出通道数必须与输入相同
                          kernel_size=1, stride=1, bias=False)
            )
            self._layers.append(conv_block)
        # --- 修正结束 ---

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对每一个残差块，将输入x与块的输出相加，实现残差连接
        for layer in self._layers:
            x = x + layer(x)
        # 在整个堆栈的输出上应用一次ReLU激活
        # 注意：在原始的DeepMind实现中，最后的激活可能是不必要的，取决于后续层。
        # 但在这里保留它可以增加非线性。
        return F.relu(x)

class Encoder(nn.Module):
    """
    VQ-VAE 的一维卷积编码器。
    它通过一系列卷积和下采样操作，将输入的时间序列压缩成一个较短的潜在表示。
    """
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int, compression_factor: int):
        super(Encoder, self).__init__()
        # 使用compression_factor来动态调整下采样层，但这里使用了固定的两层stride=2的卷积，等效于4倍压缩
        # 第一个卷积层：下采样，通道数增加
        self._conv_1 = nn.Conv1d(in_channels, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        # 第二个卷积层：再次下采样，通道数增加
        self._conv_2 = nn.Conv1d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1)
        # 第三个卷积层：不改变序列长度，用于特征提取
        self._conv_3 = nn.Conv1d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1)
        # 核心的残差堆栈，用于深化网络
        self._residual_stack = ResidualStack(num_hiddens, num_hiddens, num_residual_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self._conv_1(x))
        x = F.relu(self._conv_2(x))
        x = self._conv_3(x)
        # 将输入传递给残差堆栈进行深度特征提取
        x = self._residual_stack(x)
        return x

class Decoder(nn.Module):
    """
    VQ-VAE 的一维转置卷积解码器。
    它的结构与编码器相反，通过一系列上采样和卷积操作，将量化后的潜在表示重构成原始时间序列。
    """
    def __init__(self, in_channels: int, num_hiddens: int, out_channels: int, num_residual_layers: int, compression_factor: int):
        super(Decoder, self).__init__()
        # 第一个卷积层，不改变序列长度，为解码做准备
        self._conv_1 = nn.Conv1d(in_channels, num_hiddens, kernel_size=3, stride=1, padding=1)
        # 残差堆栈
        self._residual_stack = ResidualStack(num_hiddens, num_hiddens, num_residual_layers)
        # 第一个转置卷积层：上采样，通道数减少
        self._conv_trans_1 = nn.ConvTranspose1d(num_hiddens, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        # 第二个转置卷积层：再次上采样，恢复到原始通道数
        self._conv_trans_2 = nn.ConvTranspose1d(num_hiddens // 2, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_1(x)
        x = self._residual_stack(x)
        x = F.relu(self._conv_trans_1(x))
        x = self._conv_trans_2(x)
        return x

class VectorQuantizer(nn.Module):
    """
    矢量量化器。这是 VQ-VAE 的核心。
    它将编码器输出的连续潜在向量，映射到码本（codebook）中最接近的离散向量。
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super(VectorQuantizer, self).__init__()
        self._num_embeddings = num_embeddings     # 码本中的向量数量 (K)
        self._embedding_dim = embedding_dim       # 每个码本向量的维度 (D)
        self._commitment_cost = commitment_cost   # 承诺损失的权重 (beta)

        # 创建码本，这是一个可学习的嵌入层
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        # 对码本权重进行均匀分布初始化
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)

    def forward(self, inputs: torch.Tensor):
        # 输入形状: [Batch, Channels, Length] (N, C, L)，其中 C == embedding_dim
        # 将输入从 [N, C, L] 转换为 [N, L, C] 以便计算距离
        inputs_permuted = inputs.permute(0, 2, 1).contiguous()
        # 将输入展平为 [N*L, C] 以便进行矩阵运算
        flat_input = inputs_permuted.view(-1, self._embedding_dim)

        # --- 计算距离 ---
        # 计算每个输入向量与码本中所有向量之间的欧氏距离的平方
        # (a-b)^2 = a^2 - 2ab + b^2
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # --- 编码 ---
        # 找到每个输入向量的最近邻码本向量的索引
        # encoding_indices 的形状为 [N*L]
        encoding_indices = torch.argmin(distances, dim=1)

        # --- 量化 ---
        # 使用索引从码本中提取量化后的向量
        quantized_flat = self._embedding(encoding_indices)
        # 将量化后的向量重塑为 [N, L, C]
        quantized = quantized_flat.view(inputs_permuted.shape)

        # --- 计算损失 ---
        # 1. 承诺损失 (Commitment Loss): e_latent_loss
        #    鼓励编码器的输出（inputs）接近所选的码本向量。
        #    我们使用 .detach() 来阻止梯度流向量化器，只更新编码器。
        e_latent_loss = F.mse_loss(quantized.detach(), inputs_permuted)
        # 2. 码本损失 (Codebook Loss): q_latent_loss
        #    鼓励码本向量接近编码器的输出。
        #    我们使用 .detach() 来阻止梯度流向编码器，只更新码本。
        q_latent_loss = F.mse_loss(quantized, inputs_permuted.detach())
        # 总的 VQ 损失
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # --- 直通估计器 (Straight-Through Estimator) ---
        # 在反向传播时，梯度会跳过量化操作，直接从解码器的输入传到编码器的输出。
        # 这解决了量化操作不可导的问题。
        quantized = inputs_permuted + (quantized - inputs_permuted).detach()

        # 将量化后的张量和索引的形状恢复到期望的输出格式
        # 量化后的张量恢复为 [N, C, L]
        quantized = quantized.permute(0, 2, 1).contiguous()
        # 索引恢复为 [N, L]
        final_indices = encoding_indices.view(inputs_permuted.shape[0], inputs_permuted.shape[1])

        return quantized, loss, final_indices