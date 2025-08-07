# clgm/utils/revin.py
import torch
import torch.nn as nn

class RevIN(nn.Module):
    """
    可逆实例归一化 (Reverse Instance Normalization for Time Series)。
    论文地址: https://openreview.net/forum?id=cGDAkQo1C0p
    该模块对每个时间序列实例独立进行归一化，并存储其均值和标准差，
    以便后续可以逆转归一化过程，恢复原始数据尺度。
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        """
        初始化 RevIN 模块。
        
        Args:
            num_features (int): 特征或通道的数量。对于单变量时间序列，此值为1。
            eps (float): 一个很小的数，加在分母上以防止除以零。
            affine (bool): 如果为True，此模块将包含可学习的仿射参数（权重和偏置），
                         允许模型在归一化后学习一个最优的线性变换。
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            # 定义可学习的权重和偏置
            self.weight = nn.Parameter(torch.ones(self.num_features))
            self.bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        定义前向传播逻辑。
        
        Args:
            x (torch.Tensor): 输入张量，形状通常为 [Batch, Length, Features]。
            mode (str): 操作模式，'norm' 表示归一化，'denorm' 表示反归一化。
        
        Returns:
            torch.Tensor: 处理后的张量。
        """
        if mode == 'norm':
            # 先计算并存储统计量
            self._get_statistics(x)
            # 然后进行归一化
            x = self._normalize(x)
        elif mode == 'denorm':
            # 进行反归一化
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"模式 {mode} 未实现。")
        return x

    def _get_statistics(self, x: torch.Tensor):
        """计算并存储输入的均值和标准差。"""
        # 沿时间序列长度维度（dim=1）计算均值和标准差
        # .detach() 是为了确保这些统计量不参与梯度计算
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """执行归一化操作。"""
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            # 应用可学习的仿射变换
            x = x * self.weight
            x = x + self.bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """执行反归一化操作。"""
        if self.affine:
            # 首先逆转仿射变换
            x = x - self.bias
            x = x / (self.weight + self.eps * self.eps) # 加上eps防止除以零
        # 然后逆转标准归一化
        x = x * self.stdev
        x = x + self.mean
        return x