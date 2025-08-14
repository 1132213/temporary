# clgm/utils/evaluation.py
import numpy as np

def calculate_ts_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    为时间序列评估计算回归指标。

    Args:
        y_true (np.ndarray): 真实的时间序列值。
        y_pred (np.ndarray): 预测的时间序列值。

    Returns:
        dict: 一个包含均方误差（MSE）和平均绝对误差（MAE）的字典。
    """
    # 确保输入是一维数组，以简化比较和计算
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
        
    # 如果两个序列的形状不匹配（例如，在预测任务中，预测长度可能不同）
    # 则将它们截断为较短的那个序列的长度，以便进行比较。
    if y_true.shape != y_pred.shape:
        min_len = min(y_true.shape[0], y_pred.shape[0])
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
    # 计算均方误差 (Mean Squared Error)
    mse = np.mean((y_true - y_pred)**2)
    # 计算平均绝对误差 (Mean Absolute Error)
    mae = np.mean(np.abs(y_true - y_pred))
    
    return {
        "MSE": mse,
        "MAE": mae
    }
