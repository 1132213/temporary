# scripts/evaluate_vq_vae.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from clgm.models.vq_vae import VQVAE
from clgm.utils.revin import RevIN
from clgm.utils.evaluation import calculate_ts_metrics
from configs.config import PATCH_SIZE, VQ_VAE_TRAIN_CONFIG

# --- 新增: 定义输出目录 ---
OUTPUT_DIR = "output"

def generate_synthetic_timeseries(length=1024, save_path="synthetic_sample.npy"):
    """
    生成一段合成的时间序列用于测试。
    它由一个正弦波、一个线性上升趋势和一些随机噪声组成。
    """
    print(f"\n正在生成长度为 {length} 的合成时间序列样本...")
    time = np.linspace(0, 10 * np.pi, length)
    sine_wave = np.sin(time) * 1.5
    linear_trend = np.linspace(0, 5, length)
    noise = np.random.randn(length) * 0.4
    synthetic_ts = sine_wave + linear_trend + noise
    
    # --- 修改: 确保保存路径的目录存在 ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, synthetic_ts)
    print(f"合成样本已保存至: {save_path}")
    return synthetic_ts

def evaluate_reconstruction(model_path, ts_data, patch_size, device):
    """
    加载训练好的VQ-VAE模型，对给定的时间序列样本进行重构，并进行定性和定量评估。
    """
    # 1. 加载模型 (保持不变)
    print(f"正在从 {model_path} 加载模型...")
    model = VQVAE().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"模型加载成功，使用设备: {device}")

    # 2. 预处理数据 (保持不变)
    print("正在处理样本数据...")
    revin = RevIN(num_features=1, affine=False)
    
    ts_data = ts_data.astype(np.float32)
    if ts_data.ndim > 1:
        ts_data = ts_data.flatten()
    
    num_patches = len(ts_data) // patch_size
    if num_patches == 0:
        print(f"错误：时间序列太短 (长度 {len(ts_data)})，无法切分出一个完整的补丁 (需要 {patch_size})。")
        return
        
    original_ts = ts_data[:num_patches * patch_size]
    patches = original_ts.reshape(num_patches, patch_size)
    
    reconstructed_patches = []
    with torch.no_grad():
        for i in range(patches.shape[0]):
            patch_np = patches[i].astype(np.float32)
            patch_tensor = torch.from_numpy(patch_np).unsqueeze(0).unsqueeze(-1).to(device)
            
            norm_patch = revin(patch_tensor, mode='norm')
            norm_patch = norm_patch.permute(0, 2, 1)
            
            decoded_patch, _, _ = model(norm_patch)
            
            denorm_patch = revin(decoded_patch.permute(0, 2, 1), mode='denorm')
            
            reconstructed_patches.append(denorm_patch.squeeze().cpu().numpy())

    reconstructed_ts = np.concatenate(reconstructed_patches)

    # 3. 定量评估 (保持不变)
    metrics = calculate_ts_metrics(original_ts, reconstructed_ts)
    print("\n--- 定量评估结果 ---")
    print(f"  均方误差 (MSE): {metrics['MSE']:.6f}")
    print(f"  平均绝对误差 (MAE): {metrics['MAE']:.6f}")

    # 4. 定性评估 (修改了可视化部分)
    print("\n--- 定性评估结果 ---")
    plt.figure(figsize=(15, 6))
    plt.title("VQ-VAE Reconstruction effect comparison (using automatically generated samples)")
    plt.plot(original_ts, label='Original Time Series', color='blue', alpha=0.8, linewidth=1.5)
    plt.plot(reconstructed_ts, label='Reconstructed Time Series', color='red', linestyle='--', alpha=0.7, linewidth=1.0)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    # --- 修改: 将图片保存到指定的输出目录 ---
    output_fig_path = os.path.join(OUTPUT_DIR, "reconstruction_comparison.png")
    plt.savefig(output_fig_path)
    print(f"对比图像已保存至: {output_fig_path}")
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 VQ-VAE 模型的重构性能")
    parser.add_argument("--gpu", type=int, default=0, help="要使用的 GPU 索引 (例如, 0, 1, 2, ...)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
    else:
        device = "cpu"
    print(f"--- 将要使用的设备是: {device} ---")

    # --- 新增: 确保输出目录存在 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MODEL_CHECKPOINT_PATH = os.path.join(VQ_VAE_TRAIN_CONFIG["checkpoint_dir"], 'best_model.pth')
    
    # --- 修改: 指定 .npy 文件的保存路径 ---
    synthetic_npy_path = os.path.join(OUTPUT_DIR, "synthetic_sample.npy")
    synthetic_ts_data = generate_synthetic_timeseries(save_path=synthetic_npy_path)
    
    evaluate_reconstruction(
        model_path=MODEL_CHECKPOINT_PATH,
        ts_data=synthetic_ts_data,
        patch_size=PATCH_SIZE,
        device=device
    )