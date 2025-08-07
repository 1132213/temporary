# scripts/run_train_vq_vae.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from tqdm import tqdm # 用于显示美观的进度条

from clgm.models.vq_vae import VQVAE
from clgm.data.datasets import UnsupervisedTimeSeriesDataset
from configs.config import VQ_VAE_TRAIN_CONFIG, PATCH_SIZE, DEVICE

def main():
    # --- 1. 设置 ---
    config = VQ_VAE_TRAIN_CONFIG
    # 创建用于保存模型检查点的目录，如果目录已存在则不报错
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    print(f"使用设备: {DEVICE}")
    print(f"模型检查点将保存在: {config['checkpoint_dir']}")

    # --- 2. 数据 ---
    print("正在加载数据集...")
    dataset = UnsupervisedTimeSeriesDataset(
        data_dir=config["data_dir"],
        patch_size=PATCH_SIZE
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,  # 打乱数据以获得更好的训练效果
        num_workers=4, # 使用多个子进程加载数据，提高效率
        pin_memory=True if DEVICE == 'cuda' else False # 如果使用GPU，固定内存可以加速数据传输
    )
    print("数据集加载完毕。")

    # --- 3. 模型与优化器 ---
    model = VQVAE().to(DEVICE) # 实例化模型并移动到指定设备
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # --- 4. 训练循环 ---
    best_loss = float('inf') # 初始化最佳损失为无穷大
    print("开始训练 VQ-VAE...")
    for epoch in range(config["num_epochs"]):
        model.train() # 将模型设置为训练模式
        total_recon_loss = 0.0 # 初始化当前epoch的总重构损失
        total_vq_loss = 0.0    # 初始化当前epoch的总量化损失
        
        # 使用tqdm包装dataloader以显示进度条
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for i, batch in enumerate(progress_bar):
            batch = batch.to(DEVICE) # 将数据批次移动到指定设备
            optimizer.zero_grad()    # 清空上一轮的梯度

            # 前向传播，获取重构的输出、量化损失和编码
            decoded_x, vq_loss, _ = model(batch)
            
            # 计算重构损失（MSE），即原始输入和解码器输出之间的差距
            recon_loss = F.mse_loss(decoded_x, batch)
            # 总损失是重构损失和量化损失之和
            loss = recon_loss + vq_loss
            
            # 反向传播，计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()

            # 累加损失值用于后续计算平均值
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()

            # 定期更新进度条上的显示信息
            # **修正**: 使用 total_*_loss / (i + 1) 来计算到目前为止的平均损失，而不是依赖不稳定的瞬时值
            avg_recon_loss = total_recon_loss / (i + 1)
            avg_vq_loss = total_vq_loss / (i + 1)
            progress_bar.set_postfix({
                "Avg Recon Loss": f"{avg_recon_loss:.4f}",
                "Avg VQ Loss": f"{avg_vq_loss:.4f}"
            })

        # 计算整个epoch的平均损失
        avg_loss_epoch = (total_recon_loss + total_vq_loss) / len(dataloader)
        print(f"Epoch {epoch+1} 完成。平均总损失: {avg_loss_epoch:.4f}")

        # --- 5. 保存检查点 ---
        # 如果当前epoch的损失比历史最佳损失还要低
        if avg_loss_epoch < best_loss:
            best_loss = avg_loss_epoch
            # 保存模型状态字典、优化器状态、epoch数和损失值
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(config["checkpoint_dir"], 'best_model.pth'))
            print(f"损失创新低，已将新最佳模型保存到 {config['checkpoint_dir']}。")

if __name__ == "__main__":
    main()