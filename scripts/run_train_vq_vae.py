# scripts/run_train_vq_vae.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from tqdm import tqdm 

from clgm.models.vq_vae import VQVAE
from clgm.data.datasets import UnsupervisedTimeSeriesDataset
from configs.config import VQ_VAE_TRAIN_CONFIG, PATCH_SIZE, DEVICE

def main():
    config = VQ_VAE_TRAIN_CONFIG
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    print(f"使用设备: {DEVICE}")
    print(f"模型检查点将保存在: {config['checkpoint_dir']}")

    print("正在加载数据集...")
    dataset = UnsupervisedTimeSeriesDataset(
        data_dir=config["data_dir"],
        patch_size=PATCH_SIZE
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True, 
        num_workers=4, 
        pin_memory=True if DEVICE == 'cuda' else False 
    )
    print("数据集加载完毕。")

    model = VQVAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    best_loss = float('inf')
    print("开始训练 VQ-VAE...")
    for epoch in range(config["num_epochs"]):
        model.train() 
        total_recon_loss = 0.0 
        total_vq_loss = 0.0   
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for i, batch in enumerate(progress_bar):
            batch = batch.to(DEVICE) 
            optimizer.zero_grad()    

            decoded_x, vq_loss, _ = model(batch)
            
            recon_loss = F.mse_loss(decoded_x, batch)
            loss = recon_loss + vq_loss
            
            loss.backward()
            optimizer.step()

            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()

            avg_recon_loss = total_recon_loss / (i + 1)
            avg_vq_loss = total_vq_loss / (i + 1)
            progress_bar.set_postfix({
                "Avg Recon Loss": f"{avg_recon_loss:.4f}",
                "Avg VQ Loss": f"{avg_vq_loss:.4f}"
            })

        avg_loss_epoch = (total_recon_loss + total_vq_loss) / len(dataloader)
        print(f"Epoch {epoch+1} 完成。平均总损失: {avg_loss_epoch:.4f}")

        if avg_loss_epoch < best_loss:
            best_loss = avg_loss_epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(config["checkpoint_dir"], 'best_model.pth'))
            print(f"new best: {config['checkpoint_dir']}。")

if __name__ == "__main__":
    main()