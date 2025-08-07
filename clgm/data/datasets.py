# clgm/data/datasets.py
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import json
from typing import Dict, List, Any

from clgm.models.vq_vae import VQVAE
from clgm.utils.revin import RevIN
from transformers import AutoTokenizer

class UnsupervisedTimeSeriesDataset(Dataset):
    """
    用于 VQ-VAE 无监督训练的数据集。
    它会加载目录中所有的 .npy 时间序列，将它们切分成不重叠的补丁（patches），
    并对每个补丁进行实例归一化。
    """
    def __init__(self, data_dir: str, patch_size: int):
        self.patch_size = patch_size
        self.revin = RevIN(num_features=1, affine=False)
        self.patches = []

        # 获取数据目录下所有.npy文件
        file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        
        print(f"正在从 {len(file_paths)} 个文件中加载和处理数据...")
        for path in file_paths:
            # 1. 加载单个时间序列文件
            ts_data = np.load(path).astype(np.float32)
            
            # 2. 确保数据是一维的
            if len(ts_data.shape) > 1:
                ts_data = ts_data.flatten()
            
            # 3. 计算可以切分出的完整补丁数量
            num_patches = len(ts_data) // self.patch_size
            if num_patches > 0:
                # 舍弃末尾不足一个补丁的剩余部分
                trimmed_data = ts_data[:num_patches * self.patch_size]
                # 将一维数组重塑为 [num_patches, patch_size]
                patches_from_file = trimmed_data.reshape(num_patches, self.patch_size)
                
                # 5. 对每个补丁应用RevIN并存储
                # 注意：这里将所有补丁加载到内存。如果数据集极大，可能会导致内存问题。
                # 更优化的方案是只存储文件路径，在 __getitem__ 中动态加载和处理。
                for i in range(patches_from_file.shape[0]):
                    patch = torch.from_numpy(patches_from_file[i])
                    # RevIN期望的输入是 [Batch, Length, Features]，我们传入 [1, patch_size, 1]
                    patch = patch.unsqueeze(0).unsqueeze(-1) # -> [1, patch_size, 1]
                    
                    # 进行实例归一化
                    norm_patch = self.revin(patch, mode='norm')
                    
                    # 调整形状为 VQ-VAE 模型期望的 [Channels, Length]，即 [1, patch_size]
                    norm_patch = norm_patch.squeeze(0).permute(1, 0) # -> [1, patch_size]
                    
                    self.patches.append(norm_patch)
        
        print(f"数据加载完成。总共找到 {len(self.patches)} 个补丁。")

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.patches[idx]

class PairedTSTextDataset(Dataset):
    """用于阶段二：多模态对齐微调的数据集。"""
    def __init__(self, data_path: str, vq_vae: VQVAE, tokenizer: AutoTokenizer, patch_size: int):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.vq_vae = vq_vae
        self.tokenizer = tokenizer
        self.patch_size = patch_size
        self.revin = RevIN(num_features=1, affine=False)
        
        # 预先获取特殊词元的ID，提高效率
        self.text_start_id = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map['text_start'])
        self.text_end_id = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map['text_end'])
        self.ts_start_id = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map['ts_start'])
        self.ts_end_id = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map['ts_end'])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        item = self.data[idx]
        
        # --- 处理文本 ---
        text = item['text']
        # add_special_tokens=False 因为我们会手动添加自定义的结构化token
        text_tokens = self.tokenizer(text, add_special_tokens=False)['input_ids']
        
        # --- 处理时间序列 ---
        # **修正**: 将 np.float3t32 修正为 np.float32
        ts_data = np.load(item['ts_path']).astype(np.float32)
        if len(ts_data.shape) > 1: ts_data = ts_data.flatten()
        
        num_patches = len(ts_data) // self.patch_size
        ts_data = ts_data[:num_patches * self.patch_size]
        
        ts_tensor = torch.from_numpy(ts_data)
        # 归一化处理
        ts_tensor = ts_tensor.unsqueeze(0).unsqueeze(-1) # -> [1, Length, 1]
        ts_tensor_norm = self.revin(ts_tensor, mode='norm') # -> [1, Length, 1]
        
        # 调整形状为 VQ-VAE 模型期望的 [Batch, Channels, Length]
        ts_tensor_norm = ts_tensor_norm.permute(0, 2, 1) # -> [1, 1, Length]

        # 使用冻结的 VQ-VAE 编码器将时间序列分词
        with torch.no_grad():
            # to(self.vq_vae.device) 确保张量在模型所在的设备上
            device = next(self.vq_vae.parameters()).device
            ts_indices = self.vq_vae.encode(ts_tensor_norm.to(device)).squeeze(0)
        
        # 将 VQ-VAE 输出的索引转换为 LLM 词汇表中对应的 <ts_motif_id>
        ts_token_ids = [self.tokenizer.convert_tokens_to_ids(f"<ts_motif_{i}>") for i in ts_indices]

        # --- 组合成单一序列 ---
        # 格式: <text_start>...text...<text_end><ts_start>...ts_tokens...<ts_end>
        input_ids = ([self.text_start_id] + text_tokens + [self.text_end_id] +
                     [self.ts_start_id] + ts_token_ids + [self.ts_end_id])
        
        return torch.LongTensor(input_ids)

class InstructionTuningDataset(PairedTSTextDataset):
    """用于阶段三：指令微调的数据集。继承自 PairedTSTextDataset。"""
    def __init__(self, data_path: str, vq_vae: VQVAE, tokenizer: AutoTokenizer, patch_size: int):
        # 调用父类的构造函数
        super().__init__(data_path, vq_vae, tokenizer, patch_size)
        # 获取指令相关的特殊token ID
        self.instruction_start_id = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map['instruction'])
        self.instruction_end_id = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map['end_instruction'])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        instruction = item['instruction']
        
        # 将指令文本分词
        instruction_tokens = self.tokenizer(instruction, add_special_tokens=False)['input_ids']
        
        # --- 处理输入部分 (文本 和/或 时间序列) ---
        input_tokens = []
        if 'input_text' in item and item['input_text']:
            text_tokens = self.tokenizer(item['input_text'], add_special_tokens=False)['input_ids']
            input_tokens.extend([self.text_start_id] + text_tokens + [self.text_end_id])
            
        if 'input_ts_path' in item and item['input_ts_path']:
            # --- 修正部分: 补全缺失的时间序列处理逻辑 ---
            ts_data = np.load(item['input_ts_path']).astype(np.float32)
            if len(ts_data.shape) > 1: ts_data = ts_data.flatten()
            num_patches = len(ts_data) // self.patch_size
            ts_data = ts_data[:num_patches * self.patch_size]
            
            ts_tensor = torch.from_numpy(ts_data).unsqueeze(0).unsqueeze(-1)
            ts_tensor_norm = self.revin(ts_tensor, mode='norm')
            ts_tensor_norm = ts_tensor_norm.permute(0, 2, 1)

            with torch.no_grad():
                device = next(self.vq_vae.parameters()).device
                ts_indices = self.vq_vae.encode(ts_tensor_norm.to(device)).squeeze(0)
            
            ts_token_ids = [self.tokenizer.convert_tokens_to_ids(f"<ts_motif_{i}>") for i in ts_indices]
            input_tokens.extend([self.ts_start_id] + ts_token_ids + [self.ts_end_id])
            # --- 修正结束 ---

        # --- 处理输出部分 (文本 或 时间序列) ---
        output_tokens = []
        if 'output_text' in item and item['output_text']:
            otext_tokens = self.tokenizer(item['output_text'], add_special_tokens=False)['input_ids']
            output_tokens.extend([self.text_start_id] + otext_tokens + [self.text_end_id])
        
        # 数据集中可能包含预先分词好的时间序列token
        if 'output_ts_tokens' in item and item['output_ts_tokens']:
            ots_token_ids = [self.tokenizer.convert_tokens_to_ids(f"<ts_motif_{i}>") for i in item['output_ts_tokens']]
            output_tokens.extend([self.ts_start_id] + ots_token_ids + [self.ts_end_id])
            
        # --- 为因果语言模型（Causal LM）训练组合最终序列 ---
        # 格式: <instruction>...</instruction> [INPUTS] [OUTPUTS]
        # 注意：[INPUTS] 和 [OUTPUTS] 包含了它们各自的<start>和<end> token
        instruction_part = [self.instruction_start_id] + instruction_tokens + [self.instruction_end_id]
        full_sequence = instruction_part + input_tokens + output_tokens
        
        # --- 创建标签（labels） ---
        # 这是指令微调的关键：我们只希望模型在预测 "输出" 部分时计算损失。
        # 因此，我们将 "指令" 和 "输入" 部分的标签设置为 -100，这是 Hugging Face 库中忽略损失计算的特殊值。
        ignore_part_len = len(instruction_part) + len(input_tokens)
        labels = [-100] * ignore_part_len + output_tokens
        
        # 确保 input_ids 和 labels 长度相同
        assert len(full_sequence) == len(labels)
        
        return {
            "input_ids": torch.LongTensor(full_sequence),
            "labels": torch.LongTensor(labels)
        }