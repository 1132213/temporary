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

from configs.config import SPECIAL_TOKENS

class UnsupervisedTimeSeriesDataset(Dataset):
    """
    用于 VQ-VAE 无监督训练的数据集。
    """
    def __init__(self, data_dir: str, patch_size: int):
        self.patch_size = patch_size
        self.revin = RevIN(num_features=1, affine=False)
        self.patches = []

        file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        
        print(f"正在从 {len(file_paths)} 个文件中加载和处理数据...")
        for path in file_paths:
            ts_data = np.load(path).astype(np.float32)
            if len(ts_data.shape) > 1:
                ts_data = ts_data.flatten()
            
            num_patches = len(ts_data) // self.patch_size
            if num_patches > 0:
                trimmed_data = ts_data[:num_patches * self.patch_size]
                patches_from_file = trimmed_data.reshape(num_patches, self.patch_size)
                
                for i in range(patches_from_file.shape[0]):
                    patch = torch.from_numpy(patches_from_file[i])
                    patch = patch.unsqueeze(0).unsqueeze(-1)
                    norm_patch = self.revin(patch, mode='norm')
                    norm_patch = norm_patch.squeeze(0).permute(1, 0)
                    self.patches.append(norm_patch)
        
        print(f"数据加载完成。总共找到 {len(self.patches)} 个补丁。")

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.patches[idx]

class PairedTSTextDataset(Dataset):
    """
    用于阶段二：多模态对齐微调的数据集。
    """
    def __init__(self, data_path: str, vq_vae: VQVAE, tokenizer: AutoTokenizer, patch_size: int):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.vq_vae = vq_vae
        self.tokenizer = tokenizer
        self.patch_size = patch_size
        self.revin = RevIN(num_features=1, affine=False)
        
        self.text_start_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['text_start'])
        self.text_end_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['text_end'])
        self.ts_start_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['ts_start'])
        self.ts_end_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['ts_end'])

    def __len__(self) -> int:
        # 因为每个样本都会产生两种顺序，所以数据集的有效长度加倍
        return len(self.data) * 2

    def __getitem__(self, idx: int) -> torch.Tensor:
        # 计算原始数据索引和我们想要的顺序
        original_idx = idx // 2
        order_is_ts_first = (idx % 2 == 1)

        item = self.data[original_idx]
        
        # --- 处理文本 ---
        text = item['text']
        text_tokens = self.tokenizer(text, add_special_tokens=False)['input_ids']
        
        # --- 处理时间序列 ---
        ts_data = np.load(item['ts_path']).astype(np.float32)
        if len(ts_data.shape) > 1: ts_data = ts_data.flatten()
        
        num_patches = len(ts_data) // self.patch_size
        if num_patches == 0:
            # 如果时间序列太短，无法处理，则只返回一个简单的文本序列
            return torch.LongTensor([self.text_start_id] + text_tokens + [self.text_end_id] + [self.tokenizer.eos_token_id])

        ts_data = ts_data[:num_patches * self.patch_size]
        
        ts_tensor = torch.from_numpy(ts_data).unsqueeze(0).unsqueeze(-1)
        ts_tensor_norm = self.revin(ts_tensor, mode='norm')
        ts_tensor_norm = ts_tensor_norm.permute(0, 2, 1)

        with torch.no_grad():
            device = next(self.vq_vae.parameters()).device
            ts_indices = self.vq_vae.encode(ts_tensor_norm.to(device)).squeeze(0)
        
        ts_token_ids = [self.tokenizer.convert_tokens_to_ids(f"<ts_motif_{i}>") for i in ts_indices]

        # --- 根据 order_is_ts_first 决定拼接顺序 ---
        text_part = [self.text_start_id] + text_tokens + [self.text_end_id]
        ts_part = [self.ts_start_id] + ts_token_ids + [self.ts_end_id]
        
        if order_is_ts_first:
            # 顺序 2: [TIME_SERIES][TEXT][EOS]
            input_ids = ts_part + text_part + [self.tokenizer.eos_token_id]
        else:
            # 顺序 1: [TEXT][TIME_SERIES][EOS]
            input_ids = text_part + ts_part + [self.tokenizer.eos_token_id]
        
        return torch.LongTensor(input_ids)

class InstructionTuningDataset(PairedTSTextDataset):
    """
    用于阶段三：指令微调的数据集。
    """
    def __init__(self, data_path: str, vq_vae: VQVAE, tokenizer: AutoTokenizer, patch_size: int):
        # 调用父类的构造函数，但这里我们重写它，因为长度计算不同
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.vq_vae = vq_vae
        self.tokenizer = tokenizer
        self.patch_size = patch_size
        self.revin = RevIN(num_features=1, affine=False)

        # 获取指令相关的特殊token ID
        self.instruction_start_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['instruction'])
        self.instruction_end_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['end_instruction'])
        # 同样需要文本和时序的token
        self.text_start_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['text_start'])
        self.text_end_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['text_end'])
        self.ts_start_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['ts_start'])
        self.ts_end_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['ts_end'])

    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        instruction = item['instruction']
        instruction_tokens = self.tokenizer(instruction, add_special_tokens=False)['input_ids']
        
        # --- 处理输入部分 ---
        input_tokens = []
        if 'input_text' in item and item['input_text']:
            text_tokens = self.tokenizer(item['input_text'], add_special_tokens=False)['input_ids']
            input_tokens.extend([self.text_start_id] + text_tokens + [self.text_end_id])
            
        if 'input_ts_path' in item and item['input_ts_path']:
            ts_data = np.load(item['input_ts_path']).astype(np.float32)
            if len(ts_data.shape) > 1: ts_data = ts_data.flatten()
            num_patches = len(ts_data) // self.patch_size
            if num_patches > 0:
                ts_data = ts_data[:num_patches * self.patch_size]
                ts_tensor = torch.from_numpy(ts_data).unsqueeze(0).unsqueeze(-1)
                ts_tensor_norm = self.revin(ts_tensor, mode='norm')
                ts_tensor_norm = ts_tensor_norm.permute(0, 2, 1)

                with torch.no_grad():
                    device = next(self.vq_vae.parameters()).device
                    ts_indices = self.vq_vae.encode(ts_tensor_norm.to(device)).squeeze(0)
                
                ts_token_ids = [self.tokenizer.convert_tokens_to_ids(f"<ts_motif_{i}>") for i in ts_indices]
                input_tokens.extend([self.ts_start_id] + ts_token_ids + [self.ts_end_id])

        # --- 处理输出部分 ---
        output_tokens = []
        if 'output_text' in item and item['output_text']:
            otext_tokens = self.tokenizer(item['output_text'], add_special_tokens=False)['input_ids']
            output_tokens.extend([self.text_start_id] + otext_tokens + [self.text_end_id])
        
        if 'output_ts_path' in item and item['output_ts_path']: # 支持从路径加载输出TS
            ts_data = np.load(item['output_ts_path']).astype(np.float32)
            if len(ts_data.shape) > 1: ts_data = ts_data.flatten()
            num_patches = len(ts_data) // self.patch_size
            if num_patches > 0:
                ts_data = ts_data[:num_patches * self.patch_size]
                ts_tensor = torch.from_numpy(ts_data).unsqueeze(0).unsqueeze(-1)
                ts_tensor_norm = self.revin(ts_tensor, mode='norm')
                ts_tensor_norm = ts_tensor_norm.permute(0, 2, 1)

                with torch.no_grad():
                    device = next(self.vq_vae.parameters()).device
                    ts_indices = self.vq_vae.encode(ts_tensor_norm.to(device)).squeeze(0)
                
                ts_token_ids = [self.tokenizer.convert_tokens_to_ids(f"<ts_motif_{i}>") for i in ts_indices]
                output_tokens.extend([self.ts_start_id] + ts_token_ids + [self.ts_end_id])

        # --- 为因果语言模型组合最终序列 ---
        instruction_part = [self.instruction_start_id] + instruction_tokens + [self.instruction_end_id]
        #+eos_token
        full_sequence = instruction_part + input_tokens + output_tokens + [self.tokenizer.eos_token_id]
        
        ignore_part_len = len(instruction_part) + len(input_tokens)
        labels = [-100] * ignore_part_len + output_tokens + [self.tokenizer.eos_token_id]
        
        assert len(full_sequence) == len(labels)
        
        return {
            "input_ids": torch.LongTensor(full_sequence),
            "labels": torch.LongTensor(labels)
        }