# clgm/data/data_collators.py
import torch
from typing import List, Dict, Any

class CausalLMCollator:
    """
    用于因果语言建模（Causal LM）的数据整理器。
    它将一个批次中的多个序列进行填充（padding），以使它们具有相同的长度，
    从而能够组合成一个单一的张量（tensor）。
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        """
        这个函数会被 DataLoader 在每个批次上调用。
        
        Args:
            batch: 一个数据样本的列表。
                   - 在阶段二，它是一个 torch.Tensor 的列表。
                   - 在阶段三，它是一个字典的列表，形如 [{'input_ids': ..., 'labels': ...}, ...]。
        
        Returns:
            一个包含 "input_ids", "attention_mask", 和 "labels" 的字典，
            其中的值都是已经填充好并堆叠成批次的张量。
        """
        # 检查批次中第一个元素的类型来判断数据格式
        is_dict_batch = isinstance(batch[0], dict)

        if is_dict_batch:
            # 阶段三格式：从字典列表中提取 input_ids 和 labels
            input_ids = [item['input_ids'] for item in batch]
            labels = [item['labels'] for item in batch]
        else:
            # 阶段二格式：批次本身就是 input_ids 的列表
            input_ids = batch
            # 在这种情况下，标签与输入是相同的，因为模型需要预测每一个token
            labels = [item.clone() for item in batch]

        # 1. 填充 input_ids
        # 找到当前批次中最长的序列长度
        max_len = max(len(ids) for ids in input_ids)
        
        # 对每个序列在右侧进行填充，直到其长度达到 max_len
        padded_input_ids = [
            torch.cat([
                ids,
                torch.full((max_len - len(ids),), self.tokenizer.pad_token_id, dtype=torch.long)
            ])
            for ids in input_ids
        ]
        # 将填充后的序列列表堆叠成一个 [batch_size, max_len] 的张量
        padded_input_ids = torch.stack(padded_input_ids)

        # 2. 填充 labels
        padded_labels = [
            torch.cat([
                lbl,
                torch.full((max_len - len(lbl),), -100, dtype=torch.long) # 使用-100填充标签
            ])
            for lbl in labels
        ]
        padded_labels = torch.stack(padded_labels)
        
        # 3. 创建注意力掩码 (attention_mask)
        # 掩码中，真实token的位置是1，填充token的位置是0
        attention_mask = padded_input_ids.ne(self.tokenizer.pad_token_id).long()
        
        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": padded_labels
        }