# inference/generate.py
import torch
import numpy as np
from transformers import AutoTokenizer
import argparse
import os

from clgm.models.clgm_core import CLGM
from clgm.utils.revin import RevIN
from configs.config import INFERENCE_CONFIG, PATCH_SIZE, VQ_CODEBOOK_SIZE, TS_MOTIF_PREFIX

class Generator:
    """封装了从加载模型到生成多模态输出的完整推理流程。"""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.config = INFERENCE_CONFIG
        
        print("正在加载模型和分词器...")
        # 1. 加载最终模型和扩展后的分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["extended_tokenizer_path"])
        self.model = CLGM.from_pretrained(self.config["clgm_checkpoint_path"]).to(self.device)
        self.model.eval() # 切换到评估模式
        
        # 2. 获取冻结的VQ-VAE和RevIN实例
        self.vq_vae = self.model.get_vq_vae().to(self.device)
        self.revin = RevIN(num_features=1, affine=False)
        self.input_stats_available = False # 用于跟踪是否有可用的输入统计信息

        # 3. 预先计算并缓存token ID，以加速后处理
        # 创建一个从时间序列token ID到其数值索引的映射
        self.ts_token_id_to_idx = {
            self.tokenizer.convert_tokens_to_ids(f"{TS_MOTIF_PREFIX}{i}>"): i
            for i in range(VQ_CODEBOOK_SIZE)
        }
        # 创建一个包含所有时间序列token ID的集合，用于快速判断
        self.ts_token_ids_set = set(self.ts_token_id_to_idx.keys())
        
        # 缓存所有特殊控制token的ID
        self.special_token_ids = {
            'ts_start': self.tokenizer.convert_tokens_to_ids('<ts_start>'),
            'ts_end': self.tokenizer.convert_tokens_to_ids('<ts_end>'),
            'text_start': self.tokenizer.convert_tokens_to_ids('<text_start>'),
            'text_end': self.tokenizer.convert_tokens_to_ids('<text_end>'),
            'eos': self.tokenizer.eos_token_id
        }

    def _build_prompt(self, instruction: str, ts_path: str = None, text: str = None) -> str:
        """根据输入构建符合模型训练格式的提示字符串。"""
        prompt = f"<instruction>{instruction}</instruction>"
        if text:
            prompt += f"<text_start>{text}</text_end>"
        if ts_path:
            # --- 动态地将输入的时间序列编码为token ---
            ts_data = np.load(ts_path).astype(np.float32)
            if ts_data.ndim > 1: ts_data = ts_data.flatten()
            
            # 裁剪到可以被patch_size整除的长度
            num_patches = len(ts_data) // PATCH_SIZE
            if num_patches == 0: return prompt # 如果太短，无法处理
            ts_data = ts_data[:num_patches * PATCH_SIZE]
            
            ts_tensor = torch.from_numpy(ts_data).unsqueeze(0).unsqueeze(-1) # [1, L, 1]
            
            # **修正**: 在这里进行归一化并保存统计量
            ts_tensor_norm = self.revin(ts_tensor, mode='norm') # 这会更新 self.revin.mean 和 self.revin.stdev
            self.input_stats_available = True # 标记统计量已可用
            
            # 准备给VQ-VAE的输入
            ts_tensor_norm = ts_tensor_norm.permute(0, 2, 1) # [1, 1, L]
            
            with torch.no_grad():
                ts_indices = self.vq_vae.encode(ts_tensor_norm.to(self.device)).squeeze(0)
            
            ts_tokens_str = "".join([f"<ts_motif_{i}>" for i in ts_indices])
            prompt += f"<ts_start>{ts_tokens_str}</ts_end>"
            
        return prompt

    def generate(self, instruction: str, ts_path: str = None, text: str = None, max_new_tokens=512):
        """执行完整的生成和后处理流程。"""
        # 重置状态
        self.input_stats_available = False
        
        prompt = self._build_prompt(instruction, ts_path, text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        print("\n--- 正在生成响应 ---")
        # 使用LLM生成token ID序列
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.special_token_ids['eos'],
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False # 使用贪心解码以保证结果可复现
        )
        
        # 提取新生成的token部分
        generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
        
        # --- 后处理：多模态输出的核心解析逻辑 ---
        text_output = ""
        ts_indices = []
        in_text_block = False
        in_ts_block = False

        for token_id in generated_ids:
            token_id_item = token_id.item()
            
            # 如果遇到结束符，则停止解析
            if token_id_item == self.special_token_ids['eos']:
                break
            
            # 状态机，根据特殊token切换解析模式
            if token_id_item == self.special_token_ids['text_start']: in_text_block = True; continue
            if token_id_item == self.special_token_ids['text_end']: in_text_block = False; continue
            if token_id_item == self.special_token_ids['ts_start']: in_ts_block = True; continue
            if token_id_item == self.special_token_ids['ts_end']: in_ts_block = False; continue
            
            # 根据当前模式处理token
            if in_text_block:
                text_output += self.tokenizer.decode(token_id)
            
            if in_ts_block and token_id_item in self.ts_token_ids_set:
                ts_indices.append(self.ts_token_id_to_idx[token_id_item])

        print(f"生成的文本: {text_output}")
        
        if ts_indices:
            print(f"生成了 {len(ts_indices)} 个时间序列 token。")
            # 将离散的token索引解码回连续波形
            indices_tensor = torch.LongTensor(ts_indices).unsqueeze(0).to(self.device)
            with torch.no_grad():
                generated_ts_norm = self.vq_vae.decode(indices_tensor) # [1, 1, L_gen]
                
                # --- 修正: 处理反归一化逻辑 ---
                if self.input_stats_available:
                    print("检测到输入时间序列，使用其统计数据进行反归一化。")
                    generated_ts = self.revin(generated_ts_norm.permute(0, 2, 1), mode='denorm')
                else:
                    print("警告: 未提供输入时间序列，无法进行反归一化。输出为归一化后的波形。")
                    generated_ts = generated_ts_norm # 直接输出归一化后的结果
            
            ts_waveform = generated_ts.squeeze().cpu().numpy()
            print("成功重构时间序列波形。")
            
            output_filename = "generated_timeseries.npy"
            np.save(output_filename, ts_waveform)
            print(f"已将生成的时间序列保存至 {output_filename}")
            return text_output, ts_waveform
            
        return text_output, None

def main():
    parser = argparse.ArgumentParser(description="CLGM 推理脚本")
    parser.add_argument("--task", type=str, required=True, choices=['ts_to_text', 'text_to_ts', 'forecast'], help="要执行的任务类型")
    parser.add_argument("--instruction", type=str, required=True, help="给模型的指令")
    parser.add_argument("--ts_path", type=str, help="输入的时间序列.npy文件路径 (用于ts_to_text, forecast)")
    parser.add_argument("--text", type=str, help="输入的文本 (用于text_to_ts)")
    parser.add_argument("--max_tokens", type=int, default=512, help="要生成的最大token数量")
    args = parser.parse_args()

    generator = Generator()
    
    if args.task == 'ts_to_text':
        if not args.ts_path or not os.path.exists(args.ts_path):
            raise ValueError("--ts_path 是 'ts_to_text' 任务的必需参数，且文件必须存在。")
        generator.generate(instruction=args.instruction, ts_path=args.ts_path, max_new_tokens=args.max_tokens)
    
    elif args.task == 'text_to_ts':
        # Text-to-TS任务不需要输入文本，指令本身已经足够
        generator.generate(instruction=args.instruction, text=args.text, max_new_tokens=args.max_tokens)
        
    elif args.task == 'forecast':
        if not args.ts_path or not os.path.exists(args.ts_path):
            raise ValueError("--ts_path 是 'forecast' 任务的必需参数，且文件必须存在。")
        # forecasting 任务的指令可以更具说明性
        generator.generate(instruction=args.instruction, ts_path=args.ts_path, max_new_tokens=args.max_tokens)
        
if __name__ == "__main__":
    # 使用示例:
    # 1. 时间序列到文本 (例如, 总结)
    # python inference/generate.py --task ts_to_text --instruction "总结这段心电图数据，并指出任何异常。" --ts_path ./data/sample_ecg.npy
    
    # 2. 文本到时间序列 (例如, 条件生成)
    # python inference/generate.py --task text_to_ts --instruction "生成一个代表股票价格的序列，该股票经历一次暴跌，然后缓慢且波动地恢复。"
    
    # 3. 预测
    # python inference/generate.py --task forecast --instruction "基于给定的历史数据，预测接下来6个时间步的用电量。" --ts_path ./data/sample_electricity.npy --max_tokens 10
    main()
