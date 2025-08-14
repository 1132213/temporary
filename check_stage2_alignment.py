# check_stage2_alignment.py
import torch
import numpy as np
import os
import argparse
from safetensors.torch import load_file
import matplotlib.pyplot as plt

# 假设此脚本在项目根目录下运行
try:
    from clgm.models.clgm_core import CLGM, CLGMConfig
    from clgm.utils.revin import RevIN
    from configs.config import PATCH_SIZE, VQ_CODEBOOK_SIZE, TS_MOTIF_PREFIX, SPECIAL_TOKENS
    from transformers import AutoTokenizer
except ImportError:
    print("错误：无法导入项目模块。")
    print("请确保您将此脚本放置在 'clgm-project' 项目的根目录下，并从那里运行它。")
    exit()

# --- 配置区 ---
# 请确保此路径指向您最新、最成功的第二阶段模型
STAGE2_MODEL_PATH = "./checkpoints/clgm/stage2_aligned_1.5e-5_4/"
# STAGE2_MODEL_PATH = "./checkpoints/clgm/stage3_final/"
TOKENIZER_PATH = STAGE2_MODEL_PATH

# 检验用的样本文件路径 (请根据您的实际文件路径进行调整)
SAMPLE_NPY_PATH = "./data/qwen_generated_dataset_en_merge/npy_files/series_en_1.npy"

class Stage2ModelTester:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.vq_vae = None
        self.revin = RevIN(num_features=1, affine=False)
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        print(f"--- 正在从 '{STAGE2_MODEL_PATH}' 加载第二阶段对齐模型 ---")
        if not os.path.exists(STAGE2_MODEL_PATH):
            print(f"错误：找不到模型路径 '{STAGE2_MODEL_PATH}'。请确认路径是否正确。")
            exit()
            
        try:
            print("--- 步骤 1: 加载扩展后的分词器 ---")
            self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
            
            print("--- 步骤 2: 初始化 CLGM 模型结构 ---")
            clgm_config = CLGMConfig.from_pretrained(STAGE2_MODEL_PATH, local_files_only=True)
            self.model = CLGM(clgm_config)
            
            print(f"--- 步骤 3: 调整模型嵌入层大小至 {len(self.tokenizer)} ---")
            self.model.llm.resize_token_embeddings(len(self.tokenizer))
            
            print("--- 步骤 4: 加载已保存的权重文件 ---")
            state_dict_path = os.path.join(STAGE2_MODEL_PATH, 'model.safetensors')
            if not os.path.exists(state_dict_path):
                 state_dict_path = os.path.join(STAGE2_MODEL_PATH, 'pytorch_model.bin')
                 if not os.path.exists(state_dict_path):
                    raise FileNotFoundError(f"在目录 {STAGE2_MODEL_PATH} 中找不到模型权重文件。")
            
            if 'safetensors' in state_dict_path:
                state_dict = load_file(state_dict_path, device=self.device)
            else:
                state_dict = torch.load(state_dict_path, map_location=self.device, weights_only=False)

            print("--- 步骤 5: 将权重加载到模型中 (strict=False) ---")
            self.model.load_state_dict(state_dict, strict=False)
            
            self.model.to(self.device)
            self.model.eval()
            self.vq_vae = self.model.get_vq_vae().to(self.device)
            print("--- 模型和分词器加载成功 ---")
        except Exception as e:
            print(f"加载模型时发生严重错误: {e}")
            exit()

        self.special_token_ids = {k: self.tokenizer.convert_tokens_to_ids(v) for k, v in SPECIAL_TOKENS.items()}
        self.special_token_ids['eos'] = self.tokenizer.eos_token_id or self.special_token_ids['pad_token']

    def build_prompt(self, ts_path: str = None, text: str = None, task_type: str = 'ts_to_text'):
        """根据任务类型构建正确的引导提示。"""
        prompt_ids = []

        # 准备文本部分
        if text:
            text_tokens = self.tokenizer(text, add_special_tokens=False)['input_ids']
            text_part = [self.special_token_ids['text_start']] + text_tokens + [self.special_token_ids['text_end']]
        
        # 准备时间序列部分
        if ts_path:
            if not os.path.exists(ts_path):
                print(f"错误: 找不到时间序列文件 '{ts_path}'"); return None
            ts_data = np.load(ts_path).astype(np.float32)

            # (新增) 保存原始图像用于对比
            if task_type == 'ts_to_text':
                plt.figure(figsize=(15, 5))
                plt.title("Original Time Series for Visual Check")
                plt.plot(ts_data)
                plt.grid(True)
                plt.savefig("original_ts_plot.png")
                plt.close()
                print("\n--- 原始时间序列图像已保存至: original_ts_plot.png ---")

            if ts_data.ndim > 1: ts_data = ts_data.flatten()
            num_patches = len(ts_data) // PATCH_SIZE
            if num_patches == 0: return None
            ts_data = ts_data[:num_patches * PATCH_SIZE]
            ts_tensor = torch.from_numpy(ts_data).unsqueeze(0).unsqueeze(-1)
            ts_tensor_norm = self.revin(ts_tensor, mode='norm')
            ts_tensor_norm = ts_tensor_norm.permute(0, 2, 1)
            with torch.no_grad():
                ts_indices = self.vq_vae.encode(ts_tensor_norm.to(self.device)).squeeze(0)
            ts_token_ids = [self.tokenizer.convert_tokens_to_ids(f"<ts_motif_{i}>") for i in ts_indices]
            ts_part = [self.special_token_ids['ts_start']] + ts_token_ids + [self.special_token_ids['ts_end']]

        # 根据任务类型决定拼接顺序
        if task_type == 'ts_to_text' and ts_part:
            # 顺序: [TIME_SERIES][TEXT_START] (引导模型开始生成文本)
            prompt_ids = ts_part + [self.special_token_ids['text_start']]
        elif task_type == 'text_to_ts' and text_part:
            # 顺序: [TEXT][TS_START] (引导模型开始生成时间序列)
            prompt_ids = text_part + [self.special_token_ids['ts_start']]
        
        return self.tokenizer.decode(prompt_ids) # 返回字符串prompt

    def run_inference(self, prompt: str, max_new_tokens=256):
        """执行生成并解析输出。"""
        if not prompt: return "Prompt is empty."
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        print("\n--- 正在生成模型输出 ---")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.special_token_ids['eos'],
            pad_token_id=self.special_token_ids['pad_token'],
            do_sample=False
        )
        
        generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
        full_decoded_output = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        print(f"\n模型原始生成内容:\n---\n{full_decoded_output}\n---")
        return full_decoded_output

def main():
    parser = argparse.ArgumentParser(description="检验第二阶段对齐模型的功能 (最终版)。")
    parser.add_argument("--gpu", type=int, default=0, help="要使用的 GPU 索引。")
    args = parser.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"--- 使用设备: {device} ---")
    tester = Stage2ModelTester(device=device)

    # --- 检验任务 1: 时间序列 -> 文本 ---
    print("\n\n=========================================================")
    print("检验任务 1: 时间序列到文本 (TS -> Text)")
    print("目标: 给模型一段真实的时间序列，看它能否生成相关的文本描述。")
    print(f"使用样本: {SAMPLE_NPY_PATH}")
    print("=========================================================")
    prompt1 = tester.build_prompt(ts_path=SAMPLE_NPY_PATH, task_type='ts_to_text')
    tester.run_inference(prompt=prompt1)
    print("\n--- 如何解读检验任务1的结果 ---")
    print("请将下面的生成文本与保存的图像 'original_ts_plot.png' 进行对比。")
    print("好的迹象: 模型生成的文本应该与时间序列的实际特征（如趋势、周期性）大致相符。")
    print("糟糕的迹象: 生成的文本完全不相关，或者只是重复特殊词元。")

    # --- 检验任务 2: 文本 -> 时间序列 ---
    print("\n\n=========================================================")
    print("检验任务 2: 文本到时间序列 (Text -> TS)")
    print("目标: 给模型一段文本描述，看它能否生成对应的时间序列<ts_motif>词元。")
    print("=========================================================")
    text_description = "A signal with a clear upward trend and some periodic fluctuations."
    print(f"输入的文本描述: '{text_description}'")
    prompt2 = tester.build_prompt(text=text_description, task_type='text_to_ts')
    tester.run_inference(prompt=prompt2)
    print("\n--- 如何解读检验任务2的结果 ---")
    print("好的迹象: 生成的内容中应该只包含 <ts_motif_...> 词元，并以 <ts_end> 或 EOS 结束。")
    print("糟糕的迹象: 生成了无关的文本，或者完全没有生成时间序列词元。")
    print("=========================================================")

if __name__ == "__main__":
    main()