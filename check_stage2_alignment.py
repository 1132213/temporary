# check_stage2_alignment.py
import torch
import numpy as np
import os
import argparse
from safetensors.torch import load_file
# 假设此脚本在项目根目录下运行，因此可以直接从 clgm 包导入
try:
    from clgm.models.clgm_core import CLGM, CLGMConfig
    from clgm.utils.revin import RevIN
    from configs.config import PATCH_SIZE, VQ_CODEBOOK_SIZE, TS_MOTIF_PREFIX
    from transformers import AutoTokenizer
except ImportError:
    print("错误：无法导入项目模块。")
    print("请确保您将此脚本放置在 'clgm-project' 项目的根目录下，并从那里运行它。")
    exit()

# --- 配置区 ---
# 直接硬编码第二阶段模型的路径，避免混淆
STAGE2_MODEL_PATH = "./checkpoints/clgm/stage2_aligned2/"
TOKENIZER_PATH = STAGE2_MODEL_PATH # 分词器和模型在同一个目录下

# 检验用的样本文件路径 (请根据您的实际文件路径进行调整)
# 我们假设使用您之前生成的数据
SAMPLE_NPY_PATH = "./data/paired_ts_text/uts_llm_qa_npy_files/uts_llm_qa_0.npy"

class Stage2ModelTester:
    """
    一个专门用于加载和测试第二阶段对齐模型的类。
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.vq_vae = None
        self.revin = RevIN(num_features=1, affine=False)
        self.input_stats_available = False
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """
        加载第二阶段模型和对应的分词器 (safetensors 最终修复版)。
        """
        print(f"--- 正在从 '{STAGE2_MODEL_PATH}' 加载第二阶段对齐模型 ---")
        if not os.path.exists(STAGE2_MODEL_PATH):
            print(f"错误：找不到模型路径 '{STAGE2_MODEL_PATH}'。请确认路径是否正确。")
            exit()
            
        try:
            # 步骤 1: 加载分词器以获取正确的词汇表大小
            print("--- 步骤 1: 加载扩展后的分词器 ---")
            self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
            
            # 步骤 2: 从保存的配置初始化一个全新的、空白的CLGM模型结构
            print("--- 步骤 2: 初始化 CLGM 模型结构 ---")
            clgm_config = CLGMConfig.from_pretrained(STAGE2_MODEL_PATH, local_files_only=True)
            self.model = CLGM(clgm_config)
            
            # 步骤 3: (关键!) 手动将模型的词嵌入层调整到与分词器匹配的正确大小
            print(f"--- 步骤 3: 调整模型嵌入层大小至 {len(self.tokenizer)} ---")
            self.model.llm.resize_token_embeddings(len(self.tokenizer))
            
            # 步骤 4: (关键!) 使用 safetensors 库加载权重文件
            print("--- 步骤 4: 使用 safetensors 加载已保存的权重文件 ---")
            state_dict_path = os.path.join(STAGE2_MODEL_PATH, 'model.safetensors')
            if not os.path.exists(state_dict_path):
                 # 如果 safetensors 文件不存在，给出清晰的错误提示
                 raise FileNotFoundError(f"在目录 {STAGE2_MODEL_PATH} 中找不到 'model.safetensors' 文件。")
            
            # 使用 safetensors.torch.load_file 来加载
            state_dict = load_file(state_dict_path, device=self.device)
            
            # 步骤 5: 将加载的权重应用到模型上
            print("--- 步骤 5: 将权重加载到模型中 ---")
            self.model.load_state_dict(state_dict,strict=False)
            
            self.model.to(self.device)
            self.model.eval()
            self.vq_vae = self.model.get_vq_vae().to(self.device)
            print("--- 模型和分词器加载成功 ---")

        except Exception as e:
            print(f"加载模型时发生严重错误: {e}")
            exit()

        # ... (该函数的其余部分保持不变) ...
        self.ts_token_id_to_idx = {
            self.tokenizer.convert_tokens_to_ids(f"{TS_MOTIF_PREFIX}{i}>"): i
            for i in range(VQ_CODEBOOK_SIZE)
        }
        self.ts_token_ids_set = set(self.ts_token_id_to_idx.keys())
        self.special_token_ids = {
            'ts_start': self.tokenizer.convert_tokens_to_ids('<ts_start>'),
            'ts_end': self.tokenizer.convert_tokens_to_ids('<ts_end>'),
            'text_start': self.tokenizer.convert_tokens_to_ids('<text_start>'),
            'text_end': self.tokenizer.convert_tokens_to_ids('<text_end>'),
            'eos': self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
        }

    def _build_prompt_for_check(self, instruction: str, ts_path: str = None, text: str = None) -> str:
        """根据输入构建符合模型训练格式的提示字符串。"""
        # 注意：第二阶段的模型不是为指令设计的，但我们需要一个引导提示
        # 这里我们使用一个简单的组合，模拟训练时的输入格式
        prompt_parts = []
        if text:
            prompt_parts.append(f"<text_start>{text}<text_end>")
        
        if ts_path:
            if not os.path.exists(ts_path):
                print(f"错误: 找不到时间序列文件 '{ts_path}'")
                return None
            ts_data = np.load(ts_path).astype(np.float32)
            if ts_data.ndim > 1: ts_data = ts_data.flatten()
            
            num_patches = len(ts_data) // PATCH_SIZE
            if num_patches == 0: return "".join(prompt_parts)
            ts_data = ts_data[:num_patches * PATCH_SIZE]
            
            ts_tensor = torch.from_numpy(ts_data).unsqueeze(0).unsqueeze(-1)
            ts_tensor_norm = self.revin(ts_tensor, mode='norm')
            self.input_stats_available = True
            
            ts_tensor_norm = ts_tensor_norm.permute(0, 2, 1)
            
            with torch.no_grad():
                ts_indices = self.vq_vae.encode(ts_tensor_norm.to(self.device)).squeeze(0)
            
            ts_tokens_str = "".join([f"<ts_motif_{i}>" for i in ts_indices])
            prompt_parts.append(f"<ts_start>{ts_tokens_str}</ts_end>")
        
        # 将指令加在最前面，作为一个简单的引导
        return instruction + "".join(prompt_parts)

    def run_inference(self, instruction: str, ts_path: str = None, text: str = None, max_new_tokens=128):
        """执行生成并解析输出。"""
        self.input_stats_available = False
        
        prompt = self._build_prompt_for_check(instruction, ts_path, text)
        if prompt is None: return

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        print("\n--- 正在生成模型输出 ---")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.special_token_ids['eos'],
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False
        )
        
        generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
        
        # 直接解码并打印完整输出，用于定性分析
        full_decoded_output = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        print(f"\n模型原始生成内容:\n---\n{full_decoded_output}\n---")
        
        return full_decoded_output

def main():
    """执行所有检验任务的主函数。"""
    parser = argparse.ArgumentParser(description="检验第二阶段对齐模型的功能。")
    parser.add_argument("--gpu", type=int, default=0, help="要使用的 GPU 索引 (例如, 0, 1, ...)")
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
    
    # 简单的提示语，引导模型进行描述
    ts_to_text_instruction = "This time series can be described as follows: "
    tester.run_inference(instruction=ts_to_text_instruction, ts_path=SAMPLE_NPY_PATH)
    
    print("\n--- 如何解读检验任务1的结果 ---")
    print("好的迹象: 模型生成的文本应该与时间序列的实际特征（如趋势、周期性）大致相符。")
    print("          例如，如果输入一段上升的序列，模型可能会生成 'shows an upward trend' 或 'increasing over time'。")
    print("注意: 这个阶段的模型可能语法不通顺或描述不完美，我们主要关注“相关性”。")
    print("糟糕的迹象: 生成的文本完全不相关，或者只是重复特殊词元。")


    # --- 检验任务 2: 文本 -> 时间序列 ---
    print("\n\n=========================================================")
    print("检验任务 2: 文本到时间序列 (Text -> TS)")
    print("目标: 给模型一段文本描述，看它能否生成对应的时间序列<ts_motif>词元。")
    print("=========================================================")
    
    text_to_ts_instruction = "" # 在这个任务中，文本本身就是提示
    text_description = "A signal with a clear upward trend and some periodic fluctuations."
    
    print(f"输入的文本描述: '{text_description}'")
    
    tester.run_inference(instruction=text_to_ts_instruction, text=text_description)

    print("\n--- 如何解读检验任务2的结果 ---")
    print("好的迹象: 生成的内容中应该包含 <ts_start>...<ts_end> 模块，并且模块内填充了 <ts_motif_...> 词元。")
    print("糟糕的迹象: 完全没有生成 <ts_start> 模块，或者生成的内容是混乱的、不相关的文本。")
    print("=========================================================")


if __name__ == "__main__":
    main()