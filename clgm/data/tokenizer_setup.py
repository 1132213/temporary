# clgm/data/tokenizer_setup.py
import os
from transformers import AutoTokenizer
from configs.config import LLM_MODEL_NAME, TS_MOTIF_PREFIX, VQ_CODEBOOK_SIZE, SPECIAL_TOKENS, LLM_FINETUNE_CONFIG

def setup_tokenizer(save_path: str):
    """
    加载一个预训练的分词器，添加新的特殊词元和时间序列词元，
    然后将扩展后的分词器保存到指定路径。

    Args:
        save_path (str): 用于保存扩展后分词器的目录。
    """
    print(f"正在加载基础分词器: {LLM_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

    new_special_tokens = list(SPECIAL_TOKENS.values())
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = SPECIAL_TOKENS["pad_token"]
        print(f"已将 pad_token 设置为 {SPECIAL_TOKENS['pad_token']}")

    temporal_tokens = [f"{TS_MOTIF_PREFIX}{i}>" for i in range(VQ_CODEBOOK_SIZE)]
    
    num_added_toks = tokenizer.add_tokens(new_special_tokens + temporal_tokens, special_tokens=True)
    print(f"已向分词器中添加 {num_added_toks} 个新词元。")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"扩展后的分词器已保存至: {save_path}")

if __name__ == "__main__":
    tokenizer_save_path = os.path.join(LLM_FINETUNE_CONFIG["clgm_checkpoint_dir"], "tokenizer")
    setup_tokenizer(tokenizer_save_path)