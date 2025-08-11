# generate_with_qwen.py (最终修复版)
import numpy as np
import json
import os
import random
from tqdm import tqdm
import argparse # 1. 导入argparse

# --- 2. 关键！在导入torch或vllm之前，根据命令行参数设置可见的GPU ---
parser = argparse.ArgumentParser(description="Generate high-quality dataset using Qwen-14B and vLLM.")
parser.add_argument(
    "--gpu-id",
    type=int,
    default=0,
    help="The ID of the GPU to use for inference."
)
args = parser.parse_args()

print(f"--- 将要使用的GPU ID是: {args.gpu_id} ---")
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# 现在再导入vLLM
from vllm import LLM, SamplingParams

# --- 配置区 ---

# 1. 模型与数据配置
LOCAL_LLM_PATH = "/root/emhua/btwu/Qwen14B"
NUM_SAMPLES_TO_GENERATE = 20000
BATCH_SIZE = 128 

# 2. 输出目录配置
OUTPUT_DIR = "data/qwen_generated_dataset_en/"
NPY_SUBDIR = "npy_files"
JSONL_FILENAME = "alignment_data_en.jsonl"

NPY_OUTPUT_PATH = os.path.join(OUTPUT_DIR, NPY_SUBDIR)
os.makedirs(NPY_OUTPUT_PATH, exist_ok=True)
OUTPUT_JSONL_PATH = os.path.join(OUTPUT_DIR, JSONL_FILENAME)


# ... (generate_complex_timeseries 和 generate_descriptions_in_batch 函数保持不变) ...
def generate_complex_timeseries(length=256):
    """
    Generates a complex time series and a dictionary of its ground-truth features in English.
    """
    time = np.arange(length)
    timeseries = np.zeros(length)
    ground_truth_features = {}

    # Trend Component
    trend_type = random.choice(['steady increase', 'gentle decrease', 'v-shape', 'a-shape', 'flat'])
    if trend_type == 'steady increase':
        slope = random.uniform(0.005, 0.02)
        timeseries += time * slope
        ground_truth_features['trend'] = f"a steady linear increasing trend with a slope of {slope:.3f}"
    elif trend_type == 'gentle decrease':
        slope = random.uniform(-0.02, -0.005)
        timeseries += time * slope
        ground_truth_features['trend'] = f"a gentle linear decreasing trend"
    
    # Seasonality Component
    num_seasonalities = random.randint(1, 2)
    seasons = []
    for _ in range(num_seasonalities):
        period = random.choice([24, 48, 168]) # e.g., daily, bi-daily, weekly
        amplitude = random.uniform(0.8, 4)
        timeseries += np.sin(2 * np.pi * time / period) * amplitude
        seasons.append(f"a regular fluctuation with a period of {period} and an amplitude of {amplitude:.2f}")
    if seasons:
        ground_truth_features['seasonality'] = " and ".join(seasons)

    # Noise Component
    noise_level = random.uniform(0.1, 0.6)
    timeseries += np.random.randn(length) * noise_level
    ground_truth_features['noise'] = f"a baseline noise with a standard deviation of approximately {noise_level:.2f}"

    # Anomaly Component
    num_anomalies = random.randint(0, 3)
    anomalies = []
    for _ in range(num_anomalies):
        pos = random.randint(int(length * 0.1), int(length * 0.9))
        if random.random() > 0.5:
            magnitude = random.uniform(8, 20)
            timeseries[pos] += magnitude
            anomalies.append(f"a significant upward spike around timestep {pos}")
        else:
            magnitude = random.uniform(-20, -8)
            timeseries[pos] += magnitude
            anomalies.append(f"a sudden downward dip around timestep {pos}")
    if anomalies:
        ground_truth_features['anomalies'] = ", and also ".join(anomalies)

    return timeseries, ground_truth_features


def generate_descriptions_in_batch(llm_engine, prompts, batch_size):
    """
    Uses the vLLM engine to generate descriptions for a list of prompts in batches.
    """
    all_outputs = []
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)

    for i in tqdm(range(0, len(prompts), batch_size), desc="LLM Batch-Inferencing"):
        batch = prompts[i:i + batch_size]
        request_outputs = llm_engine.generate(batch, sampling_params)
        for output in request_outputs:
            all_outputs.append(output.outputs[0].text.strip())
            
    return all_outputs


if __name__ == "__main__":
    # --- Part 1: 生成所有的时间序列和Prompts ---
    print(f"--- Part 1: Generating {NUM_SAMPLES_TO_GENERATE} timeseries and prompts ---")
    all_timeseries_data = []
    all_prompts = []
    for _ in tqdm(range(NUM_SAMPLES_TO_GENERATE), desc="Generating timeseries"):
        ts_data, ts_features = generate_complex_timeseries()
        all_timeseries_data.append(ts_data)
        
        fact_list = [f"- Trend: {ts_features.get('trend', 'none')}."]
        if 'seasonality' in ts_features: fact_list.append(f"- Seasonality: {ts_features['seasonality']}.")
        if 'anomalies' in ts_features: fact_list.append(f"- Anomalies: {ts_features['anomalies']}.")
        fact_list.append(f"- Noise: {ts_features.get('noise', 'none')}.")
        facts_as_string = "\n".join(fact_list)

        prompt = f"""You are a professional data scientist writing a report. Your task is to summarize the key characteristics of a time series based on the following technical facts. Write a concise, fluent, and professional paragraph in English.

**Technical Facts:**
{facts_as_string}

**Analyst's Summary:**
"""
        all_prompts.append(prompt)

    # --- Part 2: 初始化vLLM并批量生成文本描述 ---
    print(f"\n--- Part 2: Initializing Qwen-14B with vLLM (this may take a moment) ---")
    llm = LLM(
        model=LOCAL_LLM_PATH,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.85  # 3. 降低显存占用率
    )
    
    print(f"\n--- Part 3: Generating text descriptions in batches ---")
    generated_texts = generate_descriptions_in_batch(llm, all_prompts, BATCH_SIZE)

    # --- Part 4: 保存最终结果 ---
    print(f"\n--- Part 4: Saving final dataset to {OUTPUT_JSONL_PATH} ---")
    with open(OUTPUT_JSONL_PATH, 'w', encoding='utf-8') as f:
        for i, ts_data in enumerate(tqdm(all_timeseries_data, desc="Saving results")):
            npy_filename = f"series_en_{i}.npy"
            npy_filepath = os.path.join(NPY_OUTPUT_PATH, npy_filename)
            np.save(npy_filepath, ts_data.astype(np.float32))
            
            output_data = {
                "text": generated_texts[i],
                "ts_path": npy_filepath
            }
            f.write(json.dumps(output_data, ensure_ascii=False) + '\n')

    print(f"\n--- High-quality English dataset generation complete! ---")