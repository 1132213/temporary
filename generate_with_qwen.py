# generate_with_qwen.py
import numpy as np
import json
import os
import random
from tqdm import tqdm
import argparse
import re
import multiprocessing
import time

PROMPT_TEMPLATES = [
    ("Your task is to rewrite the following bullet points into a single, cohesive paragraph. You MUST incorporate every fact listed.", "Synthesized Paragraph:"),
    ("Weave together all of the following factual points about a time series into a fluent, descriptive paragraph. Do not omit any details.", "Comprehensive Description:"),
    ("Combine the following distinct observations about a time series into one paragraph. Ensure every point is mentioned.", "Combined Summary:"),
    ("You are a summarization AI. Your input is a list of technical facts. Your output must be a natural language paragraph that includes all of the information from the input list.", "Full Summary:"),
    ("Convert the following checklist of time series features into a well-written paragraph. Make sure every item on the checklist is covered in your response.", "Narrative Version:")
]

def generate_complex_timeseries(length):
    time = np.arange(length)
    timeseries = np.zeros(length)
    ground_truth_features = {}
    if random.random() > 0.1:
        trend_type = random.choice(['steady increase', 'gentle decrease', 'v-shape', 'flat'])
        if trend_type == 'steady increase':
            slope = random.uniform(0.005, 0.02)
            timeseries += time * slope
            ground_truth_features['trend'] = "it exhibits a steady linear increasing trend"
        elif trend_type == 'gentle decrease':
            slope = random.uniform(-0.02, -0.005)
            timeseries += time * slope
            ground_truth_features['trend'] = "it exhibits a gentle linear decreasing trend"
        else:
             ground_truth_features['trend'] = "it is relatively flat with no significant trend"
    if random.random() > 0.1:
        period = random.choice([24, 48, 96])
        amplitude = random.uniform(1.0, 6.0)
        timeseries += np.sin(2 * np.pi * time / period) * amplitude
        ground_truth_features['seasonality'] = f"it has a clear seasonal pattern with a period of {period}"
    if random.random() > 0.5:
        pos = random.randint(int(length * 0.1), int(length * 0.9))
        magnitude = random.uniform(8, 20) * random.choice([-1, 1])
        timeseries[pos] += magnitude
        anomaly_type = "an upward spike" if magnitude > 0 else "a downward dip"
        ground_truth_features['anomaly'] = f"there is a significant {anomaly_type} around timestep {pos}"
    noise_level = random.uniform(0.1, 0.6)
    timeseries += np.random.randn(length) * noise_level
    ground_truth_features['noise'] = f"it is subject to a baseline level of random noise"
    if not ground_truth_features:
        ground_truth_features['overall'] = "it is a simple, noisy signal"
    return timeseries, ground_truth_features

def run_generation_on_gpu(gpu_id, num_gpus, gpu_rank, total_samples, batch_size, output_dir, local_llm_path, min_len, max_len):
    print(f"--- [Process for GPU {gpu_id}] Rank {gpu_rank}/{num_gpus} starting. ---")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from vllm import LLM, SamplingParams
    
    samples_per_gpu = total_samples // num_gpus
    start_index = gpu_rank * samples_per_gpu
    end_index = start_index + samples_per_gpu
    if gpu_rank == num_gpus - 1:
        end_index = total_samples
    num_samples_for_this_process = end_index - start_index
    print(f"--- [Process for GPU {gpu_id}] Will generate {num_samples_for_this_process} samples (from index {start_index} to {end_index-1}). ---")
    print(f"--- [Process for GPU {gpu_id}] Time series length will be random between {min_len} and {max_len}. ---")
    npy_output_path = os.path.join(output_dir, "npy_files")
    jsonl_filename = f"alignment_data_en_gpu{gpu_rank}.jsonl"
    output_jsonl_path = os.path.join(output_dir, jsonl_filename)
    all_timeseries_data = []
    all_prompts = []
    for _ in tqdm(range(num_samples_for_this_process), desc=f"Generating Prompts for GPU {gpu_id}"):
        current_length = random.randint(min_len, max_len)
        ts_data, ts_features = generate_complex_timeseries(length=current_length)
        all_timeseries_data.append(ts_data)
        fact_list = [f"- {desc}." for desc in ts_features.values()]
        facts_as_string = "\n".join(fact_list)
        instruction, response_prefix = random.choice(PROMPT_TEMPLATES)
        prompt = f"""{instruction}\n\n**Facts to incorporate:**\n{facts_as_string}\n\n**{response_prefix}**"""
        all_prompts.append(prompt)
    llm = LLM(model=local_llm_path, tensor_parallel_size=1, trust_remote_code=True, gpu_memory_utilization=0.85)
    all_outputs = []
    sampling_params = SamplingParams(temperature=0.5, top_p=0.9, max_tokens=256)
    for i in tqdm(range(0, len(all_prompts), batch_size), desc=f"LLM Inference on GPU {gpu_id}"):
        batch = all_prompts[i:i + batch_size]
        request_outputs = llm.generate(batch, sampling_params)
        for output in request_outputs:
            all_outputs.append(output.outputs[0].text.strip())
    prefixes_to_clean = [prefix for _, prefix in PROMPT_TEMPLATES]
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for i, ts_data in enumerate(tqdm(all_timeseries_data, desc=f"Cleaning & Saving for GPU {gpu_id}")):
            global_index = start_index + i
            npy_filename = f"series_en_{global_index}.npy"
            npy_filepath = os.path.join(npy_output_path, npy_filename)
            np.save(npy_filepath, ts_data.astype(np.float32))
            text = all_outputs[i].split('\n')[0]
            for prefix in prefixes_to_clean:
                clean_prefix = prefix.replace(":", "").strip()
                text = re.sub(r'(\*\*|)?' + re.escape(clean_prefix) + r'(\*\*|)?:\s*', '', text, count=1, flags=re.IGNORECASE)
            clean_text = re.sub(r'[^\w\.\?!]+$', '', text).strip()
            output_data = {"text": clean_text, "ts_path": npy_filepath}
            f.write(json.dumps(output_data, ensure_ascii=False) + '\n')
    print(f"--- [Process for GPU {gpu_id}] Task complete. Results saved to {output_jsonl_path} ---")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Run parallel data generation across multiple GPUs.")
    parser.add_argument('--gpus', type=str, default="0,1", help='A comma-separated list of GPU IDs to use (e.g., "0,1,2,3").')
    main_args = parser.parse_args()
    GPUS_TO_USE = [int(x) for x in main_args.gpus.split(',')]
    NUM_GPUS = len(GPUS_TO_USE)
    TOTAL_SAMPLES_TO_GENERATE = 20000
    BATCH_SIZE = 128
    OUTPUT_DIR = "data/qwen_generated_dataset_en_merge/"
    LOCAL_LLM_PATH = "/root/emhua/btwu/Qwen14B"
    MIN_LENGTH = 128
    MAX_LENGTH = 512
    if os.path.exists(OUTPUT_DIR):
        print(f"--- Cleaning up previous output directory: {OUTPUT_DIR} ---")
        for root, dirs, files in os.walk(OUTPUT_DIR, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(OUTPUT_DIR)
    os.makedirs(os.path.join(OUTPUT_DIR, "npy_files"), exist_ok=True)
    print(f"--- Starting data generation on {NUM_GPUS} GPUs: {GPUS_TO_USE} ---")
    processes = []
    for rank, gpu_id in enumerate(GPUS_TO_USE):
        process_args = (gpu_id, NUM_GPUS, rank, TOTAL_SAMPLES_TO_GENERATE, BATCH_SIZE, OUTPUT_DIR, LOCAL_LLM_PATH, MIN_LENGTH, MAX_LENGTH)
        p = multiprocessing.Process(target=run_generation_on_gpu, args=process_args)
        processes.append(p)
        p.start()
        time.sleep(5)
    for p in processes:
        p.join()
    print("\n--- All parallel generation processes have completed. ---")
    print("--- Merging final dataset... ---")
    merged_filepath = os.path.join(OUTPUT_DIR, "alignment_data_en_merged.jsonl")
    with open(merged_filepath, 'wb') as wfd:
        for rank in range(NUM_GPUS):
            chunk_filepath = os.path.join(OUTPUT_DIR, f"alignment_data_en_gpu{rank}.jsonl")
            if os.path.exists(chunk_filepath):
                with open(chunk_filepath, 'rb') as rfd:
                    wfd.write(rfd.read())
                os.remove(chunk_filepath)
    print(f"--- Success! All data merged into: {merged_filepath} ---")