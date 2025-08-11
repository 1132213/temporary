# check_data_quality.py
# 更新版：将图表保存为文件，以适应SSH环境

import json
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# --- 配置区 ---
# 指向您在第二阶段训练时使用的那个 .jsonl 文件
DATA_JSONL_PATH = "./data/paired_ts_text/alignment_data.jsonl" 

# 新增：定义一个文件夹来存放所有生成的检验图表
OUTPUT_IMAGE_DIR = "data_quality_check_plots"

# 您想抽查多少个样本？
NUM_SAMPLES_TO_CHECK = 10 # 建议可以适当增加抽查数量

def check_samples():
    # 确保输出目录存在
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    print(f"--- 图表将保存在 '{OUTPUT_IMAGE_DIR}/' 文件夹中 ---")

    if not os.path.exists(DATA_JSONL_PATH):
        print(f"错误: 找不到数据文件 '{DATA_JSONL_PATH}'。请确认路径是否正确。")
        return

    # 先读取所有数据行
    with open(DATA_JSONL_PATH, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    if len(all_lines) < NUM_SAMPLES_TO_CHECK:
        print(f"数据量不足 {NUM_SAMPLES_TO_CHECK} 条，将检查所有数据。")
        samples_to_check_indices = range(len(all_lines))
    else:
        # 随机抽取样本的索引
        samples_to_check_indices = random.sample(range(len(all_lines)), NUM_SAMPLES_TO_CHECK)

    print(f"--- 随机抽查 {len(samples_to_check_indices)} 个样本 ---")

    for i, line_idx in enumerate(samples_to_check_indices):
        line = all_lines[line_idx]
        data = json.loads(line)
        text_description = data.get("text")
        ts_path = data.get("ts_path")
        
        # 为保存的图片文件创建一个唯一的文件名
        output_image_path = os.path.join(OUTPUT_IMAGE_DIR, f"sample_{i+1}_line_{line_idx}.png")

        print(f"\n--- 正在处理样本 {i+1} (来自原始文件第 {line_idx+1} 行) ---")
        print(f"文本描述: {text_description}")
        print(f"时间序列路径: {ts_path}")

        try:
            # 加载并绘制时间序列
            ts_data = np.load(ts_path)
            plt.figure(figsize=(15, 5))
            # 将描述文本直接作为图表标题，方便对比
            plt.title(f"Description: {text_description}", wrap=True) 
            plt.plot(ts_data)
            plt.grid(True)
            plt.xlabel("Time Step")
            plt.ylabel("Value")
            
            # --- 关键修改：保存图表到文件，而不是显示它 ---
            plt.savefig(output_image_path)
            plt.close() # 关闭图表以释放内存，在循环中很重要
            
            print(f"✅ 图表已保存至: {output_image_path}")

        except FileNotFoundError:
            print(f"!! 警告: 找不到 .npy 文件 '{ts_path}' !!")
        except Exception as e:
            print(f"处理或绘图时发生错误: {e}")

if __name__ == "__main__":
    check_samples()