# tmp.py
import pandas as pd
import numpy as np
import os

# --- 数据预处理脚本 ---
# 功能：将原始CSV格式的时间序列数据转换为Numpy的.npy格式，
#      为VQ-VAE的无监督训练做准备。

# 1. 定义输入和输出文件夹路径
input_dir = 'raw_data/'
output_dir = 'data/unsupervised_ts/'

# 2. 确保输出文件夹存在，如果不存在则创建
os.makedirs(output_dir, exist_ok=True)

# 检查输入文件夹是否存在，如果不存在则给出提示
if not os.path.isdir(input_dir):
    print(f"错误：输入文件夹 '{input_dir}' 不存在。")
    print("请先创建 'raw_data' 文件夹并将您的CSV文件放入其中。")
else:
    # 3. 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_dir):
        # 4. 检查文件是否是CSV文件
        if filename.endswith('.csv'):
            # 构建完整的CSV文件读取路径
            file_path = os.path.join(input_dir, filename)
            
            # 从文件名中提取基本名称（不含.csv后缀），用于命名输出文件
            base_filename = os.path.splitext(filename)[0]
            
            print(f"\n--- 正在处理文件: {filename} ---")
            
            try:
                # 读取CSV文件
                df = pd.read_csv(file_path)
                
                # 5. 遍历DataFrame中的所有列
                for col in df.columns:
                    # 如果列名不是 'date'（通常日期列不是我们需要处理的数值序列）
                    if col.lower() != 'date':
                        # 提取该列数据，并强制转换为float32类型的numpy数组
                        # .values 将Pandas Series转换为Numpy数组
                        ts_data = df[col].values.astype(np.float32)
                        
                        # 构建输出的.npy文件名，格式为：[原文件名]_[列名].npy
                        output_npy_filename = f'{base_filename}_{col}.npy'
                        output_npy_path = os.path.join(output_dir, output_npy_filename)
                        
                        # 6. 使用Numpy的save函数保存为.npy文件
                        np.save(output_npy_path, ts_data)
                        print(f"已将列 '{col}' 保存到: {output_npy_path}")
                        
            except Exception as e:
                print(f"处理文件 {filename} 时发生错误: {e}")

    print("\n--- 所有CSV文件处理完成 ---")