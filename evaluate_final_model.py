# evaluate_final_model.py
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# --- 关键：从您的项目中导入所需的模块 ---
# 假设此脚本与 inference 和 clgm 目录在同一级别
try:
    from inference.generate import Generator
    from clgm.utils.evaluation import calculate_ts_metrics
except ImportError as e:
    print(f"错误：无法导入项目模块。 {e}")
    print("请确保此脚本位于 'clgm-project' 项目的根目录下，并从那里运行它。")
    exit()

# --- 配置区 ---
# 定义一个统一的输出目录，用于存放所有生成的图表和.npy文件
EVAL_OUTPUT_DIR = "evaluation_results"

# --- 任务一：检验“时序到文本”能力 ---
def test_ts_to_text(generator: Generator, ts_path: str):
    """
    定性评估：给定一个时间序列，检验模型生成的文本描述是否准确。
    """
    print("\n=======================================================")
    print("  检验任务 1: 时间序列到文本 (TS -> Text)")
    print("=======================================================")
    
    if not os.path.exists(ts_path):
        print(f"错误：找不到时间序列文件 '{ts_path}'")
        return

    instruction = "Please provide a detailed description of the main features of this time series, including its trend, seasonality, and any notable anomalies."
    print(f"使用的指令: '{instruction}'")
    print(f"输入的时间序列: '{ts_path}'")
    
    # 使用Generator生成文本
    generated_text, _ = generator.generate(instruction=instruction, ts_path=ts_path)

    # 绘制原始时间序列图以供对比
    ts_data = np.load(ts_path)
    plt.figure(figsize=(15, 5))
    plt.plot(ts_data, label='Original Time Series')
    plt.title(f"Input Time Series for TS-to-Text Evaluation\n(File: {os.path.basename(ts_path)})")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    output_image_path = os.path.join(EVAL_OUTPUT_DIR, "ts_to_text_input.png")
    plt.savefig(output_image_path)
    plt.close()

    print("\n--- 检验结果 ---")
    print(f"模型生成的文本描述:\n---\n{generated_text}\n---")
    print(f"原始时间序列的图表已保存至: {output_image_path}")
    
    print("\n--- 如何解读检验结果 ---")
    print("1. 打开上面保存的PNG图片。")
    print("2. 阅读模型生成的文本描述。")
    print("3. 判断：文本描述是否准确地反映了图表中的特征？（例如，趋势是否正确？周期性是否被提及？异常点是否被捕捉？）。")
    print("好的迹象：文本与图表高度一致。糟糕的迹象：描述与事实不符或过于宽泛。")


# --- 任务二：检验“文本到时序”能力 ---
def test_text_to_ts(generator: Generator, text_description: str):
    """
    定性评估：给定一段文本描述，检验模型生成的时间序列是否符合描述。
    """
    print("\n=======================================================")
    print("  检验任务 2: 文本到时间序列 (Text -> TS)")
    print("=======================================================")
    
    instruction = "Generate a time series that matches the following description."
    print(f"使用的指令: '{instruction}'")
    print(f"输入的文本描述: '{text_description}'")

    # 使用Generator生成时间序列
    _, generated_ts = generator.generate(instruction=instruction, text=text_description)
    
    if generated_ts is None:
        print("\n--- 检验失败 ---")
        print("模型未能生成时间序列。请检查模型或输入。")
        return

    # 绘制生成的时间序列图
    plt.figure(figsize=(15, 5))
    plt.plot(generated_ts, label='Generated Time Series', color='green')
    plt.title("Generated Time Series from Text Description")
    # 将描述作为图表的副标题或注释
    plt.suptitle(text_description, y=0.92, fontsize=9, wrap=True)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    output_image_path = os.path.join(EVAL_OUTPUT_DIR, "text_to_ts_output.png")
    plt.savefig(output_image_path)
    plt.close()

    print("\n--- 检验结果 ---")
    print("模型已成功生成时间序列。")
    print(f"生成的时间序列图表已保存至: {output_image_path}")
    
    print("\n--- 如何解读检验结果 ---")
    print("1. 打开上面保存的PNG图片。")
    print("2. 回顾您输入的文本描述。")
    print("3. 判断：图表的走势是否符合文本描述？（例如，如果描述包含“急剧上升”，图表中是否有此特征？）。")


# --- 任务三：检验“时序预测”能力 ---
def test_forecasting(generator: Generator, ts_path: str, horizon: int = 50):
    """
    定量与定性评估：给定一段完整的时序，检验模型对未来的预测能力。
    """
    print("\n=======================================================")
    print(f"  检验任务 3: 时间序列预测 (Forecasting) - 预测步长: {horizon}")
    print("=======================================================")
    
    if not os.path.exists(ts_path):
        print(f"错误：找不到时间序列文件 '{ts_path}'")
        return

    # 1. 加载并切分数据
    full_ts_data = np.load(ts_path)
    if len(full_ts_data) < horizon + 20: # 确保历史数据足够长
        print(f"错误：数据太短（长度{len(full_ts_data)}），无法进行预测。至少需要 {horizon + 20} 个点。")
        return
        
    history_data = full_ts_data[:-horizon]
    ground_truth_future = full_ts_data[-horizon:]
    
    # 将历史数据保存为临时文件，以符合Generator的输入格式
    history_path = os.path.join(EVAL_OUTPUT_DIR, "temp_history.npy")
    np.save(history_path, history_data)
    
    print(f"总数据长度: {len(full_ts_data)} | 历史数据长度: {len(history_data)} | 真实未来长度: {len(ground_truth_future)}")
    
    # 2. 使用Generator进行预测
    instruction = f"Based on the provided history, forecast the next {horizon} time steps."
    _, predicted_future = generator.generate(instruction=instruction, ts_path=history_path, max_new_tokens=horizon + 10)

    # 清理临时文件
    os.remove(history_path)

    if predicted_future is None:
        print("\n--- 检验失败 ---")
        print("模型未能生成预测序列。")
        return

    # 3. 定量评估
    print("\n--- 定量评估结果 ---")
    metrics = calculate_ts_metrics(ground_truth_future, predicted_future)
    print(f"  均方误差 (MSE): {metrics['MSE']:.6f}")
    print(f"  平均绝对误差 (MAE): {metrics['MAE']:.6f}")
    
    # 4. 定性评估 (绘图)
    plt.figure(figsize=(18, 6))
    
    # 绘制历史数据
    plt.plot(np.arange(len(history_data)), history_data, label='History', color='blue')
    
    # 绘制真实未来
    future_range = np.arange(len(history_data), len(full_ts_data))
    plt.plot(future_range, ground_truth_future, label='Ground Truth Future', color='green', marker='.')
    
    # 绘制预测未来
    # 确保预测长度与真实未来对齐以供比较
    pred_len = min(len(predicted_future), horizon)
    plt.plot(future_range[:pred_len], predicted_future[:pred_len], label='Predicted Future', color='red', linestyle='--', marker='x')
    
    plt.title("Forecasting Performance Evaluation")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    output_image_path = os.path.join(EVAL_OUTPUT_DIR, "forecasting_comparison.png")
    plt.savefig(output_image_path)
    plt.close()

    print("\n--- 定性评估结果 ---")
    print(f"预测对比图已保存至: {output_image_path}")

    print("\n--- 如何解读检验结果 ---")
    print("1. 查看MSE和MAE指标。值越小，说明预测越精准。")
    print("2. 打开PNG图片，观察红色虚线（预测）与绿色实线（真实）的贴合程度。")
    print("好的迹象：两条线走势非常接近。糟糕的迹象：两条线走势差异巨大。")


def main():
    parser = argparse.ArgumentParser(
        description="检验最终CLGM三阶段模型的效果。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # --- 通用参数 ---
    parser.add_argument("--gpu", type=int, default=0, help="要使用的GPU索引 (例如, 0, 1, ...)")

    # --- 创建子命令，用于选择不同的检验任务 ---
    subparsers = parser.add_subparsers(dest="task", required=True, help="选择要执行的检验任务")

    # 子命令 1: ts_to_text
    parser_t2t = subparsers.add_parser("ts_to_text", help="检验时间序列到文本的生成能力")
    parser_t2t.add_argument("--ts_path", type=str, required=True, help="输入的时间序列.npy文件路径")
    
    # 子命令 2: text_to_ts
    parser_txt2ts = subparsers.add_parser("text_to_ts", help="检验文本到时间序列的生成能力")
    parser_txt2ts.add_argument("--text", type=str, required=True, help="输入的文本描述 (请使用引号括起来)")
    
    # 子命令 3: forecast
    parser_forecast = subparsers.add_parser("forecast", help="检验时间序列预测能力")
    parser_forecast.add_argument("--ts_path", type=str, required=True, help="输入的全量时间序列.npy文件路径，脚本会自动切分")
    parser_forecast.add_argument("--horizon", type=int, default=50, help="要预测的未来步长")

    args = parser.parse_args()
    
    # --- 初始化 ---
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"--- 使用设备: {device} ---")
    
    # 加载最终的生成器模型
    # 注意：Generator会从configs/config.py中的INFERENCE_CONFIG加载模型路径
    # 请确保INFERENCE_CONFIG指向您最终的、第三阶段训练好的模型！
    try:
        generator = Generator(device=device)
    except Exception as e:
        print(f"\n错误：在初始化Generator时失败: {e}")
        print("请检查您的'configs/config.py'中的INFERENCE_CONFIG是否正确配置，并指向最终模型路径。")
        return
        
    # --- 根据选择的任务执行相应的函数 ---
    if args.task == "ts_to_text":
        test_ts_to_text(generator, args.ts_path)
    elif args.task == "text_to_ts":
        test_text_to_ts(generator, args.text)
    elif args.task == "forecast":
        test_forecasting(generator, args.ts_path, args.horizon)

if __name__ == "__main__":
    main()