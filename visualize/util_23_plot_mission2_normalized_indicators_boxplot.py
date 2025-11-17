import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_normalized_indicators_boxplot(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    csv_file_path = input_paths['mission2_normalized_indicators_csv']
    if not os.path.exists(csv_file_path):
        print(f"错误: 任务二标准化指标数据文件未找到: {csv_file_path}")
        return

    try:
        # 应用 seaborn 主题
        sns.set_theme(style="whitegrid", palette="viridis")

        # --- 在 sns.set_theme() 之后设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        df = pd.read_csv(csv_file_path, index_col=0)
        
        df_plot = df.apply(pd.to_numeric, errors='coerce')
        df_plot = df_plot.dropna()

        if df_plot.empty:
            print("处理后的标准化指标数据为空，无法绘制。")
            return

        plt.figure(figsize=(14, 8))
        sns.boxplot(data=df_plot, palette='viridis')
        
        plt.title('各国标准化风险指标箱线图', fontsize=16, fontweight='bold')
        plt.xlabel('风险指标', fontsize=12)
        plt.ylabel('标准化值', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission2_normalized_indicators_boxplot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制标准化指标箱线图时发生错误: {e}")

if __name__ == "__main__":
    test_input_paths = {'mission2_normalized_indicators_csv': "outputs/mission2/normalized_indicators.csv"}
    test_output_dir = "outputs/figures/png"
    plot_normalized_indicators_boxplot(test_input_paths, test_output_dir)
