import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_normalized_indicators_heatmap(input_paths, output_dir):
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
        
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()

        if df.empty:
            print("处理后的标准化指标数据为空，无法绘制。")
            return

        plt.figure(figsize=(12, 8))
        sns.heatmap(df, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=.5, linecolor='black', cbar_kws={'label': '标准化值'})
        
        plt.title('各国标准化风险指标热力图', fontsize=16, fontweight='bold')
        plt.xlabel('风险指标', fontsize=12)
        plt.ylabel('国家', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission2_normalized_indicators_heatmap.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制标准化指标热力图时发生错误: {e}")

if __name__ == "__main__":
    test_input_paths = {'mission2_normalized_indicators_csv': "outputs/mission2/normalized_indicators.csv"}
    test_output_dir = "outputs/figures/png"
    plot_normalized_indicators_heatmap(test_input_paths, test_output_dir)
