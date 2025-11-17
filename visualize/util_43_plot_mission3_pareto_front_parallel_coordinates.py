import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from pandas.plotting import parallel_coordinates

def plot_pareto_front_parallel_coordinates(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    csv_file_path = input_paths['mission3_nsga2_results_csv']
    if not os.path.exists(csv_file_path):
        print(f"错误: 任务三NSGA-II结果文件未找到: {csv_file_path}")
        return

    try:
        # --- 设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        df = pd.read_csv(csv_file_path)
        
        if df.empty:
            print("NSGA-II结果数据为空，无法绘制。")
            return

        df['环境影响E30'] = pd.to_numeric(df['环境影响E30'], errors='coerce')
        df['成本C'] = pd.to_numeric(df['成本C'], errors='coerce')
        df['达标时间t\''] = pd.to_numeric(df['达标时间t\''], errors='coerce')
        df = df.dropna()

        # 确保每个方案只有一个结果 (如果NSGA-II结果包含重复方案)
        df_unique_schemes = df.drop_duplicates(subset=['方案']).set_index('方案')

        # 对目标函数进行标准化，以便在平行坐标图中比较
        # 目标都是越小越好，所以直接Min-Max标准化到 [0, 1]
        df_normalized = df_unique_schemes[['环境影响E30', '成本C', '达标时间t\'']].copy()
        for col in df_normalized.columns:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            if max_val - min_val > 1e-9:
                df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
            else:
                df_normalized[col] = 0.5 # 如果所有值相同，设为中间值
        
        df_normalized['方案'] = df_normalized.index.map(lambda x: f'方案 {int(x)}')

        plt.figure(figsize=(12, 8))
        parallel_coordinates(df_normalized, '方案', 
                             color=sns.color_palette("viridis", n_colors=len(df_normalized)),
                             linewidth=3, alpha=0.8)
        
        plt.title('处理方案帕累托前沿平行坐标图 (标准化目标值)', fontsize=16, fontweight='bold')
        plt.xlabel('目标', fontsize=12)
        plt.ylabel('标准化值', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title='处理方案', loc='upper right', bbox_to_anchor=(1.2, 1), fontsize=10, title_fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission3_pareto_front_parallel_coordinates.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制帕累托前沿平行坐标图时发生错误: {e}")

if __name__ == "__main__":
    test_input_paths = {'mission3_nsga2_results_csv': "outputs/mission3/nsga2_results.csv"}
    test_output_dir = "outputs/figures/png"
    plot_pareto_front_parallel_coordinates(test_input_paths, test_output_dir)
