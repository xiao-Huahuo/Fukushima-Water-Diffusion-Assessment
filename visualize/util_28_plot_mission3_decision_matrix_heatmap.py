import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_decision_matrix_heatmap(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    csv_file_path = input_paths['mission3_nsga2_results_csv']
    if not os.path.exists(csv_file_path):
        print(f"错误: 任务三NSGA-II结果文件未找到: {csv_file_path}")
        return

    try:
        # 应用 seaborn 主题
        sns.set_theme(style="whitegrid", palette="viridis")

        # --- 在 sns.set_theme() 之后设置字体 ---
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

        # 构建决策矩阵
        decision_matrix = df_unique_schemes[['环境影响E30', '成本C', '达标时间t\'']]
        decision_matrix.columns = ['环境影响', '总成本', '达标时间']

        # 对数据进行标准化，以便在热图中更好地比较
        # 目标都是越小越好，所以直接Min-Max标准化到 [0, 1]
        normalized_matrix = decision_matrix.copy()
        for col in decision_matrix.columns:
            min_val = decision_matrix[col].min()
            max_val = decision_matrix[col].max()
            if max_val - min_val > 1e-9:
                normalized_matrix[col] = (decision_matrix[col] - min_val) / (max_val - min_val)
            else:
                normalized_matrix[col] = 0.5 # 如果所有值相同，设为中间值

        plt.figure(figsize=(8, 6))
        sns.heatmap(normalized_matrix, annot=True, cmap='RdYlGn_r', fmt=".2f", linewidths=.5, linecolor='black',
                    cbar_kws={'label': '标准化目标值 (越绿越好)'}) # _r 表示反转颜色，使得小值对应绿色

        plt.title('处理方案决策矩阵热图 (标准化目标值)', fontsize=16, fontweight='bold')
        plt.xlabel('目标', fontsize=12)
        plt.ylabel('处理方案', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission3_decision_matrix_heatmap.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制决策矩阵热图时发生错误: {e}")

if __name__ == "__main__":
    test_input_paths = {'mission3_nsga2_results_csv': "outputs/mission3/nsga2_results.csv"}
    test_output_dir = "outputs/figures/png"
    plot_decision_matrix_heatmap(test_input_paths, test_output_dir)
