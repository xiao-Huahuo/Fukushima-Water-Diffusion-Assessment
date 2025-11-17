import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import os
from mpl_toolkits.mplot3d import Axes3D 
import seaborn as sns

def plot_pareto_front_3d_scatter(input_paths, output_dir):
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

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        schemes = sorted(df['方案'].unique()) # 确保方案顺序
        colors = plt.cm.get_cmap('viridis', len(schemes)) # 使用viridis颜色映射

        for i, scheme in enumerate(schemes):
            scheme_df = df[df['方案'] == scheme]
            ax.scatter(scheme_df['环境影响E30'], scheme_df['成本C'], scheme_df['达标时间t\''],
                       color=colors(i), label=f'方案 {scheme}', s=100, alpha=0.8, edgecolors='w')

        ax.set_title('任务三：NSGA-II 帕累托前沿 (环境影响 vs. 成本 vs. 达标时间)', fontsize=16, fontweight='bold')
        ax.set_xlabel('环境影响 E30', fontsize=12)
        ax.set_ylabel('总成本 C', fontsize=12) # 明确为总成本
        ax.set_zlabel('最大达标时间 t\' (年)', fontsize=12)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='z', labelsize=10)
        ax.legend(title='处理方案', loc='best', fontsize=10, title_fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        # 调整视角以获得更好的可视化效果
        ax.view_init(elev=20, azim=-60) # 可以根据需要调整

        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission3_pareto_front_3d_scatter.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制帕累托前沿三维散点图时发生错误: {e}")

if __name__ == "__main__":
    test_input_paths = {'mission3_nsga2_results_csv': "outputs/mission3/nsga2_results.csv"}
    test_output_dir = "outputs/figures/png"
    plot_pareto_front_3d_scatter(test_input_paths, test_output_dir)
