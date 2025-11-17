import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_pareto_front_scatter_matrix(input_paths, output_dir):
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
        df_unique_schemes = df.drop_duplicates(subset=['方案'])
        
        # 绘制散点矩阵图
        g = sns.PairGrid(df_unique_schemes, vars=['环境影响E30', '成本C', '达标时间t\''], 
                         hue='方案', palette='viridis', diag_kind='kde')
        g.map_upper(sns.scatterplot, s=100, edgecolor='black', alpha=0.8)
        g.map_lower(sns.kdeplot, fill=True)
        g.map_diag(sns.histplot, kde=True)
        
        g.add_legend(title='处理方案')
        g.fig.suptitle('处理方案帕累托前沿散点矩阵图', y=1.02, fontsize=16, fontweight='bold') # y调整标题位置

        # 调整轴标签
        for i, ax in enumerate(g.axes.flatten()):
            if i % g.n_vars == 0: # 左侧Y轴
                ax.set_ylabel(ax.get_ylabel(), fontsize=12)
            if i >= g.n_vars * (g.n_vars - 1): # 底部X轴
                ax.set_xlabel(ax.get_xlabel(), fontsize=12)
            ax.tick_params(labelsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout(rect=[0, 0, 1, 0.98]) # 调整布局以适应suptitle

        output_path = os.path.join(output_dir, "mission3_pareto_front_scatter_matrix.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制帕累托前沿散点矩阵图时发生错误: {e}")

if __name__ == "__main__":
    test_input_paths = {'mission3_nsga2_results_csv': "outputs/mission3/nsga2_results.csv"}
    test_output_dir = "outputs/figures/png"
    plot_pareto_front_scatter_matrix(test_input_paths, test_output_dir)
