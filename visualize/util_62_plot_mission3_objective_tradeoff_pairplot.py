import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_objective_tradeoff_pairplot(input_paths, output_dir):
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
        
        # 绘制pairplot
        g = sns.pairplot(df_unique_schemes, vars=['环境影响E30', '成本C', '达标时间t\''], 
                         hue='方案', palette='viridis', diag_kind='kde',
                         plot_kws={'s': 100, 'alpha': 0.8, 'edgecolor': 'black'})
        
        g.fig.suptitle('处理方案目标权衡两两关系图', y=1.02, fontsize=16, fontweight='bold') # y调整标题位置

        # 调整轴标签和刻度
        for i, ax in enumerate(g.axes.flatten()):
            if i % g.n_vars == 0: # 左侧Y轴
                ax.set_ylabel(ax.get_ylabel(), fontsize=12)
            if i >= g.n_vars * (g.n_vars - 1): # 底部X轴
                ax.set_xlabel(ax.get_xlabel(), fontsize=12)
            ax.tick_params(labelsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        plt.tight_layout(rect=[0, 0, 1, 0.98]) # 调整布局以适应suptitle

        output_path = os.path.join(output_dir, "mission3_objective_tradeoff_pairplot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制目标权衡两两关系图时发生错误: {e}")

if __name__ == "__main__":
    # 模拟数据用于测试
    test_df = pd.DataFrame({
        '方案': [1, 2, 3],
        '环境影响E30': [1.0e12, 5.0e11, 1.0e11],
        '成本C': [5.0e9, 8.0e9, 12.0e9],
        '达标时间t\'': [100.0, 50.0, 10.0]
    })
    test_output_dir_temp = "outputs/mission3/"
    os.makedirs(test_output_dir_temp, exist_ok=True)
    test_csv_path = os.path.join(test_output_dir_temp, "nsga2_results.csv")
    test_df.to_csv(test_csv_path, index=False)

    test_input_paths = {'mission3_nsga2_results_csv': test_csv_path}
    test_output_dir = "outputs/figures/png"
    plot_objective_tradeoff_pairplot(test_input_paths, test_output_dir)
