import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_pareto_front_2d_projections(input_paths, output_dir):
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

        schemes = sorted(df['方案'].unique())
        colors = plt.cm.get_cmap('viridis', len(schemes))

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        axes = axes.flatten()

        # 投影1: 成本 vs 环境影响
        ax = axes[0]
        for i, scheme in enumerate(schemes):
            scheme_df = df[df['方案'] == scheme]
            sns.scatterplot(x='环境影响E30', y='成本C', data=scheme_df, ax=ax,
                            color=colors(i), label=f'方案 {scheme}', s=100, alpha=0.8, edgecolors='w')
        ax.set_title('成本 vs. 环境影响', fontsize=14, fontweight='bold')
        ax.set_xlabel('环境影响 E30', fontsize=12)
        ax.set_ylabel('总成本 C', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)

        # 投影2: 成本 vs 达标时间
        ax = axes[1]
        for i, scheme in enumerate(schemes):
            scheme_df = df[df['方案'] == scheme]
            sns.scatterplot(x='达标时间t\'', y='成本C', data=scheme_df, ax=ax,
                            color=colors(i), label=f'方案 {scheme}', s=100, alpha=0.8, edgecolors='w')
        ax.set_title('成本 vs. 达标时间', fontsize=14, fontweight='bold')
        ax.set_xlabel('最大达标时间 t\' (年)', fontsize=12)
        ax.set_ylabel('总成本 C', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)

        # 投影3: 环境影响 vs 达标时间
        ax = axes[2]
        for i, scheme in enumerate(schemes):
            scheme_df = df[df['方案'] == scheme]
            sns.scatterplot(x='环境影响E30', y='达标时间t\'', data=scheme_df, ax=ax,
                            color=colors(i), label=f'方案 {scheme}', s=100, alpha=0.8, edgecolors='w')
        ax.set_title('环境影响 vs. 达标时间', fontsize=14, fontweight='bold')
        ax.set_xlabel('环境影响 E30', fontsize=12)
        ax.set_ylabel('最大达标时间 t\' (年)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)

        fig.suptitle('任务三：NSGA-II 帕累托前沿二维投影', fontsize=18, fontweight='bold', y=1.05)
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        output_path = os.path.join(output_dir, "mission3_pareto_front_2d_projections.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制帕累托前沿二维投影图时发生错误: {e}")

if __name__ == "__main__":
    test_input_paths = {'mission3_nsga2_results_csv': "outputs/mission3/nsga2_results.csv"}
    test_output_dir = "outputs/figures/png"
    plot_pareto_front_2d_projections(test_input_paths, test_output_dir)
