import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_cost_vs_environmental_impact_jointplot(input_paths, output_dir):
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
        
        # 绘制 jointplot
        g = sns.jointplot(x='环境影响E30', y='成本C', data=df_unique_schemes, 
                          kind='scatter', height=8, ratio=5, marginal_ticks=True,
                          hue='方案', palette='viridis', s=100, edgecolor='black', linewidth=1)
        
        # 添加方案标签
        for i, row in df_unique_schemes.iterrows():
            g.ax_joint.text(row['环境影响E30'], row['成本C'], f'方案{int(row["方案"])}', 
                            ha='center', va='bottom', fontsize=9, color='black')

        g.set_axis_labels('环境影响 E30', '总成本 C', fontsize=12)
        g.fig.suptitle('处理方案：成本 vs. 环境影响 (联合分布)', fontsize=16, fontweight='bold', y=1.02)
        
        # 调整刻度为科学计数法
        g.ax_joint.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        g.ax_joint.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        plt.tight_layout(rect=[0, 0, 1, 0.98]) # 调整布局以适应suptitle

        output_path = os.path.join(output_dir, "mission3_cost_vs_environmental_impact_jointplot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制成本 vs 环境影响联合分布图时发生错误: {e}")

if __name__ == "__main__":
    test_input_paths = {'mission3_nsga2_results_csv': "outputs/mission3/nsga2_results.csv"}
    test_output_dir = "outputs/figures/png"
    plot_cost_vs_environmental_impact_jointplot(test_input_paths, test_output_dir)
