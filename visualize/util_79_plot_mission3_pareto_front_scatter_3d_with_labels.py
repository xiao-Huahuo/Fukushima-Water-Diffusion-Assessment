import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def plot_pareto_front_scatter_3d_with_labels(input_paths, output_dir):
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

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        schemes = sorted(df_unique_schemes['方案'].unique())
        colors = plt.cm.get_cmap('viridis', len(schemes))

        for i, scheme in enumerate(schemes):
            scheme_df = df_unique_schemes[df_unique_schemes['方案'] == scheme]
            ax.scatter(scheme_df['环境影响E30'], scheme_df['成本C'], scheme_df['达标时间t\''],
                       color=colors(i), label=f'方案 {scheme}', s=200, alpha=0.9, edgecolors='w')
            
            # 添加标签
            ax.text(scheme_df['环境影响E30'].values[0], 
                    scheme_df['成本C'].values[0], 
                    scheme_df['达标时间t\''].values[0], 
                    f'方案{int(scheme)}', color='black', fontsize=10, ha='center', va='bottom')

        ax.set_title('任务三：NSGA-II 帕累托前沿 (环境影响 vs. 成本 vs. 达标时间)', fontsize=16, fontweight='bold')
        ax.set_xlabel('环境影响 E30', fontsize=12)
        ax.set_ylabel('总成本 C', fontsize=12)
        ax.set_zlabel('最大达标时间 t\' (年)', fontsize=12)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='z', labelsize=10)
        ax.legend(title='处理方案', loc='best', fontsize=10, title_fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        # 调整视角以获得更好的可视化效果
        ax.view_init(elev=20, azim=-60) # 可以根据需要调整

        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission3_pareto_front_3d_scatter_with_labels.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制帕累托前沿三维散点图 (带标签) 时发生错误: {e}")

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
    plot_pareto_front_scatter_3d_with_labels(test_input_paths, test_output_dir)
