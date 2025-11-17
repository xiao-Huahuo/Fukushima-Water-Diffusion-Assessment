import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_cost_violin_by_scheme(input_paths, output_dir):
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

        df['成本C'] = pd.to_numeric(df['成本C'], errors='coerce')
        df = df.dropna(subset=['成本C'])

        # 确保每个方案只有一个结果 (如果NSGA-II结果包含重复方案)
        df_unique_schemes = df.drop_duplicates(subset=['方案'])
        
        # 将方案转换为分类变量
        df_unique_schemes['方案_str'] = df_unique_schemes['方案'].astype(str).apply(lambda x: f'方案 {x}')

        plt.figure(figsize=(10, 7))
        sns.violinplot(x='方案_str', y='成本C', data=df_unique_schemes, inner="quartile", palette='rocket')
        
        plt.title('不同处理方案下总成本分布小提琴图', fontsize=16, fontweight='bold')
        plt.xlabel('处理方案', fontsize=12)
        plt.ylabel('总成本 C', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.yscale('log') # 成本可能跨越多个数量级
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission3_cost_violin_by_scheme.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制总成本小提琴图时发生错误: {e}")

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
    plot_cost_violin_by_scheme(test_input_paths, test_output_dir)
