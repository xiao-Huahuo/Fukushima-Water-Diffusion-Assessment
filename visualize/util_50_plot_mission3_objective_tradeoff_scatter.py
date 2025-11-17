import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_objective_tradeoff_scatter(input_paths, output_dir):
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
        
        plt.figure(figsize=(12, 8))
        
        # 使用 seaborn 绘制散点图，将达标时间映射到颜色
        scatter = sns.scatterplot(x='环境影响E30', y='成本C', hue='达标时间t\'', size='达标时间t\'', 
                                  sizes=(100, 1000), # 散点大小范围
                                  data=df_unique_schemes, palette='coolwarm', # 达标时间越短越好，所以用coolwarm
                                  legend='full', alpha=0.8, edgecolor='black')
        
        # 添加方案标签
        for i, row in df_unique_schemes.iterrows():
            plt.text(row['环境影响E30'], row['成本C'], f'方案{int(row["方案"])}', 
                     ha='center', va='bottom', fontsize=9, color='black')

        plt.title('处理方案目标权衡散点图 (环境影响 vs. 成本, 达标时间着色)', fontsize=16, fontweight='bold')
        plt.xlabel('环境影响 E30', fontsize=12)
        plt.ylabel('总成本 C', fontsize=12)
        plt.xscale('log') # 目标值可能跨越多个数量级
        plt.yscale('log') # 目标值可能跨越多个数量级
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, which="both", ls="--", c='0.7', alpha=0.7)
        
        # 调整图例位置
        plt.legend(title='最大达标时间 t\' (年)', loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, title_fontsize=12)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission3_objective_tradeoff_scatter.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制目标权衡散点图时发生错误: {e}")

if __name__ == "__main__":
    test_input_paths = {'mission3_nsga2_results_csv': "outputs/mission3/nsga2_results.csv"}
    test_output_dir = "outputs/figures/png"
    plot_objective_tradeoff_scatter(test_input_paths, test_output_dir)
