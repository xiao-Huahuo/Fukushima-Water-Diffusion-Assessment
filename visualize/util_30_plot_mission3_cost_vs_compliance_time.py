import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_cost_vs_compliance_time(input_paths, output_dir):
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
        
        plt.figure(figsize=(12, 8))
        
        # 使用 seaborn 绘制散点图，将达标时间映射到颜色
        scatter = sns.scatterplot(x='达标时间t\'', y='成本C', hue='环境影响E30', size='环境影响E30', 
                                  sizes=(100, 1000), # 散点大小范围
                                  data=df, palette='viridis_r', # _r 反转颜色，使小环境影响为绿色
                                  legend='full', alpha=0.8, edgecolor='black')
        
        # 添加方案标签
        for i, row in df.iterrows():
            plt.text(row['达标时间t\''], row['成本C'], f'方案{int(row["方案"])}', 
                     ha='center', va='bottom', fontsize=9, color='black')

        plt.title('处理方案：成本 vs. 达标时间 (环境影响着色)', fontsize=16, fontweight='bold')
        plt.xlabel('最大达标时间 t\' (年)', fontsize=12)
        plt.ylabel('总成本 C', fontsize=12)
        plt.xscale('log') # 达标时间可能跨越多个数量级
        plt.yscale('log') # 成本可能跨越多个数量级
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, which="both", ls="--", c='0.7', alpha=0.7)
        
        # 调整图例位置
        plt.legend(title='环境影响 E30', loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, title_fontsize=12)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission3_cost_vs_compliance_time.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制成本 vs 达标时间散点图时发生错误: {e}")

if __name__ == "__main__":
    test_input_paths = {'mission3_nsga2_results_csv': "outputs/mission3/nsga2_results.csv"}
    test_output_dir = "outputs/figures/png"
    plot_cost_vs_compliance_time(test_input_paths, test_output_dir)
