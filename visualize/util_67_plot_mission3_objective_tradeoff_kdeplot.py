import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_objective_tradeoff_kdeplot(input_paths, output_dir):
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
        
        plt.figure(figsize=(10, 8))
        
        # 绘制两个目标之间的KDE图
        sns.kdeplot(x='环境影响E30', y='成本C', data=df_unique_schemes, 
                    fill=True, cmap='viridis', levels=5, alpha=0.7)
        
        # 在KDE图上叠加散点，并用达标时间着色
        scatter = sns.scatterplot(x='环境影响E30', y='成本C', hue='达标时间t\'', size='达标时间t\'', 
                                  sizes=(100, 1000), # 散点大小范围
                                  data=df_unique_schemes, palette='coolwarm', 
                                  legend='full', alpha=0.9, edgecolor='black', zorder=2)
        
        # 添加方案标签
        for i, row in df_unique_schemes.iterrows():
            plt.text(row['环境影响E30'], row['成本C'], f'方案{int(row["方案"])}', 
                     ha='center', va='bottom', fontsize=9, color='black')

        plt.title('处理方案目标权衡KDE图 (环境影响 vs. 成本, 达标时间着色)', fontsize=16, fontweight='bold')
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

        output_path = os.path.join(output_dir, "mission3_objective_tradeoff_kdeplot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制目标权衡KDE图时发生错误: {e}")

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
    plot_objective_tradeoff_kdeplot(test_input_paths, test_output_dir)
