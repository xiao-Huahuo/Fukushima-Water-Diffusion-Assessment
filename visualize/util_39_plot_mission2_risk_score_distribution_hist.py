import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_risk_score_distribution_hist(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    csv_file_path = input_paths['mission2_clustering_results_csv']
    if not os.path.exists(csv_file_path):
        print(f"错误: 任务二聚类结果数据文件未找到: {csv_file_path}")
        return

    try:
        # --- 设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        df = pd.read_csv(csv_file_path)
        
        df['综合得分'] = pd.to_numeric(df['综合得分'], errors='coerce')
        df = df.dropna(subset=['综合得分'])

        if df.empty:
            print("处理后的聚类结果数据为空，无法绘制。")
            return

        plt.figure(figsize=(10, 7))
        sns.histplot(df['综合得分'], kde=True, bins=5, color='skyblue', edgecolor='black')
        
        plt.title('各国综合风险得分分布直方图与KDE', fontsize=16, fontweight='bold')
        plt.xlabel('综合风险得分', fontsize=12)
        plt.ylabel('国家数量', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission2_risk_score_distribution_hist.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制综合风险得分分布直方图时发生错误: {e}")

if __name__ == "__main__":
    test_input_paths = {'mission2_clustering_results_csv': "outputs/mission2/clustering_results.csv"}
    test_output_dir = "outputs/figures/png"
    plot_risk_score_distribution_hist(test_input_paths, test_output_dir)
