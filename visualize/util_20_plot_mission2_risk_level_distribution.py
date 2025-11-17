import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_risk_level_distribution(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    csv_file_path = input_paths['mission2_clustering_results_csv']
    if not os.path.exists(csv_file_path):
        print(f"错误: 任务二聚类结果数据文件未找到: {csv_file_path}")
        return

    try:
        # 应用 seaborn 主题
        sns.set_theme(style="whitegrid", palette="viridis")

        # --- 在 sns.set_theme() 之后设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        df = pd.read_csv(csv_file_path)
        
        if df.empty:
            print("处理后的聚类结果数据为空，无法绘制。")
            return

        # 定义风险等级的颜色映射
        risk_colors = {
            '高风险': '#E41A1C', # 红色
            '中风险': '#FF7F00', # 橙色
            '低风险': '#4DAF4A'  # 绿色
        }
        # 确保风险等级的顺序
        df['风险等级'] = pd.Categorical(df['风险等级'], categories=['高风险', '中风险', '低风险'], ordered=True)
        
        # 计算每个风险等级的国家数量
        risk_counts = df['风险等级'].value_counts().sort_index()

        plt.figure(figsize=(8, 8))
        wedges, texts, autotexts = plt.pie(risk_counts, 
                                           labels=risk_counts.index, 
                                           autopct='%1.1f%%', 
                                           colors=[risk_colors[r] for r in risk_counts.index],
                                           startangle=90,
                                           pctdistance=0.85,
                                           wedgeprops=dict(width=0.3, edgecolor='w')) # 环形图

        plt.setp(autotexts, size=10, weight="bold", color="white")
        plt.setp(texts, size=12)

        plt.title('各国风险等级分布', fontsize=16, fontweight='bold')

        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission2_risk_level_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制风险等级分布饼图时发生错误: {e}")

if __name__ == "__main__":
    test_input_paths = {'mission2_clustering_results_csv': "outputs/mission2/clustering_results.csv"}
    test_output_dir = "outputs/figures/png"
    plot_risk_level_distribution(test_input_paths, test_output_dir)
