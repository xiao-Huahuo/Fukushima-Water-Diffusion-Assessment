import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_risk_score_classification_bar_chart(input_paths, output_dir):
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
        
        df['综合得分'] = pd.to_numeric(df['综合得分'], errors='coerce')
        df = df.dropna(subset=['综合得分'])

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
        df = df.sort_values('综合得分', ascending=False) # 按得分降序排列

        plt.figure(figsize=(10, 7))
        sns.barplot(x='国家', y='综合得分', hue='风险等级', data=df, 
                       palette=risk_colors, dodge=False, edgecolor='black')

        plt.title('各国综合风险得分与风险等级', fontsize=16, fontweight='bold')
        plt.xlabel('国家', fontsize=14)
        plt.ylabel('综合风险得分', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 添加图例
        handles = [plt.Rectangle((0,0),1,1, color=risk_colors[label]) for label in risk_colors]
        plt.legend(handles, risk_colors.keys(), title="风险等级", loc='upper left', fontsize=10, title_fontsize=12)

        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission2_risk_score_classification_bar_chart.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制综合风险得分与分类柱状图时发生错误: {e}")

if __name__ == "__main__":
    test_input_paths = {'mission2_clustering_results_csv': "outputs/mission2/clustering_results.csv"}
    test_output_dir = "outputs/figures/png"
    plot_risk_score_classification_bar_chart(test_input_paths, test_output_dir)
