import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_risk_score_distribution_kde_facetgrid(input_paths, output_dir):
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

        # 确保风险等级的顺序
        df['风险等级'] = pd.Categorical(df['风险等级'], categories=['高风险', '中风险', '低风险'], ordered=True)
        
        # 绘制 FacetGrid
        g = sns.FacetGrid(df, col='风险等级', col_wrap=3, height=4, aspect=1.2, sharey=False)
        g.map(sns.kdeplot, '综合得分', fill=True, alpha=0.6, color='skyblue')
        
        g.set_axis_labels("综合风险得分", "概率密度")
        g.set_titles("风险等级: {col_name}")
        g.tight_layout()
        plt.suptitle('各国综合风险得分概率密度分布 (按风险等级分面)', fontsize=16, fontweight='bold', y=1.02)

        output_path = os.path.join(output_dir, "mission2_risk_score_distribution_kde_facetgrid.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制综合风险得分 KDE 分面图时发生错误: {e}")

if __name__ == "__main__":
    # 模拟数据用于测试
    test_df = pd.DataFrame({
        '国家': ['日本', '韩国', '美国', '加拿大', '中国', '澳大利亚'],
        '综合得分': [0.9, 0.8, 0.4, 0.3, 0.1, 0.05],
        '风险等级': ['高风险', '高风险', '中风险', '中风险', '低风险', '低风险']
    })
    test_output_dir_temp = "outputs/mission2/"
    os.makedirs(test_output_dir_temp, exist_ok=True)
    test_csv_path = os.path.join(test_output_dir_temp, "clustering_results.csv")
    test_df.to_csv(test_csv_path, index=False)

    test_input_paths = {'mission2_clustering_results_csv': test_csv_path}
    test_output_dir = "outputs/figures/png"
    plot_risk_score_distribution_kde_facetgrid(test_input_paths, test_output_dir)
