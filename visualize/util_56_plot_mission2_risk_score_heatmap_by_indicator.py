import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_risk_score_heatmap_by_indicator(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    normalized_csv_path = input_paths['mission2_normalized_indicators_csv']
    clustering_csv_path = input_paths['mission2_clustering_results_csv']
    
    if not os.path.exists(normalized_csv_path):
        print(f"错误: 任务二标准化指标数据文件未找到: {normalized_csv_path}")
        return
    if not os.path.exists(clustering_csv_path):
        print(f"错误: 任务二聚类结果数据文件未找到: {clustering_csv_path}")
        return

    try:
        # --- 设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        df_norm = pd.read_csv(normalized_csv_path, index_col=0)
        df_clusters = pd.read_csv(clustering_csv_path, index_col=0)
        
        df_norm = df_norm.apply(pd.to_numeric, errors='coerce').dropna()
        df_clusters = df_clusters.dropna(subset=['风险等级'])

        if df_norm.empty or df_clusters.empty:
            print("处理后的标准化指标或聚类结果数据为空，无法绘制。")
            return

        # 合并风险等级信息
        df_merged = df_norm.merge(df_clusters[['风险等级']], left_index=True, right_index=True, how='left')
        
        # 按风险等级排序，使热图更具可读性
        df_merged['风险等级'] = pd.Categorical(df_merged['风险等级'], categories=['高风险', '中风险', '低风险'], ordered=True)
        df_merged = df_merged.sort_values('风险等级')

        # 准备热图数据
        heatmap_data = df_merged.drop(columns=['风险等级'])
        
        # 绘制热图
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=.5, linecolor='black',
                    cbar_kws={'label': '标准化指标值'})
        
        plt.title('各国标准化风险指标热图 (按风险等级排序)', fontsize=16, fontweight='bold')
        plt.xlabel('风险指标', fontsize=12)
        plt.ylabel('国家', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission2_risk_score_heatmap_by_indicator.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制按风险等级排序的标准化指标热图时发生错误: {e}")

if __name__ == "__main__":
    # 模拟数据用于测试
    test_norm_df = pd.DataFrame({
        'A1': [1.0, 0.8, 0.2, 0.1, 0.05, 0.0],
        'A2': [1.0, 0.9, 0.7, 0.5, 0.3, 0.1],
        'B1': [0.9, 0.7, 0.4, 0.2, 0.1, 0.0],
        'C1': [1.0, 0.6, 0.3, 0.1, 0.05, 0.0],
        'C5': [0.8, 0.7, 0.5, 0.3, 0.2, 0.1],
    }, index=['日本', '韩国', '美国', '加拿大', '中国', '澳大利亚'])
    test_cluster_df = pd.DataFrame({
        '国家': ['日本', '韩国', '美国', '加拿大', '中国', '澳大利亚'],
        '综合得分': [0.9, 0.8, 0.4, 0.3, 0.1, 0.05],
        '风险等级': ['高风险', '高风险', '中风险', '中风险', '低风险', '低风险']
    }).set_index('国家')

    test_output_dir_temp = "outputs/mission2/"
    os.makedirs(test_output_dir_temp, exist_ok=True)
    test_norm_csv_path = os.path.join(test_output_dir_temp, "normalized_indicators.csv")
    test_cluster_csv_path = os.path.join(test_output_dir_temp, "clustering_results.csv")
    test_norm_df.to_csv(test_norm_csv_path)
    test_cluster_df.to_csv(test_cluster_csv_path)

    test_input_paths = {
        'mission2_normalized_indicators_csv': test_norm_csv_path,
        'mission2_clustering_results_csv': test_cluster_csv_path
    }
    test_output_dir = "outputs/figures/png"
    plot_risk_score_heatmap_by_indicator(test_input_paths, test_output_dir)
