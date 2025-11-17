import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_indicator_correlation_clustermap(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    csv_file_path = input_paths['mission2_normalized_indicators_csv']
    if not os.path.exists(csv_file_path):
        print(f"错误: 任务二标准化指标数据文件未找到: {csv_file_path}")
        return

    try:
        # --- 设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        df = pd.read_csv(csv_file_path, index_col=0)
        
        df_plot = df.apply(pd.to_numeric, errors='coerce')
        df_plot = df_plot.dropna()

        if df_plot.empty:
            print("处理后的标准化指标数据为空，无法计算相关性。")
            return

        # 计算指标之间的相关性矩阵
        correlation_matrix = df_plot.corr()

        plt.figure(figsize=(10, 10))
        sns.clustermap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black',
                       cbar_kws={'label': '相关系数'})
        
        plt.title('标准化风险指标相关性聚类热图', fontsize=16, fontweight='bold', y=1.02)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.98]) # 调整布局以适应suptitle

        output_path = os.path.join(output_dir, "mission2_indicator_correlation_clustermap.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制指标相关性聚类热图时发生错误: {e}")

if __name__ == "__main__":
    # 模拟数据用于测试
    test_df = pd.DataFrame({
        'A1': [1.0, 0.8, 0.2, 0.1, 0.05, 0.0],
        'A2': [1.0, 0.9, 0.7, 0.5, 0.3, 0.1],
        'B1': [0.9, 0.7, 0.4, 0.2, 0.1, 0.0],
        'B2': [0.8, 0.6, 0.3, 0.15, 0.08, 0.0],
        'B3': [0.7, 0.5, 0.25, 0.12, 0.06, 0.0],
        'C1': [1.0, 0.6, 0.3, 0.1, 0.05, 0.0],
        'C2': [0.9, 0.5, 0.25, 0.08, 0.04, 0.0],
        'C3': [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        'C4': [0.8, 0.7, 0.5, 0.3, 0.2, 0.1],
        'C5': [0.7, 0.6, 0.4, 0.2, 0.1, 0.05],
    }, index=['日本', '韩国', '美国', '加拿大', '中国', '澳大利亚'])
    test_output_dir_temp = "outputs/mission2/"
    os.makedirs(test_output_dir_temp, exist_ok=True)
    test_csv_path = os.path.join(test_output_dir_temp, "normalized_indicators.csv")
    test_df.to_csv(test_csv_path)

    test_input_paths = {'mission2_normalized_indicators_csv': test_csv_path}
    test_output_dir = "outputs/figures/png"
    plot_indicator_correlation_clustermap(test_input_paths, test_output_dir)
