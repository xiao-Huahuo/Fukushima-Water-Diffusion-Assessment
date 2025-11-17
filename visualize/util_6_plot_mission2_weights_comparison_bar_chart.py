import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_weights_comparison_bar_chart(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    csv_file_path = input_paths['mission2_weights_csv']
    if not os.path.exists(csv_file_path):
        print(f"错误: 任务二权重计算数据文件未找到: {csv_file_path}")
        return

    try:
        # 应用 seaborn 主题
        sns.set_theme(style="whitegrid", palette="viridis")

        # --- 在 sns.set_theme() 之后设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        df = pd.read_csv(csv_file_path)
        
        df[['熵权法Wo', 'AHP法Ws', '组合权重Wj']] = df[['熵权法Wo', 'AHP法Ws', '组合权重Wj']].apply(pd.to_numeric, errors='coerce')
        df = df.dropna()

        if df.empty:
            print("处理后的权重数据为空，无法绘制。")
            return

        # 将DataFrame转换为长格式，以便seaborn处理
        df_long = df.melt(id_vars='指标', var_name='权重类型', value_name='权重值')

        # 绘制分组柱状图
        plt.figure(figsize=(14, 8))
        sns.barplot(x='指标', y='权重值', hue='权重类型', data=df_long, 
                    palette='Paired', edgecolor='black')

        plt.title('风险指标权重对比 (熵权法 vs. AHP法 vs. 组合权重)', fontsize=16, fontweight='bold')
        plt.xlabel('风险指标', fontsize=12)
        plt.ylabel('权重值', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title='权重类型', loc='upper right', fontsize=10, title_fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission2_weights_comparison_bar_chart.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制权重对比柱状图时发生错误: {e}")

if __name__ == "__main__":
    test_input_paths = {'mission2_weights_csv': "outputs/mission2/weights_calculation.csv"}
    test_output_dir = "outputs/figures/png"
    plot_weights_comparison_bar_chart(test_input_paths, test_output_dir)
