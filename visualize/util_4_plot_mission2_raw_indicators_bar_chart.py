import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_raw_indicators_bar_chart(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    csv_file_path = input_paths['mission2_raw_indicators_csv']
    if not os.path.exists(csv_file_path):
        print(f"错误: 任务二原始指标数据文件未找到: {csv_file_path}")
        return

    try:
        # 应用 seaborn 主题
        sns.set_theme(style="whitegrid", palette="viridis")

        # --- 在 sns.set_theme() 之后设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        df = pd.read_csv(csv_file_path, index_col=0)
        
        df_plot = df.apply(pd.to_numeric, errors='coerce')
        df_plot = df_plot.dropna()

        if df_plot.empty:
            print("处理后的原始指标数据为空，无法绘制。")
            return

        # 将DataFrame转换为长格式，以便seaborn处理
        df_long = df_plot.reset_index().melt(id_vars='index', var_name='指标', value_name='指标值')
        df_long = df_long.rename(columns={'index': '国家'})

        # 绘制分组柱状图
        plt.figure(figsize=(18, 10))
        sns.barplot(x='国家', y='指标值', hue='指标', data=df_long, 
                    palette='tab20', edgecolor='black')

        plt.title('各国原始风险指标对比', fontsize=18, fontweight='bold')
        plt.xlabel('国家', fontsize=14)
        plt.ylabel('指标值', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='指标', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, title_fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission2_raw_indicators_bar_chart.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制原始指标柱状图时发生错误: {e}")

if __name__ == "__main__":
    test_input_paths = {'mission2_raw_indicators_csv': "outputs/mission2/raw_indicators.csv"}
    test_output_dir = "outputs/figures/png"
    plot_raw_indicators_bar_chart(test_input_paths, test_output_dir)
