import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_weights_stacked_bar_chart(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    csv_file_path = input_paths['mission2_weights_csv']
    if not os.path.exists(csv_file_path):
        print(f"错误: 任务二权重计算数据文件未找到: {csv_file_path}")
        return

    try:
        # --- 设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        df = pd.read_csv(csv_file_path)
        
        df[['熵权法Wo', 'AHP法Ws', '组合权重Wj']] = df[['熵权法Wo', 'AHP法Ws', '组合权重Wj']].apply(pd.to_numeric, errors='coerce')
        df = df.dropna()

        if df.empty:
            print("处理后的权重数据为空，无法绘制。")
            return

        # 绘制堆叠柱状图
        # 这里我们只堆叠 Wo 和 Ws，组合权重 Wj 可以作为参考线或单独的图
        df_plot = df.set_index('指标')[['熵权法Wo', 'AHP法Ws']]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        df_plot.plot(kind='bar', stacked=True, ax=ax, cmap='coolwarm', edgecolor='black')

        # 添加组合权重 Wj 作为折线图或散点图
        ax.plot(np.arange(len(df)), df['组合权重Wj'], marker='o', color='green', linestyle='--', linewidth=2, label='组合权重Wj')

        ax.set_title('风险指标权重构成 (熵权法 vs. AHP法)', fontsize=16, fontweight='bold')
        ax.set_xlabel('风险指标', fontsize=12)
        ax.set_ylabel('权重值', fontsize=12)
        ax.tick_params(axis='x', rotation=45, ha='right', fontsize=10)
        ax.tick_params(axis='y', fontsize=10)
        ax.legend(title='权重类型', loc='upper right', fontsize=10, title_fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission2_weights_stacked_bar_chart.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制权重堆叠柱状图时发生错误: {e}")

if __name__ == "__main__":
    # 模拟数据用于测试
    test_df = pd.DataFrame({
        '指标': ['A1', 'A2', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'C4', 'C5'],
        '熵权法Wo': [0.115, 0.093, 0.110, 0.095, 0.084, 0.159, 0.103, 0.069, 0.086, 0.036],
        'AHP法Ws': [0.124, 0.085, 0.105, 0.092, 0.078, 0.163, 0.100, 0.067, 0.089, 0.037],
        '组合权重Wj': [0.118, 0.090, 0.108, 0.094, 0.082, 0.161, 0.102, 0.068, 0.087, 0.036]
    })
    test_output_dir_temp = "outputs/mission2/"
    os.makedirs(test_output_dir_temp, exist_ok=True)
    test_csv_path = os.path.join(test_output_dir_temp, "weights_calculation.csv")
    test_df.to_csv(test_csv_path, index=False)

    test_input_paths = {'mission2_weights_csv': test_csv_path}
    test_output_dir = "outputs/figures/png"
    plot_weights_stacked_bar_chart(test_input_paths, test_output_dir)
