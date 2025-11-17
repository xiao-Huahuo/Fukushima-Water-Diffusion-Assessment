import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_indicator_contribution_to_score(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    normalized_csv_path = input_paths['mission2_normalized_indicators_csv']
    weights_csv_path = input_paths['mission2_weights_csv']
    
    if not os.path.exists(normalized_csv_path):
        print(f"错误: 任务二标准化指标数据文件未找到: {normalized_csv_path}")
        return
    if not os.path.exists(weights_csv_path):
        print(f"错误: 任务二权重计算数据文件未找到: {weights_csv_path}")
        return

    try:
        # 应用 seaborn 主题
        sns.set_theme(style="whitegrid", palette="viridis")

        # --- 在 sns.set_theme() 之后设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        df_norm = pd.read_csv(normalized_csv_path, index_col=0)
        df_weights = pd.read_csv(weights_csv_path, index_col='指标')
        
        df_norm = df_norm.apply(pd.to_numeric, errors='coerce').dropna()
        df_weights['组合权重Wj'] = pd.to_numeric(df_weights['组合权重Wj'], errors='coerce').dropna()

        if df_norm.empty or df_weights.empty:
            print("处理后的标准化指标或权重数据为空，无法绘制。")
            return

        # 计算每个指标对综合得分的贡献度: 贡献度 = 标准化值 * 组合权重
        contribution_df = df_norm.multiply(df_weights['组合权重Wj'], axis=1)

        # 绘制堆叠柱状图
        fig, ax = plt.subplots(figsize=(14, 8))
        contribution_df.plot(kind='bar', stacked=True, ax=ax, cmap='tab20', edgecolor='black')

        ax.set_title('各指标对国家综合风险得分的贡献度', fontsize=16, fontweight='bold')
        ax.set_xlabel('国家', fontsize=12)
        ax.set_ylabel('贡献度 (标准化值 * 组合权重)', fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.legend(title='指标', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, title_fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission2_indicator_contribution_to_score.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制指标贡献度柱状图时发生错误: {e}")

if __name__ == "__main__":
    test_input_paths = {
        'mission2_normalized_indicators_csv': "outputs/mission2/normalized_indicators.csv",
        'mission2_weights_csv': "outputs/mission2/weights_calculation.csv"
    }
    test_output_dir = "outputs/figures/png"
    plot_indicator_contribution_to_score(test_input_paths, test_output_dir)
