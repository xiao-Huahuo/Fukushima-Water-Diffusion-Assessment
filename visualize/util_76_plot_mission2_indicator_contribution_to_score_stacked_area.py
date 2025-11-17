import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_indicator_contribution_to_score_stacked_area(input_paths, output_dir):
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
        # --- 设置字体 ---
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

        # 绘制堆叠面积图
        # 将国家作为X轴，指标作为堆叠的Y轴
        contribution_df_transposed = contribution_df.T # 转置，指标为行，国家为列
        
        plt.figure(figsize=(14, 8))
        # 使用 seaborn 的 lineplot 绘制堆叠面积图
        # 需要将数据转换为长格式
        df_long = contribution_df.reset_index().melt(id_vars='index', var_name='指标', value_name='贡献度')
        df_long = df_long.rename(columns={'index': '国家'})

        sns.lineplot(x='国家', y='贡献度', hue='指标', data=df_long, 
                     marker='o', linewidth=1, palette='tab20',
                     estimator=None) # 不进行估计，直接绘制每个点的线

        # 绘制堆叠面积图
        # plt.stackplot(contribution_df.index, contribution_df.values.T, 
        #               labels=contribution_df.columns, alpha=0.8, cmap='tab20')
        
        plt.title('各指标对国家综合风险得分的贡献度 (堆叠面积图)', fontsize=16, fontweight='bold')
        plt.xlabel('国家', fontsize=12)
        plt.ylabel('贡献度 (标准化值 * 组合权重)', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title='指标', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, title_fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission2_indicator_contribution_to_score_stacked_area.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制指标贡献度堆叠面积图时发生错误: {e}")

if __name__ == "__main__":
    # 模拟数据用于测试
    test_norm_df = pd.DataFrame({
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
    test_weights_df = pd.DataFrame({
        '指标': ['A1', 'A2', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'C4', 'C5'],
        '组合权重Wj': [0.118, 0.090, 0.108, 0.094, 0.082, 0.161, 0.102, 0.068, 0.087, 0.036]
    }).set_index('指标')

    test_output_dir_temp = "outputs/mission2/"
    os.makedirs(test_output_dir_temp, exist_ok=True)
    test_norm_csv_path = os.path.join(test_output_dir_temp, "normalized_indicators.csv")
    test_weights_csv_path = os.path.join(test_output_dir_temp, "weights_calculation.csv")
    test_norm_df.to_csv(test_norm_csv_path)
    test_weights_df.to_csv(test_weights_csv_path)

    test_input_paths = {
        'mission2_normalized_indicators_csv': test_norm_csv_path,
        'mission2_weights_csv': test_weights_csv_path
    }
    test_output_dir = "outputs/figures/png"
    plot_indicator_contribution_to_score_stacked_area(test_input_paths, test_output_dir)
