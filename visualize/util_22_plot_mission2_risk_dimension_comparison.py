import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_risk_dimension_comparison(input_paths, output_dir):
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

        # 定义一级指标及其包含的二级指标
        dimensions = {
            '海洋生态': ['A1', 'A2'],
            '渔业经济': ['B1', 'B2', 'B3'],
            '食品安全': ['C1', 'C2', 'C3', 'C4', 'C5']
        }

        # 计算每个国家在每个一级指标上的加权得分
        dimension_scores = pd.DataFrame(index=df_norm.index, columns=dimensions.keys())
        for country in df_norm.index:
            for dim, indicators in dimensions.items():
                score = 0
                for ind in indicators:
                    if ind in df_norm.columns and ind in df_weights.index:
                        score += df_norm.loc[country, ind] * df_weights.loc[ind, '组合权重Wj']
                dimension_scores.loc[country, dim] = score
        
        # 将DataFrame转换为长格式，以便seaborn处理
        df_long = dimension_scores.reset_index().melt(id_vars='index', var_name='风险维度', value_name='维度加权得分')
        df_long = df_long.rename(columns={'index': '国家'})

        # 绘制分组柱状图
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(x='国家', y='维度加权得分', hue='风险维度', data=df_long, 
                    palette='Set2', edgecolor='black')

        ax.set_title('各国在不同风险维度上的加权得分', fontsize=16, fontweight='bold')
        ax.set_xlabel('国家', fontsize=12)
        ax.set_ylabel('维度加权得分', fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.legend(title='风险维度', loc='upper left', fontsize=10, title_fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission2_risk_dimension_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制风险维度对比图时发生错误: {e}")

if __name__ == "__main__":
    test_input_paths = {
        'mission2_normalized_indicators_csv': "outputs/mission2/normalized_indicators.csv",
        'mission2_weights_csv': "outputs/mission2/weights_calculation.csv"
    }
    test_output_dir = "outputs/figures/png"
    plot_risk_dimension_comparison(test_input_paths, test_output_dir)
