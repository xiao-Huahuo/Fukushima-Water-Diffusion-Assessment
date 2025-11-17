import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_risk_dimension_comparison_radar(input_paths, output_dir):
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
        
        # 对维度得分进行标准化，以便在雷达图中比较
        df_normalized_dims = dimension_scores.copy()
        for col in df_normalized_dims.columns:
            min_val = df_normalized_dims[col].min()
            max_val = df_normalized_dims[col].max()
            if max_val - min_val > 1e-9:
                df_normalized_dims[col] = (df_normalized_dims[col] - min_val) / (max_val - min_val)
            else:
                df_normalized_dims[col] = 0.5 # 如果所有值相同，设为中间值

        # 雷达图的标签
        labels = df_normalized_dims.columns
        num_vars = len(labels)

        # 计算每个角度
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1] # 闭合雷达图

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # 绘制每个国家的雷达图
        for i, country in enumerate(df_normalized_dims.index):
            values = df_normalized_dims.loc[country].tolist()
            values += values[:1] # 闭合雷达图
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=country, alpha=0.8)
            ax.fill(angles, values, alpha=0.1) # 填充区域

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0) # 径向轴标签位置
        plt.xticks(angles[:-1], labels, color='grey', size=10)

        plt.yticks(np.arange(0, 1.1, 0.2), ['0', '0.2', '0.4', '0.6', '0.8', '1.0'], color='grey', size=8)
        plt.ylim(0, 1)

        ax.set_title('各国风险维度加权得分雷达图 (标准化)', fontsize=16, fontweight='bold', y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission2_risk_dimension_comparison_radar.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制风险维度对比雷达图时发生错误: {e}")

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
    plot_risk_dimension_comparison_radar(test_input_paths, test_output_dir)
