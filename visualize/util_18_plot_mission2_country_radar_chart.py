import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_country_radar_chart(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    csv_file_path = input_paths['mission2_normalized_indicators_csv']
    if not os.path.exists(csv_file_path):
        print(f"错误: 任务二标准化指标数据文件未找到: {csv_file_path}")
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
            print("处理后的标准化指标数据为空，无法绘制。")
            return

        # 选择要绘制的指标 (所有指标)
        labels = df_plot.columns
        num_vars = len(labels)

        # 计算每个角度
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1] # 闭合雷达图

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # 绘制每个国家的雷达图
        for i, country in enumerate(df_plot.index):
            values = df_plot.loc[country].tolist()
            values += values[:1] # 闭合雷达图
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=country, alpha=0.8)
            ax.fill(angles, values, alpha=0.1) # 填充区域

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0) # 径向轴标签位置
        plt.xticks(angles[:-1], labels, color='grey', size=10)

        plt.yticks(np.arange(0, 1.1, 0.2), ['0', '0.2', '0.4', '0.6', '0.8', '1.0'], color='grey', size=8)
        plt.ylim(0, 1)

        ax.set_title('各国标准化风险指标雷达图', fontsize=16, fontweight='bold', y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission2_country_radar_chart.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制各国风险指标雷达图时发生错误: {e}")

if __name__ == "__main__":
    test_input_paths = {'mission2_normalized_indicators_csv': "outputs/mission2/normalized_indicators.csv"}
    test_output_dir = "outputs/figures/png"
    plot_country_radar_chart(test_input_paths, test_output_dir)
