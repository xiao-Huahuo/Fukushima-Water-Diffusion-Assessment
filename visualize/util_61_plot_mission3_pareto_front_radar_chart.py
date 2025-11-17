import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_pareto_front_radar_chart(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    csv_file_path = input_paths['mission3_nsga2_results_csv']
    if not os.path.exists(csv_file_path):
        print(f"错误: 任务三NSGA-II结果文件未找到: {csv_file_path}")
        return

    try:
        # --- 设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        df = pd.read_csv(csv_file_path)
        
        if df.empty:
            print("NSGA-II结果数据为空，无法绘制。")
            return

        df['环境影响E30'] = pd.to_numeric(df['环境影响E30'], errors='coerce')
        df['成本C'] = pd.to_numeric(df['成本C'], errors='coerce')
        df['达标时间t\''] = pd.to_numeric(df['达标时间t\''], errors='coerce')
        df = df.dropna()

        # 确保每个方案只有一个结果 (如果NSGA-II结果包含重复方案)
        df_unique_schemes = df.drop_duplicates(subset=['方案']).set_index('方案')

        # 对目标函数进行标准化，以便在雷达图中比较
        # 目标都是越小越好，所以直接Min-Max标准化到 [0, 1]
        df_normalized = df_unique_schemes[['环境影响E30', '成本C', '达标时间t\'']].copy()
        for col in df_normalized.columns:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            if max_val - min_val > 1e-9:
                df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
            else:
                df_normalized[col] = 0.5 # 如果所有值相同，设为中间值
        
        # 雷达图的标签
        labels = ['环境影响', '总成本', '达标时间']
        num_vars = len(labels)

        # 计算每个角度
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1] # 闭合雷达图

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # 绘制每个方案的雷达图
        for i, scheme in enumerate(df_normalized.index):
            values = df_normalized.loc[scheme, ['环境影响E30', '成本C', '达标时间t\'']].tolist()
            values += values[:1] # 闭合雷达图
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=f'方案 {int(scheme)}', alpha=0.8)
            ax.fill(angles, values, alpha=0.1) # 填充区域

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0) # 径向轴标签位置
        plt.xticks(angles[:-1], labels, color='grey', size=10)

        plt.yticks(np.arange(0, 1.1, 0.2), ['0', '0.2', '0.4', '0.6', '0.8', '1.0'], color='grey', size=8)
        plt.ylim(0, 1)

        ax.set_title('处理方案帕累托前沿雷达图 (标准化目标值)', fontsize=16, fontweight='bold', y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission3_pareto_front_radar_chart.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制帕累托前沿雷达图时发生错误: {e}")

if __name__ == "__main__":
    # 模拟数据用于测试
    test_df = pd.DataFrame({
        '方案': [1, 2, 3],
        '环境影响E30': [1.0e12, 5.0e11, 1.0e11],
        '成本C': [5.0e9, 8.0e9, 12.0e9],
        '达标时间t\'': [100.0, 50.0, 10.0]
    })
    test_output_dir_temp = "outputs/mission3/"
    os.makedirs(test_output_dir_temp, exist_ok=True)
    test_csv_path = os.path.join(test_output_dir_temp, "nsga2_results.csv")
    test_df.to_csv(test_csv_path, index=False)

    test_input_paths = {'mission3_nsga2_results_csv': test_csv_path}
    test_output_dir = "outputs/figures/png"
    plot_pareto_front_radar_chart(test_input_paths, test_output_dir)
