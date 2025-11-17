import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_environmental_impact_timeseries_stacked(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    csv_file_path = input_paths['mission3_e_t_timeseries_csv']
    if not os.path.exists(csv_file_path):
        print(f"错误: 任务三环境影响时间序列数据文件未找到: {csv_file_path}")
        print("请确保已运行 utils/Et.py 来生成该文件。")
        return

    try:
        # --- 设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        df_impact = pd.read_csv(csv_file_path)
        
        if df_impact.empty:
            print("环境影响时间序列数据为空，无法绘制。")
            return

        # 假设 df_impact 包含 'Year' 列和 'Scheme_1', 'Scheme_2', 'Scheme_3' 等列
        # 并且这些 Scheme_X 列代表了不同方案下的环境影响
        schemes = [col for col in df_impact.columns if col.startswith('Scheme_')]
        
        # 将数据从宽格式转换为长格式，以便seaborn处理堆叠面积图
        df_long = df_impact.melt(id_vars=['Year'], value_vars=schemes, 
                                 var_name='方案', value_name='环境影响')
        df_long['方案'] = df_long['方案'].str.replace('Scheme_', '方案 ')

        plt.figure(figsize=(12, 7))
        sns.lineplot(x='Year', y='环境影响', hue='方案', data=df_long, 
                     marker='o', linewidth=2, palette='viridis')
        
        plt.title('不同处理方案下环境影响随时间变化', fontsize=16, fontweight='bold')
        plt.xlabel('年份', fontsize=12)
        plt.ylabel('环境影响 ($E_{Total}(t)$)', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, which="both", ls="--", c='0.7', alpha=0.7)
        plt.legend(title='处理方案', loc='best', fontsize=10, title_fontsize=12)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission3_environmental_impact_timeseries_line.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制环境影响时间序列图时发生错误: {e}")

if __name__ == "__main__":
    # 模拟一个 E_t_timeseries.csv 文件用于测试
    test_df = pd.DataFrame({
        'Year': np.arange(1, 31),
        'Scheme_1': np.linspace(1e12, 5e11, 30) + np.random.rand(30) * 1e11,
        'Scheme_2': np.linspace(8e11, 3e11, 30) + np.random.rand(30) * 5e10,
        'Scheme_3': np.linspace(5e11, 1e11, 30) + np.random.rand(30) * 2e10,
    })
    test_output_dir_temp = "outputs/mission3/"
    os.makedirs(test_output_dir_temp, exist_ok=True)
    test_csv_path = os.path.join(test_output_dir_temp, "E_t_timeseries.csv")
    test_df.to_csv(test_csv_path, index=False)

    test_input_paths = {'mission3_e_t_timeseries_csv': test_csv_path}
    test_output_dir = "outputs/figures/png"
    plot_environmental_impact_timeseries_stacked(test_input_paths, test_output_dir)
