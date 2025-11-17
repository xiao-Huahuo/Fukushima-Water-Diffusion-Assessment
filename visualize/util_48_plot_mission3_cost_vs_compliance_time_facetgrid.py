import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_cost_vs_compliance_time_facetgrid(input_paths, output_dir):
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
        df_unique_schemes = df.drop_duplicates(subset=['方案'])
        
        # 将方案转换为分类变量，以便在FacetGrid中使用
        df_unique_schemes['方案_str'] = df_unique_schemes['方案'].astype(str).apply(lambda x: f'方案 {x}')

        # 绘制 FacetGrid
        g = sns.FacetGrid(df_unique_schemes, col='方案_str', col_wrap=2, height=5, aspect=1.2, sharex=False, sharey=False)
        g.map(sns.scatterplot, '达标时间t\'', '成本C', 
              hue='环境影响E30', palette='viridis_r', 
              s=200, edgecolor='black', linewidth=1, legend='full')
        
        g.set_axis_labels('最大达标时间 t\' (年)', '总成本 C')
        g.set_titles(col_template="{col_name}")
        g.set(xscale="log", yscale="log") # 使用对数尺度
        g.add_legend(title='环境影响 E30')
        g.tight_layout()
        plt.suptitle('处理方案：成本 vs. 达标时间 (按方案分面)', fontsize=16, fontweight='bold', y=1.02)

        output_path = os.path.join(output_dir, "mission3_cost_vs_compliance_time_facetgrid.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制成本 vs 达标时间分面图时发生错误: {e}")

if __name__ == "__main__":
    test_input_paths = {'mission3_nsga2_results_csv': "outputs/mission3/nsga2_results.csv"}
    test_output_dir = "outputs/figures/png"
    plot_cost_vs_compliance_time_facetgrid(test_input_paths, test_output_dir)
