import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_indicator_pairplot(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    csv_file_path = input_paths['mission2_normalized_indicators_csv']
    if not os.path.exists(csv_file_path):
        print(f"错误: 任务二标准化指标数据文件未找到: {csv_file_path}")
        return

    try:
        # --- 设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        df = pd.read_csv(csv_file_path, index_col=0)
        
        df_plot = df.apply(pd.to_numeric, errors='coerce')
        df_plot = df_plot.dropna()

        if df_plot.empty:
            print("处理后的标准化指标数据为空，无法绘制。")
            return

        # 添加风险等级信息，如果可用
        clustering_csv_path = input_paths.get('mission2_clustering_results_csv')
        if clustering_csv_path and os.path.exists(clustering_csv_path):
            df_clusters = pd.read_csv(clustering_csv_path, index_col=0)
            df_plot = df_plot.merge(df_clusters[['风险等级']], left_index=True, right_index=True, how='left')
            # 确保风险等级的顺序
            df_plot['风险等级'] = pd.Categorical(df_plot['风险等级'], categories=['高风险', '中风险', '低风险'], ordered=True)
            hue_var = '风险等级'
            palette = {'高风险': '#E41A1C', '中风险': '#FF7F00', '低风险': '#4DAF4A'}
        else:
            hue_var = None
            palette = None

        # 绘制pairplot
        # 限制绘制的指标数量，避免图表过于拥挤
        if len(df_plot.columns) > 6: # 例如，如果指标超过6个，只选择前6个
            cols_to_plot = df_plot.columns[:6].tolist()
            if hue_var and hue_var not in cols_to_plot:
                cols_to_plot.append(hue_var)
            df_plot_subset = df_plot[cols_to_plot]
        else:
            df_plot_subset = df_plot

        g = sns.pairplot(df_plot_subset, hue=hue_var, palette=palette, diag_kind='kde')
        g.fig.suptitle('标准化风险指标两两关系图', y=1.02, fontsize=16, fontweight='bold') # y调整标题位置
        
        plt.tight_layout(rect=[0, 0, 1, 0.98]) # 调整布局以适应suptitle

        output_path = os.path.join(output_dir, "mission2_indicator_pairplot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制指标两两关系图时发生错误: {e}")

if __name__ == "__main__":
    test_input_paths = {
        'mission2_normalized_indicators_csv': "outputs/mission2/normalized_indicators.csv",
        'mission2_clustering_results_csv': "outputs/mission2/clustering_results.csv"
    }
    test_output_dir = "outputs/figures/png"
    plot_indicator_pairplot(test_input_paths, test_output_dir)
