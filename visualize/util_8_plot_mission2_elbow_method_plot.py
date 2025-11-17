import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def plot_elbow_method_plot(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    excel_file_path = input_paths['mission2_results_excel']
    if not os.path.exists(excel_file_path):
        print(f"错误: 任务二Excel报告文件未找到: {excel_file_path}")
        return

    try:
        # 应用 seaborn 主题
        sns.set_theme(style="whitegrid", palette="viridis")

        # --- 在 sns.set_theme() 之后设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        df_wcss = pd.read_excel(excel_file_path, sheet_name='5-肘部法则')
        
        if df_wcss.empty:
            print("肘部法则数据为空，无法绘制。")
            return

        plt.figure(figsize=(8, 6))
        sns.lineplot(x='K值', y='WCSS', data=df_wcss, marker='o', color='darkblue', linewidth=2)
        
        plt.title('K-means 肘部法则图 (WCSS vs. K值)', fontsize=16, fontweight='bold')
        plt.xlabel('聚类数量 (K)', fontsize=12)
        plt.ylabel('簇内平方和 (WCSS)', fontsize=12)
        plt.xticks(df_wcss['K值'], fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission2_elbow_method_plot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制肘部法则图时发生错误: {e}")

if __name__ == "__main__":
    test_input_paths = {'mission2_results_excel': "outputs/mission2/risk_assessment_results.xlsx"}
    test_output_dir = "outputs/figures/png"
    plot_elbow_method_plot(test_input_paths, test_output_dir)
