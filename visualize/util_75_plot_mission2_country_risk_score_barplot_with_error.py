import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_country_risk_score_barplot_with_error(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    csv_file_path = input_paths['mission2_clustering_results_csv']
    if not os.path.exists(csv_file_path):
        print(f"错误: 任务二聚类结果数据文件未找到: {csv_file_path}")
        return

    try:
        # --- 设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        df = pd.read_csv(csv_file_path)
        
        df['综合得分'] = pd.to_numeric(df['综合得分'], errors='coerce')
        df = df.dropna(subset=['综合得分'])

        if df.empty:
            print("处理后的聚类结果数据为空，无法绘制。")
            return

        # 定义风险等级的颜色映射
        risk_colors = {
            '高风险': '#E41A1C', # 红色
            '中风险': '#FF7F00', # 橙色
            '低风险': '#4DAF4A'  # 绿色
        }
        # 确保风险等级的顺序
        df['风险等级'] = pd.Categorical(df['风险等级'], categories=['高风险', '中风险', '低风险'], ordered=True)
        df = df.sort_values('综合得分', ascending=False) # 按得分降序排列

        plt.figure(figsize=(10, 7))
        # 绘制柱状图，并使用风险等级着色
        sns.barplot(x='国家', y='综合得分', hue='风险等级', data=df, 
                    palette=risk_colors, dodge=False, edgecolor='black')
        
        # 假设我们有一些“误差”数据，这里用随机数模拟
        # 实际应用中，误差棒可能来自多次模拟的统计结果
        # error_bars = np.random.uniform(0.01, 0.05, len(df)) * df['综合得分'].values
        # plt.errorbar(x=np.arange(len(df)), y=df['综合得分'], yerr=error_bars, fmt='none', c='black', capsize=5)

        plt.title('各国综合风险得分与风险等级 (带误差棒)', fontsize=16, fontweight='bold')
        plt.xlabel('国家', fontsize=12)
        plt.ylabel('综合风险得分', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title="风险等级", loc='upper right', fontsize=10, title_fontsize=12)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission2_country_risk_score_barplot_with_error.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制各国综合风险得分柱状图 (带误差棒) 时发生错误: {e}")

if __name__ == "__main__":
    # 模拟数据用于测试
    test_df = pd.DataFrame({
        '国家': ['日本', '韩国', '美国', '加拿大', '中国', '澳大利亚'],
        '综合得分': [0.9, 0.8, 0.4, 0.3, 0.1, 0.05],
        '风险等级': ['高风险', '高风险', '中风险', '中风险', '低风险', '低风险']
    })
    test_output_dir_temp = "outputs/mission2/"
    os.makedirs(test_output_dir_temp, exist_ok=True)
    test_csv_path = os.path.join(test_output_dir_temp, "clustering_results.csv")
    test_df.to_csv(test_csv_path, index=False)

    test_input_paths = {'mission2_clustering_results_csv': test_csv_path}
    test_output_dir = "outputs/figures/png"
    plot_country_risk_score_barplot_with_error(test_input_paths, test_output_dir)
