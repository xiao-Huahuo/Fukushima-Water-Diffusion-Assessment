import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_weights_pie_chart(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    csv_file_path = input_paths['mission2_weights_csv']
    if not os.path.exists(csv_file_path):
        print(f"错误: 任务二权重计算数据文件未找到: {csv_file_path}")
        return

    try:
        # --- 设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        df = pd.read_csv(csv_file_path)
        
        df['组合权重Wj'] = pd.to_numeric(df['组合权重Wj'], errors='coerce')
        df = df.dropna(subset=['组合权重Wj'])

        if df.empty:
            print("处理后的权重数据为空，无法绘制。")
            return

        # 绘制饼图
        plt.figure(figsize=(10, 10))
        wedges, texts, autotexts = plt.pie(df['组合权重Wj'], 
                                           labels=df['指标'], 
                                           autopct='%1.1f%%', 
                                           startangle=90, 
                                           pctdistance=0.85,
                                           wedgeprops=dict(width=0.4, edgecolor='w'), # 环形图
                                           colors=sns.color_palette("Paired", len(df)))

        plt.setp(autotexts, size=10, weight="bold", color="white")
        plt.setp(texts, size=12)

        plt.title('风险指标组合权重分布', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission2_weights_pie_chart.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制权重饼图时发生错误: {e}")

if __name__ == "__main__":
    test_input_paths = {'mission2_weights_csv': "outputs/mission2/weights_calculation.csv"}
    test_output_dir = "outputs/figures/png"
    plot_weights_pie_chart(test_input_paths, test_output_dir)
