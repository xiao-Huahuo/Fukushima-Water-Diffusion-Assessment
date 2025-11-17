import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_mission1_source_term_comparison(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 假设 mission1.py 的 Config 类中的核素参数是可访问的
    from mission1 import Config as Mission1Config

    try:
        # --- 设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        nuclides = list(Mission1Config.NUCLIDES_SIMULATION.keys())
        
        source_term_data = []
        for nuclide in nuclides:
            # 模拟 Q_BQ 的计算，因为 Q_BQ 是动态赋值的
            # 这里使用 BASE_Q_RATE 和 Q_factor 来计算相对源项强度
            q_rate_tons_per_day = Mission1Config.BASE_Q_RATE * Mission1Config.NUCLIDES_SIMULATION[nuclide]["Q_factor"]
            # 转换为 Bq/s (假设 1吨 = 1e12 Bq，这是在 mission1.py 中使用的转换因子)
            q_bq_per_s = q_rate_tons_per_day * 1e12 / 86400 
            
            source_term_data.append({
                '核素': nuclide,
                '源项强度 (Bq/s)': q_bq_per_s
            })
        
        df_source = pd.DataFrame(source_term_data)
        df_source = df_source.sort_values('源项强度 (Bq/s)', ascending=False)

        plt.figure(figsize=(10, 7))
        sns.barplot(x='核素', y='源项强度 (Bq/s)', data=df_source, palette='rocket')
        
        plt.title('各核素源项强度对比', fontsize=16, fontweight='bold')
        plt.xlabel('核素', fontsize=12)
        plt.ylabel('源项强度 (Bq/s)', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.yscale('log') # 源项强度可能跨越多个数量级
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission1_source_term_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制源项强度对比图时发生错误: {e}")

if __name__ == "__main__":
    # 模拟 mission1.py 的 Config
    class MockMission1Config:
        NUCLIDES_SIMULATION = {
            "H3": {"T12": 12.3, "Q_factor": 1.0},
            "C14": {"T12": 5730.0, "Q_factor": 0.5},
            "Sr90": {"T12": 28.8, "Q_factor": 2.0},
            "I129": {"T12": 1.57e7, "Q_factor": 0.3}
        }
        BASE_Q_RATE = 100.0 # 吨/天

    import sys
    sys.modules['mission1'] = type('module', (object,), {'Config': MockMission1Config})()

    test_input_paths = {} # 此图不需要外部CSV输入
    test_output_dir = "outputs/figures/png"
    plot_mission1_source_term_comparison(test_input_paths, test_output_dir)
