import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_nuclide_eta_comparison_heatmap(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 假设 mission3.py 的 Config 类中的核素参数是可访问的
    from mission3 import Config as Mission3Config

    try:
        # --- 设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        nuclides = list(Mission3Config.NUCLIDES.keys())
        schemes = [1, 2, 3]
        
        eta_data = {}
        for nuclide in nuclides:
            eta_data[nuclide] = [Mission3Config.NUCLIDES[nuclide]["eta"][s] for s in schemes]
        
        df_eta = pd.DataFrame(eta_data, index=[f'方案 {s}' for s in schemes])

        plt.figure(figsize=(10, 7))
        sns.heatmap(df_eta, annot=True, cmap='YlGnBu', fmt=".1%", linewidths=.5, linecolor='black',
                    cbar_kws={'label': '去除率'})
        
        plt.title('各核素在不同处理方案下的去除率对比', fontsize=16, fontweight='bold')
        plt.xlabel('核素', fontsize=12)
        plt.ylabel('处理方案', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission3_nuclide_eta_comparison_heatmap.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制核素去除率对比热图时发生错误: {e}")

if __name__ == "__main__":
    # 模拟 mission3.py 的 Config
    class MockMission3Config:
        NUCLIDES = {
            "H3": {"eta": {1: 0.10, 2: 0.20, 3: 0.30}},
            "C14": {"eta": {1: 0.90, 2: 0.95, 3: 0.999}},
            "Sr90": {"eta": {1: 0.85, 2: 0.95, 3: 0.995}},
            "I129": {"eta": {1: 0.90, 2: 0.995, 3: 0.9999}}
        }
    
    import sys
    sys.modules['mission3'] = type('module', (object,), {'Config': MockMission3Config})()

    test_input_paths = {} # 此图不需要外部CSV输入
    test_output_dir = "outputs/figures/png"
    plot_nuclide_eta_comparison_heatmap(test_input_paths, test_output_dir)
