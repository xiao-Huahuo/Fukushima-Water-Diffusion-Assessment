import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_nuclide_decay_half_life_bar(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 假设 mission3.py 的 Config 类中的核素参数是可访问的
    from mission3 import Config as Mission3Config

    try:
        # --- 设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        nuclides = list(Mission3Config.NUCLIDES.keys())
        
        half_life_data = []
        for nuclide in nuclides:
            half_life_data.append({
                '核素': nuclide,
                '半衰期 (年)': Mission3Config.NUCLIDES[nuclide]["T12"]
            })
        
        df_half_life = pd.DataFrame(half_life_data)
        df_half_life = df_half_life.sort_values('半衰期 (年)', ascending=False)

        plt.figure(figsize=(10, 7))
        sns.barplot(x='核素', y='半衰期 (年)', data=df_half_life, palette='viridis')
        
        plt.title('各核素半衰期对比', fontsize=16, fontweight='bold')
        plt.xlabel('核素', fontsize=12)
        plt.ylabel('半衰期 (年)', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.yscale('log') # 半衰期可能跨越多个数量级
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission3_nuclide_decay_half_life_bar.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制核素半衰期对比图时发生错误: {e}")

if __name__ == "__main__":
    # 模拟 mission3.py 的 Config
    class MockMission3Config:
        NUCLIDES = {
            "H3": {"T12": 12.3},
            "C14": {"T12": 5730},
            "Sr90": {"T12": 28.8},
            "I129": {"T12": 1.57e7}
        }
    
    import sys
    sys.modules['mission3'] = type('module', (object,), {'Config': MockMission3Config})()

    test_input_paths = {} # 此图不需要外部CSV输入
    test_output_dir = "outputs/figures/png"
    plot_nuclide_decay_half_life_bar(test_input_paths, test_output_dir)
