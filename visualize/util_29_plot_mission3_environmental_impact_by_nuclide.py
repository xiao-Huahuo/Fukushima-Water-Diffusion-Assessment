import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_environmental_impact_by_nuclide(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 假设 mission3.py 的 Config 类中的核素参数是可访问的
    # 并且 compute_environment_impact 函数可以被模拟或直接调用
    from mission3 import Config as Mission3Config

    # 假设有一个函数可以计算每个核素的环境影响
    # 由于 compute_environment_impact 是针对总影响的，这里需要一个模拟函数
    # 或者从一个预先计算好的CSV中读取
    
    # 为了演示，我们模拟一个数据
    try:
        # 应用 seaborn 主题
        sns.set_theme(style="whitegrid", palette="viridis")

        # --- 在 sns.set_theme() 之后设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        nuclides = list(Mission3Config.NUCLIDES.keys())
        schemes = [1, 2, 3]
        
        # 模拟每个核素在每个方案下的环境影响贡献
        # 实际应用中，这部分数据应该由 mission3.py 额外输出
        np.random.seed(42)
        impact_data = {}
        for scheme in schemes:
            impact_data[f'方案 {scheme}'] = {}
            total_impact_for_scheme = np.random.rand() * 1e12 + 1e11 # 模拟总影响
            nuclide_impacts = np.random.rand(len(nuclides))
            nuclide_impacts = nuclide_impacts / nuclide_impacts.sum() * total_impact_for_scheme # 归一化并分配
            for i, nuclide in enumerate(nuclides):
                impact_data[f'方案 {scheme}'][nuclide] = nuclide_impacts[i]
        
        df_impact = pd.DataFrame(impact_data).T
        df_impact.index.name = '方案'

        # 绘制堆叠柱状图
        fig, ax = plt.subplots(figsize=(12, 8))
        df_impact.plot(kind='bar', stacked=True, ax=ax, cmap='viridis', edgecolor='black')

        ax.set_title('不同处理方案下各核素对环境影响的贡献', fontsize=16, fontweight='bold')
        ax.set_xlabel('处理方案', fontsize=12)
        ax.set_ylabel('环境影响贡献 (Bq·m³·年)', fontsize=12)
        ax.tick_params(axis='x', rotation=0, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.legend(title='核素', loc='upper left', fontsize=10, title_fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission3_environmental_impact_by_nuclide.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制各核素环境影响贡献图时发生错误: {e}")

if __name__ == "__main__":
    # 模拟 mission3.py 的 Config
    class MockMission3Config:
        NUCLIDES = {
            "H3": {}, "C14": {}, "Sr90": {}, "I129": {}
        }
    
    import sys
    sys.modules['mission3'] = type('module', (object,), {'Config': MockMission3Config})()

    test_input_paths = {} # 此图不需要外部CSV输入
    test_output_dir = "outputs/figures/png"
    plot_environmental_impact_by_nuclide(test_input_paths, test_output_dir)
