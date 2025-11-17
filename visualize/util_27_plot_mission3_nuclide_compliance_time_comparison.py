import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_nuclide_compliance_time_comparison(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 假设 mission3.py 的 Config 类中的核素参数是可访问的
    # 并且 compute_decay_time 函数可以被模拟或直接调用
    from mission3 import Config as Mission3Config
    from mission3 import compute_decay_time # 导入计算达标时间的函数

    try:
        # 应用 seaborn 主题
        sns.set_theme(style="whitegrid", palette="viridis")

        # --- 在 sns.set_theme() 之后设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        nuclides = Mission3Config.NUCLIDES.keys()
        schemes = [1, 2, 3]
        
        compliance_times = []
        for nuclide in nuclides:
            for scheme in schemes:
                # 调用 mission3.py 中的函数计算达标时间
                t_prime = compute_decay_time(nuclide, scheme)
                compliance_times.append({
                    '核素': nuclide,
                    '方案': f'方案 {scheme}',
                    '达标时间 (年)': t_prime
                })
        
        df_compliance = pd.DataFrame(compliance_times)

        # 绘制分组柱状图
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(x='方案', y='达标时间 (年)', hue='核素', data=df_compliance, 
                    palette='tab10', edgecolor='black')

        ax.set_title('各核素在不同处理方案下的达标时间', fontsize=16, fontweight='bold')
        ax.set_xlabel('处理方案', fontsize=12)
        ax.set_ylabel('达标时间 (年)', fontsize=12)
        ax.tick_params(axis='x', rotation=0, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_yscale('log') # 达标时间可能跨越多个数量级
        ax.legend(title='核素', loc='upper left', fontsize=10, title_fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission3_nuclide_compliance_time_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制核素达标时间对比图时发生错误: {e}")

if __name__ == "__main__":
    # 模拟 mission3.py 的 Config 和 compute_decay_time
    class MockMission3Config:
        NUCLIDES = {
            "H3": {"A0": 5000.0, "T12": 12.3, "thr": 1000, "eta": {1: 0.10, 2: 0.20, 3: 0.30}},
            "C14": {"A0": 2.0, "T12": 5730, "thr": 0.1, "eta": {1: 0.90, 2: 0.95, 3: 0.999}},
            "Sr90": {"A0": 100.0, "T12": 28.8, "thr": 10, "eta": {1: 0.85, 2: 0.95, 3: 0.995}},
            "I129": {"A0": 0.02, "T12": 1.57e7, "thr": 0.01, "eta": {1: 0.90, 2: 0.995, 3: 0.9999}}
        }

    def mock_compute_decay_time(nuclide_name, scheme):
        params = MockMission3Config.NUCLIDES[nuclide_name]
        A0 = params["A0"]
        T12 = params["T12"]
        thr = params["thr"]
        eta = params["eta"][scheme]
        lam = np.log(2) / T12
        A_treated = A0 * (1 - eta)
        if A_treated <= thr:
            return 0.0
        try:
            t_prime = np.log(A_treated / thr) / lam
            return max(0.0, t_prime)
        except:
            return 1e6

    import sys
    sys.modules['mission3'] = type('module', (object,), {
        'Config': MockMission3Config,
        'compute_decay_time': mock_compute_decay_time
    })()

    test_input_paths = {} # 此图不需要外部CSV输入
    test_output_dir = "outputs/figures/png"
    plot_nuclide_compliance_time_comparison(test_input_paths, test_output_dir)
