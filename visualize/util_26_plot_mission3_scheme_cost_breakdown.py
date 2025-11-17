import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_scheme_cost_breakdown(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 假设 mission3.py 的 Config 类中的成本参数是可访问的
    # 或者从一个模拟的配置文件中读取
    # 这里直接使用 mission3.py 中的 Config 值
    from mission3 import Config as Mission3Config

    try:
        # 应用 seaborn 主题
        sns.set_theme(style="whitegrid", palette="viridis")

        # --- 在 sns.set_theme() 之后设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        # 模拟成本数据
        schemes = [1, 2, 3]
        cost_data = []

        for scheme in schemes:
            build_cost = Mission3Config.BUILD_COST[scheme]
            op_cost_total = Mission3Config.OP_COST * Mission3Config.YEARS
            storage_cost_initial = 0
            storage_cost_annual_total = 0

            if scheme == 3:
                storage_cost_initial = Mission3Config.STORAGE_COST_INITIAL
                storage_cost_annual_total = Mission3Config.STORAGE_COST_ANNUAL * Mission3Config.YEARS
                op_cost_total = 0 # 方案3的运维成本可能被存储成本取代

            cost_data.append({
                '方案': f'方案 {scheme}',
                '建造成本': build_cost / 1e9, # 单位转换为十亿
                '运行维护成本': op_cost_total / 1e9,
                '初始存储设施成本': storage_cost_initial / 1e9,
                '年度存储成本': storage_cost_annual_total / 1e9
            })
        
        df_costs = pd.DataFrame(cost_data)
        df_costs = df_costs.set_index('方案')

        # 绘制堆叠柱状图
        fig, ax = plt.subplots(figsize=(12, 8))
        df_costs.plot(kind='bar', stacked=True, ax=ax, cmap='Paired', edgecolor='black')

        ax.set_title('不同处理方案成本构成', fontsize=16, fontweight='bold')
        ax.set_xlabel('处理方案', fontsize=12)
        ax.set_ylabel('成本 (十亿)', fontsize=12)
        ax.tick_params(axis='x', rotation=0, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.legend(title='成本类型', loc='upper left', fontsize=10, title_fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission3_scheme_cost_breakdown.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制方案成本构成图时发生错误: {e}")

if __name__ == "__main__":
    # 假设 mission3.py 存在于当前路径或已添加到PYTHONPATH
    # 或者直接模拟 Config 数据
    class MockMission3Config:
        YEARS = 30
        OP_COST = 5.5e8
        BUILD_COST = {1: 1.2e9, 2: 2.0e9, 3: 8.0e9}
        STORAGE_COST_ANNUAL = 2.0e8
        STORAGE_COST_INITIAL = 5.0e9
    
    # 替换 mission3.Config 为 MockMission3Config
    import sys
    sys.modules['mission3'] = type('module', (object,), {'Config': MockMission3Config})()

    test_input_paths = {} # 此图不需要外部CSV输入
    test_output_dir = "outputs/figures/png"
    plot_scheme_cost_breakdown(test_input_paths, test_output_dir)
