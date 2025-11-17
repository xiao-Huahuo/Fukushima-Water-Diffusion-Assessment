import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd

def plot_nuclide_max_concentration_comparison(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    nuclides = ["H3", "C14", "Sr90", "I129"]
    all_max_conc_data = []

    for nuclide in nuclides:
        nc_file_path = input_paths[f'mission1_{nuclide.lower()}_nc']
        if not os.path.exists(nc_file_path):
            print(f"警告: 任务一 {nuclide} 数据文件未找到: {nc_file_path}，跳过。")
            continue

        try:
            ds = xr.open_dataset(nc_file_path)
            
            # 计算每个时间步的全球最大浓度
            max_concentrations = ds['concentration'].max(dim=['depth', 'latitude', 'longitude']).values
            
            df_nuclide = pd.DataFrame({
                '时间': ds['time'].values,
                '最大浓度': max_concentrations,
                '核素': nuclide
            })
            all_max_conc_data.append(df_nuclide)
            ds.close()
        except Exception as e:
            print(f"读取 {nuclide} 最大浓度数据时发生错误: {e}")
            if 'ds' in locals() and ds is not None:
                ds.close()
            continue

    if not all_max_conc_data:
        print("没有可用的核素数据来绘制最大浓度对比图。")
        return

    df_all = pd.concat(all_max_conc_data)

    # --- 设置字体 ---
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False  

    plt.figure(figsize=(14, 8))
    sns.lineplot(x='时间', y='最大浓度', hue='核素', data=df_all, 
                 marker='o', markersize=4, linewidth=2, palette='viridis')

    plt.title('各核素全球最大浓度随时间变化', fontsize=16, fontweight='bold')
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('全球最大浓度 (Bq/m³)', fontsize=12)
    plt.yscale('log') # 浓度可能跨越多个数量级，使用对数坐标
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='核素', fontsize=10, title_fontsize=12)
    plt.grid(True, which="both", ls="--", c='0.7', alpha=0.7)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "mission1_nuclide_max_concentration_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已生成: {output_path}")

if __name__ == "__main__":
    test_input_paths = {
        'mission1_h3_nc': "outputs/mission1/H3/H3.nc",
        'mission1_c14_nc': "outputs/mission1/C14/C14.nc",
        'mission1_sr90_nc': "outputs/mission1/Sr90/Sr90.nc",
        'mission1_i129_nc': "outputs/mission1/I129/I129.nc"
    }
    test_output_dir = "outputs/figures/png"
    plot_nuclide_max_concentration_comparison(test_input_paths, test_output_dir)
