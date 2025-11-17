import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd # 导入 pandas 用于数据整理

def plot_h3_concentration_exceedance_area(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    nc_file_path = input_paths['mission1_h3_nc']
    if not os.path.exists(nc_file_path):
        print(f"错误: 任务一 H3 数据文件未找到: {nc_file_path}")
        return

    # 假设 H3 的阈值 (Bq/m³)
    H3_THRESHOLD = 0.1 # 与 mission1.py 中的 C_THRESHOLD 和 Config.C_THRESHOLD 保持一致

    try:
        # 应用 seaborn 主题
        sns.set_theme(style="whitegrid", palette="viridis")

        # --- 在 sns.set_theme() 之后设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        ds = xr.open_dataset(nc_file_path)
        
        # 提取所有时间步的表面浓度 (depth=0)
        if 'depth' in ds.dims and len(ds['depth']) > 1:
            surface_concentrations = ds['concentration'].isel(depth=0)
        else:
            surface_concentrations = ds['concentration'].isel(depth=0)

        time_coords = ds['time'].values
        
        # 计算每个时间步浓度超过阈值的网格点数量
        exceedance_counts = []
        for t_idx in range(len(time_coords)):
            exceedance_mask = surface_concentrations.isel(time=t_idx) > H3_THRESHOLD
            exceedance_counts.append(exceedance_mask.sum().item())
        
        # 将数据转换为DataFrame，以便seaborn处理
        df_exceedance = pd.DataFrame({
            '时间': time_coords,
            '超过阈值的网格点数量': exceedance_counts
        })

        plt.figure(figsize=(12, 7))
        sns.lineplot(x='时间', y='超过阈值的网格点数量', data=df_exceedance, 
                     marker='o', linestyle='-', color='red', linewidth=2, alpha=0.8)

        plt.title(f'H3 表面浓度超过 {H3_THRESHOLD} Bq/m³ 的网格点数量随时间变化', fontsize=16, fontweight='bold')
        plt.xlabel('时间', fontsize=12)
        plt.ylabel('超过阈值的网格点数量', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, which="both", ls="--", c='0.7', alpha=0.7)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission1_h3_exceedance_area_timeseries.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制 H3 浓度超阈值区域时间序列图时发生错误: {e}")
    finally:
        if 'ds' in locals() and ds is not None:
            ds.close()

if __name__ == "__main__":
    test_input_paths = {'mission1_h3_nc': "outputs/mission1/H3/H3.nc"}
    test_output_dir = "outputs/figures/png"
    plot_h3_concentration_exceedance_area(test_input_paths, test_output_dir)
