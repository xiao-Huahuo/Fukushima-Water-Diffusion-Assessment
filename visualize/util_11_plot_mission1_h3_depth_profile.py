import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd # 导入 pandas 用于数据整理

def plot_h3_depth_profile(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    nc_file_path = input_paths['mission1_h3_nc']
    if not os.path.exists(nc_file_path):
        print(f"错误: 任务一 H3 数据文件未找到: {nc_file_path}")
        return

    try:
        # 应用 seaborn 主题
        sns.set_theme(style="whitegrid", palette="viridis")

        # --- 在 sns.set_theme() 之后设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        ds = xr.open_dataset(nc_file_path)
        
        # 选择一个时间步（例如最后一个时间步）和一个经纬度点进行剖面图绘制
        # 假设源点附近的一个点
        target_lon = 141.03
        target_lat = 37.42
        
        lon_idx = np.abs(ds['longitude'].values - target_lon).argmin()
        lat_idx = np.abs(ds['latitude'].values - target_lat).argmin()

        # 提取最后一个时间步的深度剖面数据
        depth_profile = ds['concentration'].isel(time=-1, latitude=lat_idx, longitude=lon_idx)
        
        if 'depth' not in ds.coords:
            print("数据集中没有深度坐标，无法绘制深度剖面图。")
            return

        # 将xarray DataArray转换为pandas Series，以便seaborn处理
        df_profile = pd.DataFrame({
            '浓度': depth_profile.values,
            '深度': ds['depth'].values
        })

        plt.figure(figsize=(8, 10))
        sns.lineplot(x='浓度', y='深度', data=df_profile, marker='o', color='blue', linewidth=2)
        plt.gca().invert_yaxis() # 深度通常是从上到下增加的
        plt.xscale('log') # 浓度可能跨越多个数量级，使用对数坐标

        plt.title(f'H3 浓度深度剖面图 (经度: {ds["longitude"].values[lon_idx]:.2f}°, 纬度: {ds["latitude"].values[lat_idx]:.2f}°)\n时间: {str(depth_profile.time.values)[:10]}', fontsize=16, fontweight='bold')
        plt.xlabel('H3 浓度 (Bq/m³)', fontsize=12)
        plt.ylabel('深度 (m)', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, which="both", ls="--", c='0.7', alpha=0.7)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission1_h3_depth_profile.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制 H3 深度剖面图时发生错误: {e}")
    finally:
        if 'ds' in locals() and ds is not None:
            ds.close()

if __name__ == "__main__":
    test_input_paths = {'mission1_h3_nc': "outputs/mission1/H3/H3.nc"}
    test_output_dir = "outputs/figures/png"
    plot_h3_depth_profile(test_input_paths, test_output_dir)
