import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_h3_depth_time_heatmap(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    nc_file_path = input_paths['mission1_h3_nc']
    if not os.path.exists(nc_file_path):
        print(f"错误: 任务一 H3 数据文件未找到: {nc_file_path}")
        return

    try:
        # --- 设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        ds = xr.open_dataset(nc_file_path)
        
        # 选择一个经纬度点（例如源点）
        target_lon = 141.03
        target_lat = 37.42
        
        lon_idx = np.abs(ds['longitude'].values - target_lon).argmin()
        lat_idx = np.abs(ds['latitude'].values - target_lat).argmin()

        # 提取该点随时间变化的深度剖面数据
        concentration_profile = ds['concentration'].isel(latitude=lat_idx, longitude=lon_idx)
        
        if 'depth' not in ds.coords:
            print("数据集中没有深度坐标，无法绘制深度-时间热图。")
            return
        if 'time' not in ds.coords:
            print("数据集中没有时间坐标，无法绘制深度-时间热图。")
            return

        # 将时间坐标转换为年份
        time_years = (ds['time'].values - ds['time'].values[0]) / np.timedelta64(1, 'Y')

        plt.figure(figsize=(12, 8))
        # 使用pcolormesh绘制热图，x轴为时间，y轴为深度
        mesh = plt.pcolormesh(time_years, ds['depth'].values, concentration_profile.values.T, 
                              cmap='viridis', shading='auto',
                              norm=plt.Normalize(vmin=0, vmax=np.percentile(concentration_profile.values[concentration_profile.values > 0], 99) if np.any(concentration_profile.values > 0) else 1.0))
        
        plt.gca().invert_yaxis() # 深度通常是从上到下增加的
        cbar = plt.colorbar(mesh, label='H3 浓度 (Bq/m³)')
        cbar.ax.tick_params(labelsize=10)

        plt.title(f'H3 浓度深度-时间热图 (经度: {ds["longitude"].values[lon_idx]:.2f}°, 纬度: {ds["latitude"].values[lat_idx]:.2f}°)', fontsize=16, fontweight='bold')
        plt.xlabel('时间 (年)', fontsize=12)
        plt.ylabel('深度 (m)', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission1_h3_depth_time_heatmap.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制 H3 深度-时间热图时发生错误: {e}")
    finally:
        if 'ds' in locals() and ds is not None:
            ds.close()

if __name__ == "__main__":
    test_input_paths = {'mission1_h3_nc': "outputs/mission1/H3/H3.nc"}
    test_output_dir = "outputs/figures/png"
    plot_h3_depth_time_heatmap(test_input_paths, test_output_dir)
