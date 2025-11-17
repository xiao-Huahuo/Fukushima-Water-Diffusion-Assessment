import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
import pandas as pd

def plot_sr90_arrival_time_kde_map(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    nc_file_path = input_paths['mission1_sr90_nc']
    if not os.path.exists(nc_file_path):
        print(f"错误: 任务一 Sr90 数据文件未找到: {nc_file_path}")
        return

    try:
        # --- 设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        ds = xr.open_dataset(nc_file_path)
        
        # 提取表面 (depth=0) 的到达时间
        if 'depth' in ds.dims and len(ds['depth']) > 1:
            arrival_time_data = ds['arrival_time'].isel(depth=0)
        else:
            arrival_time_data = ds['arrival_time'].isel(depth=0)

        lats = ds['latitude'].values
        lons = ds['longitude'].values

        # 将到达时间从天转换为年
        arrival_time_years = arrival_time_data.values / 365.25

        # 准备用于KDE的数据
        lon_flat, lat_flat = np.meshgrid(lons, lats)
        lon_flat = lon_flat.flatten()
        lat_flat = lat_flat.flatten()
        time_flat = arrival_time_years.flatten()

        # 过滤掉NaN值和0值
        valid_mask = ~np.isnan(time_flat) & (time_flat > 0.01)
        lon_valid = lon_flat[valid_mask]
        lat_valid = lat_flat[valid_mask]
        time_valid = time_flat[valid_mask]
        
        if len(lon_valid) == 0:
            print("Sr90 到达时间数据为空或全部为零，无法绘制KDE地图。")
            return

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=165))
        ax.set_extent([50, 280, -60, 65], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND, color='lightgray', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray', zorder=1)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=0)

        # 使用 seaborn.kdeplot 绘制核密度估计图
        df_kde = pd.DataFrame({'longitude': lon_valid, 'latitude': lat_valid, 'arrival_time': time_valid})
        sns.kdeplot(x='longitude', y='latitude', data=df_kde, fill=True, cmap='YlOrRd_r', 
                    alpha=0.7, levels=10, ax=ax, transform=ccrs.PlateCarree()) # _r 反转颜色，使短时间为红色
        
        ax.set_title('Sr90 表面浓度达到阈值时间核密度估计图', fontsize=16, fontweight='bold')
        ax.set_xlabel('经度', fontsize=12)
        ax.set_ylabel('纬度', fontsize=12)
        ax.tick_params(labelsize=10)
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--', color='gray')
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission1_sr90_arrival_time_kde_map.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制 Sr90 到达时间KDE地图时发生错误: {e}")
    finally:
        if 'ds' in locals() and ds is not None:
            ds.close()

if __name__ == "__main__":
    test_input_paths = {'mission1_sr90_nc': "outputs/mission1/Sr90/Sr90.nc"}
    test_output_dir = "outputs/figures/png"
    plot_sr90_arrival_time_kde_map(test_input_paths, test_output_dir)
