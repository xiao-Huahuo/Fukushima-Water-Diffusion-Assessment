import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns

def plot_sr90_arrival_time_map(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    nc_file_path = input_paths['mission1_sr90_nc']
    if not os.path.exists(nc_file_path):
        print(f"错误: 任务一 Sr90 数据文件未找到: {nc_file_path}")
        return

    try:
        # 应用 seaborn 主题
        sns.set_theme(style="whitegrid", palette="viridis")

        # --- 在 sns.set_theme() 之后设置字体 ---
        # 这样可以覆盖 seaborn 的默认字体设置
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

        # 将到达时间从天转换为年，并处理NaN值
        arrival_time_years = arrival_time_data.values / 365.25
        arrival_time_years[np.isnan(arrival_time_years)] = np.nan # 保持NaN

        # 创建图表
        fig = plt.figure(figsize=(14, 10))
        # 使用以太平洋为中心的投影
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=165))
        # 调整 extent 以太平洋为中心
        ax.set_extent([50, 280, -60, 65], crs=ccrs.PlateCarree())

        # 绘制地图特征
        ax.add_feature(cfeature.LAND, color='lightgray', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray', zorder=1)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=0)

        # 绘制热力图
        vmax = np.nanpercentile(arrival_time_years[~np.isnan(arrival_time_years)], 99) if np.any(~np.isnan(arrival_time_years)) else 10
        mesh = ax.pcolormesh(lons, lats, arrival_time_years, cmap='plasma_r', shading='auto', # plasma_r 使小值（早到达）颜色更深
                             norm=plt.Normalize(vmin=0, vmax=vmax),
                             transform=ccrs.PlateCarree(), zorder=2)

        cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.05, shrink=0.7)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Sr90 浓度达到阈值所需时间 (年)', fontsize=12)

        ax.set_title('Sr90 表面浓度达到阈值时间分布图', fontsize=16, fontweight='bold')
        ax.set_xlabel('经度', fontsize=12)
        ax.set_ylabel('纬度', fontsize=12)
        ax.tick_params(labelsize=10)
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--', color='gray')
        plt.tight_layout()

        # 保存图片
        output_path = os.path.join(output_dir, "mission1_sr90_arrival_time_map.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制 Sr90 到达时间地图时发生错误: {e}")
    finally:
        if 'ds' in locals() and ds is not None:
            ds.close()

if __name__ == "__main__":
    test_input_paths = {'mission1_sr90_nc': "outputs/mission1/Sr90/Sr90.nc"}
    test_output_dir = "outputs/figures/png"
    plot_sr90_arrival_time_map(test_input_paths, test_output_dir)
