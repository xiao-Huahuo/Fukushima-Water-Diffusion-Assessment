import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns

def plot_sr90_surface_concentration_contour(input_paths, output_dir):
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
        
        # 提取最后一个时间步的表面浓度 (depth=0)
        if 'depth' in ds.dims and len(ds['depth']) > 1:
            surface_concentration = ds['concentration'].isel(time=-1, depth=0)
        else:
            surface_concentration = ds['concentration'].isel(time=-1, depth=0)

        data_to_plot = surface_concentration.values
        lats = ds['latitude'].values
        lons = ds['longitude'].values

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=165))
        ax.set_extent([50, 280, -60, 65], crs=ccrs.PlateCarree())

        # 绘制地图特征
        ax.add_feature(cfeature.LAND, color='lightgray', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray', zorder=1)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=0)

        # 绘制等高线图
        # 设定等高线级别，可以根据数据范围调整
        levels = np.logspace(np.log10(np.nanmin(data_to_plot[data_to_plot > 0]) + 1e-10), 
                             np.log10(np.nanpercentile(data_to_plot[data_to_plot > 0], 99)), 10)
        
        contourf = ax.contourf(lons, lats, data_to_plot, levels=levels, cmap='YlGnBu', 
                               transform=ccrs.PlateCarree(), zorder=2, extend='max')
        contour = ax.contour(lons, lats, data_to_plot, levels=levels, colors='black', 
                             linewidths=0.5, transform=ccrs.PlateCarree(), zorder=3)
        
        cbar = plt.colorbar(contourf, ax=ax, orientation='vertical', pad=0.05, shrink=0.7)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Sr90 浓度 (Bq/m³)', fontsize=12)

        ax.set_title(f'Sr90 表面浓度等高线图 (时间: {str(surface_concentration.time.values)[:10]})', fontsize=16, fontweight='bold')
        ax.set_xlabel('经度', fontsize=12)
        ax.set_ylabel('纬度', fontsize=12)
        ax.tick_params(labelsize=10)
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--', color='gray')
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission1_sr90_surface_concentration_contour.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制 Sr90 表面浓度等高线图时发生错误: {e}")
    finally:
        if 'ds' in locals() and ds is not None:
            ds.close()

if __name__ == "__main__":
    test_input_paths = {'mission1_sr90_nc': "outputs/mission1/Sr90/Sr90.nc"}
    test_output_dir = "outputs/figures/png"
    plot_sr90_surface_concentration_contour(test_input_paths, test_output_dir)
