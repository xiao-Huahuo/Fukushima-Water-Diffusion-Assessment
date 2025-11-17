import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_h3_surface_concentration_heatmap(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    nc_file_path = input_paths['mission1_h3_nc']
    if not os.path.exists(nc_file_path):
        print(f"错误: 任务一 H3 数据文件未找到: {nc_file_path}")
        return

    try:
        # 应用 seaborn 主题
        sns.set_theme(style="whitegrid", palette="viridis")

        # --- 在 sns.set_theme() 之后设置字体 ---
        # 这样可以覆盖 seaborn 的默认字体设置
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

        fig = plt.figure(figsize=(12, 8))
        # 使用以太平洋为中心的投影
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=165))
        # 调整 extent 以太平洋为中心
        ax.set_extent([50, 280, -60, 65], crs=ccrs.PlateCarree())
        
        # 绘制地图特征
        ax.add_feature(cfeature.LAND, color='lightgray', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray', zorder=1)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=0)

        # 限制vmax避免离群值影响，使颜色分布更合理
        vmax_val = np.percentile(data_to_plot[data_to_plot > 0], 99) if np.any(data_to_plot > 0) else 1.0
        mesh = ax.pcolormesh(lons, lats, data_to_plot, cmap='viridis', shading='auto',
                              norm=plt.Normalize(vmin=0, vmax=vmax_val),
                              transform=ccrs.PlateCarree()) # transform 保持数据坐标系
        
        cbar = plt.colorbar(mesh, label='H3 浓度 (Bq/m³)', pad=0.02, ax=ax, orientation='vertical', shrink=0.7)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('H3 浓度 (Bq/m³)', fontsize=12)

        ax.set_title(f'H3 表面浓度热图 (时间: {str(surface_concentration.time.values)[:10]})', fontsize=16, fontweight='bold')
        ax.set_xlabel('经度', fontsize=12)
        ax.set_ylabel('纬度', fontsize=12)
        ax.tick_params(labelsize=10)
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--', color='gray')
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission1_h3_surface_concentration_heatmap.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制 H3 表面浓度热图时发生错误: {e}")
    finally:
        if 'ds' in locals() and ds is not None:
            ds.close()

if __name__ == "__main__":
    test_input_paths = {'mission1_h3_nc': "outputs/mission1/H3/H3.nc"}
    test_output_dir = "outputs/figures/png"
    plot_h3_surface_concentration_heatmap(test_input_paths, test_output_dir)
