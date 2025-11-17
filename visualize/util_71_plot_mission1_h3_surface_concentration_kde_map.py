import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
import pandas as pd

def plot_h3_surface_concentration_kde_map(input_paths, output_dir):
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
        
        # 提取最后一个时间步的表面浓度 (depth=0)
        if 'depth' in ds.dims and len(ds['depth']) > 1:
            surface_concentration = ds['concentration'].isel(time=-1, depth=0)
        else:
            surface_concentration = ds['concentration'].isel(time=-1, depth=0)

        data_to_plot = surface_concentration.values
        lats = ds['latitude'].values
        lons = ds['longitude'].values

        # 过滤掉0值，只保留有浓度的地方
        valid_lons = lons[np.any(data_to_plot > 1e-10, axis=0)]
        valid_lats = lats[np.any(data_to_plot > 1e-10, axis=1)]
        
        # 为了绘制KDE，需要将经纬度数据展平，并根据浓度值进行加权
        # 这里简化为只对有浓度值的区域绘制KDE
        lon_flat, lat_flat = np.meshgrid(lons, lats)
        lon_flat = lon_flat.flatten()
        lat_flat = lat_flat.flatten()
        conc_flat = data_to_plot.flatten()

        # 过滤掉浓度为0的区域
        valid_mask = conc_flat > 1e-10
        lon_valid = lon_flat[valid_mask]
        lat_valid = lat_flat[valid_mask]
        
        if len(lon_valid) == 0:
            print("H3 表面浓度数据为空或全部为零，无法绘制KDE地图。")
            return

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=165))
        ax.set_extent([50, 280, -60, 65], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND, color='lightgray', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray', zorder=1)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=0)

        # 使用 seaborn.kdeplot 绘制核密度估计图
        # 注意：kdeplot直接在ax上绘制，但需要将数据转换为DataFrame
        df_kde = pd.DataFrame({'longitude': lon_valid, 'latitude': lat_valid})
        sns.kdeplot(x='longitude', y='latitude', data=df_kde, fill=True, cmap='Reds', 
                    alpha=0.7, levels=10, ax=ax, transform=ccrs.PlateCarree())
        
        ax.set_title(f'H3 表面浓度核密度估计图 (时间: {str(surface_concentration.time.values)[:10]})', fontsize=16, fontweight='bold')
        ax.set_xlabel('经度', fontsize=12)
        ax.set_ylabel('纬度', fontsize=12)
        ax.tick_params(labelsize=10)
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--', color='gray')
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission1_h3_surface_concentration_kde_map.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制 H3 表面浓度KDE地图时发生错误: {e}")
    finally:
        if 'ds' in locals() and ds is not None:
            ds.close()

if __name__ == "__main__":
    test_input_paths = {'mission1_h3_nc': "outputs/mission1/H3/H3.nc"}
    test_output_dir = "outputs/figures/png"
    plot_h3_surface_concentration_kde_map(test_input_paths, test_output_dir)
