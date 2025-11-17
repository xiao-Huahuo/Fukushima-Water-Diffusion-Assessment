import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns

def plot_i129_global_diffusion_snapshots(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    nc_file_path = input_paths['mission1_i129_nc']
    if not os.path.exists(nc_file_path):
        print(f"错误: 任务一 I129 数据文件未找到: {nc_file_path}")
        return

    try:
        # 应用 seaborn 主题
        sns.set_theme(style="whitegrid", palette="viridis")

        # --- 在 sns.set_theme() 之后设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        ds = xr.open_dataset(nc_file_path)
        
        # 选择几个代表性的时间步进行快照 (例如，开始，1/4，1/2，3/4，结束)
        time_indices = np.linspace(0, len(ds['time']) - 1, 4, dtype=int)
        
        lats = ds['latitude'].values
        lons = ds['longitude'].values

        fig, axes = plt.subplots(2, 2, figsize=(18, 12), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=165)})
        axes = axes.flatten()

        # 过滤掉0值，计算一个合理的vmax
        if 'depth' in ds.dims and len(ds['depth']) > 1:
            all_surface_data = ds['concentration'].isel(depth=0).values
        else:
            all_surface_data = ds['concentration'].isel(depth=0).values
        valid_data = all_surface_data[all_surface_data > 0]
        global_vmax = np.percentile(valid_data, 99) if len(valid_data) > 0 else 1.0
        if global_vmax == 0: global_vmax = 1.0 # 避免vmax为0


        for i, idx in enumerate(time_indices):
            ax = axes[i]
            
            if 'depth' in ds.dims and len(ds['depth']) > 1:
                surface_concentration = ds['concentration'].isel(time=idx, depth=0)
            else:
                surface_concentration = ds['concentration'].isel(time=idx, depth=0)

            data_to_plot = surface_concentration.values
            
            # 调整 extent 以太平洋为中心
            ax.set_extent([50, 280, -60, 65], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, color='lightgray', zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=1)
            ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray', zorder=1)
            ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=0)

            mesh = ax.pcolormesh(lons, lats, data_to_plot, cmap='viridis', shading='auto',
                                 norm=plt.Normalize(vmin=0, vmax=global_vmax), # 使用统一的vmax
                                 transform=ccrs.PlateCarree(), zorder=2)
            
            ax.set_title(f'时间: {str(surface_concentration.time.values)[:10]}', fontsize=12)
            ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--', color='gray')

        fig.suptitle('I129 表面浓度全球扩散快照', fontsize=18, fontweight='bold', y=1.02)
        fig.colorbar(mesh, ax=axes.ravel().tolist(), orientation='vertical', pad=0.02, shrink=0.7, label='I129 浓度 (Bq/m³)')
        
        plt.tight_layout(rect=[0, 0, 1, 0.98]) # 调整布局以适应suptitle
        
        output_path = os.path.join(output_dir, "mission1_i129_global_diffusion_snapshots.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制 I129 全球扩散快照时发生错误: {e}")
    finally:
        if 'ds' in locals() and ds is not None:
            ds.close()

if __name__ == "__main__":
    test_input_paths = {'mission1_i129_nc': "outputs/mission1/I129/I129.nc"}
    test_output_dir = "outputs/figures/png"
    plot_i129_global_diffusion_snapshots(test_input_paths, test_output_dir)
