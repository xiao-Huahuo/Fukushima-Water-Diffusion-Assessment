import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns

def plot_all_nuclides_surface_comparison(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    nuclides = ["H3", "C14", "Sr90", "I129"]
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=165)})
    axes = axes.flatten()
    
    global_vmax = 0.0 # 用于统一色标

    # 应用 seaborn 主题
    sns.set_theme(style="whitegrid", palette="viridis")

    # --- 在 sns.set_theme() 之后设置字体 ---
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False  

    for i, nuclide in enumerate(nuclides):
        nc_file_path = input_paths[f'mission1_{nuclide.lower()}_nc']

        if not os.path.exists(nc_file_path):
            print(f"警告: 任务一 {nuclide} 数据文件未找到: {nc_file_path}，跳过。")
            axes[i].set_title(f'{nuclide} (数据缺失)', fontsize=14)
            axes[i].set_extent([50, 280, -60, 65], crs=ccrs.PlateCarree())
            axes[i].add_feature(cfeature.LAND, color='lightgray', zorder=0)
            axes[i].add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=1)
            axes[i].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--', color='gray')
            continue

        try:
            ds = xr.open_dataset(nc_file_path)
            ax = axes[i]
            
            if 'depth' in ds.dims and len(ds['depth']) > 1:
                surface_concentration = ds['concentration'].isel(time=-1, depth=0)
            else:
                surface_concentration = ds['concentration'].isel(time=-1, depth=0)

            data_to_plot = surface_concentration.values
            lats = ds['latitude'].values
            lons = ds['longitude'].values
            
            # 调整 extent 以太平洋为中心
            ax.set_extent([50, 280, -60, 65], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, color='lightgray', zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=1)
            ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray', zorder=1)
            ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=0)

            # 更新全局最大值
            current_vmax = np.percentile(data_to_plot[data_to_plot > 0], 99) if np.any(data_to_plot > 0) else 1.0
            global_vmax = max(global_vmax, current_vmax)

            mesh = ax.pcolormesh(lons, lats, data_to_plot, cmap='viridis', shading='auto',
                                 norm=plt.Normalize(vmin=0, vmax=current_vmax), # 先用各自的vmax
                                 transform=ccrs.PlateCarree())
            
            ax.set_title(f'{nuclide} 表面浓度', fontsize=14)
            ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--', color='gray')
            ds.close()

        except Exception as e:
            print(f"绘制 {nuclide} 表面浓度对比图时发生错误: {e}")
            if 'ds' in locals() and ds is not None:
                ds.close()
            axes[i].set_title(f'{nuclide} (绘图失败)', fontsize=14)
            axes[i].set_extent([50, 280, -60, 65], crs=ccrs.PlateCarree())
            axes[i].add_feature(cfeature.LAND, color='lightgray', zorder=0)
            axes[i].add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=1)
            axes[i].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--', color='gray')
            continue

    # 重新绘制所有图，使用统一的色标
    for i, nuclide in enumerate(nuclides):
        nc_file_path = input_paths[f'mission1_{nuclide.lower()}_nc']
        
        if not os.path.exists(nc_file_path):
            continue # 跳过数据缺失或绘图失败的子图

        try:
            ds = xr.open_dataset(nc_file_path)
            ax = axes[i]
            
            if 'depth' in ds.dims and len(ds['depth']) > 1:
                surface_concentration = ds['concentration'].isel(time=-1, depth=0)
            else:
                surface_concentration = ds['concentration'].isel(time=-1, depth=0)

            data_to_plot = surface_concentration.values
            lats = ds['latitude'].values
            lons = ds['longitude'].values

            # 清除旧的pcolormesh，重新绘制
            for collection in ax.collections:
                collection.remove()
            
            mesh = ax.pcolormesh(lons, lats, data_to_plot, cmap='viridis', shading='auto',
                                 norm=plt.Normalize(vmin=0, vmax=global_vmax), # 使用统一的vmax
                                 transform=ccrs.PlateCarree())
            ds.close()
        except Exception as e:
            print(f"重新绘制 {nuclide} 表面浓度对比图时发生错误: {e}")
            if 'ds' in locals() and ds is not None:
                ds.close()
            continue


    fig.suptitle(f'各核素表面浓度对比 (时间: {str(ds["time"].isel(time=-1).values)[:10]})', fontsize=18, fontweight='bold', y=1.02)
    fig.colorbar(mesh, ax=axes.ravel().tolist(), orientation='vertical', pad=0.02, shrink=0.7, label='浓度 (Bq/m³)')
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # 调整布局以适应suptitle
    
    output_path = os.path.join(output_dir, "mission1_all_nuclides_surface_comparison.png")
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
    plot_all_nuclides_surface_comparison(test_input_paths, test_output_dir)
