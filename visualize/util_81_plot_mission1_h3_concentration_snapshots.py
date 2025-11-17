import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns

def plot_concentration_snapshots(input_paths, output_dir):
    """
    绘制单个核素在四个不同时间点的表面浓度快照。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 以 H3 为例，这个函数可以适用于任何一个核素的NC文件
    nc_file_path = input_paths.get('mission1_h3_nc', '') 
    if not nc_file_path or not os.path.exists(nc_file_path):
        print(f"错误: 数据文件未找到: {nc_file_path}")
        return

    try:
        # 应用 seaborn 主题
        sns.set_theme(style="whitegrid", palette="viridis")

        # --- 在 sns.set_theme() 之后设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        ds = xr.open_dataset(nc_file_path)
        
        # 选取四个均匀分布的时间点
        time_indices = np.linspace(0, len(ds['time']) - 1, 4, dtype=int)
        
        lats = ds['latitude'].values
        lons = ds['longitude'].values

        # 创建 2x2 子图网格，并设置共享的投影
        fig, axes = plt.subplots(2, 2, figsize=(18, 12), 
                                 subplot_kw={'projection': ccrs.PlateCarree(central_longitude=165)})
        axes = axes.flatten()

        # 提取表面浓度数据
        if 'depth' in ds.dims and len(ds['depth']) > 1:
            surface_data = ds['concentration'].isel(depth=0)
        else:
            surface_data = ds['concentration']
        
        # --- 优化vmax的计算 ---
        # 以最后一个时间快照的99百分位数为基准，确保后期细节可见
        last_snapshot_data = surface_data.isel(time=time_indices[-1]).values
        valid_last_data = last_snapshot_data[last_snapshot_data > 0]
        global_vmax = np.percentile(valid_last_data, 99) if len(valid_last_data) > 0 else 1.0
        if global_vmax == 0: global_vmax = 1.0

        # 遍历四个时间点和四个子图
        for i, time_idx in enumerate(time_indices):
            ax = axes[i]
            
            concentration_slice = surface_data.isel(time=time_idx)
            data_to_plot = concentration_slice.values
            
            # 设置地图范围
            ax.set_extent([50, 280, -60, 65], crs=ccrs.PlateCarree())
            
            # 添加地图特征
            ax.add_feature(cfeature.LAND, color='lightgray', zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=1)
            ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray', zorder=1)
            ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=0)

            # 绘制浓度数据
            mesh = ax.pcolormesh(lons, lats, data_to_plot, cmap='viridis', shading='auto',
                                 norm=plt.Normalize(vmin=0, vmax=global_vmax),
                                 transform=ccrs.PlateCarree(), zorder=2)
            
            # 设置子图标题
            time_str = str(concentration_slice.time.values)[:10]
            ax.set_title(f'时间: {time_str}', fontsize=12)
            ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--', color='gray')

        # --- 优化颜色条位置 ---
        # 调整子图布局，为颜色条留出空间
        fig.subplots_adjust(right=0.85) 
        # 在图的右侧创建一个新的坐标轴用于颜色条
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
        fig.colorbar(mesh, cax=cbar_ax, label='H3 浓度 (Bq/m³)')

        # 添加总标题
        fig.suptitle('H3 表面浓度扩散快照', fontsize=18, fontweight='bold')
        
        output_path = os.path.join(output_dir, "mission1_h3_concentration_snapshots.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制浓度快照图时发生错误: {e}")
    finally:
        if 'ds' in locals() and ds is not None:
            ds.close()

if __name__ == "__main__":
    # 使用 mission1 的 H3 数据进行测试
    test_input_paths = {'mission1_h3_nc': "outputs/mission1/H3/H3.nc"}
    test_output_dir = "outputs/figures/png"
    plot_concentration_snapshots(test_input_paths, test_output_dir)
