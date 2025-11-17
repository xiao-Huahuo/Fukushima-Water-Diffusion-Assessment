import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns

def plot_sr90_depth_integrated_concentration_map(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    nc_file_path = input_paths['mission1_sr90_nc']
    if not os.path.exists(nc_file_path):
        print(f"错误: 任务一 Sr90 数据文件未找到: {nc_file_path}")
        return

    try:
        # 应用 seaborn 主题
        sns.set_theme(style="whitegrid", palette="viridis")

        # --- 在 sns.set_theme() 之后设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        ds = xr.open_dataset(nc_file_path)
        
        # 提取最后一个时间步的所有深度层浓度
        concentration_data = ds['concentration'].isel(time=-1)
        lats = ds['latitude'].values
        lons = ds['longitude'].values
        depths = ds['depth'].values

        if len(depths) < 2:
            print("深度层数不足，无法计算深度积分浓度。")
            return

        # 计算深度积分浓度
        # 假设深度间隔均匀，或者使用 xarray 的 integrate 方法
        # 这里简化为对深度轴求和，乘以平均深度间隔
        depth_diffs = np.diff(depths)
        avg_dz = np.mean(depth_diffs) if len(depth_diffs) > 0 else 10.0 # 默认10m
        
        # 简单的深度积分：C_integrated = sum(C * dz)
        # 如果ds.depth是层中心，可以考虑梯形积分
        # 这里为了简化，直接求和并乘以一个代表性的深度间隔
        depth_integrated_concentration = concentration_data.sum(dim='depth') * avg_dz

        data_to_plot = depth_integrated_concentration.values
        
        fig = plt.figure(figsize=(14, 10))
        # 使用以太平洋为中心的投影
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=165))
        # 调整 extent 以太平洋为中心
        ax.set_extent([50, 280, -60, 65], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND, color='lightgray', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray', zorder=1)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=0)

        vmax_val = np.percentile(data_to_plot[data_to_plot > 0], 99) if np.any(data_to_plot > 0) else 1.0
        mesh = ax.pcolormesh(lons, lats, data_to_plot, cmap='viridis', shading='auto',
                             norm=plt.Normalize(vmin=0, vmax=vmax_val),
                             transform=ccrs.PlateCarree(), zorder=2)

        cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.05, shrink=0.7)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Sr90 深度积分浓度 (Bq/m²)', fontsize=12)

        ax.set_title(f'Sr90 深度积分浓度热图 (时间: {str(concentration_data.time.values)[:10]})', fontsize=16, fontweight='bold')
        ax.set_xlabel('经度', fontsize=12)
        ax.set_ylabel('纬度', fontsize=12)
        ax.tick_params(labelsize=10)
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--', color='gray')
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission1_sr90_depth_integrated_concentration_map.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制 Sr90 深度积分浓度地图时发生错误: {e}")
    finally:
        if 'ds' in locals() and ds is not None:
            ds.close()

if __name__ == "__main__":
    test_input_paths = {'mission1_sr90_nc': "outputs/mission1/Sr90/Sr90.nc"}
    test_output_dir = "outputs/figures/png"
    plot_sr90_depth_integrated_concentration_map(test_input_paths, test_output_dir)
