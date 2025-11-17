import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns

def plot_i129_global_diffusion_quiver_map(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    nc_file_path = input_paths['mission1_i129_nc']
    if not os.path.exists(nc_file_path):
        print(f"错误: 任务一 I129 数据文件未找到: {nc_file_path}")
        return

    try:
        # --- 设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        ds = xr.open_dataset(nc_file_path)
        
        lats = ds['latitude'].values
        lons = ds['longitude'].values

        # 提取最后一个时间步的表面浓度 (depth=0)
        if 'depth' in ds.dims and len(ds['depth']) > 1:
            surface_concentration = ds['concentration'].isel(time=-1, depth=0)
            u_vel = ds['u'].isel(time=-1, depth=0) if 'u' in ds else None
            v_vel = ds['v'].isel(time=-1, depth=0) if 'v' in ds else None
        else:
            surface_concentration = ds['concentration'].isel(time=-1, depth=0)
            u_vel = ds['u'].isel(time=-1, depth=0) if 'u' in ds else None
            v_vel = ds['v'].isel(time=-1, depth=0) if 'v' in ds else None

        data_to_plot = surface_concentration.values
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=165))
        ax.set_extent([50, 280, -60, 65], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND, color='lightgray', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray', zorder=1)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=0)

        # 绘制浓度热图作为背景
        vmax_val = np.percentile(data_to_plot[data_to_plot > 0], 99) if np.any(data_to_plot > 0) else 1.0
        mesh = ax.pcolormesh(lons, lats, data_to_plot, cmap='viridis', shading='auto',
                             norm=plt.Normalize(vmin=0, vmax=vmax_val),
                             transform=ccrs.PlateCarree(), zorder=2)
        
        cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.05, shrink=0.7)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('I129 浓度 (Bq/m³)', fontsize=12)

        # 绘制洋流矢量图
        if u_vel is not None and v_vel is not None:
            # 稀疏化矢量图，避免过于密集
            skip = 5 # 每隔5个点绘制一个矢量
            ax.quiver(lons[::skip], lats[::skip], u_vel.values[::skip, ::skip], v_vel.values[::skip, ::skip],
                      color='red', scale=5, transform=ccrs.PlateCarree(), zorder=3) # scale调整矢量长度
            print("洋流矢量图已绘制。")
        else:
            print("警告: 数据集中缺少 'u' 或 'v' 速度分量，无法绘制洋流矢量图。")

        ax.set_title(f'I129 表面浓度与洋流矢量图 (时间: {str(surface_concentration.time.values)[:10]})', fontsize=16, fontweight='bold')
        ax.set_xlabel('经度', fontsize=12)
        ax.set_ylabel('纬度', fontsize=12)
        ax.tick_params(labelsize=10)
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--', color='gray')
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission1_i129_global_diffusion_quiver_map.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制 I129 表面浓度与洋流矢量图时发生错误: {e}")
    finally:
        if 'ds' in locals() and ds is not None:
            ds.close()

if __name__ == "__main__":
    test_input_paths = {'mission1_i129_nc': "outputs/mission1/I129/I129.nc"}
    test_output_dir = "outputs/figures/png"
    plot_i129_global_diffusion_quiver_map(test_input_paths, test_output_dir)
