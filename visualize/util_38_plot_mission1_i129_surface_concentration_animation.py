import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation

def plot_i129_surface_concentration_animation(input_paths, output_dir):
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

        if 'depth' in ds.dims and len(ds['depth']) > 1:
            surface_concentrations = ds['concentration'].isel(depth=0)
        else:
            surface_concentrations = ds['concentration'].isel(depth=0)

        # 过滤掉0值，计算一个合理的vmax
        valid_data = surface_concentrations.values[surface_concentrations.values > 0]
        vmax_val = np.percentile(valid_data, 99) if len(valid_data) > 0 else 1.0
        if vmax_val == 0: vmax_val = 1.0 # 避免vmax为0

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=165))
        ax.set_extent([50, 280, -60, 65], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND, color='lightgray', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray', zorder=1)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=0)

        # 初始帧
        mesh = ax.pcolormesh(lons, lats, surface_concentrations.isel(time=0).values, 
                             cmap='viridis', shading='auto',
                             norm=plt.Normalize(vmin=0, vmax=vmax_val),
                             transform=ccrs.PlateCarree(), zorder=2)
        
        cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', pad=0.05, shrink=0.7)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('I129 浓度 (Bq/m³)', fontsize=12)

        title = ax.set_title(f'I129 表面浓度扩散动画 (时间: {str(surface_concentrations.time.values[0])[:10]})', 
                             fontsize=16, fontweight='bold')
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--', color='gray')

        def update(frame):
            data = surface_concentrations.isel(time=frame).values
            mesh.set_array(data.ravel())
            title.set_text(f'I129 表面浓度扩散动画 (时间: {str(surface_concentrations.time.values[frame])[:10]})')
            return mesh, title

        # 动画帧数可以调整，这里选择一部分帧以避免文件过大
        num_frames = len(ds['time'])
        # 每隔N帧取一帧，或者只取前M帧
        frames_to_animate = np.arange(0, num_frames, max(1, num_frames // 50)) # 最多50帧
        
        ani = FuncAnimation(fig, update, frames=frames_to_animate, blit=True, interval=200)

        output_path = os.path.join(output_dir, "mission1_i129_surface_concentration_animation.gif")
        ani.save(output_path, writer='pillow', dpi=150) # 使用pillow writer保存gif
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制 I129 表面浓度扩散动画时发生错误: {e}")
    finally:
        if 'ds' in locals() and ds is not None:
            ds.close()

if __name__ == "__main__":
    test_input_paths = {'mission1_i129_nc': "outputs/mission1/I129/I129.nc"}
    test_output_dir = "outputs/figures/gif" # 动画通常保存为gif
    plot_i129_surface_concentration_animation(test_input_paths, test_output_dir)
