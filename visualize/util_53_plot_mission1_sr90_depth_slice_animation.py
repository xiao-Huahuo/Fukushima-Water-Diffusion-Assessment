import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from matplotlib.animation import FuncAnimation

def plot_sr90_depth_slice_animation(input_paths, output_dir):
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
        
        lats = ds['latitude'].values
        lons = ds['longitude'].values
        depths = ds['depth'].values

        if len(depths) < 2:
            print("深度层数不足，无法绘制深度切片动画。")
            return
        
        # 选择一个时间步（例如最后一个时间步）
        concentration_data_at_time = ds['concentration'].isel(time=-1)

        # 过滤掉0值，计算一个合理的vmax
        valid_data = concentration_data_at_time.values[concentration_data_at_time.values > 0]
        vmax_val = np.percentile(valid_data, 99) if len(valid_data) > 0 else 1.0
        if vmax_val == 0: vmax_val = 1.0 # 避免vmax为0

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)

        # 初始帧
        mesh = ax.pcolormesh(lons, lats, concentration_data_at_time.isel(depth=0).values, 
                             cmap='viridis', shading='auto',
                             norm=plt.Normalize(vmin=0, vmax=vmax_val))
        
        cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', pad=0.05, shrink=0.7)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Sr90 浓度 (Bq/m³)', fontsize=12)

        title = ax.set_title(f'Sr90 浓度深度切片动画 (深度: {depths[0]:.1f}m, 时间: {str(concentration_data_at_time.time.values)[:10]})', 
                             fontsize=16, fontweight='bold')
        ax.set_xlabel('经度', fontsize=12)
        ax.set_ylabel('纬度', fontsize=12)
        ax.tick_params(labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)

        def update(frame):
            data = concentration_data_at_time.isel(depth=frame).values
            mesh.set_array(data.ravel())
            title.set_text(f'Sr90 浓度深度切片动画 (深度: {depths[frame]:.1f}m, 时间: {str(concentration_data_at_time.time.values)[:10]})')
            return mesh, title

        ani = FuncAnimation(fig, update, frames=len(depths), blit=True, interval=500)

        output_path = os.path.join(output_dir, "mission1_sr90_depth_slice_animation.gif")
        ani.save(output_path, writer='pillow', dpi=150) # 使用pillow writer保存gif
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制 Sr90 深度切片动画时发生错误: {e}")
    finally:
        if 'ds' in locals() and ds is not None:
            ds.close()

if __name__ == "__main__":
    test_input_paths = {'mission1_sr90_nc': "outputs/mission1/Sr90/Sr90.nc"}
    test_output_dir = "outputs/figures/gif" # 动画通常保存为gif
    plot_sr90_depth_slice_animation(test_input_paths, test_output_dir)
