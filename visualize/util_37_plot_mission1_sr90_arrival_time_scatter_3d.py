import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def plot_sr90_arrival_time_scatter_3d(input_paths, output_dir):
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

        # 提取到达时间数据
        arrival_time_data = ds['arrival_time'].values # shape (depth, lat, lon)
        
        # 将到达时间从天转换为年
        arrival_time_years = arrival_time_data / 365.25

        # 准备用于3D散点图的数据
        # 过滤掉NaN值，只绘制有到达时间的点
        valid_indices = ~np.isnan(arrival_time_years)
        
        # 使用np.meshgrid创建完整的经度、纬度、深度网格
        lon_grid, lat_grid, depth_grid = np.meshgrid(lons, lats, depths, indexing='ij')

        # 展平并过滤
        plot_lons = lon_grid[valid_indices]
        plot_lats = lat_grid[valid_indices]
        plot_depths = depth_grid[valid_indices]
        plot_arrival_times = arrival_time_years[valid_indices]

        if len(plot_arrival_times) == 0:
            print("Sr90 到达时间数据为空，无法绘制3D散点图。")
            return

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 使用seaborn的调色板进行颜色映射
        # 根据到达时间进行颜色编码
        scatter = ax.scatter(plot_lons, plot_lats, plot_depths, 
                             c=plot_arrival_times, cmap='plasma_r', 
                             s=10, alpha=0.6, edgecolor='none') # s控制点的大小

        ax.set_title('Sr90 浓度达到阈值的三维空间分布与时间', fontsize=16, fontweight='bold')
        ax.set_xlabel('经度', fontsize=12)
        ax.set_ylabel('纬度', fontsize=12)
        ax.set_zlabel('深度 (m)', fontsize=12)
        ax.tick_params(labelsize=10)
        
        # 深度轴反向
        ax.invert_zaxis()

        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Sr90 到达时间 (年)', fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        # 调整视角
        ax.view_init(elev=20, azim=-60)

        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission1_sr90_arrival_time_scatter_3d.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制 Sr90 到达时间3D散点图时发生错误: {e}")
    finally:
        if 'ds' in locals() and ds is not None:
            ds.close()

if __name__ == "__main__":
    test_input_paths = {'mission1_sr90_nc': "outputs/mission1/Sr90/Sr90.nc"}
    test_output_dir = "outputs/figures/png"
    plot_sr90_arrival_time_scatter_3d(test_input_paths, test_output_dir)
