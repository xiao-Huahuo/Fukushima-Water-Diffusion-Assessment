import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
# from mpl_toolkits.mplot3d import Axes3D # 这个库在这个图里用不到

def plot_c14_3d_concentration_slice(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    nc_file_path = input_paths['mission1_c14_nc']
    if not os.path.exists(nc_file_path):
        print(f"错误: 任务一 C14 数据文件未找到: {nc_file_path}")
        return

    try:
        # 应用 seaborn 主题
        sns.set_theme(style="whitegrid", palette="viridis")

        # --- 在 sns.set_theme() 之后设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        ds = xr.open_dataset(nc_file_path)
        
        # 选择一个时间步（例如最后一个时间步）
        concentration_data = ds['concentration'].isel(time=-1)
        lons = ds['longitude'].values
        lats = ds['latitude'].values
        depths = ds['depth'].values

        if len(depths) < 2:
            print("深度层数不足，无法绘制三维切片图。")
            return

        # 选择一个深度切片 (例如中间深度)
        mid_depth_idx = len(depths) // 2
        horizontal_slice = concentration_data.isel(depth=mid_depth_idx)

        # 选择一个经度切片 (例如源点经度)
        source_lon = 141.03
        vertical_slice_lon_idx = np.abs(lons - source_lon).argmin()
        vertical_slice_lon = concentration_data.isel(longitude=vertical_slice_lon_idx)

        # 创建图表
        fig = plt.figure(figsize=(18, 8))

        # 子图1: 水平切片
        ax1 = fig.add_subplot(121)
        vmax_val = np.percentile(horizontal_slice.values[horizontal_slice.values > 0], 99) if np.any(horizontal_slice.values > 0) else 1.0
        mesh1 = ax1.pcolormesh(lons, lats, horizontal_slice.values, cmap='viridis', shading='auto',
                               norm=plt.Normalize(vmin=0, vmax=vmax_val))
        fig.colorbar(mesh1, ax=ax1, label='C14 浓度 (Bq/m³)')
        ax1.set_title(f'C14 水平浓度切片 (深度: {depths[mid_depth_idx]:.1f}m)\n时间: {str(horizontal_slice.time.values)[:10]}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('经度', fontsize=12)
        ax1.set_ylabel('纬度', fontsize=12)
        ax1.tick_params(labelsize=10)
        ax1.grid(True, linestyle='--', alpha=0.6)

        # 子图2: 垂直切片
        ax2 = fig.add_subplot(122)
        vmax_val_vert = np.percentile(vertical_slice_lon.values[vertical_slice_lon.values > 0], 99) if np.any(vertical_slice_lon.values > 0) else 1.0
        mesh2 = ax2.pcolormesh(lats, depths, vertical_slice_lon.values.T, cmap='viridis', shading='auto',
                               norm=plt.Normalize(vmin=0, vmax=vmax_val_vert))
        fig.colorbar(mesh2, ax=ax2, label='C14 浓度 (Bq/m³)')
        ax2.set_title(f'C14 垂直浓度切片 (经度: {lons[vertical_slice_lon_idx]:.2f}°)\n时间: {str(vertical_slice_lon.time.values)[:10]}', fontsize=14, fontweight='bold')
        ax2.set_xlabel('纬度', fontsize=12)
        ax2.set_ylabel('深度 (m)', fontsize=12)
        ax2.invert_yaxis() # 深度从上到下增加
        ax2.tick_params(labelsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission1_c14_3d_concentration_slice.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制 C14 三维浓度切片图时发生错误: {e}")
    finally:
        if 'ds' in locals() and ds is not None:
            ds.close()

if __name__ == "__main__":
    test_input_paths = {'mission1_c14_nc': "outputs/mission1/C14/C14.nc"}
    test_output_dir = "outputs/figures/png"
    plot_c14_3d_concentration_slice(test_input_paths, test_output_dir)
