import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D # Import for 3D plotting

def plot_i129_surface_concentration_3d_final(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    nuclide_name = "I129"
    time_description = "final"
    depth_target = 0 # 0 for surface, 50 for 50m, 200 for 200m
    output_filename_suffix = "surface_concentration_3d_final"

    nc_file_path = input_paths.get(f'mission1_{nuclide_name.lower()}_nc', '')
    if not nc_file_path or not os.path.exists(nc_file_path):
        print(f"错误: 任务一 {nuclide_name} 数据文件未找到: {nc_file_path}")
        return

    try:
        sns.set_theme(style="whitegrid", palette="viridis")
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        ds = xr.open_dataset(nc_file_path)

        # Determine time index
        time_len = len(ds['time'])
        if time_description == "initial":
            time_step_idx = 0
        elif time_description == "middle":
            time_step_idx = time_len // 2
        elif time_description == "final":
            time_step_idx = -1
        else: # Fallback to final
            time_step_idx = -1

        # Determine depth index
        depth_level_idx = 0
        depth_label = "表面"
        if 'depth' in ds.dims and depth_target is not None:
            depths = ds['depth'].values
            if len(depths) > 1:
                depth_level_idx = np.abs(depths - depth_target).argmin()
                depth_label = f"{depths[depth_level_idx]:.0f}m 深度"
            else:
                depth_target = None # Only surface available

        # Select data slice
        if 'depth' in ds.dims and len(ds['depth']) > 1:
            data_slice = ds['concentration'].isel(time=time_step_idx, depth=depth_level_idx)
        else:
            data_slice = ds['concentration'].isel(time=time_step_idx)

        lons = ds['longitude'].values
        lats = ds['latitude'].values
        
        X, Y = np.meshgrid(lons, lats)
        Z = data_slice.values

        # Filter out very small concentrations for better visualization
        Z_masked = np.copy(Z)
        Z_masked[Z_masked < 1e-10] = np.nan # Mask very small values

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Use a colormap and normalize for color mapping
        valid_Z = Z_masked[~np.isnan(Z_masked)]
        vmax_val = np.percentile(valid_Z, 99) if len(valid_Z) > 0 else 1.0
        vmin_val = np.nanmin(valid_Z) if len(valid_Z) > 0 else 0.0
        if vmax_val == 0: vmax_val = 1.0
        if vmin_val == vmax_val: vmin_val = 0.0 # Avoid zero range

        norm = plt.Normalize(vmin=vmin_val, vmax=vmax_val)
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z_masked, cmap='viridis', norm=norm,
                               linewidth=0, antialiased=False, rstride=1, cstride=1)

        # Set labels and title
        ax.set_xlabel('经度 (°E)', fontsize=12)
        ax.set_ylabel('纬度 (°N)', fontsize=12)
        ax.set_zlabel(f'{nuclide_name} 浓度 (Bq/m³)', fontsize=12)
        
        time_str = str(data_slice.time.values)[:10]
        ax.set_title(f'{nuclide_name} {depth_label}浓度 3D 曲面图 ({time_description}时间)\n时间: {time_str}', fontsize=16, fontweight='bold')

        # Add color bar
        cbar = fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
        cbar.set_label(f'{nuclide_name} 浓度 (Bq/m³)', fontsize=12)

        # Set view angle
        ax.view_init(elev=30, azim=240) # Consistent view

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"mission1_{nuclide_name.lower()}_{output_filename_suffix}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制 {nuclide_name} 浓度 3D 曲面图时发生错误: {e}")
    finally:
        if 'ds' in locals() and ds is not None:
            ds.close()

if __name__ == "__main__":
    test_input_paths = {'mission1_i129_nc': "outputs/mission1/I129/I129.nc"}
    test_output_dir = "outputs/figures/png"
    plot_i129_surface_concentration_3d_final(test_input_paths, test_output_dir)