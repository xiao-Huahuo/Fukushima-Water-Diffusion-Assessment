import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_h3_surface_concentration_violin(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    nc_file_path = input_paths['mission1_h3_nc']
    if not os.path.exists(nc_file_path):
        print(f"错误: 任务一 H3 数据文件未找到: {nc_file_path}")
        return

    try:
        # --- 设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        ds = xr.open_dataset(nc_file_path)
        
        # 提取最后一个时间步的表面浓度 (depth=0)
        if 'depth' in ds.dims and len(ds['depth']) > 1:
            surface_concentration = ds['concentration'].isel(time=-1, depth=0)
        else:
            surface_concentration = ds['concentration'].isel(time=-1, depth=0)

        # 将xarray DataArray转换为pandas Series，以便seaborn处理
        # 过滤掉0值，因为它们可能代表未受影响区域，会扭曲分布
        data_to_plot = surface_concentration.values.flatten()
        data_to_plot = data_to_plot[data_to_plot > 1e-10] # 过滤掉接近0的值

        if len(data_to_plot) == 0:
            print("H3 表面浓度数据为空或全部为零，无法绘制小提琴图。")
            return

        plt.figure(figsize=(10, 7))
        sns.violinplot(y=np.log10(data_to_plot), inner="quartile", palette="viridis")
        
        plt.title(f'H3 表面浓度分布 (对数尺度) (时间: {str(surface_concentration.time.values)[:10]})', fontsize=16, fontweight='bold')
        plt.ylabel('H3 浓度 (log10 Bq/m³)', fontsize=12)
        plt.xticks([]) # 隐藏X轴刻度，因为只有一个分布
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission1_h3_surface_concentration_violin.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制 H3 表面浓度小提琴图时发生错误: {e}")
    finally:
        if 'ds' in locals() and ds is not None:
            ds.close()

if __name__ == "__main__":
    test_input_paths = {'mission1_h3_nc': "outputs/mission1/H3/H3.nc"}
    test_output_dir = "outputs/figures/png"
    plot_h3_surface_concentration_violin(test_input_paths, test_output_dir)
