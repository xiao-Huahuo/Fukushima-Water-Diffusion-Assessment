import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd

def plot_c14_depth_distribution_boxen(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    nc_file_path = input_paths['mission1_c14_nc']
    if not os.path.exists(nc_file_path):
        print(f"错误: 任务一 C14 数据文件未找到: {nc_file_path}")
        return

    try:
        # --- 设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        ds = xr.open_dataset(nc_file_path)
        
        if 'depth' not in ds.coords or len(ds['depth']) < 2:
            print("数据集中没有深度坐标或深度层数不足，无法绘制深度分布图。")
            return

        # 提取最后一个时间步的所有深度层浓度
        concentration_data = ds['concentration'].isel(time=-1)
        depths = ds['depth'].values

        # 将xarray DataArray转换为pandas DataFrame，以便seaborn处理
        # 每一列代表一个深度层，行是经纬度点
        df_conc = concentration_data.to_dataframe(name='concentration').reset_index()
        df_conc = df_conc[df_conc['concentration'] > 1e-10] # 过滤掉接近0的值

        if df_conc.empty:
            print("C14 浓度数据为空或全部为零，无法绘制深度分布图。")
            return

        # 将深度转换为分类变量，以便在x轴上显示
        df_conc['depth_str'] = df_conc['depth'].astype(str) + 'm'

        plt.figure(figsize=(14, 8))
        sns.boxenplot(x='depth_str', y=np.log10(df_conc['concentration']), data=df_conc, palette='viridis')
        
        plt.title(f'C14 浓度在不同深度层的分布 (对数尺度) (时间: {str(concentration_data.time.values)[:10]})', fontsize=16, fontweight='bold')
        plt.xlabel('深度层', fontsize=12)
        plt.ylabel('C14 浓度 (log10 Bq/m³)', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission1_c14_depth_distribution_boxen.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制 C14 深度分布箱线图时发生错误: {e}")
    finally:
        if 'ds' in locals() and ds is not None:
            ds.close()

if __name__ == "__main__":
    test_input_paths = {'mission1_c14_nc': "outputs/mission1/C14/C14.nc"}
    test_output_dir = "outputs/figures/png"
    plot_c14_depth_distribution_boxen(test_input_paths, test_output_dir)
