import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd # 导入 pandas 用于数据整理
from datetime import datetime

def plot_sr90_time_series_at_ports(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    nc_file_path = input_paths['mission1_sr90_nc']
    if not os.path.exists(nc_file_path):
        print(f"错误: 任务一 Sr90 数据文件未找到: {nc_file_path}")
        return

    # 定义港口坐标 (示例，可根据实际情况调整)
    ports = {
        '上海': {'lon': 121.47, 'lat': 31.23},
        '洛杉矶': {'lon': 241.73, 'lat': 33.70}, # 约 -118.27 E
        '釜山': {'lon': 129.04, 'lat': 35.18}
    }
    
    try:
        # 应用 seaborn 主题
        sns.set_theme(style="whitegrid", palette="viridis")

        # --- 在 sns.set_theme() 之后设置字体 ---
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  

        ds = xr.open_dataset(nc_file_path)
        time_coords = ds['time'].values
        
        all_port_data = []

        for port_name, coords in ports.items():
            lon_idx = np.abs(ds['longitude'].values - coords['lon']).argmin()
            lat_idx = np.abs(ds['latitude'].values - coords['lat']).argmin()
            
            # 提取表面 (depth=0) 的浓度时间序列
            if 'depth' in ds.dims and len(ds['depth']) > 1:
                conc_series = ds['concentration'].isel(depth=0, latitude=lat_idx, longitude=lon_idx).values
            else:
                conc_series = ds['concentration'].isel(depth=0, latitude=lat_idx, longitude=lon_idx).values

            df_port = pd.DataFrame({
                '时间': time_coords,
                '浓度': conc_series,
                '港口': f'{port_name} ({ds["longitude"].values[lon_idx]:.2f}°, {ds["latitude"].values[lat_idx]:.2f}°)'
            })
            all_port_data.append(df_port)
        
        df_all_ports = pd.concat(all_port_data)

        plt.figure(figsize=(14, 8))
        sns.lineplot(x='时间', y='浓度', hue='港口', data=df_all_ports, 
                     marker='o', markersize=4, linewidth=2, palette='deep')

        plt.title('Sr90 在主要港口附近海域表面浓度时间序列', fontsize=16, fontweight='bold')
        plt.xlabel('时间', fontsize=12)
        plt.ylabel('Sr90 浓度 (Bq/m³)', fontsize=12)
        plt.yscale('log') # 浓度可能跨越多个数量级，使用对数坐标
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title='港口', fontsize=10, title_fontsize=12)
        plt.grid(True, which="both", ls="--", c='0.7', alpha=0.7)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "mission1_sr90_time_series_at_ports.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"绘制 Sr90 港口浓度时间序列图时发生错误: {e}")
    finally:
        if 'ds' in locals() and ds is not None:
            ds.close()

if __name__ == "__main__":
    test_input_paths = {'mission1_sr90_nc': "outputs/mission1/Sr90/Sr90.nc"}
    test_output_dir = "outputs/figures/png"
    plot_sr90_time_series_at_ports(test_input_paths, test_output_dir)
