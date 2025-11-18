import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from datetime import datetime
from matplotlib.ticker import LogFormatterExponent # 导入 LogFormatterExponent

# 定义关键监测点 (经纬度)
# 经度范围是 50.5E 到 279.5E (即 80.5W)，所以旧金山使用正值
KEY_POINTS = {
    "Shanghai": {"lon": 121.47, "lat": 31.23},
    "Busan": {"lon": 129.04, "lat": 35.18},
    "San Francisco Nearshore": {"lon": 237.0, "lat": 37.5}, # 旧金山近海的海洋点
}

# 阈值浓度 (Bq/m³)
THRESHOLD_CONCENTRATION = 0.1

# 自定义配色方案 (深蓝色, 紫色, 蓝色, 深绿色)
# 使用 Matplotlib 的颜色名称或十六进制代码
COLORS = ['#00008B', '#800080', '#4682B4', '#228B22'] # DarkBlue, Purple, SteelBlue, ForestGreen

def plot_key_points_concentration(input_paths, output_dir):
    """
    绘制关键监测点核素浓度十年变化曲线（对数尺度）。
    Args:
        input_paths (dict): 包含所有输入文件路径的字典。
        output_dir (str): 图片输出目录。
    """
    # --- 全局中文显示设置 ---
    plt.rcParams['font.sans-serif'] = ["SimHei"]  # 使用SimHei字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 从 input_paths 中推断分辨率
    h3_path = input_paths.get('mission1_h3_nc')
    if not h3_path:
        print("  错误: 无法从 input_paths 中找到 mission1_h3_nc 路径。")
        return
    
    res_str_from_path = os.path.basename(os.path.dirname(os.path.dirname(h3_path)))
    try:
        if res_str_from_path == "1":
            resolution = 1.0
        else:
            resolution = float(res_str_from_path.replace('p', '.'))
    except ValueError:
        resolution = 1.0 # 无法解析则默认为 1.0

    print(f"\n--- 绘制关键监测点核素浓度曲线 (分辨率: {resolution}°), 包含旧金山近海点 ---")

    nuclides = ['H3', 'C14', 'Sr90', 'I129']
    
    # 创建一个包含三个子图的图表，每个子图对应一个监测点
    fig, axes = plt.subplots(nrows=len(KEY_POINTS), ncols=1, figsize=(12, 6 * len(KEY_POINTS)), sharex=True)
    if len(KEY_POINTS) == 1: # 如果只有一个监测点，axes不是数组
        axes = [axes]

    for i, (point_name, coords) in enumerate(KEY_POINTS.items()):
        ax = axes[i]
        ax.set_title(f'{point_name} ({coords["lon"]:.2f}°E, {coords["lat"]:.2f}°N)', fontsize=14)
        # 修改纵轴标签单位为 10^x Bq/m^3
        ax.set_ylabel('Concentration (10$^x$ Bq/m$^3$)', fontsize=12) 
        ax.set_yscale('log') # 对数尺度
        ax.grid(True, which="both", ls="--", c='0.7')
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # 显式设置 LogFormatterExponent 来处理负号显示
        ax.yaxis.set_major_formatter(LogFormatterExponent(labelOnlyBase=False, minor_thresholds=(np.inf, np.inf)))

        # 添加阈值线
        ax.axhline(y=THRESHOLD_CONCENTRATION, color='r', linestyle='--', label=f'Threshold ({THRESHOLD_CONCENTRATION} Bq/m$^3$)')

        for n_idx, nuclide in enumerate(nuclides):
            nuclide_key = f'mission1_{nuclide.lower()}_nc'
            file_path = input_paths.get(nuclide_key)

            if not file_path or not os.path.exists(file_path):
                print(f"  警告: 文件不存在，跳过 {nuclide} for {point_name}: {file_path if file_path else '路径未提供'}")
                continue

            try:
                ds = xr.open_dataset(file_path)
                # print(f"  成功加载 {nuclide} 数据 for {point_name}")

                # 找到最近的网格点
                lon_idx = np.abs(ds.longitude.values - coords["lon"]).argmin()
                lat_idx = np.abs(ds.latitude.values - coords["lat"]).argmin()
                
                # 提取表层浓度时间序列 (深度索引为 0)
                if 'concentration' in ds.data_vars and len(ds.concentration.dims) == 4:
                    if ds.concentration.shape[1] > 0:
                        concentration_series = ds.concentration[:, 0, lat_idx, lon_idx].values
                        # X轴数值以“年”为单位
                        time_values_for_plot = (ds.time - ds.time[0]).astype('timedelta64[D]').astype(float) / 365.25
                        
                        # 替换0或负值为一个非常小的正数，以便在对数坐标轴上显示
                        concentration_series[concentration_series <= 0] = 1e-10 

                        ax.plot(time_values_for_plot, concentration_series, label=nuclide, color=COLORS[n_idx % len(COLORS)], linewidth=1.5)
                    else:
                        print(f"    警告: {nuclide} 数据没有深度维度，无法提取表层浓度。")
                else:
                    print(f"    警告: {nuclide} 数据中缺少 'concentration' 变量或维度不正确。")

                ds.close()

            except Exception as e:
                print(f"  处理 {nuclide} for {point_name} 时发生错误: {e}")
        
        ax.legend(loc='upper right', fontsize=10) # 统一图例位置

    # 设置最底部子图的X轴标签
    axes[-1].set_xlabel('Time (Days)', fontsize=12) # 标签显示 Days
    
    # 调整整体布局
    fig.suptitle(f'Nuclide Concentration at Key Monitoring Points (Resolution: {resolution}°)', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # 调整布局以适应suptitle
    
    plot_filename = os.path.join(output_dir, f'all_nuclides_key_points_concentration_{res_str_from_path}deg.png')
    plt.savefig(plot_filename, dpi=300)
    print(f"  图表已保存: {plot_filename}")
    plt.close(fig) # 关闭整个图表，释放内存
