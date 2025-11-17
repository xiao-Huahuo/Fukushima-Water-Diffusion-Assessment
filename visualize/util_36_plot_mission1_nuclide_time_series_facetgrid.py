import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd

def plot_nuclide_time_series_facetgrid(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    nuclides = ["H3", "C14", "Sr90", "I129"]
    all_nuclide_data = []

    # 从 mission1.py 的 Config 中获取源点坐标 (这些是固定值，可以作为绘图脚本的内部常量)
    SOURCE_LON = 141.03
    SOURCE_LAT = 37.42
    SOURCE_DEPTH_IDX = 0 # 假设源点在最表层

    for nuclide in nuclides:
        nc_file_path = input_paths[f'mission1_{nuclide.lower()}_nc']
        if not os.path.exists(nc_file_path):
            print(f"警告: 任务一 {nuclide} 数据文件未找到: {nc_file_path}，跳过。")
            continue

        try:
            ds = xr.open_dataset(nc_file_path)
            
            lon_idx = np.abs(ds['longitude'].values - SOURCE_LON).argmin()
            lat_idx = np.abs(ds['latitude'].values - SOURCE_LAT).argmin()

            if 'depth' in ds.dims and len(ds['depth']) > 1:
                conc_series = ds['concentration'][:, SOURCE_DEPTH_IDX, lat_idx, lon_idx].values
            else:
                conc_series = ds['concentration'][:, SOURCE_DEPTH_IDX, lat_idx, lon_idx].values

            df_nuclide = pd.DataFrame({
                '时间': ds['time'].values,
                '浓度': conc_series,
                '核素': nuclide
            })
            all_nuclide_data.append(df_nuclide)
            ds.close()
        except Exception as e:
            print(f"读取 {nuclide} 源点浓度数据时发生错误: {e}")
            if 'ds' in locals() and ds is not None:
                ds.close()
            continue

    if not all_nuclide_data:
        print("没有可用的核素数据来绘制时间序列分面图。")
        return

    df_all = pd.concat(all_nuclide_data)

    # --- 设置字体 ---
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False  

    # 绘制分面图
    g = sns.relplot(
        data=df_all,
        x="时间", y="浓度",
        col="核素", col_wrap=2, # 每行显示2个核素
        height=4, aspect=1.5,
        kind="line",
        facet_kws={'sharey': False, 'sharex': True} # 每个核素有独立的Y轴，共享X轴
    )
    g.set_axis_labels("时间", "浓度 (Bq/m³)")
    g.set_titles("核素: {col_name}")
    g.set(yscale="log") # 浓度可能跨越多个数量级，使用对数坐标
    g.tight_layout()
    plt.suptitle(f'各核素在源点 ({SOURCE_LON}°E, {SOURCE_LAT}°N) 浓度随时间变化', fontsize=16, fontweight='bold', y=1.02)

    output_path = os.path.join(output_dir, "mission1_nuclide_time_series_facetgrid.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已生成: {output_path}")

if __name__ == "__main__":
    test_input_paths = {
        'mission1_h3_nc': "outputs/mission1/H3/H3.nc",
        'mission1_c14_nc': "outputs/mission1/C14/C14.nc",
        'mission1_sr90_nc': "outputs/mission1/Sr90/Sr90.nc",
        'mission1_i129_nc': "outputs/mission1/I129/I129.nc"
    }
    test_output_dir = "outputs/figures/png"
    plot_nuclide_time_series_facetgrid(test_input_paths, test_output_dir)
