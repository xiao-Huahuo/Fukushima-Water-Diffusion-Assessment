import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_nuclide_arrival_time_kde(input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    nuclides = ["H3", "C14", "Sr90", "I129"]
    arrival_times_data = {}

    for nuclide in nuclides:
        nc_file_path = input_paths[f'mission1_{nuclide.lower()}_nc']
        if not os.path.exists(nc_file_path):
            print(f"警告: 任务一 {nuclide} 数据文件未找到: {nc_file_path}，跳过。")
            continue

        try:
            ds = xr.open_dataset(nc_file_path)
            # 提取所有深度层的到达时间，并转换为年
            arrival_time_years = ds['arrival_time'].values.flatten() / 365.25
            # 过滤掉NaN值和0值（0值可能表示未到达或源点）
            arrival_time_years = arrival_time_years[~np.isnan(arrival_time_years)]
            arrival_time_years = arrival_time_years[arrival_time_years > 0.01] # 过滤掉非常小的到达时间

            if len(arrival_time_years) > 0:
                arrival_times_data[nuclide] = arrival_time_years
            ds.close()
        except Exception as e:
            print(f"读取 {nuclide} 到达时间数据时发生错误: {e}")
            if 'ds' in locals() and ds is not None:
                ds.close()
            continue

    if not arrival_times_data:
        print("没有可用的核素到达时间数据来绘制 KDE 图。")
        return

    # --- 设置字体 ---
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False  

    plt.figure(figsize=(12, 8))
    for nuclide, data in arrival_times_data.items():
        sns.kdeplot(data, label=nuclide, fill=True, alpha=0.5, linewidth=2)

    plt.title('各核素浓度达到阈值时间的概率密度分布', fontsize=16, fontweight='bold')
    plt.xlabel('到达时间 (年)', fontsize=12)
    plt.ylabel('概率密度', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='核素', fontsize=10, title_fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "mission1_nuclide_arrival_time_kde.png")
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
    plot_nuclide_arrival_time_kde(test_input_paths, test_output_dir)
