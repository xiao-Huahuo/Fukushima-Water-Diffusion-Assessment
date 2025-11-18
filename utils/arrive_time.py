# # 原本的analyze_arrival_times.py,直接以命令行呈现
# import xarray as xr
# import numpy as np
# import os
# from datetime import datetime, timedelta
#
# # 城市及其经纬度
# TARGET_CITIES = {
#     "Busan_Nearshore": (130.0, 35.0),
#     "Shanghai_Nearshore": (123.0, 31.0),
#     "SanFrancisco_Nearshore": (236.0, 37.0)
# }
#
# # 阈值
# C_THRESHOLD = 1e-5  # [Bq/m³]
#
# # 核素列表（根据你的输出目录）
# NUCLIDES = ["H3", "C14", "Sr90", "I129"]
# OUTPUT_DIR = "../outputs/mission1"
#
#
# def find_nearest_index(array, value):
#     return np.abs(array - value).argmin()
#
#
# def main():
#     results = {}
#     for nuclide in NUCLIDES:
#         nc_file = os.path.join(OUTPUT_DIR, nuclide, f"{nuclide}.nc")
#         if not os.path.exists(nc_file):
#             print(f"[跳过] 文件不存在: {nc_file}")
#             continue
#
#         ds = xr.open_dataset(nc_file)
#         conc = ds['concentration'].values  # [time, depth, lat, lon]
#         time_vals = ds['time'].values
#         lat_vals = ds['latitude'].values
#         lon_vals = ds['longitude'].values
#
#         results[nuclide] = {}
#         for city, (lon_c, lat_c) in TARGET_CITIES.items():
#             i_lon = find_nearest_index(lon_vals, lon_c)
#             i_lat = find_nearest_index(lat_vals, lat_c)
#
#             # 假设海面层为第0层
#             conc_city = conc[:, 0, i_lat, i_lon]
#
#             # 找到第一个超过阈值的时间
#             exceed_indices = np.where(conc_city >= C_THRESHOLD)[0]
#             if len(exceed_indices) == 0:
#                 arrival_time = None
#             else:
#                 arrival_time = time_vals[exceed_indices[0]]
#
#             results[nuclide][city] = arrival_time
#
#     # 输出结果
#     print("=" * 60)
#     print(f"核素到达阈值 {C_THRESHOLD} Bq/m³ 的时间 (近似)")
#     print("=" * 60)
#     for nuclide, city_times in results.items():
#         print(f"\n核素: {nuclide}")
#         for city, arrival in city_times.items():
#             if arrival is None:
#                 print(f"  {city}: 未达到阈值")
#             else:
#                 # 转换为 datetime
#                 if isinstance(arrival, np.datetime64):
#                     arrival_dt = pd.to_datetime(arrival)
#                     print(f"  {city}: {arrival_dt.strftime('%Y-%m-%d')}")
#                 else:
#                     print(f"  {city}: {arrival}")
#     print("=" * 60)
#
#
# if __name__ == "__main__":
#     import pandas as pd
#
#     main()


"""
生成核素最早到达时间的全球地图数据
用于任务一图1：最早到达时间地图
"""
import xarray as xr
import numpy as np
import pandas as pd
import os
from datetime import datetime

# 配置
NUCLIDES = ["H3", "C14", "Sr90", "I129"]
OUTPUT_DIR = "../outputs/mission1"
C_THRESHOLD = 0.1  # Bq/m³ (根据题目要求)


def compute_arrival_time_map(nuclide):
    """计算单个核素的到达时间地图"""
    print(f"\n处理核素: {nuclide}")

    nc_file = os.path.join(OUTPUT_DIR, nuclide, f"{nuclide}.nc")
    if not os.path.exists(nc_file):
        print(f"  [错误] 文件不存在: {nc_file}")
        return None

    # 加载数据
    ds = xr.open_dataset(nc_file)
    conc = ds['concentration'].values  # [time, depth, lat, lon]
    time_vals = ds['time'].values
    lat_vals = ds['latitude'].values
    lon_vals = ds['longitude'].values

    print(f"  数据形状: {conc.shape}")
    print(f"  时间范围: {len(time_vals)} 步")

    # 初始化到达时间地图 (使用表层数据)
    surface_conc = conc[:, 0, :, :]  # [time, lat, lon]
    n_time, n_lat, n_lon = surface_conc.shape

    arrival_time = np.full((n_lat, n_lon), np.nan)  # 单位：天数
    arrival_date = np.full((n_lat, n_lon), '', dtype='U20')  # 日期字符串

    print(f"  开始计算到达时间...")
    count_reached = 0

    # 遍历每个网格点
    for i in range(n_lat):
        if i % 20 == 0:
            print(f"    处理进度: {i}/{n_lat} ({100 * i / n_lat:.1f}%)")

        for j in range(n_lon):
            series = surface_conc[:, i, j]

            # 找到首次超过阈值的时间索引
            exceed_idx = np.where(series >= C_THRESHOLD)[0]

            if len(exceed_idx) > 0:
                first_idx = exceed_idx[0]

                # 转换为天数（相对于开始时间）
                if isinstance(time_vals[first_idx], np.datetime64):
                    delta = (time_vals[first_idx] - time_vals[0]) / np.timedelta64(1, 'D')
                    arrival_time[i, j] = delta
                    arrival_date[i, j] = pd.to_datetime(time_vals[first_idx]).strftime('%Y-%m-%d')
                else:
                    arrival_time[i, j] = first_idx  # 如果是时间步数
                    arrival_date[i, j] = f"Step_{first_idx}"

                count_reached += 1

    print(f"  完成！到达网格数: {count_reached}/{n_lat * n_lon} ({100 * count_reached / (n_lat * n_lon):.2f}%)")

    ds.close()

    return {
        'arrival_time': arrival_time,
        'arrival_date': arrival_date,
        'lat': lat_vals,
        'lon': lon_vals
    }


def save_arrival_data(nuclide, data):
    """保存到达时间数据为CSV和NPY格式"""
    if data is None:
        return

    output_subdir = os.path.join(OUTPUT_DIR, nuclide)

    # 保存为NumPy数组（方便MATLAB加载）
    npy_file = os.path.join(output_subdir, f"{nuclide}_arrival_time.npy")
    np.save(npy_file, data['arrival_time'])
    print(f"  已保存: {npy_file}")

    # 保存坐标
    np.save(os.path.join(output_subdir, f"{nuclide}_lat.npy"), data['lat'])
    np.save(os.path.join(output_subdir, f"{nuclide}_lon.npy"), data['lon'])

    # 保存为CSV（可读性更好）
    csv_file = os.path.join(output_subdir, f"{nuclide}_arrival_time.csv")
    df = pd.DataFrame(
        data['arrival_time'],
        index=data['lat'],
        columns=data['lon']
    )
    df.to_csv(csv_file)
    print(f"  已保存: {csv_file}")

    # 统计信息
    valid_times = data['arrival_time'][~np.isnan(data['arrival_time'])]
    if len(valid_times) > 0:
        print(f"  到达时间统计:")
        print(f"    最早: {valid_times.min():.1f} 天")
        print(f"    最晚: {valid_times.max():.1f} 天")
        print(f"    平均: {valid_times.mean():.1f} 天")


def main():
    print("=" * 60)
    print("生成核素最早到达时间地图数据")
    print(f"阈值: {C_THRESHOLD} Bq/m³")
    print("=" * 60)

    for nuclide in NUCLIDES:
        data = compute_arrival_time_map(nuclide)
        save_arrival_data(nuclide, data)

    print("\n" + "=" * 60)
    print("全部完成！")
    print("MATLAB 读取示例:")
    print("  arrival_time = readNPY('../outputs/H3/H3_arrival_time.npy');")
    print("  lat = readNPY('../outputs/H3/H3_lat.npy');")
    print("  lon = readNPY('../outputs/H3/H3_lon.npy');")
    print("  pcolor(lon, lat, arrival_time); shading interp;")
    print("=" * 60)


if __name__ == "__main__":
    main()