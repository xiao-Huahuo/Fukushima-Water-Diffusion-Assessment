import xarray as xr
import numpy as np
import os
import pandas as pd
import sys

# --- 配置部分 ---
BASE_MISSION1_OUTPUT_DIR = "outputs/mission1/"
BASE_MODEL_INPUT_DIR = "raw_data/output/" # 原始模型输入文件目录
NUCLIDES_TO_EVALUATE = ["H3", "C14", "Sr90", "I129"]

# 目标点位：北美西海岸附近太平洋 (用户指定)
TARGET_POINT = {
    "name": "North Pacific (User Defined)",
    "lon": 237.0, # 237E = 123W
    "lat": 37.0
}

def get_res_string(resolution):
    """根据分辨率生成对应的字符串，例如 1.0 -> '1', 0.5 -> '0p5'"""
    if resolution == 1.0:
        return "1"
    else:
        return str(resolution).replace('.', 'p')

def find_nearest_ocean_point_for_target(model_input_ds, target_lon, target_lat, search_radius=10):
    """
    在给定半径内寻找距离目标经纬度最近的海洋网格点。
    使用 model_input_ds 来获取陆地掩码。
    返回 (ocean_lon, ocean_lat) 或 (None, None)
    """
    # 找到目标经纬度在数据集中的最近索引
    initial_lon_idx = np.abs(model_input_ds.longitude.values - target_lon).argmin()
    initial_lat_idx = np.abs(model_input_ds.latitude.values - target_lat).argmin()
    
    # 假设深度为0 (表层)
    depth_idx = 0 

    min_dist_sq = float('inf')
    nearest_ocean_coords = (None, None)
    
    # 获取陆地掩码 (假设 u, v 速度为 0 的地方是陆地)
    # 陆地掩码是 (depth, lat, lon) 形状
    land_mask = (model_input_ds.u.isel(time=0).values == 0) & (model_input_ds.v.isel(time=0).values == 0)

    # 遍历周围网格
    for d_lat in range(-search_radius, search_radius + 1):
        for d_lon in range(-search_radius, search_radius + 1):
            current_lat_idx = initial_lat_idx + d_lat
            current_lon_idx = initial_lon_idx + d_lon

            # 检查索引是否在有效范围内
            if not (0 <= current_lat_idx < len(model_input_ds.latitude) and 
                    0 <= current_lon_idx < len(model_input_ds.longitude)):
                continue

            # 检查是否是海洋点
            # 注意: land_mask 的第一个维度是 depth，所以需要指定 depth_idx
            if not land_mask[depth_idx, current_lat_idx, current_lon_idx]:
                # 计算距离平方 (避免开方，提高效率)
                dist_sq = d_lat**2 + d_lon**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    nearest_ocean_coords = (model_input_ds.longitude.values[current_lon_idx], model_input_ds.latitude.values[current_lat_idx])
    
    return nearest_ocean_coords


def extract_and_print_series(resolution=1.0):
    print(f"--- {TARGET_POINT['name']} 点位浓度时间序列提取开始 (分辨率: {resolution}°) ---")

    lon = TARGET_POINT['lon']
    lat = TARGET_POINT['lat']
    res_str = get_res_string(resolution) # 用于构建 model_input 文件名

    # 尝试加载原始模型输入文件来获取网格信息和陆地掩码
    model_input_file = os.path.join(BASE_MODEL_INPUT_DIR, f"model_input_{res_str}deg.nc")
    if not os.path.exists(model_input_file):
        print(f"⚠️ 警告: 未找到模型输入文件 {model_input_file}。请确保 regridder 脚本已运行。")
        return

    try:
        model_input_ds = xr.open_dataset(model_input_file)
        
        # 检查初始目标点是否在陆地上
        initial_lon_idx = np.abs(model_input_ds.longitude.values - lon).argmin()
        initial_lat_idx = np.abs(model_input_ds.latitude.values - lat).argmin()
        
        # 假设陆地掩码在第一个时间步是稳定的，且深度为0 (表层)
        land_mask_initial_point = (model_input_ds.u.isel(time=0).values == 0) & \
                                  (model_input_ds.v.isel(time=0).values == 0)
        
        if land_mask_initial_point[0, initial_lat_idx, initial_lon_idx]: # 检查表层
            print(f"⚠️ 警告: 初始目标点 {TARGET_POINT['name']} ({lon}°E, {lat}°N) 位于陆地掩码内。尝试寻找最近的海洋点...")
            new_lon, new_lat = find_nearest_ocean_point_for_target(model_input_ds, lon, lat)
            if new_lon is not None and new_lat is not None:
                TARGET_POINT["lon"] = new_lon
                TARGET_POINT["lat"] = new_lat
                print(f"✅ 已将 {TARGET_POINT['name']} 移动到最近的海洋点: ({new_lon:.2f}°E, {new_lat:.2f}°N)")
                lon = new_lon # 更新当前函数中的 lon, lat
                lat = new_lat
            else:
                print(f"❌ 未能在周围找到 {TARGET_POINT['name']} 的海洋点。将无法提取有效数据。")
                model_input_ds.close()
                return
        model_input_ds.close()

    except Exception as e:
        print(f"处理模型输入文件 {model_input_file} 时出错: {e}。无法检查陆地掩码。")
        return


    for nuclide in NUCLIDES_TO_EVALUATE:
        nc_file = os.path.join(BASE_MISSION1_OUTPUT_DIR, nuclide, f"{nuclide}.nc")

        if not os.path.exists(nc_file):
            print(f"⚠️ 警告: 未找到 {nuclide} 的输出文件: {nc_file}")
            continue

        try:
            ds = xr.open_dataset(nc_file)

            if not (ds.longitude.min() <= lon <= ds.longitude.max() and
                    ds.latitude.min() <= lat <= ds.latitude.max()):
                print(f"⚠️ 警告: 目标点位 ({lon}°E, {lat}°N) 超出 {nuclide} 数据集的经纬度范围。")
                ds.close()
                continue

            conc_data = ds['concentration'].sel(
                longitude=lon,
                latitude=lat,
                method='nearest'
            ).isel(depth=0)

            time_series = conc_data.to_series()

            max_conc = time_series.max()
            final_conc = time_series.iloc[-1]

            print(f"\n{'=' * 70}")
            print(f"🔬 核素: {nuclide} | 点位: {TARGET_POINT['name']} ({lon}°E, {lat}°N)")
            print(f"  单位: Bq/m³")
            print(f"  * 10年内最大浓度: {max_conc:.6e}")
            print(f"  * 最终（10年末）浓度: {final_conc:.6e}")
            print('-' * 70)

            print("📅 浓度时间序列 (所有时间步):")
            print(time_series.to_string(float_format='{:,.6e}'.format))

            ds.close()

        except Exception as e:
            print(f"处理 {nuclide} 文件时出错: {e}")
            continue

    print("\n--- 提取完成 ---")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            res = float(sys.argv[1])
            extract_and_print_series(res)
        except ValueError:
            print("用法: python one_city_density_change_with_time.py [分辨率 (例如: 0.5, 1.0)]")
    else:
        extract_and_print_series() # 默认运行 1.0 度分辨率
