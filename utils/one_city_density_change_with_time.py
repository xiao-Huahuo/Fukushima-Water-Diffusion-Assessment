import xarray as xr
import numpy as np
import os
import pandas as pd

# --- 配置部分 ---
OUTPUT_DIR = "../outputs/mission1/"
NUCLIDES_TO_EVALUATE = ["H3", "C14", "Sr90", "I129"]

# 目标点位：釜山近海 (根据您提供的坐标，选取最近的海洋网格中心)
TARGET_POINT = {
    "name": "Busan_Nearshore",
    "lon": 130.0,
    "lat": 35.0
}


def extract_and_print_series():
    print("--- 釜山点位浓度时间序列提取开始 ---")

    lon = TARGET_POINT['lon']
    lat = TARGET_POINT['lat']

    for nuclide in NUCLIDES_TO_EVALUATE:
        nc_file = os.path.join(OUTPUT_DIR, nuclide, f"{nuclide}.nc")

        if not os.path.exists(nc_file):
            print(f"⚠️ 警告: 未找到 {nuclide} 的输出文件: {nc_file}")
            continue

        try:
            # 1. 加载数据
            ds = xr.open_dataset(nc_file)

            # 2. 提取时间序列 (表层 depth=0)
            conc_data = ds['concentration'].sel(
                longitude=lon,
                latitude=lat,
                method='nearest'
            ).isel(depth=0)

            # 3. 转换为 Pandas Series
            time_series = conc_data.to_series()

            # 4. 打印摘要
            max_conc = time_series.max()
            final_conc = time_series.iloc[-1]

            print(f"\n{'=' * 70}")
            print(f"🔬 核素: {nuclide} | 点位: {TARGET_POINT['name']} ({lon}°E, {lat}°N)")
            print(f"  单位: Bq/m³")
            print(f"  * 10年内最大浓度: {max_conc:.6e}")
            print(f"  * 最终（10年末）浓度: {final_conc:.6e}")
            print('-' * 70)

            # 5. 打印完整时间序列
            print("📅 浓度时间序列 (所有时间步):")
            # 使用 float_format 确保输出为科学计数法
            print(time_series.to_string(float_format='{:,.6e}'.format))

            ds.close()

        except Exception as e:
            print(f"处理 {nuclide} 文件时出错: {e} (可能缺少xarray库或坐标不在海洋网格内)")
            continue

    print("\n--- 提取完成 ---")


if __name__ == "__main__":
    extract_and_print_series()