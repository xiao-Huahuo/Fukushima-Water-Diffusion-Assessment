import xarray as xr
import numpy as np
import os
import sys

def test_model_input_nc(resolution=0.5):
    """
    测试 model_input_{resolution}deg.nc 文件是否正确生成，并检查其内容。
    """
    res_str = str(resolution).replace('.', 'p')
    file_path = os.path.join("output", f"model_input_{res_str}deg.nc")

    print(f"--- 正在测试文件: {file_path} ---")

    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在。请确保 regridder_precise.py 已运行。")
        return

    try:
        ds = xr.open_dataset(file_path)
        print("\n✅ 文件成功加载。")

        print("\n--- 数据集信息 ---")
        print(ds)

        print("\n--- 维度和坐标 ---")
        print(f"时间步数: {len(ds.time)}")
        print(f"深度层数: {len(ds.depth)}")
        print(f"纬度点数: {len(ds.latitude)}")
        print(f"经度点数: {len(ds.longitude)}")

        # 验证分辨率
        # 确保维度长度大于1，否则无法计算分辨率
        actual_lat_res = round(float(ds.latitude[1] - ds.latitude[0]), 2) if len(ds.latitude) > 1 else resolution
        actual_lon_res = round(float(ds.longitude[1] - ds.longitude[0]), 2) if len(ds.longitude) > 1 else resolution
        print(f"实际纬度分辨率: {actual_lat_res}° (预期: {resolution}°)")
        print(f"实际经度分辨率: {actual_lon_res}° (预期: {resolution}°)")

        if actual_lat_res != resolution or actual_lon_res != resolution:
            print(f"⚠️ 警告: 实际分辨率 ({actual_lat_res}x{actual_lon_res}) 与预期 ({resolution}x{resolution}) 不符。")

        print("\n--- 变量统计 ---")
        for var_name in ['u', 'v', 'w', 'T', 'S', 'H', 'h_level', 'land_mask']: # 增加 h_level 和 land_mask
            if var_name in ds:
                print(f"  变量 '{var_name}':")
                print(f"    Dimensions: {ds[var_name].dims}")
                print(f"    Coordinates: {ds[var_name].coords}") # 打印坐标信息，有助于调试
                
                data_to_process = ds[var_name]
                if 'time' in ds[var_name].dims:
                    # 对于时间依赖变量，取前几个时间步的切片进行统计，避免加载全部数据
                    data_to_process = ds[var_name].isel(time=slice(0, min(3, len(ds.time))))
                
                # 将 xarray.DataArray 转换为 NumPy 数组进行统计，以避免 xarray 内部的维度处理问题
                data_to_process_np = data_to_process.values

                if data_to_process_np.size > 0:
                    # 使用 NumPy 的 nanmin/nanmax/nanmean 来处理可能存在的 NaN 值
                    min_val = np.nanmin(data_to_process_np)
                    max_val = np.nanmax(data_to_process_np)
                    mean_val = np.nanmean(data_to_process_np)
                    nan_count = np.sum(np.isnan(data_to_process_np))

                    print(f"    Min: {min_val:.4f}")
                    print(f"    Max: {max_val:.4f}")
                    print(f"    Mean: {mean_val:.4f}")
                    print(f"    NaN count: {nan_count}")
                    # 检查是否全为零 (或全为 NaN)
                    if min_val == 0 and max_val == 0 and mean_val == 0 and nan_count == 0:
                        print(f"    ⚠️ 警告: 变量 '{var_name}' 在统计切片中全为零。")
                else:
                    print("    (数据切片为空，无法计算统计信息)")
            else:
                print(f"  变量 '{var_name}' 不存在于数据集中。")
        
        print("\n✅ 文件内容检查完毕。")

    except Exception as e:
        print(f"处理文件 '{file_path}' 时出错: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            res = float(sys.argv[1])
            test_model_input_nc(res)
        except ValueError:
            print("用法: python test_nc.py [分辨率 (例如: 0.5, 1.0)]")
    else:
        test_model_input_nc()
