import xarray as xr
import numpy as np
import os
from dask.diagnostics import ProgressBar  # <-- 1. 导入进度条

# --- 1. 定义你的文件路径 (根据你的描述) ---
INPUT_DIR = "CMEMS/"
OUTPUT_DIR = "output/"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "model_input_1deg.nc")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 你下载的 5 个核心文件
FILES_TO_PROCESS = {
    'uv': os.path.join(INPUT_DIR, 'uv.nc'),
    'w': os.path.join(INPUT_DIR, 'w.nc'),
    't': os.path.join(INPUT_DIR, 'T.nc'),
    's': os.path.join(INPUT_DIR, 'S.nc'),
    'h': os.path.join(INPUT_DIR, 'H.nc')
}

# --- 2. 定义你的 1°x1° 目标网格 ---
# (网格中心点从 -59.5°S 到 64.5°N)
target_lats = np.arange(-59.5, 65, 1.0)

# (网格中心点从 50.5°E 到 279.5°E)
# (280°E 即 80°W)
target_lons = np.arange(50.5, 280, 1.0)

# 创建一个只包含经纬度坐标的"模板"网格
target_grid = xr.Dataset(
    {
        "latitude": (("latitude",), target_lats),
        "longitude": (("longitude",), target_lons),
    }
)

print(f"目标网格已创建: {len(target_lats)} 纬度 x {len(target_lons)} 经度")
print(f"纬度范围: {target_lats.min()}° to {target_lats.max()}°")
print(f"经度范围: {target_lons.min()}° to {target_lons.max()}°")

# --- 3. 循环处理所有文件 ---
all_regridded_datasets = []

# 强制 xarray 使用 dask  <-- (你把下面这行删掉)
# xr.set_options(use_dask=True)

for var_key, file_path in FILES_TO_PROCESS.items():
    print(f"\n--- 正在处理: {file_path} ---")

    try:
        # 打开 NetCDF 文件 (使用 dask 'chunks' 加快读取)
        ds_high_res = xr.open_dataset(file_path, chunks={'time': 1})

        print("正在准备降采样 (Regridding) 到 1°x1° 网格...")
        ds_regridded_lazy = ds_high_res.interp_like(
            target_grid,
            method="linear"  # 线性插值法, 速度最快
        )

        # (可选) 清理变量名，使其更统一
        if var_key == 'uv':
            ds_regridded_lazy = ds_regridded_lazy.rename({'uo': 'u', 'vo': 'v'})
        elif var_key == 'w':
            ds_regridded_lazy = ds_regridded_lazy.rename({'wo': 'w'})
        elif var_key == 't':
            ds_regridded_lazy = ds_regridded_lazy.rename({'thetao': 'T'})
        elif var_key == 's':
            ds_regridded_lazy = ds_regridded_lazy.rename({'so': 'S'})
        elif var_key == 'h':
            ds_regridded_lazy = ds_regridded_lazy.rename({
                'deptho': 'H',
                'deptho_lev': 'h_level',
                'mask': 'land_mask'
            })

        # --- 2. 这是你需要的“日志” (进度条) ---
        print(f"*** {var_key} 开始执行降采样 (此步耗时较长，请等待进度条)... ***")
        with ProgressBar():
            # .load() 会强制 xarray 立即执行“惰性”计算
            # 并显示一个 dask 进度条
            ds_regridded_computed = ds_regridded_lazy.load()

        all_regridded_datasets.append(ds_regridded_computed)
        print(f"*** 文件 {var_key} 处理完毕。 ***")

    except FileNotFoundError:
        print(f"警告: 文件未找到 {file_path}, 将跳过。")
    except Exception as e:
        print(f"处理 {file_path} 时出错: {e}")

# --- 4. 最终合并 ---
if len(all_regridded_datasets) != 5:
    print(f"错误: 只处理了 {len(all_regridded_datasets)} / 5 个文件。请检查文件路径。")
else:
    # (这一步现在会非常快, 因为计算已在循环中完成)
    print("\n--- 正在合并所有降采样后的数据 (此步很快) ---")
    final_dataset = xr.merge(all_regridded_datasets)

    final_dataset.attrs['description'] = 'Final 1x1 deg model input for diffusion simulation.'
    final_dataset.attrs['source'] = 'CMEMS (001_024) regridded to 1-deg'

    # --- 5. 保存到磁盘 ---
    print(f"正在保存最终的模型输入文件到: {OUTPUT_FILE}")
    try:
        final_dataset.to_netcdf(OUTPUT_FILE)
        print("--- ✅ 全部完成! ---")
        print(f"你的模型输入文件 '{OUTPUT_FILE}' 已准备就绪。")
    except Exception as e:
        print(f"保存失败: {e}")