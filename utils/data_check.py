import xarray as xr
import numpy as np
import os

# --- 配置部分 ---
OUTPUT_DIR = "../outputs/mission1/"
NUCLIDES_TO_CHECK = ["H3", "C14", "Sr90", "I129"]
# 明确定义阈值，与原模型脚本 (mission1.py) 中的值保持一致，用于报告说明
C_THRESHOLD = 0.1


def check_data_existence():
    print("--- NC 文件数据存在性检查开始 ---")

    # 定义表格头部
    header = ["核素", "文件状态", "C > 0 网格数 (最终)", "Arrival Time > 0 网格数"]
    results_table = []

    for nuclide in NUCLIDES_TO_CHECK:
        nuclide_data = [nuclide]
        # 构造输出文件名
        nc_file = os.path.join(OUTPUT_DIR, nuclide, f"{nuclide}.nc")

        if not os.path.exists(nc_file):
            nuclide_data.extend(["❌ 缺失", "N/A", "N/A"])
            results_table.append(nuclide_data)
            continue

        nuclide_data.append("✅ 存在")

        try:
            # 1. 加载数据
            ds = xr.open_dataset(nc_file)

            # 2. 检查浓度数据 (concentration)
            # 检查最终时间步 C > 0 的网格数
            concentration = ds['concentration'].values
            final_c = concentration[-1, ...]
            # 使用一个极小值作为判断标准，确保排除浮点数误差带来的0
            active_c_count = np.sum(final_c > 1e-12)

            # 3. 检查到达时间数据 (arrival_time)
            arrival_time = ds['arrival_time'].values

            # 找到非NaN（即已记录时间）的网格点数量
            arrival_count = np.sum(~np.isnan(arrival_time))

            nuclide_data.append(f"{active_c_count:,}")
            nuclide_data.append(f"{arrival_count:,}")

        except Exception as e:
            nuclide_data.extend([f"⚠️ 加载错误: {e}", "N/A"])

        results_table.append(nuclide_data)

    # 打印最终表格
    print("\n" + "=" * 70)
    print("🌊 核素数据存在性及扩散范围报告 🌊")
    print("=" * 70)

    # 确定列宽
    col_widths = [max(len(str(item)) for item in col) for col in zip(*results_table, header)]

    # 打印头部
    print(
        f"{header[0]:<{col_widths[0]}} | {header[1]:<{col_widths[1]}} | {header[2]:<{col_widths[2]}} | {header[3]:<{col_widths[3]}}")
    print("-" * 70)

    # 打印数据行
    for row in results_table:
        print(
            f"{row[0]:<{col_widths[0]}} | {row[1]:<{col_widths[1]}} | {row[2]:<{col_widths[2]}} | {row[3]:<{col_widths[3]}}")

    print("=" * 70)
    print(f"\n💡 C > 0 网格数: 表示在模拟结束时浓度仍大于零的网格点总数。")
    print(f"💡 Arrival Time > 0 网格数: 表示浓度达到 {C_THRESHOLD} Bq/m³ 阈值的网格点总数。")


if __name__ == "__main__":
    check_data_existence()