# utils/generate_E_t_timeseries.py
"""
计算并保存三个方案的环境影响时间序列 E(t)
用于任务三图8：环境影响随时间变化曲线
"""
import xarray as xr
import numpy as np
import pandas as pd
import os
import sys

# 添加父目录到路径（如果需要导入任务三的配置）
sys.path.append('..')

# 配置
NUCLIDES = ['H3', 'C14', 'Sr90', 'I129']
OUTPUT_DIR = "../outputs/mission1"
MISSION3_OUTPUT = "../outputs/mission3"
YEARS = 30

# 核素参数（与任务三一致）
NUCLIDES_PARAMS = {
    "H3": {"eta": {1: 0.10, 2: 0.20, 3: 0.30}},
    "C14": {"eta": {1: 0.90, 2: 0.95, 3: 0.999}},
    "Sr90": {"eta": {1: 0.85, 2: 0.95, 3: 0.995}},
    "I129": {"eta": {1: 0.90, 2: 0.995, 3: 0.9999}}
}

# 网格权重
WEIGHT_COAST = 3
WEIGHT_FISHERY = 5
WEIGHT_OCEAN = 1


def generate_weight_map(lat, lon):
    """生成权重地图 W(x,y)"""
    W = np.ones((len(lat), len(lon))) * WEIGHT_OCEAN

    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            # 日本渔场区域
            if 30 <= la <= 45 and 130 <= lo <= 150:
                W[i, j] = WEIGHT_FISHERY
            # 近岸区域
            elif 20 <= la <= 50 and lo < 140:
                W[i, j] = WEIGHT_COAST

    return W


def compute_E_t_for_scheme(scheme, C_dict, V, W, years=30):
    """计算单个方案的E(t)时间序列"""
    print(f"\n计算方案 {scheme} 的 E(t)...")

    E_t = []

    for t in range(years):
        if t % 5 == 0:
            print(f"  处理时间步: {t}/{years}")

        E_sum = 0.0

        for nuclide_name, C_nuclide in C_dict.items():
            # 获取去除率
            eta = NUCLIDES_PARAMS[nuclide_name]["eta"][scheme]

            # 获取该时间步的浓度
            T0 = C_nuclide.shape[0]
            if t < T0:
                C_t = C_nuclide[t]  # shape: (depth, lat, lon)
            else:
                C_t = C_nuclide[-1]  # 使用最后时刻数据

            # 应用处理效率
            C_treated = C_t * (1 - eta)

            # 对深度求和
            C_depth_sum = C_treated.sum(axis=0)  # shape: (lat, lon)

            # 计算环境影响：E = Σ C * W * V
            E_nuclide = np.sum(C_depth_sum * W * V)
            E_sum += E_nuclide

        E_t.append(E_sum)

    print(f"  完成！E_总和 = {np.sum(E_t):.4e}")
    return E_t


def load_concentration_data():
    """加载所有核素的浓度数据"""
    print("\n加载核素浓度数据...")
    C_dict = {}
    V = None
    lat = None
    lon = None

    for nuclide in NUCLIDES:
        file_path = os.path.join(OUTPUT_DIR, nuclide, f"{nuclide}.nc")

        if not os.path.exists(file_path):
            print(f"  [警告] 文件不存在，跳过: {file_path}")
            continue

        try:
            ds = xr.open_dataset(file_path)
            C_nuclide = ds["concentration"].values
            C_dict[nuclide] = C_nuclide
            print(f"  成功加载 {nuclide}: 形状 {C_nuclide.shape}")

            # 只在第一次计算网格参数
            if V is None:
                depth = ds.depth.values
                lat = ds.latitude.values
                lon = ds.longitude.values

                dz = np.abs(np.diff(depth)).mean() if len(depth) > 1 else 10.0
                dy = 111320.0
                dx = 111320.0 * np.cos(np.radians(lat))
                V = dz * dy * dx[:, np.newaxis]  # shape (lat, lon)

            ds.close()

        except Exception as e:
            print(f"  [错误] 加载失败 {file_path}: {e}")

    return C_dict, V, lat, lon


def main():
    print("=" * 60)
    print("生成三个方案的环境影响时间序列 E(t)")
    print("=" * 60)

    # 加载数据
    C_dict, V, lat, lon = load_concentration_data()

    if not C_dict or V is None:
        print("\n[错误] 数据加载失败，终止程序")
        return

    # 生成权重地图
    print("\n生成权重地图...")
    W = generate_weight_map(lat, lon)

    # 计算三个方案的E(t)
    results = {}
    for scheme in [1, 2, 3]:
        E_t = compute_E_t_for_scheme(scheme, C_dict, V, W, YEARS)
        results[f'Scheme_{scheme}'] = E_t

    # 合并为DataFrame
    df = pd.DataFrame({
        'Year': range(1, YEARS + 1),
        **results
    })

    # 保存
    os.makedirs(MISSION3_OUTPUT, exist_ok=True)
    output_csv = os.path.join(MISSION3_OUTPUT, "E_t_timeseries.csv")
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print(f"已保存到: {output_csv}")
    print("\n数据预览:")
    print(df.head(10).to_string(index=False))
    print("...")
    print(df.tail(5).to_string(index=False))

    # 统计信息
    print("\n30年累计环境影响 E^30:")
    for scheme in [1, 2, 3]:
        col = f'Scheme_{scheme}'
        E30 = df[col].sum()
        print(f"  方案 {scheme}: {E30:.4e}")

    # MATLAB绘图提示
    print("\n" + "=" * 60)
    print("MATLAB 绘图代码:")
    print("=" * 60)
    print("""
data = readtable('../outputs/mission3/E_t_timeseries.csv');
figure;
hold on;
plot(data.Year, data.Scheme_1, '-o', 'LineWidth', 2, 'DisplayName', '方案1');
plot(data.Year, data.Scheme_2, '-s', 'LineWidth', 2, 'DisplayName', '方案2');
plot(data.Year, data.Scheme_3, '-^', 'LineWidth', 2, 'DisplayName', '方案3');
xlabel('时间 (年)');
ylabel('环境影响 E(t)');
title('三种方案环境影响随时间变化');
legend('Location', 'best');
grid on;
    """)
    print("=" * 60)


if __name__ == "__main__":
    main()