# 任务一脚本 - 优化版：正确的边界条件 + 向量化性能
import xarray as xr
import numpy as np
import os
from datetime import datetime, timedelta
import logging
import sys
from math import log


# ============================================================================
# 配置（同修正版）
# ============================================================================
class Config:
    RESOLUTION = 1.0  # 默认分辨率为 1.0 度
    INPUT_FILE = "" # 将在 update_paths 中设置
    BASE_OUTPUT_DIR = "outputs/mission1/" # 基础输出目录
    OUTPUT_DIR = "" # 将在 update_paths 中设置
    OUTPUT_NC = ""
    DATA_LOG = ""
    STATUS_LOG = ""
    CHECKPOINT_DIR = ""

    D_H = 500.0
    D_V = 1e-4
    LAMBDA_DECAY = 0.0
    Q_RATE = 0.0
    Q_BQ = 0.0

    SOURCE_LON = 141.03
    SOURCE_LAT = 37.42
    SOURCE_DEPTH = 0.0

    START_DATE = "2023-08-24"
    DT = 3600.0 * 6
    N_YEARS = 10
    N_STEPS = int(365 * 24 / 6 * N_YEARS)
    SAVE_INTERVAL = 7 * 4
    LOG_INTERVAL = 7 * 4
    CHECKPOINT_INTERVAL = 365 * 4

    C_THRESHOLD = 0.1

    NUCLIDES_SIMULATION = {
        "H3": {"T12": 12.3, "Q_rate": 100.0, "activity_per_ton": 3.7e14},
        "C14": {"T12": 5730.0, "Q_rate": 50.0, "activity_per_ton": 1.6e11},
        "Sr90": {"T12": 28.8, "Q_rate": 200.0, "activity_per_ton": 5.1e15},
        "I129": {"T12": 1.57e7, "Q_rate": 30.0, "activity_per_ton": 6.5e9}
    }

    @staticmethod
    def get_lambda_decay(T12_years):
        SECONDS_PER_YEAR = 365.25 * 24 * 3600.0
        return log(2) / (T12_years * SECONDS_PER_YEAR)

    @classmethod
    def get_res_string(cls):
        """根据分辨率生成对应的字符串，例如 1.0 -> '1', 0.5 -> '0p5'"""
        if cls.RESOLUTION == 1.0:
            return "1"
        else:
            return str(cls.RESOLUTION).replace('.', 'p')

    @classmethod
    def update_paths(cls, nuclide_name):
        """根据分辨率和核素名更新所有文件路径"""
        res_str_for_filename = cls.get_res_string()
        
        cls.INPUT_FILE = os.path.join("raw_data", "output", f"model_input_{res_str_for_filename}deg.nc")
        # 输出路径不变，覆盖
        cls.OUTPUT_DIR = os.path.join(cls.BASE_OUTPUT_DIR, nuclide_name)
        
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True) # 确保输出目录存在

        cls.OUTPUT_NC = os.path.join(cls.OUTPUT_DIR, f"{nuclide_name}.nc")
        cls.DATA_LOG = os.path.join(cls.OUTPUT_DIR, f"data_log_{nuclide_name}.txt")
        cls.STATUS_LOG = os.path.join(cls.OUTPUT_DIR, f"status_log_{nuclide_name}.txt")
        cls.CHECKPOINT_DIR = os.path.join(cls.OUTPUT_DIR, f"checkpoints_{nuclide_name}")


def setup_logging(nuclide_name):
    # 确保在设置日志前更新路径
    Config.update_paths(nuclide_name)

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    data_logger = logging.getLogger('data_log')
    data_logger.setLevel(logging.INFO)
    data_logger.handlers.clear()
    data_handler = logging.FileHandler(Config.DATA_LOG, mode='w', encoding='utf-8')
    data_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    data_logger.addHandler(data_handler)

    status_logger = logging.getLogger('status_log')
    status_logger.setLevel(logging.INFO)
    status_logger.handlers.clear()
    status_handler = logging.FileHandler(Config.STATUS_LOG, mode='w', encoding='utf-8')
    status_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s'))
    status_logger.addHandler(status_handler)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    status_logger.addHandler(console)

    return data_logger, status_logger


# ============================================================================
# 优化版求解器：向量化 + 正确边界条件
# ============================================================================
class OptimizedSolver:
    def __init__(self, ds, data_logger, status_logger):
        self.ds = ds
        self.data_log = data_logger
        self.status_log = status_logger

        self.lat = ds.latitude.values
        self.lon = ds.longitude.values
        self.depth = ds.depth.values if 'depth' in ds.coords else np.array([0])

        self.n_lat = len(self.lat)
        self.n_lon = len(self.lon)
        self.n_depth = len(self.depth)
        self.n_time_ocean = len(ds.time)

        # 确保 dx, dy, dz 的计算与网格分辨率无关，而是动态从 ds 中获取
        # dx 随纬度变化
        self.dx = 111320.0 * np.cos(np.radians(self.lat)) * (self.lon[1] - self.lon[0])
        self.dy = 111320.0 * (self.lat[1] - self.lat[0])
        self.dz = np.abs(np.diff(self.depth)).mean() if self.n_depth > 1 else 10.0 # 默认10m

        self.C = np.zeros((self.n_depth, self.n_lat, self.n_lon), dtype=np.float32)

        self.source_i_lon = np.abs(self.lon - Config.SOURCE_LON).argmin()
        self.source_i_lat = np.abs(self.lat - Config.SOURCE_LAT).argmin()
        self.source_i_depth = 0

        self.status_log.info(f"网格: {self.n_depth}×{self.n_lat}×{self.n_lon} (分辨率: {Config.RESOLUTION}°)")
        self.status_log.info(f"初始源点: ({self.lon[self.source_i_lon]:.2f}°E, {self.lat[self.source_i_lat]:.2f}°N)")

        self.arrival_time = np.full((self.n_depth, self.n_lat, self.n_lon), np.nan, dtype=np.float32)

        # 预加载速度场
        self.status_log.info("预加载速度场...")
        self.u_monthly = []
        self.v_monthly = []
        self.w_monthly = []

        for t in range(self.n_time_ocean):
            u = self.ds.u.isel(time=t).values if 'u' in self.ds else np.zeros((self.n_depth, self.n_lat, self.n_lon))
            v = self.ds.v.isel(time=t).values if 'v' in self.ds else np.zeros((self.n_depth, self.n_lat, self.n_lon))
            w = self.ds.w.isel(time=t).values if 'w' in self.ds else np.zeros((self.n_depth, self.n_lat, self.n_lon))

            self.u_monthly.append(np.nan_to_num(u, nan=0.0).astype(np.float32))
            self.v_monthly.append(np.nan_to_num(v, nan=0.0).astype(np.float32))
            self.w_monthly.append(np.nan_to_num(w, nan=0.0).astype(np.float32))

        # 陆地掩码
        # land_mask 应该是一个布尔数组，形状与 C 相同
        # 假设 land_mask 是 (depth, lat, lon) 形状
        self.land_mask = (self.u_monthly[0] == 0) & (self.v_monthly[0] == 0)
        self.status_log.info(f"陆地网格数: {np.sum(self.land_mask)}")

        # 检查初始源点是否在陆地上
        if self.land_mask[self.source_i_depth, self.source_i_lat, self.source_i_lon]:
            self.status_log.warning(
                f"⚠️ 警告: 初始源点 ({self.lon[self.source_i_lon]:.2f}°E, {self.lat[self.source_i_lat]:.2f}°N) "
                f"位于陆地掩码内。"
            )
            # 如果是 0.5 度分辨率，尝试寻找最近的海洋点
            if Config.RESOLUTION == 0.5:
                self.status_log.info("尝试在周围寻找最近的海洋网格点作为新的源点...")
                new_source_lat_idx, new_source_lon_idx = self.find_nearest_ocean_point(
                    self.source_i_lat, self.source_i_lon, search_radius=2
                )
                if new_source_lat_idx is not None and new_source_lon_idx is not None:
                    self.source_i_lat = new_source_lat_idx
                    self.source_i_lon = new_source_lon_idx
                    self.status_log.info(
                        f"✅ 新的源点已找到: ({self.lon[self.source_i_lon]:.2f}°E, {self.lat[self.source_i_lat]:.2f}°N)"
                    )
                else:
                    self.status_log.error("❌ 未能在周围找到合适的海洋网格点。将停止核素释放。")
                    Config.Q_BQ = 0 # 停止核素释放
            else:
                self.status_log.error("❌ 源点位于陆地掩码内，且未配置自动寻找海洋点。将停止核素释放。")
                Config.Q_BQ = 0 # 停止核素释放

    def find_nearest_ocean_point(self, start_lat_idx, start_lon_idx, search_radius=2):
        """
        在给定半径内寻找最近的海洋网格点。
        返回 (lat_idx, lon_idx) 或 (None, None)
        """
        min_dist_sq = float('inf')
        nearest_ocean_point = (None, None)

        for d_lat in range(-search_radius, search_radius + 1):
            for d_lon in range(-search_radius, search_radius + 1):
                current_lat_idx = start_lat_idx + d_lat
                current_lon_idx = start_lon_idx + d_lon

                # 检查边界
                if not (0 <= current_lat_idx < self.n_lat and 0 <= current_lon_idx < self.n_lon):
                    continue

                # 检查是否是海洋点
                if not self.land_mask[self.source_i_depth, current_lat_idx, current_lon_idx]:
                    # 计算距离平方 (避免开方，提高效率)
                    dist_sq = d_lat**2 + d_lon**2
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        nearest_ocean_point = (current_lat_idx, current_lon_idx)
        return nearest_ocean_point


    def get_velocity_at_time(self, step):
        """获取速度场（带年际趋势）"""
        time_idx = (step * 6 // 24) % self.n_time_ocean
        current_year = (step * Config.DT) / (365.25 * 86400)
        year_factor = 1.0 - 0.02 * current_year

        return (self.u_monthly[time_idx] * year_factor,
                self.v_monthly[time_idx] * year_factor,
                self.w_monthly[time_idx] * year_factor)

    def add_source_term(self, dt):
        """添加源项"""
        # 检查源点是否在陆地上，如果在陆地上则不添加源项
        if self.land_mask[self.source_i_depth, self.source_i_lat, self.source_i_lon]:
            # 如果 Config.Q_BQ 已经被设置为 0，则不再打印警告
            if Config.Q_BQ != 0:
                self.status_log.warning("源点在陆地上，不添加源项。")
            return

        # 如果 Config.Q_BQ 为 0，也不添加源项
        if Config.Q_BQ == 0:
            return

        # 使用实际的网格单元体积
        cell_volume = self.dx[self.source_i_lat] * self.dy * self.dz
        source_strength = Config.Q_BQ / cell_volume
        self.C[self.source_i_depth, self.source_i_lat, self.source_i_lon] += source_strength * dt

    def compute_advection_vectorized(self, C, u, v, w):
        """
        向量化平流项计算 - 使用掩码避免循环回绕
        关键：不使用 np.roll()，改用 np.pad() + 切片
        """
        # 对数组进行零填充（模拟开放边界）
        C_padded = np.pad(C, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)

        # X方向（经度）- 上风格式
        u_pos = np.maximum(u, 0)  # 正速度
        u_neg = np.minimum(u, 0)  # 负速度

        # 向量化单侧差分
        # 注意：这里 self.dx 是一个数组，需要广播
        dC_dx_backward = (C_padded[1:-1, 1:-1, 1:-1] - C_padded[1:-1, 1:-1, :-2]) / self.dx[:, np.newaxis]
        dC_dx_forward = (C_padded[1:-1, 1:-1, 2:] - C_padded[1:-1, 1:-1, 1:-1]) / self.dx[:, np.newaxis]

        advection_x = u_pos * dC_dx_backward + u_neg * dC_dx_forward

        # Y方向（纬度）
        v_pos = np.maximum(v, 0)
        v_neg = np.minimum(v, 0)

        dC_dy_backward = (C_padded[1:-1, 1:-1, 1:-1] - C_padded[1:-1, :-2, 1:-1]) / self.dy
        dC_dy_forward = (C_padded[1:-1, 2:, 1:-1] - C_padded[1:-1, 1:-1, 1:-1]) / self.dy

        advection_y = v_pos * dC_dy_backward + v_neg * dC_dy_forward

        # Z方向（深度）
        advection_z = np.zeros_like(C)
        if C.shape[0] > 2:
            w_pos = np.maximum(w, 0)
            w_neg = np.minimum(w, 0)

            dC_dz_backward = (C_padded[1:-1, 1:-1, 1:-1] - C_padded[:-2, 1:-1, 1:-1]) / self.dz
            dC_dz_forward = (C_padded[2:, 1:-1, 1:-1] - C_padded[1:-1, 1:-1, 1:-1]) / self.dz

            advection_z = w_pos * dC_dz_backward + w_neg * dC_dz_forward

        return advection_x, advection_y, advection_z

    def compute_diffusion_vectorized(self, C):
        """向量化扩散项计算"""
        # 填充（Neumann边界条件：边界值=邻近内部值）
        C_padded = np.pad(C, ((1, 1), (1, 1), (1, 1)), mode='edge')

        # 水平扩散（X + Y）
        # 注意：这里 self.dx 是一个数组，需要广播
        d2C_dx2 = (C_padded[1:-1, 1:-1, 2:] - 2 * C_padded[1:-1, 1:-1, 1:-1] +
                   C_padded[1:-1, 1:-1, :-2]) / (self.dx[:, np.newaxis] ** 2)

        d2C_dy2 = (C_padded[1:-1, 2:, 1:-1] - 2 * C_padded[1:-1, 1:-1, 1:-1] +
                   C_padded[1:-1, :-2, 1:-1]) / (self.dy ** 2)

        diffusion_h = Config.D_H * (d2C_dx2 + d2C_dy2)

        # 垂直扩散
        diffusion_v = np.zeros_like(C)
        if C.shape[0] > 2:
            d2C_dz2 = (C_padded[2:, 1:-1, 1:-1] - 2 * C_padded[1:-1, 1:-1, 1:-1] +
                       C_padded[:-2, 1:-1, 1:-1]) / (self.dz ** 2)
            diffusion_v = Config.D_V * d2C_dz2

        return diffusion_h, diffusion_v

    def solve_step_optimized(self, step, current_time):
        """优化版时间步进：向量化 + 正确边界"""
        dt = Config.DT

        # 1. 首先清零陆地上的浓度 (重要: 在添加源项之前)
        self.C[self.land_mask] = 0

        # 2. 添加源项 (如果源点不在陆地上)
        self.add_source_term(dt)

        # 3. 获取速度场
        u, v, w = self.get_velocity_at_time(step)

        # 4. 计算平流项（向量化，无循环回绕）
        adv_x = np.zeros_like(self.C)
        adv_y = np.zeros_like(self.C)
        adv_z = np.zeros_like(self.C)
        
        # 只有当C的维度大于1时才计算平流项，避免维度错误
        if self.C.shape[0] > 0 and self.C.shape[1] > 0 and self.C.shape[2] > 0:
            adv_x, adv_y, adv_z = self.compute_advection_vectorized(self.C, u, v, w)

        # 5. 计算扩散项（向量化）
        diff_h = np.zeros_like(self.C)
        diff_v = np.zeros_like(self.C)
        if self.C.shape[0] > 0 and self.C.shape[1] > 0 and self.C.shape[2] > 0:
            diff_h, diff_v = self.compute_diffusion_vectorized(self.C)

        # 6. 衰变项
        decay = Config.LAMBDA_DECAY * self.C

        # 7. 更新浓度
        dC_dt = -adv_x - adv_y - adv_z + diff_h + diff_v - decay
        self.C = np.maximum(0, self.C + dC_dt * dt)

        # 8. 海面和海底边界（Neumann）
        if self.n_depth > 1:
            self.C[0] = self.C[1]
            self.C[-1] = self.C[-2]

        # 9. 记录到达时间
        mask = (np.isnan(self.arrival_time)) & (self.C >= Config.C_THRESHOLD)
        if np.any(mask):
            self.arrival_time[mask] = current_time

        # 10. 统计
        max_conc = np.max(self.C)
        mean_conc = np.mean(self.C[self.C > 0]) if np.any(self.C > 0) else 0.0
        total_mass = np.sum(self.C) * self.dx.mean() * self.dy * self.dz

        return max_conc, mean_conc, total_mass

    def save_checkpoint(self, year, current_dt):
        """保存检查点"""
        checkpoint_file = os.path.join(Config.CHECKPOINT_DIR, f"checkpoint_year_{year}.npz")
        np.savez_compressed(checkpoint_file, C=self.C, arrival_time=self.arrival_time,
                            year=year, date=current_dt.strftime('%Y-%m-%d'))
        self.status_log.info(f"检查点已保存: 第{year}年")


# ============================================================================
# 主程序
# ============================================================================
def main():
    if len(sys.argv) < 2:
        print("用法: python mission1.py <核素名> [分辨率(例如: 0.5, 1.0)]")
        print(f"可选核素: {list(Config.NUCLIDES_SIMULATION.keys())}")
        sys.exit(1)

    nuclide_name = sys.argv[1]
    
    if len(sys.argv) > 2:
        try:
            Config.RESOLUTION = float(sys.argv[2])
            if Config.RESOLUTION <= 0:
                raise ValueError("分辨率必须是正数")
        except ValueError as e:
            print(f"错误: 无效的分辨率参数. {e}")
            sys.exit(1)

    if nuclide_name not in Config.NUCLIDES_SIMULATION:
        print(f"错误: 未知核素 '{nuclide_name}'")
        sys.exit(1)

    params = Config.NUCLIDES_SIMULATION[nuclide_name]

    Config.LAMBDA_DECAY = Config.get_lambda_decay(params["T12"])
    Q_ton_per_day = params["Q_rate"]
    activity_per_ton = params["activity_per_ton"]
    Config.Q_BQ = (Q_ton_per_day * activity_per_ton) / 86400

    data_log, status_log = setup_logging(nuclide_name) # 传入核素名以更新路径

    status_log.info("=" * 60)
    status_log.info(f"核废水扩散模型 - 优化版 - 核素: {nuclide_name}")
    status_log.info(f"分辨率: {Config.RESOLUTION}°")
    status_log.info("=" * 60)
    status_log.info(f"T1/2={params['T12']:.1f}年, λ={Config.LAMBDA_DECAY:.2e} 1/s")
    status_log.info(f"源项: {Q_ton_per_day:.1f}吨/天 = {Config.Q_BQ:.2e} Bq/s")
    status_log.info(f"输入文件: {Config.INPUT_FILE}")


    try:
        ds = xr.open_dataset(Config.INPUT_FILE)
        status_log.info(f"数据维度: {dict(ds.sizes)}")
    except Exception as e:
        status_log.error(f"加载失败: {Config.INPUT_FILE}. 请确保该文件已生成。错误: {e}")
        sys.exit(1)

    status_log.info("初始化优化求解器...")
    solver = OptimizedSolver(ds, data_log, status_log)

    n_saves = Config.N_STEPS // Config.SAVE_INTERVAL + 1
    C_history = np.zeros((n_saves, solver.n_depth, solver.n_lat, solver.n_lon), dtype=np.float32)
    time_coords = []

    status_log.info("开始计算...")
    start_dt = datetime.strptime(Config.START_DATE, "%Y-%m-%d")
    import time
    start_time = time.time()

    for step in range(Config.N_STEPS):
        current_time = step * Config.DT / 86400.0
        current_dt = start_dt + timedelta(seconds=step * Config.DT)
        current_year = int(current_time / 365) + 1

        max_c, mean_c, mass = solver.solve_step_optimized(step, current_time)

        if step % Config.SAVE_INTERVAL == 0:
            save_idx = step // Config.SAVE_INTERVAL
            C_history[save_idx] = solver.C
            time_coords.append(current_dt)

        if step % Config.LOG_INTERVAL == 0:
            progress = (step / Config.N_STEPS) * 100
            elapsed = time.time() - start_time
            eta = elapsed / max(step, 1) * (Config.N_STEPS - step)
            source_conc = solver.C[solver.source_i_depth, solver.source_i_lat, solver.source_i_lon]

            status_log.info(
                f"进度: {progress:.1f}% | {nuclide_name} | "
                f"最大: {max_c:.2e} Bq/m³ | 源点: {source_conc:.2e} Bq/m³ | "
                f"ETA: {eta / 60:.1f}分钟")

            data_log.info(f"step={step}, max={max_c:.6e}, mean={mean_c:.6e}, mass={mass:.6e}")

        if step > 0 and step % Config.CHECKPOINT_INTERVAL == 0:
            solver.save_checkpoint(current_year, current_dt)

    status_log.info("保存结果...")

    output_ds = xr.Dataset(
        {
            'concentration': (['time', 'depth', 'latitude', 'longitude'], C_history),
            'arrival_time': (['depth', 'latitude', 'longitude'], solver.arrival_time),
        },
        coords={
            'time': time_coords,
            'depth': solver.depth,
            'latitude': solver.lat,
            'longitude': solver.lon,
        },
        attrs={
            'title': f'核废水扩散模拟 - {nuclide_name} (优化版)',
            'method': '有限体积法 + 向量化 + 正确边界条件',
            'nuclide': nuclide_name,
            'T_half_years': params['T12'],
            'source_Bq_per_s': Config.Q_BQ,
            'resolution_deg': Config.RESOLUTION, # 添加分辨率属性
        }
    )

    output_ds.to_netcdf(Config.OUTPUT_NC)
    total_time = time.time() - start_time

    status_log.info("=" * 60)
    status_log.info(f"{nuclide_name} 模拟完成!")
    status_log.info(f"  分辨率: {Config.RESOLUTION}°")
    status_log.info(f"  耗时: {total_time / 60:.1f}分钟 ({total_time / 3600:.2f}小时)")
    status_log.info(f"  最大浓度: {np.max(solver.C):.2e} Bq/m³")
    status_log.info(f"  到达网格数: {np.sum(~np.isnan(solver.arrival_time))}")
    status_log.info(f"  输出: {Config.OUTPUT_NC}")
    status_log.info("=" * 60)


if __name__ == "__main__":
    main()