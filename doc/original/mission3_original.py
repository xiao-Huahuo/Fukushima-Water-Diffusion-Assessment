# -*- coding: utf-8 -*-
"""
任务三：三维评估模型 + 多目标优化（NSGA-II）- 修正版
--------------------------------------------

输入：
    - outputs/[Nuclide]/[Nuclide].nc (任务一输出的四个核素浓度场)
    - 任务三.md 中给定的核素参数、半衰期、阈值、去除率、成本参数等

输出：
    outputs/mission3/：
        ├── nsga2_results.csv             # 30 年最优方案帕累托前沿
        ├── environment_impact.csv        # 环境影响结果
        ├── mission3_log.txt              # 详细日志
"""
import sys
import numpy as np
import xarray as xr
import pandas as pd
import os
import logging
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.variable import Integer


# ==============================================================
# 配置
# ==============================================================

class Config:
    OUTPUT_DIR = "outputs/mission3/"
    INPUT_DIR= "outputs/mission1/"
    # 任务一输出文件路径格式
    NUCLIDE_PATHS = {
        "H3": os.path.join(INPUT_DIR, "H3", "H3.nc"),
        "C14": os.path.join(INPUT_DIR, "C14", "C14.nc"),
        "Sr90": os.path.join(INPUT_DIR, "Sr90", "Sr90.nc"),
        "I129": os.path.join(INPUT_DIR, "I129", "I129.nc")
    }

    # 年度时间步
    YEARS = 30
    DT = 1.0

    # 网格权重 W(x,y,z)
    WEIGHT_COAST = 3
    WEIGHT_FISHERY = 5
    WEIGHT_OCEAN = 1

    # 成本模型（单位：元）
    OP_COST = 5.5e8  # 运行维护费：5.5 亿/年
    BUILD_COST = {
        1: 1.2e9,  # 方案1：基础处理
        2: 2.0e9,  # 方案2：强化处理
        3: 8.0e9  # 方案3：零排放（更高的建造成本）
    }
    # 方案3的额外成本
    STORAGE_COST_ANNUAL = 2.0e8  # 每年2亿存储成本
    STORAGE_COST_INITIAL = 5.0e9  # 初始存储设施50亿

    # 核素参数（按问题三要求）
    # A0是初始排放浓度（Bq/L）- 根据实际福岛排放数据调整
    # 注意：这里的A0应该是经过ALPS初步处理后的浓度
    NUCLIDES = {
        "H3": {
            "A0": 5000.0,  # 氚初始浓度较高（ALPS无法去除氚）
            "T12": 12.3,  # 半衰期（年）
            "thr": 1000,  # 阈值（Bq/L）
            "eta": {  # 不同方案的去除率
                1: 0.10,  # 方案1：基础去除10%
                2: 0.20,  # 方案2：中等去除20%
                3: 0.30  # 方案3：超净化30%（按题目）
            }
        },
        "C14": {
            "A0": 0.15,  # C14初始浓度降低（ALPS已处理大部分）
            "T12": 5730,
            "thr": 0.1,
            "eta": {
                1: 0.90,  # 方案1：90% → 处理后0.015，已达标
                2: 0.95,  # 方案2：95%
                3: 0.999  # 方案3：99.9%（按题目）
            }
        },
        "Sr90": {
            "A0": 100.0,  # Sr90初始浓度（ALPS处理后仍有残留）
            "T12": 28.8,
            "thr": 10,
            "eta": {
                1: 0.85,  # 方案1：85%
                2: 0.95,  # 方案2：95%
                3: 0.995  # 方案3：99.5%（按题目）
            }
        },
        "I129": {
            "A0": 0.02,  # I129初始浓度极低（接近阈值0.01）
            "T12": 1.57e7,  # 极长半衰期
            "thr": 0.01,
            "eta": {
                1: 0.90,  # 方案1：90%（处理后0.002，已达标）
                2: 0.995,  # 方案2：99.5%
                3: 0.9999  # 方案3：99.99%（按题目）
            }
        }
    }


# ==============================================================
# 日志系统
# ==============================================================

def setup_logging():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    logger = logging.getLogger("mission3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 清除已有handlers

    fh = logging.FileHandler(os.path.join(Config.OUTPUT_DIR, "mission3_log.txt"),
                             mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(ch)

    return logger


# ==============================================================
# 数据加载
# ==============================================================

def load_concentration_data(logger):
    """
    加载所有四种核素的浓度场
    返回：字典 {nuclide_name: concentration_array}, V, lat, lon
    """
    logger.info("开始加载任务一输出的四种核素浓度场...")

    C_dict = {}
    V = None
    lat = None
    lon = None

    for nuclide_name, file_path in Config.NUCLIDE_PATHS.items():
        try:
            if not os.path.exists(file_path):
                logger.warning(f"  文件不存在: {file_path}，跳过该核素")
                continue

            ds = xr.open_dataset(file_path)
            C_nuclide = ds["concentration"].values  # shape = (time, depth, lat, lon)
            C_dict[nuclide_name] = C_nuclide
            logger.info(f"  成功加载 {nuclide_name}: 数据形状 {C_nuclide.shape}")

            if V is None:
                # 仅在第一次加载时计算网格参数
                depth = ds.depth.values
                lat = ds.latitude.values
                lon = ds.longitude.values

                dz = np.abs(np.diff(depth)).mean() if len(depth) > 1 else 10.0
                dy = 111320.0  # 纬度1度约111.32km
                dx = 111320.0 * np.cos(np.radians(lat))  # 经度随纬度变化
                V = dz * dy * dx[:, np.newaxis]  # shape (lat, lon)

            ds.close()

        except Exception as e:
            logger.error(f"  加载文件失败: {file_path}. 错误: {e}")

    if not C_dict:
        logger.error("未能加载任何核素数据，程序终止")
        return None, None, None, None

    logger.info(f"浓度场加载完成，共加载 {len(C_dict)} 种核素")
    return C_dict, V, lat, lon


def generate_weight_map(lat, lon):
    """
    生成权重地图 W(x,y)
    """
    W = np.ones((len(lat), len(lon))) * Config.WEIGHT_OCEAN

    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            # 日本渔场区域（示例）
            if 30 <= la <= 45 and 130 <= lo <= 150:
                W[i, j] = Config.WEIGHT_FISHERY
            # 近岸区域（示例）
            elif 20 <= la <= 50 and lo < 140:
                W[i, j] = Config.WEIGHT_COAST

    return W


# ==============================================================
# 核心计算函数
# ==============================================================

def compute_environment_impact(C_dict, scheme, V, W, years=30):
    """
    计算方案j在30年内的环境影响
    E_j(t) = Σ C_j,i(x,y,z,t) * W * V

    返回：E30 (30年累计总影响)
    """
    E_t = []

    for t in range(years):
        E_sum = 0.0

        for nuclide_name, C_nuclide in C_dict.items():
            # 获取该核素在方案j下的去除率
            eta = Config.NUCLIDES[nuclide_name]["eta"][scheme]

            # 处理后的浓度：C_j,i = (1 - eta_j,i) * C_i
            T0 = C_nuclide.shape[0]
            if t < T0:
                C_t = C_nuclide[t]  # shape: (depth, lat, lon)
            else:
                C_t = C_nuclide[-1]  # 使用最后时刻数据填充

            # 应用去除率
            C_treated = C_t * (1 - eta)

            # 对深度求和
            C_depth_sum = C_treated.sum(axis=0)  # shape: (lat, lon)

            # 计算环境影响：E = Σ C * W * V
            E_nuclide = np.sum(C_depth_sum * W * V)
            E_sum += E_nuclide

        E_t.append(E_sum)

    E30 = np.sum(E_t) * Config.DT  # 累计30年
    return E30


def compute_cost(scheme):
    """
    计算方案j的总成本

    方案1、2：建造成本 + 30年运行维护成本
    方案3：建造成本 + 初始存储设施 + 30年存储运维成本
    """
    if scheme == 3:
        # 零排放：高建造成本 + 初始存储设施 + 持续存储成本
        return (Config.BUILD_COST[3] +
                Config.STORAGE_COST_INITIAL +
                Config.STORAGE_COST_ANNUAL * Config.YEARS)
    else:
        # 方案1、2：建造 + 30年运行维护
        return Config.BUILD_COST[scheme] + Config.OP_COST * Config.YEARS


def compute_decay_time(nuclide_name, scheme):
    """
    计算核素i在方案j下的达标时间 t'_i

    按照问题描述：
    1. 排放后经过处理：活度降为 A_0 * (1 - η)
    2. 然后随时间衰变：A(t) = A_0 * (1 - η) * exp(-λt)
    3. 求解 A(t) = C_thr 的时间 t'
    """
    params = Config.NUCLIDES[nuclide_name]
    A0 = params["A0"]
    T12 = params["T12"]
    thr = params["thr"]
    eta = params["eta"][scheme]

    lam = np.log(2) / T12

    # 处理后的初始活度
    A_treated = A0 * (1 - eta)

    # 如果处理后就已经达标，立即达标
    if A_treated <= thr:
        return 0.0

    # 否则需要等待衰变到阈值以下
    # A_treated * exp(-λt) = thr
    # t = ln(A_treated / thr) / λ
    try:
        t_prime = np.log(A_treated / thr) / lam

        # 对于I129这种超长半衰期的核素，如果达标时间超长，
        # 说明该方案对该核素效果不佳
        if t_prime > 1e6:  # 超过100万年
            return 1e6  # 返回上限值

        return max(0.0, t_prime)
    except:
        return 1e6  # 计算异常时返回大值


def compute_max_decay_time(scheme):
    """计算方案j对所有核素的最大达标时间"""
    t_list = []
    for nuclide_name in Config.NUCLIDES.keys():
        t = compute_decay_time(nuclide_name, scheme)
        t_list.append(t)
    return max(t_list)


# ==============================================================
# NSGA-II 多目标优化问题定义
# ==============================================================

class MultiObjectiveProblem(ElementwiseProblem):
    """
    三目标优化问题：
    min f1 = 环境影响 E_j^30
    min f2 = 总成本 C_j
    min f3 = 最大达标时间 t'_j

    决策变量：方案编号 j ∈ {1, 2, 3}
    """

    def __init__(self, C_dict, V, W, logger):
        super().__init__(
            n_var=1,
            n_obj=3,
            n_ieq_constr=0,
            xl=np.array([1]),
            xu=np.array([3]),
            vtype=int
        )
        self.C_dict = C_dict
        self.V = V
        self.W = W
        self.logger = logger
        self.eval_count = 0

    def _evaluate(self, x, out, *args, **kwargs):
        scheme = int(x[0])  # 方案编号 1, 2, 或 3

        # 计算三个目标
        f1 = compute_environment_impact(self.C_dict, scheme, self.V, self.W)
        f2 = compute_cost(scheme)
        f3 = compute_max_decay_time(scheme)

        self.eval_count += 1
        if self.eval_count % 10 == 0:
            self.logger.info(f"  评估 #{self.eval_count}: 方案{scheme} -> E={f1:.2e}, C={f2:.2e}, t'={f3:.1f}年")

        out["F"] = [f1, f2, f3]


# ==============================================================
# 主程序
# ==============================================================

def main():
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("任务三：三维评估模型 + NSGA-II 多目标优化")
    logger.info("=" * 60)

    # 1. 加载浓度数据
    C_dict, V, lat, lon = load_concentration_data(logger)

    if C_dict is None or V is None:
        logger.error("数据加载失败，程序终止")
        return

    # 2. 生成权重地图
    logger.info("生成环境权重地图 W(x,y)...")
    W = generate_weight_map(lat, lon)
    logger.info(f"  权重统计: 渔场={Config.WEIGHT_FISHERY}, 近岸={Config.WEIGHT_COAST}, 公海={Config.WEIGHT_OCEAN}")

    # 3. 先计算每个方案的单独结果（用于验证）
    logger.info("\n" + "=" * 60)
    logger.info("单方案评估（验证）:")
    logger.info("=" * 60)

    for scheme in [1, 2, 3]:
        logger.info(f"\n方案 {scheme}:")
        E = compute_environment_impact(C_dict, scheme, V, W)
        C = compute_cost(scheme)
        t = compute_max_decay_time(scheme)
        logger.info(f"  环境影响 E^30 = {E:.4e}")
        logger.info(f"  总成本 C = {C:.4e}")
        logger.info(f"  最大达标时间 t' = {t:.2f} 年")

    # 4. 建立并运行NSGA-II优化
    logger.info("\n" + "=" * 60)
    logger.info("开始 NSGA-II 多目标优化...")
    logger.info("=" * 60)

    problem = MultiObjectiveProblem(C_dict, V, W, logger)

    # 创建初始种群：确保每个方案都有足够代表
    initial_pop = []
    for scheme in [1, 2, 3]:
        for _ in range(10):  # 每个方案重复10次
            initial_pop.append([scheme])
    initial_pop = np.array(initial_pop)

    algorithm = NSGA2(
        pop_size=30,
        sampling=initial_pop,  # 使用预设的初始种群
        eliminate_duplicates=False  # 允许相同方案（因为只有3个离散值）
    )

    termination = get_termination("n_gen", 50)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=False
    )

    # 5. 处理和保存结果
    logger.info("\n" + "=" * 60)
    logger.info("保存优化结果...")
    logger.info("=" * 60)

    X = res.X.flatten() if res.X.ndim > 1 else res.X
    F = res.F

    # 创建结果DataFrame
    results = []
    for i in range(len(X)):
        results.append({
            "方案": int(X[i]),
            "环境影响E30": F[i, 0],
            "成本C": F[i, 1],
            "达标时间t'": F[i, 2]
        })

    df = pd.DataFrame(results)

    # 去重：对于离散方案，每个方案只保留一个结果
    df = df.drop_duplicates(subset=["方案"])
    df = df.sort_values("方案")  # 按方案编号排序

    logger.info(f"\n优化结果去重后保留 {len(df)} 个唯一方案")

    output_file = os.path.join(Config.OUTPUT_DIR, "nsga2_results.csv")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logger.info(f"  帕累托前沿解已保存: {output_file}")

    # 显示结果摘要
    logger.info("\n帕累托前沿解集:")
    logger.info(df.to_string(index=False))

    logger.info("\n" + "=" * 60)
    logger.info("任务三完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()