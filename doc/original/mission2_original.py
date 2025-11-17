import xarray as xr
import numpy as np
import pandas as pd
import os
from datetime import datetime
import logging
import json
from sklearn.cluster import KMeans
import sys


# ============================================================================
# 配置部分 - 已更新为动态输入数据
# ============================================================================
class Config:
    # 文件路径
    INPUT_DIR="outputs/mission1/"
    OUTPUT_DIR = "outputs/mission2/"
    INPUT_NC=''
    # 输出文件
    RESULTS_EXCEL = os.path.join(OUTPUT_DIR, "risk_assessment_results.xlsx")
    RAW_DATA_CSV = os.path.join(OUTPUT_DIR, "raw_indicators.csv")
    NORMALIZED_CSV = os.path.join(OUTPUT_DIR, "normalized_indicators.csv")
    WEIGHTS_CSV = os.path.join(OUTPUT_DIR, "weights_calculation.csv")
    CLUSTERING_CSV = os.path.join(OUTPUT_DIR, "clustering_results.csv")
    SUMMARY_JSON = os.path.join(OUTPUT_DIR, "summary_report.json")
    LOG_FILE = os.path.join(OUTPUT_DIR, "mission2_log.txt")

    # 核心常数
    K_BIOCONCENTRATION = 5e4  # 生物富集系数 K = 5 x 10^4

    # 多核素配置：定义需要处理的核素列表及其在风险评估中的相对权重 (W_Risk)
    NUCLIDES_TO_PROCESS = ['H3', 'C14', 'Sr90', 'I129']
    # 严谨风险权重 (W_Risk) - 基于 ICRP 60/119 成人摄入有效剂量系数 (Sv/Bq) 相对 H3 (1.8e-11 Sv/Bq) 的归一化。
    # C_Total-Risk = SUM(C_i * W_i)
    NUCLIDE_RISK_WEIGHTS = {
        'H3': 1.0,  # (1.8e-11 / 1.8e-11)
        'C14': 32.22,  # (5.8e-10 / 1.8e-11)
        'Sr90': 1555.56,  # (2.8e-8 / 1.8e-11)
        'I129': 6111.11  # (1.1e-7 / 1.8e-11)
    }

    # 直接使用文档表格中给定的海域核素浓度（已经是经过任务一模拟的10年平均值）
    CUSTOM_INPUTS = {
        '日本': {'C_avg': 8.2, 'D_adult': 0.20},
        '韩国': {'C_avg': 6.8, 'D_adult': 0.18},
        '美国': {'C_avg': 2.5, 'D_adult': 0.12},
        '加拿大': {'C_avg': 2.1, 'D_adult': 0.10},
        '中国': {'C_avg': 0.8, 'D_adult': 0.07},
        '澳大利亚': {'C_avg': 0.5, 'D_adult': 0.13}
    }

    # 国家的经纬度坐标 (用于从NC文件提取浓度)
    COUNTRIES = {
        '日本': {'lon': 141.03, 'lat': 37.42, 'name_en': 'Japan'},
        '韩国': {'lon': 129.04, 'lat': 35.18, 'name_en': 'South Korea'},
        '美国': {'lon': 237.78, 'lat': 37.77, 'name_en': 'USA'},
        '加拿大': {'lon': 236.64, 'lat': 49.28, 'name_en': 'Canada'},
        '中国': {'lon': 121.47, 'lat': 31.23, 'name_en': 'China'},
        '澳大利亚': {'lon': 151.21, 'lat': -33.87, 'name_en': 'Australia'}
    }

    # 固定的 D_adult 值 (用于计算C1)
    FIXED_D_ADULT = {
        '日本': 0.20,
        '韩国': 0.18,
        '美国': 0.12,
        '加拿大': 0.10,
        '中国': 0.07,
        '澳大利亚': 0.13
    }

    # AHP主观权重（直接使用文档提供的全局权重）
    AHP_WEIGHTS = {
        'A1': 0.232, 'A2': 0.077,
        'B1': 0.075, 'B2': 0.013, 'B3': 0.022,
        'C1': 0.247, 'C2': 0.103, 'C3': 0.159, 'C4': 0.049, 'C5': 0.024
    }


# ============================================================================
# 日志系统
# ============================================================================
def setup_logging():
    """设置日志系统"""
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    logger = logging.getLogger('mission2')
    logger.setLevel(logging.INFO)
    # 文件处理器
    fh = logging.FileHandler(Config.LOG_FILE, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s'))
    logger.addHandler(fh)
    # 控制台处理器
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(ch)
    return logger


# ============================================================================
# 多核素数据预处理模块 (新增)
# ============================================================================
class MultiNuclideProcessor:
    """读取多核素NC文件，计算 C_Total-Risk，并填充 Config.CUSTOM_INPUTS"""

    def __init__(self, logger):
        self.logger = logger
        self.countries = Config.COUNTRIES.keys()
        self.nuclides = Config.NUCLIDES_TO_PROCESS
        self.weights = Config.NUCLIDE_RISK_WEIGHTS

    def find_nearest_point_in_ds(self, ds, lon, lat):
        """找到最接近给定坐标的网格点"""
        lon_idx = np.abs(ds.longitude.values - lon).argmin()
        lat_idx = np.abs(ds.latitude.values - lat).argmin()
        return lon_idx, lat_idx

    def extract_and_weight(self):
        """核心处理逻辑：读取4个文件，提取平均浓度，并加权求和"""

        country_conc_sums = {country: 0.0 for country in self.countries}

        self.logger.info("\n" + "=" * 60)
        self.logger.info("多核素数据处理：读取NC文件并计算 C_Total-Risk")
        self.logger.info("=" * 60)

        # ===拼接路径===
        for nuclide in self.nuclides:
            file_path = Config.INPUT_DIR+ nuclide+'/'+f"{nuclide}.nc" # ===拼接路径===
            weight = self.weights[nuclide]

            try:
                ds = xr.open_dataset(file_path)
                self.logger.info(f"成功加载文件: {nuclide} ({file_path}). 风险权重: {weight:.1f}")
            except Exception as e:
                self.logger.error(f"加载文件失败: {file_path}. 请确保任务一已运行。跳过该核素。错误: {e}")
                continue

            for country, coords in Config.COUNTRIES.items():
                # 1. 提取浓度 C_i (Bq/m³)
                # 假设只考虑表层 (深度索引 z=0)
                lon_idx, lat_idx = self.find_nearest_point_in_ds(ds, coords['lon'], coords['lat'])
                conc_series = ds.concentration[:, 0, lat_idx, lon_idx].values

                # 排除0值并计算平均值 (Bq/m³)
                conc_valid = conc_series[conc_series > 0]
                avg_conc_bq_m3 = np.mean(conc_valid) if len(conc_valid) > 0 else 0.0

                # 转换为 Bq/L (1 m³ = 1000 L)
                avg_conc_bq_l = avg_conc_bq_m3 / 1000.0

                # 2. 加权求和: C_Total-Risk += C_i * W_i
                country_conc_sums[country] += avg_conc_bq_l * weight

                self.logger.debug(f"  {country} - {nuclide}: C_avg={avg_conc_bq_l:.6e} Bq/L * W={weight:.1f}")

        # 3. 填充 Config.CUSTOM_INPUTS
        self.logger.info("\n开始填充 Config.CUSTOM_INPUTS:")

        for country in self.countries:
            total_risk_conc = country_conc_sums[country]
            d_adult = Config.FIXED_D_ADULT[country]

            Config.CUSTOM_INPUTS[country] = {
                'C_avg': total_risk_conc,  # C_Total-Risk (加权求和后的当量浓度 Bq/L)
                'D_adult': d_adult  # 使用固定的消费量 D
            }
            self.logger.info(f"  {country}: C_Total-Risk = {total_risk_conc:.6e} Bq/L, D_adult = {d_adult:.2f} kg/天")

        self.logger.info("多核素数据处理完成。任务二将使用 C_Total-Risk 进行评估。")


# ============================================================================
# 数据提取模块 - 使用 Config.CUSTOM_INPUTS
# ============================================================================
class ConcentrationExtractor:
    """直接从配置中获取各国给定的浓度数据"""

    def __init__(self, nc_file, logger):
        self.logger = logger
        self.logger.info("跳过NC文件加载。使用 MultiNuclideProcessor 计算的 C_Total-Risk 进行计算。")
        # 尝试加载NC文件，但其数据不再用于浓度提取
        try:
            self.ds = xr.open_dataset(nc_file)
        except Exception:
            self.ds = None

    def extract_all_countries(self):
        """直接从配置字典中提取六国给定浓度"""
        concentrations = {}
        self.logger.info("\n" + "=" * 60)
        self.logger.info("开始使用 C_Total-Risk 作为六国平均浓度 (Bq/L)")
        self.logger.info("=" * 60)

        for country, data in Config.CUSTOM_INPUTS.items():
            conc = data['C_avg']
            concentrations[country] = conc
            self.logger.info(f"  {country}: {conc:.6e} Bq/L, 对应日均消费量 D={data['D_adult']:.2f} kg/天")

        return concentrations


# ============================================================================
# 指标计算模块 - 已更新 D 的获取方式
# ============================================================================
class IndicatorCalculator:
    """计算10个二级指标"""

    def __init__(self, concentrations, logger):
        self.C = concentrations  # 字典：国家名 -> 浓度(Bq/L)
        self.logger = logger
        self.K = Config.K_BIOCONCENTRATION  # 使用 K 值 5e4

    def get_D_value(self, country):
        """获取国家特定的日均消费量 D"""
        return Config.CUSTOM_INPUTS[country]['D_adult']

    def calculate_A1(self, C):
        """A1: 核素累积量 = C × K"""
        return C * self.K

    def calculate_A2(self, C):
        """A2: 浮游生物死亡率 (Logistic模型)"""
        # C为 Bq/L 单位
        return 1.0 / (1.0 + np.exp(-(C - 0.5) / 0.1))

    def calculate_B1(self, C):
        """B1: 渔获量下降率 = 0.02 × C (C单位：Bq/L)"""
        return 0.02 * C

    def calculate_B2(self, B1):
        """B2: 出口量损失 = 0.8 × B1"""
        return 0.8 * B1

    def calculate_B3(self, B1):
        """B3: 渔民收入减少率 = 0.9 × B1"""
        return 0.9 * B1

    def calculate_C1(self, C, country):
        """C1: 成人年摄入量 = D × C × 365 (使用各国特定的 D 值)"""
        D = self.get_D_value(country)
        return D * C * 365

    def calculate_C2(self, C1):
        """C2: 儿童年摄入量 = 0.5 × C1"""
        return 0.5 * C1

    def calculate_C3(self, C1):
        """C3: 超标风险率 (阈值 100 Bq/年)"""
        # C1的单位为 Bq/年
        return 1.0 if C1 > 100 else 0.0

    def calculate_C4(self, C):
        """C4: 食品信任度下降 = 0.03 × C"""
        return 0.03 * C

    def calculate_C5(self, C):
        """C5: 食品替代成本 = 0.01 × C"""
        return 0.01 * C

    def calculate_all_indicators(self):
        """计算所有国家的所有指标"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("开始计算10个二级指标")
        self.logger.info("=" * 60)
        results = {}
        for country, C in self.C.items():
            D = self.get_D_value(country)
            self.logger.info(f"\n计算国家: {country} (C={C:.6e} Bq/L, K={self.K:.1e}, D={D:.2f} kg/天)")
            A1 = self.calculate_A1(C)
            A2 = self.calculate_A2(C)
            B1 = self.calculate_B1(C)
            B2 = self.calculate_B2(B1)
            B3 = self.calculate_B3(B1)
            C1 = self.calculate_C1(C, country)
            C2 = self.calculate_C2(C1)
            C3 = self.calculate_C3(C1)
            C4 = self.calculate_C4(C)
            C5 = self.calculate_C5(C)
            results[country] = {
                'A1': A1, 'A2': A2,
                'B1': B1, 'B2': B2, 'B3': B3,
                'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4, 'C5': C5
            }
            self.logger.info(f"  A1(核素累积): {A1:.6e}, A2(死亡率): {A2:.6f}, B1(渔获下降): {B1:.6f}")
            self.logger.info(f"  B2(出口损失): {B2:.6f}, B3(收入减少): {B3:.6f}")
            self.logger.info(f"  C1(成人摄入/年): {C1:.6f}, C2(儿童摄入/年): {C2:.6f}, C3(超标风险): {C3:.0f}")
            self.logger.info(f"  C4(信任度): {C4:.6f}, C5(替代成本): {C5:.6f}")
        return results


# ============================================================================
# 标准化与权重计算
# ============================================================================
class WeightCalculator:
    """计算熵权法权重和组合权重"""

    def __init__(self, normalized_df, logger):
        self.df = normalized_df
        self.logger = logger
        self.indicators = list(normalized_df.columns)  # 确保顺序正确

    def calculate_entropy_weights(self):
        """熵权法计算客观权重"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("熵权法计算客观权重 (Wo)")
        self.logger.info("=" * 60)
        n = len(self.df)  # 6个国家
        weights = {}
        entropies = {}
        for indicator in self.indicators:
            # 1. 计算概率 Pij
            col_sum = self.df[indicator].sum()
            # 避免除以0或总和为0，此时权重无效或均匀
            if col_sum == 0:
                P = np.zeros(n)
            else:
                P = self.df[indicator].values / col_sum

            # 2. 计算信息熵 Ej
            # 避免 log(0)，使用 1e-10 替代 0
            P_safe = np.where(P > 1e-10, P, 1e-10)
            E = -np.sum(P_safe * np.log(P_safe)) / np.log(n)
            entropies[indicator] = E
            self.logger.info(f"{indicator}: E={E:.6f}")

        # 3. 计算客观权重 Wo
        sum_1_minus_E = sum(1 - E for E in entropies.values())
        if sum_1_minus_E == 0:
            # 如果所有熵值都为1，则权重平均分配
            Wo_default = 1.0 / len(self.indicators)
            self.logger.warning("所有指标的熵值都为1，客观权重平均分配。")
            for indicator in self.indicators:
                weights[indicator] = Wo_default
        else:
            for indicator in self.indicators:
                weights[indicator] = (1 - entropies[indicator]) / sum_1_minus_E

        # Log权重
        for indicator, Wo in weights.items():
            self.logger.info(f"{indicator}: Wo={Wo:.6f}")

        return weights

    def combine_weights(self, entropy_weights, ahp_weights):
        """组合权重: Wj = 0.6*Wo + 0.4*Ws"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("组合权重计算 (Wj = 0.6*Wo + 0.4*Ws)")
        self.logger.info("=" * 60)
        combined = {}
        for indicator in self.indicators:
            Wo = entropy_weights[indicator]
            Ws = ahp_weights[indicator]
            Wj = 0.6 * Wo + 0.4 * Ws
            combined[indicator] = Wj
            self.logger.info(f"{indicator}: Wo={Wo:.6f}, Ws={Ws:.6f}, Wj={Wj:.6f}")
        return combined


# ============================================================================
# K-means聚类分析
# ============================================================================
class RiskClassifier:
    """K-means聚类进行风险分级"""

    def __init__(self, scores_df, logger):
        self.df = scores_df
        self.logger = logger

    def calculate_wcss(self, k):
        """计算指定K值的簇内平方和"""
        X = self.df[['综合得分']].values
        # 设置n_init=auto
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X)
        return kmeans.inertia_

    def elbow_method(self):
        """肘部法则确定最佳K值"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("肘部法则确定聚类数K")
        self.logger.info("=" * 60)
        wcss_values = {}
        # 尝试 K=1 到 K=4
        for k in range(1, min(5, len(self.df))):
            wcss = self.calculate_wcss(k)
            wcss_values[k] = wcss
            self.logger.info(f"K={k}: WCSS={wcss:.6f}")
        return wcss_values

    def kmeans_clustering(self, k=3):
        """执行K-means聚类 (K=3)"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"K-means聚类 (K={k})")
        self.logger.info("=" * 60)
        X = self.df[['综合得分']].values

        # 优化初始中心选择：使用综合得分最大、中间、最小值作为初始中心，确保聚类结果稳定且可解释
        scores_sorted = sorted(enumerate(X.flatten()), key=lambda x: x[1], reverse=True)
        if len(scores_sorted) >= 3:
            init_centers = np.array([
                [scores_sorted[0][1]],  # 最大值 (高风险)
                [scores_sorted[len(scores_sorted) // 2][1]],  # 中间值 (中风险)
                [scores_sorted[-1][1]]  # 最小值 (低风险)
            ])
            n_init_val = 1  # 使用自定义初始化，只运行一次
        else:
            # 如果样本太少，使用默认初始化
            init_centers = 'k-means++'
            n_init_val = 'auto'

        self.logger.info(f"初始聚类中心设置: {init_centers if isinstance(init_centers, np.ndarray) else 'k-means++'}")

        # 执行K-means
        kmeans = KMeans(n_clusters=k, init=init_centers, n_init=n_init_val, random_state=42)
        kmeans.fit(X)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # 按中心值排序标签 (0=高风险, 1=中风险, 2=低风险)
        center_order = np.argsort(centers.flatten())[::-1]
        label_mapping = {old: new for new, old in enumerate(center_order)}
        labels = np.array([label_mapping[l] for l in labels])

        # 排序后的中心值
        sorted_centers = sorted(centers.flatten(), reverse=True)

        self.logger.info("\n最终聚类中心:")
        risk_levels = ['高风险', '中风险', '低风险']
        for i, center in enumerate(sorted_centers):
            self.logger.info(f"  {risk_levels[i]}: μ = {center:.6f}")

        # 记录每个国家的聚类结果
        self.logger.info("\n聚类结果:")
        final_results = []
        for i, (country, score) in enumerate(zip(self.df.index, X.flatten())):
            risk_level = risk_levels[labels[i]]
            self.logger.info(f"  {country}: {risk_level} (得分={score:.6f})")
            final_results.append((country, score, risk_level))

        return labels, sorted_centers, final_results


# ============================================================================
# 主程序
# ============================================================================
def main():
    # 设置日志
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("任务二：核废水影响评估与风险分级 - 使用 C_Total-Risk 进行评估")
    logger.info(f"K (生物富集系数) = {Config.K_BIOCONCENTRATION:.1e}")
    logger.info("=" * 60)

    # # ========== 步骤0: 多核素数据整合 (新增) ==========
    # processor = MultiNuclideProcessor(logger)
    # processor.extract_and_weight()

    # ========== 步骤1: 提取六国平均浓度 (从Config.CUSTOM_INPUTS中获取) ==========
    extractor = ConcentrationExtractor(Config.INPUT_NC, logger)
    concentrations = extractor.extract_all_countries()

    # ========== 步骤2: 计算原始指标 ==========
    calculator = IndicatorCalculator(concentrations, logger)
    raw_indicators = calculator.calculate_all_indicators()

    # 转换为DataFrame
    raw_df = pd.DataFrame(raw_indicators).T
    raw_df.to_csv(Config.RAW_DATA_CSV, encoding='utf-8-sig')
    logger.info(f"\n原始指标已保存: {Config.RAW_DATA_CSV}")

    # ========== 步骤3: Min-Max标准化 ==========
    logger.info("\n" + "=" * 60)
    logger.info("Min-Max标准化")
    logger.info("=" * 60)
    normalized_df = raw_df.copy()
    for col in raw_df.columns:
        min_val = raw_df[col].min()
        max_val = raw_df[col].max()
        logger.info(f"{col}: min={min_val:.6e}, max={max_val:.6e}")
        if max_val - min_val > 1e-12:  # 增加一个小阈值防止除以几乎为零的数
            normalized_df[col] = (raw_df[col] - min_val) / (max_val - min_val)
        else:
            normalized_df[col] = 0.0  # 如果所有值相同，标准化结果为0

    normalized_df.to_csv(Config.NORMALIZED_CSV, encoding='utf-8-sig')
    logger.info(f"\n标准化数据已保存: {Config.NORMALIZED_CSV}")

    # ========== 步骤4: 计算权重 ==========
    weight_calc = WeightCalculator(normalized_df, logger)
    entropy_weights = weight_calc.calculate_entropy_weights()
    combined_weights = weight_calc.combine_weights(entropy_weights, Config.AHP_WEIGHTS)

    # 保存权重
    weights_df = pd.DataFrame({
        '指标': list(combined_weights.keys()),
        '熵权法Wo': [entropy_weights[k] for k in combined_weights.keys()],
        'AHP法Ws': [Config.AHP_WEIGHTS[k] for k in combined_weights.keys()],
        '组合权重Wj': list(combined_weights.values())
    })
    weights_df.to_csv(Config.WEIGHTS_CSV, index=False, encoding='utf-8-sig')
    logger.info(f"\n权重数据已保存: {Config.WEIGHTS_CSV}")

    # ========== 步骤5: 计算综合得分 ==========
    logger.info("\n" + "=" * 60)
    logger.info("计算综合风险得分")
    logger.info("=" * 60)
    scores = {}
    for country in normalized_df.index:
        score = sum(normalized_df.loc[country, ind] * combined_weights[ind]
                    for ind in combined_weights.keys())
        scores[country] = score
        logger.info(f"{country}: {score:.6f}")

    scores_df = pd.DataFrame(list(scores.items()), columns=['国家', '综合得分'])
    scores_df = scores_df.set_index('国家')

    # ========== 步骤6: K-means聚类 ==========
    classifier = RiskClassifier(scores_df, logger)

    # 肘部法则
    wcss_values = classifier.elbow_method()

    # K=3聚类
    labels, centers, final_results = classifier.kmeans_clustering(k=3)

    # 添加风险等级
    risk_levels = ['高风险', '中风险', '低风险']
    scores_df['风险等级'] = [risk_levels[l] for l in labels]
    scores_df['聚类标签'] = labels
    scores_df.to_csv(Config.CLUSTERING_CSV, encoding='utf-8-sig')
    logger.info(f"\n聚类结果已保存: {Config.CLUSTERING_CSV}")

    # ========== 步骤7: 生成Excel报告 ==========
    logger.info("\n" + "=" * 60)
    logger.info("生成完整Excel报告")
    logger.info("=" * 60)

    with pd.ExcelWriter(Config.RESULTS_EXCEL, engine='openpyxl') as writer:
        # 工作表1: 原始指标
        raw_df.to_excel(writer, sheet_name='1-原始指标')
        # 工作表2: 标准化指标
        normalized_df.to_excel(writer, sheet_name='2-标准化指标')
        # 工作表3: 权重
        weights_df.to_excel(writer, sheet_name='3-权重计算', index=False)
        # 工作表4: 综合得分与聚类
        scores_df.to_excel(writer, sheet_name='4-综合得分与风险等级')
        # 工作表5: WCSS值
        wcss_df = pd.DataFrame(list(wcss_values.items()), columns=['K值', 'WCSS'])
        wcss_df.to_excel(writer, sheet_name='5-肘部法则', index=False)

    logger.info(f"Excel报告已保存: {Config.RESULTS_EXCEL}")

    # ========== 步骤8: 生成JSON摘要 ==========
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'parameters': {
            'K_bioconcentration': Config.K_BIOCONCENTRATION,
            'nuclide_risk_weights': Config.NUCLIDE_RISK_WEIGHTS,
            'fixed_daily_consumption': Config.FIXED_D_ADULT,
        },
        'concentrations_used': {
            country: {
                'C_avg': float(data['C_avg']),
                'D_adult': float(data['D_adult'])
            }
            for country, data in Config.CUSTOM_INPUTS.items()
        },
        'risk_classification': {
            country: {
                'score': float(scores_df.loc[country, '综合得分']),
                'risk_level': scores_df.loc[country, '风险等级']
            }
            for country in scores_df.index
        },
        'cluster_centers': [float(c) for c in centers]
    }

    with open(Config.SUMMARY_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"JSON摘要已保存: {Config.SUMMARY_JSON}")

    # ========== 最终总结 ==========
    logger.info("\n" + "=" * 60)
    logger.info("任务二完成！最终风险分级结果:")
    logger.info("=" * 60)
    for risk_level in ['高风险', '中风险', '低风险']:
        countries = scores_df[scores_df['风险等级'] == risk_level].index.tolist()
        if countries:
            logger.info(f"{risk_level}: {', '.join(countries)}")
    logger.info("\n所有输出文件已保存到: " + Config.OUTPUT_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()