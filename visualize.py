import os
import sys
import importlib
import matplotlib.pyplot as plt # 导入matplotlib

# 导入所有绘图脚本中的函数 (已更新为新的命名和函数签名)
from visualize.util_1_plot_mission1_h3_surface_concentration_heatmap import plot_h3_surface_concentration_heatmap
from visualize.util_2_plot_mission1_sr90_arrival_time_map import plot_sr90_arrival_time_map
from visualize.util_3_plot_mission1_nuclide_comparison_line_plot import plot_nuclide_comparison_line_plot
from visualize.util_4_plot_mission2_raw_indicators_bar_chart import plot_raw_indicators_bar_chart
from visualize.util_5_plot_mission2_normalized_indicators_heatmap import plot_normalized_indicators_heatmap
from visualize.util_6_plot_mission2_weights_comparison_bar_chart import plot_weights_comparison_bar_chart
from visualize.util_7_plot_mission2_risk_score_classification_bar_chart import plot_risk_score_classification_bar_chart
from visualize.util_8_plot_mission2_elbow_method_plot import plot_elbow_method_plot
from visualize.util_9_plot_mission3_pareto_front_3d_scatter import plot_pareto_front_3d_scatter
from visualize.util_10_plot_mission3_environmental_impact_timeseries import plot_environmental_impact_timeseries

# 新增的20个绘图脚本导入
from visualize.util_11_plot_mission1_h3_depth_profile import plot_h3_depth_profile
from visualize.util_12_plot_mission1_sr90_time_series_at_ports import plot_sr90_time_series_at_ports
from visualize.util_13_plot_mission1_c14_3d_concentration_slice import plot_c14_3d_concentration_slice
from visualize.util_14_plot_mission1_i129_global_diffusion_snapshots import plot_i129_global_diffusion_snapshots
from visualize.util_15_plot_mission1_all_nuclides_surface_comparison import plot_all_nuclides_surface_comparison
from visualize.util_16_plot_mission1_h3_concentration_exceedance_area import plot_h3_concentration_exceedance_area
from visualize.util_17_plot_mission1_sr90_depth_integrated_concentration_map import plot_sr90_depth_integrated_concentration_map
from visualize.util_18_plot_mission2_country_radar_chart import plot_country_radar_chart
from visualize.util_19_plot_mission2_risk_score_scatter_with_clusters import plot_risk_score_scatter_with_clusters
from visualize.util_20_plot_mission2_risk_level_distribution import plot_risk_level_distribution
from visualize.util_21_plot_mission2_indicator_contribution_to_score import plot_indicator_contribution_to_score
from visualize.util_22_plot_mission2_risk_dimension_comparison import plot_risk_dimension_comparison
from visualize.util_23_plot_mission2_normalized_indicators_boxplot import plot_normalized_indicators_boxplot
from visualize.util_24_plot_mission2_correlation_heatmap import plot_correlation_heatmap
from visualize.util_25_plot_mission3_pareto_front_2d_projections import plot_pareto_front_2d_projections
from visualize.util_26_plot_mission3_scheme_cost_breakdown import plot_scheme_cost_breakdown
from visualize.util_27_plot_mission3_nuclide_compliance_time_comparison import plot_nuclide_compliance_time_comparison
from visualize.util_28_plot_mission3_decision_matrix_heatmap import plot_decision_matrix_heatmap
from visualize.util_29_plot_mission3_environmental_impact_by_nuclide import plot_environmental_impact_by_nuclide
from visualize.util_30_plot_mission3_cost_vs_compliance_time import plot_cost_vs_compliance_time

# 新增的30个绘图脚本导入
from visualize.util_31_plot_mission1_h3_surface_concentration_violin import plot_h3_surface_concentration_violin
from visualize.util_32_plot_mission1_nuclide_arrival_time_kde import plot_nuclide_arrival_time_kde
from visualize.util_33_plot_mission1_h3_depth_time_heatmap import plot_h3_depth_time_heatmap
from visualize.util_34_plot_mission1_sr90_surface_concentration_contour import plot_sr90_surface_concentration_contour
from visualize.util_35_plot_mission1_c14_depth_distribution_boxen import plot_c14_depth_distribution_boxen
from visualize.util_36_plot_mission1_nuclide_time_series_facetgrid import plot_nuclide_time_series_facetgrid
from visualize.util_37_plot_mission1_sr90_arrival_time_scatter_3d import plot_sr90_arrival_time_scatter_3d
from visualize.util_38_plot_mission1_i129_surface_concentration_animation import plot_i129_surface_concentration_animation
from visualize.util_39_plot_mission2_risk_score_distribution_hist import plot_risk_score_distribution_hist
from visualize.util_40_plot_mission2_indicator_pairplot import plot_indicator_pairplot
from visualize.util_41_plot_mission2_weights_pie_chart import plot_weights_pie_chart
from visualize.util_42_plot_mission2_risk_score_swarmplot import plot_risk_score_swarmplot
from visualize.util_43_plot_mission3_pareto_front_parallel_coordinates import plot_pareto_front_parallel_coordinates
from visualize.util_44_plot_mission3_nuclide_eta_comparison_heatmap import plot_nuclide_eta_comparison_heatmap
from visualize.util_45_plot_mission3_cost_vs_environmental_impact_jointplot import plot_cost_vs_environmental_impact_jointplot
from visualize.util_46_plot_mission3_pareto_front_scatter_matrix import plot_pareto_front_scatter_matrix
from visualize.util_47_plot_mission3_nuclide_decay_half_life_bar import plot_nuclide_decay_half_life_bar
from visualize.util_48_plot_mission3_cost_vs_compliance_time_facetgrid import plot_cost_vs_compliance_time_facetgrid
from visualize.util_49_plot_mission3_environmental_impact_timeseries_stacked import plot_environmental_impact_timeseries_stacked
from visualize.util_50_plot_mission3_objective_tradeoff_scatter import plot_objective_tradeoff_scatter
# from visualize.util_51_plot_mission1_all_nuclides_depth_integrated_comparison import plot_all_nuclides_depth_integrated_comparison
from visualize.util_52_plot_mission1_nuclide_max_concentration_comparison import plot_nuclide_max_concentration_comparison
from visualize.util_53_plot_mission1_sr90_depth_slice_animation import plot_sr90_depth_slice_animation
# from visualize.util_54_plot_mission1_nuclide_decay_rate_comparison import plot_nuclide_decay_rate_comparison
from visualize.util_55_plot_mission1_source_term_comparison import plot_mission1_source_term_comparison
from visualize.util_56_plot_mission2_risk_score_heatmap_by_indicator import plot_risk_score_heatmap_by_indicator
from visualize.util_57_plot_mission2_risk_score_distribution_kde_facetgrid import plot_risk_score_distribution_kde_facetgrid
from visualize.util_58_plot_mission2_weights_stacked_bar_chart import plot_weights_stacked_bar_chart
from visualize.util_59_plot_mission2_risk_score_violin_by_dimension import plot_risk_score_violin_by_dimension
from visualize.util_60_plot_mission2_risk_score_clustermap import plot_risk_score_clustermap
from visualize.util_61_plot_mission3_pareto_front_radar_chart import plot_pareto_front_radar_chart
from visualize.util_62_plot_mission3_objective_tradeoff_pairplot import plot_objective_tradeoff_pairplot
from visualize.util_63_plot_mission3_environmental_impact_boxplot_by_scheme import plot_environmental_impact_boxplot_by_scheme
from visualize.util_64_plot_mission3_cost_distribution_hist import plot_cost_distribution_hist
from visualize.util_65_plot_mission3_compliance_time_distribution_hist import plot_compliance_time_distribution_hist
from visualize.util_66_plot_mission3_environmental_impact_distribution_hist import plot_environmental_impact_distribution_hist
from visualize.util_67_plot_mission3_objective_tradeoff_kdeplot import plot_objective_tradeoff_kdeplot
from visualize.util_68_plot_mission3_environmental_impact_violin_by_scheme import plot_environmental_impact_violin_by_scheme
from visualize.util_69_plot_mission3_cost_violin_by_scheme import plot_cost_violin_by_scheme
from visualize.util_70_plot_mission3_compliance_time_violin_by_scheme import plot_compliance_time_violin_by_scheme
from visualize.util_71_plot_mission1_h3_surface_concentration_kde_map import plot_h3_surface_concentration_kde_map
from visualize.util_72_plot_mission1_sr90_arrival_time_kde_map import plot_sr90_arrival_time_kde_map
from visualize.util_73_plot_mission1_c14_depth_time_contourf import plot_c14_depth_time_contourf
from visualize.util_74_plot_mission1_i129_global_diffusion_quiver_map import plot_i129_global_diffusion_quiver_map
from visualize.util_75_plot_mission2_country_risk_score_barplot_with_error import plot_country_risk_score_barplot_with_error
from visualize.util_76_plot_mission2_indicator_contribution_to_score_stacked_area import plot_indicator_contribution_to_score_stacked_area
from visualize.util_77_plot_mission2_risk_dimension_comparison_radar import plot_risk_dimension_comparison_radar
from visualize.util_78_plot_mission2_indicator_correlation_clustermap import plot_indicator_correlation_clustermap
from visualize.util_79_plot_mission3_pareto_front_scatter_3d_with_labels import plot_pareto_front_scatter_3d_with_labels
from visualize.util_80_plot_mission3_objective_tradeoff_hexbin import plot_objective_tradeoff_hexbin
from visualize.util_81_plot_mission1_h3_concentration_snapshots import plot_concentration_snapshots

# 新增的19个3D曲面图脚本导入
from visualize.util_82_plot_mission1_h3_surface_concentration_3d_initial import plot_h3_surface_concentration_3d_initial
from visualize.util_83_plot_mission1_h3_surface_concentration_3d_middle import plot_h3_surface_concentration_3d_middle
from visualize.util_84_plot_mission1_h3_surface_concentration_3d_final import plot_h3_surface_concentration_3d_final
from visualize.util_85_plot_mission1_c14_surface_concentration_3d_initial import plot_c14_surface_concentration_3d_initial
from visualize.util_86_plot_mission1_c14_surface_concentration_3d_middle import plot_c14_surface_concentration_3d_middle
from visualize.util_87_plot_mission1_c14_surface_concentration_3d_final import plot_c14_surface_concentration_3d_final
from visualize.util_88_plot_mission1_sr90_surface_concentration_3d_initial import plot_sr90_surface_concentration_3d_initial
from visualize.util_89_plot_mission1_sr90_surface_concentration_3d_middle import plot_sr90_surface_concentration_3d_middle
from visualize.util_90_plot_mission1_sr90_surface_concentration_3d_final import plot_sr90_surface_concentration_3d_final
from visualize.util_91_plot_mission1_i129_surface_concentration_3d_initial import plot_i129_surface_concentration_3d_initial
from visualize.util_92_plot_mission1_i129_surface_concentration_3d_middle import plot_i129_surface_concentration_3d_middle
from visualize.util_93_plot_mission1_i129_surface_concentration_3d_final import plot_i129_surface_concentration_3d_final
from visualize.util_94_plot_mission1_h3_50m_concentration_3d_final import plot_h3_50m_concentration_3d_final
from visualize.util_95_plot_mission1_c14_50m_concentration_3d_final import plot_c14_50m_concentration_3d_final
from visualize.util_96_plot_mission1_sr90_50m_concentration_3d_final import plot_sr90_50m_concentration_3d_final
from visualize.util_97_plot_mission1_i129_50m_concentration_3d_final import plot_i129_50m_concentration_3d_final
from visualize.util_98_plot_mission1_h3_200m_concentration_3d_final import plot_h3_200m_concentration_3d_final
from visualize.util_99_plot_mission1_c14_200m_concentration_3d_final import plot_c14_200m_concentration_3d_final
from visualize.util_100_plot_mission1_sr90_200m_concentration_3d_final import plot_sr90_200m_concentration_3d_final


def run_all_visualizations():
    print("=" * 60)
    print("开始运行所有可视化脚本...")
    print("=" * 60)

    # --- 全局中文显示设置 ---
    plt.rcParams['font.sans-serif'] = ["SimHei"]  # 使用SimHei字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 统一设置数据输入路径
    INPUT_PATHS = {
        # Mission 1 outputs
        'mission1_h3_nc': "outputs/mission1/H3/H3.nc",
        'mission1_c14_nc': "outputs/mission1/C14/C14.nc",
        'mission1_sr90_nc': "outputs/mission1/Sr90/Sr90.nc",
        'mission1_i129_nc': "outputs/mission1/I129/I129.nc",
        
        # Mission 2 outputs
        'mission2_raw_indicators_csv': "outputs/mission2/raw_indicators.csv",
        'mission2_normalized_indicators_csv': "outputs/mission2/normalized_indicators.csv",
        'mission2_weights_csv': "outputs/mission2/weights_calculation.csv",
        'mission2_clustering_results_csv': "outputs/mission2/clustering_results.csv",
        'mission2_results_excel': "outputs/mission2/risk_assessment_results.xlsx",
        
        # Mission 3 outputs
        'mission3_nsga2_results_csv': "outputs/mission3/nsga2_results.csv",
        'mission3_e_t_timeseries_csv': "outputs/mission3/E_t_timeseries.csv",
    }

    # 统一设置图片输出目录
    OUTPUT_PNG_DIR = "outputs/figures/png"
    OUTPUT_GIF_DIR="outputs/figures/gif"
    os.makedirs(OUTPUT_PNG_DIR, exist_ok=True)
    print(f"所有生成的图片将保存到: {OUTPUT_PNG_DIR}")


    # 定义所有要运行的绘图函数及其名称
    plot_functions = [
        ("任务一 H3 表面浓度热图", plot_h3_surface_concentration_heatmap, OUTPUT_PNG_DIR),
        ("任务一 Sr90 到达时间地图", plot_sr90_arrival_time_map, OUTPUT_PNG_DIR),
        ("任务一 各核素源点浓度对比曲线", plot_nuclide_comparison_line_plot, OUTPUT_PNG_DIR),
        ("任务二 原始指标各国对比柱状图", plot_raw_indicators_bar_chart, OUTPUT_PNG_DIR),
        ("任务二 标准化指标热力图", plot_normalized_indicators_heatmap, OUTPUT_PNG_DIR),
        ("任务二 权重对比柱状图", plot_weights_comparison_bar_chart, OUTPUT_PNG_DIR),
        ("任务二 综合风险得分与分类柱状图", plot_risk_score_classification_bar_chart, OUTPUT_PNG_DIR),
        ("任务二 K-means 肘部法则图", plot_elbow_method_plot, OUTPUT_PNG_DIR),
        ("任务三 帕累托前沿三维散点图", plot_pareto_front_3d_scatter, OUTPUT_PNG_DIR),
        ("任务三 环境影响时间序列图", plot_environmental_impact_timeseries, OUTPUT_PNG_DIR),
        ("任务一 H3 浓度深度剖面图", plot_h3_depth_profile, OUTPUT_PNG_DIR),
        ("任务一 Sr90 港口浓度时间序列图", plot_sr90_time_series_at_ports, OUTPUT_PNG_DIR),
        ("任务一 C14 三维浓度切片图", plot_c14_3d_concentration_slice, OUTPUT_PNG_DIR),
        ("任务一 I129 全球扩散快照", plot_i129_global_diffusion_snapshots, OUTPUT_PNG_DIR),
        ("任务一 各核素表面浓度对比图", plot_all_nuclides_surface_comparison, OUTPUT_PNG_DIR),
        ("任务一 H3 浓度超阈值区域时间序列图", plot_h3_concentration_exceedance_area, OUTPUT_PNG_DIR),
        ("任务一 Sr90 深度积分浓度地图", plot_sr90_depth_integrated_concentration_map, OUTPUT_PNG_DIR),
        ("任务二 各国风险指标雷达图", plot_country_radar_chart, OUTPUT_PNG_DIR),
        ("任务二 综合风险得分聚类散点图", plot_risk_score_scatter_with_clusters, OUTPUT_PNG_DIR),
        ("任务二 风险等级分布饼图", plot_risk_level_distribution, OUTPUT_PNG_DIR),
        ("任务二 各指标对综合风险得分的贡献度", plot_indicator_contribution_to_score, OUTPUT_PNG_DIR),
        ("任务二 各国在不同风险维度上的加权得分", plot_risk_dimension_comparison, OUTPUT_PNG_DIR),
        ("任务二 标准化风险指标箱线图", plot_normalized_indicators_boxplot, OUTPUT_PNG_DIR),
        ("任务二 标准化风险指标相关性热图", plot_correlation_heatmap, OUTPUT_PNG_DIR),
        ("任务三 帕累托前沿二维投影图", plot_pareto_front_2d_projections, OUTPUT_PNG_DIR),
        ("任务三 处理方案成本构成图", plot_scheme_cost_breakdown, OUTPUT_PNG_DIR),
        ("任务三 各核素达标时间对比图", plot_nuclide_compliance_time_comparison, OUTPUT_PNG_DIR),
        ("任务三 处理方案决策矩阵热图", plot_decision_matrix_heatmap, OUTPUT_PNG_DIR),
        ("任务三 各核素对环境影响的贡献图", plot_environmental_impact_by_nuclide, OUTPUT_PNG_DIR),
        ("任务三 成本 vs 达标时间散点图", plot_cost_vs_compliance_time, OUTPUT_PNG_DIR),
        ("任务一 H3 表面浓度小提琴图", plot_h3_surface_concentration_violin, OUTPUT_PNG_DIR),
        ("任务一 各核素到达时间KDE图", plot_nuclide_arrival_time_kde, OUTPUT_PNG_DIR),
        ("任务一 H3 深度-时间热图", plot_h3_depth_time_heatmap, OUTPUT_PNG_DIR),
        ("任务一 Sr90 表面浓度等高线图", plot_sr90_surface_concentration_contour, OUTPUT_PNG_DIR),
        ("任务一 C14 深度分布箱线图", plot_c14_depth_distribution_boxen, OUTPUT_PNG_DIR),
        ("任务一 各核素时间序列分面图", plot_nuclide_time_series_facetgrid, OUTPUT_PNG_DIR),
        ("任务一 Sr90 到达时间3D散点图", plot_sr90_arrival_time_scatter_3d, OUTPUT_PNG_DIR),
        ("任务一 I129 表面浓度扩散动画", plot_i129_surface_concentration_animation, OUTPUT_GIF_DIR),
        ("任务二 综合风险得分分布直方图", plot_risk_score_distribution_hist, OUTPUT_PNG_DIR),
        ("任务二 指标两两关系图", plot_indicator_pairplot, OUTPUT_PNG_DIR),
        ("任务二 权重饼图", plot_weights_pie_chart, OUTPUT_PNG_DIR),
        ("任务二 综合风险得分Swarmplot", plot_risk_score_swarmplot, OUTPUT_PNG_DIR),
        ("任务三 帕累托前沿平行坐标图", plot_pareto_front_parallel_coordinates, OUTPUT_PNG_DIR),
        ("任务三 核素去除率对比热图", plot_nuclide_eta_comparison_heatmap, OUTPUT_PNG_DIR),
        ("任务三 成本 vs 环境影响联合分布图", plot_cost_vs_environmental_impact_jointplot, OUTPUT_PNG_DIR),
        ("任务三 帕累托前沿散点矩阵图", plot_pareto_front_scatter_matrix, OUTPUT_PNG_DIR),
        ("任务三 核素半衰期对比图", plot_nuclide_decay_half_life_bar, OUTPUT_PNG_DIR),
        ("任务三 成本 vs 达标时间分面图", plot_cost_vs_compliance_time_facetgrid, OUTPUT_PNG_DIR),
        ("任务三 环境影响时间序列线图", plot_environmental_impact_timeseries_stacked, OUTPUT_PNG_DIR),
        ("任务三 目标权衡散点图", plot_objective_tradeoff_scatter, OUTPUT_PNG_DIR),
        ("任务一 各核素全球最大浓度对比图", plot_nuclide_max_concentration_comparison, OUTPUT_PNG_DIR),
        ("任务一 Sr90 深度切片动画", plot_sr90_depth_slice_animation, OUTPUT_GIF_DIR),
        ("任务一 源项强度对比图", plot_mission1_source_term_comparison, OUTPUT_PNG_DIR),
        ("任务二 风险等级排序热图", plot_risk_score_heatmap_by_indicator, OUTPUT_PNG_DIR),
        ("任务二 风险得分KDE分面图", plot_risk_score_distribution_kde_facetgrid, OUTPUT_PNG_DIR),
        ("任务二 权重堆叠柱状图", plot_weights_stacked_bar_chart, OUTPUT_PNG_DIR),
        ("任务二 风险维度小提琴图", plot_risk_score_violin_by_dimension, OUTPUT_PNG_DIR),
        ("任务二 风险得分聚类热图", plot_risk_score_clustermap, OUTPUT_PNG_DIR),
        ("任务三 帕累托前沿雷达图", plot_pareto_front_radar_chart, OUTPUT_PNG_DIR),
        ("任务三 目标权衡两两关系图", plot_objective_tradeoff_pairplot, OUTPUT_PNG_DIR),
        ("任务三 环境影响箱线图", plot_environmental_impact_boxplot_by_scheme, OUTPUT_PNG_DIR),
        ("任务三 成本分布直方图", plot_cost_distribution_hist, OUTPUT_PNG_DIR),
        ("任务三 达标时间分布直方图", plot_compliance_time_distribution_hist, OUTPUT_PNG_DIR),
        ("任务三 环境影响分布直方图", plot_environmental_impact_distribution_hist, OUTPUT_PNG_DIR),
        ("任务三 目标权衡KDE图", plot_objective_tradeoff_kdeplot, OUTPUT_PNG_DIR),
        ("任务三 环境影响小提琴图", plot_environmental_impact_violin_by_scheme, OUTPUT_PNG_DIR),
        ("任务三 成本小提琴图", plot_cost_violin_by_scheme, OUTPUT_PNG_DIR),
        ("任务三 达标时间小提琴图", plot_compliance_time_violin_by_scheme, OUTPUT_PNG_DIR),
        ("任务一 H3 表面浓度KDE地图", plot_h3_surface_concentration_kde_map, OUTPUT_PNG_DIR),
        ("任务一 Sr90 到达时间KDE地图", plot_sr90_arrival_time_kde_map, OUTPUT_PNG_DIR),
        ("任务一 C14 深度-时间等值线图", plot_c14_depth_time_contourf, OUTPUT_PNG_DIR),
        ("任务一 I129 表面浓度与洋流矢量图", plot_i129_global_diffusion_quiver_map, OUTPUT_PNG_DIR),
        ("任务二 各国综合风险得分柱状图 (带误差棒)", plot_country_risk_score_barplot_with_error, OUTPUT_PNG_DIR),
        ("任务二 指标贡献度堆叠面积图", plot_indicator_contribution_to_score_stacked_area, OUTPUT_PNG_DIR),
        ("任务二 风险维度对比雷达图", plot_risk_dimension_comparison_radar, OUTPUT_PNG_DIR),
        ("任务二 指标相关性聚类热图", plot_indicator_correlation_clustermap, OUTPUT_PNG_DIR),
        ("任务三 帕累托前沿三维散点图 (带标签)", plot_pareto_front_scatter_3d_with_labels, OUTPUT_PNG_DIR),
        ("任务三 目标权衡Hexbin图", plot_objective_tradeoff_hexbin, OUTPUT_PNG_DIR),
        ("任务一 H3 浓度快照", plot_concentration_snapshots, OUTPUT_PNG_DIR),
        # 新增的19个3D曲面图
        ("任务一 H3 表面浓度 3D 曲面图 (初始时间)", plot_h3_surface_concentration_3d_initial, OUTPUT_PNG_DIR),
        ("任务一 H3 表面浓度 3D 曲面图 (中期时间)", plot_h3_surface_concentration_3d_middle, OUTPUT_PNG_DIR),
        ("任务一 H3 表面浓度 3D 曲面图 (最终时间)", plot_h3_surface_concentration_3d_final, OUTPUT_PNG_DIR),
        ("任务一 C14 表面浓度 3D 曲面图 (初始时间)", plot_c14_surface_concentration_3d_initial, OUTPUT_PNG_DIR),
        ("任务一 C14 表面浓度 3D 曲面图 (中期时间)", plot_c14_surface_concentration_3d_middle, OUTPUT_PNG_DIR),
        ("任务一 C14 表面浓度 3D 曲面图 (最终时间)", plot_c14_surface_concentration_3d_final, OUTPUT_PNG_DIR),
        ("任务一 Sr90 表面浓度 3D 曲面图 (初始时间)", plot_sr90_surface_concentration_3d_initial, OUTPUT_PNG_DIR),
        ("任务一 Sr90 表面浓度 3D 曲面图 (中期时间)", plot_sr90_surface_concentration_3d_middle, OUTPUT_PNG_DIR),
        ("任务一 Sr90 表面浓度 3D 曲面图 (最终时间)", plot_sr90_surface_concentration_3d_final, OUTPUT_PNG_DIR),
        ("任务一 I129 表面浓度 3D 曲面图 (初始时间)", plot_i129_surface_concentration_3d_initial, OUTPUT_PNG_DIR),
        ("任务一 I129 表面浓度 3D 曲面图 (中期时间)", plot_i129_surface_concentration_3d_middle, OUTPUT_PNG_DIR),
        ("任务一 I129 表面浓度 3D 曲面图 (最终时间)", plot_i129_surface_concentration_3d_final, OUTPUT_PNG_DIR),
        ("任务一 H3 50m 深度浓度 3D 曲面图 (最终时间)", plot_h3_50m_concentration_3d_final, OUTPUT_PNG_DIR),
        ("任务一 C14 50m 深度浓度 3D 曲面图 (最终时间)", plot_c14_50m_concentration_3d_final, OUTPUT_PNG_DIR),
        ("任务一 Sr90 50m 深度浓度 3D 曲面图 (最终时间)", plot_sr90_50m_concentration_3d_final, OUTPUT_PNG_DIR),
        ("任务一 I129 50m 深度浓度 3D 曲面图 (最终时间)", plot_i129_50m_concentration_3d_final, OUTPUT_PNG_DIR),
        ("任务一 H3 200m 深度浓度 3D 曲面图 (最终时间)", plot_h3_200m_concentration_3d_final, OUTPUT_PNG_DIR),
        ("任务一 C14 200m 深度浓度 3D 曲面图 (最终时间)", plot_c14_200m_concentration_3d_final, OUTPUT_PNG_DIR),
        ("任务一 Sr90 200m 深度浓度 3D 曲面图 (最终时间)", plot_sr90_200m_concentration_3d_final, OUTPUT_PNG_DIR),
    ]

    results = []

    for plot_name, plot_func, output_dir in plot_functions:
        print(f"\n--- 正在生成: {plot_name} ---")
        try:
            # 将统一的路径传递给每个绘图函数
            plot_func(INPUT_PATHS, output_dir)
            results.append(f"[成功] {plot_name}")
        except Exception as e:
            results.append(f"[失败] {plot_name}: {e}")
            print(f"!!! 错误: 生成 {plot_name} 时发生异常: {e} !!!")
            import traceback
            traceback.print_exc() # 打印完整的错误堆栈

    print("\n" + "=" * 60)
    print("所有可视化脚本运行完毕。")
    print("运行结果摘要:")
    for res in results:
        print(res)
    print("=" * 60)

if __name__ == "__main__":
    # 将 visualize 目录添加到 Python 路径，以便正确导入
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # if current_dir not in sys.path:
    #     sys.path.insert(0, current_dir)
    run_all_visualizations()
