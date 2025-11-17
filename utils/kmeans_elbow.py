# utils/extract_kmeans_elbow_data.py
"""
从任务二的Excel结果中提取K-means肘部法则数据
用于任务二图6：K-means肘部法则图
"""
import pandas as pd
import os

# 配置
INPUT_EXCEL = "../outputs/mission2/risk_assessment_results.xlsx"
OUTPUT_CSV = "../outputs/mission2/kmeans_elbow_plot_data.csv"


def extract_elbow_data():
    """从Excel文件中提取肘部法则数据"""
    print("=" * 60)
    print("提取 K-means 肘部法则数据")
    print("=" * 60)

    if not os.path.exists(INPUT_EXCEL):
        print(f"[错误] Excel文件不存在: {INPUT_EXCEL}")
        print("请先运行任务二代码生成结果文件。")
        return

    try:
        # 读取Excel中的肘部法则工作表
        df = pd.read_excel(INPUT_EXCEL, sheet_name='5-肘部法则')

        print(f"\n成功读取数据:")
        print(df.to_string(index=False))

        # 保存为独立的CSV
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print(f"\n已保存到: {OUTPUT_CSV}")

        # 计算肘部指标（变化率）
        if len(df) > 1:
            df['WCSS_Change'] = df['WCSS'].diff().abs()
            df['Change_Percentage'] = (df['WCSS_Change'] / df['WCSS'].shift(1) * 100).round(2)

            print("\n变化率分析:")
            print(df[['K值', 'WCSS', 'WCSS_Change', 'Change_Percentage']].to_string(index=False))

            # 找到最大变化率（肘部位置）
            max_change_idx = df['WCSS_Change'].idxmax()
            if pd.notna(max_change_idx):
                optimal_k = df.loc[max_change_idx, 'K值']
                print(f"\n推荐的K值: {optimal_k} (变化率最大点)")

        # 生成MATLAB绘图提示
        print("\n" + "=" * 60)
        print("MATLAB 绘图代码:")
        print("=" * 60)
        print("""
data = readtable('../outputs/mission2/kmeans_elbow_plot_data.csv');
figure;
plot(data.K___, data.WCSS, '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('聚类数 K');
ylabel('簇内平方和 WCSS');
title('K-means 肘部法则');
grid on;
% 标注K=3
hold on;
plot(3, data.WCSS(3), 'ro', 'MarkerSize', 12, 'LineWidth', 2);
text(3, data.WCSS(3), '  K=3 (选定)', 'FontSize', 12);
        """)
        print("=" * 60)

    except Exception as e:
        print(f"[错误] 读取Excel失败: {e}")


def main():
    extract_elbow_data()


if __name__ == "__main__":
    main()