# 数维杯D题-福岛核污水扩散模拟和影响评估

## 数据收集和预处理
### 数据收集:
所有数据均来自Copernicus Marine Service的公开数据集
[Global Ocean Physics Analysis and Forecast](https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/services)
(数据集ID: `GLOBAL_ANALYSISFORECAST_PHY_001_024`)
### 数据预处理:
分别运行raw_data/download_py下的五个文件:
```shell
python raw_data/download_py/pull_u_v.py
python raw_data/download_py/pull_w.py
python raw_data/download_py/pull_T.py
python raw_data/download_py/pull_S.py
python raw_data/download_py/pull_H.py
```
下载的裸数据出现在raw_data/CMEMS/目录下.(包括uv.nc,w.nc,T.nc,S.nc,H.nc).
### 数据标准化
运行regridder脚本:
##### 1°✖1°精度
```shell
python raw_data/regridder.py
```
##### 0.5°✖0.5°精度
```shell
python raw_data/regridder_precise.py
```
1. regridder脚本会将五个nc文件**汇总+归一化**为一个$1°\times1°$的网格化数据源raw_data/output/model_input_1deg_nc;
2. regridder_precise脚本会将五个nc文件**汇总+归一化**为一个$0.5°\times0.5°$的网格化数据源raw_data/output/model_input_0.5deg_nc.

### 文件完整性测试
最后进行文件完整性测试:
##### 1°✖1°精度
```shell
cd raw_data
python test_nc.py
```
##### 0.5°✖0.5°精度
```shell
cd raw_data
python test_nc.py 0.5
```

---

## 数据分析



### 任务一
分别运行四个进程:
##### 1°✖1°精度
```shell
# 默认:不输入精度参数则为1°✖1°精度
python mission1.py H3
python mission1.py C14
python mission1.py Sr90
python mission1.py I129
```
1°✖1°精度约需要准备**40分钟**时间.
##### 0.5°✖0.5°精度
```shell
python mission1.py H3 0.5
python mission1.py C14 0.5
python mission1.py Sr90 0.5
python mission1.py I129 0.5
```
0.5°✖0.5°精度约需要准备**2小时40分钟**时间.
#####  输入
raw_data/output/目录下:
model_input_1deg.nc或model_input_0.5deg.nc
##### 输出
outputs/mission1/目录下:
- H3/H3.nc
- C14/C14.nc
- Sr90/Sr90.nc
- I129/I129.nc

**注**:
1. 运行完任务一后运行data_check来检验资源完整性:
```shell
python utils/data_check.py
```
运行test_dim来检验维度一致性:
```shell
python utils/test_dim.py
```
2. 如果使用的是0.5精度,则后面的任务二和任务三都应该使用0.5精度的脚本运行命令.

---

### 任务二:
##### 1°✖1°精度 & 0.5°✖0.5°精度
```shell
python mission2.py
```
##### 输入
任务一生成的outputs/mission1/目录下:
- H3/H3.nc
- C14/C14.nc
- Sr90/Sr90.nc
- I129/I129.nc
##### 输出
outputs/mission2/目录下:
- risk_assessment_results.xlsx (包含 5 个工作表，详细记录了原始指标、标准化结果、权重计算和最终聚类分级。)
- normalized_indicators.csv (各国指标的 Min-Max 标准化结果。)
- weights_calculation.csv (熵权法 $W_o$、AHP $W_s$ 和组合权重 $W_j$。)
- clustering_results.csv (各国综合得分、聚类标签和最终风险等级。)
- summary_report.json (包含关键参数、聚类中心和最终风险分级的 JSON 摘要。)
- mission2_log.txt (详细的程序运行日志。)

---

### 任务三
##### 1°✖1°精度 & 0.5°✖0.5°精度
```shell
python mission3.py
```
##### 输入
任务一生成的outputs/mission1/目录下:
- H3/H3.nc
- C14/C14.nc
- Sr90/Sr90.nc
- I129/I129.nc
##### 输出
outputs/mission3/目录下:
- 最优解集nsga2_results.csv (包含帕累托前沿上的所有最优方案，每行对应一个方案及其三个目标函数值：环境影响 $E^{30}$、总成本 $C$、最大达标时间 $t'$。)
- mission3_log.txt (详细的程序运行日志。)

### 其他
##### 文档与记录
文档和建模记录存放在doc/目录下.
##### 任务一最早到达时间
- 获取三大城市(上海,釜山,旧金山)四大核素浓度到达$10^{-5}Bq/m^3$的时间:
```shell
python utils/arrive_time.py
```
文件输出于
```md
outputs/mission1/
├── H3/
│   ├── H3_arrival_time.npy      ← 输出
│   ├── H3_arrival_time.csv      ← 输出
│   ├── H3_lat.npy               ← 输出
│   ├── H3_lon.npy               ← 输出
│   ├── checkpoints_H3
│   ├── H3.nc
│   ├── status_log_H3.txt
│   └── data_log_H3.txt
├── C14/, Sr90/, I129/           ← 同样结构
...
```
- 获取单个城市四大核素浓度的随时间变化的序列:
```shell
python utils/one_city_density_change_with_time.py
```
- 提取K-means肘部数据（需要先运行任务二）:
```shell
python utils/kmeans_elbow.py
```
文件输出于`ouputs/mission2/kmeans_elbow_plot_data.csv`. 
- 生成环境影响时间序列E(t):
```shell
python utils/Et.py
```
文件输出于`ouputs/mission3/E_t_timeseries.csv`. 
- 生成单核素随时间变化的扩散gif图
```shell
python utils/gif.py
```
gif文件输出于`outputs/figures/gif/H3_surface_animation.gif`.

---

## 数据可视化
- 调用visualize.py脚本可以直接调用visualize/下的所有绘图脚本,生成的图片存储于`outputs/figures/png/`中.
(something wrong with it)
`outputs/fine_photos/`目录存储的是多次多脚本生成的最好看的图片们.
- 生成单核素随时间变化的扩散gif图,gif文件输出于`outputs/figures/gif/H3_surface_animation.gif`.
```shell
python utils/gif.py
```
