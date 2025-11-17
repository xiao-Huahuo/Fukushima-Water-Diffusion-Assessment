import copernicusmarine
import os
import logging
import sys

# 配置日志记录，强制 copernicusmarine 库输出所有 DEBUG (调试) 信息
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
# 定义一个你想保存文件的文件夹
DOWNLOAD_DIR = "../CMEMS"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# 定义你想保存的文件名
file_path = os.path.join(DOWNLOAD_DIR, "w.nc")

print(f"准备下载 W 变量 (2023年)，保存到: {file_path}")

try:
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy-wcur_anfc_0.083deg_P1M-m",
        variables=["wo"],
        minimum_longitude=50,
        maximum_longitude=280,
        minimum_latitude=-60,
        maximum_latitude=65,
        start_datetime="2023-08-24T00:00:00",
        end_datetime="2024-08-24T00:00:00",
        minimum_depth=0.49402499198913574,
        maximum_depth=541.0889282226562,

        username='sruixi2',
        password='1589523904Xx!',
        output_filename=file_path
    )
    print("下载请求提交成功！")

except Exception as e:
    print(f"下载失败: {e}")
    print("请确保你已经创建了用户名和密码的凭证文件！")