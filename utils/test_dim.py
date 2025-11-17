import xarray as xr
import os
import sys

def test_nuclide_dimensions(base_output_dir="outputs/mission1/"):
    """
    检查 outputs/mission1/ 目录下所有核素NC文件的维度。
    """
    nuclides = ['H3', 'C14', 'Sr90', 'I129']
    
    print("--- 正在检查 Mission 1 输出文件的维度 ---")
    print(f"基础输出目录: {base_output_dir}")
    print("------------------------------------------")

    all_dims_consistent = True
    first_file_dims = None
    first_file_path = None

    for nuclide in nuclides:
        # 构建文件路径
        file_path = os.path.join(base_output_dir, nuclide, f"{nuclide}.nc")
        
        print(f"\n检查文件: {file_path}")

        if not os.path.exists(file_path):
            print(f"  ❌ 错误: 文件不存在。请确保 mission1.py 已运行并生成了该文件。")
            all_dims_consistent = False
            continue

        try:
            ds = xr.open_dataset(file_path)
            
            current_dims = {
                'time': len(ds.time) if 'time' in ds.dims else 'N/A',
                'depth': len(ds.depth) if 'depth' in ds.dims else 'N/A',
                'latitude': len(ds.latitude) if 'latitude' in ds.dims else 'N/A',
                'longitude': len(ds.longitude) if 'longitude' in ds.dims else 'N/A',
            }
            print(f"  ✅ 成功加载。维度信息: {current_dims}")

            if first_file_dims is None:
                first_file_dims = current_dims
                first_file_path = file_path
            else:
                # 比较纬度和经度维度
                if current_dims['latitude'] != first_file_dims['latitude'] or \
                   current_dims['longitude'] != first_file_dims['longitude']:
                    print(f"  ⚠️ 警告: 该文件 (纬度: {current_dims['latitude']}, 经度: {current_dims['longitude']}) "
                          f"与第一个文件 ({os.path.basename(first_file_path)} 纬度: {first_file_dims['latitude']}, 经度: {first_file_dims['longitude']}) "
                          f"的纬度/经度维度不一致。")
                    all_dims_consistent = False
                else:
                    print(f"  维度与第一个文件 ({os.path.basename(first_file_path)}) 一致。")
            
            ds.close()

        except Exception as e:
            print(f"  ❌ 错误: 处理文件时发生异常: {e}")
            all_dims_consistent = False

    print("\n------------------------------------------")
    if all_dims_consistent:
        print("✅ 所有核素文件的纬度/经度维度一致。")
    else:
        print("❌ 发现核素文件的纬度/经度维度不一致。请确保所有 mission1 模拟都以相同的分辨率运行。")
    print("------------------------------------------")

if __name__ == "__main__":
    # 默认检查 outputs/mission1/ 目录
    # 如果需要检查特定分辨率的子目录，例如 outputs/mission1/0p5/，可以这样运行：
    # python test_dim.py outputs/mission1/0p5/
    if len(sys.argv) > 1:
        test_nuclide_dimensions(sys.argv[1])
    else:
        test_nuclide_dimensions()
