import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
from tqdm import tqdm


def read_npy_file(file_path):
    """
    读取 .npy 文件并返回其中的数组数据

    Args:
        file_path (str): .npy 文件路径

    Returns:
        numpy.ndarray: 从文件中读取的数组
    """
    try:
        data = np.load(file_path)
        return data
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None


def normalize_band(band, p_lo=1, p_hi=99, gamma=2.2):
    """归一化波段数据 - 使用百分位数方法+Gamma 校正"""
    lo, hi = np.percentile(band, (p_lo, p_hi))
    normalized = np.clip((band.astype(float) - lo) / (hi - lo + 1e-10), 0, 1)
    # 应用 Gamma 校正
    normalized = np.power(normalized, 1.0 / gamma)
    return np.clip(normalized, 0, 1)


def convert_npy_to_jpg(npy_file_path, output_jpg_path):
    """
    将 8 通道 npy文件转换为 RGB JPG 图像

    Args:
       npy_file_path (str): 输入的.npy文件路径
       output_jpg_path (str): 输出的.jpg 文件路径
    """
    # 读取 npy文件
    data = read_npy_file(npy_file_path)
    if data is None:
        return False

    # 检查数据形状
    if data.shape != (8, 1200, 900):
        print(f"警告：{npy_file_path} 形状为 {data.shape}，不是预期的 (8, 1200, 900)")
        return False

    # 选择合适的波段作为 RGB 通道
    # 使用与可视化光谱立方体一致的波段：第 5、3、2 波段（索引从 0 开始）
    rgb_bands = [data[4], data[2], data[1]]  # R, G, B

    # 归一化每个波段（使用 1%-99% 百分位+Gamma 校正）
    normalized_bands = [normalize_band(band) for band in rgb_bands]

    # 组合成 RGB 图像 (1200, 900, 3)
    rgb_image = np.stack([
        normalized_bands[0],  # Red
        normalized_bands[1],  # Green
        normalized_bands[2]   # Blue
    ], axis=-1)

    # 将数值范围从 [0,1] 转换到 [0,255]
    rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)

    # 使用 PIL 保存为 JPG 图像
    img = Image.fromarray(rgb_image_uint8)
    img.save(output_jpg_path, 'JPEG')

    return True


def batch_convert_npy_to_jpg(input_folder, output_folder):
    """
    批量将输入文件夹中的所有.npy文件转换为.jpg文件

    Args:
        input_folder (str): 包含.npy文件的输入文件夹路径
        output_folder (str): 保存.jpg文件的输出文件夹路径
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取输入文件夹中所有.npy文件
    npy_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.npy')]

    print(f"找到 {len(npy_files)} 个.npy文件")

    success_count = 0
    fail_count = 0

    # 使用tqdm创建进度条
    for npy_file in tqdm(npy_files, desc="转换进度", unit="file"):
        # 构建完整的文件路径
        npy_path = os.path.join(input_folder, npy_file)

        # 生成输出文件名（替换扩展名为.jpg）
        jpg_filename = os.path.splitext(npy_file)[0] + '.jpg'
        jpg_path = os.path.join(output_folder, jpg_filename)

        # 转换文件
        if convert_npy_to_jpg(npy_path, jpg_path):
            success_count += 1
            # tqdm.write(f"已保存: {jpg_filename}")  # 在进度条外输出保存信息
        else:
            fail_count += 1

    print(f"\n转换完成!")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")


# 使用示例
if __name__ == "__main__":
    # 设置输入和输出文件夹路径
    input_folder = r"数据集路径"  # 替换为你的输入文件夹路径
    output_folder = r"输出路径"      # 替换为你的输出文件夹路径

    # 执行批量转换
    batch_convert_npy_to_jpg(input_folder, output_folder)
