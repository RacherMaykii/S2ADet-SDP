import os
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

def generate_features(npy_input_dir, output_dir, phase="train"):
    """
    读取指定文件夹下所有.npy 文件，处理并生成 Sobel 和 NDVI 图像

    Args:
        npy_input_dir: .npy 文件输入目录路径
        output_dir: 输出目录根路径
        phase: 数据集阶段，如 train/val/test
    """

    # 创建输出目录
    sobel_output_dir = os.path.join(output_dir, "sobel", "images", phase)
    ndvi_output_dir = os.path.join(output_dir, "NDVI", "images", phase)
    os.makedirs(sobel_output_dir, exist_ok=True)
    os.makedirs(ndvi_output_dir, exist_ok=True)

    # 计算 Sobel 梯度（多波段）
    def sobel_gradient_multiband(data):
        """对多波段数据计算 Sobel 梯度并融合"""
        sobel_magnitudes = []

        # 对每个波段计算 Sobel 梯度
        for i in range(data.shape[0]):  # 遍历所有波段
            band = data[i]
            # 计算 x 和 y 方向的梯度
            sobel_x = cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3)
            # 计算梯度幅值
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_magnitudes.append(gradient_magnitude)

        # 将所有波段的梯度进行平均融合
        combined_sobel = np.mean(sobel_magnitudes, axis=0)
        return combined_sobel

    # 计算 NDVI
    def calculate_ndvi(band4, band6):
        # 避免除零错误
        denominator = band6 + band4
        denominator = np.where(denominator == 0, 1e-10, denominator)
        ndvi = (band6 - band4) / denominator
        # 在归一化前处理无效值
        ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)
        # 将 NDVI 范围从 [-1,1] 映射到 [0,65535] 用于 16 位图像
        ndvi_normalized = (ndvi + 1) * 32767.5
        ndvi_normalized = np.clip(ndvi_normalized, 0, 65535).astype(np.uint16)
        return ndvi_normalized

    # 获取所有.npy 文件
    npy_files = list(Path(npy_input_dir).glob("*.npy"))

    if not npy_files:
        print(f"No .npy files found in {npy_input_dir}")
        return

    print(f"Found {len(npy_files)} .npy files in {npy_input_dir}")

    # 处理所有文件
    for npy_file_path in npy_files:
        file_name = npy_file_path.stem  # 获取不带后缀的文件名

        # 加载.npy 文件
        try:
            data = np.load(npy_file_path)
        except Exception as e:
            print(f"Error loading {npy_file_path}: {e}")
            continue

        # 检查数据形状
        if data.shape != (8, 1200, 900):
            print(f"Warning: {npy_file_path} has unexpected shape {data.shape}, expected (8, 1200, 900)")
            continue

        data = data.astype(np.float32)
        # 选择合适的波段进行处理
        red_band = data[4]  # 红光波段
        nir_band = data[6]  # 近红外波段

        # 计算 Sobel 梯度
        sobel_result = sobel_gradient_multiband(data)
        # 将 Sobel 结果归一化到 0-65535 范围用于 16 位 PNG
        sobel_normalized = ((sobel_result - sobel_result.min()) /
                           (sobel_result.max() - sobel_result.min()) * 65535).astype(np.uint16)

        # 计算 NDVI
        ndvi_result = calculate_ndvi(red_band, nir_band)

        # 保存 Sobel 图像
        sobel_img_path = os.path.join(sobel_output_dir, f"{file_name}.png")
        sobel_img = Image.fromarray(sobel_normalized, mode='I;16')
        sobel_img.save(sobel_img_path)

        # 保存 NDVI 图像
        ndvi_img_path = os.path.join(ndvi_output_dir, f"{file_name}.png")
        ndvi_img = Image.fromarray(ndvi_result)
        ndvi_img.save(ndvi_img_path)

        print(f"Processed {file_name}")

    print(f"\nDone! Output saved to:")
    print(f"  Sobel: {sobel_output_dir}")
    print(f"  NDVI: {ndvi_output_dir}")

# 使用示例
if __name__ == "__main__":
    # 直接使用示例
    npy_input_dir = r"数据集路径"
    output_dir = r"输出路径"

    generate_features(npy_input_dir, output_dir, phase="test")
