import os
import numpy as np

# 类别映射字典
class_mapping_dict = {
    "car": 0,
    "pedestrian": 1,
    "bike": 2,
    "awning-bike": 3,
    "van": 4,
    "truck": 5,
    "bus": 6,
    "tricycle": 7,
}

# 图像尺寸（根据你的数据设置，原程序中是1200x900）
IMG_WIDTH = 900  # 图像宽度（像素）
IMG_HEIGHT = 1200  # 图像高度（像素）


def read_label_file(label_file_path):
    """
    读取标签文件内容

    Args:
        label_file_path (str): 标签文件路径

    Returns:
        list: 标签文件的每行内容列表，如果出错返回None
    """
    try:
        with open(label_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return lines
    except FileNotFoundError:
        print(f"错误：找不到文件 {label_file_path}")
        return None
    except Exception as e:
        print(f"读取文件 {label_file_path} 时发生错误：{e}")
        return None


def write_label_file(label_file_path, lines):
    """
    将转换后的标签内容写入文件

    Args:
        label_file_path (str): 输出文件路径
        lines (list): 要写入的行内容列表
    """
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(label_file_path)
        os.makedirs(output_dir, exist_ok=True)

        with open(label_file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"成功写入文件：{label_file_path}")
    except Exception as e:
        print(f"写入文件 {label_file_path} 时发生错误：{e}")


def rotated_box_to_yolo_horizontal_box(rotated_coords, img_width, img_height):
    """
    将旋转框（8个像素坐标点）转换为YOLO格式的水平框（归一化xywh）

    Args:
        rotated_coords (list): 8个像素坐标点的列表 [x1,y1,x2,y2,x3,y3,x4,y4]
        img_width (int): 图像宽度（像素）
        img_height (int): 图像高度（像素）

    Returns:
        tuple: (x_center, y_center, width, height) 归一化后的YOLO格式坐标
    """
    # 将坐标点转换为numpy数组，形状为 (4, 2)（4个角点，每个点x,y坐标）
    points = np.array(rotated_coords).reshape(-1, 2)

    # 计算水平包围框的像素坐标（xyxy）
    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])
    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])

    # 转换为YOLO格式（归一化的xywh）
    # 计算中心坐标
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0

    # 计算宽高
    width = x_max - x_min
    height = y_max - y_min

    # 归一化到0-1之间
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    # 确保数值在0-1范围内（防止边界情况超出）
    x_center_norm = np.clip(x_center_norm, 0.0, 1.0)
    y_center_norm = np.clip(y_center_norm, 0.0, 1.0)
    width_norm = np.clip(width_norm, 0.0, 1.0)
    height_norm = np.clip(height_norm, 0.0, 1.0)

    return x_center_norm, y_center_norm, width_norm, height_norm


def process_single_label_file(input_path, output_path, img_width, img_height):
    """
    处理单个标签文件，将旋转框转换为YOLO格式的水平框

    Args:
        input_path (str): 输入标签文件路径
        output_path (str): 输出标签文件路径
        img_width (int): 图像宽度（像素）
        img_height (int): 图像高度（像素）
    """
    # 读取原始标签文件
    lines = read_label_file(input_path)
    if lines is None:
        return

    converted_lines = []

    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line:  # 跳过空行
            converted_lines.append('\n')
            continue

        parts = line.split()
        if len(parts) < 9:  # 至少8个坐标 + 1个类别标签
            print(f"警告：{input_path} 第{line_idx + 1}行格式不正确，跳过该行: {line}")
            converted_lines.append(line + '\n')
            continue

        # 提取坐标和类别信息
        try:
            rotated_coords = [float(x) for x in parts[:8]]
        except ValueError:
            print(f"警告：{input_path} 第{line_idx + 1}行坐标不是数字，跳过该行: {line}")
            converted_lines.append(line + '\n')
            continue

        class_name = parts[8]

        # 处理类别映射
        if class_name in class_mapping_dict:
            class_id = class_mapping_dict[class_name]
        else:
            print(f"警告：{input_path} 第{line_idx + 1}行类别 {class_name} 不在映射字典中，跳过该行")
            converted_lines.append(line + '\n')
            continue

        # 将旋转框转换为YOLO格式的水平框
        x_center, y_center, width, height = rotated_box_to_yolo_horizontal_box(
            rotated_coords, img_width, img_height
        )

        # 构建新的行内容（YOLO格式：class_id x_center y_center width height）
        # 保留6位小数，符合YOLO格式规范
        converted_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        converted_lines.append(converted_line)

    # 写入转换后的内容
    write_label_file(output_path, converted_lines)


def batch_convert_rotated_to_yolo_horizontal(input_dir, output_dir, img_width, img_height):
    """
    批量处理指定文件夹下的所有标签文件，将旋转框转换为YOLO格式的水平框

    Args:
        input_dir (str): 输入目录路径（包含待处理的txt标签文件）
        output_dir (str): 输出目录路径（保存转换后的标签文件）
        img_width (int): 图像宽度（像素）
        img_height (int): 图像高度（像素）
    """
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误：输入目录不存在 {input_dir}")
        return

    # 获取目录下所有txt文件
    txt_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.txt')]

    if not txt_files:
        print(f"警告：在目录 {input_dir} 中未找到任何txt文件")
        return

    print(f"找到 {len(txt_files)} 个标签文件，开始转换为YOLO格式水平框...")
    print(f"图像尺寸：宽={img_width}px, 高={img_height}px")

    # 逐个处理文件
    for filename in txt_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        print(f"\n处理文件：{filename}")
        process_single_label_file(input_path, output_path, img_width, img_height)

    print(f"\n批量转换完成！共处理 {len(txt_files)} 个文件")
    print(f"输出目录：{output_dir}")


# 使用示例
if __name__ == "__main__":
    # ====================== 配置参数 ======================
    # 输入目录（包含旋转框标签文件的文件夹）
    INPUT_DIR = r"数据集标签文件夹"

    # 输出目录（保存YOLO格式水平框标签文件）
    OUTPUT_DIR = r"水平框输出文件夹"

    # 图像尺寸（根据你的实际数据修改！）
    # 原程序中是 1200（高）x 900（宽）
    CUSTOM_IMG_WIDTH = 640
    CUSTOM_IMG_HEIGHT = 640
    # =====================================================

    # 执行批量转换
    batch_convert_rotated_to_yolo_horizontal(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        img_width=CUSTOM_IMG_WIDTH,
        img_height=CUSTOM_IMG_HEIGHT
    )