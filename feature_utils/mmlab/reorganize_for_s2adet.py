import os
import shutil
import argparse
from pathlib import Path

def reorganize(src_root, dst_root, phases=('train', 'val')):
    """
    src_root: 分割脚本的输出根目录（包含 train/, val/ 等）
    dst_root: 新数据集的根目录，将创建 images/ 和 labels/ 子目录
    """
    for phase in phases:
        src_img_dir = os.path.join(src_root, phase, 'images')
        src_ann_dir = os.path.join(src_root, phase, 'annfiles')
        dst_img_dir = os.path.join(dst_root, 'images', phase)
        dst_ann_dir = os.path.join(dst_root, 'labels', phase)

        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_ann_dir, exist_ok=True)

        if not os.path.isdir(src_img_dir):
            print(f"Warning: {src_img_dir} does not exist, skipping {phase}")
            continue

        for img_file in os.listdir(src_img_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                continue
            # 复制图像
            shutil.copy2(
                os.path.join(src_img_dir, img_file),
                os.path.join(dst_img_dir, img_file)
            )
            # 复制对应的标注文件（假设与图像同名，扩展名 .txt）
            ann_file = os.path.splitext(img_file)[0] + '.txt'
            src_ann_path = os.path.join(src_ann_dir, ann_file)
            if os.path.exists(src_ann_path):
                shutil.copy2(src_ann_path, os.path.join(dst_ann_dir, ann_file))
            else:
                print(f"Warning: annotation not found for {img_file}")

    print("Reorganization completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reorganize split dataset for S2ADet')
    parser.add_argument('src_root', help='Root directory of split output (containing train/, val/ etc.)')
    parser.add_argument('dst_root', help='Destination root directory for S2ADet format')
    parser.add_argument('--phases', nargs='+', default=['train', 'val'], help='Phases to process')
    args = parser.parse_args()
    reorganize(args.src_root, args.dst_root, args.phases)