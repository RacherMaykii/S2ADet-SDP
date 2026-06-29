# 切片尺寸（可以有多个尺度，通常只用一个）
patch_sizes = [640]
patch_overlap_sizes = [200]       # 重叠区域大小（步长 = size - overlap）
img_resize_ratio = [1]             # 图像不缩放
min_img_ratio = 0.6                # 窗口有效面积占比阈值
iof_thr = 0.5                      # 标注与窗口的 IOF 阈值
padding = False                    # 不填充边缘
padding_value = [0, 0, 0]          # 填充值（padding=True 时使用，根据图像通道设置）

# 使用示例
# python mmlab/dota_split.py mmlab/ss_640.py 输入路径 输出路径 --ann-subdir labels --phase train test --nproc 8 --save-ext png --overwrite