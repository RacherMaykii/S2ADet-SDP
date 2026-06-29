import os
import csv
from pathlib import Path
from tqdm import tqdm

def count_pedestrian_difficulty(file_path):
    """
    统计单个文件中 pedestrian 且难度不为 0 的数量

    Args:
        file_path (str): 标签文件路径

    Returns:
        int: pedestrian 且难度不为 0 的数量
    """
    count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 10:
                class_name = parts[8]
                difficulty = int(parts[9])

                if class_name == 'pedestrian' and difficulty != 0:
                    count += 1
    except Exception as e:
        print(f"错误：读取文件 {file_path} 失败 - {e}")

    return count

def scan_and_generate_csv(root_dir, output_csv):
    """
    扫描指定目录下所有标签文件，统计 pedestrian 且难度不为 0 的文件，生成 CSV

    Args:
        root_dir (str): 根目录
        output_csv (str): 输出 CSV 文件路径
    """
    # 使用 glob 递归查找所有子文件夹中的 *.txt
    pattern = os.path.join(root_dir, '**', '*.txt')
    txt_files = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))

    print(f"找到 {len(txt_files)} 个标注文件")

    results = []

    for txt_file in tqdm(txt_files, desc="统计进度"):
        count = count_pedestrian_difficulty(txt_file)
        if count > 0:
            # 获取文件名（不含后缀）
            file_name = os.path.splitext(os.path.basename(txt_file))[0]
            results.append({
                'file_name': file_name,
                'pedestrian_difficulty_count': count
            })

    # 按数量降序排序
    results.sort(key=lambda x: x['pedestrian_difficulty_count'], reverse=True)

    # 写入 CSV 文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file_name', 'pedestrian_difficulty_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # 输出统计结果（只显示前 10 个）
    print("\n" + "="*60)
    print("统计结果")
    print("="*60)
    print(f"包含 pedestrian 且难度不为 0 的文件总数：{len(results)}")

    top_10 = results[:10] if len(results) > 10 else results
    print(f"\n显示前 {len(top_10)} 个文件（按数量排序）:")
    for i, file_info in enumerate(top_10, 1):
        print(f"  {i}. {file_info['file_name']} - {file_info['pedestrian_difficulty_count']} 个")

    print(f"\nCSV 文件已保存到：{output_csv}")
    print("="*60)

def main():
    # 配置参数
    ROOT_DIRECTORY = r"annfiles path"  # 修改为你的标签文件夹路径
    OUTPUT_CSV = r"pedestrian_difficulty_stats.csv"  # 输出 CSV 文件名

    print("开始扫描和统计...")
    scan_and_generate_csv(ROOT_DIRECTORY, OUTPUT_CSV)
    print("\n完成！")

if __name__ == "__main__":
    main()
