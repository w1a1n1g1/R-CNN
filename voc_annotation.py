"""
VOC2012数据集标注格式转换脚本
功能：将VOC2012原生XML标注文件转换为TXT格式，适配模型加载需求
输出格式：图像路径 x1,y1,x2,y2,label_id x1,y1,x2,y2,label_id ...
用法：python voc_annotation.py
"""

import xml.etree.ElementTree as ET
import os

# ===================== 配置区域 =====================

# VOC2012数据集根目录（根据实际路径修改）
# 用户数据集路径: F:\AI\VOC2012\VOCdevkit
VOC_root = "F:/AI/VOC2012/VOCdevkit/VOC2012"

# 数据集划分
sets = [('2012', 'train'), ('2012', 'val')]

# VOC2012的20个目标类别
classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# ===================== 核心函数 =====================

def convert_annotation(year, image_id, list_file):
    """
    解析单张图像的XML标注文件，将目标框信息写入TXT文件

    参数:
        year: 数据集年份
        image_id: 图像ID（不含扩展名）
        list_file: 输出文件句柄
    """
    in_file = open(f'{VOC_root}/Annotations/{image_id}.xml')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = int(obj.find('difficult').text)

        cls_name = obj.find('name').text
        # 跳过不在类别列表中的目标或标记为困难的样本
        if cls_name not in classes or difficult == 1:
            continue

        cls_id = classes.index(cls_name)
        xmlbox = obj.find('bndbox')
        b = (
            int(float(xmlbox.find('xmin').text)),
            int(float(xmlbox.find('ymin').text)),
            int(float(xmlbox.find('xmax').text)),
            int(float(xmlbox.find('ymax').text))
        )
        # 写入格式：x1,y1,x2,y2,label_id
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


# ===================== 主程序 =====================

if __name__ == '__main__':
    # 检查数据集目录是否存在
    if not os.path.exists(VOC_root):
        print(f"错误：数据集目录不存在: {VOC_root}")
        print("请先下载并解压VOC2012数据集，确保目录结构如下：")
        print("  VOCdevkit/")
        print("  └── VOC2012/")
        print("      ├── Annotations/")
        print("      ├── ImageSets/Main/")
        print("      └── JPEGImages/")
        exit(1)

    for year, image_set in sets:
        image_sets_path = f'{VOC_root}/ImageSets/Main/{image_set}.txt'

        if not os.path.exists(image_sets_path):
            print(f"错误：数据集划分文件不存在: {image_sets_path}")
            continue

        image_ids = open(image_sets_path).read().strip().split()
        list_file = open(f'{year}_{image_set}.txt', 'w')

        for image_id in image_ids:
            list_file.write(f'{VOC_root}/JPEGImages/{image_id}.jpg')
            convert_annotation(year, image_id, list_file)
            list_file.write('\n')

        list_file.close()
        print(f'生成 {year}_{image_set}.txt 完成，共 {len(image_ids)} 张图像')

    print("\n标注转换完成！生成的文件：")
    print("  - 2012_train.txt（训练集标注）")
    print("  - 2012_val.txt（验证集标注）")
