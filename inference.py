"""
Faster R-CNN 模型推理脚本
功能：加载训练完成的模型权重，对自定义图像进行目标检测
      可视化检测框、类别与置信度，保存检测结果图像
用法：python inference.py --model faster_rcnn_final.pth --image test.jpg
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
import argparse
import os

import matplotlib
matplotlib.use('Agg')  # 无图形界面环境使用Agg后端
import matplotlib.pyplot as plt


# =====================================================================
# 模型构建函数（与train.py保持一致）
# =====================================================================

def get_model(num_classes):
    """构建Faster R-CNN模型"""
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# =====================================================================
# VOC类别名称
# =====================================================================

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

# 为不同类别分配颜色（便于可视化区分）
def get_color(class_id):
    """根据类别ID生成固定颜色"""
    np.random.seed(class_id + 42)
    return tuple(np.random.randint(50, 255, 3).tolist())


# =====================================================================
# 推理函数
# =====================================================================

def inference_image(model_path, image_path, device, confidence_threshold=0.5,
                    output_path="detection_result.jpg"):
    """
    使用训练好的模型对单张图像进行推理

    参数:
        model_path: 训练好的模型权重路径（.pth文件）
        image_path: 待检测图像路径
        device: 推理设备（cuda或cpu）
        confidence_threshold: 置信度阈值，低于此值的结果将被过滤
        output_path: 检测结果保存路径

    返回:
        result_img: 标注了检测框的图像（BGR格式）
    """
    # -------------------- 加载模型 --------------------
    print(f"正在加载模型: {model_path}")
    num_classes = 21  # VOC2012: 20类别 + 1背景
    model = get_model(num_classes)

    # 加载权重（支持checkpoint格式和纯state_dict格式）
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在: {model_path}")
        return None

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '未知')
        print(f"加载检查点: Epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print("模型加载完成！")

    # -------------------- 读取图像 --------------------
    if not os.path.exists(image_path):
        print(f"错误：图像文件不存在: {image_path}")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图像: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width = image.shape[:2]
    print(f"图像尺寸: {original_width}x{original_height}")

    # -------------------- 图像预处理 --------------------
    # 等比例缩放到模型输入尺寸范围
    min_size, max_size = 800, 1333
    scale = min(min_size / min(original_height, original_width),
                max_size / max(original_height, original_width))
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized = cv2.resize(image_rgb, (new_width, new_height))

    # 转换为张量并归一化
    image_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

    # ImageNet归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std

    # 增加批次维度
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # -------------------- 推理 --------------------
    print("正在进行目标检测...")
    with torch.no_grad():
        predictions = model(image_tensor)

    # -------------------- 解析结果 --------------------
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()

    # 置信度阈值筛选
    mask = pred_scores >= confidence_threshold
    pred_boxes = pred_boxes[mask]
    pred_labels = pred_labels[mask]
    pred_scores = pred_scores[mask]

    print(f"检测到 {len(pred_boxes)} 个目标（置信度阈值: {confidence_threshold}）")

    # -------------------- 可视化 --------------------
    result_img = image.copy()

    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        # 将缩放后的坐标映射回原始图像尺寸
        x1 = int(box[0] / scale)
        y1 = int(box[1] / scale)
        x2 = int(box[2] / scale)
        y2 = int(box[3] / scale)

        # 确保坐标在图像范围内
        x1 = max(0, min(x1, original_width - 1))
        y1 = max(0, min(y1, original_height - 1))
        x2 = max(0, min(x2, original_width - 1))
        y2 = max(0, min(y2, original_height - 1))

        class_name = VOC_CLASSES[label] if label < len(VOC_CLASSES) else f"Class_{label}"
        color = get_color(label)

        # 绘制检测框
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

        # 绘制标签背景
        label_text = f"{class_name}: {score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(result_img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)

        # 绘制标签文字
        cv2.putText(result_img, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # -------------------- 保存结果 --------------------
    cv2.imwrite(output_path, result_img)
    print(f"检测结果已保存: {output_path}")

    # 使用matplotlib保存高质量图像
    plt.figure(figsize=(14, 10))
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Faster R-CNN 检测结果 (共 {len(pred_boxes)} 个目标)", fontsize=16)
    plt.axis('off')
    plt.tight_layout()

    output_png = output_path.rsplit('.', 1)[0] + '_hd.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"高清结果已保存: {output_png}")
    plt.close()

    # -------------------- 打印详情 --------------------
    print(f"\n{'=' * 60}")
    print(f"检测结果详情：")
    print(f"{'=' * 60}")
    for i, (box, label, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
        class_name = VOC_CLASSES[label] if label < len(VOC_CLASSES) else f"Class_{label}"
        print(f"  {i + 1:2d}. {class_name:15s}  置信度: {score:.4f}  "
              f"框: ({int(box[0]):4d},{int(box[1]):4d},{int(box[2]):4d},{int(box[3]):4d})")
    print(f"{'=' * 60}")

    return result_img


# =====================================================================
# 批量推理函数
# =====================================================================

def inference_batch(model_path, image_dir, device, confidence_threshold=0.5,
                    output_dir="results"):
    """
    对文件夹中的所有图像进行批量推理

    参数:
        model_path: 模型权重路径
        image_dir: 图像文件夹路径
        device: 推理设备
        confidence_threshold: 置信度阈值
        output_dir: 结果保存目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 支持的图像格式
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    image_files = [
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ]

    if len(image_files) == 0:
        print(f"错误：在 {image_dir} 中未找到图像文件")
        return

    print(f"找到 {len(image_files)} 张图像，开始批量检测...")

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        out_path = os.path.join(output_dir, f"det_{img_file}")

        print(f"\n[{i + 1}/{len(image_files)}] 处理: {img_file}")
        inference_image(model_path, img_path, device, confidence_threshold, out_path)

    print(f"\n批量检测完成！结果保存在: {output_dir}")


# =====================================================================
# 主程序
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Faster R-CNN 目标检测推理')
    parser.add_argument('--model', type=str, default='faster_rcnn_final.pth',
                        help='模型权重路径')
    parser.add_argument('--image', type=str, default='./test.jpg',
                        help='待检测图像路径')
    parser.add_argument('--dir', type=str, default=None,
                        help='批量检测：图像文件夹路径')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='置信度阈值（默认0.5）')
    parser.add_argument('--output', type=str, default='detection_result.jpg',
                        help='检测结果保存路径')
    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    if args.dir:
        # 批量推理模式
        inference_batch(args.model, args.dir, device, args.conf, args.output)
    else:
        # 单图推理模式
        result = inference_image(
            args.model, args.image, device,
            confidence_threshold=args.conf,
            output_path=args.output
        )
