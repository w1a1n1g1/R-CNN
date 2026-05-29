"""
R-CNN目标检测实验代码
实验目的：使用PyTorch调用预训练的Faster R-CNN模型进行目标检测
作者：AI Assistant
日期：2026-05-29
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 固定随机种子，保证实验结果可复现
torch.manual_seed(42)

# 定义设备（优先使用GPU，无GPU则使用CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

# 定义目标类别名称（对应VOC数据集的20个类别，索引0为背景）
class_names = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]

# 定义IoU计算函数（计算两个检测框的交并比）
def calculate_iou(box1, boxes):
    """
    计算单个检测框与多个检测框的IoU
    box1: 单个检测框坐标（x1, y1, x2, y2）
    boxes: 多个检测框坐标（n×4）
    """
    x1, y1, x2, y2 = box1
    x1s, y1s, x2s, y2s = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # 计算交集面积
    intersection_x1 = np.maximum(x1, x1s)
    intersection_y1 = np.maximum(y1, y1s)
    intersection_x2 = np.minimum(x2, x2s)
    intersection_y2 = np.minimum(y2, y2s)
    intersection_area = np.maximum(0, intersection_x2 - intersection_x1) * np.maximum(0, intersection_y2 - intersection_y1)

    # 计算每个检测框的面积
    box1_area = (x2 - x1) * (y2 - y1)
    boxes_area = (x2s - x1s) * (y2s - y1s)

    # 计算IoU
    iou = intersection_area / (box1_area + boxes_area - intersection_area + 1e-6)  # 添加小值避免除零
    return iou


def load_model():
    """
    加载预训练的Faster R-CNN模型
    """
    print("正在加载预训练模型...")
    # 加载预训练的Faster R-CNN模型（基于ResNet50 backbone）
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)  # 将模型移动到指定设备
    model.eval()  # 切换模型为评估模式
    print("模型加载完成！")
    return model


def preprocess_image(img_path):
    """
    图像预处理
    """
    # 定义图像预处理流程（适配预训练模型输入）
    transform = transforms.Compose([
        transforms.ToPILImage(),  # 将OpenCV读取的BGR图像转换为PIL图像
        transforms.Resize((600, 800)),  # 调整图像尺寸，适配模型输入
        transforms.ToTensor(),  # 转换为张量（维度：C×H×W）
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # 读取输入图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像：{img_path}，请检查路径是否正确")

    img_copy = img.copy()  # 保存原始图像，用于后续可视化
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

    # 图像预处理
    img_tensor = transform(img_rgb)
    img_tensor = img_tensor.unsqueeze(0)  # 增加批次维度
    img_tensor = img_tensor.to(device)

    return img_tensor, img_copy, img_rgb


def detect_objects(model, img_tensor, confidence_threshold=0.5, iou_threshold=0.5):
    """
    执行目标检测
    """
    # 禁用梯度计算，提升推理速度
    with torch.no_grad():
        predictions = model(img_tensor)  # 执行目标检测

    # 解析预测结果
    predictions = {k: v.cpu().numpy() for k, v in predictions[0].items()}
    boxes = predictions['boxes']  # 检测框坐标
    labels = predictions['labels']  # 目标类别索引
    scores = predictions['scores']  # 检测置信度

    print(f"原始检测结果数量：{len(boxes)}")

    # 步骤1：根据置信度阈值筛选结果
    mask = scores >= confidence_threshold
    filtered_boxes = boxes[mask]
    filtered_labels = labels[mask]
    filtered_scores = scores[mask]

    print(f"置信度筛选后结果数量：{len(filtered_boxes)}")

    # 步骤2：非极大值抑制（NMS），去除重叠检测框
    keep = []
    if len(filtered_boxes) > 0:
        # 按置信度从高到低排序
        indices = np.argsort(filtered_scores)[::-1]

        while len(indices) > 0:
            # 保留置信度最高的检测框
            current_idx = indices[0]
            keep.append(current_idx)

            if len(indices) == 1:
                break

            # 计算当前检测框与其他所有检测框的IoU
            current_box = filtered_boxes[current_idx]
            other_boxes = filtered_boxes[indices[1:]]
            ious = calculate_iou(current_box, other_boxes)

            # 保留IoU < 阈值的检测框索引
            indices = indices[1:][ious < iou_threshold]

    # 最终筛选后的检测结果
    final_boxes = filtered_boxes[keep] if keep else np.array([])
    final_labels = filtered_labels[keep] if keep else np.array([])
    final_scores = filtered_scores[keep] if keep else np.array([])

    print(f"NMS筛选后最终结果数量：{len(final_boxes)}")

    return final_boxes, final_labels, final_scores


def visualize_results(img, boxes, labels, scores, save_path=None):
    """
    可视化检测结果
    """
    img_result = img.copy()

    # 为不同类别定义不同颜色
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
        (255, 128, 128), (128, 255, 128), (128, 128, 255), (192, 192, 192),
        (64, 64, 64)
    ]

    # 在原始图像上绘制检测框和标注信息
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x1, y1, x2, y2 = box.astype(int)
        class_name = class_names[label]
        color = colors[label % len(colors)]

        # 绘制矩形检测框
        cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 2)

        # 标注类别和置信度
        text = f"{class_name}: {score:.2f}"
        # 计算文本背景
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_result, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        cv2.putText(img_result, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 显示检测结果
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    plt.title(f"R-CNN目标检测结果 (检测到{len(boxes)}个目标)")
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"结果已保存至：{save_path}")

    plt.show()

    return img_result


def run_detection(img_path, confidence_threshold=0.5, iou_threshold=0.5, save_path=None):
    """
    完整的目标检测流程
    """
    print("=" * 50)
    print("R-CNN目标检测实验")
    print("=" * 50)
    print(f"图像路径：{img_path}")
    print(f"置信度阈值：{confidence_threshold}")
    print(f"IoU阈值：{iou_threshold}")
    print("-" * 50)

    # 加载模型
    model = load_model()

    # 图像预处理
    print("正在进行图像预处理...")
    img_tensor, img_copy, img_rgb = preprocess_image(img_path)

    # 执行目标检测
    print("正在进行目标检测...")
    boxes, labels, scores = detect_objects(
        model, img_tensor,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold
    )

    # 可视化结果
    print("正在生成可视化结果...")
    result_img = visualize_results(img_copy, boxes, labels, scores, save_path)

    # 打印检测结果详情
    print("\n检测结果详情：")
    print("-" * 50)
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        class_name = class_names[label]
        print(f"{i+1}. 类别：{class_name}, 置信度：{score:.4f}, 位置：({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f})")

    print("=" * 50)
    print("检测完成！")
    print("=" * 50)

    return boxes, labels, scores, result_img


def parameter_tuning_experiment(img_path):
    """
    参数调试实验：对比不同参数设置下的检测效果
    """
    print("\n" + "=" * 60)
    print("参数调试实验")
    print("=" * 60)

    # 测试不同的置信度阈值
    confidence_thresholds = [0.3, 0.5, 0.7]
    iou_threshold = 0.5

    print("\n实验1：不同置信度阈值的检测效果对比")
    print("-" * 60)

    for conf_thresh in confidence_thresholds:
        print(f"\n置信度阈值 = {conf_thresh}")
        save_path = f"result_conf_{conf_thresh}.jpg"
        boxes, labels, scores, _ = run_detection(
            img_path,
            confidence_threshold=conf_thresh,
            iou_threshold=iou_threshold,
            save_path=save_path
        )

    # 测试不同的IoU阈值
    iou_thresholds = [0.3, 0.5, 0.7]
    confidence_threshold = 0.5

    print("\n实验2：不同IoU阈值的检测效果对比")
    print("-" * 60)

    for iou_thresh in iou_thresholds:
        print(f"\nIoU阈值 = {iou_thresh}")
        save_path = f"result_iou_{iou_thresh}.jpg"
        boxes, labels, scores, _ = run_detection(
            img_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_thresh,
            save_path=save_path
        )


# 主程序入口
if __name__ == "__main__":
    # 默认测试图像路径（用户可修改为自己的图像路径）
    test_image_path = "test_image.jpg"

    # 检查命令行参数
    import sys
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]

    print(f"测试图像路径：{test_image_path}")

    # 运行单次检测（推荐参数：置信度0.5，IoU 0.5）
    try:
        run_detection(
            test_image_path,
            confidence_threshold=0.5,
            iou_threshold=0.5,
            save_path="rcnn_detection_result.jpg"
        )

        # 运行参数调试实验（可选，取消注释以运行）
        # parameter_tuning_experiment(test_image_path)

    except Exception as e:
        print(f"错误：{e}")
        print("\n提示：")
        print("1. 请确保已安装必要的库：pip install torch torchvision opencv-python matplotlib numpy")
        print("2. 请确保测试图像存在，或修改代码中的test_image_path变量")
        print("3. 首次运行会自动下载预训练模型（约150MB），请确保网络通畅")
