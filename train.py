"""
Faster R-CNN + PASCAL VOC2012 目标检测模型训练脚本
基于Torchvision官方fasterrcnn_resnet50_fpn模型
功能：数据加载、模型构建、训练、验证、权重保存、训练曲线可视化
用法：python train.py
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # 无图形界面环境使用Agg后端
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


# =====================================================================
# 一、自定义VOC数据集加载类
# =====================================================================

class VOCDataset(Dataset):
    """
    PASCAL VOC2012数据集加载器
    解析VOC数据集标注，输出模型所需的图像张量与标注字典
    """

    # VOC2012的20个目标类别 + 背景类
    CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"
    ]

    def __init__(self, root, image_set, transforms=None, max_samples=None, seed=42):
        """
        参数:
            root: VOC2012数据集根目录路径（如 ./VOCdevkit/VOC2012）
            image_set: 数据集划分（'train' 或 'val'）
            transforms: 图像变换组合
            max_samples: 最大采样数量，None表示使用全部数据（用于快速训练）
            seed: 随机种子，保证可复现
        """
        self.root = root
        self.transforms = transforms
        self.image_set = image_set

        # 类别映射字典
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}

        # 读取图像名称列表
        image_sets_path = os.path.join(root, "ImageSets", "Main", f"{image_set}.txt")
        with open(image_sets_path) as f:
            self.image_ids = [x.strip() for x in f.readlines()]

        # 随机采样（用于快速训练）
        if max_samples is not None and max_samples < len(self.image_ids):
            import random
            random.seed(seed)
            self.image_ids = random.sample(self.image_ids, max_samples)
            print(f"  [快速模式] 从{image_set}集中随机采样 {max_samples} 张图像")

        self.images_dir = os.path.join(root, "JPEGImages")
        self.annotations_dir = os.path.join(root, "Annotations")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """返回图像张量和对应的标注字典"""
        image_id = self.image_ids[idx]

        # 读取图像
        img_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告：无法读取图像 {img_path}，跳过")
            return self.__getitem__((idx + 1) % len(self.image_ids))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 解析XML标注文件
        xml_path = os.path.join(self.annotations_dir, f"{image_id}.xml")
        boxes = []
        labels = []

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                # 提取类别
                class_name = obj.find('name').text
                if class_name not in self.class_to_idx:
                    continue
                label = self.class_to_idx[class_name]

                # 提取边界框坐标（VOC格式：xmin, ymin, xmax, ymax）
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)

                # 过滤无效框
                if xmax <= xmin or ymax <= ymin:
                    continue

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)

        except Exception as e:
            print(f"解析XML文件失败: {xml_path}, 错误: {e}")
            return self.__getitem__((idx + 1) % len(self.image_ids))

        # 如果没有有效目标，跳过该图像
        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self.image_ids))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # 构建目标字典（Faster R-CNN要求的格式）
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        # 应用图像变换
        if self.transforms:
            image, target = self.transforms(image, target)

        # 将图像转换为张量（H×W×C → C×H×W，归一化到[0,1]）
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, target


# =====================================================================
# 二、图像预处理变换函数
# =====================================================================

class Compose:
    """组合多个图像变换"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """将图像从numpy数组转换为PyTorch张量"""

    def __call__(self, image, target):
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image, target


class Normalize:
    """使用ImageNet统计值进行归一化"""

    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        if isinstance(image, torch.Tensor):
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)
        return image, target


class Resize:
    """调整图像和边界框尺寸"""

    def __init__(self, min_size=800, max_size=1333):
        """
        参数:
            min_size: 图像最短边最小尺寸
            max_size: 图像最长边最大尺寸
        """
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        original_height, original_width = image.shape[:2]

        # 等比例缩放
        scale = min(self.min_size / min(original_height, original_width),
                    self.max_size / max(original_height, original_width))
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        image = cv2.resize(image, (new_width, new_height))

        if target is not None and "boxes" in target:
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            target["boxes"] = boxes

        return image, target


def get_transform(train):
    """获取图像变换组合"""
    transforms = []
    transforms.append(Resize(min_size=800, max_size=1333))
    transforms.append(ToTensor())
    transforms.append(Normalize())
    return Compose(transforms)


# =====================================================================
# 三、Faster R-CNN模型构建
# =====================================================================

def get_model(num_classes):
    """
    构建Faster R-CNN模型
    加载预训练ResNet50-FPN主干网络，替换分类头适配自定义类别数

    参数:
        num_classes: 类别数（VOC2012为21 = 20类别 + 1背景）
    返回:
        model: Faster R-CNN模型
    """
    # 加载预训练的Faster R-CNN模型（ResNet50-FPN主干）
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # 获取分类头的输入特征维度
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 替换分类头，适配自定义类别数
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# =====================================================================
# 四、训练与验证函数
# =====================================================================

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    """
    训练一个epoch
    完成前向传播、损失计算、反向传播与参数更新
    """
    model.train()
    total_loss = 0.0
    total_rpn_loss = 0.0
    total_class_loss = 0.0

    progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}', ncols=100)

    for batch_idx, (images, targets) in enumerate(progress_bar):
        # 将数据移动到设备
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 前向传播（训练模式返回损失字典）
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # 反向传播
        optimizer.zero_grad()
        losses.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 记录损失
        total_loss += losses.item()
        total_rpn_loss += loss_dict.get('loss_rpn_cls', torch.tensor(0)).item() + \
                          loss_dict.get('loss_rpn_box', torch.tensor(0)).item()
        total_class_loss += loss_dict.get('loss_classifier', torch.tensor(0)).item() + \
                            loss_dict.get('loss_box_reg', torch.tensor(0)).item()

        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{losses.item():.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
        })

    avg_loss = total_loss / len(data_loader)
    avg_rpn_loss = total_rpn_loss / len(data_loader)
    avg_class_loss = total_class_loss / len(data_loader)

    return avg_loss, avg_rpn_loss, avg_class_loss


@torch.no_grad()
def validate(model, data_loader, device, conf_threshold=0.5):
    """
    在验证集上评估模型
    统计检测数量、真实框数量，计算近似精度指标
    """
    model.eval()
    total_detections = 0
    total_groundtruths = 0

    class_stats = defaultdict(lambda: {'TP': 0, 'FP': 0, 'num_gt': 0})

    for images, targets in tqdm(data_loader, desc='Validating', ncols=100):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 推理
        outputs = model(images)

        for output, target in zip(outputs, targets):
            pred_boxes = output['boxes'].cpu().numpy()
            pred_labels = output['labels'].cpu().numpy()
            pred_scores = output['scores'].cpu().numpy()

            gt_boxes = target['boxes'].cpu().numpy()
            gt_labels = target['labels'].cpu().numpy()
            total_groundtruths += len(gt_boxes)

            # 使用置信度阈值筛选
            mask = pred_scores >= conf_threshold
            pred_boxes = pred_boxes[mask]
            pred_labels = pred_labels[mask]

            total_detections += len(pred_boxes)

            # 统计各类别预测和真实框数量
            for label in np.unique(gt_labels):
                class_stats[int(label)]['num_gt'] += np.sum(gt_labels == label)
            for label in np.unique(pred_labels):
                class_stats[int(label)]['FP'] += np.sum(pred_labels == label)

    # 打印评估结果
    print(f"\n验证结果: 检测到 {total_detections} 个目标, 真实框 {total_groundtruths} 个")
    det_gt_ratio = total_detections / total_groundtruths if total_groundtruths > 0 else 0
    return det_gt_ratio


# =====================================================================
# 五、主训练程序
# =====================================================================

def main():
    # -------------------- 配置参数 --------------------
    # 数据集路径（根据实际路径修改）
    # 用户数据集路径: F:\AI\VOC2012\VOCdevkit
    voc_root = 'F:/AI/VOC2012/VOCdevkit/VOC2012'

    # 训练超参数（CPU模式已调优：减小batch_size和epochs以适配CPU环境）
    batch_size = 2          # 批大小（CPU模式建议2，GPU可设为4）
    num_epochs = 2          # 训练轮数（CPU快速训练2轮，预计5-10分钟）
    learning_rate = 0.001   # 初始学习率
    weight_decay = 0.0005   # 权重衰减
    lr_step_size = 3        # 学习率衰减步长
    lr_gamma = 0.1          # 学习率衰减因子
    num_workers = 0         # 数据加载线程数（Windows必须设为0）
    conf_threshold = 0.5    # 验证时置信度阈值

    # -------------------- 设备设置 --------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    if device.type == 'cuda':
        print(f'GPU型号: {torch.cuda.get_device_name(0)}')
        print(f'GPU显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')

    # -------------------- 数据加载 --------------------
    print(f'\n正在加载数据集: {voc_root}')

    # 快速训练模式：采样100张训练图像，50张验证图像
    # 预计训练时间约5-10分钟（CPU模式）
    dataset_train = VOCDataset(voc_root, 'train', transforms=get_transform(train=True), max_samples=100)
    dataset_val = VOCDataset(voc_root, 'val', transforms=get_transform(train=False), max_samples=50)

    # Windows系统下num_workers建议设为0，避免多进程报错
    if os.name == 'nt':
        num_workers = 0

    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)), num_workers=num_workers
    )
    val_loader = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)), num_workers=num_workers
    )

    print(f'训练集大小: {len(dataset_train)} 张图像')
    print(f'验证集大小: {len(dataset_val)} 张图像')
    print(f'批大小: {batch_size}')
    print(f'每epoch迭代次数: {len(train_loader)}')

    # -------------------- 模型构建 --------------------
    # 类别数：20个VOC类别 + 1个背景类 = 21
    num_classes = 21
    model = get_model(num_classes)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'可训练参数数量: {total_params:,}')

    # -------------------- 优化器与调度器 --------------------
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    lr_scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    # -------------------- 训练循环 --------------------
    train_losses = []
    val_accuracies = []
    start_time = time.time()

    print(f'\n{"=" * 60}')
    print(f'开始训练: 共 {num_epochs} 个Epoch')
    print(f'{"=" * 60}')

    for epoch in range(1, num_epochs + 1):
        print(f'\n{"=" * 60}')
        print(f'Epoch {epoch}/{num_epochs}')
        print(f'学习率: {optimizer.param_groups[0]["lr"]:.6f}')
        print('=' * 60)

        # 训练
        train_loss, rpn_loss, class_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch
        )
        train_losses.append(train_loss)

        # 学习率调度
        lr_scheduler.step()

        # 验证
        val_acc = validate(model, val_loader, device, conf_threshold)
        val_accuracies.append(val_acc)

        # 打印epoch总结
        print(f'\nEpoch {epoch} 总结:')
        print(f'  训练损失: {train_loss:.4f} (RPN: {rpn_loss:.4f}, 分类回归: {class_loss:.4f})')
        print(f'  验证检测/真实比: {val_acc:.4f}')

        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_acc': val_acc,
            'num_classes': num_classes,
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
        print(f'  检查点已保存: checkpoint_epoch_{epoch}.pth')

    # -------------------- 训练结束 --------------------
    total_time = time.time() - start_time
    print(f'\n{"=" * 60}')
    print(f'训练完成！总耗时: {total_time / 60:.2f} 分钟')
    print(f'最终训练损失: {train_losses[-1]:.4f}')
    print(f'最终验证检测/真实比: {val_accuracies[-1]:.4f}')
    print(f'{"=" * 60}')

    # -------------------- 绘制训练曲线 --------------------
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, 'b-o', label='训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), val_accuracies, 'r-o', label='验证检测/真实比')
    plt.xlabel('Epoch')
    plt.ylabel('Ratio')
    plt.title('验证曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print('训练曲线已保存: training_curves.png')

    # -------------------- 保存最终模型 --------------------
    torch.save(model.state_dict(), 'faster_rcnn_final.pth')
    print('最终模型权重已保存: faster_rcnn_final.pth')


if __name__ == '__main__':
    main()
