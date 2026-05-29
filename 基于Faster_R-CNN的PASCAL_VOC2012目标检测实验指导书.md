**基于Faster R-CNN的PASCAL VOC2012目标检测实验指导书**

**一、Faster R-CNN算法概述**

Faster R-CNN是经典的**两阶段（two-stage）**目标检测框架，区别于YOLO等单阶段检测算法，其通过分阶段完成目标检测任务，兼顾检测精度与推理效率。

该算法的核心创新为提出**区域建议网络（RPN, Region Proposal Network）**，彻底替代了传统检测算法中耗时的选择性搜索（Selective Search）手工候选区域生成方法，实现了真正的端到端可学习训练，是R-CNN系列算法的里程碑式改进。

Faster R-CNN整体遵循三段式检测范式：**特征提取 → 区域建议生成 → 候选框分类与精调**。第一阶段由RPN从共享特征图中快速筛选高质量目标候选区域，第二阶段对候选区域完成类别分类和边界框回归优化。

**补充说明**：本次实验基于Torchvision官方封装的fasterrcnn_resnet50_fpn模型开发，无需手动搭建RPN、ROI Pooling等核心模块，可快速完成完整的模型训练与推理流程。

**二、实验环境搭建**

**2.1 硬件要求**

- CPU：Intel Core i5 8代及以上（4核8线程）
- 内存：最低8GB，推荐16GB及以上
- GPU（强烈建议）：NVIDIA独立显卡，显存≥6GB（GTX1060、RTX2060、RTX3060等）
- 硬盘空间：≥30GB（用于存放数据集、模型权重、训练缓存）

**2.2 软件环境**

- 操作系统：Windows10/11、Ubuntu18.04/20.04/22.04
- Python版本：3.8~3.10（推荐3.8）
- PyTorch版本：1.8及以上

**2.3 环境安装步骤**

**步骤1：安装Anaconda**

从Anaconda官网下载最新版本安装包，安装时勾选“Add Anaconda to my PATH environment variable”，自动配置系统环境变量。

**步骤2：创建专属虚拟环境**

打开Anaconda Prompt（Windows）或终端（Linux/Mac），执行以下命令创建并激活虚拟环境，避免项目库版本冲突：

```python
# 创建名为faster_rcnn的虚拟环境，指定Python3.8版本
conda create -n faster_rcnn python=3.8 -y

# 激活虚拟环境
conda activate faster_rcnn
```

**步骤3：安装PyTorch与Torchvision**

根据设备是否搭载NVIDIA GPU，选择对应安装命令，本次实验指定版本为torch2.0.1、torchvision0.15.2。

**GPU版本（CUDA11.8）**：

```python
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

**CPU版本（无GPU）**：

```python
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
```

**环境验证命令**：

```python
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

输出True代表GPU可用，False代表当前为CPU运行模式。

**步骤4：安装依赖库**

```python
pip install opencv-python matplotlib numpy pillow pycocotools tqdm tensorboard
```

依赖库功能说明：

- opencv-python：图像读取与预处理
- matplotlib：训练曲线、检测结果可视化
- numpy：数值计算
- pillow：图像格式处理
- pycocotools：模型精度评估工具
- tqdm：训练进度条显示
- tensorboard：训练日志可视化

**三、PASCAL VOC2012数据集准备**

**3.1 数据集简介**

PASCAL VOC2012是目标检测经典基准数据集，包含**20个目标类别**：aeroplane、bicycle、bird、boat、bottle、bus、car、cat、chair、cow、diningtable、dog、horse、motorbike、person、pottedplant、sheep、sofa、train、tvmonitor。数据集包含训练集5717张、验证集5823张图像，标注格式规范，适用于通用目标检测模型训练。

**3.2 数据集下载**

**方法一：官方链接下载**

下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

文件大小约2GB，下载完成后得到VOCtrainval_11-May-2012.tar压缩包。

**方法二：Linux/Mac终端命令下载**

```python
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```

**方法三：备用方案**：百度网盘第三方资源下载。

**3.3 数据集解压与目录结构**

Windows使用WinRAR/7-Zip解压，Linux/Mac执行解压命令：

```python
tar -xvf VOCtrainval_11-May-2012.tar
```

解压后生成VOCdevkit文件夹，标准目录结构如下：

```python
VOCdevkit/
└── VOC2012/
    ├── Annotations/      # 图像XML标注文件（目标框、类别信息）
    ├── ImageSets/        # 数据集划分文件
    │   └── Main/         # train.txt、val.txt、trainval.txt
    ├── JPEGImages/       # 原始训练图像（JPG格式）
    ├── SegmentationClass/# 语义分割标注（目标检测无需使用）
    └── SegmentationObject/# 实例分割标注（目标检测无需使用）
```

核心文件夹说明：

- Annotations：每张图像对应一个XML文件，存储目标框坐标（xmin、ymin、xmax、ymax）和类别名称
- ImageSets/Main：划分训练集、验证集图像名称列表
- JPEGImages：原始图像文件，与标注文件一一对应

**3.4 标注格式转换（XML转TXT）**

为适配模型加载需求，将原生XML标注文件转换为TXT格式。在项目根目录创建voc_annotation.py文件，写入以下代码并运行。

**转换脚本代码**

```python
import xml.etree.ElementTree as ET
import os

# 设置路径
VOC_root = "./VOCdevkit/VOC2012"
sets = [('2012', 'train'), ('2012', 'val')]  # 训练和验证集
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
           "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def convert_annotation(year, image_id, list_file):
    in_file = open(f'{VOC_root}/Annotations/{image_id}.xml')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = int(obj.find('difficult').text)
        cls_name = obj.find('name').text
        if cls_name not in classes or difficult == 1:
            continue
        cls_id = classes.index(cls_name)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

for year, image_set in sets:
    image_ids = open(f'{VOC_root}/ImageSets/Main/{image_set}.txt').read().strip().split()
    list_file = open(f'{year}_{image_set}.txt', 'w')
    for image_id in image_ids:
        list_file.write(f'{VOC_root}/JPEGImages/{image_id}.jpg')
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()
    print(f'生成 {year}_{image_set}.txt 完成，共{len(image_ids)}张图像')
```

**脚本执行命令**

```python
python voc_annotation.py
```

执行后生成2012_train.txt、2012_val.txt文件，格式为：图像路径 x1,y1,x2,y2,label_id。

**四、模型训练完整代码实现**

**4.1 库文件导入**

```python
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
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
```

**4.2 自定义VOC数据集加载类**

解析VOC数据集标注，输出模型所需的图像张量与标注字典，是数据加载的核心模块。

```python
class VOCDataset(Dataset):
    """PASCAL VOC数据集加载器"""
    def __init__(self, root, image_set, transforms=None):
        self.root = root
        self.transforms = transforms
        self.image_set = image_set

        # 类别映射：20个VOC类别 + 背景类
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train", "tvmonitor"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # 读取图像名称列表
        image_sets_path = os.path.join(root, "ImageSets", "Main", f"{image_set}.txt")
        with open(image_sets_path) as f:
            self.image_ids = [x.strip() for x in f.readlines()]

        self.images_dir = os.path.join(root, "JPEGImages")
        self.annotations_dir = os.path.join(root, "Annotations")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """返回图像和对应的标注信息"""
        image_id = self.image_ids[idx]

        # 读取图像
        img_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 解析XML标注文件，提取目标框和类别
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

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)
        except Exception as e:
            print(f"解析XML文件失败: {xml_path}, 错误: {e}")
            return self.__getitem__((idx + 1) % len(self.image_ids))

        # 如果没有目标，跳过该图像
        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self.image_ids))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # 构建目标字典
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        # 应用图像变换
        if self.transforms:
            image, target = self.transforms(image, target)

        # 将图像转换为张量（H×W×C → C×H×W）
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, target

```

**4.3 图像预处理变换函数**

实现图像缩放、归一化、张量转换，适配预训练模型的输入规范。

```python
class Compose:
    """组合多个变换"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    """将图像转换为张量"""
    def __call__(self, image, target):
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image, target

class Normalize:
    """归一化（使用ImageNet统计值）"""
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        if isinstance(image, torch.Tensor):
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)
        return image, target

class Resize:
    """调整图像和边界框尺寸"""
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        original_height, original_width = image.shape[:2]
        image = cv2.resize(image, (self.size[1], self.size[0]))

        if target is not None and "boxes" in target:
            boxes = target["boxes"].clone()
            # 调整边界框坐标
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * (self.size[1] / original_width)
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * (self.size[0] / original_height)
            target["boxes"] = boxes

        return image, target

def get_transform(train):
    """获取图像变换"""
    transforms = []
    # 训练时将所有图像缩放到统一尺寸
    transforms.append(Resize((800, 800)))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return Compose(transforms)
```

**4.4 Faster R-CNN模型构建**

加载预训练ResNet50-FPN主干网络，替换分类头适配VOC21类（20类别+背景）。

```python
def get_model(num_classes):
    """构建Faster R-CNN模型"""
    # 加载预训练的Faster R-CNN模型（ResNet50-FPN主干）
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # 获取输入特征维度
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 替换输出层，适配自定义类别数
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
```

**4.5 单轮训练函数**

完成模型前向传播、损失计算、反向传播与参数更新，统计各类损失值。

```python
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_rpn_loss = 0.0
    total_class_loss = 0.0

    # tqdm显示进度条
    progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}', ncols=100)

    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 前向传播，计算损失
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # 反向传播
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # 记录损失
        total_loss += losses.item()
        total_rpn_loss += loss_dict.get('loss_rpn_cls', 0).item() + loss_dict.get('loss_rpn_box', 0).item()
        total_class_loss += loss_dict.get('loss_classifier', 0).item() + loss_dict.get('loss_box_reg', 0).item()

        # 更新进度条显示
        progress_bar.set_postfix({
            'loss': f'{losses.item():.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
        })

    avg_loss = total_loss / len(data_loader)
    avg_rpn_loss = total_rpn_loss / len(data_loader)
    avg_class_loss = total_class_loss / len(data_loader)

    return avg_loss, avg_rpn_loss, avg_class_loss
```

**4.6 模型验证函数**

在验证集上推理，统计目标检测数量、真实框数量，评估模型检测效果。

```python
@torch.no_grad()
def validate(model, data_loader, device):
    """在验证集上评估模型"""
    model.eval()
    total_detections = 0
    total_groundtruths = 0

    # 统计每个类别的TP和FP
    class_stats = defaultdict(lambda: {'TP': 0, 'FP': 0, 'num_gt': 0})

    for images, targets in tqdm(data_loader, desc='Validating', ncols=100):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 推理
        outputs = model(images)

        # 统计检测结果
        for output, target in zip(outputs, targets):
            pred_boxes = output['boxes'].cpu().numpy()
            pred_labels = output['labels'].cpu().numpy()
            pred_scores = output['scores'].cpu().numpy()

            gt_boxes = target['boxes'].cpu().numpy()
            gt_labels = target['labels'].cpu().numpy()
            total_groundtruths += len(gt_boxes)

            # 使用置信度阈值0.5筛选预测结果
            mask = pred_scores >= 0.5
            pred_boxes = pred_boxes[mask]
            pred_labels = pred_labels[mask]
            pred_scores = pred_scores[mask]

            total_detections += len(pred_boxes)

            # 简单评估：记录每个类别的预测和真实框数量
            for label in np.unique(gt_labels):
                class_stats[int(label)]['num_gt'] += np.sum(gt_labels == label)
            for label in np.unique(pred_labels):
                class_stats[int(label)]['FP'] += np.sum(pred_labels == label)

    # 打印评估结果
    print(f"\n验证结果: 检测到 {total_detections} 个目标, 真实框 {total_groundtruths} 个")
    return total_detections / total_groundtruths if total_groundtruths > 0 else 0
```

**4.7 主训练程序**

整合数据加载、模型训练、验证、日志记录、权重保存、曲线可视化全流程。

```python
def main():
    # 设置设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'使用设备: {device}')

    # 数据集路径
    voc_root = './VOCdevkit/VOC2012'

    # 构建数据集和数据加载器
    dataset_train = VOCDataset(voc_root, 'train', transforms=get_transform(train=True))
    dataset_val = VOCDataset(voc_root, 'val', transforms=get_transform(train=False))

    # batch_size根据GPU显存调整，显存不足时减小
    train_loader = DataLoader(dataset_train, batch_size=4, shuffle=True,
                              collate_fn=lambda x: tuple(zip(*x)), num_workers=4)
    val_loader = DataLoader(dataset_val, batch_size=4, shuffle=False,
                            collate_fn=lambda x: tuple(zip(*x)), num_workers=4)

    print(f'训练集大小: {len(dataset_train)} 张图像')
    print(f'验证集大小: {len(dataset_val)} 张图像')

    # 类别数：20个VOC类别 + 1个背景类 = 21
    num_classes = 21
    model = get_model(num_classes)
    model.to(device)

    # 打印模型可训练参数数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'可训练参数数量: {total_params:,}')

    # 配置优化器（Adam或SGD）
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    # 学习率调度器：每3个epoch将学习率乘以0.1
    lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    # 训练参数
    num_epochs = 10  # 可根据需要调整

    # 记录训练历史
    train_losses = []
    val_accuracies = []

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch}/{num_epochs}')
        print(f'学习率: {optimizer.param_groups[0]["lr"]:.6f}')
        print('='*60)

        # 训练一个epoch
        train_loss, rpn_loss, class_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        train_losses.append(train_loss)

        # 学习率调度
        lr_scheduler.step()

        # 验证
        val_acc = validate(model, val_loader, device)
        val_accuracies.append(val_acc)

        # 打印epoch总结
        print(f'\nEpoch {epoch} 总结:')
        print(f'  训练损失: {train_loss:.4f} (RPN: {rpn_loss:.4f}, 分类回归: {class_loss:.4f})')
        print(f'  验证F1近似值: {val_acc:.4f}')

        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_acc': val_acc,
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
        print(f'  检查点已保存: checkpoint_epoch_{epoch}.pth')

    # 训练结束
    total_time = time.time() - start_time
    print(f'\n{"="*60}')
    print(f'训练完成！总耗时: {total_time/60:.2f} 分钟')
    print(f'最终训练损失: {train_losses[-1]:.4f}')
    print(f'最终验证F1: {val_accuracies[-1]:.4f}')

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, 'b-', label='训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失曲线')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), val_accuracies, 'r-', label='验证F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.title('验证曲线')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()

    # 保存最终模型
    torch.save(model.state_dict(), 'faster_rcnn_final.pth')
    print('最终模型已保存: faster_rcnn_final.pth')


if __name__ == '__main__':
    main()
```

**五、模型推理代码（单图测试）**

加载训练完成的模型权重，对自定义图像进行目标检测，可视化检测框、类别与置信度。

```python
def inference_image(model_path, image_path, device, confidence_threshold=0.5):
    """使用训练好的模型对单张图像进行推理"""
    # 构建模型并加载权重
    num_classes = 21
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # VOC类别名称（用于显示标签）
    class_names = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant",
                   "sheep", "sofa", "train", "tvmonitor"]

    # 读取和预处理图像
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width = image.shape[:2]

    # 缩放到模型输入尺寸
    resized = cv2.resize(image_rgb, (800, 800))
    image_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    image_tensor = (image_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        predictions = model(image_tensor)

    # 解析结果
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()

    # 筛选
    mask = pred_scores >= confidence_threshold
    pred_boxes = pred_boxes[mask]
    pred_labels = pred_labels[mask]
    pred_scores = pred_scores[mask]

    # 在图像上绘制检测框
    result_img = image.copy()
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        # 将800×800尺寸的坐标映射回原始图像尺寸
        x1 = int(box[0] * original_width / 800)
        y1 = int(box[1] * original_height / 800)
        x2 = int(box[2] * original_width / 800)
        y2 = int(box[3] * original_height / 800)

        class_name = class_names[label] if label < len(class_names) else f"Class_{label}"

        # 绘制矩形框
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制标签
        label_text = f"{class_name}: {score:.2f}"
        cv2.putText(result_img, label_text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示结果
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Faster R-CNN 检测结果 (共 {len(pred_boxes)} 个目标)")
    plt.axis('off')
    plt.savefig("detection_result.jpg", dpi=150, bbox_inches='tight')
    plt.show()

    print(f"检测到 {len(pred_boxes)} 个目标:")
    for i, (box, label, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
        class_name = class_names[label] if label < len(class_names) else f"Class_{label}"
        print(f"  {i+1}. {class_name}: 置信度={score:.3f}, 框坐标=({int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])})")

    return result_img


if __name__ == '__main__':
    # 使用示例
    model_path = 'faster_rcnn_final.pth'      # 训练好的模型权重
    image_path = './test.jpg'                  # 测试图像路径
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    result = inference_image(model_path, image_path, device, confidence_threshold=0.5)
```

**六、实验启动与训练输出**

**6.1 训练启动命令**

将所有训练代码保存为train.py，终端执行：

```python
python train.py
```

**6.2 典型训练输出**

```python
使用设备: cuda
训练集大小: 5717 张图像
验证集大小: 5823 张图像
可训练参数数量: 41,234,821

============================================================
Epoch 1/10
学习率: 0.001000
============================================================
Epoch 1: 100%|██████████| 1430/1430 [12:34<00:00,  1.89it/s, loss=0.5823, avg_loss=0.7534]

Epoch 1 总结:
  训练损失: 0.7534 (RPN: 0.4215, 分类回归: 0.3319)
  验证F1近似值: 0.6823
  检查点已保存: checkpoint_epoch_1.pth

============================================================
训练完成！总耗时: 125.34 分钟
最终训练损失: 0.2134
最终验证F1: 0.7821
最终模型已保存: faster_rcnn_final.pth
```

**七、常见问题排查**

**7.1 CUDA out of memory（显存溢出）**

解决方案：减小batch_size（4→2/1）、降低图像缩放尺寸（800→640）、切换CPU训练。

**7.2 训练损失不收敛**

解决方案：降低初始学习率、检查数据集标注是否正常、替换SGD优化器、增加训练轮数。

**7.3 预训练权重下载失败**

解决方案：手动下载官方预训练权重，放置到~/.cache/torch/hub/checkpoints/目录，或更换国内镜像源。

**7.4 模型推理速度过慢**

解决方案：使用GPU推理、开启FP16半精度加速、替换轻量化主干网络。

**八、核心概念总结**

| 核心概念 | 详细说明 |
| --- | --- |
| RPN | 区域建议网络，在特征图上滑动锚点窗口，快速生成目标候选区域，替代传统选择性搜索 |
| Anchor（锚点） | 预定义多尺度、多长宽比的边界框模板，适配图像中不同大小、形态的目标 |
| ROI Pooling | 将不同尺寸的候选区域特征统一缩放到固定尺寸，适配全连接层输入要求 |
| NMS非极大值抑制 | 去除重叠度高的重复检测框，保留最优检测结果 |
| 多任务损失 | 包含RPN分类损失、RPN回归损失、目标分类损失、边界框回归损失，联合优化模型 |
| 迁移学习 | 复用ImageNet预训练主干网络权重，仅训练RPN和检测头，加速收敛、防止过拟合 |
| TP/FP/FN | TP正确检测、FP误检、FN漏检，是模型精度评估的基础指标 |
| IoU交并比 | 预测框与真实框的重叠占比，IoU>=0.5通常判定为检测正确 |
| mAP | 多类别平均精度均值，是目标检测领域衡量模型综合性能的核心指标 |

> （注：文档部分内容可能由 AI 生成）

