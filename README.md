# Faster R-CNN + PASCAL VOC2012 目标检测实验

基于Torchvision官方 `fasterrcnn_resnet50_fpn` 模型，在PASCAL VOC2012数据集上完成目标检测的完整训练与推理流程。

---

## 项目结构

```
R-CNN/
├── VOCdevkit/                  # VOC2012数据集（需自行下载）
│   └── VOC2012/
│       ├── Annotations/        # XML标注文件
│       ├── ImageSets/Main/     # train.txt / val.txt
│       └── JPEGImages/         # 原始图像
├── voc_annotation.py           # 标注格式转换脚本（XML → TXT）
├── train.py                    # 模型训练主程序
├── inference.py                # 模型推理脚本（单图/批量）
├── faster_rcnn_final.pth       # 训练好的最终模型权重（训练后生成）
├── checkpoint_epoch_*.pth      # 各epoch检查点（训练后生成）
├── training_curves.png         # 训练曲线图（训练后生成）
└── detection_result.jpg        # 检测结果图（推理后生成）
```

---

## 环境要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| CPU | Intel Core i5 8代 | Intel Core i7 |
| 内存 | 8GB | 16GB+ |
| GPU | — | NVIDIA显卡，显存≥6GB |
| 硬盘 | 30GB | 50GB+ SSD |
| Python | 3.8 | 3.8~3.10 |
| PyTorch | 1.8+ | 2.0.1 |

---

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
conda create -n faster_rcnn python=3.8 -y
conda activate faster_rcnn

# 安装PyTorch（GPU版本）
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 安装PyTorch（CPU版本，无GPU时使用）
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install opencv-python matplotlib numpy pillow pycocotools tqdm tensorboard
```

### 2. 准备数据集

下载PASCAL VOC2012数据集（约2GB）：

- 官方地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
- 解压后确保目录结构为 `VOCdevkit/VOC2012/`

### 3. 标注格式转换

```bash
python voc_annotation.py
```

生成 `2012_train.txt` 和 `2012_val.txt`。

### 4. 模型训练

```bash
python train.py
```

训练参数可在 `train.py` 的 `main()` 函数中修改：
- `batch_size`：批大小（显存不足时减小）
- `num_epochs`：训练轮数
- `learning_rate`：初始学习率

### 5. 模型推理

```bash
# 单张图像检测
python inference.py --model faster_rcnn_final.pth --image test.jpg --conf 0.5

# 批量检测
python inference.py --model faster_rcnn_final.pth --dir ./test_images/ --output ./results/
```

---

## 命令行参数说明

### train.py

所有参数在代码中直接配置，无需命令行参数。

### inference.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型权重路径 | `faster_rcnn_final.pth` |
| `--image` | 待检测图像路径 | `./test.jpg` |
| `--dir` | 批量检测图像文件夹 | `None` |
| `--conf` | 置信度阈值 | `0.5` |
| `--output` | 结果保存路径 | `detection_result.jpg` |

---

## VOC2012数据集20个类别

| 类别 | 中文 | 类别 | 中文 |
|------|------|------|------|
| aeroplane | 飞机 | person | 人 |
| bicycle | 自行车 | pottedplant | 盆栽 |
| bird | 鸟 | sheep | 羊 |
| boat | 船 | sofa | 沙发 |
| bottle | 瓶子 | train | 火车 |
| bus | 公共汽车 | tvmonitor | 电视 |
| car | 汽车 | cat | 猫 |
| chair | 椅子 | dog | 狗 |
| cow | 牛 | horse | 马 |
| diningtable | 餐桌 | motorbike | 摩托车 |

---

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| CUDA out of memory | 减小 `batch_size`（4→2→1），或降低图像缩放尺寸 |
| 训练损失不收敛 | 降低学习率、检查数据标注、增加训练轮数 |
| 预训练权重下载失败 | 手动下载到 `~/.cache/torch/hub/checkpoints/` |
| Windows多进程报错 | 将 `num_workers` 设为 0 |
| 推理速度慢 | 使用GPU推理，或开启FP16半精度加速 |
