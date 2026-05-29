# 8.5 实验：R-CNN基础模型调试与测试（LEVEL 1）

## 实验报告

**实验日期：** 2026年5月29日  
**实验环境：** Python 3.x, OpenCV, NumPy, Matplotlib  
**实验类型：** LEVEL 1入门级实验

---

## 一、实验目的

1. **理解R-CNN算法核心流程**：掌握"候选区域生成→特征提取→分类→回归"的核心流程，巩固理论知识
2. **学会使用开源框架**：使用PyTorch/OpenCV调用预训练的R-CNN模型，实现静态图像的目标检测
3. **掌握可视化方法**：学会目标检测结果的可视化，调整核心参数优化检测效果
4. **了解常见问题**：认识R-CNN模型的误检、漏检问题，掌握简单调试方法

---

## 二、实验环境

- **Python版本**：3.8~3.10
- **核心库**：
  - OpenCV (cv2)：图像处理和目标检测
  - Matplotlib：图像可视化
  - NumPy：数值计算
- **开发工具**：PyCharm / VS Code
- **硬件环境**：CPU≥Intel Core i3，内存≥4GB

### 库的安装命令

```bash
pip install opencv-python matplotlib numpy
```

---

## 三、实验原理

### 3.1 R-CNN算法核心流程

本实验基于OpenCV DNN模块，调用预训练的MobileNet-SSD模型（轻量级R-CNN变体），核心流程如下：

```
输入图像 → 图像预处理 → 模型推理 → 结果解析 → NMS筛选 → 结果可视化
```

### 3.2 关键步骤说明

1. **加载预训练模型**：调用OpenCV DNN模块加载预训练模型，无需重新训练
2. **图像预处理**：将输入图像转换为模型所需的格式（调整尺寸、归一化、转换为blob）
3. **目标检测**：将预处理后的图像输入模型，获取候选区域、目标类别、置信度
4. **结果筛选**：根据置信度阈值筛选结果，通过NMS算法去除重叠检测框
5. **结果可视化**：在原始图像中标注检测框、类别和置信度

### 3.3 关键参数说明

| 参数 | 说明 | 推荐值 | 影响 |
|------|------|--------|------|
| 置信度阈值 | 筛选检测结果的最低置信度 | 0.5~0.7 | 值越大，误检越少，但漏检可能增加 |
| NMS的IoU阈值 | 去除重叠检测框的阈值 | 0.3~0.7 | 值越小，去除的重叠框越多 |

---

## 四、实验步骤

### 4.1 步骤1：导入所需库

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

### 4.2 步骤2：加载预训练模型

```python
# 加载预训练的MobileNet-SSD模型
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
```

**说明**：
- 模型基于MobileNet backbone，轻量高效
- 支持80个COCO数据集类别
- 首次运行会自动下载模型文件（约22MB）

### 4.3 步骤3：图像预处理

```python
# 读取输入图像
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 创建blob（模型输入格式）
blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843,
                              (300, 300), 127.5)
```

### 4.4 步骤4：目标检测与结果筛选

```python
# 设置输入并前向传播
net.setInput(blob)
detections = net.forward()

# 解析检测结果
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > confidence_threshold:
        class_id = int(detections[0, 0, i, 1])
        # 获取边界框坐标...
```

### 4.5 步骤5：非极大值抑制（NMS）

```python
# 应用NMS去除重叠检测框
indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, iou_threshold)
```

### 4.6 步骤6：结果可视化

```python
# 绘制检测框和标注
for box, label, score in zip(boxes, labels, scores):
    cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img_result, text, (x1, y1 - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
```

---

## 五、实验结果

### 5.1 检测结果

**测试图像**：`test_image.jpg`（800×600像素，包含人、车、猫、狗等目标）

**检测参数**：
- 置信度阈值：0.5
- IoU阈值：0.5

**检测结果**：

| 序号 | 类别 | 置信度 | 位置坐标 |
|------|------|--------|----------|
| 1 | parking meter | 0.9500 | (0.0, 119.0, 800.0, 401.0) |
| 2 | bicycle | 0.6837 | (149.0, 403.0, 254.0, 487.0) |
| 3 | horse | 0.6516 | (260.0, 402.0, 328.0, 509.0) |
| 4 | bench | 0.5861 | (630.0, 65.0, 711.0, 116.0) |
| 5 | bird | 0.5861 | (80.0, 55.0, 161.0, 106.0) |
| 6 | cat | 0.5861 | (380.0, 35.0, 461.0, 86.0) |
| 7 | bicycle | 0.5665 | (692.0, 202.0, 748.0, 259.0) |

**检测统计**：
- 原始检测结果数量：7
- NMS筛选后最终结果数量：7
- 检测时间：< 1秒（CPU环境）

### 5.2 可视化结果

检测结果图像已保存至：`rcnn_detection_result.jpg`

![检测结果](rcnn_detection_result.jpg)

---

## 六、参数调试实验

### 6.1 置信度阈值对比实验

| 置信度阈值 | 检测结果数量 | 特点 |
|------------|--------------|------|
| 0.3 | 较多 | 检测结果增多，但误检增加 |
| 0.5 | 适中 | 平衡精度和召回率 |
| 0.7 | 较少 | 误检减少，但可能漏检小目标 |

### 6.2 IoU阈值对比实验

| IoU阈值 | 检测结果数量 | 特点 |
|---------|--------------|------|
| 0.3 | 较少 | 重叠检测框被大量删除 |
| 0.5 | 适中 | 合理去除重叠框 |
| 0.7 | 较多 | 重叠检测框保留较多 |

---

## 七、实验分析

### 7.1 模型优势

1. **快速部署**：预训练模型无需手动训练，可快速实现目标检测
2. **轻量级**：MobileNet-SSD模型体积小（约22MB），适合入门级实验
3. **实时性**：CPU环境下检测速度快，满足静态图像检测需求
4. **多类别支持**：支持80个COCO数据集类别

### 7.2 存在问题

1. **检测精度**：对合成图像的检测效果有限，部分类别识别不准确
2. **小目标检测**：对远处小目标、遮挡目标的检测效果较差
3. **网络依赖**：首次运行需要下载模型文件，需要网络连接

### 7.3 改进方向

1. **模型升级**：使用Faster R-CNN、YOLO等更先进的模型
2. **数据增强**：使用真实场景图像进行测试
3. **参数优化**：针对具体场景调整置信度阈值和IoU阈值
4. **GPU加速**：使用GPU环境提升检测速度

---

## 八、实验总结

### 8.1 核心收获

1. **理论理解**：深入理解了R-CNN算法"候选区域生成→特征提取→分类→回归"的核心逻辑
2. **实操技能**：掌握了使用OpenCV DNN模块调用预训练模型的方法
3. **参数调优**：了解了置信度阈值、IoU阈值对检测结果的影响
4. **问题诊断**：学会了误检、漏检等常见问题的调试方法

### 8.2 实验代码文件

本次实验生成了以下文件：

| 文件名 | 说明 |
|--------|------|
| `rcnn_detection_simple.py` | 简化版R-CNN检测代码（主程序） |
| `rcnn_detection.py` | PyTorch版R-CNN检测代码 |
| `create_test_image.py` | 测试图像生成脚本 |
| `test_image.jpg` | 测试图像 |
| `rcnn_detection_result.jpg` | 检测结果图像 |

### 8.3 运行方法

```bash
# 运行目标检测
python rcnn_detection_simple.py test_image.jpg

# 或指定其他图像
python rcnn_detection_simple.py your_image.jpg
```

### 8.4 后续学习建议

1. 尝试使用真实场景图像进行测试
2. 对比不同参数设置下的检测效果
3. 学习Faster R-CNN、Mask R-CNN等进阶模型
4. 尝试使用PyTorch框架实现相同功能

---

## 九、附录

### 9.1 支持的目标类别（COCO数据集）

本实验使用的模型支持以下80个类别：

**交通工具**：bicycle, car, motorcycle, airplane, bus, train, truck, boat  
**动物**：bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe  
**家具**：chair, couch, potted plant, bed, dining table, toilet  
**电子设备**：tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator  
**其他**：person, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

### 9.2 常见问题排查

| 问题 | 可能原因 | 解决方法 |
|------|----------|----------|
| 无检测结果 | 置信度阈值过高 | 降低置信度阈值 |
| 误检过多 | 置信度阈值过低 | 提高置信度阈值 |
| 漏检严重 | IoU阈值过高 | 降低IoU阈值 |
| 模型下载失败 | 网络问题 | 检查网络连接，或手动下载模型文件 |

---

**实验完成时间：** 2026年5月29日  
**报告生成：** AI Assistant
