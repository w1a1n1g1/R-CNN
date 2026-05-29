"""
Faster R-CNN 目标检测 Web 应用
带美化UI的图片上传与目标检测工具
运行命令: streamlit run app.py
"""

import streamlit as st
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
from PIL import Image
import io
import time
import os

# ===================== 页面配置 =====================

st.set_page_config(
    page_title="Faster R-CNN 目标检测",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===================== 自定义CSS样式 =====================

st.markdown("""
<style>
    /* 全局字体和颜色 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* 主标题样式 */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    /* 副标题样式 */
    .subtitle {
        font-size: 1.1rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* 上传区域样式 */
    .upload-box {
        border: 2px dashed #667eea;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(102,126,234,0.05) 0%, rgba(118,75,162,0.05) 100%);
        transition: all 0.3s ease;
    }

    /* 检测结果卡片样式 */
    .result-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }

    /* 统计数字样式 */
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
    }

    .stat-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* 类别标签样式 */
    .class-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.2rem;
    }

    /* 进度条样式 */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }

    /* 侧边栏样式 */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }

    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.6);
    }

    /* 图片容器样式 */
    .image-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }

    /* 警告框样式 */
    .stAlert {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ===================== VOC类别定义 =====================

VOC_CLASSES = [
    "背景", "飞机", "自行车", "鸟", "船", "瓶子", "公交车",
    "汽车", "猫", "椅子", "牛", "餐桌", "狗", "马", "摩托车",
    "人", "盆栽", "羊", "沙发", "火车", "显示器"
]

# 类别颜色映射
CLASS_COLORS = {
    0: (128, 128, 128),   # 背景
    1: (255, 0, 0),       # 飞机
    2: (0, 255, 0),       # 自行车
    3: (0, 0, 255),       # 鸟
    4: (255, 255, 0),     # 船
    5: (255, 0, 255),     # 瓶子
    6: (0, 255, 255),     # 公交车
    7: (128, 0, 0),       # 汽车
    8: (0, 128, 0),       # 猫
    9: (0, 0, 128),       # 椅子
    10: (128, 128, 0),   # 牛
    11: (128, 0, 128),   # 餐桌
    12: (0, 128, 128),   # 狗
    13: (255, 128, 0),   # 马
    14: (255, 0, 128),   # 摩托车
    15: (128, 255, 0),   # 人
    16: (0, 255, 128),   # 盆栽
    17: (128, 0, 255),   # 羊
    18: (0, 128, 255),   # 沙发
    19: (255, 128, 128), # 火车
    20: (128, 255, 128), # 显示器
}

# ===================== 模型加载 =====================

@st.cache_resource
def load_model():
    """加载Faster R-CNN模型"""
    try:
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 21)

        # 加载训练好的权重
        model_path = "faster_rcnn_final.pth"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            return model, "训练模型"
        else:
            return model, "预训练模型(COCO)"
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None, None

# ===================== 检测函数 =====================

def detect_objects(image, model, conf_threshold=0.5):
    """执行目标检测"""
    if model is None:
        return None, None, None

    # 图像预处理
    img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    original_height, original_width = img_rgb.shape[:2]

    # 缩放到模型输入尺寸
    min_size, max_size = 800, 1333
    scale = min(min_size / min(original_height, original_width),
                max_size / max(original_height, original_width))
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized = cv2.resize(img_rgb, (new_width, new_height))

    # 转换为张量
    image_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    image_tensor = image_tensor.unsqueeze(0)

    # 推理
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)

    # 解析结果
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()

    # 筛选
    mask = pred_scores >= conf_threshold
    pred_boxes = pred_boxes[mask]
    pred_labels = pred_labels[mask]
    pred_scores = pred_scores[mask]

    # 坐标映射回原图
    pred_boxes[:, [0, 2]] = pred_boxes[:, [0, 2]] / scale
    pred_boxes[:, [1, 3]] = pred_boxes[:, [1, 3]] / scale

    return pred_boxes, pred_labels, pred_scores

def draw_detections(image, boxes, labels, scores):
    """绘制检测结果"""
    result_img = image.copy()

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        label_idx = int(label)

        if label_idx >= len(VOC_CLASSES):
            continue

        class_name = VOC_CLASSES[label_idx]
        color = CLASS_COLORS.get(label_idx, (255, 0, 0))

        # 绘制边界框
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

        # 绘制标签背景
        text = f"{class_name} {score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result_img, (x1, y1 - text_h - 15), (x1 + text_w, y1), color, -1)

        # 绘制标签文字
        cv2.putText(result_img, text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return result_img

# ===================== 主界面 =====================

def main():
    # 标题
    st.markdown('<p class="main-title">🎯 Faster R-CNN 目标检测</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">基于深度学习的高精度目标检测工具 | 支持20类常见物体识别</p>', unsafe_allow_html=True)

    # 加载模型
    with st.spinner('正在加载模型，请稍候...'):
        model, model_type = load_model()

    # 侧边栏设置
    with st.sidebar:
        st.header("⚙️ 检测设置")

        conf_threshold = st.slider(
            "置信度阈值",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="只显示置信度高于此值的检测结果"
        )

        st.divider()

        st.subheader("📊 支持的类别")
        st.markdown("""
        - 🚗 **交通工具**: 汽车、摩托车、公交车、火车
        - 🐾 **动物**: 猫、狗、鸟、马、牛、羊
        - 🧑 **人物**: 人
        - 🏠 **家具**: 椅子、沙发、餐桌、盆栽
        - 📺 **电子**: 显示器、瓶子
        - ✈️ **其他**: 飞机、船、自行车
        """)

        st.divider()

        st.caption(f"模型类型: {model_type if model_type else '未加载'}")
        st.caption("训练数据: PASCAL VOC2012")

    # 主内容区
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📤 上传图片")

        uploaded_file = st.file_uploader(
            "支持 JPG, PNG 格式",
            type=['jpg', 'jpeg', 'png'],
            help="选择一张包含常见物体的图片进行检测"
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="原始图片", use_container_width=True)

            # 检测按钮
            if st.button("🚀 开始检测", use_container_width=True):
                if model is not None:
                    with st.spinner('正在进行目标检测...'):
                        start_time = time.time()

                        boxes, labels, scores = detect_objects(image, model, conf_threshold)

                        if boxes is not None and len(boxes) > 0:
                            result_img = draw_detections(np.array(image), boxes, labels, scores)

                            # 保存结果
                            result_bytes = cv2.imencode('.png', cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))[1].tobytes()
                            st.session_state['result_image'] = result_img
                            st.session_state['detection_data'] = {
                                'boxes': boxes,
                                'labels': labels,
                                'scores': scores
                            }
                            st.session_state['detection_time'] = time.time() - start_time

                            st.success(f"✅ 检测完成！发现 {len(boxes)} 个目标，耗时 {time.time() - start_time:.2f} 秒")
                        else:
                            st.warning("未检测到任何目标，请尝试降低置信度阈值或更换图片")

    with col2:
        st.markdown("### 📊 检测结果")

        if 'result_image' in st.session_state:
            result_img = st.session_state['result_image']
            detection_data = st.session_state['detection_data']
            detection_time = st.session_state['detection_time']

            # 统计信息
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(f'<div class="stat-number">{len(detection_data["boxes"])}</div>', unsafe_allow_html=True)
                st.markdown('<div class="stat-label">检测数量</div>', unsafe_allow_html=True)
            with col_b:
                unique_classes = len(set(detection_data['labels']))
                st.markdown(f'<div class="stat-number">{unique_classes}</div>', unsafe_allow_html=True)
                st.markdown('<div class="stat-label">类别数</div>', unsafe_allow_html=True)
            with col_c:
                st.markdown(f'<div class="stat-number">{detection_time:.2f}s</div>', unsafe_allow_html=True)
                st.markdown('<div class="stat-label">检测耗时</div>', unsafe_allow_html=True)

            st.divider()

            # 显示结果图片
            st.image(result_img, caption="检测结果", use_container_width=True)

            # 检测详情
            with st.expander("📋 查看详细检测结果"):
                for i, (box, label, score) in enumerate(zip(
                    detection_data['boxes'],
                    detection_data['labels'],
                    detection_data['scores']
                )):
                    label_idx = int(label)
                    if label_idx < len(VOC_CLASSES):
                        st.markdown(f"""
                        **{i+1}. {VOC_CLASSES[label_idx]}**
                        - 置信度: `{score:.4f}`
                        - 边界框: `({int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])})`
                        """)
        else:
            st.info("👆 请上传图片并点击「开始检测」")
            # 显示示例图片提示
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 12px; margin-top: 2rem;">
                <p style="font-size: 3rem;">🖼️</p>
                <p style="color: #6c757d;">上传包含以下物体的图片以获得最佳效果：</p>
                <p style="color: #6c757d; font-size: 0.9rem;">人物、车辆、猫狗、桌椅等日常物体</p>
            </div>
            """, unsafe_allow_html=True)

    # 底部信息
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #6c757d; font-size: 0.85rem;">
        <p>Powered by Faster R-CNN + PyTorch | 基于PASCAL VOC2012数据集训练</p>
        <p>支持20类常见物体检测：人、动物、交通工具、家具、电子设备等</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
