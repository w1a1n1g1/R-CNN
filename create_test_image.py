"""
创建测试图像用于R-CNN目标检测实验
生成包含多个目标的合成图像
"""

import cv2
import numpy as np

def create_test_image():
    """
    创建一个包含多个目标的测试图像
    包括：人、车、猫、狗等常见目标
    """
    # 创建画布（800x600像素）
    img = np.ones((600, 800, 3), dtype=np.uint8) * 240  # 浅灰色背景

    # 绘制街道场景背景
    # 天空
    cv2.rectangle(img, (0, 0), (800, 200), (135, 206, 235), -1)

    # 道路
    cv2.rectangle(img, (0, 400), (800, 600), (128, 128, 128), -1)
    # 道路标线
    for i in range(0, 800, 100):
        cv2.rectangle(img, (i+30, 490), (i+70, 510), (255, 255, 255), -1)

    # 绘制建筑物（简化表示）
    cv2.rectangle(img, (50, 150), (200, 400), (139, 69, 19), -1)  # 棕色建筑
    cv2.rectangle(img, (600, 120), (750, 400), (105, 105, 105), -1)  # 灰色建筑

    # 绘制窗户
    for i in range(70, 180, 40):
        for j in range(170, 380, 50):
            cv2.rectangle(img, (i, j), (i+25, j+30), (255, 255, 224), -1)

    # 绘制树木
    def draw_tree(img, x, y):
        # 树干
        cv2.rectangle(img, (x-10, y), (x+10, y+80), (139, 69, 19), -1)
        # 树冠
        cv2.circle(img, (x, y-20), 50, (34, 139, 34), -1)
        cv2.circle(img, (x-25, y+10), 35, (34, 139, 34), -1)
        cv2.circle(img, (x+25, y+10), 35, (34, 139, 34), -1)

    draw_tree(img, 250, 350)
    draw_tree(img, 550, 340)

    # 绘制汽车（简化表示）
    def draw_car(img, x, y, color, scale=1.0):
        # 车身
        body_w, body_h = int(120*scale), int(40*scale)
        cv2.rectangle(img, (x, y), (x+body_w, y+body_h), color, -1)
        # 车顶
        roof_w, roof_h = int(80*scale), int(30*scale)
        cv2.rectangle(img, (x+20, y-roof_h), (x+20+roof_w, y), color, -1)
        # 车轮
        wheel_r = int(12*scale)
        cv2.circle(img, (x+25, y+body_h), wheel_r, (50, 50, 50), -1)
        cv2.circle(img, (x+95, y+body_h), wheel_r, (50, 50, 50), -1)
        # 车窗
        window_color = (200, 220, 255)
        cv2.rectangle(img, (x+25, y-int(25*scale)), (x+55, y-5), window_color, -1)
        cv2.rectangle(img, (x+65, y-int(25*scale)), (x+95, y-5), window_color, -1)

    # 绘制红色汽车
    draw_car(img, 150, 420, (0, 0, 255), 1.0)
    # 绘制蓝色汽车
    draw_car(img, 450, 450, (255, 100, 0), 0.9)

    # 绘制行人（简化表示）
    def draw_person(img, x, y, scale=1.0):
        # 头部
        head_r = int(15*scale)
        cv2.circle(img, (x, y-int(80*scale)), head_r, (255, 220, 177), -1)
        # 身体
        cv2.rectangle(img, (x-int(15*scale), y-int(65*scale)),
                     (x+int(15*scale), y-int(20*scale)), (0, 100, 200), -1)
        # 腿
        cv2.rectangle(img, (x-int(10*scale), y-int(20*scale)),
                     (x-int(2*scale), y), (50, 50, 50), -1)
        cv2.rectangle(img, (x+int(2*scale), y-int(20*scale)),
                     (x+int(10*scale), y), (50, 50, 50), -1)
        # 手臂
        cv2.line(img, (x-int(15*scale), y-int(50*scale)),
                (x-int(30*scale), y-int(30*scale)), (255, 220, 177), int(5*scale))
        cv2.line(img, (x+int(15*scale), y-int(50*scale)),
                (x+int(30*scale), y-int(30*scale)), (255, 220, 177), int(5*scale))

    # 绘制行人
    draw_person(img, 350, 400, 1.0)
    draw_person(img, 650, 380, 0.9)

    # 绘制猫（简化表示）
    def draw_cat(img, x, y, scale=1.0):
        # 身体
        body_w, body_h = int(50*scale), int(30*scale)
        cv2.ellipse(img, (x, y), (body_w//2, body_h//2), 0, 0, 360, (255, 165, 0), -1)
        # 头
        head_r = int(18*scale)
        cv2.circle(img, (x-int(25*scale), y-int(10*scale)), head_r, (255, 165, 0), -1)
        # 耳朵
        ear_pts = np.array([
            [x-int(35*scale), y-int(20*scale)],
            [x-int(30*scale), y-int(35*scale)],
            [x-int(20*scale), y-int(18*scale)]
        ], np.int32)
        cv2.fillPoly(img, [ear_pts], (255, 165, 0))
        ear_pts2 = np.array([
            [x-int(20*scale), y-int(18*scale)],
            [x-int(15*scale), y-int(35*scale)],
            [x-int(10*scale), y-int(15*scale)]
        ], np.int32)
        cv2.fillPoly(img, [ear_pts2], (255, 165, 0))
        # 尾巴
        cv2.ellipse(img, (x+int(30*scale), y-int(5*scale)),
                   (int(15*scale), int(5*scale)), 45, 0, 360, (255, 165, 0), -1)
        # 眼睛
        cv2.circle(img, (x-int(30*scale), y-int(12*scale)), int(3*scale), (0, 0, 0), -1)
        cv2.circle(img, (x-int(18*scale), y-int(10*scale)), int(3*scale), (0, 0, 0), -1)

    # 绘制猫
    draw_cat(img, 500, 350, 0.8)

    # 绘制狗（简化表示）
    def draw_dog(img, x, y, scale=1.0):
        # 身体
        body_w, body_h = int(60*scale), int(35*scale)
        cv2.ellipse(img, (x, y), (body_w//2, body_h//2), 0, 0, 360, (139, 69, 19), -1)
        # 头
        head_r = int(22*scale)
        cv2.circle(img, (x-int(30*scale), y-int(15*scale)), head_r, (139, 69, 19), -1)
        # 耳朵
        cv2.ellipse(img, (x-int(45*scale), y-int(25*scale)),
                   (int(8*scale), int(15*scale)), 30, 0, 360, (101, 67, 33), -1)
        cv2.ellipse(img, (x-int(15*scale), y-int(25*scale)),
                   (int(8*scale), int(15*scale)), -30, 0, 360, (101, 67, 33), -1)
        # 鼻子
        cv2.circle(img, (x-int(42*scale), y-int(12*scale)), int(4*scale), (0, 0, 0), -1)
        # 眼睛
        cv2.circle(img, (x-int(35*scale), y-int(20*scale)), int(3*scale), (0, 0, 0), -1)
        # 腿
        for offset in [-20, -5, 5, 20]:
            cv2.rectangle(img, (x+int((offset-3)*scale), y+int(10*scale)),
                         (x+int((offset+3)*scale), y+int(30*scale)), (101, 67, 33), -1)
        # 尾巴
        cv2.ellipse(img, (x+int(35*scale), y-int(10*scale)),
                   (int(20*scale), int(5*scale)), -30, 0, 360, (139, 69, 19), -1)

    # 绘制狗
    draw_dog(img, 280, 480, 0.9)

    # 绘制交通标志（简化表示）
    def draw_sign(img, x, y):
        # 标志杆
        cv2.rectangle(img, (x-3, y), (x+3, y+100), (128, 128, 128), -1)
        # 圆形标志
        cv2.circle(img, (x, y-20), 25, (0, 0, 200), -1)
        cv2.circle(img, (x, y-20), 25, (255, 255, 255), 3)
        # 停止标志文字（简化用横线表示）
        cv2.line(img, (x-10, y-25), (x+10, y-25), (255, 255, 255), 3)

    draw_sign(img, 720, 250)

    # 添加一些细节
    # 云朵
    def draw_cloud(img, x, y):
        cv2.circle(img, (x, y), 20, (255, 255, 255), -1)
        cv2.circle(img, (x+20, y), 25, (255, 255, 255), -1)
        cv2.circle(img, (x+40, y), 20, (255, 255, 255), -1)

    draw_cloud(img, 100, 80)
    draw_cloud(img, 400, 60)
    draw_cloud(img, 650, 90)

    return img


if __name__ == "__main__":
    # 创建测试图像
    test_img = create_test_image()

    # 保存图像
    output_path = "test_image.jpg"
    cv2.imwrite(output_path, test_img)
    print(f"测试图像已创建并保存至：{output_path}")
    print(f"图像尺寸：{test_img.shape}")

    # 显示图像（如果在有图形界面的环境中）
    try:
        cv2.imshow("Test Image", test_img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    except:
        print("无法显示图像（可能是在无图形界面环境中），但文件已保存")
