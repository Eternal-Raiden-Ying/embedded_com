import time
import os
import sys
import aidcv as cv2
import numpy as np

# 获取当前脚本所在目录的父目录（即项目根目录）
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR) 

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from backend.camera import HardwareCamera
except ImportError as e:
    print(f"🚨 找不到 backend.camera 模块，请确认文件路径是否正确。")
    raise e

def print_menu():
    print("="*50)
    print("📷 请选择要测试的相机模式:")
    print("  1. RGB 彩色相机  (/dev/video6, YUYV -> RGB)")
    print("  2. Depth 深度相机 (/dev/video2, Z16 -> GRAY16_LE)")
    print("  3. IR 红外相机    (/dev/video4, GREY -> GRAY8)")
    print("="*50)
    choice = input("请输入选项 (1/2/3): ").strip()
    return choice

def main():
    choice = print_menu()
    
    # 默认参数初始化
    cam_kwargs = {
        "out_w": 640, "out_h": 480, # 统一窗口显示大小
    }
    
    current_exposure = 166
    current_brightness = 0
    mode_name = ""

    if choice == '1':
        mode_name = "RGB Mode"
        cam_kwargs.update({
            "device": '/dev/video6',
            "in_w": 1280, "in_h": 720,
            "in_format": "YUY2", "format": "RGB",
            "auto_exposure": False,        # 强制关闭自动曝光
            "exposure": current_exposure,  # 初始曝光值
            "brightness": current_brightness
        })
    elif choice == '2':
        mode_name = "Depth Mode (16-bit)"
        cam_kwargs.update({
            "device": '/dev/video2',
            "in_w": 1280, "in_h": 720,     # D435 深度图常见分辨率
            "in_format": "GRAY16_LE", "format": "GRAY16_LE",
        })
    elif choice == '3':
        mode_name = "IR Mode (8-bit)"
        cam_kwargs.update({
            "device": '/dev/video4',
            "in_w": 640, "in_h": 480,
            "in_format": "GRAY8", "format": "GRAY8",
        })
    else:
        print("❌ 输入无效，程序退出。")
        return

    print(f"\n🚀 正在启动 {mode_name}... 请打开 AidLux 网页端 [cvs] 查看。")
    cv2.namedWindow("AidLux Camera")
    
    # 记录帧率用的变量
    prev_time = time.time()
    
    # 使用 with 语法，确保程序退出时自动调用 cam.release()
    with HardwareCamera(**cam_kwargs) as cam:
        while True:
            # 1. 获取底层显存直通的 NumPy 数组
            hw_frame = cam.read_frame()
            
            # 预热防呆处理
            if hw_frame is None or hw_frame.size == 0:
                cv2.waitKey(10)
                continue

            # 2. 图像渲染与格式转换处理
            if choice == '1':
                # RGB 转为 BGR (OpenCV 默认格式)
                display_frame = cv2.cvtColor(hw_frame, cv2.COLOR_RGB2BGR)
            
            elif choice == '2':
                # 16位深度图 (0-65535) 无法直接漂亮地显示，需要归一化到 8位 (0-255)
                # norm_img = cv2.normalize(hw_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                # 转成伪彩色，或者转为 BGR 以便写彩色字
                display_frame = hw_frame
                
            elif choice == '3':
                # 单通道 8 位灰度图转 BGR 以便写彩色字
                display_frame = cv2.cvtColor(hw_frame, cv2.COLOR_GRAY2BGR)

            # ⚠️ 释放底层硬件内存！(必须，否则管道会死锁)
            del hw_frame 
            
            # 3. 计算并绘制文字信息
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time

            cv2.putText(display_frame, f"FPS: {fps:.1f} | {mode_name}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if choice == '1':
                cv2.putText(display_frame, f"Exposure: {current_exposure} (Keys: A/D)", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display_frame, f"Brightness: {current_brightness} (Keys: W/S)", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 4. 显示画面
            cv2.imshow("AidLux Camera", display_frame)
            
            current_exposure = max(66, current_exposure - 10)
            
            # 5. 处理键盘事件 (动态控制)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'): # ESC 或 Q 键退出
                break
                
            # 仅 RGB 模式下响应曝光和亮度调节
            if choice == '1':
                if key == ord('a'):   # 减少曝光 (抗拖影)
                    current_exposure = max(1, current_exposure - 10)
                    cam.set_exposure(current_exposure)
                elif key == ord('d'): # 增加曝光
                    current_exposure = min(10000, current_exposure + 10)
                    cam.set_exposure(current_exposure)
                elif key == ord('s'): # 减少亮度
                    current_brightness = max(-64, current_brightness - 5)
                    cam.set_brightness(current_brightness)
                elif key == ord('w'): # 增加亮度
                    current_brightness = min(64, current_brightness + 5)
                    cam.set_brightness(current_brightness)

    print("\n✅ 测试结束，相机已安全关闭。")

if __name__ == "__main__":
    main()