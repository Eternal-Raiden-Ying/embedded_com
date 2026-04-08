import os
import cv2
import argparse
import numpy as np
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from backend.utils.yolo_utils import load_yolo_model, predict_target_masks


def generate_binary_mask(model, img_path, class_id):
    bgr_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise ValueError(f"Failed to read image: {img_path}")
    binary_mask, _, overlay_img, result_info = predict_target_masks(model, bgr_image, class_id)
    return binary_mask, overlay_img, result_info

def main(args):
    # 1. 设置 Ultralytics 全局设置 (如权重缓存路径)
    # 这会告诉 Ultralytics 去这个目录寻找或下载模型权重
    print(f"[*] 权重目录已设置为: {args.weights_dir}")

    # 2. 加载模型
    print(f"[*] 正在加载模型: {args.model}")
    model = load_yolo_model(args.model, args.weights_dir)

    # 3. 进行推理
    print(f"[*] 正在对图片 {args.img_path} 进行推理...")
    print(f"[*] 指定类别 ID: {args.class_id}")
    
    # retina_masks=True 用于生成与原图分辨率匹配的高精度掩码
    binary_mask, overlay_img, result = generate_binary_mask(model, args.img_path, args.class_id)

    # 如果检测到目标，则将对应的多边形区域填充为 1
    if result['found']:
        print(f"[*] 检测到目标！掩码矩阵包含的像素类别: {np.unique(binary_mask)}")
    else:
        print("[!] 未检测到指定类别的目标，输出全 0 掩码。")

    # 6. 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    idx = args.img_path.strip('.png')[-5: ]  # 获取文件名（不带扩展名）
    
    npy_path = os.path.join(args.output_dir, f"seg_{idx}.npy")
    img_path = os.path.join(args.output_dir, f"seg_{idx}_overlay.jpg")

    # 保存 numpy 二值矩阵
    np.save(npy_path, binary_mask)
    # 保存渲染效果图
    cv2.imwrite(img_path, overlay_img)

    print(f"[*] 推理完成！结果已保存:")
    print(f"    - Numpy 掩码: {npy_path}")
    print(f"    - 渲染效果图: {img_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultralytics YOLO Segmentation Script")
    
    # 路径与模型参数
    parser.add_argument('--img_path', type=str, required=True, help='输入图片的路径')
    parser.add_argument('--weights_dir', type=str, default='../weights', help='权重缓存路径')
    parser.add_argument('--model', type=str, default='yolo26m-seg.pt', help='模型名称或路径')
    
    # 推理参数
    parser.add_argument('--class_id', type=int, default=46, help='指定过滤的类别 ID (例如 0 通常为人)')  # 46: banana
    parser.add_argument('--output_dir', type=str, default='./data/seg', help='结果保存目录')

    args = parser.parse_args()
    main(args)
