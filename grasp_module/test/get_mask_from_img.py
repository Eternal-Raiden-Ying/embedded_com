import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO, settings


def generate_binary_mask(model, img_path, class_id):
    results = model.predict(
        source=img_path,
        classes=[class_id],
        retina_masks=True,
        save=False,
    )

    result = results[0]
    orig_img = result.orig_img
    h, w = orig_img.shape[:2]
    binary_mask = np.zeros((h, w), dtype=np.uint8)

    if result.masks is not None:
        for seg in result.masks.xy:
            pts = np.array(seg, dtype=np.int32)
            cv2.fillPoly(binary_mask, [pts], 1)

    overlay_img = result.plot(boxes=False, labels=False)
    return binary_mask, overlay_img, result

def main(args):
    # 1. 设置 Ultralytics 全局设置 (如权重缓存路径)
    # 这会告诉 Ultralytics 去这个目录寻找或下载模型权重
    settings.update({'weights_dir': args.weights_dir})
    print(f"[*] 权重目录已设置为: {settings['weights_dir']}")

    # 2. 加载模型
    print(f"[*] 正在加载模型: {args.model}")
    model = YOLO(args.model)

    # 3. 进行推理
    print(f"[*] 正在对图片 {args.img_path} 进行推理...")
    print(f"[*] 指定类别 ID: {args.class_id}")
    
    # retina_masks=True 用于生成与原图分辨率匹配的高精度掩码
    binary_mask, overlay_img, result = generate_binary_mask(model, args.img_path, args.class_id)

    # 如果检测到目标，则将对应的多边形区域填充为 1
    if result.masks is not None:
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
