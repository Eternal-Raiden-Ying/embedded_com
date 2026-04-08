import os
import sys
import numpy as np
import argparse
import cv2
import torch
from ultralytics import YOLO, settings

# ================= 1. 环境准备：将父级目录加入环境变量 =================
# 假设当前脚本位于：/project/test/test_predictor.py
# 而代码结构是：/project/backend/engine.py
# 我们需要把 /project 加入 sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR) 
sys.path.append(PARENT_DIR)

# 尝试导入 Predictor
try:
    # 假设你的类定义在 backend/engine.py 中
    from backend.engine import RealSenseGraspPredictor
    from get_mask_from_img import generate_binary_mask
    print("-> Successfully imported RealSenseGraspPredictor from backend.engine")
except ImportError as e:
    print(f"-> Import Error: {e}")
    print(f"-> Current sys.path: {sys.path}")
    sys.exit(1)

def main():
    # ================= 2. 参数解析 (保留并扩展原逻辑) =================
    parser = argparse.ArgumentParser(description="GraspNet Inference Test Script")
    
    # 模型配置相关
    parser.add_argument('--checkpoint_path', default='../weights/minkuresunet_realsense.tar', help='Model checkpoint path')
    parser.add_argument('--dump_dir', default='./debug_res', help='Folder to save debug outputs')
    parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
    parser.add_argument('--num_point', type=int, default=15000, help='Point Number')
    parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
    parser.add_argument('--collision_thresh', type=float, default=-1, help='Collision Threshold (set to -1 to disable)')
    parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed for reproducible point sampling')
    
    # 输入数据相关
    parser.add_argument('--rgb_path', type=str, default='./data/color/color_00000.png', help='Path to RGB image (jpg/png)')
    parser.add_argument('--depth_path', type=str, default='./data/depth/depth_raw_00000.png', help='Path to Depth image (png, 16bit)')
    parser.add_argument('--seg_path', type=str, default='./data/seg/seg_00000.npy', help='Path to Segmentation mask (png/npy)')
    parser.add_argument('--camera_metadata', type=str, default='../config/realsense_metadata.json', help='Path to camera metadata JSON file')

    # YOLO 即时分割验证
    parser.add_argument('--run_yolo', action='store_true', help='Run YOLO on rgb_path to generate segmentation mask at runtime')
    parser.add_argument('--yolo_model', type=str, default='yolo26m-seg.pt', help='YOLO segmentation model name or path')
    parser.add_argument('--yolo_weights_dir', type=str, default='../weights', help='Ultralytics weights cache directory')
    parser.add_argument('--yolo_class_id', type=int, default=46, help='YOLO target class id')
    parser.add_argument('--save_runtime_seg', action='store_true', help='Save runtime YOLO mask and overlay to dump_dir')
    
    # Debug 开关
    parser.add_argument('--debug', action='store_true', default=True, help='Enable debug mode to save visualizations')

    cfgs = parser.parse_args()

    np.random.seed(cfgs.random_seed)
    torch.manual_seed(cfgs.random_seed)
    print(f"-> Random seed: {cfgs.random_seed}")

    def normalize_depth_shape(depth_img):
        if depth_img is None:
            return depth_img
        if depth_img.ndim == 3 and depth_img.shape[2] == 1:
            return depth_img[:, :, 0]
        return depth_img

    def normalize_seg_shape(seg_mask):
        if seg_mask is None:
            return seg_mask
        if seg_mask.ndim == 3:
            if seg_mask.shape[2] == 1:
                return seg_mask[:, :, 0]
            return cv2.cvtColor(seg_mask, cv2.COLOR_BGR2GRAY)
        return seg_mask

    # ================= 3. 数据加载逻辑 =================
    # 如果提供了路径则加载真实数据，否则生成 Dummy 数据
    if cfgs.rgb_path and cfgs.depth_path:
        print(f"-> Loading real data from: {cfgs.rgb_path}")
        # 读取 RGB
        color_img = cv2.cvtColor(cv2.imread(cfgs.rgb_path), cv2.COLOR_BGR2RGB)
        # 读取 Depth (通常为 16bit PNG, 单位毫米)
        depth_img = normalize_depth_shape(cv2.imread(cfgs.depth_path, cv2.IMREAD_UNCHANGED))
        
        # 读取或生成 Seg Mask
        if cfgs.run_yolo:
            settings.update({'weights_dir': cfgs.yolo_weights_dir})
            print(f"-> Running YOLO segmentation on current RGB: {cfgs.yolo_model}, class={cfgs.yolo_class_id}")
            model = YOLO(cfgs.yolo_model)
            seg_mask, overlay_img, result = generate_binary_mask(model, cfgs.rgb_path, cfgs.yolo_class_id)
            if cfgs.save_runtime_seg:
                os.makedirs(cfgs.dump_dir, exist_ok=True)
                np.save(os.path.join(cfgs.dump_dir, 'runtime_seg.npy'), seg_mask)
                cv2.imwrite(os.path.join(cfgs.dump_dir, 'runtime_seg_overlay.jpg'), overlay_img)
                print(f"-> Saved runtime seg outputs to: {cfgs.dump_dir}")
            if result.masks is None:
                print("-> YOLO found no target mask; using empty seg mask.")
        elif cfgs.seg_path:
            if cfgs.seg_path.endswith('.npy'):
                seg_mask = normalize_seg_shape(np.load(cfgs.seg_path))
            else:
                seg_mask = normalize_seg_shape(cv2.imread(cfgs.seg_path, cv2.IMREAD_UNCHANGED))
        else:
            print("-> No seg_mask provided, using full image mask.")
            seg_mask = np.ones_like(depth_img, dtype=np.uint8)
    else:
        print("-> No input paths provided, generating dummy data for testing...")
        H, W = 720, 1280
        color_img = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        depth_img = (np.random.rand(H, W) * 1000).astype(np.uint16)
        seg_mask = np.ones((H, W), dtype=np.uint8)

    depth_img = normalize_depth_shape(depth_img)
    seg_mask = normalize_seg_shape(seg_mask)

    if depth_img is None:
        raise ValueError(f"Failed to read depth image: {cfgs.depth_path}")
    if color_img is None:
        raise ValueError(f"Failed to read color image: {cfgs.rgb_path}")
    if seg_mask is None:
        raise ValueError("Segmentation mask is None")

    if depth_img.ndim != 2:
        raise ValueError(f"Depth image must be 2D after normalization, got shape {depth_img.shape}")
    if seg_mask.ndim != 2:
        raise ValueError(f"Segmentation mask must be 2D after normalization, got shape {seg_mask.shape}")

    if seg_mask.shape != depth_img.shape:
        print(f"-> Seg/depth shape mismatch: seg={seg_mask.shape}, depth={depth_img.shape}. Resizing seg with nearest-neighbor.")
        seg_mask = cv2.resize(seg_mask.astype(np.uint8), (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    seg_mask = (seg_mask > 0).astype(np.uint8)
    mask_pixels = int(seg_mask.sum())
    valid_depth_pixels = int((depth_img > 0).sum())
    masked_valid_pixels = int(((depth_img > 0) & (seg_mask > 0)).sum())
    print(f"-> RGB shape: {color_img.shape}, Depth shape: {depth_img.shape}, Seg shape: {seg_mask.shape}")
    print(f"-> Valid depth pixels: {valid_depth_pixels}")
    print(f"-> Seg mask pixels: {mask_pixels}")
    print(f"-> Valid masked pixels: {masked_valid_pixels}")

    # ================= 4. 执行推理 =================
    # 初始化推理引擎
    predictor = RealSenseGraspPredictor(cfgs)

    # 运行推理
    print("-> Starting inference...")
    grasp_results = predictor.infer(color_img, depth_img, seg_mask)

    # ================= 5. 结果输出 =================
    if grasp_results is not None:
        print(f"-> Found {len(grasp_results)} grasps.")
        if len(grasp_results) > 0:
            top_grasp = grasp_results[0]
            print(f"-> Best Grasp Score: {top_grasp.score:.4f}")
            print(f"-> Best Grasp Translation: {top_grasp.translation}")
    else:
        print("-> Inference failed or no grasps found.")

if __name__ == '__main__':
    main()
