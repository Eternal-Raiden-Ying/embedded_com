import os
import sys
import logging
import numpy as np
import cv2
import torch

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
    from config.logging_config import configure_grasp_logger
    from config.predictor_config import build_predictor_arg_parser
except ImportError as e:
    print(f"-> Import Error: {e}")
    print(f"-> Current sys.path: {sys.path}")
    sys.exit(1)


logger = logging.getLogger("vision.grasp")

def main():
    configure_grasp_logger(level=logging.INFO)
    logger.info("Successfully imported RealSenseGraspPredictor from backend.engine")

    default_overrides = {
        'debug': True,
        'dump_dir': os.path.join(CURRENT_DIR, 'debug_res'),
        'rgb_path': os.path.join(CURRENT_DIR, 'data', 'color', 'color_00000.png'),
        'depth_path': os.path.join(CURRENT_DIR, 'data', 'depth', 'depth_raw_00000.png'),
        'seg_path': os.path.join(CURRENT_DIR, 'data', 'seg', 'seg_00000.npy'),
        'camera_metadata': os.path.join(PARENT_DIR, 'config', 'realsense_metadata.json'),
        'yolo_weights_dir': os.path.join(PARENT_DIR, 'weights'),
    }
    parser = build_predictor_arg_parser(description='GraspNet Inference Test Script', default_overrides=default_overrides)

    cfgs = parser.parse_args()

    np.random.seed(cfgs.random_seed)
    torch.manual_seed(cfgs.random_seed)
    logger.info("Random seed: %s", cfgs.random_seed)

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
        logger.info("Loading real data from: %s", cfgs.rgb_path)
        # 读取 RGB
        color_img = cv2.cvtColor(cv2.imread(cfgs.rgb_path), cv2.COLOR_BGR2RGB)
        # 读取 Depth (通常为 16bit PNG, 单位毫米)
        depth_img = normalize_depth_shape(cv2.imread(cfgs.depth_path, cv2.IMREAD_UNCHANGED))
        
        # 读取或生成 Seg Mask
        target_input = None
        if cfgs.run_yolo:
            target_input = cfgs.yolo_class_id
        elif cfgs.seg_path:
            if cfgs.seg_path.endswith('.npy'):
                seg_mask = normalize_seg_shape(np.load(cfgs.seg_path))
            else:
                seg_mask = normalize_seg_shape(cv2.imread(cfgs.seg_path, cv2.IMREAD_UNCHANGED))
            target_input = seg_mask
        else:
            raise ValueError("seg_mask is required for real inference. Provide --seg_path or enable --run_yolo.")
    else:
        logger.info("No input paths provided, generating dummy data for testing...")
        H, W = 720, 1280
        color_img = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        depth_img = (np.random.rand(H, W) * 1000).astype(np.uint16)
        target_input = np.ones((H, W), dtype=np.uint8)

    depth_img = normalize_depth_shape(depth_img)
    seg_mask = None if cfgs.run_yolo else normalize_seg_shape(target_input)

    if depth_img is None:
        raise ValueError(f"Failed to read depth image: {cfgs.depth_path}")
    if color_img is None:
        raise ValueError(f"Failed to read color image: {cfgs.rgb_path}")
    if not cfgs.run_yolo and seg_mask is None:
        raise ValueError("Segmentation mask is None")

    if depth_img.ndim != 2:
        raise ValueError(f"Depth image must be 2D after normalization, got shape {depth_img.shape}")
    if not cfgs.run_yolo and seg_mask.ndim != 2:
        raise ValueError(f"Segmentation mask must be 2D after normalization, got shape {seg_mask.shape}")

    if not cfgs.run_yolo and seg_mask.shape != depth_img.shape:
        logger.info("Seg/depth shape mismatch: seg=%s, depth=%s. Resizing seg with nearest-neighbor.", seg_mask.shape, depth_img.shape)
        seg_mask = cv2.resize(seg_mask.astype(np.uint8), (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    if cfgs.run_yolo:
        logger.info("Using online YOLO segmentation in engine. target_class_id=%s", cfgs.yolo_class_id)
    else:
        seg_mask = (seg_mask > 0).astype(np.uint8)
        target_input = seg_mask

    mask_pixels = int(seg_mask.sum()) if seg_mask is not None else -1
    valid_depth_pixels = int((depth_img > 0).sum())
    masked_valid_pixels = int(((depth_img > 0) & (seg_mask > 0)).sum()) if seg_mask is not None else -1
    seg_shape = seg_mask.shape if seg_mask is not None else 'online_yolo'
    logger.info("RGB shape: %s, Depth shape: %s, Seg shape: %s", color_img.shape, depth_img.shape, seg_shape)
    logger.info("Valid depth pixels: %s", valid_depth_pixels)
    if seg_mask is not None:
        logger.info("Seg mask pixels: %s", mask_pixels)
        logger.info("Valid masked pixels: %s", masked_valid_pixels)

    # ================= 4. 执行推理 =================
    # 初始化推理引擎
    predictor = RealSenseGraspPredictor(cfgs)

    # 运行推理
    logger.info("Starting inference...")
    grasp_results = predictor.infer(color_img, depth_img, target_input)

    # ================= 5. 结果输出 =================
    if grasp_results is not None:
        logger.info("Found %s grasps.", len(grasp_results))
        if len(grasp_results) > 0:
            top_grasp = grasp_results[0]
            logger.info("Best Grasp Score: %.4f", top_grasp.score)
            logger.info("Best Grasp Translation: %s", top_grasp.translation)
    else:
        logger.warning("Inference failed or no grasps found")

if __name__ == '__main__':
    main()
