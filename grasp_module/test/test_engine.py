import os
import sys
import logging
import cv2
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

try:
    from backend.engine import RealSenseGraspPredictor
    from config.logging_config import configure_grasp_logger
    from config.predictor_config import build_predictor_arg_parser
except ImportError as e:
    print(f"-> Import Error: {e}")
    print(f"-> Current sys.path: {sys.path}")
    sys.exit(1)


logger = logging.getLogger("vision.grasp")


def normalize_depth_shape(depth_img):
    if depth_img is None:
        return depth_img
    if depth_img.ndim == 3 and depth_img.shape[2] == 1:
        return depth_img[:, :, 0]
    return depth_img


def main():
    configure_grasp_logger(level=logging.INFO)
    logger.info("Successfully imported RealSenseGraspPredictor from backend.engine")

    default_overrides = {
        'debug': True,
        'dump_dir': os.path.join(CURRENT_DIR, 'debug_res'),
        'rgb_path': os.path.join(CURRENT_DIR, 'data', 'color', 'color_00000.png'),
        'depth_path': os.path.join(CURRENT_DIR, 'data', 'depth', 'depth_raw_00000.png'),
        'camera_metadata': os.path.join(PARENT_DIR, 'config', 'realsense_metadata.json'),
        'yolo_weights_dir': os.path.join(PARENT_DIR, 'weights'),
    }
    parser = build_predictor_arg_parser(
        description='GraspNet inference test script (class_id only)',
        default_overrides=default_overrides,
    )
    cfgs = parser.parse_args()

    torch.manual_seed(cfgs.random_seed)
    logger.info("Random seed: %s", cfgs.random_seed)

    logger.info("Loading RGB-D data from: %s", cfgs.rgb_path)
    color_bgr = cv2.imread(cfgs.rgb_path, cv2.IMREAD_COLOR)
    color_img = None if color_bgr is None else cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    depth_img = normalize_depth_shape(cv2.imread(cfgs.depth_path, cv2.IMREAD_UNCHANGED))

    if depth_img is None:
        raise ValueError(f"Failed to read depth image: {cfgs.depth_path}")
    if color_img is None:
        raise ValueError(f"Failed to read color image: {cfgs.rgb_path}")
    if depth_img.ndim != 2:
        raise ValueError(f"Depth image must be 2D after normalization, got shape {depth_img.shape}")

    logger.info("Using internal YOLO segmentation. target_class_id=%s", cfgs.yolo_class_id)
    logger.info("RGB shape: %s, Depth shape: %s", color_img.shape, depth_img.shape)
    logger.info("Valid depth pixels: %s", int((depth_img > 0).sum()))

    predictor = RealSenseGraspPredictor(cfgs)

    logger.info("Starting inference...")
    grasp_results = predictor.infer(color_img, depth_img, cfgs.yolo_class_id)

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
