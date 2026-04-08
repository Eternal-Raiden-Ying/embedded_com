import argparse
import os
from dataclasses import asdict, dataclass


CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.dirname(CONFIG_DIR)
PROJECT_ROOT = os.path.dirname(MODULE_DIR)


@dataclass
class PredictorConfig:
    checkpoint_path: str = os.path.join(MODULE_DIR, 'weights', 'minkuresunet_realsense.tar')
    dump_dir: str = os.path.join(MODULE_DIR, 'test', 'debug_res')
    seed_feat_dim: int = 512
    num_point: int = 15000
    voxel_size: float = 0.005
    collision_thresh: float = -1.0
    voxel_size_cd: float = 0.01
    random_seed: int = 0
    scene_max_depth: float = 3.0
    debug_grasp_count: int = 15

    rgb_path: str = os.path.join(MODULE_DIR, 'test', 'data', 'color', 'color_00000.png')
    depth_path: str = os.path.join(MODULE_DIR, 'test', 'data', 'depth', 'depth_raw_00000.png')
    seg_path: str = os.path.join(MODULE_DIR, 'test', 'data', 'seg', 'seg_00000.npy')
    camera_metadata: str = os.path.join(CONFIG_DIR, 'realsense_metadata.json')

    run_yolo: bool = False
    yolo_model: str = 'yolo26m-seg.pt'
    yolo_weights_dir: str = os.path.join(MODULE_DIR, 'weights')
    yolo_class_id: int = 46
    yolo_conf: float = 0.25
    yolo_iou: float = 0.7
    bbox_expand_scale: float = 2.0
    collision_depth_margin: float = 0.15

    debug: bool = False


def build_predictor_defaults(default_overrides=None):
    defaults = asdict(PredictorConfig())
    if default_overrides:
        defaults.update(default_overrides)
    return defaults


def add_predictor_args(parser, default_overrides=None):
    defaults = build_predictor_defaults(default_overrides)

    parser.add_argument('--checkpoint_path', type=str, default=defaults['checkpoint_path'], help='Model checkpoint path')
    parser.add_argument('--dump_dir', type=str, default=defaults['dump_dir'], help='Folder to save debug outputs')
    parser.add_argument('--seed_feat_dim', type=int, default=defaults['seed_feat_dim'], help='Point wise feature dim')
    parser.add_argument('--num_point', type=int, default=defaults['num_point'], help='Point Number')
    parser.add_argument('--voxel_size', type=float, default=defaults['voxel_size'], help='Voxel size for sparse convolution')
    parser.add_argument('--collision_thresh', type=float, default=defaults['collision_thresh'], help='Collision threshold, <=0 disables collision detection')
    parser.add_argument('--voxel_size_cd', type=float, default=defaults['voxel_size_cd'], help='Voxel size for collision detection')
    parser.add_argument('--random_seed', type=int, default=defaults['random_seed'], help='Random seed for reproducible point sampling')
    parser.add_argument('--scene_max_depth', type=float, default=defaults['scene_max_depth'], help='Maximum depth in meters for debug scene cloud')
    parser.add_argument('--debug_grasp_count', type=int, default=defaults['debug_grasp_count'], help='Number of top grasps to export in debug mesh')

    parser.add_argument('--rgb_path', type=str, default=defaults['rgb_path'], help='Path to RGB image')
    parser.add_argument('--depth_path', type=str, default=defaults['depth_path'], help='Path to depth image')
    parser.add_argument('--seg_path', type=str, default=defaults['seg_path'], help='Path to segmentation mask')
    parser.add_argument('--camera_metadata', type=str, default=defaults['camera_metadata'], help='Path to camera metadata JSON file')

    parser.add_argument('--run_yolo', action='store_true', default=defaults['run_yolo'], help='Use engine internal YOLO segmentation with class id input')
    parser.add_argument('--yolo_model', type=str, default=defaults['yolo_model'], help='YOLO segmentation model name or path')
    parser.add_argument('--yolo_weights_dir', type=str, default=defaults['yolo_weights_dir'], help='Ultralytics weights cache directory')
    parser.add_argument('--yolo_class_id', type=int, default=defaults['yolo_class_id'], help='YOLO target class id')
    parser.add_argument('--yolo_conf', type=float, default=defaults['yolo_conf'], help='YOLO confidence threshold')
    parser.add_argument('--yolo_iou', type=float, default=defaults['yolo_iou'], help='YOLO NMS IoU threshold')
    parser.add_argument('--bbox_expand_scale', type=float, default=defaults['bbox_expand_scale'], help='Expand bbox width/height around center by this scale')
    parser.add_argument('--collision_depth_margin', type=float, default=defaults['collision_depth_margin'], help='Depth margin in meters for bbox collision cloud')

    parser.add_argument('--debug', action='store_true', default=defaults['debug'], help='Enable debug outputs')
    return parser


def build_predictor_arg_parser(description=None, default_overrides=None):
    parser = argparse.ArgumentParser(description=description)
    return add_predictor_args(parser, default_overrides=default_overrides)


def create_predictor_config(default_overrides=None, **overrides):
    values = build_predictor_defaults(default_overrides)
    values.update(overrides)
    return PredictorConfig(**values)
