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

    # ---- Debug gripper mesh (PLY visualisation only) ----
    # These only affect save_debug_visualizations() / test replay mesh export.
    # They do NOT influence collision detection or downstream protocol output.
    gripper_height_m: float = 0.028
    gripper_finger_width_m: float = 0.018
    gripper_depth_base_m: float = 0.0
    gripper_tail_length_m: float = 0.04

    # ---- Collision detection occupancy parameters ----
    # Used by ModelFreeCollisionDetector. Independent of the debug mesh above.
    collision_finger_width_m: float = 0.02
    collision_finger_length_m: float = 0.07
    collision_height_override_m: float = -1.0

    rgb_path: str = os.path.join(MODULE_DIR, 'test', 'data', 'color', 'color_00000.png')
    depth_path: str = os.path.join(MODULE_DIR, 'test', 'data', 'depth', 'depth_raw_00000.png')
    camera_metadata: str = os.path.join(CONFIG_DIR, 'realsense_metadata.json')

    yolo_model: str = 'yolo26m-seg.pt'
    yolo_weights_dir: str = os.path.join(MODULE_DIR, 'weights')
    yolo_class_id: int = 47
    yolo_conf: float = 0.25
    yolo_iou: float = 0.7
    bbox_expand_scale: float = 2.0
    collision_depth_margin: float = 0.15
    protocol_depth_base: float = -0.03  # 0.02
    protocol_feasible_distance_cm: float = 2.0
    protocol_min_score: float = 0.3
    response_max_targets: int = 5
    reference_line_x_cm: float = 0.0
    reference_line_y_cm: float = 0.0
    reposition_max_distance_cm: float = 20.0
    robot_cam_rotation_csv: str = '0.00428801,-0.63729195,0.77061053,-0.99996824,0.00244406,0.00758549,-0.00671759,-0.77061858,-0.63726123'
    robot_cam_translation_cm_csv: str = '-10.30593831,0.93004589,35.4166982'
    robot_calibration_translation_cm_csv: str = ''

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
    parser.add_argument('--gripper_height_m', type=float, default=defaults['gripper_height_m'], help='Repo-local debug gripper mesh total height in meters')
    parser.add_argument('--gripper_finger_width_m', type=float, default=defaults['gripper_finger_width_m'], help='Repo-local debug gripper finger thickness in meters')
    parser.add_argument('--gripper_depth_base_m', type=float, default=defaults['gripper_depth_base_m'], help='Repo-local debug gripper rear offset from grasp origin in meters')
    parser.add_argument('--gripper_tail_length_m', type=float, default=defaults['gripper_tail_length_m'], help='Repo-local debug gripper tail length in meters')
    parser.add_argument('--collision_finger_width_m', type=float, default=defaults['collision_finger_width_m'], help='Collision occupancy finger thickness in meters')
    parser.add_argument('--collision_finger_length_m', type=float, default=defaults['collision_finger_length_m'], help='Collision occupancy finger length in meters')
    parser.add_argument('--collision_height_override_m', type=float, default=defaults['collision_height_override_m'], help='Override grasp total height used in collision detection; <=0 keeps raw grasp height')

    parser.add_argument('--rgb_path', type=str, default=defaults['rgb_path'], help='Path to RGB image')
    parser.add_argument('--depth_path', type=str, default=defaults['depth_path'], help='Path to depth image')
    parser.add_argument('--camera_metadata', type=str, default=defaults['camera_metadata'], help='Path to camera metadata JSON file')

    parser.add_argument('--yolo_model', type=str, default=defaults['yolo_model'], help='YOLO segmentation model name or path')
    parser.add_argument('--yolo_weights_dir', type=str, default=defaults['yolo_weights_dir'], help='Ultralytics weights cache directory')
    parser.add_argument('--yolo_class_id', type=int, default=defaults['yolo_class_id'], help='YOLO target class id')
    parser.add_argument('--yolo_conf', type=float, default=defaults['yolo_conf'], help='YOLO confidence threshold')
    parser.add_argument('--yolo_iou', type=float, default=defaults['yolo_iou'], help='YOLO NMS IoU threshold')
    parser.add_argument('--bbox_expand_scale', type=float, default=defaults['bbox_expand_scale'], help='Expand bbox width/height around center by this scale')
    parser.add_argument('--collision_depth_margin', type=float, default=defaults['collision_depth_margin'], help='Depth margin in meters for bbox collision cloud')
    parser.add_argument('--protocol_depth_base', type=float, default=defaults['protocol_depth_base'], help='Rear-edge offset from grasp origin along negative approach axis, in meters')
    parser.add_argument('--protocol_feasible_distance_cm', type=float, default=defaults['protocol_feasible_distance_cm'], help='Maximum allowed 3D distance (cm) between the approach line and the reference Z-line')
    parser.add_argument('--protocol_min_score', type=float, default=defaults['protocol_min_score'], help='Minimum score required for a grasp to be reported as executable; <=0 disables this filter')
    parser.add_argument('--response_max_targets', type=int, default=defaults['response_max_targets'], help='Maximum number of protocol targets returned to downstream clients')
    parser.add_argument('--reference_line_x_cm', type=float, default=defaults['reference_line_x_cm'], help='X coordinate of the reference Z-parallel line in robot frame (cm)')
    parser.add_argument('--reference_line_y_cm', type=float, default=defaults['reference_line_y_cm'], help='Y coordinate of the reference Z-parallel line in robot frame (cm)')
    parser.add_argument('--reposition_max_distance_cm', type=float, default=defaults['reposition_max_distance_cm'], help='Maximum allowed XY distance from reference line to grasp in reposition proposal (cm)')
    parser.add_argument('--robot_cam_rotation_csv', type=str, default=defaults['robot_cam_rotation_csv'], help='Rotation from native camera frame to robot frame, in row-major CSV format (9 values)')
    parser.add_argument('--robot_cam_translation_cm_csv', type=str, default=defaults['robot_cam_translation_cm_csv'], help='Translation from native camera frame to robot frame, in centimeters CSV format (3 values)')
    parser.add_argument('--robot_calibration_translation_cm_csv', type=str, default=defaults['robot_calibration_translation_cm_csv'], help='Deprecated compatibility alias for robot_cam_translation_cm_csv')

    parser.add_argument('--debug', action='store_true', default=defaults['debug'], help='Enable debug outputs')
    return parser


def build_predictor_arg_parser(description=None, default_overrides=None):
    parser = argparse.ArgumentParser(description=description)
    return add_predictor_args(parser, default_overrides=default_overrides)


def create_predictor_config(default_overrides=None, **overrides):
    values = build_predictor_defaults(default_overrides)
    values.update(overrides)
    return PredictorConfig(**values)
