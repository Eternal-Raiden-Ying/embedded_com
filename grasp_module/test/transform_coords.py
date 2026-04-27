import argparse
import json
import logging
import os
import sys

import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(PARENT_DIR)
sys.path.append(ROOT_DIR)

try:
    from grasp_module.backend.utils.frames import FrameTransformer, parse_csv_floats
    from grasp_module.config.logging_config import configure_grasp_logger
    from grasp_module.config.predictor_config import create_predictor_config
except ImportError as e:
    print(f"-> Import Error: {e}")
    print(f"-> Current sys.path: {sys.path}")
    sys.exit(1)


logger = logging.getLogger("vision.grasp")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert camera-frame points/vectors into robot frame using main predictor config."
    )
    parser.add_argument(
        "--point_cam_m_csv",
        type=str,
        default="",
        help="Camera-frame point in meters, CSV format: x,y,z",
    )
    parser.add_argument(
        "--vector_cam_csv",
        type=str,
        default="",
        help="Camera-frame direction vector, CSV format: x,y,z",
    )
    parser.add_argument(
        "--robot_cam_rotation_csv",
        type=str,
        default="",
        help="Optional override of camera-to-robot rotation, row-major CSV format (9 values). Default comes from predictor config.",
    )
    parser.add_argument(
        "--robot_cam_translation_cm_csv",
        type=str,
        default="",
        help="Optional override of camera-to-robot translation, CSV format (3 values, cm). Default comes from predictor config.",
    )
    parser.add_argument(
        "--robot_calibration_translation_cm_csv",
        type=str,
        default="",
        help="Deprecated compatibility alias for robot_cam_translation_cm_csv.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON only.",
    )
    return parser


def vector_to_list(array_like):
    return [float(v) for v in np.asarray(array_like, dtype=np.float64).reshape(-1)]


def matrix_to_list(array_like):
    return [[float(v) for v in row] for row in np.asarray(array_like, dtype=np.float64)]


def build_transformer(args):
    overrides = {}
    if args.robot_cam_rotation_csv:
        overrides["robot_cam_rotation_csv"] = args.robot_cam_rotation_csv
    if args.robot_cam_translation_cm_csv:
        overrides["robot_cam_translation_cm_csv"] = args.robot_cam_translation_cm_csv
    if args.robot_calibration_translation_cm_csv:
        overrides["robot_calibration_translation_cm_csv"] = args.robot_calibration_translation_cm_csv
    cfg = create_predictor_config(**overrides)
    frames = FrameTransformer.from_config(cfg)
    return cfg, frames


def build_point_report(frames, point_cam_m):
    point_cam_m = np.asarray(point_cam_m, dtype=np.float64)
    point_robot_cm = frames.camera_point_to_robot_cm(point_cam_m)
    return {
        "camera_m": vector_to_list(point_cam_m),
        "robot_cm": vector_to_list(point_robot_cm),
    }


def build_vector_report(frames, vector_cam):
    vector_cam = np.asarray(vector_cam, dtype=np.float64)
    vector_robot = frames.camera_vector_to_robot(vector_cam)
    return {
        "camera": vector_to_list(vector_cam),
        "robot": vector_to_list(vector_robot),
    }


def main():
    configure_grasp_logger(level=logging.INFO)
    parser = build_parser()
    args = parser.parse_args()

    cfg, frames = build_transformer(args)
    payload = {
        "config": {
            "robot_cam_rotation_csv": cfg.robot_cam_rotation_csv,
            "robot_cam_translation_cm_csv": cfg.robot_cam_translation_cm_csv,
            "robot_calibration_translation_cm_csv": cfg.robot_calibration_translation_cm_csv,
        },
        "matrices": {
            "rotation_camera_to_robot": matrix_to_list(frames.rotation_camera_to_robot),
            "translation_camera_to_robot_cm": None
            if frames.translation_camera_to_robot_cm is None
            else vector_to_list(frames.translation_camera_to_robot_cm),
        },
    }

    if args.point_cam_m_csv:
        point_cam_m = parse_csv_floats(args.point_cam_m_csv, 3)
        payload["point_report"] = build_point_report(frames, point_cam_m)
    if args.vector_cam_csv:
        vector_cam = parse_csv_floats(args.vector_cam_csv, 3)
        payload["vector_report"] = build_vector_report(frames, vector_cam)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    logger.info("Transform config:")
    logger.info(" - robot_cam_rotation_csv: %s", cfg.robot_cam_rotation_csv)
    logger.info(" - robot_cam_translation_cm_csv: %s", cfg.robot_cam_translation_cm_csv)
    logger.info(" - robot_calibration_translation_cm_csv: %s", cfg.robot_calibration_translation_cm_csv)
    logger.info("Matrices:")
    logger.info(" - rotation_camera_to_robot:\n%s", frames.rotation_camera_to_robot)
    logger.info(" - translation_camera_to_robot_cm: %s", frames.translation_camera_to_robot_cm)

    if "point_report" in payload:
        report = payload["point_report"]
        logger.info("Point transform report:")
        logger.info(" - camera_m: %s", report["camera_m"])
        logger.info(" - robot_cm: %s", report["robot_cm"])

    if "vector_report" in payload:
        report = payload["vector_report"]
        logger.info("Vector transform report:")
        logger.info(" - camera: %s", report["camera"])
        logger.info(" - robot: %s", report["robot"])


if __name__ == "__main__":
    main()
