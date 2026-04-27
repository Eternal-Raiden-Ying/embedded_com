import logging
from dataclasses import dataclass

import numpy as np


logger = logging.getLogger("vision.grasp")


def parse_csv_floats(raw_value, expected_len):
    values = [float(item.strip()) for item in str(raw_value).split(",") if item.strip()]
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated values, got {len(values)}")
    return values


@dataclass
class FrameTransformer:
    rotation_camera_to_robot: np.ndarray | None = None
    translation_camera_to_robot_cm: np.ndarray | None = None

    def __post_init__(self):
        if self.rotation_camera_to_robot is None:
            self.rotation_camera_to_robot = np.eye(3, dtype=np.float64)

    @classmethod
    def from_config(cls, cfgs):
        rotation = None
        translation = None

        rotation_csv = getattr(cfgs, "robot_cam_rotation_csv", "")
        if rotation_csv:
            rotation = np.asarray(parse_csv_floats(rotation_csv, 9), dtype=np.float64).reshape(3, 3)
            logger.info("Loaded camera-to-robot rotation:\n%s", rotation)

        translation_csv = getattr(cfgs, "robot_cam_translation_cm_csv", "")
        if not translation_csv:
            translation_csv = getattr(cfgs, "robot_calibration_translation_cm_csv", "")
        if translation_csv:
            translation = np.asarray(parse_csv_floats(translation_csv, 3), dtype=np.float64)
            logger.info("Loaded camera-to-robot translation (cm): %s", translation)

        return cls(
            rotation_camera_to_robot=rotation,
            translation_camera_to_robot_cm=translation,
        )

    def camera_vector_to_robot(self, vector):
        vector = np.asarray(vector, dtype=np.float64)
        return self.rotation_camera_to_robot @ vector

    def camera_point_to_robot_cm(self, point_m):
        point_m = np.asarray(point_m, dtype=np.float64)
        rotated = 100.0 * (self.rotation_camera_to_robot @ point_m)
        if self.translation_camera_to_robot_cm is None:
            return rotated
        return rotated + self.translation_camera_to_robot_cm
