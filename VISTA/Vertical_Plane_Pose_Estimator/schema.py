#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 0.001


@dataclass
class ROIConfig:
    x_min_ratio: float = 0.15
    x_max_ratio: float = 0.85
    y_min_ratio: float = 0.45
    y_max_ratio: float = 0.95


@dataclass
class PlaneRansacConfig:
    iterations: int = 80
    residual_threshold_m: float = 0.015
    max_abs_normal_y: float = 0.20
    min_abs_normal_z: float = 0.45
    min_inliers: int = 48
    min_inlier_ratio: float = 0.30
    min_confidence_inliers: int = 96
    target_inlier_ratio: float = 0.65


@dataclass
class TemporalConfig:
    yaw_alpha: float = 0.35
    dist_alpha: float = 0.30
    hold_confidence_threshold: float = 0.35


@dataclass
class VerticalPlaneEstimatorConfig:
    roi: ROIConfig = field(default_factory=ROIConfig)
    ransac: PlaneRansacConfig = field(default_factory=PlaneRansacConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    downsample_stride: int = 4
    min_depth_m: float = 0.20
    max_depth_m: float = 2.00
    min_camera_y_m: float = -0.20
    max_camera_y_m: float = 0.35
    target_distance_m: float = 0.50
    reference_point_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
