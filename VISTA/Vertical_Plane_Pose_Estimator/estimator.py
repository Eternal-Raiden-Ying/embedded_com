#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from .schema import CameraIntrinsics, VerticalPlaneEstimatorConfig
except ImportError:
    from schema import CameraIntrinsics, VerticalPlaneEstimatorConfig


@dataclass
class PlaneFitDebug:
    roi_box: Tuple[int, int, int, int]
    candidate_count: int
    inlier_count: int
    inlier_ratio: float
    confidence: float
    plane: Optional[Tuple[float, float, float, float]] = None
    normal: Optional[Tuple[float, float, float]] = None


@dataclass
class PoseEstimate:
    yaw_err_rad: float
    dist_err_m: float
    confidence: float
    valid: bool
    held: bool
    plane: Optional[Tuple[float, float, float, float]]
    debug: PlaneFitDebug


class VerticalPlanePoseEstimator:
    def __init__(self, cfg: Optional[VerticalPlaneEstimatorConfig] = None):
        self.cfg = cfg or VerticalPlaneEstimatorConfig()
        self._last_valid_yaw: Optional[float] = None
        self._last_valid_dist: Optional[float] = None
        self._last_valid_plane: Optional[Tuple[float, float, float, float]] = None

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(v)))

    @staticmethod
    def _normalize(v: np.ndarray) -> Optional[np.ndarray]:
        norm = float(np.linalg.norm(v))
        if norm < 1e-9:
            return None
        return v / norm

    def _roi_bounds(self, image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        height, width = image_shape[:2]
        x0 = int(round(self._clamp(self.cfg.roi.x_min_ratio, 0.0, 1.0) * width))
        x1 = int(round(self._clamp(self.cfg.roi.x_max_ratio, 0.0, 1.0) * width))
        y0 = int(round(self._clamp(self.cfg.roi.y_min_ratio, 0.0, 1.0) * height))
        y1 = int(round(self._clamp(self.cfg.roi.y_max_ratio, 0.0, 1.0) * height))
        x0 = max(0, min(width - 1, x0))
        x1 = max(x0 + 1, min(width, x1))
        y0 = max(0, min(height - 1, y0))
        y1 = max(y0 + 1, min(height, y1))
        return x0, y0, x1, y1

    def _extract_candidate_points(self, depth_map: np.ndarray, intrinsics: CameraIntrinsics) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        x0, y0, x1, y1 = self._roi_bounds(depth_map.shape[:2])
        step = max(1, int(self.cfg.downsample_stride))
        depth_roi = depth_map[y0:y1:step, x0:x1:step].astype(np.float32) * float(intrinsics.depth_scale)

        yy, xx = np.mgrid[y0:y1:step, x0:x1:step]
        valid = (depth_roi > float(self.cfg.min_depth_m)) & (depth_roi < float(self.cfg.max_depth_m))
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32), (x0, y0, x1, y1)

        z = depth_roi[valid]
        u = xx[valid].astype(np.float32)
        v = yy[valid].astype(np.float32)
        x = (u - float(intrinsics.cx)) * z / float(intrinsics.fx)
        y = (v - float(intrinsics.cy)) * z / float(intrinsics.fy)

        points = np.stack([x, y, z], axis=1)
        height_mask = (
            (points[:, 1] >= float(self.cfg.min_camera_y_m)) &
            (points[:, 1] <= float(self.cfg.max_camera_y_m))
        )
        return points[height_mask], (x0, y0, x1, y1)

    def _is_vertical_front_plane_normal(self, normal: np.ndarray) -> bool:
        if abs(float(normal[1])) > float(self.cfg.ransac.max_abs_normal_y):
            return False
        if abs(float(normal[2])) < float(self.cfg.ransac.min_abs_normal_z):
            return False
        return True

    def _orient_normal_toward_camera(self, normal: np.ndarray, d: float) -> Tuple[np.ndarray, float]:
        if float(normal[2]) > 0.0:
            return -normal, -float(d)
        return normal, float(d)

    def _plane_from_points(self, pts: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        if pts.shape[0] < 3:
            return None
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        normal = self._normalize(vh[-1, :])
        if normal is None:
            return None
        d = -float(np.dot(normal, centroid))
        normal, d = self._orient_normal_toward_camera(normal, d)
        if not self._is_vertical_front_plane_normal(normal):
            return None
        return normal, d

    @staticmethod
    def _point_plane_residual(points: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
        return np.abs(points @ normal + float(d))

    def _fit_plane_ransac(self, points_xyz: np.ndarray) -> Tuple[Optional[Tuple[np.ndarray, float]], np.ndarray]:
        if points_xyz.shape[0] < 3:
            return None, np.zeros((0,), dtype=bool)

        best = None
        best_inliers = None
        best_count = 0
        best_mean_residual = None

        for _ in range(max(8, int(self.cfg.ransac.iterations))):
            idx = np.random.choice(points_xyz.shape[0], size=3, replace=False)
            plane = self._plane_from_points(points_xyz[idx])
            if plane is None:
                continue
            normal, d = plane
            residual = self._point_plane_residual(points_xyz, normal, d)
            inliers = residual <= float(self.cfg.ransac.residual_threshold_m)
            count = int(inliers.sum())
            if count < int(self.cfg.ransac.min_inliers):
                continue
            mean_residual = float(residual[inliers].mean()) if count > 0 else float("inf")
            if count > best_count or (count == best_count and best_mean_residual is not None and mean_residual < best_mean_residual):
                best = plane
                best_inliers = inliers
                best_count = count
                best_mean_residual = mean_residual

        if best is None or best_inliers is None:
            return None, np.zeros((points_xyz.shape[0],), dtype=bool)

        refined = self._plane_from_points(points_xyz[best_inliers])
        if refined is None:
            return best, best_inliers
        normal, d = refined
        residual = self._point_plane_residual(points_xyz, normal, d)
        inliers = residual <= float(self.cfg.ransac.residual_threshold_m)
        return (normal, d), inliers

    def _compute_confidence(self, inlier_count: int, candidate_count: int) -> float:
        if candidate_count <= 0:
            return 0.0
        ratio = float(inlier_count) / float(candidate_count)
        count_conf = min(1.0, float(inlier_count) / float(max(1, self.cfg.ransac.min_confidence_inliers)))
        ratio_conf = min(1.0, ratio / float(max(1e-6, self.cfg.ransac.target_inlier_ratio)))
        return 0.5 * count_conf + 0.5 * ratio_conf

    @staticmethod
    def _blend_scalar(prev: float, cur: float, alpha: float) -> float:
        a = max(0.0, min(1.0, float(alpha)))
        return (1.0 - a) * float(prev) + a * float(cur)

    @staticmethod
    def _blend_angle(prev: float, cur: float, alpha: float) -> float:
        delta = math.atan2(math.sin(cur - prev), math.cos(cur - prev))
        return prev + max(0.0, min(1.0, float(alpha))) * delta

    def _solve_pose_from_plane(self, normal: np.ndarray, d: float) -> Tuple[float, float]:
        yaw_err = math.atan2(float(normal[0]), -float(normal[2]))
        ref = np.asarray(self.cfg.reference_point_xyz, dtype=np.float32)
        signed_distance = float(np.dot(normal, ref) + float(d))
        dist_err = signed_distance - float(self.cfg.target_distance_m)
        return yaw_err, dist_err

    def estimate_from_points(self, points_xyz: np.ndarray) -> PoseEstimate:
        candidate_count = int(points_xyz.shape[0])
        debug = PlaneFitDebug(
            roi_box=(0, 0, 0, 0),
            candidate_count=candidate_count,
            inlier_count=0,
            inlier_ratio=0.0,
            confidence=0.0,
            plane=None,
            normal=None,
        )

        plane, inlier_mask = self._fit_plane_ransac(points_xyz)
        if plane is None:
            return self._hold_or_invalid(debug)

        normal, d = plane
        inlier_count = int(inlier_mask.sum())
        inlier_ratio = float(inlier_count) / float(max(1, candidate_count))
        confidence = self._compute_confidence(inlier_count, candidate_count)
        debug.inlier_count = inlier_count
        debug.inlier_ratio = inlier_ratio
        debug.confidence = confidence
        debug.plane = (float(normal[0]), float(normal[1]), float(normal[2]), float(d))
        debug.normal = (float(normal[0]), float(normal[1]), float(normal[2]))

        yaw_err, dist_err = self._solve_pose_from_plane(normal, d)
        if confidence < float(self.cfg.temporal.hold_confidence_threshold):
            return self._hold_or_invalid(debug)

        if self._last_valid_yaw is not None and self._last_valid_dist is not None:
            yaw_err = self._blend_angle(self._last_valid_yaw, yaw_err, self.cfg.temporal.yaw_alpha)
            dist_err = self._blend_scalar(self._last_valid_dist, dist_err, self.cfg.temporal.dist_alpha)

        self._last_valid_yaw = yaw_err
        self._last_valid_dist = dist_err
        self._last_valid_plane = debug.plane
        return PoseEstimate(
            yaw_err_rad=float(yaw_err),
            dist_err_m=float(dist_err),
            confidence=float(confidence),
            valid=True,
            held=False,
            plane=debug.plane,
            debug=debug,
        )

    def _hold_or_invalid(self, debug: PlaneFitDebug) -> PoseEstimate:
        if self._last_valid_yaw is not None and self._last_valid_dist is not None:
            return PoseEstimate(
                yaw_err_rad=float(self._last_valid_yaw),
                dist_err_m=float(self._last_valid_dist),
                confidence=float(debug.confidence),
                valid=True,
                held=True,
                plane=self._last_valid_plane,
                debug=debug,
            )
        return PoseEstimate(
            yaw_err_rad=0.0,
            dist_err_m=0.0,
            confidence=float(debug.confidence),
            valid=False,
            held=False,
            plane=None,
            debug=debug,
        )

    def estimate_from_depth(self, depth_map: np.ndarray, intrinsics: CameraIntrinsics) -> PoseEstimate:
        points_xyz, roi_box = self._extract_candidate_points(depth_map, intrinsics)
        estimate = self.estimate_from_points(points_xyz)
        estimate.debug.roi_box = roi_box
        return estimate


def _make_mock_vertical_plane_points(
    yaw_deg: float = 18.0,
    orth_dist_m: float = 0.58,
    width_m: float = 0.80,
    height_m: float = 0.45,
    x_samples: int = 36,
    y_samples: int = 20,
    noise_std_m: float = 0.006,
    outlier_count: int = 40,
) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    normal = np.asarray([math.sin(yaw), 0.0, -math.cos(yaw)], dtype=np.float32)
    normal = normal / np.linalg.norm(normal)
    center = -orth_dist_m * normal
    tangent_x = np.asarray([math.cos(yaw), 0.0, math.sin(yaw)], dtype=np.float32)
    tangent_y = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    xs = np.linspace(-width_m * 0.5, width_m * 0.5, x_samples)
    ys = np.linspace(-height_m * 0.5, height_m * 0.5, y_samples)
    grid = []
    for x in xs:
        for y in ys:
            pt = center + x * tangent_x + y * tangent_y
            grid.append(pt)
    points = np.asarray(grid, dtype=np.float32)
    points += np.random.normal(0.0, noise_std_m, size=points.shape).astype(np.float32)

    if outlier_count > 0:
        outliers = np.empty((outlier_count, 3), dtype=np.float32)
        outliers[:, 0] = np.random.uniform(-0.8, 0.8, size=(outlier_count,))
        outliers[:, 1] = np.random.uniform(-0.4, 0.4, size=(outlier_count,))
        outliers[:, 2] = np.random.uniform(0.25, 1.40, size=(outlier_count,))
        points = np.concatenate([points, outliers], axis=0)
    return points


def main() -> None:
    cfg = VerticalPlaneEstimatorConfig()
    estimator = VerticalPlanePoseEstimator(cfg)

    mock_points = _make_mock_vertical_plane_points()
    t0 = time.perf_counter()
    estimate = estimator.estimate_from_points(mock_points)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print("=== Mock Vertical Plane Pose Estimator ===")
    print("candidate_count =", estimate.debug.candidate_count)
    print("inlier_count    =", estimate.debug.inlier_count)
    print("inlier_ratio    = %.4f" % estimate.debug.inlier_ratio)
    print("confidence      = %.4f" % estimate.confidence)
    print("held            =", estimate.held)
    print("valid           =", estimate.valid)
    print("elapsed_ms      = %.3f" % elapsed_ms)
    if estimate.plane is not None:
        A, B, C, D = estimate.plane
        print("plane           = %.6f x + %.6f y + %.6f z + %.6f = 0" % (A, B, C, D))
    print("yaw_err_rad     = %.6f" % estimate.yaw_err_rad)
    print("yaw_err_deg     = %.3f" % (estimate.yaw_err_rad * 180.0 / math.pi))
    print("dist_err_m      = %.6f" % estimate.dist_err_m)


if __name__ == "__main__":
    main()
