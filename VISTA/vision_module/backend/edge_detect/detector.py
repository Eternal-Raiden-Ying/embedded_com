#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from .schema import DetectorConfig
except ImportError:
    from schema import DetectorConfig


@dataclass
class CameraCalib:
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 0.001


@dataclass
class EdgeDetectResult:
    edge_found: bool
    yaw_err_rad: float
    dist_err_m: float
    edge_confidence: float
    line_k: Optional[float] = None
    line_b: Optional[float] = None
    point_count: int = 0
    table_point_count: int = 0
    raw_found: bool = False
    pose_found: bool = False
    valid_for_control: bool = False
    pose_source: str = "none"
    plane_found: bool = False
    line_found: bool = False
    plane_confidence: float = 0.0
    line_confidence: float = 0.0
    plane_residual_mean: float = 0.0
    line_residual_mean: float = 0.0
    plane_x_span_m: float = 0.0
    line_x_span_m: float = 0.0
    candidate_count: int = 0
    inlier_count: int = 0
    stable_count: int = 0
    roi_source: str = ""
    front_face_area_ratio: float = 0.0
    reject_reason: str = ""
    plane_yaw_err_rad: Optional[float] = None
    plane_dist_err_m: Optional[float] = None
    line_yaw_err_rad: Optional[float] = None
    line_dist_err_m: Optional[float] = None
    plane_k: Optional[float] = None
    plane_b: Optional[float] = None
    image_line_k: Optional[float] = None
    image_line_b: Optional[float] = None
    upper_line_found: bool = False
    upper_line_confidence: float = 0.0
    upper_line_candidate_count: int = 0
    upper_line_inlier_count: int = 0
    upper_line_residual_mean: float = 0.0
    upper_line_x_span_m: float = 0.0
    upper_line_y_norm_mean: float = 0.0
    upper_line_k: Optional[float] = None
    upper_line_b: Optional[float] = None
    upper_line_yaw_err_rad: Optional[float] = None
    upper_line_dist_err_m: Optional[float] = None
    lower_line_found: bool = False
    lower_line_confidence: float = 0.0
    lower_line_candidate_count: int = 0
    lower_line_inlier_count: int = 0
    lower_line_residual_mean: float = 0.0
    lower_line_x_span_m: float = 0.0
    lower_line_y_norm_mean: float = 0.0
    lower_line_k: Optional[float] = None
    lower_line_b: Optional[float] = None
    lower_line_yaw_err_rad: Optional[float] = None
    lower_line_dist_err_m: Optional[float] = None
    selected_line_type: str = "none"
    table_geometry_score: float = 0.0
    front_plane_score: float = 0.0
    line_score: float = 0.0
    plane_line_consistency_score: float = 0.0
    roi_boundary_score: float = 0.0
    temporal_score: float = 0.0
    geometry_reject_reason: str = ""
    usable_for_approach: bool = False
    usable_for_alignment: bool = False
    usable_for_stop: bool = False
    control_level: str = "none"
    control_reject_reason: str = ""
    selected_line_plane_boundary_dist: float = 0.0
    selected_line_plane_consistency: float = 0.0
    line_reject_reason: str = ""
    line_drift_rejected: bool = False
    object_like_line_score: float = 0.0
    final_pose_source: str = "none"


def load_calib(json_path: Path) -> Tuple[CameraCalib, float]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    calib = CameraCalib(
        fx=float(data["fx"]),
        fy=float(data["fy"]),
        cx=float(data["cx"]),
        cy=float(data["cy"]),
        depth_scale=float(data.get("depth_scale", 0.001)),
    )
    return calib, float(data["target_dist_m"])


class OnlineTableEdgeDetector:
    def __init__(self, calib: CameraCalib, cfg: DetectorConfig, target_dist_m: float):
        self.calib = calib
        self.cfg = cfg
        self.target_dist_m = float(target_dist_m)
        self._rng = np.random.default_rng(int(cfg.random_seed))
        self._stable_count = 0
        self._last_pose: Optional[Tuple[float, float]] = None
        self._last_selected_line_type = "none"
        self._last_pose_source = "none"

    def _cfg_roi(self) -> Tuple[int, int, int, int]:
        return (
            int(self.cfg.roi_x0),
            int(self.cfg.roi_y0),
            int(self.cfg.roi_x1),
            int(self.cfg.roi_y1),
        )

    @staticmethod
    def _parse_roi_box(value: Any) -> Optional[Tuple[int, int, int, int]]:
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            return None
        try:
            x0, y0, x1, y1 = [int(round(float(v))) for v in value[:4]]
        except (TypeError, ValueError):
            return None
        return x0, y0, x1, y1

    @staticmethod
    def _clip_roi_box(roi_box: Tuple[int, int, int, int], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
        x0, y0, x1, y1 = roi_box
        x0 = max(0, min(int(width), int(x0)))
        x1 = max(0, min(int(width), int(x1)))
        y0 = max(0, min(int(height), int(y0)))
        y1 = max(0, min(int(height), int(y1)))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def _resolve_roi(self, depth_img: np.ndarray, roi_override=None) -> Tuple[int, int, int, int]:
        if depth_img is None or len(getattr(depth_img, "shape", ())) < 2:
            raise ValueError("depth image must be a 2D array")
        height, width = int(depth_img.shape[0]), int(depth_img.shape[1])
        override_box = self._parse_roi_box(roi_override)
        if override_box is not None:
            clipped = self._clip_roi_box(override_box, width, height)
            if clipped is not None:
                return clipped

        cfg_box = self._parse_roi_box(self._cfg_roi())
        clipped_cfg = self._clip_roi_box(cfg_box, width, height) if cfg_box is not None else None
        if clipped_cfg is None:
            raise ValueError(f"invalid depth ROI: override={roi_override!r} cfg={self._cfg_roi()!r} image_shape={depth_img.shape!r}")
        return clipped_cfg

    def _preprocess_depth(self, depth_img: np.ndarray, roi_override=None):
        x0, y0, x1, y1 = self._resolve_roi(depth_img, roi_override=roi_override)
        depth_roi = depth_img[y0:y1, x0:x1]
        ksize = int(self.cfg.depth_median_ksize)
        if ksize > 1 and ksize % 2 == 1:
            depth_filtered = cv2.medianBlur(depth_roi, ksize)
        else:
            depth_filtered = depth_roi
        depth_m = depth_filtered.astype(np.float32) * float(self.calib.depth_scale)
        mask = (depth_m > float(self.cfg.z_min)) & (depth_m < float(self.cfg.z_max))
        return mask, depth_m, (x0, y0, x1, y1)

    def _depth_to_3d(self, depth_m: np.ndarray, mask: np.ndarray, roi_box) -> np.ndarray:
        x0, y0, x1, y1 = roi_box
        u, v = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        u_valid, v_valid, z_valid = u[mask], v[mask], depth_m[mask]
        x_c = (u_valid - self.calib.cx) * z_valid / self.calib.fx
        y_c = (v_valid - self.calib.cy) * z_valid / self.calib.fy
        return np.vstack((x_c, y_c, z_valid)).T

    def _depth_to_xyz_maps(self, depth_m: np.ndarray, roi_box) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x0, y0, x1, y1 = roi_box
        u, v = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
        z = depth_m.astype(np.float32, copy=False)
        x = (u - float(self.calib.cx)) * z / float(self.calib.fx)
        y = (v - float(self.calib.cy)) * z / float(self.calib.fy)
        return x, y, z

    def _find_table_plane(self, pc_cam: np.ndarray) -> np.ndarray:
        y = pc_cam[:, 1]
        lo = float(self.cfg.table_y_min)
        hi = float(self.cfg.table_y_max)
        return (y > lo) & (y < hi)

    def _fit_line_on_inliers(self, x: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
        coeff = np.polyfit(x, z, 1)
        return float(coeff[0]), float(coeff[1])

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float(default)
        return out if math.isfinite(out) else float(default)

    @staticmethod
    def _sample_pixels(xs: np.ndarray, ys: np.ndarray, limit: int = 1200) -> List[List[int]]:
        if xs.size <= 0 or ys.size <= 0:
            return []
        count = int(min(xs.size, ys.size))
        if count > limit:
            idx = np.linspace(0, count - 1, limit).astype(np.int64)
            xs = xs[idx]
            ys = ys[idx]
        return [[int(x), int(y)] for x, y in zip(xs.tolist(), ys.tolist())]

    def _fit_ransac_xz(self, points: np.ndarray, threshold_m: Optional[float] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "found": False,
            "k": None,
            "b": None,
            "yaw": 0.0,
            "dist": 0.0,
            "confidence": 0.0,
            "inlier_count": 0,
            "residual_mean": float("inf"),
            "x_span": 0.0,
            "inlier_mask": None,
        }
        if points is None or points.shape[0] < 2:
            return out
        x = points[:, 0].astype(np.float64)
        z = points[:, 2].astype(np.float64)
        finite = np.isfinite(x) & np.isfinite(z)
        if int(finite.sum()) < 2:
            return out
        x = x[finite]
        z = z[finite]
        n = int(x.shape[0])
        threshold = float(threshold_m if threshold_m is not None else self.cfg.residual_threshold_m)
        best_mask = None
        best_count = 0
        best_error = float("inf")
        for _ in range(max(8, int(self.cfg.ransac_iters))):
            idx = self._rng.choice(n, size=2, replace=False)
            x1, x2 = float(x[idx[0]]), float(x[idx[1]])
            z1, z2 = float(z[idx[0]]), float(z[idx[1]])
            if abs(x2 - x1) < 1e-6:
                continue
            k = (z2 - z1) / (x2 - x1)
            b = z1 - k * x1
            residual = np.abs(z - (k * x + b))
            mask = residual <= threshold
            count = int(mask.sum())
            mean_err = float(residual[mask].mean()) if count > 0 else float("inf")
            if count > best_count or (count == best_count and mean_err < best_error):
                best_mask = mask
                best_count = count
                best_error = mean_err
        if best_mask is None or best_count < 2:
            return out
        k, b = self._fit_line_on_inliers(x[best_mask], z[best_mask])
        residual = np.abs(z - (k * x + b))
        inlier_mask = residual <= threshold
        inlier_count = int(inlier_mask.sum())
        if inlier_count < 2:
            return out
        residual_mean = float(residual[inlier_mask].mean())
        x_span = float(x[inlier_mask].max() - x[inlier_mask].min())
        conf = float(inlier_count) / float(max(1, n))
        out.update(
            {
                "found": True,
                "k": float(k),
                "b": float(b),
                "yaw": float(math.atan(k)),
                "dist": float(b - float(self.target_dist_m)),
                "confidence": conf,
                "inlier_count": inlier_count,
                "residual_mean": residual_mean,
                "x_span": x_span,
                "inlier_mask": inlier_mask,
            }
        )
        return out

    def _fit_edge_line(self, edge_points: np.ndarray):
        fit = self._fit_ransac_xz(edge_points, threshold_m=float(self.cfg.residual_threshold_m))
        if not bool(fit.get("found")):
            return False, 0.0, 0.0, 0.0, None
        return (
            True,
            float(fit["yaw"]),
            float(fit["dist"]),
            float(fit["confidence"]),
            (float(fit["k"]), float(fit["b"])),
        )

    def _estimate_front_plane(self, depth_m: np.ndarray, valid_mask: np.ndarray, roi_box) -> Dict[str, Any]:
        x_map, y_map, z_map = self._depth_to_xyz_maps(depth_m, roi_box)
        h, w = depth_m.shape[:2]
        empty = {
            "found": False,
            "candidate_mask": np.zeros((h, w), dtype=bool),
            "inlier_mask": np.zeros((h, w), dtype=bool),
            "candidate_count": 0,
            "inlier_count": 0,
            "yaw": 0.0,
            "dist": 0.0,
            "confidence": 0.0,
            "residual_mean": float("inf"),
            "x_span": 0.0,
            "area_ratio": 0.0,
            "k": None,
            "b": None,
            "normal": None,
            "plane": None,
            "pixels": [],
            "image_y_min": None,
            "image_y_max": None,
            "image_y_mean": None,
            "image_x_min": None,
            "image_x_max": None,
        }
        if h < 3 or w < 3 or int(valid_mask.sum()) < int(self.cfg.min_all_points):
            return empty
        core_valid = valid_mask[1:-1, 1:-1] & valid_mask[1:-1, :-2] & valid_mask[1:-1, 2:] & valid_mask[:-2, 1:-1] & valid_mask[2:, 1:-1]
        vx = np.stack(
            (
                x_map[1:-1, 2:] - x_map[1:-1, :-2],
                y_map[1:-1, 2:] - y_map[1:-1, :-2],
                z_map[1:-1, 2:] - z_map[1:-1, :-2],
            ),
            axis=-1,
        )
        vy = np.stack(
            (
                x_map[2:, 1:-1] - x_map[:-2, 1:-1],
                y_map[2:, 1:-1] - y_map[:-2, 1:-1],
                z_map[2:, 1:-1] - z_map[:-2, 1:-1],
            ),
            axis=-1,
        )
        normals = np.cross(vx, vy)
        norm = np.linalg.norm(normals, axis=-1)
        normal_valid = core_valid & np.isfinite(norm) & (norm > 1e-6)
        normals_unit = np.zeros_like(normals, dtype=np.float32)
        normals_unit[normal_valid] = normals[normal_valid] / norm[normal_valid, None]
        abs_ny = np.abs(normals_unit[:, :, 1])
        abs_nz = np.abs(normals_unit[:, :, 2])
        candidate_core = (
            normal_valid
            & (abs_ny <= float(self.cfg.plane_max_abs_normal_y))
            & (abs_nz >= float(self.cfg.plane_min_abs_normal_z))
        )
        candidate_mask = np.zeros((h, w), dtype=bool)
        candidate_mask[1:-1, 1:-1] = candidate_core
        ys, xs = np.nonzero(candidate_mask)
        candidate_count = int(len(xs))
        empty["candidate_mask"] = candidate_mask
        empty["candidate_count"] = candidate_count
        if candidate_count < int(self.cfg.plane_min_inliers):
            empty["pixels"] = self._sample_pixels(xs + int(roi_box[0]), ys + int(roi_box[1]))
            return empty
        points = np.vstack((x_map[candidate_mask], y_map[candidate_mask], z_map[candidate_mask])).T.astype(np.float64)
        if points.shape[0] > 5000:
            idx = self._rng.choice(points.shape[0], size=5000, replace=False)
            fit_points = points[idx]
        else:
            fit_points = points
        best_plane = None
        best_mask = None
        best_count = 0
        best_error = float("inf")
        threshold = float(self.cfg.plane_max_residual_m)
        n = int(fit_points.shape[0])
        for _ in range(max(24, int(self.cfg.ransac_iters))):
            idx = self._rng.choice(n, size=3, replace=False)
            p1, p2, p3 = fit_points[idx]
            normal = np.cross(p2 - p1, p3 - p1)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm < 1e-8:
                continue
            normal = normal / normal_norm
            if abs(float(normal[1])) > float(self.cfg.plane_max_abs_normal_y) or abs(float(normal[2])) < float(self.cfg.plane_min_abs_normal_z):
                continue
            d = -float(np.dot(normal, p1))
            residual = np.abs(fit_points @ normal + d)
            mask = residual <= threshold
            count = int(mask.sum())
            mean_err = float(residual[mask].mean()) if count > 0 else float("inf")
            if count > best_count or (count == best_count and mean_err < best_error):
                best_plane = (normal, d)
                best_mask = mask
                best_count = count
                best_error = mean_err
        if best_plane is None or best_mask is None or best_count < int(self.cfg.plane_min_inliers):
            empty["pixels"] = self._sample_pixels(xs + int(roi_box[0]), ys + int(roi_box[1]))
            return empty
        normal, d = best_plane
        residual_all = np.abs(points @ normal + d)
        inlier_mask = residual_all <= threshold
        inlier_count = int(inlier_mask.sum())
        if inlier_count < int(self.cfg.plane_min_inliers) or abs(float(normal[2])) < 1e-6:
            empty["pixels"] = self._sample_pixels(xs + int(roi_box[0]), ys + int(roi_box[1]))
            return empty
        inliers = points[inlier_mask]
        residual_mean = float(residual_all[inlier_mask].mean())
        x_span = float(inliers[:, 0].max() - inliers[:, 0].min()) if inliers.shape[0] else 0.0
        area_ratio = float(inlier_count) / float(max(1, int(valid_mask.sum())))
        k = -float(normal[0]) / float(normal[2])
        b = -float(d) / float(normal[2])
        yaw = float(math.atan(k))
        dist = float(b - float(self.target_dist_m))
        residual_score = max(0.0, 1.0 - residual_mean / max(1e-6, float(self.cfg.front_plane_max_residual_m)))
        span_score = min(1.0, x_span / max(1e-6, float(self.cfg.front_plane_min_x_span_m)))
        area_score = min(1.0, area_ratio / max(1e-6, float(self.cfg.front_plane_min_area_ratio)))
        inlier_score = min(1.0, inlier_count / max(1.0, float(self.cfg.plane_min_inliers) * 2.0))
        conf = float(np.clip(0.30 * residual_score + 0.25 * span_score + 0.25 * area_score + 0.20 * inlier_score, 0.0, 1.0))
        all_xs = xs + int(roi_box[0])
        all_ys = ys + int(roi_box[1])
        inlier_xs = all_xs[inlier_mask]
        inlier_ys = all_ys[inlier_mask]
        plane_inlier_mask = np.zeros((h, w), dtype=bool)
        plane_inlier_mask[ys[inlier_mask], xs[inlier_mask]] = True
        return {
            "found": True,
            "candidate_mask": candidate_mask,
            "inlier_mask": plane_inlier_mask,
            "candidate_count": candidate_count,
            "inlier_count": inlier_count,
            "yaw": yaw,
            "dist": dist,
            "confidence": conf,
            "residual_mean": residual_mean,
            "x_span": x_span,
            "area_ratio": area_ratio,
            "k": float(k),
            "b": float(b),
            "normal": [float(v) for v in normal.tolist()],
            "plane": [float(normal[0]), float(normal[1]), float(normal[2]), float(d)],
            "pixels": self._sample_pixels(inlier_xs, inlier_ys),
            "image_y_min": int(inlier_ys.min()) if inlier_ys.size else None,
            "image_y_max": int(inlier_ys.max()) if inlier_ys.size else None,
            "image_y_mean": float(inlier_ys.mean()) if inlier_ys.size else None,
            "image_x_min": int(inlier_xs.min()) if inlier_xs.size else None,
            "image_x_max": int(inlier_xs.max()) if inlier_xs.size else None,
        }

    def _empty_line_hypothesis(self, line_type: str) -> Dict[str, Any]:
        return {
            "type": line_type,
            "found": False,
            "selected": False,
            "candidate_count": 0,
            "inlier_count": 0,
            "yaw": 0.0,
            "dist": 0.0,
            "confidence": 0.0,
            "selection_score": 0.0,
            "residual_mean": float("inf"),
            "x_span": 0.0,
            "y_norm_mean": 0.0,
            "k": None,
            "b": None,
            "points": np.empty((0, 3), dtype=np.float32),
            "pixels": [],
            "inlier_pixels": [],
            "image_line_k": None,
            "image_line_b": None,
            "boundary_touch_ratio": 0.0,
            "plane_boundary_dist_px": float("inf"),
            "plane_boundary_consistency": 0.0,
            "object_like_score": 0.0,
            "drift_rejected": False,
            "reject_reason": "no_candidates",
        }

    def _disabled_line_result(self) -> Dict[str, Any]:
        upper = self._empty_line_hypothesis("upper_crease")
        lower = self._empty_line_hypothesis("lower_contact")
        selected = self._empty_line_hypothesis("selected")
        for item in (upper, lower, selected):
            item["reject_reason"] = "disabled_plane_only"
        return {
            "found": False,
            "selected_line_type": "none",
            "candidate_count": 0,
            "inlier_count": 0,
            "yaw": 0.0,
            "dist": 0.0,
            "confidence": 0.0,
            "residual_mean": float("inf"),
            "x_span": 0.0,
            "y_norm_mean": 0.0,
            "k": None,
            "b": None,
            "points": np.empty((0, 3), dtype=np.float32),
            "pixels": [],
            "inlier_pixels": [],
            "image_line_k": None,
            "image_line_b": None,
            "boundary_touch_ratio": 0.0,
            "plane_boundary_dist_px": float("inf"),
            "plane_boundary_consistency": 0.0,
            "object_like_score": 0.0,
            "drift_rejected": False,
            "reject_reason": "disabled_plane_only",
            "upper": upper,
            "lower": lower,
            "upper_pixels": [],
            "lower_pixels": [],
        }

    def _line_boundary_touch_ratio(self, pixels: List[List[int]], roi_box) -> float:
        if not pixels:
            return 0.0
        x0, y0, x1, y1 = [int(v) for v in roi_box]
        margin = max(0, int(self.cfg.roi_boundary_margin_px))
        touched = 0
        for px, py in pixels:
            if px <= x0 + margin or px >= x1 - 1 - margin or py <= y0 + margin or py >= y1 - 1 - margin:
                touched += 1
        return float(touched) / float(max(1, len(pixels)))

    def _line_plane_boundary_score(
        self,
        line_type: str,
        pixels: List[List[int]],
        plane: Dict[str, Any],
    ) -> Tuple[float, float]:
        if not bool(plane.get("found")) or not pixels:
            return float("inf"), 1.0
        boundary_key = "image_y_min" if line_type == "upper_crease" else "image_y_max"
        boundary_y = plane.get(boundary_key)
        if boundary_y is None:
            return float("inf"), 1.0
        try:
            line_y = float(np.mean([float(p[1]) for p in pixels if isinstance(p, (list, tuple)) and len(p) >= 2]))
            boundary = float(boundary_y)
        except Exception:
            return float("inf"), 0.0
        if not math.isfinite(line_y) or not math.isfinite(boundary):
            return float("inf"), 0.0
        dist_px = abs(line_y - boundary)
        soft = max(0.0, float(self.cfg.line_plane_boundary_soft_dist_px))
        hard = max(soft + 1.0, float(self.cfg.line_plane_boundary_max_dist_px))
        consistency = 1.0 if dist_px <= soft else max(0.0, 1.0 - (dist_px - soft) / (hard - soft))
        return float(dist_px), float(consistency)

    def _object_like_line_score(
        self,
        candidate_count: int,
        inlier_count: int,
        x_span: float,
        boundary_consistency: float,
        boundary_touch_ratio: float,
    ) -> float:
        span_ref = max(1e-6, float(self.cfg.line_select_min_x_span_m) * 1.5)
        span_short = max(0.0, 1.0 - float(x_span) / span_ref)
        count_ref = max(1.0, float(self.cfg.trend_min_candidate_count) * 2.0)
        sparse = max(0.0, 1.0 - float(candidate_count) / count_ref)
        inlier_ratio = float(inlier_count) / float(max(1, candidate_count))
        isolated = max(0.0, 1.0 - inlier_ratio)
        boundary_far = max(0.0, 1.0 - float(boundary_consistency))
        roi_touch = min(1.0, float(boundary_touch_ratio) / max(1e-6, float(self.cfg.roi_boundary_max_touch_ratio)))
        score = 0.35 * span_short + 0.20 * sparse + 0.15 * isolated + 0.25 * boundary_far + 0.05 * roi_touch
        return float(np.clip(score, 0.0, 1.0))

    def _fit_line_hypothesis(
        self,
        line_type: str,
        points: np.ndarray,
        pixels: List[List[int]],
        y_norms: np.ndarray,
        plane: Dict[str, Any],
        roi_box,
    ) -> Dict[str, Any]:
        out = self._empty_line_hypothesis(line_type)
        candidate_count = int(points.shape[0]) if isinstance(points, np.ndarray) else 0
        out["candidate_count"] = candidate_count
        out["pixels"] = pixels[:1600]
        out["points"] = points if isinstance(points, np.ndarray) else np.empty((0, 3), dtype=np.float32)
        out["y_norm_mean"] = float(np.mean(y_norms)) if candidate_count > 0 else 0.0
        out["boundary_touch_ratio"] = self._line_boundary_touch_ratio(pixels, roi_box)
        if candidate_count < int(self.cfg.trend_min_candidate_count):
            out["reject_reason"] = "too_few_candidates"
            return out
        fit = self._fit_ransac_xz(points, threshold_m=float(self.cfg.line_select_max_residual_m))
        if not bool(fit.get("found")):
            out["reject_reason"] = "ransac_failed"
            return out
        inlier_mask = fit.get("inlier_mask")
        inlier_pixels: List[List[int]] = []
        inlier_y_norms = y_norms
        if isinstance(inlier_mask, np.ndarray) and len(inlier_mask) == len(pixels):
            inlier_pixels = [pixels[i] for i, keep in enumerate(inlier_mask.tolist()) if keep]
            inlier_y_norms = y_norms[inlier_mask] if len(y_norms) == len(inlier_mask) else y_norms
        image_line_k = None
        image_line_b = None
        if len(inlier_pixels) >= 2:
            px = np.array([p[0] for p in inlier_pixels], dtype=np.float64)
            py = np.array([p[1] for p in inlier_pixels], dtype=np.float64)
            if float(px.max() - px.min()) > 1.0:
                coeff = np.polyfit(px, py, 1)
                image_line_k = float(coeff[0])
                image_line_b = float(coeff[1])
        residual = float(fit.get("residual_mean", float("inf")))
        x_span = float(fit.get("x_span", 0.0) or 0.0)
        residual_score = max(0.0, 1.0 - residual / max(1e-6, float(self.cfg.line_select_max_residual_m))) if math.isfinite(residual) else 0.0
        span_score = min(1.0, x_span / max(1e-6, float(self.cfg.line_select_min_x_span_m)))
        count_score = min(1.0, candidate_count / max(1.0, float(self.cfg.trend_min_candidate_count) * 2.0))
        inlier_score = min(1.0, float(fit.get("inlier_count", 0) or 0) / float(max(1, candidate_count)))
        effective_pixels = inlier_pixels or pixels
        boundary_touch_ratio = self._line_boundary_touch_ratio(effective_pixels, roi_box)
        boundary_score = max(0.0, 1.0 - boundary_touch_ratio / max(1e-6, float(self.cfg.roi_boundary_max_touch_ratio)))
        plane_yaw_consistency = 1.0
        if bool(plane.get("found")):
            yaw_diff = abs(float(fit.get("yaw", 0.0) or 0.0) - float(plane.get("yaw", 0.0) or 0.0))
            plane_yaw_consistency = max(0.0, 1.0 - yaw_diff / max(1e-6, float(self.cfg.line_select_max_plane_yaw_diff_rad)))
        plane_boundary_dist, plane_boundary_consistency = self._line_plane_boundary_score(line_type, effective_pixels, plane)
        object_like_score = self._object_like_line_score(
            candidate_count,
            int(fit.get("inlier_count", 0) or 0),
            x_span,
            plane_boundary_consistency,
            boundary_touch_ratio,
        )
        temporal_bonus = 0.05 if line_type == self._last_selected_line_type else 0.0
        raw_confidence = float(
            np.clip(
                0.30 * residual_score
                + 0.25 * span_score
                + 0.20 * count_score
                + 0.15 * inlier_score
                + 0.10 * boundary_score,
                0.0,
                1.0,
            )
        )
        confidence = float(
            np.clip(
                raw_confidence
                * (1.0 - float(self.cfg.line_object_like_penalty_weight) * object_like_score)
                * (1.0 - float(self.cfg.line_plane_boundary_weight) * (1.0 - plane_boundary_consistency)),
                0.0,
                1.0,
            )
        )
        selection_score = float(
            np.clip(
                0.45 * confidence
                + 0.20 * span_score
                + 0.15 * residual_score
                + 0.10 * plane_yaw_consistency
                + 0.10 * plane_boundary_consistency
                + temporal_bonus,
                0.0,
                1.0,
            )
        )
        drift_rejected = bool(
            bool(plane.get("found"))
            and (
                plane_boundary_consistency < float(self.cfg.fusion_line_min_boundary_consistency)
                or object_like_score > float(self.cfg.line_object_like_max_score)
            )
        )
        found = (
            confidence >= float(self.cfg.line_select_min_confidence)
            and x_span >= float(self.cfg.line_select_min_x_span_m)
            and residual <= float(self.cfg.line_select_max_residual_m)
            and not drift_rejected
        )
        reject_reason = ""
        if not found:
            if drift_rejected and plane_boundary_consistency < float(self.cfg.fusion_line_min_boundary_consistency):
                reject_reason = "line_plane_boundary_mismatch"
            elif drift_rejected and object_like_score > float(self.cfg.line_object_like_max_score):
                reject_reason = "object_like_line"
            elif x_span < float(self.cfg.line_select_min_x_span_m):
                reject_reason = "line_too_short"
            elif residual > float(self.cfg.line_select_max_residual_m):
                reject_reason = "line_residual_high"
            elif confidence < float(self.cfg.line_select_min_confidence):
                reject_reason = "line_confidence_low"
        out.update(
            {
                "found": bool(found),
                "candidate_count": candidate_count,
                "inlier_count": int(fit.get("inlier_count", 0) or 0) if found else 0,
                "yaw": float(fit.get("yaw", 0.0) or 0.0) if found else 0.0,
                "dist": float(fit.get("dist", 0.0) or 0.0) if found else 0.0,
                "confidence": confidence if found else 0.0,
                "selection_score": selection_score if found else 0.0,
                "residual_mean": residual if found else float("inf"),
                "x_span": x_span if found else 0.0,
                "y_norm_mean": float(np.mean(inlier_y_norms)) if len(inlier_y_norms) else out["y_norm_mean"],
                "k": fit.get("k") if found else None,
                "b": fit.get("b") if found else None,
                "inlier_pixels": inlier_pixels[:1600],
                "image_line_k": image_line_k if found else None,
                "image_line_b": image_line_b if found else None,
                "boundary_touch_ratio": boundary_touch_ratio,
                "plane_boundary_dist_px": plane_boundary_dist,
                "plane_boundary_consistency": plane_boundary_consistency,
                "object_like_score": object_like_score,
                "drift_rejected": drift_rejected,
                "reject_reason": reject_reason,
            }
        )
        return out

    def _estimate_crease_line(self, depth_m: np.ndarray, valid_mask: np.ndarray, roi_box, plane: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        x0, y0, _x1, _y1 = roi_box
        h, w = depth_m.shape[:2]
        candidates: List[Tuple[int, int, float]] = []
        win = max(3, int(self.cfg.trend_window_px))
        step = max(1, int(self.cfg.trend_col_step_px))
        min_ratio = float(self.cfg.trend_min_valid_ratio)
        min_valid = max(3, int(math.ceil(win * min_ratio)))
        min_delta = float(self.cfg.trend_min_slope_delta)
        topk = max(1, int(self.cfg.trend_topk_per_col))
        plane = dict(plane or {})
        if h < win * 2 + 1 or w <= 0:
            upper = self._empty_line_hypothesis("upper_crease")
            lower = self._empty_line_hypothesis("lower_contact")
            return {"found": False, "selected_line_type": "none", "upper": upper, "lower": lower, **self._empty_line_hypothesis("selected")}
        rows = np.arange(h, dtype=np.float64)
        for col in range(0, w, step):
            z_col = depth_m[:, col].astype(np.float64)
            valid_col = valid_mask[:, col] & np.isfinite(z_col)
            scored: List[Tuple[float, int]] = []
            for yy in range(win, h - win):
                above_mask = valid_col[yy - win : yy]
                below_mask = valid_col[yy + 1 : yy + 1 + win]
                if int(above_mask.sum()) < min_valid or int(below_mask.sum()) < min_valid:
                    continue
                ya = rows[yy - win : yy][above_mask]
                yb = rows[yy + 1 : yy + 1 + win][below_mask]
                za = z_col[yy - win : yy][above_mask]
                zb = z_col[yy + 1 : yy + 1 + win][below_mask]
                if ya.size < 2 or yb.size < 2:
                    continue
                slope_above = float(np.polyfit(ya, za, 1)[0])
                slope_below = float(np.polyfit(yb, zb, 1)[0])
                score = abs(slope_below - slope_above)
                if score >= min_delta:
                    scored.append((score, yy))
            if not scored:
                continue
            selected_rows: List[int] = []
            for score, yy in sorted(scored, key=lambda item: item[0], reverse=True):
                if len(selected_rows) >= topk:
                    break
                if any(abs(yy - prev) < max(4, win // 2) for prev in selected_rows):
                    continue
                y_start = max(0, yy - 2)
                y_end = min(h, yy + 3)
                local_valid = valid_mask[y_start:y_end, col]
                local_depth = depth_m[y_start:y_end, col]
                if not np.any(local_valid) and not valid_mask[yy, col]:
                    continue
                selected_rows.append(yy)
                candidates.append((col, yy, float(score)))
        candidate_count = int(len(candidates))
        if candidate_count <= 0:
            points = np.empty((0, 3), dtype=np.float32)
            pixels: List[List[int]] = []
            y_norms = np.empty((0,), dtype=np.float32)
        else:
            cols = np.array([c[0] for c in candidates], dtype=np.int64)
            ys = np.array([c[1] for c in candidates], dtype=np.int64)
            z = np.zeros((len(candidates),), dtype=np.float64)
            for idx, (col, yy, _score) in enumerate(candidates):
                y_start = max(0, yy - 2)
                y_end = min(h, yy + 3)
                local_valid = valid_mask[y_start:y_end, col]
                local_depth = depth_m[y_start:y_end, col]
                z[idx] = float(np.median(local_depth[local_valid])) if np.any(local_valid) else float(depth_m[yy, col])
            good = np.isfinite(z) & (z > float(self.cfg.z_min)) & (z < float(self.cfg.z_max))
            cols = cols[good]
            ys = ys[good]
            z = z[good]
            y_norms = ys.astype(np.float32) / float(max(1, h - 1))
            u = cols.astype(np.float64) + float(x0)
            v = ys.astype(np.float64) + float(y0)
            x = (u - float(self.calib.cx)) * z / float(self.calib.fx)
            y = (v - float(self.calib.cy)) * z / float(self.calib.fy)
            points = np.vstack((x, y, z)).T.astype(np.float32)
            pixels = [[int(cx + x0), int(cy + y0)] for cx, cy in zip(cols.tolist(), ys.tolist())]
            candidate_count = int(points.shape[0])
        upper_mask = (y_norms >= float(self.cfg.upper_line_y_norm_min)) & (y_norms <= float(self.cfg.upper_line_y_norm_max))
        lower_mask = (y_norms >= float(self.cfg.lower_line_y_norm_min)) & (y_norms <= float(self.cfg.lower_line_y_norm_max))
        upper_pixels = [pixels[i] for i, keep in enumerate(upper_mask.tolist()) if keep] if len(pixels) == len(upper_mask) else []
        lower_pixels = [pixels[i] for i, keep in enumerate(lower_mask.tolist()) if keep] if len(pixels) == len(lower_mask) else []
        upper = self._fit_line_hypothesis("upper_crease", points[upper_mask], upper_pixels, y_norms[upper_mask], plane, roi_box)
        lower = self._fit_line_hypothesis("lower_contact", points[lower_mask], lower_pixels, y_norms[lower_mask], plane, roi_box)
        selected = self._empty_line_hypothesis("selected")
        reliable = [item for item in (upper, lower) if bool(item.get("found"))]
        if reliable:
            selected = max(reliable, key=lambda item: float(item.get("selection_score", 0.0) or 0.0))
            selected = dict(selected)
            selected["selected"] = True
            selected_line_type = "upper_crease" if selected.get("type") == "upper_crease" else "lower_contact"
            line_reject_reason = str(selected.get("reject_reason") or "")
        else:
            selected_line_type = "none"
            rejected = max((upper, lower), key=lambda item: int(item.get("candidate_count", 0) or 0))
            line_reject_reason = str(rejected.get("reject_reason") or "no_reliable_line")
        return {
            "found": bool(selected_line_type != "none"),
            "selected_line_type": selected_line_type,
            "candidate_count": int(selected.get("candidate_count", 0) or 0),
            "inlier_count": int(selected.get("inlier_count", 0) or 0),
            "yaw": float(selected.get("yaw", 0.0) or 0.0),
            "dist": float(selected.get("dist", 0.0) or 0.0),
            "confidence": float(selected.get("confidence", 0.0) or 0.0),
            "residual_mean": selected.get("residual_mean", float("inf")),
            "x_span": float(selected.get("x_span", 0.0) or 0.0),
            "y_norm_mean": float(selected.get("y_norm_mean", 0.0) or 0.0),
            "k": selected.get("k"),
            "b": selected.get("b"),
            "points": selected.get("points", np.empty((0, 3), dtype=np.float32)),
            "pixels": pixels[:1600],
            "inlier_pixels": selected.get("inlier_pixels", [])[:1600],
            "image_line_k": selected.get("image_line_k"),
            "image_line_b": selected.get("image_line_b"),
            "boundary_touch_ratio": float(selected.get("boundary_touch_ratio", 0.0) or 0.0),
            "plane_boundary_dist_px": float(selected.get("plane_boundary_dist_px", float("inf")) or float("inf")),
            "plane_boundary_consistency": float(selected.get("plane_boundary_consistency", 0.0) or 0.0),
            "object_like_score": float(selected.get("object_like_score", 0.0) or 0.0),
            "drift_rejected": bool(selected.get("drift_rejected", False)) if selected_line_type != "none" else bool(upper.get("drift_rejected") or lower.get("drift_rejected")),
            "reject_reason": line_reject_reason,
            "upper": upper,
            "lower": lower,
            "upper_pixels": upper.get("pixels", [])[:1600],
            "lower_pixels": lower.get("pixels", [])[:1600],
        }

    def _plane_reject_reason(self, plane: Dict[str, Any]) -> str:
        if not bool(plane.get("found")):
            return "no_reliable_plane"
        area_ratio = float(plane.get("area_ratio", 0.0) or 0.0)
        residual = float(plane.get("residual_mean", float("inf")) or float("inf"))
        x_span = float(plane.get("x_span", 0.0) or 0.0)
        if area_ratio < float(self.cfg.front_plane_min_area_ratio):
            return "low_front_face_area"
        if not math.isfinite(residual) or residual > float(self.cfg.front_plane_max_residual_m):
            return "plane_residual_high"
        if x_span < float(self.cfg.front_plane_min_x_span_m):
            return "plane_x_span_short"
        if float(plane.get("confidence", 0.0) or 0.0) < float(self.cfg.front_plane_min_score):
            return "front_plane_score_low"
        return ""

    def _front_plane_pose(self, plane: Dict[str, Any], raw_found: bool) -> Dict[str, Any]:
        reject_reason = self._plane_reject_reason(plane)
        if reject_reason:
            return {
                "raw_found": raw_found,
                "pose_found": False,
                "pose_source": "none",
                "final_pose_source": "none",
                "yaw": 0.0,
                "dist": 0.0,
                "confidence": 0.0,
                "line_k": None,
                "line_b": None,
                "x_span": 0.0,
                "residual_mean": float("inf"),
                "reject_reason": reject_reason,
            }
        return {
            "raw_found": raw_found,
            "pose_found": True,
            "pose_source": "front_plane",
            "final_pose_source": "front_plane",
            "yaw": float(plane.get("yaw", 0.0) or 0.0),
            "dist": float(plane.get("dist", 0.0) or 0.0),
            "confidence": float(plane.get("confidence", 0.0) or 0.0),
            "line_k": plane.get("k"),
            "line_b": plane.get("b"),
            "x_span": float(plane.get("x_span", 0.0) or 0.0),
            "residual_mean": float(plane.get("residual_mean", float("inf"))),
            "reject_reason": "",
        }

    def _fuse_front_pose(self, plane: Dict[str, Any], line: Dict[str, Any]) -> Dict[str, Any]:
        if bool(getattr(self.cfg, "plane_only_mode", False)) or not bool(getattr(self.cfg, "enable_crease_line", True)):
            return self._front_plane_pose(plane, raw_found=bool(plane.get("found")))
        plane_reliable = (
            bool(plane.get("found"))
            and float(plane.get("area_ratio", 0.0) or 0.0) >= float(self.cfg.front_face_min_area_ratio)
            and float(plane.get("residual_mean", float("inf")) or float("inf")) <= float(self.cfg.plane_max_residual_m)
            and float(plane.get("x_span", 0.0) or 0.0) >= float(self.cfg.plane_min_x_span_m)
        )
        line_reliable = (
            bool(line.get("found"))
            and int(line.get("candidate_count", 0) or 0) >= int(self.cfg.trend_min_candidate_count)
            and float(line.get("residual_mean", float("inf")) or float("inf")) <= float(self.cfg.line_select_max_residual_m)
            and float(line.get("x_span", 0.0) or 0.0) >= float(self.cfg.line_select_min_x_span_m)
            and float(line.get("confidence", 0.0) or 0.0) >= float(self.cfg.line_select_min_confidence)
            and not bool(line.get("drift_rejected", False))
        )
        boundary_consistency = float(line.get("plane_boundary_consistency", 1.0) or 0.0)
        if plane_reliable and line_reliable and boundary_consistency < float(self.cfg.fusion_line_min_boundary_consistency):
            line_reliable = False
        raw_found = bool(plane.get("found")) or bool(line.get("found"))
        if plane_reliable and line_reliable:
            yaw_delta = abs(float(plane["yaw"]) - float(line["yaw"]))
            if yaw_delta <= float(self.cfg.fusion_yaw_consistency_rad):
                wp = max(1e-6, float(plane.get("confidence", 0.0) or 0.0))
                wl = max(1e-6, float(line.get("confidence", 0.0) or 0.0))
                if boundary_consistency < float(self.cfg.fusion_plane_prefer_boundary_consistency):
                    wl *= max(0.20, boundary_consistency)
                    wp *= 1.25
                wsum = wp + wl
                return {
                    "raw_found": raw_found,
                    "pose_found": True,
                    "pose_source": "fused",
                    "final_pose_source": "fused",
                    "yaw": float((wp * float(plane["yaw"]) + wl * float(line["yaw"])) / wsum),
                    "dist": float((wp * float(plane["dist"]) + wl * float(line["dist"])) / wsum),
                    "confidence": float(np.clip(0.5 * (wp + wl), 0.0, 1.0)),
                    "line_k": line.get("k"),
                    "line_b": line.get("b"),
                    "x_span": max(float(plane.get("x_span", 0.0) or 0.0), float(line.get("x_span", 0.0) or 0.0)),
                    "residual_mean": min(float(plane.get("residual_mean", float("inf"))), float(line.get("residual_mean", float("inf")))),
                    "reject_reason": "",
                }
            chosen = plane if float(plane.get("confidence", 0.0) or 0.0) >= float(line.get("confidence", 0.0) or 0.0) else line
            return {
                "raw_found": raw_found,
                "pose_found": True,
                "pose_source": "conflict",
                "final_pose_source": "conflict",
                "yaw": float(chosen.get("yaw", 0.0) or 0.0),
                "dist": float(chosen.get("dist", 0.0) or 0.0),
                "confidence": min(float(plane.get("confidence", 0.0) or 0.0), float(line.get("confidence", 0.0) or 0.0)),
                "line_k": line.get("k") if line.get("k") is not None else plane.get("k"),
                "line_b": line.get("b") if line.get("b") is not None else plane.get("b"),
                "x_span": max(float(plane.get("x_span", 0.0) or 0.0), float(line.get("x_span", 0.0) or 0.0)),
                "residual_mean": max(float(plane.get("residual_mean", float("inf"))), float(line.get("residual_mean", float("inf")))),
                "reject_reason": "plane_line_yaw_conflict",
            }
        if plane_reliable:
            return {
                "raw_found": raw_found,
                "pose_found": True,
                "pose_source": "front_plane",
                "final_pose_source": "front_plane",
                "yaw": float(plane.get("yaw", 0.0) or 0.0),
                "dist": float(plane.get("dist", 0.0) or 0.0),
                "confidence": float(plane.get("confidence", 0.0) or 0.0),
                "line_k": plane.get("k"),
                "line_b": plane.get("b"),
                "x_span": float(plane.get("x_span", 0.0) or 0.0),
                "residual_mean": float(plane.get("residual_mean", float("inf"))),
                "reject_reason": str(line.get("reject_reason") or "") if bool(line.get("drift_rejected", False)) else "",
            }
        if line_reliable:
            return {
                "raw_found": raw_found,
                "pose_found": True,
                "pose_source": "crease_line",
                "final_pose_source": "crease_line",
                "yaw": float(line.get("yaw", 0.0) or 0.0),
                "dist": float(line.get("dist", 0.0) or 0.0),
                "confidence": float(line.get("confidence", 0.0) or 0.0),
                "line_k": line.get("k"),
                "line_b": line.get("b"),
                "x_span": float(line.get("x_span", 0.0) or 0.0),
                "residual_mean": float(line.get("residual_mean", float("inf"))),
                "reject_reason": "",
            }
        return {
            "raw_found": raw_found,
            "pose_found": False,
            "pose_source": "none",
            "final_pose_source": "none",
            "yaw": 0.0,
            "dist": 0.0,
            "confidence": 0.0,
            "line_k": None,
            "line_b": None,
            "x_span": 0.0,
            "residual_mean": float("inf"),
            "reject_reason": "no_reliable_pose" if raw_found else "no_raw_geometry",
        }

    def _score_table_geometry(self, plane: Dict[str, Any], line: Dict[str, Any], fused: Dict[str, Any]) -> Dict[str, Any]:
        plane_residual = float(plane.get("residual_mean", float("inf")) or float("inf"))
        plane_span = float(plane.get("x_span", 0.0) or 0.0)
        area_ratio = float(plane.get("area_ratio", 0.0) or 0.0)
        if bool(plane.get("found")):
            plane_residual_score = max(0.0, 1.0 - plane_residual / max(1e-6, float(self.cfg.front_plane_max_residual_m))) if math.isfinite(plane_residual) else 0.0
            plane_span_score = min(1.0, plane_span / max(1e-6, float(self.cfg.front_plane_min_x_span_m)))
            plane_area_score = min(1.0, area_ratio / max(1e-6, float(self.cfg.front_plane_min_area_ratio)))
            front_plane_score = float(
                np.clip(
                    0.35 * float(plane.get("confidence", 0.0) or 0.0)
                    + 0.25 * plane_area_score
                    + 0.20 * plane_residual_score
                    + 0.20 * plane_span_score,
                    0.0,
                    1.0,
                )
            )
        else:
            front_plane_score = 0.0

        line_residual = float(line.get("residual_mean", float("inf")) or float("inf"))
        line_span = float(line.get("x_span", 0.0) or 0.0)
        boundary_consistency = float(line.get("plane_boundary_consistency", 1.0) or 0.0)
        object_like_score = float(line.get("object_like_score", 0.0) or 0.0)
        if bool(line.get("found")):
            line_residual_score = max(0.0, 1.0 - line_residual / max(1e-6, float(self.cfg.line_select_max_residual_m))) if math.isfinite(line_residual) else 0.0
            line_span_score = min(1.0, line_span / max(1e-6, float(self.cfg.line_select_min_x_span_m)))
            line_count_score = min(1.0, int(line.get("candidate_count", 0) or 0) / max(1.0, float(self.cfg.trend_min_candidate_count) * 2.0))
            line_score = float(
                np.clip(
                    0.40 * float(line.get("confidence", 0.0) or 0.0)
                    + 0.25 * line_residual_score
                    + 0.20 * line_span_score
                    + 0.15 * line_count_score,
                    0.0,
                    1.0,
                )
            )
            line_score = float(np.clip(line_score * max(0.0, boundary_consistency) * max(0.0, 1.0 - 0.35 * object_like_score), 0.0, 1.0))
        else:
            line_score = 0.0

        if bool(plane.get("found")) and bool(line.get("found")):
            yaw_diff = abs(float(plane.get("yaw", 0.0) or 0.0) - float(line.get("yaw", 0.0) or 0.0))
            dist_diff = abs(float(plane.get("dist", 0.0) or 0.0) - float(line.get("dist", 0.0) or 0.0))
            yaw_score = max(0.0, 1.0 - yaw_diff / max(1e-6, float(self.cfg.fusion_yaw_consistency_rad)))
            dist_score = max(0.0, 1.0 - dist_diff / max(1e-6, float(self.cfg.control_max_dist_jump_m)))
            plane_line_consistency_score = float(np.clip(0.55 * yaw_score + 0.25 * dist_score + 0.20 * boundary_consistency, 0.0, 1.0))
        elif bool(plane.get("found")) or bool(line.get("found")):
            plane_line_consistency_score = 0.55
        else:
            plane_line_consistency_score = 0.0

        boundary_touch = float(line.get("boundary_touch_ratio", 0.0) or 0.0) if bool(line.get("found")) else 0.0
        roi_boundary_score = max(0.0, 1.0 - boundary_touch / max(1e-6, float(self.cfg.roi_boundary_max_touch_ratio)))
        if not bool(line.get("found")) and bool(plane.get("found")):
            roi_boundary_score = 0.70

        pose_found = bool(fused.get("pose_found"))
        yaw = float(fused.get("yaw", 0.0) or 0.0)
        dist = float(fused.get("dist", 0.0) or 0.0)
        temporal_jump = False
        if not pose_found:
            temporal_score = 0.0
        elif self._last_pose is None:
            temporal_score = 0.60
        else:
            yaw_jump = abs(yaw - float(self._last_pose[0]))
            dist_jump = abs(dist - float(self._last_pose[1]))
            yaw_score = max(0.0, 1.0 - yaw_jump / max(1e-6, float(self.cfg.control_max_yaw_jump_rad)))
            dist_score = max(0.0, 1.0 - dist_jump / max(1e-6, float(self.cfg.control_max_dist_jump_m)))
            temporal_score = float(np.clip(0.55 * yaw_score + 0.45 * dist_score, 0.0, 1.0))
            temporal_jump = yaw_jump > float(self.cfg.control_max_yaw_jump_rad) or dist_jump > float(self.cfg.control_max_dist_jump_m)

        if bool(getattr(self.cfg, "plane_only_mode", False)) or not bool(getattr(self.cfg, "enable_crease_line", True)):
            plane_line_consistency_score = 1.0 if bool(plane.get("found")) else 0.0
            roi_boundary_score = 1.0 if bool(plane.get("found")) else 0.0
            line_score = 0.0
            table_geometry_score = float(np.clip(0.80 * front_plane_score + 0.20 * temporal_score, 0.0, 1.0))
            reason = self._plane_reject_reason(plane)
            if not reason and temporal_jump:
                reason = "temporal_jump"
            return {
                "table_geometry_score": table_geometry_score,
                "front_plane_score": front_plane_score,
                "line_score": line_score,
                "plane_line_consistency_score": plane_line_consistency_score,
                "roi_boundary_score": roi_boundary_score,
                "temporal_score": temporal_score,
                "geometry_reject_reason": reason,
                "temporal_jump": temporal_jump,
            }

        weights = {
            "front": max(0.0, float(self.cfg.front_plane_score_weight)),
            "line": max(0.0, float(self.cfg.line_score_weight)),
            "consistency": max(0.0, float(self.cfg.plane_line_consistency_weight)),
            "boundary": max(0.0, float(self.cfg.roi_boundary_score_weight)),
            "temporal": max(0.0, float(self.cfg.temporal_score_weight)),
        }
        weight_sum = max(1e-6, sum(weights.values()))
        table_geometry_score = float(
            np.clip(
                (
                    weights["front"] * front_plane_score
                    + weights["line"] * line_score
                    + weights["consistency"] * plane_line_consistency_score
                    + weights["boundary"] * roi_boundary_score
                    + weights["temporal"] * temporal_score
                )
                / weight_sum,
                0.0,
                1.0,
            )
        )

        reasons: List[Tuple[float, str]] = []
        if not bool(plane.get("found")):
            reasons.append((front_plane_score, "no_reliable_plane"))
        elif area_ratio < float(self.cfg.front_face_min_area_ratio):
            reasons.append((front_plane_score, "low_front_face_area"))
        if not bool(line.get("found")):
            reasons.append((line_score, str(line.get("reject_reason") or "no_reliable_line")))
        elif line_span < float(self.cfg.line_select_min_x_span_m):
            reasons.append((line_score, "line_too_short"))
        elif line_residual > float(self.cfg.line_select_max_residual_m):
            reasons.append((line_score, "line_residual_high"))
        elif boundary_consistency < float(self.cfg.fusion_line_min_boundary_consistency):
            reasons.append((boundary_consistency, "line_plane_boundary_mismatch"))
        elif object_like_score > float(self.cfg.line_object_like_max_score):
            reasons.append((1.0 - object_like_score, "object_like_line"))
        if plane_line_consistency_score < 0.35 and bool(plane.get("found")) and bool(line.get("found")):
            reasons.append((plane_line_consistency_score, "plane_line_yaw_conflict"))
        if roi_boundary_score < 0.35:
            reasons.append((roi_boundary_score, "touching_roi_boundary"))
        if temporal_jump:
            reasons.append((temporal_score, "temporal_jump"))
        geometry_reject_reason = min(reasons, key=lambda item: item[0])[1] if reasons else ""
        return {
            "table_geometry_score": table_geometry_score,
            "front_plane_score": front_plane_score,
            "line_score": line_score,
            "plane_line_consistency_score": plane_line_consistency_score,
            "roi_boundary_score": roi_boundary_score,
            "temporal_score": temporal_score,
            "geometry_reject_reason": geometry_reject_reason,
            "temporal_jump": temporal_jump,
        }

    def _validate_pose_for_control(self, fused: Dict[str, Any], geometry: Dict[str, Any]) -> Dict[str, Any]:
        pose_found = bool(fused.get("pose_found"))
        reason = str(fused.get("reject_reason") or "")
        valid_base = pose_found and fused.get("pose_source") != "conflict"
        yaw = float(fused.get("yaw", 0.0) or 0.0)
        dist = float(fused.get("dist", 0.0) or 0.0)
        conf = float(fused.get("confidence", 0.0) or 0.0)
        x_span = float(fused.get("x_span", 0.0) or 0.0)
        residual = float(fused.get("residual_mean", float("inf")) or float("inf"))
        geometry_score = float(geometry.get("table_geometry_score", 0.0) or 0.0)
        temporal_jump = bool(geometry.get("temporal_jump", False))
        if bool(getattr(self.cfg, "plane_only_mode", False)) or not bool(getattr(self.cfg, "enable_crease_line", True)):
            front_plane_score = float(geometry.get("front_plane_score", 0.0) or 0.0)
            valid_base = pose_found and str(fused.get("pose_source") or "") == "front_plane"
            if not valid_base and not reason:
                reason = "pose_not_found"
            if valid_base and conf < float(self.cfg.control_min_confidence):
                valid_base = False
                reason = "low_confidence"
            if valid_base and x_span < float(self.cfg.front_plane_min_x_span_m):
                valid_base = False
                reason = "plane_x_span_short"
            if valid_base and (not math.isfinite(residual) or residual > float(self.cfg.front_plane_max_residual_m)):
                valid_base = False
                reason = "plane_residual_high"
            if valid_base and abs(yaw) > float(self.cfg.control_max_yaw_rad):
                valid_base = False
                reason = "yaw_out_of_range"
            if valid_base and temporal_jump:
                valid_base = False
                reason = "temporal_jump"
            if valid_base:
                self._stable_count += 1
            else:
                self._stable_count = 0
            dist_valid = pose_found and math.isfinite(dist) and abs(dist) <= max(float(self.target_dist_m), float(self.cfg.z_max))
            usable_for_approach = bool(
                pose_found
                and dist_valid
                and front_plane_score >= float(self.cfg.control_approach_min_score)
                and self._stable_count >= int(self.cfg.control_approach_min_stable_frames)
            )
            usable_for_alignment = bool(
                valid_base
                and front_plane_score >= float(self.cfg.control_alignment_min_score)
                and self._stable_count >= int(self.cfg.control_alignment_min_stable_frames)
            )
            usable_for_stop = bool(
                valid_base
                and front_plane_score >= float(self.cfg.control_stop_min_score)
                and abs(dist) <= float(self.cfg.control_stop_dist_abs_max_m)
                and self._stable_count >= int(self.cfg.control_stop_min_stable_frames)
            )
            valid_for_control = bool(usable_for_alignment or usable_for_stop)
            if usable_for_stop:
                control_level = "stop"
            elif usable_for_alignment:
                control_level = "alignment"
            elif usable_for_approach:
                control_level = "approach"
            else:
                control_level = "none"
            if control_level == "none":
                if not pose_found:
                    control_reason = reason or "pose_not_found"
                elif front_plane_score < float(self.cfg.control_approach_min_score):
                    control_reason = geometry.get("geometry_reject_reason") or "front_plane_score_low"
                elif not valid_base:
                    control_reason = reason or "control_gate_rejected"
                else:
                    control_reason = "stabilizing"
            elif control_level == "approach" and not valid_for_control:
                control_reason = "approach_only"
            else:
                control_reason = ""
            if valid_base and control_level == "none" and not reason:
                reason = "stabilizing"
            if pose_found:
                self._last_pose = (yaw, dist)
                self._last_pose_source = "front_plane"
            return {
                "valid_for_control": valid_for_control,
                "stable_count": int(self._stable_count),
                "reject_reason": reason,
                "usable_for_approach": usable_for_approach,
                "usable_for_alignment": usable_for_alignment,
                "usable_for_stop": usable_for_stop,
                "control_level": control_level,
                "control_reject_reason": control_reason,
            }
        if not valid_base and not reason:
            reason = "pose_not_found"
        if valid_base and conf < float(self.cfg.control_min_confidence):
            valid_base = False
            reason = "low_confidence"
        min_span = min(float(self.cfg.plane_min_x_span_m), float(self.cfg.line_min_x_span_m))
        if valid_base and x_span < min_span:
            valid_base = False
            reason = "insufficient_x_span"
        max_residual = max(float(self.cfg.plane_max_residual_m), float(self.cfg.line_max_residual_m))
        if valid_base and (not math.isfinite(residual) or residual > max_residual):
            valid_base = False
            reason = "residual_too_high"
        if valid_base and abs(yaw) > float(self.cfg.control_max_yaw_rad):
            valid_base = False
            reason = "yaw_out_of_range"
        if valid_base and temporal_jump:
            valid_base = False
            reason = "temporal_jump"
        if valid_base:
            self._stable_count += 1
        else:
            self._stable_count = 0
        approach_base = (
            pose_found
            and geometry_score >= float(self.cfg.table_geometry_approach_score)
            and abs(dist) <= max(float(self.target_dist_m), float(self.cfg.z_max))
            and fused.get("pose_source") != "conflict"
        )
        usable_for_approach = bool(approach_base and self._stable_count >= int(self.cfg.control_approach_min_stable_frames))
        alignment_source_ok = str(fused.get("pose_source") or "") in {"fused", "crease_line"}
        usable_for_alignment = bool(
            valid_base
            and alignment_source_ok
            and geometry_score >= float(self.cfg.table_geometry_alignment_score)
            and self._stable_count >= int(self.cfg.control_alignment_min_stable_frames)
        )
        usable_for_stop = bool(
            valid_base
            and geometry_score >= float(self.cfg.table_geometry_stop_score)
            and abs(dist) <= float(self.cfg.control_stop_dist_abs_max_m)
            and self._stable_count >= int(self.cfg.control_stop_min_stable_frames)
        )
        valid_for_control = bool(usable_for_alignment or usable_for_stop)
        if usable_for_stop:
            control_level = "stop"
        elif usable_for_alignment:
            control_level = "alignment"
        elif usable_for_approach:
            control_level = "approach"
        else:
            control_level = "none"
        if control_level == "none":
            if not pose_found:
                control_reason = reason or "pose_not_found"
            elif geometry_score < float(self.cfg.table_geometry_approach_score):
                control_reason = geometry.get("geometry_reject_reason") or "geometry_score_low"
            elif not valid_base:
                control_reason = reason or "control_gate_rejected"
            else:
                control_reason = "stabilizing"
        elif control_level == "approach" and not valid_for_control:
            control_reason = "approach_only"
        else:
            control_reason = ""
        if valid_base and control_level == "none" and not reason:
            reason = "stabilizing"
        if pose_found:
            self._last_pose = (yaw, dist)
            self._last_pose_source = str(fused.get("pose_source") or "none")
        return {
            "valid_for_control": valid_for_control,
            "stable_count": int(self._stable_count),
            "reject_reason": reason,
            "usable_for_approach": usable_for_approach,
            "usable_for_alignment": usable_for_alignment,
            "usable_for_stop": usable_for_stop,
            "control_level": control_level,
            "control_reject_reason": control_reason,
        }

    @staticmethod
    def _finite_or_zero(value: Any) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return 0.0
        return out if math.isfinite(out) else 0.0

    @staticmethod
    def _finite_or_default(value: Any, default: float) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float(default)
        return out if math.isfinite(out) else float(default)

    def process_depth(self, depth_image_16bit: np.ndarray, roi_override=None):
        timing: Dict[str, float] = {}

        def _stage_ms(start: float) -> float:
            return float((time.perf_counter() - start) * 1000.0)

        def _finish_timing() -> None:
            for key in (
                "roi_extract_ms",
                "point_build_ms",
                "candidate_select_ms",
                "plane_fit_ms",
                "residual_eval_ms",
                "mask_build_ms",
                "obs_build_ms",
            ):
                timing.setdefault(key, 0.0)

        stage_start = time.perf_counter()
        valid_mask, depth_meters, roi_box = self._preprocess_depth(depth_image_16bit, roi_override=roi_override)
        timing["roi_extract_ms"] = _stage_ms(stage_start)

        stage_start = time.perf_counter()
        pc_cam = self._depth_to_3d(depth_meters, valid_mask, roi_box)
        timing["point_build_ms"] = _stage_ms(stage_start)
        base_debug = {
            "depth_meters": depth_meters,
            "pc_all": pc_cam,
            "pc_table": None,
            "roi_box": roi_box,
            "timing": timing,
            "front_plane_candidate_pixels": [],
            "crease_candidate_pixels": [],
            "crease_inlier_pixels": [],
            "upper_line_candidate_pixels": [],
            "upper_line_inlier_pixels": [],
            "lower_line_candidate_pixels": [],
            "lower_line_inlier_pixels": [],
        }
        if len(pc_cam) < int(self.cfg.min_all_points):
            self._stable_count = 0
            obs_start = time.perf_counter()
            result = EdgeDetectResult(
                False,
                0.0,
                0.0,
                0.0,
                point_count=len(pc_cam),
                table_point_count=0,
                reject_reason="roi_empty",
                control_reject_reason="roi_empty",
                geometry_reject_reason="no_raw_geometry",
            )
            timing["obs_build_ms"] = _stage_ms(obs_start)
            _finish_timing()
            base_debug.update({"reject_reason": result.reject_reason})
            return result, base_debug

        stage_start = time.perf_counter()
        table_mask = self._find_table_plane(pc_cam)
        table_pc = pc_cam[table_mask]
        timing["candidate_select_ms"] = _stage_ms(stage_start)

        stage_start = time.perf_counter()
        plane = self._estimate_front_plane(depth_meters, valid_mask, roi_box)
        timing["plane_fit_ms"] = _stage_ms(stage_start)

        stage_start = time.perf_counter()
        if bool(getattr(self.cfg, "plane_only_mode", False)) or not bool(getattr(self.cfg, "enable_crease_line", True)):
            line = self._disabled_line_result()
        else:
            line = self._estimate_crease_line(depth_meters.copy(), valid_mask, roi_box, plane)
        fused = self._fuse_front_pose(plane, line)
        geometry = self._score_table_geometry(plane, line, fused)
        control = self._validate_pose_for_control(fused, geometry)
        edge_found = bool(fused.get("pose_found"))
        reject_reason = str(control.get("reject_reason") or fused.get("reject_reason") or "")
        selected_line_type = str(line.get("selected_line_type") or "none")
        upper = dict(line.get("upper") or {})
        lower = dict(line.get("lower") or {})
        timing["residual_eval_ms"] = _stage_ms(stage_start)

        obs_start = time.perf_counter()
        result = EdgeDetectResult(
            edge_found,
            float(fused.get("yaw", 0.0) or 0.0),
            float(fused.get("dist", 0.0) or 0.0),
            float(fused.get("confidence", 0.0) or 0.0),
            line_k=fused.get("line_k"),
            line_b=fused.get("line_b"),
            point_count=len(pc_cam),
            table_point_count=len(table_pc),
            raw_found=bool(fused.get("raw_found")),
            pose_found=edge_found,
            valid_for_control=bool(control.get("valid_for_control")),
            pose_source=str(fused.get("pose_source") or "none"),
            plane_found=bool(plane.get("found")),
            line_found=bool(line.get("found")),
            plane_confidence=float(plane.get("confidence", 0.0) or 0.0),
            line_confidence=float(line.get("confidence", 0.0) or 0.0),
            plane_residual_mean=self._finite_or_zero(plane.get("residual_mean")),
            line_residual_mean=self._finite_or_zero(line.get("residual_mean")),
            plane_x_span_m=float(plane.get("x_span", 0.0) or 0.0),
            line_x_span_m=float(line.get("x_span", 0.0) or 0.0),
            candidate_count=int(line.get("candidate_count", 0) or 0),
            inlier_count=int(plane.get("inlier_count", 0) or 0) if bool(getattr(self.cfg, "plane_only_mode", False)) or not bool(getattr(self.cfg, "enable_crease_line", True)) else int(max(int(plane.get("inlier_count", 0) or 0), int(line.get("inlier_count", 0) or 0))),
            stable_count=int(control.get("stable_count", 0) or 0),
            front_face_area_ratio=float(plane.get("area_ratio", 0.0) or 0.0),
            reject_reason=reject_reason,
            plane_yaw_err_rad=float(plane.get("yaw")) if plane.get("found") else None,
            plane_dist_err_m=float(plane.get("dist")) if plane.get("found") else None,
            line_yaw_err_rad=float(line.get("yaw")) if line.get("found") else None,
            line_dist_err_m=float(line.get("dist")) if line.get("found") else None,
            plane_k=plane.get("k"),
            plane_b=plane.get("b"),
            image_line_k=None if bool(getattr(self.cfg, "plane_only_mode", False)) or not bool(getattr(self.cfg, "enable_crease_line", True)) else line.get("image_line_k"),
            image_line_b=None if bool(getattr(self.cfg, "plane_only_mode", False)) or not bool(getattr(self.cfg, "enable_crease_line", True)) else line.get("image_line_b"),
            upper_line_found=bool(upper.get("found")),
            upper_line_confidence=float(upper.get("confidence", 0.0) or 0.0),
            upper_line_candidate_count=int(upper.get("candidate_count", 0) or 0),
            upper_line_inlier_count=int(upper.get("inlier_count", 0) or 0),
            upper_line_residual_mean=self._finite_or_zero(upper.get("residual_mean")),
            upper_line_x_span_m=float(upper.get("x_span", 0.0) or 0.0),
            upper_line_y_norm_mean=float(upper.get("y_norm_mean", 0.0) or 0.0),
            upper_line_k=upper.get("k"),
            upper_line_b=upper.get("b"),
            upper_line_yaw_err_rad=float(upper.get("yaw")) if upper.get("found") else None,
            upper_line_dist_err_m=float(upper.get("dist")) if upper.get("found") else None,
            lower_line_found=bool(lower.get("found")),
            lower_line_confidence=float(lower.get("confidence", 0.0) or 0.0),
            lower_line_candidate_count=int(lower.get("candidate_count", 0) or 0),
            lower_line_inlier_count=int(lower.get("inlier_count", 0) or 0),
            lower_line_residual_mean=self._finite_or_zero(lower.get("residual_mean")),
            lower_line_x_span_m=float(lower.get("x_span", 0.0) or 0.0),
            lower_line_y_norm_mean=float(lower.get("y_norm_mean", 0.0) or 0.0),
            lower_line_k=lower.get("k"),
            lower_line_b=lower.get("b"),
            lower_line_yaw_err_rad=float(lower.get("yaw")) if lower.get("found") else None,
            lower_line_dist_err_m=float(lower.get("dist")) if lower.get("found") else None,
            selected_line_type=selected_line_type,
            table_geometry_score=float(geometry.get("table_geometry_score", 0.0) or 0.0),
            front_plane_score=float(geometry.get("front_plane_score", 0.0) or 0.0),
            line_score=float(geometry.get("line_score", 0.0) or 0.0),
            plane_line_consistency_score=float(geometry.get("plane_line_consistency_score", 0.0) or 0.0),
            roi_boundary_score=float(geometry.get("roi_boundary_score", 0.0) or 0.0),
            temporal_score=float(geometry.get("temporal_score", 0.0) or 0.0),
            geometry_reject_reason=str(geometry.get("geometry_reject_reason") or ""),
            usable_for_approach=bool(control.get("usable_for_approach")),
            usable_for_alignment=bool(control.get("usable_for_alignment")),
            usable_for_stop=bool(control.get("usable_for_stop")),
            control_level=str(control.get("control_level") or "none"),
            control_reject_reason=str(control.get("control_reject_reason") or ""),
            selected_line_plane_boundary_dist=self._finite_or_default(line.get("plane_boundary_dist_px"), -1.0),
            selected_line_plane_consistency=float(line.get("plane_boundary_consistency", 0.0) or 0.0),
            line_reject_reason=str(line.get("reject_reason") or ""),
            line_drift_rejected=bool(line.get("drift_rejected", False)),
            object_like_line_score=float(line.get("object_like_score", 0.0) or 0.0),
            final_pose_source=str(fused.get("final_pose_source") or fused.get("pose_source") or "none"),
        )
        self._last_selected_line_type = selected_line_type
        timing["obs_build_ms"] = _stage_ms(obs_start)

        mask_start = time.perf_counter()
        base_debug.update(
            {
                "pc_table": table_pc,
                "front_plane": plane,
                "crease_line": line,
                "fused_pose": fused,
                "table_geometry": geometry,
                "control_gate": control,
                "front_plane_candidate_pixels": plane.get("pixels", []),
                "crease_candidate_pixels": line.get("pixels", []),
                "crease_inlier_pixels": line.get("inlier_pixels", []),
                "upper_line_candidate_pixels": upper.get("pixels", []),
                "upper_line_inlier_pixels": upper.get("inlier_pixels", []),
                "lower_line_candidate_pixels": lower.get("pixels", []),
                "lower_line_inlier_pixels": lower.get("inlier_pixels", []),
                "reject_reason": reject_reason,
            }
        )
        timing["mask_build_ms"] = _stage_ms(mask_start)
        _finish_timing()
        return result, base_debug
