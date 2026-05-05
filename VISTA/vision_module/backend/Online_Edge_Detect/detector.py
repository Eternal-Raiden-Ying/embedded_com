#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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

    def _find_table_plane(self, pc_cam: np.ndarray) -> np.ndarray:
        y = pc_cam[:, 1]
        lo = float(self.cfg.table_y_min)
        hi = float(self.cfg.table_y_max)
        return (y > lo) & (y < hi)

    def _fit_line_on_inliers(self, x: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
        coeff = np.polyfit(x, z, 1)
        return float(coeff[0]), float(coeff[1])

    def _fit_edge_line(self, edge_points: np.ndarray):
        if edge_points.shape[0] < 2:
            return False, 0.0, 0.0, 0.0, None

        x = edge_points[:, 0]
        z = edge_points[:, 2]
        best_mask = None
        best_count = 0
        best_error = None
        threshold = float(self.cfg.residual_threshold_m)
        n = edge_points.shape[0]

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
            if count > best_count or (count == best_count and best_error is not None and mean_err < best_error):
                best_mask = mask
                best_count = count
                best_error = mean_err

        if best_mask is None or best_count < 2:
            return False, 0.0, 0.0, 0.0, None

        k, b = self._fit_line_on_inliers(x[best_mask], z[best_mask])
        yaw_err = math.atan(k)
        dist_err = b - float(self.target_dist_m)
        conf = float(best_count) / float(max(1, n))
        return True, float(yaw_err), float(dist_err), float(conf), (float(k), float(b))

    def process_depth(self, depth_image_16bit: np.ndarray, roi_override=None):
        valid_mask, depth_meters, roi_box = self._preprocess_depth(depth_image_16bit, roi_override=roi_override)
        pc_cam = self._depth_to_3d(depth_meters, valid_mask, roi_box)
        if len(pc_cam) < int(self.cfg.min_all_points):
            return EdgeDetectResult(False, 0.0, 0.0, 0.0, point_count=len(pc_cam), table_point_count=0), {
                "depth_meters": depth_meters,
                "pc_all": pc_cam,
                "pc_table": None,
                "roi_box": roi_box,
            }

        table_mask = self._find_table_plane(pc_cam)
        table_pc = pc_cam[table_mask]
        if len(table_pc) < int(self.cfg.min_table_points):
            return EdgeDetectResult(False, 0.0, 0.0, 0.0, point_count=len(pc_cam), table_point_count=len(table_pc)), {
                "depth_meters": depth_meters,
                "pc_all": pc_cam,
                "pc_table": table_pc,
                "roi_box": roi_box,
            }

        success, yaw_err, dist_err, conf, line_params = self._fit_edge_line(table_pc)
        line_k = line_params[0] if line_params is not None else None
        line_b = line_params[1] if line_params is not None else None
        result = EdgeDetectResult(
            success,
            yaw_err,
            dist_err,
            conf,
            line_k=line_k,
            line_b=line_b,
            point_count=len(pc_cam),
            table_point_count=len(table_pc),
        )
        return result, {
            "depth_meters": depth_meters,
            "pc_all": pc_cam,
            "pc_table": table_pc,
            "roi_box": roi_box,
        }
