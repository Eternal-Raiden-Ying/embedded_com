#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
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
        residual_score = max(0.0, 1.0 - residual_mean / max(1e-6, float(self.cfg.plane_max_residual_m)))
        span_score = min(1.0, x_span / max(1e-6, float(self.cfg.plane_min_x_span_m)))
        area_score = min(1.0, area_ratio / max(1e-6, float(self.cfg.front_face_min_area_ratio)))
        inlier_score = min(1.0, inlier_count / max(1.0, float(self.cfg.plane_min_inliers) * 2.0))
        conf = float(np.clip(0.30 * residual_score + 0.25 * span_score + 0.25 * area_score + 0.20 * inlier_score, 0.0, 1.0))
        all_xs = xs + int(roi_box[0])
        all_ys = ys + int(roi_box[1])
        return {
            "found": True,
            "candidate_mask": candidate_mask,
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
            "pixels": self._sample_pixels(all_xs, all_ys),
        }

    def _estimate_crease_line(self, depth_m: np.ndarray, valid_mask: np.ndarray, roi_box) -> Dict[str, Any]:
        x0, y0, _x1, _y1 = roi_box
        h, w = depth_m.shape[:2]
        candidates: List[Tuple[int, int, float]] = []
        win = max(3, int(self.cfg.trend_window_px))
        step = max(1, int(self.cfg.trend_col_step_px))
        min_ratio = float(self.cfg.trend_min_valid_ratio)
        min_valid = max(3, int(math.ceil(win * min_ratio)))
        min_delta = float(self.cfg.trend_min_slope_delta)
        if h < win * 2 + 1 or w <= 0:
            return {
                "found": False,
                "candidate_count": 0,
                "inlier_count": 0,
                "yaw": 0.0,
                "dist": 0.0,
                "confidence": 0.0,
                "residual_mean": float("inf"),
                "x_span": 0.0,
                "k": None,
                "b": None,
                "points": np.empty((0, 3), dtype=np.float32),
                "pixels": [],
                "inlier_pixels": [],
                "image_line_k": None,
                "image_line_b": None,
            }
        rows = np.arange(h, dtype=np.float64)
        for col in range(0, w, step):
            z_col = depth_m[:, col].astype(np.float64)
            valid_col = valid_mask[:, col] & np.isfinite(z_col)
            best_y = None
            best_score = 0.0
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
                if score > best_score:
                    best_score = score
                    best_y = yy
            if best_y is None or best_score < min_delta:
                continue
            y_start = max(0, best_y - 2)
            y_end = min(h, best_y + 3)
            local_valid = valid_mask[y_start:y_end, col]
            local_depth = depth_m[y_start:y_end, col]
            if np.any(local_valid):
                z_value = float(np.median(local_depth[local_valid]))
                depth_m[best_y, col] = z_value
            elif not valid_mask[best_y, col]:
                continue
            candidates.append((col, best_y, float(best_score)))
        candidate_count = int(len(candidates))
        if candidate_count <= 0:
            points = np.empty((0, 3), dtype=np.float32)
            pixels: List[List[int]] = []
        else:
            cols = np.array([c[0] for c in candidates], dtype=np.int64)
            ys = np.array([c[1] for c in candidates], dtype=np.int64)
            z = depth_m[ys, cols].astype(np.float64)
            good = np.isfinite(z) & (z > float(self.cfg.z_min)) & (z < float(self.cfg.z_max))
            cols = cols[good]
            ys = ys[good]
            z = z[good]
            u = cols.astype(np.float64) + float(x0)
            v = ys.astype(np.float64) + float(y0)
            x = (u - float(self.calib.cx)) * z / float(self.calib.fx)
            y = (v - float(self.calib.cy)) * z / float(self.calib.fy)
            points = np.vstack((x, y, z)).T.astype(np.float32)
            pixels = [[int(cx + x0), int(cy + y0)] for cx, cy in zip(cols.tolist(), ys.tolist())]
            candidate_count = int(points.shape[0])
        fit = self._fit_ransac_xz(points, threshold_m=float(self.cfg.line_max_residual_m))
        found = bool(fit.get("found")) and candidate_count >= int(self.cfg.trend_min_candidate_count)
        inlier_pixels: List[List[int]] = []
        image_line_k = None
        image_line_b = None
        if found:
            inlier_mask = fit.get("inlier_mask")
            if isinstance(inlier_mask, np.ndarray) and len(inlier_mask) == len(pixels):
                inlier_pixels = [pixels[i] for i, keep in enumerate(inlier_mask.tolist()) if keep]
            if len(inlier_pixels) >= 2:
                px = np.array([p[0] for p in inlier_pixels], dtype=np.float64)
                py = np.array([p[1] for p in inlier_pixels], dtype=np.float64)
                if float(px.max() - px.min()) > 1.0:
                    coeff = np.polyfit(px, py, 1)
                    image_line_k = float(coeff[0])
                    image_line_b = float(coeff[1])
        residual = float(fit.get("residual_mean", float("inf")))
        x_span = float(fit.get("x_span", 0.0) or 0.0)
        residual_score = max(0.0, 1.0 - residual / max(1e-6, float(self.cfg.line_max_residual_m))) if math.isfinite(residual) else 0.0
        span_score = min(1.0, x_span / max(1e-6, float(self.cfg.line_min_x_span_m)))
        count_score = min(1.0, candidate_count / max(1.0, float(self.cfg.trend_min_candidate_count) * 2.0))
        inlier_score = float(fit.get("confidence", 0.0) or 0.0)
        conf = float(np.clip(0.30 * residual_score + 0.25 * span_score + 0.25 * count_score + 0.20 * inlier_score, 0.0, 1.0))
        return {
            "found": bool(found),
            "candidate_count": candidate_count,
            "inlier_count": int(fit.get("inlier_count", 0) or 0) if found else 0,
            "yaw": float(fit.get("yaw", 0.0) or 0.0) if found else 0.0,
            "dist": float(fit.get("dist", 0.0) or 0.0) if found else 0.0,
            "confidence": conf if found else 0.0,
            "residual_mean": residual if found else float("inf"),
            "x_span": x_span if found else 0.0,
            "k": fit.get("k") if found else None,
            "b": fit.get("b") if found else None,
            "points": points,
            "pixels": pixels[:1600],
            "inlier_pixels": inlier_pixels[:1600],
            "image_line_k": image_line_k,
            "image_line_b": image_line_b,
        }

    def _fuse_front_pose(self, plane: Dict[str, Any], line: Dict[str, Any]) -> Dict[str, Any]:
        plane_reliable = (
            bool(plane.get("found"))
            and float(plane.get("area_ratio", 0.0) or 0.0) >= float(self.cfg.front_face_min_area_ratio)
            and float(plane.get("residual_mean", float("inf")) or float("inf")) <= float(self.cfg.plane_max_residual_m)
            and float(plane.get("x_span", 0.0) or 0.0) >= float(self.cfg.plane_min_x_span_m)
        )
        line_reliable = (
            bool(line.get("found"))
            and int(line.get("candidate_count", 0) or 0) >= int(self.cfg.trend_min_candidate_count)
            and float(line.get("residual_mean", float("inf")) or float("inf")) <= float(self.cfg.line_max_residual_m)
            and float(line.get("x_span", 0.0) or 0.0) >= float(self.cfg.line_min_x_span_m)
        )
        raw_found = bool(plane.get("found")) or bool(line.get("found"))
        if plane_reliable and line_reliable:
            yaw_delta = abs(float(plane["yaw"]) - float(line["yaw"]))
            if yaw_delta <= float(self.cfg.fusion_yaw_consistency_rad):
                wp = max(1e-6, float(plane.get("confidence", 0.0) or 0.0))
                wl = max(1e-6, float(line.get("confidence", 0.0) or 0.0))
                wsum = wp + wl
                return {
                    "raw_found": raw_found,
                    "pose_found": True,
                    "pose_source": "fused",
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
                "yaw": float(plane.get("yaw", 0.0) or 0.0),
                "dist": float(plane.get("dist", 0.0) or 0.0),
                "confidence": float(plane.get("confidence", 0.0) or 0.0),
                "line_k": plane.get("k"),
                "line_b": plane.get("b"),
                "x_span": float(plane.get("x_span", 0.0) or 0.0),
                "residual_mean": float(plane.get("residual_mean", float("inf"))),
                "reject_reason": "",
            }
        if line_reliable:
            return {
                "raw_found": raw_found,
                "pose_found": True,
                "pose_source": "crease_line",
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
            "yaw": 0.0,
            "dist": 0.0,
            "confidence": 0.0,
            "line_k": None,
            "line_b": None,
            "x_span": 0.0,
            "residual_mean": float("inf"),
            "reject_reason": "no_reliable_pose" if raw_found else "no_raw_geometry",
        }

    def _validate_pose_for_control(self, fused: Dict[str, Any]) -> Dict[str, Any]:
        pose_found = bool(fused.get("pose_found"))
        reason = str(fused.get("reject_reason") or "")
        valid_base = pose_found and fused.get("pose_source") != "conflict"
        yaw = float(fused.get("yaw", 0.0) or 0.0)
        dist = float(fused.get("dist", 0.0) or 0.0)
        conf = float(fused.get("confidence", 0.0) or 0.0)
        x_span = float(fused.get("x_span", 0.0) or 0.0)
        residual = float(fused.get("residual_mean", float("inf")) or float("inf"))
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
        if valid_base and self._last_pose is not None:
            last_yaw, last_dist = self._last_pose
            if abs(yaw - last_yaw) > float(self.cfg.control_max_yaw_jump_rad):
                valid_base = False
                reason = "yaw_jump"
            elif abs(dist - last_dist) > float(self.cfg.control_max_dist_jump_m):
                valid_base = False
                reason = "dist_jump"
        if valid_base:
            self._stable_count += 1
        else:
            self._stable_count = 0
        valid_for_control = bool(valid_base and self._stable_count >= int(self.cfg.control_min_stable_frames))
        if valid_base and not valid_for_control:
            reason = "stabilizing"
        if pose_found:
            self._last_pose = (yaw, dist)
        return {"valid_for_control": valid_for_control, "stable_count": int(self._stable_count), "reject_reason": reason}

    @staticmethod
    def _finite_or_zero(value: Any) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return 0.0
        return out if math.isfinite(out) else 0.0

    def process_depth(self, depth_image_16bit: np.ndarray, roi_override=None):
        valid_mask, depth_meters, roi_box = self._preprocess_depth(depth_image_16bit, roi_override=roi_override)
        pc_cam = self._depth_to_3d(depth_meters, valid_mask, roi_box)
        base_debug = {
            "depth_meters": depth_meters,
            "pc_all": pc_cam,
            "pc_table": None,
            "roi_box": roi_box,
            "front_plane_candidate_pixels": [],
            "crease_candidate_pixels": [],
            "crease_inlier_pixels": [],
        }
        if len(pc_cam) < int(self.cfg.min_all_points):
            self._stable_count = 0
            result = EdgeDetectResult(
                False,
                0.0,
                0.0,
                0.0,
                point_count=len(pc_cam),
                table_point_count=0,
                reject_reason="roi_empty",
            )
            base_debug.update({"reject_reason": result.reject_reason})
            return result, base_debug

        table_mask = self._find_table_plane(pc_cam)
        table_pc = pc_cam[table_mask]
        plane = self._estimate_front_plane(depth_meters, valid_mask, roi_box)
        line = self._estimate_crease_line(depth_meters.copy(), valid_mask, roi_box)
        fused = self._fuse_front_pose(plane, line)
        control = self._validate_pose_for_control(fused)
        edge_found = bool(fused.get("pose_found"))
        reject_reason = str(control.get("reject_reason") or fused.get("reject_reason") or "")
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
            inlier_count=int(max(int(plane.get("inlier_count", 0) or 0), int(line.get("inlier_count", 0) or 0))),
            stable_count=int(control.get("stable_count", 0) or 0),
            front_face_area_ratio=float(plane.get("area_ratio", 0.0) or 0.0),
            reject_reason=reject_reason,
            plane_yaw_err_rad=float(plane.get("yaw")) if plane.get("found") else None,
            plane_dist_err_m=float(plane.get("dist")) if plane.get("found") else None,
            line_yaw_err_rad=float(line.get("yaw")) if line.get("found") else None,
            line_dist_err_m=float(line.get("dist")) if line.get("found") else None,
            plane_k=plane.get("k"),
            plane_b=plane.get("b"),
            image_line_k=line.get("image_line_k"),
            image_line_b=line.get("image_line_b"),
        )
        base_debug.update(
            {
                "pc_table": table_pc,
                "front_plane": plane,
                "crease_line": line,
                "fused_pose": fused,
                "control_gate": control,
                "front_plane_candidate_pixels": plane.get("pixels", []),
                "crease_candidate_pixels": line.get("pixels", []),
                "crease_inlier_pixels": line.get("inlier_pixels", []),
                "reject_reason": reject_reason,
            }
        )
        return result, base_debug
