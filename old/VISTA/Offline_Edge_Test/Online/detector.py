#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

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
    image_line_k: Optional[float] = None
    image_line_b: Optional[float] = None
    point_count: int = 0
    table_point_count: int = 0
    edge_point_count: int = 0


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

    def _resolve_roi(self, depth_img: np.ndarray):
        height, width = depth_img.shape[:2]
        x0 = int(self.cfg.roi_x0)
        x1 = int(self.cfg.roi_x1)
        y0 = int(self.cfg.roi_y0)
        y1 = int(self.cfg.roi_y1)

        # Legacy defaults were tuned on roughly 640x480 preview frames.
        # When we run on 424x240 depth frames, directly clipping those values
        # pushes the ROI to the bottom-right. We rescale them first.
        if x1 > width or y1 > height:
            ref_w = 640.0
            ref_h = 480.0
            sx = float(width) / ref_w
            sy = float(height) / ref_h
            x0 = int(round(x0 * sx))
            x1 = int(round(x1 * sx))
            y0 = int(round(y0 * sy))
            y1 = int(round(y1 * sy))

        x0 = max(0, min(width - 1, x0))
        x1 = max(x0 + 1, min(width, x1))
        y0 = max(0, min(height - 1, y0))
        y1 = max(y0 + 1, min(height, y1))
        return x0, y0, x1, y1

    def _preprocess_depth(self, depth_img: np.ndarray):
        x0, y0, x1, y1 = self._resolve_roi(depth_img)
        if y1 <= y0 or x1 <= x0:
            raise ValueError("invalid ROI range")
        depth_roi = depth_img[y0:y1, x0:x1]
        ksize = int(self.cfg.depth_median_ksize)
        if ksize > 1 and ksize % 2 == 1:
            depth_filtered = cv2.medianBlur(depth_roi, ksize)
        else:
            depth_filtered = depth_roi
        depth_m = depth_filtered.astype(np.float32) * float(self.calib.depth_scale)
        mask = (depth_m > float(self.cfg.z_min)) & (depth_m < float(self.cfg.z_max))
        return mask, depth_m, (x0, y0, x1, y1)

    def _depth_to_3d(self, depth_m: np.ndarray, mask: np.ndarray, roi_box):
        x0, y0, x1, y1 = roi_box
        u, v = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        u_valid, v_valid, z_valid = u[mask], v[mask], depth_m[mask]
        x_c = (u_valid - self.calib.cx) * z_valid / self.calib.fx
        y_c = (v_valid - self.calib.cy) * z_valid / self.calib.fy
        pc = np.vstack((x_c, y_c, z_valid)).T
        uv = np.vstack((u_valid, v_valid)).T.astype(np.int32)
        return pc, uv

    def _find_table_plane(self, pc_cam: np.ndarray) -> np.ndarray:
        y = pc_cam[:, 1]
        return (y > float(self.cfg.table_y_min)) & (y < float(self.cfg.table_y_max))

    def _fit_line_on_inliers(self, x: np.ndarray, z: np.ndarray):
        coeff = np.polyfit(x, z, 1)
        return float(coeff[0]), float(coeff[1])

    def _fit_ransac_line(self, x: np.ndarray, y: np.ndarray, threshold: float):
        if x.shape[0] < 2:
            return False, 0.0, 0.0, 0.0, None
        best_mask = None
        best_count = 0
        best_error = None
        n = x.shape[0]
        for _ in range(max(8, int(self.cfg.ransac_iters))):
            idx = self._rng.choice(n, size=2, replace=False)
            x1, x2 = float(x[idx[0]]), float(x[idx[1]])
            y1, y2 = float(y[idx[0]]), float(y[idx[1]])
            if abs(x2 - x1) < 1e-6:
                continue
            k = (y2 - y1) / (x2 - x1)
            b = y1 - k * x1
            residual = np.abs(y - (k * x + b))
            mask = residual <= threshold
            count = int(mask.sum())
            mean_err = float(residual[mask].mean()) if count > 0 else float("inf")
            if count > best_count or (count == best_count and best_error is not None and mean_err < best_error):
                best_mask = mask
                best_count = count
                best_error = mean_err
        if best_mask is None or best_count < 2:
            return False, 0.0, 0.0, 0.0, None
        k, b = self._fit_line_on_inliers(x[best_mask], y[best_mask])
        conf = float(best_count) / float(max(1, n))
        return True, float(k), float(b), float(conf), best_mask

    def _extract_edge_frontier(self, table_pc: np.ndarray, table_uv: np.ndarray):
        if table_pc.shape[0] == 0:
            return None, None
        u = table_uv[:, 0]
        unique_u = np.unique(u)
        edge_pc = []
        edge_uv = []
        for cur_u in unique_u:
            idx = np.where(u == cur_u)[0]
            if idx.size == 0:
                continue
            col_points = table_pc[idx]
            nearest_idx = idx[int(np.argmin(col_points[:, 2]))]
            edge_pc.append(table_pc[nearest_idx])
            edge_uv.append(table_uv[nearest_idx])
        if len(edge_pc) < 2:
            return None, None
        edge_pc = np.asarray(edge_pc, dtype=np.float32)
        edge_uv = np.asarray(edge_uv, dtype=np.int32)

        order = np.argsort(edge_uv[:, 0])
        edge_pc = edge_pc[order]
        edge_uv = edge_uv[order]

        # Keep the longest visually continuous frontier segment instead of
        # blindly fitting all "nearest" points. Large camera tilt often creates
        # multiple disconnected candidate fragments, and mixing them tends to
        # collapse the final fit toward a near-horizontal compromise line.
        u_vals = edge_uv[:, 0].astype(np.float32)
        v_vals = edge_uv[:, 1].astype(np.float32)
        z_vals = edge_pc[:, 2].astype(np.float32)

        if len(edge_uv) >= 5:
            # Simple 1D smoothing to suppress isolated spikes before continuity checks.
            kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
            kernel /= kernel.sum()
            v_smooth = np.convolve(v_vals, kernel, mode="same")
            z_smooth = np.convolve(z_vals, kernel, mode="same")
        else:
            v_smooth = v_vals
            z_smooth = z_vals

        max_u_gap = 2.0
        max_v_jump = max(8.0, 0.06 * 240.0)
        max_z_jump = 0.08

        segments = []
        seg_start = 0
        for idx in range(1, len(edge_uv)):
            du = abs(u_vals[idx] - u_vals[idx - 1])
            dv = abs(v_smooth[idx] - v_smooth[idx - 1])
            dz = abs(z_smooth[idx] - z_smooth[idx - 1])
            if du > max_u_gap or dv > max_v_jump or dz > max_z_jump:
                segments.append((seg_start, idx))
                seg_start = idx
        segments.append((seg_start, len(edge_uv)))

        best = None
        best_score = None
        for start, end in segments:
            length = end - start
            if length < 12:
                continue
            seg_u = u_vals[start:end]
            seg_v = v_smooth[start:end]
            seg_z = z_smooth[start:end]
            coverage = float(seg_u[-1] - seg_u[0]) if length > 1 else 0.0
            v_span = float(np.max(seg_v) - np.min(seg_v)) if length > 0 else 0.0
            z_span = float(np.max(seg_z) - np.min(seg_z)) if length > 0 else 0.0
            # Favor long, wide, diagonally varying frontier segments.
            score = (length, coverage, v_span + 12.0 * z_span)
            if best_score is None or score > best_score:
                best = (start, end)
                best_score = score

        if best is not None:
            start, end = best
            edge_pc = edge_pc[start:end]
            edge_uv = edge_uv[start:end]

        if len(edge_pc) < 2:
            return None, None
        return edge_pc, edge_uv

    def process_depth(self, depth_image_16bit: np.ndarray):
        valid_mask, depth_meters, roi_box = self._preprocess_depth(depth_image_16bit)
        pc_cam, uv_cam = self._depth_to_3d(depth_meters, valid_mask, roi_box)
        if len(pc_cam) < int(self.cfg.min_all_points):
            return EdgeDetectResult(False, 0.0, 0.0, 0.0, point_count=len(pc_cam), table_point_count=0), {
                "depth_meters": depth_meters,
                "pc_all": pc_cam,
                "uv_all": uv_cam,
                "pc_table": None,
                "uv_table": None,
                "edge_uv": None,
                "roi_box": roi_box,
            }
        table_mask = self._find_table_plane(pc_cam)
        table_pc = pc_cam[table_mask]
        table_uv = uv_cam[table_mask]
        if len(table_pc) < int(self.cfg.min_table_points):
            return EdgeDetectResult(False, 0.0, 0.0, 0.0, point_count=len(pc_cam), table_point_count=len(table_pc)), {
                "depth_meters": depth_meters,
                "pc_all": pc_cam,
                "uv_all": uv_cam,
                "pc_table": table_pc,
                "uv_table": table_uv,
                "edge_uv": None,
                "roi_box": roi_box,
            }
        edge_pc, edge_uv = self._extract_edge_frontier(table_pc, table_uv)
        if edge_pc is None or edge_uv is None or len(edge_pc) < 2:
            return EdgeDetectResult(
                False,
                0.0,
                0.0,
                0.0,
                point_count=len(pc_cam),
                table_point_count=len(table_pc),
                edge_point_count=0,
            ), {
                "depth_meters": depth_meters,
                "pc_all": pc_cam,
                "uv_all": uv_cam,
                "pc_table": table_pc,
                "uv_table": table_uv,
                "edge_uv": None,
                "roi_box": roi_box,
            }

        success_3d, line_k, line_b, conf_3d, mask_3d = self._fit_ransac_line(
            edge_pc[:, 0],
            edge_pc[:, 2],
            float(self.cfg.residual_threshold_m),
        )
        success_img, image_k, image_b, conf_img, mask_img = self._fit_ransac_line(
            edge_uv[:, 0].astype(np.float32),
            edge_uv[:, 1].astype(np.float32),
            3.5,
        )

        success = bool(success_3d and success_img)
        yaw_err = math.atan(line_k) if success_3d else 0.0
        dist_err = float(line_b) - float(self.target_dist_m) if success_3d else 0.0
        conf = min(conf_3d, conf_img) if success else max(conf_3d, conf_img)
        result = EdgeDetectResult(
            success,
            yaw_err,
            dist_err,
            conf,
            line_k=(float(line_k) if success_3d else None),
            line_b=(float(line_b) if success_3d else None),
            image_line_k=(float(image_k) if success_img else None),
            image_line_b=(float(image_b) if success_img else None),
            point_count=len(pc_cam),
            table_point_count=len(table_pc),
            edge_point_count=len(edge_pc),
        )
        return result, {
            "depth_meters": depth_meters,
            "pc_all": pc_cam,
            "uv_all": uv_cam,
            "pc_table": table_pc,
            "uv_table": table_uv,
            "edge_uv": edge_uv,
            "edge_uv_inliers": (edge_uv[mask_img] if (success_img and mask_img is not None) else edge_uv),
            "roi_box": roi_box,
        }
