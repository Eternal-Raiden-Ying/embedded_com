#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
VISTA_ROOT = HERE.parents[2]
for item in (REPO_ROOT, VISTA_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))


DEFAULT_FRAMES = (300, 360, 390, 420, 450, 480, 510, 594, 744)
CSV_COLUMNS = (
    "frame_seq",
    "u",
    "v",
    "depth_m",
    "rgb_r",
    "rgb_g",
    "rgb_b",
    "X_cam",
    "Y_cam",
    "Z_cam",
    "X_robot",
    "Y_robot",
    "Z_robot",
    "x_bin",
    "column_bin",
    "y_cluster",
    "is_sampled",
    "is_height_candidate",
    "is_vertical_support",
    "is_representative",
    "is_fit_inlier",
    "reject_stage",
    "point_stage",
    "support_count",
    "z_span_m",
    "cluster_y_center",
    "representative_id",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export point-level fast table-plane diagnostics for key bag frames.")
    parser.add_argument("--bag", type=Path, default=VISTA_ROOT / "20260516_161436.bag", help="RealSense .bag path.")
    parser.add_argument("--frames", default=",".join(str(v) for v in DEFAULT_FRAMES), help="Comma-separated frame_seq values.")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "runs" / "point_debug_keyframes", help="Output directory.")
    parser.add_argument("--config", type=Path, default=VISTA_ROOT / "configs" / "vision_params.yaml", help="Vision params YAML.")
    return parser


def _parse_frames(raw: str) -> List[int]:
    return sorted({int(part.strip()) for part in str(raw or "").split(",") if part.strip()})


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return str(value)


def _stats(values: Any) -> Dict[str, Optional[float]]:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return {"min": None, "p10": None, "p50": None, "p90": None, "max": None}
    return {
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _ratio(mask: np.ndarray, denom: int) -> float:
    return float(np.sum(mask)) / float(max(1, int(denom)))


def _resolve_roi(manager: Any, depth: np.ndarray) -> Tuple[Tuple[int, int, int, int], Dict[str, Any]]:
    roi_meta = manager._select_roi(depth)
    roi_box = roi_meta.get("depth_edge_roi") if roi_meta.get("roi_source") != "static_fallback" else manager._static_roi()
    if roi_box is None:
        roi_box = manager._static_roi()
    try:
        roi_box = manager._detector._resolve_roi(depth, roi_override=roi_box) if manager._detector is not None else tuple(int(v) for v in roi_box)
    except Exception:
        roi_box = tuple(int(v) for v in manager._static_roi() or (0, 0, depth.shape[1], depth.shape[0]))
    x0, y0, x1, y1 = [int(v) for v in roi_box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(int(depth.shape[1]), x1), min(int(depth.shape[0]), y1)
    return (x0, y0, x1, y1), roi_meta


def _select_representatives_with_labels(
    x_robot: np.ndarray,
    y_robot: np.ndarray,
    z_robot: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    *,
    x_bin_width_m: float,
    y_cluster_bin_m: float,
    min_support_points: int,
    min_z_span_m: float,
) -> Dict[str, Any]:
    n = int(min(len(x_robot), len(y_robot), len(z_robot), len(px), len(py)))
    support_mask = np.zeros(n, dtype=bool)
    representative_mask = np.zeros(n, dtype=bool)
    representative_id = np.full(n, -1, dtype=np.int32)
    support_count = np.zeros(n, dtype=np.int32)
    z_span_out = np.full(n, np.nan, dtype=np.float32)
    cluster_y_center = np.full(n, np.nan, dtype=np.float32)
    y_cluster = np.full(n, -1, dtype=np.int32)
    reps: List[Dict[str, Any]] = []
    if n <= 0:
        return {
            "count": 0,
            "support_mask": support_mask,
            "representative_mask": representative_mask,
            "representative_id": representative_id,
            "support_count": support_count,
            "z_span_point": z_span_out,
            "cluster_y_center_point": cluster_y_center,
            "y_cluster": y_cluster,
        }

    x_bins = np.floor(x_robot[:n] / max(1e-6, float(x_bin_width_m))).astype(np.int32)
    y_radius_bins = max(1, int(math.ceil(0.08 / max(1e-6, float(y_cluster_bin_m)))))
    for xb in sorted(set(int(v) for v in x_bins.tolist())):
        x_mask = x_bins == xb
        if int(x_mask.sum()) < int(min_support_points):
            continue
        idxs = np.nonzero(x_mask)[0]
        y_bins = np.floor(y_robot[idxs] / max(1e-6, float(y_cluster_bin_m))).astype(np.int32)
        best = None
        for yb in sorted(set(int(v) for v in y_bins.tolist())):
            local = idxs[np.abs(y_bins - int(yb)) <= y_radius_bins]
            support = int(len(local))
            if support < int(min_support_points):
                continue
            z_span = float(np.max(z_robot[local]) - np.min(z_robot[local])) if support > 1 else 0.0
            if z_span < float(min_z_span_m):
                continue
            y_spread = float(np.percentile(y_robot[local], 90) - np.percentile(y_robot[local], 10)) if support > 2 else 0.0
            if y_spread > max(0.16, float(y_cluster_bin_m) * float(2 * y_radius_bins + 1)):
                continue
            score = float(support) * float(z_span) / max(0.04, y_spread + 0.02)
            med_x, med_y, med_z = float(np.median(x_robot[local])), float(np.median(y_robot[local])), float(np.median(z_robot[local]))
            nearest = int(local[np.argmin((x_robot[local] - med_x) ** 2 + (y_robot[local] - med_y) ** 2 + (z_robot[local] - med_z) ** 2)])
            item = {
                "score": score,
                "support": support,
                "z_span": z_span,
                "x": med_x,
                "y": med_y,
                "z": med_z,
                "px": int(np.median(px[local])),
                "py": int(np.median(py[local])),
                "y_spread": y_spread,
                "local": local,
                "nearest": nearest,
                "yb": int(yb),
            }
            if best is None or item["score"] > best["score"]:
                best = item
        if best is not None:
            reps.append(best)

    reps.sort(key=lambda item: item["x"])
    for rid, item in enumerate(reps):
        local = np.asarray(item["local"], dtype=np.int32)
        support_mask[local] = True
        representative_id[local] = int(rid)
        support_count[local] = int(item["support"])
        z_span_out[local] = float(item["z_span"])
        cluster_y_center[local] = float(item["y"])
        y_cluster[local] = int(item["yb"])
        representative_mask[int(item["nearest"])] = True

    if not reps:
        return {
            "count": 0,
            "support_mask": support_mask,
            "representative_mask": representative_mask,
            "representative_id": representative_id,
            "support_count": support_count,
            "z_span_point": z_span_out,
            "cluster_y_center_point": cluster_y_center,
            "y_cluster": y_cluster,
        }
    return {
        "count": int(len(reps)),
        "x": np.asarray([r["x"] for r in reps], dtype=np.float32),
        "y": np.asarray([r["y"] for r in reps], dtype=np.float32),
        "z": np.asarray([r["z"] for r in reps], dtype=np.float32),
        "px": np.asarray([r["px"] for r in reps], dtype=np.int32),
        "py": np.asarray([r["py"] for r in reps], dtype=np.int32),
        "support": np.asarray([r["support"] for r in reps], dtype=np.int32),
        "z_span": np.asarray([r["z_span"] for r in reps], dtype=np.float32),
        "y_spread": np.asarray([r["y_spread"] for r in reps], dtype=np.float32),
        "support_total": int(sum(int(r["support"]) for r in reps)),
        "support_mask": support_mask,
        "representative_mask": representative_mask,
        "representative_id": representative_id,
        "support_count": support_count,
        "z_span_point": z_span_out,
        "cluster_y_center_point": cluster_y_center,
        "y_cluster": y_cluster,
    }


def _fit_representatives(reps: Dict[str, Any], min_front_face_columns: int, residual_threshold: float) -> Dict[str, Any]:
    x_t = np.asarray(reps.get("x", []), dtype=np.float32)
    y_t = np.asarray(reps.get("y", []), dtype=np.float32)
    if len(x_t) < int(min_front_face_columns):
        return {"fit_attempted": False, "inlier": np.zeros(len(x_t), dtype=bool), "reject_reason": "front_face_columns_low"}
    try:
        k, b = np.polyfit(x_t, y_t, 1)
        residual = np.abs(y_t - (float(k) * x_t + float(b)))
        inlier = residual <= float(residual_threshold)
        if int(inlier.sum()) >= int(min_front_face_columns):
            k, b = np.polyfit(x_t[inlier], y_t[inlier], 1)
            residual = np.abs(y_t - (float(k) * x_t + float(b)))
            inlier = residual <= float(residual_threshold)
        return {
            "fit_attempted": True,
            "k": float(k),
            "b": float(b),
            "inlier": inlier,
            "residual": residual.astype(np.float32),
            "residual_mean": float(np.mean(residual[inlier])) if int(inlier.sum()) > 0 else float(np.mean(residual)),
            "residual_p90": float(np.percentile(residual[inlier], 90)) if int(inlier.sum()) > 0 else float(np.percentile(residual, 90)),
            "inlier_count": int(inlier.sum()),
        }
    except Exception as exc:
        return {"fit_attempted": True, "inlier": np.zeros(len(x_t), dtype=bool), "reject_reason": f"birdview_fit_failed:{exc}"}


def _build_point_data(manager: Any, depth: np.ndarray, rgb: Optional[np.ndarray], obs: Dict[str, Any], frame_seq: int) -> Dict[str, Any]:
    cfg = manager._detector_cfg
    table_edge_cfg = getattr(manager.cfg, "table_edge", None)
    roi_box, roi_meta = _resolve_roi(manager, depth)
    x0, y0, x1, y1 = roi_box
    stride = max(1, int(manager._fast_plane_stride))
    yy_grid, xx_grid = np.mgrid[y0:y1:stride, x0:x1:stride]
    u = xx_grid.reshape(-1).astype(np.float32)
    v = yy_grid.reshape(-1).astype(np.float32)
    raw_depth = depth[yy_grid, xx_grid].reshape(-1)
    depth_m = raw_depth.astype(np.float32, copy=False)
    if raw_depth.dtype != np.float32:
        scale = float(getattr(manager._detector.calib, "depth_scale", 0.001) if manager._detector is not None else 0.001)
        depth_m = depth_m * scale
    z_min = float(getattr(cfg, "z_min", 0.2) if cfg is not None else 0.2)
    z_max = float(getattr(cfg, "z_max", 2.0) if cfg is not None else 2.0)
    valid = (depth_m > z_min) & (depth_m < z_max) & np.isfinite(depth_m)

    n = int(len(depth_m))
    x_cam = np.full(n, np.nan, dtype=np.float32)
    y_cam = np.full(n, np.nan, dtype=np.float32)
    z_cam = np.full(n, np.nan, dtype=np.float32)
    x_robot = np.full(n, np.nan, dtype=np.float32)
    y_robot = np.full(n, np.nan, dtype=np.float32)
    z_robot = np.full(n, np.nan, dtype=np.float32)

    if manager._detector is not None and int(valid.sum()) > 0:
        calib = manager._detector.calib
        z_cam[valid] = depth_m[valid]
        x_cam[valid] = (u[valid] - float(calib.cx)) * z_cam[valid] / float(calib.fx)
        y_cam[valid] = (v[valid] - float(calib.cy)) * z_cam[valid] / float(calib.fy)
        pitch_deg = float(getattr(table_edge_cfg, "camera_pitch_deg", 30.0) or 30.0)
        camera_height_m = float(getattr(table_edge_cfg, "camera_height_m", 0.60) or 0.60)
        xr, yr, zr = manager._camera_points_to_robot(x_cam[valid], y_cam[valid], z_cam[valid], pitch_deg=pitch_deg, camera_height_m=camera_height_m)
        x_robot[valid], y_robot[valid], z_robot[valid] = xr, yr, zr

    table_height_m = float(getattr(table_edge_cfg, "table_height_m", 0.40) or 0.40)
    front_face_z_min = float(getattr(table_edge_cfg, "front_face_z_min_m", 0.03) or 0.03)
    front_face_z_max = float(getattr(table_edge_cfg, "front_face_z_max_m", 0.43) or 0.43)
    min_vertical_z_span = float(getattr(table_edge_cfg, "min_vertical_z_span_m", 0.12) or 0.12)
    min_vertical_support = max(1, int(float(getattr(table_edge_cfg, "min_vertical_support_points", 3) or 3)))
    x_bin_width_m = float(getattr(table_edge_cfg, "x_bin_width_m", 0.04) or 0.04)
    y_cluster_bin_m = float(getattr(table_edge_cfg, "y_cluster_bin_m", 0.04) or 0.04)
    min_front_face_columns = max(2, int(float(getattr(table_edge_cfg, "min_front_face_columns", 3) or 3)))
    min_front_face_x_span = float(getattr(table_edge_cfg, "min_front_face_x_span_m", 0.07) or 0.07)
    residual_threshold = max(0.035, float(getattr(cfg, "front_plane_max_residual_m", 0.035) if cfg is not None else 0.035) * 1.5)
    height_candidate = valid & (z_robot > front_face_z_min) & (z_robot < front_face_z_max)
    candidate_idx = np.nonzero(height_candidate)[0]
    reps = _select_representatives_with_labels(
        x_robot[candidate_idx],
        y_robot[candidate_idx],
        z_robot[candidate_idx],
        u[candidate_idx].astype(np.int32),
        v[candidate_idx].astype(np.int32),
        x_bin_width_m=x_bin_width_m,
        y_cluster_bin_m=y_cluster_bin_m,
        min_support_points=min_vertical_support,
        min_z_span_m=min_vertical_z_span,
    )
    fit = _fit_representatives(reps, min_front_face_columns, residual_threshold)

    vertical_support = np.zeros(n, dtype=bool)
    representative = np.zeros(n, dtype=bool)
    fit_inlier = np.zeros(n, dtype=bool)
    rep_id = np.full(n, -1, dtype=np.int32)
    support_count = np.zeros(n, dtype=np.int32)
    z_span_point = np.full(n, np.nan, dtype=np.float32)
    cluster_y_center = np.full(n, np.nan, dtype=np.float32)
    y_cluster = np.full(n, -1, dtype=np.int32)
    if candidate_idx.size:
        vertical_support[candidate_idx] = np.asarray(reps.get("support_mask", []), dtype=bool)
        representative[candidate_idx] = np.asarray(reps.get("representative_mask", []), dtype=bool)
        rep_id[candidate_idx] = np.asarray(reps.get("representative_id", []), dtype=np.int32)
        support_count[candidate_idx] = np.asarray(reps.get("support_count", []), dtype=np.int32)
        z_span_point[candidate_idx] = np.asarray(reps.get("z_span_point", []), dtype=np.float32)
        cluster_y_center[candidate_idx] = np.asarray(reps.get("cluster_y_center_point", []), dtype=np.float32)
        y_cluster[candidate_idx] = np.asarray(reps.get("y_cluster", []), dtype=np.int32)
        rep_inlier = np.asarray(fit.get("inlier", []), dtype=bool)
        local_rep_id = np.asarray(reps.get("representative_id", []), dtype=np.int32)
        if rep_inlier.size:
            local_inlier = (local_rep_id >= 0) & (local_rep_id < rep_inlier.size) & rep_inlier[np.clip(local_rep_id, 0, max(0, rep_inlier.size - 1))]
            fit_inlier[candidate_idx] = local_inlier

    x_bin = np.full(n, -1, dtype=np.int32)
    finite_x = np.isfinite(x_robot)
    x_bin[finite_x] = np.floor(x_robot[finite_x] / max(1e-6, x_bin_width_m)).astype(np.int32)
    reject_stage = np.full(n, "depth_invalid", dtype=object)
    reject_stage[valid] = "height_rejected"
    reject_stage[height_candidate] = "candidate_no_vertical_support"
    reject_stage[vertical_support] = "vertical_support"
    reject_stage[representative] = "representative"
    reject_stage[fit_inlier] = "fit_inlier"

    rgb_r = np.full(n, -1, dtype=np.int16)
    rgb_g = np.full(n, -1, dtype=np.int16)
    rgb_b = np.full(n, -1, dtype=np.int16)
    if rgb is not None and getattr(rgb, "ndim", 0) == 3:
        h, w = rgb.shape[:2]
        uu = np.clip(u.astype(np.int32), 0, w - 1)
        vv = np.clip(v.astype(np.int32), 0, h - 1)
        # RealSense color frames are RGB in bag replay.
        rgb_r = rgb[vv, uu, 0].astype(np.int16)
        rgb_g = rgb[vv, uu, 1].astype(np.int16)
        rgb_b = rgb[vv, uu, 2].astype(np.int16)

    return {
        "frame_seq": np.full(n, int(frame_seq), dtype=np.int32),
        "u": u,
        "v": v,
        "depth_m": depth_m.astype(np.float32),
        "rgb_r": rgb_r,
        "rgb_g": rgb_g,
        "rgb_b": rgb_b,
        "X_cam": x_cam,
        "Y_cam": y_cam,
        "Z_cam": z_cam,
        "X_robot": x_robot,
        "Y_robot": y_robot,
        "Z_robot": z_robot,
        "x_bin": x_bin,
        "column_bin": x_bin.copy(),
        "y_cluster": y_cluster,
        "is_sampled": np.ones(n, dtype=bool),
        "is_height_candidate": height_candidate,
        "is_vertical_support": vertical_support,
        "is_representative": representative,
        "is_fit_inlier": fit_inlier,
        "reject_stage": reject_stage,
        "point_stage": reject_stage.copy(),
        "support_count": support_count,
        "z_span_m": z_span_point,
        "cluster_y_center": cluster_y_center,
        "representative_id": rep_id,
        "_roi_box": roi_box,
        "_roi_meta": roi_meta,
        "_reps": reps,
        "_fit": fit,
        "_obs": obs,
        "_params": {
            "stride": stride,
            "table_height_m": table_height_m,
            "camera_height_m": float(getattr(table_edge_cfg, "camera_height_m", 0.60) or 0.60),
            "front_face_z_min_m": front_face_z_min,
            "front_face_z_max_m": front_face_z_max,
            "min_vertical_z_span_m": min_vertical_z_span,
            "min_vertical_support_points": min_vertical_support,
            "x_bin_width_m": x_bin_width_m,
            "y_cluster_bin_m": y_cluster_bin_m,
            "min_front_face_columns": min_front_face_columns,
            "min_front_face_x_span_m": min_front_face_x_span,
            "residual_threshold_m": residual_threshold,
        },
    }


def _write_csv(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(len(data["u"]))
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for i in range(n):
            row = {}
            for key in CSV_COLUMNS:
                value = data[key][i]
                if isinstance(value, np.generic):
                    value = value.item()
                row[key] = value
            writer.writerow(row)


def _write_npz(path: Path, data: Dict[str, Any]) -> None:
    arrays = {key: value for key, value in data.items() if not key.startswith("_") and isinstance(value, np.ndarray)}
    np.savez_compressed(path, **arrays)


def _blank_plot(title: str, xlabel: str, ylabel: str, *, width: int = 1100, height: int = 820) -> Tuple[Any, Tuple[int, int, int, int]]:
    import cv2

    img = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(img, title, (70, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(img, xlabel, (width // 2 - 120, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(img, ylabel, (16, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (25, 25, 25), 1, cv2.LINE_AA)
    plot = (90, 60, width - 40, height - 80)
    cv2.rectangle(img, (plot[0], plot[1]), (plot[2], plot[3]), (40, 40, 40), 1)
    return img, plot


def _bounds(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size <= 0:
        return 0.0, 1.0
    lo = float(np.percentile(values, 1))
    hi = float(np.percentile(values, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-6:
        lo, hi = float(np.min(values)), float(np.max(values))
    pad = max(0.02, abs(hi - lo) * 0.06)
    return lo - pad, hi + pad


def _project_points(x: np.ndarray, y: np.ndarray, bounds: Tuple[float, float, float, float], plot: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    x_min, x_max, y_min, y_max = bounds
    x0, y0, x1, y1 = plot
    sx = (x1 - x0) / max(1e-6, x_max - x_min)
    sy = (y1 - y0) / max(1e-6, y_max - y_min)
    px = x0 + (x - x_min) * sx
    py = y1 - (y - y_min) * sy
    return px.astype(np.int32), py.astype(np.int32)


def _draw_grid(img: np.ndarray, plot: Tuple[int, int, int, int], bounds: Tuple[float, float, float, float]) -> None:
    import cv2

    x0, y0, x1, y1 = plot
    x_min, x_max, y_min, y_max = bounds
    for frac in np.linspace(0, 1, 6):
        x = int(x0 + frac * (x1 - x0))
        y = int(y1 - frac * (y1 - y0))
        cv2.line(img, (x, y0), (x, y1), (225, 225, 225), 1)
        cv2.line(img, (x0, y), (x1, y), (225, 225, 225), 1)
        cv2.putText(img, f"{x_min + frac * (x_max - x_min):.2f}", (x - 22, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(img, f"{y_min + frac * (y_max - y_min):.2f}", (x0 - 62, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)


def _draw_legend(img: np.ndarray, labels: Iterable[Tuple[str, Tuple[int, int, int]]]) -> None:
    import cv2

    x, y = 760, 78
    for idx, (label, color) in enumerate(labels):
        yy = y + idx * 24
        cv2.circle(img, (x, yy - 5), 6, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(img, label, (x + 16, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (35, 35, 35), 1, cv2.LINE_AA)


def _plot_overlay(path: Path, depth: np.ndarray, rgb: Optional[np.ndarray], data: Dict[str, Any]) -> None:
    import cv2

    if rgb is not None and getattr(rgb, "ndim", 0) == 3:
        base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    else:
        d = depth.astype(np.float32)
        valid = np.isfinite(d) & (d > 0)
        lo = float(np.percentile(d[valid], 2)) if np.any(valid) else 0.0
        hi = float(np.percentile(d[valid], 98)) if np.any(valid) else 1.0
        norm = np.clip((d - lo) / max(1e-6, hi - lo), 0.0, 1.0)
        base = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    x0, y0, x1, y1 = data["_roi_box"]
    cv2.rectangle(base, (x0, y0), (x1, y1), (230, 230, 230), 1)

    layers = (
        ("is_sampled", (130, 130, 130), 1, 0.18, 1400),
        ("is_height_candidate", (0, 165, 255), 2, 0.65, 2600),
        ("is_vertical_support", (255, 210, 0), 3, 0.75, 2600),
        ("is_representative", (255, 0, 255), 5, 1.0, 2600),
        ("is_fit_inlier", (40, 230, 40), 4, 0.95, 2600),
    )
    overlay = base.copy()
    for key, color, radius, alpha, cap in layers:
        idx = np.nonzero(np.asarray(data[key], dtype=bool))[0]
        if idx.size <= 0:
            continue
        step = max(1, int(math.ceil(idx.size / float(cap))))
        for i in idx[::step]:
            cv2.circle(overlay, (int(data["u"][i]), int(data["v"][i])), int(radius), color, -1, lineType=cv2.LINE_AA)
        base = cv2.addWeighted(overlay, float(alpha), base, 1.0 - float(alpha), 0)
        overlay = base.copy()
    labels = [
        ("sampled", (130, 130, 130)),
        ("height", (0, 165, 255)),
        ("support", (255, 210, 0)),
        ("rep", (255, 0, 255)),
        ("inlier", (40, 230, 40)),
    ]
    for j, (text, color) in enumerate(labels):
        y = 22 + j * 20
        cv2.circle(base, (18, y - 5), 5, color, -1)
        cv2.putText(base, text, (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(base, text, (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), base)


def _draw_scatter_layers(img: np.ndarray, plot: Tuple[int, int, int, int], bounds: Tuple[float, float, float, float], data: Dict[str, Any], x_key: str, y_key: str) -> None:
    import cv2

    layers = (
        ("is_sampled", "sampled", (138, 138, 138), 2, 9000),
        ("is_height_candidate", "height_candidate", (43, 142, 242), 3, 7000),
        ("is_vertical_support", "vertical_support", (167, 121, 78), 4, 7000),
        ("is_representative", "representative", (238, 58, 178), 7, 7000),
        ("is_fit_inlier", "fit_inlier", (44, 160, 44), 5, 7000),
    )
    for mask_key, _label, color, radius, cap in layers:
        mask = np.asarray(data[mask_key], dtype=bool) & np.isfinite(data[x_key]) & np.isfinite(data[y_key])
        idx = np.nonzero(mask)[0]
        if idx.size <= 0:
            continue
        step = max(1, int(math.ceil(idx.size / float(cap))))
        px, py = _project_points(data[x_key][idx[::step]], data[y_key][idx[::step]], bounds, plot)
        for x, y in zip(px, py):
            if plot[0] <= x <= plot[2] and plot[1] <= y <= plot[3]:
                cv2.circle(img, (int(x), int(y)), int(radius), color, -1, lineType=cv2.LINE_AA)


def _plot_topview(path: Path, data: Dict[str, Any]) -> None:
    import cv2

    img, plot = _blank_plot(f"Frame {int(data['frame_seq'][0])} robot XY top view", "X_robot lateral (m)", "Y_robot forward (m)")
    x_min, x_max = _bounds(data["X_robot"])
    y_min, y_max = _bounds(data["Y_robot"])
    bounds = (x_min, x_max, y_min, y_max)
    _draw_grid(img, plot, bounds)
    _draw_scatter_layers(img, plot, bounds, data, "X_robot", "Y_robot")
    fit = data["_fit"]
    if fit.get("fit_attempted") and fit.get("k") is not None:
        xs = data["X_robot"][np.asarray(data["is_fit_inlier"], dtype=bool)]
        xs = xs[np.isfinite(xs)]
        if xs.size >= 2:
            line_x = np.linspace(float(np.min(xs)), float(np.max(xs)), 100)
            line_y = float(fit["k"]) * line_x + float(fit["b"])
            px, py = _project_points(line_x, line_y, bounds, plot)
            pts = np.stack([px, py], axis=1).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], False, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    _draw_legend(img, [("sampled", (138, 138, 138)), ("height_candidate", (43, 142, 242)), ("vertical_support", (167, 121, 78)), ("representative", (238, 58, 178)), ("fit_inlier", (44, 160, 44)), ("fit", (0, 0, 0))])
    cv2.imwrite(str(path), img)


def _plot_side_or_height(path: Path, data: Dict[str, Any], x_key: str, y_key: str, xlabel: str, ylabel: str, title: str) -> None:
    import cv2

    img, plot = _blank_plot(title, xlabel, ylabel)
    x_min, x_max = _bounds(data[x_key])
    y_min, y_max = _bounds(data[y_key])
    bounds = (x_min, x_max, y_min, y_max)
    _draw_grid(img, plot, bounds)
    _draw_scatter_layers(img, plot, bounds, data, x_key, y_key)
    if y_key == "Z_robot":
        params = data["_params"]
        for value, color, label in ((0.0, (0, 0, 0), "Z=0 ground"), (float(params["table_height_m"]), (40, 40, 200), "Z=table_top"), (float(params["camera_height_m"]), (160, 90, 140), "Z=camera_height")):
            _px, py = _project_points(np.asarray([x_min, x_max], dtype=np.float32), np.asarray([value, value], dtype=np.float32), bounds, plot)
            cv2.line(img, (int(_px[0]), int(py[0])), (int(_px[1]), int(py[1])), color, 2, lineType=cv2.LINE_AA)
            cv2.putText(img, label, (plot[0] + 8, int(py[0]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)
    _draw_legend(img, [("sampled", (138, 138, 138)), ("height_candidate", (43, 142, 242)), ("vertical_support", (167, 121, 78)), ("representative", (238, 58, 178)), ("fit_inlier", (44, 160, 44))])
    cv2.imwrite(str(path), img)


def _plot_hist(path: Path, data: Dict[str, Any]) -> None:
    import cv2

    img, plot = _blank_plot(f"Frame {int(data['frame_seq'][0])} Z_robot histogram", "Z_robot (m)", "count", height=650)
    all_values = data["Z_robot"][np.isfinite(data["Z_robot"])]
    z_min, z_max = _bounds(all_values)
    max_count = 1
    hist_items = []
    for key, label, color in (
        ("is_sampled", "sampled", (138, 138, 138)),
        ("is_height_candidate", "height_candidate", (43, 142, 242)),
        ("is_vertical_support", "vertical_support", (167, 121, 78)),
        ("is_representative", "representative", (238, 58, 178)),
    ):
        mask = np.asarray(data[key], dtype=bool) & np.isfinite(data["Z_robot"])
        values = data["Z_robot"][mask]
        if values.size:
            counts, edges = np.histogram(values, bins=60, range=(z_min, z_max))
            max_count = max(max_count, int(np.max(counts)))
            hist_items.append((counts, edges, label, color))
    bounds = (z_min, z_max, 0.0, float(max_count))
    _draw_grid(img, plot, bounds)
    for counts, edges, _label, color in hist_items:
        xs = (edges[:-1] + edges[1:]) * 0.5
        px, py = _project_points(xs.astype(np.float32), counts.astype(np.float32), bounds, plot)
        pts = np.stack([px, py], axis=1).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, color, 2, lineType=cv2.LINE_AA)
    for value, color in ((0.0, (0, 0, 0)), (float(data["_params"]["table_height_m"]), (40, 40, 200))):
        px, py = _project_points(np.asarray([value, value], dtype=np.float32), np.asarray([0.0, float(max_count)], dtype=np.float32), bounds, plot)
        cv2.line(img, (int(px[0]), int(py[0])), (int(px[1]), int(py[1])), color, 1, lineType=cv2.LINE_AA)
    _draw_legend(img, [("sampled", (138, 138, 138)), ("height_candidate", (43, 142, 242)), ("vertical_support", (167, 121, 78)), ("representative", (238, 58, 178))])
    cv2.imwrite(str(path), img)


def _median_profile(data: Dict[str, Any], value_key: str) -> Tuple[np.ndarray, np.ndarray]:
    v_vals = data["v"].astype(np.int32)
    values = np.asarray(data[value_key], dtype=np.float32)
    valid = np.isfinite(values)
    rows = sorted(set(int(v) for v in v_vals[valid].tolist()))
    med_v, med_value = [], []
    for row in rows:
        row_values = values[valid & (v_vals == row)]
        if row_values.size:
            med_v.append(row)
            med_value.append(float(np.median(row_values)))
    return np.asarray(med_v, dtype=np.float32), np.asarray(med_value, dtype=np.float32)


def _local_minima(x: np.ndarray, y: np.ndarray) -> List[Tuple[float, float]]:
    if y.size < 3:
        return []
    smooth = y.copy()
    if y.size >= 5:
        smooth = np.convolve(y, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    mins: List[Tuple[float, float]] = []
    for i in range(1, len(smooth) - 1):
        if not np.isfinite(smooth[i]):
            continue
        if smooth[i] < smooth[i - 1] and smooth[i] < smooth[i + 1]:
            left = np.nanmax(smooth[max(0, i - 8) : i + 1])
            right = np.nanmax(smooth[i : min(len(smooth), i + 9)])
            if min(left, right) - smooth[i] >= 0.015:
                mins.append((float(x[i]), float(y[i])))
    return mins[:8]


def _plot_camera_z_profile(path: Path, data: Dict[str, Any]) -> List[Tuple[float, float]]:
    import cv2

    img, plot = _blank_plot(f"Frame {int(data['frame_seq'][0])} vertical camera-depth profiles", "Z_cam / depth_m (m)", "image v (px)")
    mask = np.asarray(data["is_sampled"], dtype=bool) & np.isfinite(data["Z_cam"])
    z_min, z_max = _bounds(data["Z_cam"][mask])
    v_min, v_max = _bounds(data["v"][mask])
    bounds = (z_min, z_max, v_min, v_max)
    _draw_grid(img, plot, bounds)
    finite_bins = sorted(set(int(v) for v in data["column_bin"][mask].tolist()))
    colors = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189)]
    if finite_bins:
        picks = np.linspace(0, len(finite_bins) - 1, min(5, len(finite_bins))).round().astype(int)
        for j, idx in enumerate(picks):
            xb = finite_bins[int(idx)]
            m = mask & (data["column_bin"] == xb)
            order = np.argsort(data["v"][m])
            px, py = _project_points(data["Z_cam"][m][order], data["v"][m][order], bounds, plot)
            pts = np.stack([px, py], axis=1).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], False, colors[j % len(colors)], 1, lineType=cv2.LINE_AA)
    med_v, med_z = _median_profile(data, "Z_cam")
    minima = _local_minima(med_v, med_z)
    if med_v.size:
        px, py = _project_points(med_z, med_v, bounds, plot)
        pts = np.stack([px, py], axis=1).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, (0, 0, 0), 3, lineType=cv2.LINE_AA)
    for vv, zz in minima:
        px, py = _project_points(np.asarray([zz], dtype=np.float32), np.asarray([vv], dtype=np.float32), bounds, plot)
        cv2.drawMarker(img, (int(px[0]), int(py[0])), (40, 40, 210), cv2.MARKER_TILTED_CROSS, 16, 2, line_type=cv2.LINE_AA)
        cv2.putText(img, f"min v={vv:.0f}", (int(px[0]) + 6, int(py[0]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 210), 1, cv2.LINE_AA)
    _draw_legend(img, [("selected x_bins", (31, 119, 180)), ("ROI median Z_cam", (0, 0, 0)), ("local minima", (40, 40, 210))])
    cv2.imwrite(str(path), img)
    return minima


def _plot_robot_y_profile(path: Path, data: Dict[str, Any]) -> None:
    import cv2

    img, plot = _blank_plot(f"Frame {int(data['frame_seq'][0])} vertical robot-forward profiles", "Y_robot forward (m)", "image v (px)")
    mask_all = np.isfinite(data["Y_robot"])
    y_min, y_max = _bounds(data["Y_robot"][mask_all])
    v_min, v_max = _bounds(data["v"][mask_all])
    bounds = (y_min, y_max, v_min, v_max)
    _draw_grid(img, plot, bounds)
    med_v, med_y = _median_profile(data, "Y_robot")
    if med_v.size:
        px, py = _project_points(med_y, med_v, bounds, plot)
        pts = np.stack([px, py], axis=1).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, (0, 0, 0), 3, lineType=cv2.LINE_AA)
    mask = np.asarray(data["is_height_candidate"], dtype=bool) & np.isfinite(data["Y_robot"])
    finite_bins = sorted(set(int(v) for v in data["column_bin"][mask].tolist()))
    colors = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189)]
    if finite_bins:
        picks = np.linspace(0, len(finite_bins) - 1, min(5, len(finite_bins))).round().astype(int)
        for j, idx in enumerate(picks):
            xb = finite_bins[int(idx)]
            m = mask & (data["column_bin"] == xb)
            order = np.argsort(data["v"][m])
            px, py = _project_points(data["Y_robot"][m][order], data["v"][m][order], bounds, plot)
            pts = np.stack([px, py], axis=1).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], False, colors[j % len(colors)], 1, lineType=cv2.LINE_AA)
    _draw_legend(img, [("selected x_bins", (31, 119, 180)), ("ROI median Y_robot", (0, 0, 0))])
    cv2.imwrite(str(path), img)


def _summary(data: Dict[str, Any], minima: List[Tuple[float, float]]) -> Dict[str, Any]:
    obs = data["_obs"]
    params = data["_params"]
    sampled = np.asarray(data["is_sampled"], dtype=bool) & np.isfinite(data["Z_robot"])
    candidate = np.asarray(data["is_height_candidate"], dtype=bool) & np.isfinite(data["Z_robot"])
    support = np.asarray(data["is_vertical_support"], dtype=bool) & np.isfinite(data["Z_robot"])
    representative = np.asarray(data["is_representative"], dtype=bool) & np.isfinite(data["Z_robot"])
    inlier = np.asarray(data["is_fit_inlier"], dtype=bool) & np.isfinite(data["Z_robot"])
    z_span = data["z_span_m"][np.isfinite(data["z_span_m"])]
    x_vals = data["X_robot"][candidate]
    y_vals = data["Y_robot"][candidate]
    near_threshold = 0.06
    fit = data["_fit"]
    return {
        "frame_seq": int(data["frame_seq"][0]),
        "ROI": list(data["_roi_box"]),
        "roi_meta": _json_ready(data["_roi_meta"]),
        "sampled_count": int(len(data["u"])),
        "valid_sampled_count": int(np.sum(sampled)),
        "height_candidate_count": int(np.sum(candidate)),
        "vertical_support_count": int(np.sum(support)),
        "representative_count": int(data["_reps"].get("count", 0) or 0),
        "fit_inlier_count": int(np.sum(inlier)),
        "reject_reason": obs.get("reject_reason") or obs.get("fast_gate_reason") or obs.get("reason") or "none",
        "control_level": obs.get("control_level") or obs.get("fast_control_level") or "none",
        "robot_z_sampled": _stats(data["Z_robot"][sampled]),
        "robot_z_candidate": _stats(data["Z_robot"][candidate]),
        "robot_z_support": _stats(data["Z_robot"][support]),
        "camera_z_sampled": _stats(data["Z_cam"][sampled]),
        "candidate_ground_like_ratio": _ratio(np.abs(data["Z_robot"][candidate] - 0.0) < near_threshold, int(np.sum(candidate))),
        "candidate_top_like_ratio": _ratio(np.abs(data["Z_robot"][candidate] - float(params["table_height_m"])) < near_threshold, int(np.sum(candidate))),
        "candidate_near_ground_threshold_m": near_threshold,
        "candidate_near_table_top_threshold_m": near_threshold,
        "z_span_statistics": _stats(z_span),
        "x_span_m": float(np.max(x_vals) - np.min(x_vals)) if x_vals.size > 1 else 0.0,
        "y_span_m": float(np.max(y_vals) - np.min(y_vals)) if y_vals.size > 1 else 0.0,
        "fitted_yaw_rad": float(math.atan(float(fit["k"]))) if fit.get("k") is not None else None,
        "fitted_dist_m": float(fit["b"]) - float(obs.get("target_dist_m", 0.5) or 0.5) if fit.get("b") is not None else None,
        "fit_k": fit.get("k"),
        "fit_b": fit.get("b"),
        "fit_residual_mean": fit.get("residual_mean"),
        "fit_residual_p90": fit.get("residual_p90"),
        "camera_z_profile_local_minima": [{"v": vv, "z_cam_m": zz} for vv, zz in minima],
        "fast_obs": {
            key: obs.get(key)
            for key in (
                "edge_found",
                "valid_for_control",
                "plane_found",
                "fast_raw_confidence",
                "fast_raw_yaw_err_rad",
                "fast_raw_dist_err_m",
                "fast_raw_plane_x_span_m",
                "fast_front_face_rep_count",
                "fast_front_face_support_point_count",
                "fast_z_span_m_p50",
                "fast_z_span_m_max",
                "fast_distance_stage",
                "fast_gate_reason",
            )
        },
    }


def _classify_frame(summary: Dict[str, Any]) -> Dict[str, str]:
    rz = summary.get("robot_z_sampled") or {}
    cz = summary.get("robot_z_candidate") or {}
    rep_count = int(summary.get("representative_count") or 0)
    support_count = int(summary.get("vertical_support_count") or 0)
    candidate_count = int(summary.get("height_candidate_count") or 0)
    reason = str(summary.get("reject_reason") or "none")
    top_ratio = float(summary.get("candidate_top_like_ratio") or 0.0)
    ground_ratio = float(summary.get("candidate_ground_like_ratio") or 0.0)
    zspan_p50 = (summary.get("z_span_statistics") or {}).get("p50")
    x_span = float(summary.get("x_span_m") or 0.0)
    plausible = "yes"
    if rz.get("p50") is None or rz.get("min") is None or rz.get("max") is None:
        plausible = "no valid robot-Z samples"
    elif float(rz["p50"]) < -0.25 or float(rz["p50"]) > 1.2:
        plausible = "questionable median robot-Z"
    if candidate_count <= 0:
        dominant = "no height candidates"
    elif ground_ratio > 0.45:
        dominant = "ground-like candidates"
    elif top_ratio > 0.45:
        dominant = "table-top-like candidates"
    elif support_count > 0 and zspan_p50 is not None and float(zspan_p50) >= 0.12:
        dominant = "vertical front-face-like support"
    else:
        dominant = "mixed sparse front-face candidates"
    if rep_count <= 0:
        reps = "no representatives"
    elif x_span < 0.12:
        reps = "representatives cover a narrow patch"
    else:
        reps = "representatives span visible face"
    if reason in {"height_filter_empty", "no_robot_points"} or candidate_count <= 0:
        likely = "height filter issue or transform placed face outside height band"
    elif reason in {"vertical_support_low", "front_face_columns_low"}:
        likely = "vertical support / columns too strict for observed candidates"
    elif reason == "front_face_x_span_low" or x_span < 0.07:
        likely = "x-span / columns too narrow"
    elif top_ratio > 0.45:
        likely = "candidates dominated by table top"
    elif ground_ratio > 0.45:
        likely = "candidates dominated by ground"
    elif candidate_count > 0 and support_count <= 0:
        likely = "true front face weak or not vertically coherent"
    else:
        likely = "accepted or rejected by downstream confidence/gating"
    return {"plausible": plausible, "dominant": dominant, "representatives": reps, "likely_cause": likely}


def _write_report(path: Path, summaries: List[Dict[str, Any]]) -> None:
    lines = [
        "# Point Debug Keyframes",
        "",
        "Standalone point-level export using the current fast table-plane ROI, stride, projection, robot transform, height filter, vertical support selection, and bird-view fit. Detector behavior was not modified.",
        "",
        "| frame | valid/control | reject | sampled | candidates | support | reps | inliers | Z_robot p50 | candidate mix | likely cause |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in summaries:
        cls = _classify_frame(item)
        rz50 = (item.get("robot_z_sampled") or {}).get("p50")
        lines.append(
            f"| {item['frame_seq']} | {item.get('control_level')} | {item.get('reject_reason')} | "
            f"{item.get('sampled_count')} | {item.get('height_candidate_count')} | {item.get('vertical_support_count')} | "
            f"{item.get('representative_count')} | {item.get('fit_inlier_count')} | "
            f"{rz50 if rz50 is not None else 'NA'} | {cls['dominant']} | {cls['likely_cause']} |"
        )
    lines.extend(["", "## Frame Notes", ""])
    special = {
        300: "Special check: full detector looked visually good; inspect whether fast height filtering or vertical support lost the front face.",
        390: "Special check: valid fast result may have few representatives; check span and whether reps are a narrow patch.",
        420: "Special check: valid fast result may have few representatives; check span and whether reps are a narrow patch.",
        450: "Special check: valid fast result may have few representatives; check span and whether reps are a narrow patch.",
        480: "Special check: verify the fitted slope sign and whether an opposite-slope line is being forced.",
        510: "Special check: determine whether the front face is truly not visible.",
        594: "Special check: near-stage support validity and conservatism.",
        744: "Special check: near-stage support validity and conservatism.",
    }
    for item in summaries:
        cls = _classify_frame(item)
        minima = item.get("camera_z_profile_local_minima") or []
        minima_text = ", ".join(f"v={m['v']:.0f}/Z={m['z_cam_m']:.3f}" for m in minima[:4]) if minima else "none detected"
        yaw = item.get("fitted_yaw_rad")
        dist = item.get("fitted_dist_m")
        lines.extend(
            [
                f"### Frame {item['frame_seq']}",
                "",
                f"- Physical plausibility: {cls['plausible']}; sampled Z_robot stats={item.get('robot_z_sampled')}.",
                f"- Candidate interpretation: {cls['dominant']}; top_like={item.get('candidate_top_like_ratio'):.3f}, ground_like={item.get('candidate_ground_like_ratio'):.3f}.",
                f"- Representatives: {cls['representatives']}; reps={item.get('representative_count')}, support={item.get('vertical_support_count')}, x_span_m={item.get('x_span_m'):.3f}.",
                f"- Valid/missing: control={item.get('control_level')}, reject={item.get('reject_reason')}, likely={cls['likely_cause']}.",
                f"- Fit: yaw_rad={yaw if yaw is not None else 'NA'}, dist_m={dist if dist is not None else 'NA'}, residual_mean={item.get('fit_residual_mean')}.",
                f"- Camera vertical Z profile local minima: {minima_text}.",
            ]
        )
        if int(item["frame_seq"]) in special:
            lines.append(f"- {special[int(item['frame_seq'])]}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _process_one(manager: Any, pack: Dict[str, Any], frame_seq: int) -> Dict[str, Any]:
    capture_ts = time.time()
    frames = {
        "rgb": pack.get("rgb"),
        "depth": pack.get("depth"),
        "frame_capture_ts": capture_ts,
        "timestamp_ms": pack.get("timestamp_ms"),
    }
    obs = manager.process_camera_frame(
        frames,
        frame_seq=frame_seq,
        frame_slot={"seq": frame_seq, "ts": capture_ts, "payload": frames},
        local_perception={"has_infer": False, "box_count": 0, "infer_boxes": [], "rgb_shape": getattr(pack.get("rgb"), "shape", None)},
        runtime_status={"stage": "OFFLINE_BAG", "mode": "TABLE_EDGE_PERCEPTION"},
        source_mode="OFFLINE_POINT_DEBUG",
        count_dropped=False,
    )
    obs["bag_timestamp_ms"] = float(pack.get("timestamp_ms", 0.0) or 0.0)
    return obs


def main() -> None:
    args = build_parser().parse_args()
    frames = _parse_frames(args.frames)
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VISION_PARAMS_FILE"] = str(args.config.expanduser().resolve())

    from VISTA.vision_module.backend.table_edge_manager import TableEdgeManager
    from VISTA.vision_module.config.board_config import CONFIG
    from VISTA.vision_module.examples.bag_table_plane import iter_bag_frames

    CONFIG.table_edge.roi_preset = "center_lower"
    CONFIG.table_edge.detector_mode = "fast_plane_only"
    manager = TableEdgeManager(cfg=CONFIG)
    summaries: List[Dict[str, Any]] = []
    target_set = set(frames)
    print(f"[POINT_DEBUG] bag={args.bag} frames={frames} output={output_dir}")
    try:
        for pack in iter_bag_frames(args.bag, 0):
            frame_seq = int(pack["frame"])
            if frame_seq not in target_set:
                if frame_seq > max(target_set):
                    break
                continue
            obs = _process_one(manager, pack, frame_seq)
            data = _build_point_data(manager, pack["depth"], pack.get("rgb"), obs, frame_seq)
            stem = f"{frame_seq:06d}"
            _write_csv(output_dir / f"points_{stem}.csv", data)
            _write_npz(output_dir / f"points_{stem}.npz", data)
            _plot_overlay(output_dir / f"overlay_points_{stem}.png", pack["depth"], pack.get("rgb"), data)
            _plot_topview(output_dir / f"robot_xy_topview_{stem}.png", data)
            _plot_side_or_height(output_dir / f"robot_yz_sideview_{stem}.png", data, "Y_robot", "Z_robot", "Y_robot forward (m)", "Z_robot height (m)", f"Frame {frame_seq} robot YZ side view")
            _plot_side_or_height(output_dir / f"robot_xz_heightview_{stem}.png", data, "X_robot", "Z_robot", "X_robot lateral (m)", "Z_robot height (m)", f"Frame {frame_seq} robot XZ height view")
            _plot_hist(output_dir / f"z_hist_{stem}.png", data)
            minima = _plot_camera_z_profile(output_dir / f"camera_z_profile_{stem}.png", data)
            _plot_robot_y_profile(output_dir / f"robot_y_profile_{stem}.png", data)
            summary = _summary(data, minima)
            (output_dir / f"summary_{stem}.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            summaries.append(summary)
            print(
                "[POINT_DEBUG] "
                f"frame={frame_seq} sampled={summary['sampled_count']} cand={summary['height_candidate_count']} "
                f"support={summary['vertical_support_count']} reps={summary['representative_count']} "
                f"reject={summary['reject_reason']} control={summary['control_level']}"
            )
    finally:
        manager.release_all()
    missing = sorted(target_set - {int(item["frame_seq"]) for item in summaries})
    if missing:
        print(f"[POINT_DEBUG] missing_frames={missing}")
    _write_report(output_dir / "point_debug_report.md", summaries)
    (output_dir / "summary_all.json").write_text(json.dumps(_json_ready(summaries), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[POINT_DEBUG] wrote report={output_dir / 'point_debug_report.md'}")


if __name__ == "__main__":
    main()
