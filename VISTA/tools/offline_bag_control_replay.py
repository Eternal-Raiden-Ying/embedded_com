#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Offline RealSense bag replay through VISTA vision and dry-run control.

This script intentionally reuses the online-ish pieces:
- bag frame reader and preview canvas helpers from examples/bag_table_plane.py
- PredictorManager for YOLO inference/postprocess
- TableEdgeManager for dynamic ROI and table-edge/docking observations
- OrchestratorCore/MotionController and SimpleCarMapper for dry-run control

It never opens a preview window, never connects UART, and never sends motion.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from collections import Counter
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


HERE = Path(__file__).resolve()
VISTA_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[2]
for item in (REPO_ROOT, VISTA_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline bag replay with vision, dynamic ROI, table edge, and dry-run control.")
    parser.add_argument("--bag", type=Path, required=True, help="RealSense .bag path.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth bag frame. Default: 1.")
    parser.add_argument("--preview-stride", type=int, default=10, help="Save one preview every N processed frames. Default: 10.")
    parser.add_argument("--max-frames", type=int, default=0, help="Max bag frames to read. 0 means all.")
    parser.add_argument("--config", type=Path, default=VISTA_ROOT / "configs" / "vision_params.yaml", help="Vision params YAML.")
    return parser


def _json_ready(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if np is not None and isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "omitted": True}
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return str(value)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.fp = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.path.open("w", encoding="utf-8")
        return self

    def write(self, row: Dict[str, Any]) -> None:
        assert self.fp is not None
        self.fp.write(json.dumps(_json_ready(row), ensure_ascii=False, sort_keys=True) + "\n")

    def __exit__(self, exc_type, exc, tb):
        if self.fp is not None:
            self.fp.close()
        self.fp = None
        return False


def _shape(value: Any) -> Optional[List[int]]:
    shape = getattr(value, "shape", None)
    if not isinstance(shape, tuple):
        return None
    return [int(v) for v in shape]


def _to_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _counter(rows: Iterable[Dict[str, Any]], key: str, default: str = "none") -> Dict[str, int]:
    return dict(sorted(Counter(str(row.get(key) or default) for row in rows).items()))


def _percentiles(values: Iterable[Any]) -> Dict[str, Optional[float]]:
    vals = sorted(v for v in (_to_float(v) for v in values) if v is not None)
    if not vals:
        return {"p50": None, "p90": None, "max": None}

    def pick(q: float) -> float:
        if len(vals) == 1:
            return float(vals[0])
        pos = q * float(len(vals) - 1)
        lo = int(pos)
        hi = min(len(vals) - 1, lo + 1)
        frac = pos - float(lo)
        return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)

    return {"p50": pick(0.50), "p90": pick(0.90), "max": float(vals[-1])}


def _range(values: Iterable[Any]) -> Dict[str, Optional[float]]:
    vals = sorted(v for v in (_to_float(v) for v in values) if v is not None)
    return {"min": float(vals[0]), "max": float(vals[-1])} if vals else {"min": None, "max": None}


def _bbox_area_ratio(bbox: Any, shape: Any) -> Optional[float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        return None
    try:
        h, w = float(shape[0]), float(shape[1])
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except Exception:
        return None
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1.0, w * h)


def _bbox_center_norm(bbox: Any, shape: Any) -> Optional[List[float]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        return None
    try:
        h, w = float(shape[0]), float(shape[1])
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except Exception:
        return None
    return [max(0.0, min(1.0, ((x1 + x2) * 0.5) / max(1.0, w))), max(0.0, min(1.0, ((y1 + y2) * 0.5) / max(1.0, h)))]


def _cmd_to_row(frame_id: int, ts: float, cmd: Any, summary: Dict[str, Any]) -> Dict[str, Any]:
    cmd_dict = cmd.to_dict() if hasattr(cmd, "to_dict") else {}
    return {
        "frame_id": int(frame_id),
        "timestamp": float(ts),
        "mode": str(cmd_dict.get("mode", getattr(cmd, "mode", "")) or ""),
        "vx_norm": float(cmd_dict.get("vx_norm", getattr(cmd, "vx_norm", 0.0)) or 0.0),
        "vy_norm": float(cmd_dict.get("vy_norm", getattr(cmd, "vy_norm", 0.0)) or 0.0),
        "wz_norm": float(cmd_dict.get("wz_norm", getattr(cmd, "wz_norm", 0.0)) or 0.0),
        "hold_ms": int(cmd_dict.get("hold_ms", getattr(cmd, "hold_ms", 0)) or 0),
        "brake": bool(cmd_dict.get("brake", getattr(cmd, "brake", False))),
        "reason": str(summary.get("reason") or summary.get("lock_reason") or ""),
        "control_source": str(summary.get("control_source") or ""),
        "summary": summary,
    }


def _car_to_row(frame_id: int, ts: float, car_cmd: Any, velocity: Tuple[float, float, float], reason: str) -> Dict[str, Any]:
    raw_line = str(getattr(car_cmd, "raw_line", "") or "")
    return {
        "frame_id": int(frame_id),
        "timestamp": float(ts),
        "mode": str(getattr(car_cmd, "mode", "") or ""),
        "kind": str(getattr(car_cmd, "kind", "") or ""),
        "vx_norm": float(getattr(car_cmd, "vx_norm", 0.0) or 0.0),
        "vy_norm": float(getattr(car_cmd, "vy_norm", 0.0) or 0.0),
        "wz_norm": float(getattr(car_cmd, "wz_norm", 0.0) or 0.0),
        "hold_ms": int(getattr(car_cmd, "hold_ms", 0) or 0),
        "brake": bool(getattr(car_cmd, "brake", False)),
        "vx_mps": float(velocity[0]),
        "vy_mps": float(velocity[1]),
        "wz_radps": float(velocity[2]),
        "raw": "STOP" if str(getattr(car_cmd, "kind", "")) in {"stop", "brake"} else f"V {velocity[0]:.3f} {velocity[1]:.3f} {velocity[2]:.3f}",
        "legacy_raw": raw_line.rstrip("\r\n"),
        "reason": str(reason or ""),
        "dry_run": True,
    }


def _build_local_perception(predictor: Any, frames: Dict[str, Any], frame_seq: int) -> Dict[str, Any]:
    import time as _time
    from VISTA.vision_module.utils.table_roi import build_table_roi, find_table_bbox, table_detection_debug

    rgb = frames.get("rgb")
    rgb_shape = _shape(rgb)
    has_infer = bool(rgb is not None and predictor.is_ready())
    boxes: Any = []
    masks: Any = []
    infer_ms = None
    if has_infer:
        t0 = time.perf_counter()
        boxes, masks = predictor.predict_frame(rgb)
        infer_ms = (time.perf_counter() - t0) * 1000.0

    payload = predictor._build_local_perception_payload(rgb_shape, boxes, masks, has_infer)
    payload["obs_ts"] = _time.time()
    payload["frame_seq"] = int(frame_seq)
    payload["age_ms"] = 0.0
    payload["yolo_infer_ms"] = infer_ms
    payload["yolo_has_infer"] = bool(has_infer)
    payload["yolo_frame_seq"] = int(frame_seq)
    for key in (
        "rgb_native_shape",
        "rgb_crop_rect",
        "rgb_output_shape_config",
        "rgb_output_shape_actual",
        "depth_shape_actual",
        "rgb_config_source",
    ):
        if key in frames:
            payload[key] = frames.get(key)

    boxes_list = list(payload.get("infer_boxes") or [])
    roi_input = {"infer_boxes": boxes_list, "rgb_shape": rgb_shape}
    model_cfg = getattr(predictor.cfg, "model", None)
    yolo26_enabled = bool(getattr(model_cfg, "enable_yolo26", False))
    yolo_table_search_enabled = bool(getattr(model_cfg, "enable_yolo_table_search", False))
    detected_table_bbox = find_table_bbox(roi_input)
    table_bbox_detected = detected_table_bbox is not None
    table_bbox_used_for_search = bool(yolo_table_search_enabled and table_bbox_detected)
    if table_bbox_used_for_search:
        roi_input["table_bbox"] = detected_table_bbox
        table_source = "yolo_table_bbox"
    else:
        table_source = "yolo_table_search_disabled" if table_bbox_detected else "yolo_unavailable"
        mock_bbox = predictor._mock_table_bbox(rgb_shape)
        roi_input = {"infer_boxes": boxes_list, "rgb_shape": rgb_shape}
        if mock_bbox is not None:
            roi_input["mock_table_bbox"] = mock_bbox
            table_source = "mock_table_bbox"
    roi_meta = build_table_roi(roi_input, rgb_shape, None)
    payload.update(
        {
            "table_bbox": roi_meta.get("table_bbox"),
            "detected_table_bbox": detected_table_bbox,
            "table_quadrant": roi_meta.get("table_quadrant"),
            "rgb_search_roi": roi_meta.get("rgb_search_roi"),
            "table_roi_source": table_source,
            "yolo26_enabled": yolo26_enabled,
            "yolo_infer_running": bool(has_infer),
            "yolo_table_search_enabled": yolo_table_search_enabled,
            "table_bbox_detected": table_bbox_detected,
            "table_bbox_used_for_search": table_bbox_used_for_search,
            "table_direction_hint": table_detection_debug(payload, rgb_shape).get("direction"),
        }
    )
    return payload


def _trace_row(frame_id: int, ts: float, rgb: Any, depth: Any, local: Dict[str, Any], obs: Dict[str, Any], control: Dict[str, Any]) -> Dict[str, Any]:
    bbox = obs.get("yolo_table_bbox") or local.get("table_bbox") or local.get("detected_table_bbox")
    rgb_shape = _shape(rgb) or local.get("rgb_shape")
    bbox_found = bool(obs.get("table_bbox_found", bool(bbox)))
    return {
        "frame_id": int(frame_id),
        "timestamp": float(ts),
        "rgb_shape": _shape(rgb),
        "depth_shape": _shape(depth) or obs.get("depth_shape"),
        "table_bbox_found": bbox_found,
        "table_bbox_conf": obs.get("yolo_table_conf"),
        "table_bbox_conf_raw": obs.get("table_bbox_conf_raw", obs.get("yolo_table_conf")),
        "table_bbox_conf_used_for_gate": bool(obs.get("table_bbox_conf_used_for_gate", False)),
        "table_bbox_xyxy": bbox,
        "table_bbox_area_ratio": obs.get("table_bbox_area_ratio", obs.get("yolo_bbox_area_norm", obs.get("yolo_bbox_area_ratio", _bbox_area_ratio(bbox, rgb_shape)))),
        "table_bbox_center_norm": _bbox_center_norm(bbox, rgb_shape),
        "dynamic_roi": obs.get("depth_edge_roi") or obs.get("table_edge_roi") or obs.get("edge_roi"),
        "roi_source": obs.get("roi_source"),
        "roi_phase": obs.get("roi_phase"),
        "roi_anchor": obs.get("yolo_table_roi_anchor"),
        "mapped_depth_center": obs.get("mapped_depth_center"),
        "edge_found": bool(obs.get("edge_found")),
        "edge_valid": bool(obs.get("edge_valid")),
        "edge_yaw_err_rad": obs.get("yaw_err_rad", obs.get("yaw_err")),
        "edge_dist_m": obs.get("dist_err_m", obs.get("dist_err")),
        "edge_reject_reason": obs.get("reject_reason") or obs.get("fast_gate_reject_reason") or obs.get("reason"),
        "edge_stable_count": obs.get("yolo_table_edge_stable_count"),
        "yolo_table_control_valid": bool(obs.get("yolo_table_control_valid")),
        "yolo_valid_reason": obs.get("yolo_valid_reason"),
        "yolo_invalid_reason": obs.get("yolo_invalid_reason"),
        "docking_enabled_by_yolo": bool(control.get("docking_enabled_by_yolo", obs.get("docking_enabled_by_yolo", False))),
        "docking_allowed_by_yolo_area": bool(control.get("docking_allowed_by_yolo_area")),
        "docking_blocked_by_yolo_area": bool(control.get("docking_blocked_by_yolo_area")),
        "edge_control_allowed": bool(control.get("edge_control_allowed", obs.get("edge_control_allowed", False))),
        "edge_control_block_reason": control.get("edge_control_block_reason", obs.get("edge_control_block_reason")),
        "allow_rotate": bool(control.get("allow_rotate")),
        "rotate_block_reason": control.get("rotate_block_reason"),
        "yolo_edge_yaw_conflict": bool(control.get("yolo_edge_yaw_conflict")),
        "state": control.get("state"),
        "next_state": control.get("next_state"),
        "state_transition_reason": control.get("state_transition_reason"),
        "control_source": control.get("control_source"),
        "vx_cmd": control.get("vx_norm", control.get("final_vx")),
        "vy_cmd": control.get("vy_norm", control.get("final_vy")),
        "wz_cmd": control.get("wz_norm", control.get("final_wz")),
        "cmd_raw": control.get("cmd_raw"),
        "zero_cmd_reason": control.get("zero_cmd_reason"),
        "forward_block_reason": control.get("forward_block_reason"),
        "stale_level": control.get("stale_level"),
        "stale_source": control.get("stale_source"),
        "vision_process_ms": obs.get("vision_process_ms", obs.get("process_ms")),
    }


def _make_preview_canvas(sink: Any, rgb: Any, depth: Any, obs: Dict[str, Any], local: Dict[str, Any], control: Dict[str, Any], frame_id: int):
    import numpy as np
    from VISTA.vision_module.backend.preview.base import PreviewFrame, PreviewOverlay

    table_edge = dict(obs or {})
    if not bool(table_edge.get("table_bbox_found", table_edge.get("yolo_table_bbox") is not None)):
        table_edge.update(
            {
                "edge_found": False,
                "edge_valid": False,
                "valid_for_control": False,
                "usable_for_approach": False,
                "usable_for_alignment": False,
                "usable_for_stop": False,
                "edge_k": None,
                "edge_b": None,
                "depth_edge_roi": None,
                "table_edge_roi": None,
                "edge_roi": None,
                "plane_roi": None,
                "roi_source": "disabled_no_table_bbox",
                "roi_phase": "disabled_no_table_bbox",
                "yaw_err_rad": None,
                "dist_err_m": None,
                "reason": table_edge.get("reason") or "table_bbox_unavailable",
                "reject_reason": table_edge.get("reject_reason") or "table_bbox_unavailable",
            }
        )
    sink.debug_points_enabled = True
    metadata = {
        "preview_layout": "rgb_depth_edge",
        "runtime_status": {"stage": "OFFLINE_BAG", "mode": "FIND_EDGE", "state": control.get("state")},
        "local_perception": local,
        "table_edge_obs": table_edge,
        "target_obs": {},
        "source_cameras": ["rgb", "depth"] if rgb is not None else ["depth"],
        "show_age_ms": True,
        "show_yolo_boxes": True,
        "frame_age_s": 0.0,
        "offline_control": control,
    }
    lines = [
        f"frame={frame_id}",
        f"state={control.get('state')} source={control.get('control_source')}",
        f"vx={float(control.get('vx_norm') or 0.0):.3f} wz={float(control.get('wz_norm') or 0.0):.3f}",
        f"allow_rotate={int(bool(control.get('allow_rotate')))} block={control.get('rotate_block_reason') or ''}",
        f"roi={table_edge.get('roi_source')} edge={int(bool(table_edge.get('edge_found')))} valid={int(bool(table_edge.get('edge_valid')))}",
        f"reject={table_edge.get('reject_reason') or table_edge.get('fast_gate_reject_reason') or table_edge.get('reason') or ''}",
    ]
    frame = PreviewFrame(time.time(), {"rgb": rgb, "depth": depth}, "OFFLINE_BAG", "FIND_EDGE", PreviewOverlay("Offline Bag Control Replay", lines, metadata=metadata))
    panel_w = max(320, int(getattr(sink, "canvas_w", 1280)) // 2)
    panel_h = max(220, int(getattr(sink, "canvas_h", 720)) // 2)
    panel_size = (panel_w, panel_h)
    rgb_panel = sink._make_rgb_panel(rgb, metadata, panel_size)
    depth_panel = sink._make_depth_panel(depth, table_edge, panel_size)
    edge_panel = sink._make_edge_panel(depth, table_edge, panel_size)
    info_panel = sink._make_info_panel(frame, metadata, table_edge, {}, panel_size)
    canvas = np.vstack([np.hstack([rgb_panel, depth_panel]), np.hstack([edge_panel, info_panel])])
    if canvas.shape[:2] != (sink.canvas_h, sink.canvas_w):
        try:
            import cv2

            canvas = cv2.resize(canvas, (sink.canvas_w, sink.canvas_h), interpolation=cv2.INTER_AREA)
        except Exception:
            pass
    return canvas


def _configure_offline_table_edge(edge_processor: Any, vision_cfg: Any) -> Dict[str, Any]:
    modes = getattr(vision_cfg, "modes", None)
    find_edge = getattr(modes, "FIND_EDGE", None) if modes is not None else None
    table_edge = getattr(find_edge, "table_edge", None) if find_edge is not None else None
    keys = (
        "detector_mode",
        "update_hz",
        "light_stride",
        "fast_plane_stride",
        "require_yolo_confirm",
        "static_roi_enabled",
        "camera_pitch_deg",
        "camera_height_m",
        "camera_roll_deg",
        "camera_yaw_deg",
        "table_height_m",
        "front_face_z_min_m",
        "front_face_z_max_m",
        "min_vertical_z_span_m",
        "min_vertical_support_points",
        "x_bin_width_m",
        "y_cluster_bin_m",
        "min_front_face_columns",
        "min_front_face_x_span_m",
        "front_cluster_gap_m",
        "max_yaw_abs_rad",
        "enable_yolo_in_plane_only",
        "yolo_table_min_conf",
    )
    payload: Dict[str, Any] = {}
    if table_edge is not None:
        for key in keys:
            if hasattr(table_edge, key):
                payload[key] = getattr(table_edge, key)
    payload["detector_mode"] = "fast_plane_only"
    payload["require_yolo_confirm"] = True
    edge_processor.configure(payload)
    return payload


def _write_contact_sheet(image_paths: Sequence[Path], output_path: Path, *, thumb_w: int = 320, cols: int = 4) -> bool:
    if not image_paths:
        return False
    try:
        import cv2
        import numpy as np

        thumbs = []
        for path in image_paths:
            img = cv2.imread(str(path))
            if img is None:
                continue
            scale = float(thumb_w) / max(1.0, float(img.shape[1]))
            thumb_h = max(1, int(round(float(img.shape[0]) * scale)))
            thumbs.append(cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))
        if not thumbs:
            return False
        thumb_h = max(img.shape[0] for img in thumbs)
        padded = []
        for img in thumbs:
            if img.shape[0] < thumb_h:
                pad = np.zeros((thumb_h - img.shape[0], img.shape[1], 3), dtype=img.dtype)
                img = np.vstack([img, pad])
            padded.append(img)
        rows = []
        for start in range(0, len(padded), cols):
            row = list(padded[start : start + cols])
            while len(row) < cols:
                row.append(np.zeros((thumb_h, thumb_w, 3), dtype=padded[0].dtype))
            rows.append(np.hstack(row))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(output_path), np.vstack(rows)))
    except Exception:
        return False


SUMMARY_COLUMNS = (
    "frame_id",
    "timestamp",
    "table_bbox_found",
    "table_bbox_conf",
    "table_bbox_conf_raw",
    "table_bbox_conf_used_for_gate",
    "table_bbox_area_ratio",
    "roi_source",
    "roi_phase",
    "dynamic_roi",
    "edge_found",
    "edge_valid",
    "edge_yaw_err_rad",
    "edge_dist_m",
    "edge_reject_reason",
    "edge_stable_count",
    "yolo_table_control_valid",
    "yolo_valid_reason",
    "yolo_invalid_reason",
    "docking_enabled_by_yolo",
    "docking_allowed_by_yolo_area",
    "docking_blocked_by_yolo_area",
    "edge_control_allowed",
    "edge_control_block_reason",
    "allow_rotate",
    "rotate_block_reason",
    "yolo_edge_yaw_conflict",
    "state",
    "next_state",
    "state_transition_reason",
    "control_source",
    "vx_cmd",
    "vy_cmd",
    "wz_cmd",
    "zero_cmd_reason",
    "forward_block_reason",
    "stale_level",
    "stale_source",
    "vision_process_ms",
)


def _write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_json_ready(row.get(key)), ensure_ascii=False) if isinstance(row.get(key), (list, dict)) else row.get(key) for key in SUMMARY_COLUMNS})


def _summary_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    yolo_valid = [r for r in rows if bool(r.get("table_bbox_found") or r.get("yolo_table_control_valid"))]
    yolo_far = [r for r in yolo_valid if (_to_float(r.get("table_bbox_area_ratio")) or 0.0) < 0.40]
    no_forward = [r for r in yolo_valid if abs(float(r.get("vx_cmd") or 0.0)) <= 1e-9]
    far_with_wz = [r for r in yolo_far if abs(float(r.get("wz_cmd") or 0.0)) > 1e-9]
    far_bad_state = [r for r in yolo_far if str(r.get("state") or "") in {"COARSE_ALIGN", "SEARCH_TABLE"}]
    cmd_stop = [r for r in rows if str(r.get("state") or "") in {"STOP", "IDLE", "DONE", "ERROR_RECOVERY"}]
    cmd_v0 = [r for r in rows if str(r.get("state") or "") not in {"STOP", "IDLE", "DONE", "ERROR_RECOVERY"} and abs(float(r.get("vx_cmd") or 0.0)) <= 1e-9 and abs(float(r.get("vy_cmd") or 0.0)) <= 1e-9 and abs(float(r.get("wz_cmd") or 0.0)) <= 1e-9]
    cmd_v = [r for r in rows if abs(float(r.get("vx_cmd") or 0.0)) > 1e-9 or abs(float(r.get("vy_cmd") or 0.0)) > 1e-9 or abs(float(r.get("wz_cmd") or 0.0)) > 1e-9]
    roi_x_vals: List[float] = []
    roi_y_vals: List[float] = []
    for row in rows:
        roi = row.get("dynamic_roi")
        if isinstance(roi, (list, tuple)) and len(roi) >= 4:
            try:
                roi_x_vals.extend([float(roi[0]), float(roi[2])])
                roi_y_vals.extend([float(roi[1]), float(roi[3])])
            except Exception:
                pass
    return {
        "frames": total,
        "yolo_valid_frames": len(yolo_valid),
        "table_bbox_found_but_yolo_control_invalid_frames": int(sum(1 for r in rows if bool(r.get("table_bbox_found")) and not bool(r.get("yolo_table_control_valid")))),
        "yolo_valid_no_forward_frames": len(no_forward),
        "table_bbox_found_zero_cmd_frames": int(sum(1 for r in rows if bool(r.get("table_bbox_found")) and abs(float(r.get("vx_cmd") or 0.0)) <= 1e-9 and abs(float(r.get("wz_cmd") or 0.0)) <= 1e-9)),
        "yolo_valid_area_lt_0_40_with_wz_frames": len(far_with_wz),
        "yolo_valid_area_lt_0_40_in_coarse_or_search_frames": len(far_bad_state),
        "table_bbox_missing_but_edge_control_allowed_frames": int(sum(1 for r in rows if not bool(r.get("table_bbox_found")) and bool(r.get("edge_control_allowed")))),
        "coarse_align_stop_frames": int(sum(1 for r in rows if str(r.get("state") or "") == "COARSE_ALIGN" and abs(float(r.get("vx_cmd") or 0.0)) <= 1e-9 and abs(float(r.get("vy_cmd") or 0.0)) <= 1e-9 and abs(float(r.get("wz_cmd") or 0.0)) <= 1e-9)),
        "allow_rotate_true_frames": int(sum(1 for r in rows if bool(r.get("allow_rotate")))),
        "rotate_block_reason": _counter(rows, "rotate_block_reason", "none"),
        "control_source": _counter(rows, "control_source", "none"),
        "forward_block_reason": _counter(rows, "forward_block_reason", "none"),
        "cmd_distribution": {"STOP": len(cmd_stop), "V0": len(cmd_v0), "V_nonzero": len(cmd_v)},
        "yolo_edge_yaw_conflict_frames": int(sum(1 for r in rows if bool(r.get("yolo_edge_yaw_conflict")))),
        "dynamic_roi_x_range": _range(roi_x_vals),
        "dynamic_roi_y_range": _range(roi_y_vals),
        "edge_reject_reason": _counter(rows, "edge_reject_reason", "none"),
        "vision_process_ms": _percentiles(r.get("vision_process_ms") for r in rows),
    }


def _write_summary_md(path: Path, stats: Dict[str, Any], out_dir: Path) -> None:
    lines = [
        "# Offline Bag Control Replay Summary",
        "",
        f"- output_dir: `{out_dir}`",
        f"- frames: {stats.get('frames')}",
        f"- yolo_valid_frames: {stats.get('yolo_valid_frames')}",
        f"- table_bbox_found_but_yolo_control_invalid_frames: {stats.get('table_bbox_found_but_yolo_control_invalid_frames')}",
        f"- yolo_valid_no_forward_frames: {stats.get('yolo_valid_no_forward_frames')}",
        f"- table_bbox_found_zero_cmd_frames: {stats.get('table_bbox_found_zero_cmd_frames')}",
        f"- yolo_valid_area_lt_0_40_with_wz_frames: {stats.get('yolo_valid_area_lt_0_40_with_wz_frames')}",
        f"- yolo_valid_area_lt_0_40_in_COARSE_ALIGN_or_SEARCH_TABLE_frames: {stats.get('yolo_valid_area_lt_0_40_in_coarse_or_search_frames')}",
        f"- table_bbox_missing_but_edge_control_allowed_frames: {stats.get('table_bbox_missing_but_edge_control_allowed_frames')}",
        f"- coarse_align_stop_frames: {stats.get('coarse_align_stop_frames')}",
        f"- allow_rotate_true_frames: {stats.get('allow_rotate_true_frames')}",
        f"- yolo_edge_yaw_conflict_frames: {stats.get('yolo_edge_yaw_conflict_frames')}",
        f"- cmd_distribution: `{json.dumps(stats.get('cmd_distribution'), ensure_ascii=False, sort_keys=True)}`",
        f"- dynamic_roi_x_range: `{json.dumps(stats.get('dynamic_roi_x_range'), ensure_ascii=False, sort_keys=True)}`",
        f"- dynamic_roi_y_range: `{json.dumps(stats.get('dynamic_roi_y_range'), ensure_ascii=False, sort_keys=True)}`",
        f"- vision_process_ms: `{json.dumps(stats.get('vision_process_ms'), ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Distributions",
        "",
        f"- rotate_block_reason: `{json.dumps(stats.get('rotate_block_reason'), ensure_ascii=False, sort_keys=True)}`",
        f"- control_source: `{json.dumps(stats.get('control_source'), ensure_ascii=False, sort_keys=True)}`",
        f"- forward_block_reason: `{json.dumps(stats.get('forward_block_reason'), ensure_ascii=False, sort_keys=True)}`",
        f"- edge_reject_reason: `{json.dumps(stats.get('edge_reject_reason'), ensure_ascii=False, sort_keys=True)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _configure_offline_core(core: Any) -> None:
    # Offline replay should advance on observations, not wall-clock warmup/dwell.
    core.cfg.table_approach_warmup_s = 0.0
    core.cfg.table_approach_warmup_min_fresh_obs = 0
    core.cfg.coarse_align_min_dwell_s = 0.0
    core.cfg.controlled_approach_min_dwell_s = 0.0
    core.cfg.approach_min_dwell_s = 0.0
    core.cfg.search_table_timeout_s = 1e9
    core.cfg.approach_timeout_s = 1e9
    core.cfg.req_resend_period_s = 1e9


def run(args: argparse.Namespace) -> Dict[str, Any]:
    os.environ["VISION_PARAMS_FILE"] = str(args.config.expanduser().resolve())

    from VISTA.vision_module.config.board_config import CONFIG as VISION_CONFIG
    from VISTA.vision_module.backend.predictor_manager import PredictorManager
    from VISTA.vision_module.backend.table_edge_manager import TableEdgeManager
    from VISTA.vision_module.backend.preview.opencv_sink import OpenCVPreviewSink
    from VISTA.vision_module.examples.bag_table_plane import _compact_obs, iter_bag_frames
    from orchestrator.orchestrator_service.bridge.simple_car_protocol import SimpleCarMapper
    from orchestrator.orchestrator_service.config.board_config import CONFIG as ORCH_CONFIG
    from orchestrator.orchestrator_service.control.motion_adapter import Stm32MotionAdapter
    from orchestrator.orchestrator_service.ipc.protocol import TableEdgeObs, TaskCmd
    from orchestrator.orchestrator_service.runtime.state_machine import OrchestratorCore

    out_dir = args.out.expanduser().resolve()
    preview_dir = out_dir / "preview_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    vision_cfg = copy.deepcopy(VISION_CONFIG)
    orch_cfg = copy.deepcopy(ORCH_CONFIG)
    predictor = PredictorManager(vision_cfg)
    model_loaded = predictor.ensure_model(str(vision_cfg.model.active_model))
    predictor.set_inference_enabled(True)
    edge_processor = TableEdgeManager(cfg=vision_cfg)
    table_edge_config = _configure_offline_table_edge(edge_processor, vision_cfg)
    core = OrchestratorCore(orch_cfg.control, orch_cfg.car, orch_cfg.docking)
    _configure_offline_core(core)
    core.handle_task_cmd(TaskCmd(ts=time.time(), intent="FIND", confidence=1.0, target="offline_table", cmd_id="offline_cmd", session_id="offline_bag", epoch=1, source="offline"))
    mapper = SimpleCarMapper(orch_cfg.car)
    motion_adapter = Stm32MotionAdapter(
        uart=None,
        vx_scale=float(orch_cfg.car.vx_mps_per_norm),
        vy_scale=float(orch_cfg.car.vy_mps_per_norm),
        wz_scale=float(orch_cfg.car.wz_radps_per_norm),
    )
    preview_sink = OpenCVPreviewSink("Offline Bag Control Replay")
    preview_paths: List[Path] = []
    rows: List[Dict[str, Any]] = []
    frames_total = 0
    processed_total = 0
    previous_selected_ms: Optional[float] = None
    last_transition_key = ""

    meta = {
        "bag": str(args.bag),
        "out": str(out_dir),
        "config": str(args.config),
        "stride": int(args.stride),
        "preview_stride": int(args.preview_stride),
        "max_frames": int(args.max_frames),
        "dry_run": True,
        "uart_mocked": True,
        "opencv_window": False,
        "vision_model": str(vision_cfg.model.active_model),
        "model_loaded_or_already_ready": bool(model_loaded or predictor.is_ready()),
        "table_edge_config": table_edge_config,
        "predictor_snapshot": predictor.snapshot(),
        "loaded_vision_config_files": list(getattr(vision_cfg.runtime, "loaded_config_files", []) or []),
        "loaded_orchestrator_config_files": list(getattr(orch_cfg.runtime, "loaded_config_files", []) or []),
    }
    _write_json(out_dir / "meta.json", meta)

    try:
        with JsonlWriter(out_dir / "frame_trace.jsonl") as frame_w, JsonlWriter(out_dir / "vision_obs.jsonl") as vision_w, JsonlWriter(out_dir / "table_edge_obs.jsonl") as edge_w, JsonlWriter(out_dir / "yolo_table_trace.jsonl") as yolo_w, JsonlWriter(out_dir / "dynamic_roi_trace.jsonl") as roi_w, JsonlWriter(out_dir / "control_trace.jsonl") as ctrl_w, JsonlWriter(out_dir / "cmd_vel_dryrun.jsonl") as cmd_w, JsonlWriter(out_dir / "car_cmd_dryrun.jsonl") as car_w:
            for pack in iter_bag_frames(args.bag, int(args.max_frames)):
                frame_seq = int(pack["frame"])
                frames_total += 1
                if int(args.stride) > 1 and frame_seq % int(args.stride) != 0:
                    continue
                processed_total += 1
                bag_ts_ms = float(pack.get("timestamp_ms", 0.0) or 0.0)
                bag_update_interval_ms = None if previous_selected_ms is None else max(0.0, bag_ts_ms - float(previous_selected_ms))
                previous_selected_ms = bag_ts_ms
                capture_ts = time.time()
                frames = {
                    "rgb": pack.get("rgb"),
                    "depth": pack.get("depth"),
                    "depth_intrinsics": pack.get("depth_intrinsics"),
                    "frame_capture_ts": capture_ts,
                    "timestamp_ms": bag_ts_ms,
                }
                local = _build_local_perception(predictor, frames, frame_seq)
                obs = edge_processor.process_camera_frame(
                    frames,
                    frame_seq=frame_seq,
                    frame_slot={"seq": frame_seq, "ts": capture_ts, "payload": frames},
                    local_perception=local,
                    runtime_status={"stage": "OFFLINE_BAG", "mode": "TABLE_EDGE_PERCEPTION"},
                    source_mode="OFFLINE_BAG",
                    count_dropped=False,
                )
                obs["source"] = "offline_bag_control_replay"
                obs["bag_timestamp_ms"] = bag_ts_ms
                if bag_update_interval_ms is not None:
                    obs["bag_update_interval_ms"] = bag_update_interval_ms
                    obs["update_interval_ms"] = bag_update_interval_ms
                    obs["edge_update_interval_ms"] = bag_update_interval_ms
                obs_clean = _compact_obs(obs)

                table_obs = TableEdgeObs.from_dict(obs)
                core.handle_table_obs(table_obs)
                prev_state = core.ctx.state.value
                decision = core.tick()
                state_block = core.export_state_block()
                summary = dict(decision.control_summary or {})
                summary["state"] = str(summary.get("state") or prev_state)
                summary["next_state"] = core.ctx.state.value
                summary["state_transition_reason"] = ""
                if core.last_transition_snapshot:
                    snap = dict(core.last_transition_snapshot)
                    transition_key = "|".join(
                        [
                            str(snap.get("previous_state") or ""),
                            str(snap.get("new_state") or ""),
                            str(snap.get("reason") or ""),
                            str(snap.get("stable_ms") or ""),
                            str(snap.get("lost_ms") or ""),
                        ]
                    )
                    if transition_key != last_transition_key and snap.get("new_state") == core.ctx.state.value:
                        summary["state_transition_reason"] = str(snap.get("reason") or "")
                        last_transition_key = transition_key
                cmd_row = _cmd_to_row(frame_seq, capture_ts, decision.cmd, summary)
                car_cmd = mapper.from_cmd_vel(decision.cmd, cx_norm_abs=decision.cx_norm_abs, distance_ratio=decision.distance_ratio)
                velocity = motion_adapter.cmd_vel_to_velocity(decision.cmd)
                cmd_row.update({
                    "vx_mps": float(velocity[0]),
                    "vy_mps": float(velocity[1]),
                    "wz_radps": float(velocity[2]),
                    "vx_mps_per_norm": float(orch_cfg.car.vx_mps_per_norm),
                    "vy_mps_per_norm": float(orch_cfg.car.vy_mps_per_norm),
                    "wz_radps_per_norm": float(orch_cfg.car.wz_radps_per_norm),
                })
                reason = str(summary.get("reason") or summary.get("lock_reason") or getattr(car_cmd, "kind", "") or "")
                car_row = _car_to_row(frame_seq, capture_ts, car_cmd, velocity, reason)
                summary["cmd_raw"] = car_row["raw"]
                if abs(float(cmd_row["vx_norm"])) <= 1e-9 and abs(float(cmd_row["vy_norm"])) <= 1e-9 and abs(float(cmd_row["wz_norm"])) <= 1e-9:
                    summary["zero_cmd_reason"] = str(summary.get("forward_block_reason") or summary.get("reason") or "zero_cmd")
                summary.update({
                    "vx_norm": cmd_row["vx_norm"],
                    "vy_norm": cmd_row["vy_norm"],
                    "wz_norm": cmd_row["wz_norm"],
                })

                trace = _trace_row(frame_seq, capture_ts, pack.get("rgb"), pack.get("depth"), local, obs_clean, summary)
                rows.append(trace)
                frame_w.write(trace)
                vision_w.write({"frame_id": frame_seq, "timestamp": capture_ts, "local_perception": local, "table_edge_obs": obs_clean})
                edge_w.write(obs_clean)
                yolo_w.write({
                    "frame_id": frame_seq,
                    "timestamp": capture_ts,
                    "table_bbox_found": trace["table_bbox_found"],
                    "table_bbox_conf": trace["table_bbox_conf"],
                    "table_bbox_conf_raw": trace["table_bbox_conf_raw"],
                    "table_bbox_conf_used_for_gate": trace["table_bbox_conf_used_for_gate"],
                    "table_bbox_xyxy": trace["table_bbox_xyxy"],
                    "table_bbox_area_ratio": trace["table_bbox_area_ratio"],
                    "table_bbox_center_norm": trace["table_bbox_center_norm"],
                    "infer_box_count": int(local.get("box_count") or 0),
                    "yolo_infer_ms": local.get("yolo_infer_ms"),
                    "table_roi_source": local.get("table_roi_source"),
                    "yolo_table_control_valid": trace["yolo_table_control_valid"],
                    "yolo_valid_reason": trace["yolo_valid_reason"],
                    "yolo_invalid_reason": trace["yolo_invalid_reason"],
                })
                roi_w.write({
                    "frame_id": frame_seq,
                    "timestamp": capture_ts,
                    "dynamic_roi": trace["dynamic_roi"],
                    "roi_source": trace["roi_source"],
                    "roi_phase": trace["roi_phase"],
                    "roi_anchor": trace["roi_anchor"],
                    "mapped_depth_center": trace["mapped_depth_center"],
                    "table_bbox_xyxy": trace["table_bbox_xyxy"],
                    "yolo_table_roi_valid": obs_clean.get("yolo_table_roi_valid"),
                })
                ctrl_w.write({**summary, "frame_id": frame_seq, "timestamp": capture_ts, "state_block": state_block})
                cmd_w.write(cmd_row)
                car_w.write(car_row)

                if int(args.preview_stride) > 0 and ((processed_total - 1) % int(args.preview_stride)) == 0:
                    try:
                        import cv2

                        canvas = _make_preview_canvas(preview_sink, pack.get("rgb"), pack.get("depth"), obs, local, summary, frame_seq)
                        preview_path = preview_dir / f"preview_{frame_seq:06d}.jpg"
                        if cv2.imwrite(str(preview_path), canvas):
                            preview_paths.append(preview_path)
                    except Exception as exc:
                        obs_clean["preview_save_error"] = str(exc)

                print(
                    "[OFFLINE_BAG_CONTROL] "
                    f"frame={frame_seq} bbox={int(bool(trace['table_bbox_found']))} "
                    f"edge={int(bool(trace['edge_found']))}/{int(bool(trace['edge_valid']))} "
                    f"state={summary.get('state')} next={summary.get('next_state')} "
                    f"src={summary.get('control_source')} vx={cmd_row['vx_norm']:.3f} wz={cmd_row['wz_norm']:.3f}"
                )
    finally:
        edge_processor.release_all()
        predictor.release_all()

    contact_sheet = out_dir / "preview_contact_sheet.jpg"
    _write_contact_sheet(preview_paths, contact_sheet)
    _write_summary_csv(out_dir / "summary.csv", rows)
    stats = _summary_stats(rows)
    _write_summary_md(out_dir / "summary.md", stats, out_dir)
    meta.update(
        {
            "frames_total": int(frames_total),
            "processed_frames": int(processed_total),
            "preview_saved_count": int(len(preview_paths)),
            "contact_sheet": str(contact_sheet) if contact_sheet.exists() else None,
            "summary": stats,
        }
    )
    _write_json(out_dir / "meta.json", meta)
    return meta


def main() -> None:
    meta = run(build_parser().parse_args())
    print(f"[OFFLINE_BAG_CONTROL] wrote output={meta.get('out')} processed_frames={meta.get('processed_frames')}")


if __name__ == "__main__":
    main()
