#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Dict, Optional, Sequence


def payload_has_table_edge_obs(payload: Optional[Dict[str, object]]) -> bool:
    return isinstance(payload, dict) and (
        isinstance(payload.get("table_edge_obs"), dict) or isinstance(payload.get("mock_table_edge_obs"), dict)
    )


def default_table_edge_obs() -> Dict[str, object]:
    return {
        "table_found": False,
        "edge_found": False,
        "edge_valid": False,
        "confidence": 0.0,
        "edge_conf": 0.0,
        "yaw_err_rad": None,
        "yaw_err": None,
        "dist_err_m": None,
        "dist_err": None,
        "edge_k": None,
        "edge_b": None,
        "depth_valid": False,
        "edge_obs_unavailable": True,
        "point_count": 0,
        "table_point_count": 0,
        "obs_ts": None,
        "age_ms": None,
        "frame_id": None,
        "seq": None,
        "source_mode": "",
        "is_stale": True,
        "source": "vision_table_edge_manager",
        "type": "table_edge_obs",
    }


def _coerce_bbox(value: object) -> Optional[list]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
    except Exception:
        return None


def _shape_wh(value: object) -> Optional[tuple]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        h = float(value[0])
        w = float(value[1])
    except Exception:
        return None
    if w <= 0.0 or h <= 0.0:
        return None
    return w, h


def _bbox_metrics(bbox: Sequence[float], shape: object) -> Dict[str, object]:
    wh = _shape_wh(shape)
    if wh is None:
        return {
            "table_cx_norm": None,
            "table_size_norm": None,
            "table_bbox_area_ratio": None,
            "table_bbox_touch_left": False,
            "table_bbox_touch_right": False,
            "table_bbox_touch_bottom": False,
            "table_bbox_boundary_allowed": False,
        }
    w, h = wh
    x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
    bw = max(0.0, x1 - x0)
    bh = max(0.0, y1 - y0)
    cx_norm = (((x0 + x1) * 0.5) / max(1.0, w) - 0.5) * 2.0
    area_ratio = (bw * bh) / max(1.0, w * h)
    touch_left = x0 <= max(2.0, w * 0.02)
    touch_right = x1 >= min(w - 2.0, w * 0.98)
    touch_bottom = y1 >= min(h - 2.0, h * 0.98)
    return {
        "table_cx_norm": max(-1.0, min(1.0, float(cx_norm))),
        "table_size_norm": max(0.0, min(1.0, float(area_ratio))),
        "table_bbox_area_ratio": max(0.0, min(1.0, float(area_ratio))),
        "table_bbox_touch_left": bool(touch_left),
        "table_bbox_touch_right": bool(touch_right),
        "table_bbox_touch_bottom": bool(touch_bottom),
        "table_bbox_boundary_allowed": bool(touch_left or touch_right or touch_bottom),
    }


def merge_table_bbox_from_local_perception(
    obs: Dict[str, object],
    local_perception: object,
    *,
    tick_ts: float,
) -> Dict[str, object]:
    """Preserve current YOLO table bbox when the depth edge worker has no result."""
    if not isinstance(local_perception, dict):
        return obs
    bbox = _coerce_bbox(
        local_perception.get("table_bbox")
        or local_perception.get("table_bbox_xyxy")
        or local_perception.get("yolo_table_bbox")
    )
    if bbox is None:
        return obs
    out = dict(obs or default_table_edge_obs())
    existing_bbox = _coerce_bbox(out.get("table_bbox_xyxy") or out.get("table_bbox") or out.get("yolo_table_bbox"))
    if bool(out.get("table_bbox_found", False)) and existing_bbox is not None:
        return out

    metrics = _bbox_metrics(bbox, local_perception.get("rgb_shape"))
    source = str(local_perception.get("table_roi_source") or "local_perception_table_bbox")
    out.update(
        {
            "table_found": True,
            "edge_found": bool(out.get("edge_found", False)),
            "edge_valid": bool(out.get("edge_valid", False)),
            "edge_obs_unavailable": bool(out.get("edge_obs_unavailable", True)),
            "depth_valid": None,
            "obs_ts": float(tick_ts),
            "age_ms": 0.0,
            "is_stale": False,
            "reason": "table_bbox_from_local_perception_no_edge_result",
            "table_confirmed_by_yolo": True,
            "table_bbox_current_found": True,
            "table_bbox_control_valid": True,
            "table_bbox_found": True,
            "table_bbox_xyxy": bbox,
            "table_bbox": bbox,
            "table_bbox_source": source,
            "yolo_table_bbox": bbox,
            "yolo_reliable": True,
            "yolo_valid_reason": "table_bbox_found",
            "yolo_invalid_reason": "",
            "docking_enabled_by_yolo": True,
            "edge_control_allowed": True,
            "edge_control_block_reason": "",
            "yolo_table_control_valid": True,
            "roi_source": source,
            "roi_reason": local_perception.get("table_roi_reason") or "local_perception_table_bbox",
            "table_quadrant": local_perception.get("table_quadrant"),
            "rgb_search_roi": local_perception.get("rgb_search_roi"),
            "rgb_shape": local_perception.get("rgb_shape"),
            "source": "local_perception_table_bbox",
        }
    )
    out.update(metrics)
    out["yolo_bbox_area_norm"] = out.get("table_bbox_area_ratio")
    out["yolo_bbox_touch_left"] = out.get("table_bbox_touch_left")
    out["yolo_bbox_touch_right"] = out.get("table_bbox_touch_right")
    out["yolo_bbox_touch_bottom"] = out.get("table_bbox_touch_bottom")
    out["yolo_bbox_touch_boundary"] = bool(
        out.get("table_bbox_touch_left") or out.get("table_bbox_touch_right") or out.get("table_bbox_touch_bottom")
    )
    return out


def table_edge_obs_from_payload(payload: Optional[Dict[str, object]]) -> Dict[str, object]:
    base = default_table_edge_obs()
    source = None
    if isinstance(payload, dict):
        source = payload.get("table_edge_obs") or payload.get("mock_table_edge_obs")
    if isinstance(source, dict):
        base.update(source)
    return base


def table_edge_obs_from_results(results: Dict[str, object]) -> Optional[Dict[str, object]]:
    table_edge = (results or {}).get("table_edge_obs")
    if not isinstance(table_edge, dict):
        return None
    merged = default_table_edge_obs()
    merged.update(table_edge)
    merged["type"] = "table_edge_obs"
    if "edge_obs_unavailable" not in table_edge:
        merged["edge_obs_unavailable"] = str(merged.get("reason") or "") in {
            "depth_unavailable",
            "depth_frame_missing",
            "depth_frame_not_2d",
            "detector_unavailable",
        }
    if "is_stale" not in table_edge:
        merged["is_stale"] = False
    return merged


def table_edge_stale_ms() -> float:
    try:
        return max(0.0, float(os.getenv("VISTA_TABLE_EDGE_STALE_MS", "500") or 500.0))
    except Exception:
        return 500.0


def annotate_table_edge_obs(
    obs: Dict[str, object],
    *,
    tick_ts: float,
    source: str,
    source_mode: str,
) -> Dict[str, object]:
    out = dict(obs or default_table_edge_obs())
    out["type"] = "table_edge_obs"
    out["source_mode"] = str(source_mode or "").strip().upper()
    out["edge_conf"] = float(out.get("edge_conf", out.get("confidence", 0.0)) or 0.0)
    out["yaw_err"] = out.get("yaw_err", out.get("yaw_err_rad"))
    out["dist_err"] = out.get("dist_err", out.get("dist_err_m"))
    out.setdefault("seq", out.get("frame_seq", out.get("frame_id")))
    out.setdefault("frame_id", out.get("frame_seq", out.get("seq")))
    out["edge_obs_unavailable"] = bool(
        out.get("edge_obs_unavailable", False)
        or out.get("reason") in {"depth_unavailable", "depth_frame_missing", "depth_frame_not_2d", "detector_unavailable"}
    )
    out["edge_valid"] = bool(out.get("edge_found", False) and not out.get("edge_obs_unavailable", False))
    obs_ts = out.get("obs_ts", out.get("ts"))
    age_ms = None
    try:
        if obs_ts is not None:
            obs_ts_f = float(obs_ts)
            out["obs_ts"] = obs_ts_f
            age_ms = max(0.0, (float(tick_ts) - obs_ts_f) * 1000.0)
    except Exception:
        age_ms = None
    if age_ms is None:
        out.setdefault("obs_ts", None)
        out["age_ms"] = None
        stale = True
    else:
        upstream_age = out.get("age_ms")
        try:
            age_ms = max(float(age_ms), float(upstream_age))
        except Exception:
            pass
        out["age_ms"] = float(age_ms)
        stale = bool(age_ms > table_edge_stale_ms())
    if source != "results":
        stale = True
        out.setdefault("reason", "no_new_table_edge_obs_result")
    out["is_stale"] = bool(out.get("is_stale", False) or stale or out.get("edge_obs_unavailable", False))
    return out
