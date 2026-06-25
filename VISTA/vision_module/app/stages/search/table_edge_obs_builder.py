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
    cy_norm = (((y0 + y1) * 0.5) / max(1.0, h) - 0.5) * 2.0
    area_ratio = (bw * bh) / max(1.0, w * h)
    touch_left = x0 <= max(2.0, w * 0.02)
    touch_right = x1 >= min(w - 2.0, w * 0.98)
    touch_bottom = y1 >= min(h - 2.0, h * 0.98)
    return {
        "table_cx_norm": max(-1.0, min(1.0, float(cx_norm))),
        "table_cy_norm": max(-1.0, min(1.0, float(cy_norm))),
        "table_size_norm": max(0.0, min(1.0, float(area_ratio))),
        "table_bbox_area_ratio": max(0.0, min(1.0, float(area_ratio))),
        "table_bbox_touch_left": bool(touch_left),
        "table_bbox_touch_right": bool(touch_right),
        "table_bbox_touch_bottom": bool(touch_bottom),
        "table_bbox_boundary_allowed": bool(touch_left or touch_right or touch_bottom),
    }


def _bbox_conf(value: object) -> Optional[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 5:
        return None
    try:
        return float(value[4])
    except Exception:
        return None


def _local_frame_id(local_perception: Dict[str, object]) -> object:
    return (
        local_perception.get("frame_seq")
        or local_perception.get("frame_id")
        or local_perception.get("camera_frame_seq")
        or local_perception.get("yolo_frame_seq")
    )


def _local_obs_ts(local_perception: Dict[str, object], tick_ts: float) -> float:
    for key in ("obs_ts", "ts", "frame_capture_ts"):
        value = local_perception.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return float(tick_ts)


def _local_current_table_bbox(local_perception: Dict[str, object]) -> Optional[list]:
    bbox = _coerce_bbox(
        local_perception.get("detected_table_bbox")
        or local_perception.get("table_bbox")
        or local_perception.get("table_bbox_xyxy")
        or local_perception.get("yolo_table_bbox")
    )
    explicit_found = (
        local_perception.get("table_bbox_current_found")
        if local_perception.get("table_bbox_current_found") is not None
        else local_perception.get("table_bbox_detected", local_perception.get("table_found"))
    )
    if explicit_found is False:
        return None
    if explicit_found is not None and not bool(explicit_found):
        return None
    return bbox


def _has_local_table_bbox_signal(local_perception: Dict[str, object]) -> bool:
    for key in (
        "detected_table_bbox",
        "table_bbox",
        "table_bbox_xyxy",
        "yolo_table_bbox",
        "table_bbox_current_found",
        "table_bbox_detected",
        "table_found",
        "table_roi_source",
        "table_roi_reason",
        "yolo_table_conf",
        "table_conf",
    ):
        if key in local_perception:
            return True
    return False


def merge_table_bbox_from_local_perception(
    obs: Dict[str, object],
    local_perception: object,
    *,
    tick_ts: float,
) -> Dict[str, object]:
    """Publish the current-frame YOLO table bbox alongside table-edge output.

    Local perception is the authority for table bbox freshness.  Never let a
    stage-state or table-edge-manager payload carry an old bbox/timestamp into
    a new frame.
    """
    if not isinstance(local_perception, dict):
        return obs
    bbox = _local_current_table_bbox(local_perception)
    out = dict(obs or default_table_edge_obs())
    if bbox is None and not _has_local_table_bbox_signal(local_perception):
        return out
    obs_ts = _local_obs_ts(local_perception, tick_ts)
    frame_id = _local_frame_id(local_perception)

    if bbox is None:
        edge_present = bool(out.get("edge_found", out.get("edge_detected", False)))
        out.update(
            {
                "table_found": bool(out.get("edge_found", False) and out.get("edge_valid", False)),
                "obs_ts": float(obs_ts),
                "ts": float(obs_ts),
                "age_ms": 0.0,
                "frame_id": frame_id,
                "seq": frame_id,
                "is_stale": False,
                "table_confirmed_by_yolo": False,
                "table_bbox_current_found": False,
                "table_bbox_control_valid": False,
                "table_bbox_found": False,
                "table_bbox_xyxy": None,
                "table_bbox": None,
                "table_bbox_source": "none",
                "yolo_table_bbox": None,
                "yolo_reliable": False,
                "yolo_valid_reason": "",
                "yolo_invalid_reason": "no_current_table_bbox",
                "docking_enabled_by_yolo": False,
                "edge_control_allowed": False,
                "edge_control_block_reason": "no_current_table_bbox",
                "yolo_table_control_valid": False,
                "yolo_table_visible": False,
                "yolo_table_fresh": False,
                "yolo_table_age_ms": None,
                "reason": "no_current_table_bbox_edge_candidate_present" if edge_present else "no_current_table_bbox",
                "source": "local_perception_no_table_bbox",
            }
        )
        return out

    metrics = _bbox_metrics(bbox, local_perception.get("rgb_shape"))
    source = str(local_perception.get("table_roi_source") or "local_perception_table_bbox")
    conf = (
        local_perception.get("yolo_table_conf")
        or local_perception.get("table_conf")
        or local_perception.get("table_bbox_conf_raw")
        or _bbox_conf(local_perception.get("detected_table_bbox"))
        or _bbox_conf(local_perception.get("table_bbox"))
    )
    edge_present = bool(out.get("edge_found", out.get("edge_detected", False)))
    edge_valid = bool(out.get("edge_valid", out.get("edge_geometry_valid", False)))
    out.update(
        {
            "table_found": True,
            "edge_found": edge_present,
            "edge_valid": edge_valid,
            "edge_trusted": bool(out.get("edge_trusted", False)),
            "edge_obs_unavailable": bool(out.get("edge_obs_unavailable", True)),
            "depth_valid": None,
            "obs_ts": float(obs_ts),
            "ts": float(obs_ts),
            "age_ms": 0.0,
            "frame_id": frame_id,
            "seq": frame_id,
            "is_stale": False,
            "reason": (out.get("reason") or "edge_result_with_local_perception_table_bbox")
            if edge_present
            else "table_bbox_from_local_perception_no_edge_result",
            "table_confirmed_by_yolo": True,
            "table_bbox_current_found": True,
            "table_bbox_control_valid": True,
            "table_bbox_found": True,
            "table_bbox_xyxy": bbox,
            "table_bbox": bbox,
            "table_bbox_source": source,
            "table_conf": conf,
            "table_bbox_conf_raw": conf,
            "yolo_table_bbox": bbox,
            "yolo_reliable": True,
            "yolo_valid_reason": "table_bbox_found",
            "yolo_invalid_reason": "",
            "docking_enabled_by_yolo": True,
            "edge_control_allowed": True,
            "edge_control_block_reason": "",
            "yolo_table_control_valid": True,
            "yolo_table_visible": True,
            "yolo_table_fresh": True,
            "yolo_table_age_ms": 0.0,
            "yolo_table_conf": conf,
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
    if "edge_found" not in table_edge and "edge_detected" in table_edge:
        merged["edge_found"] = bool(table_edge.get("edge_detected"))
    if "edge_valid" not in table_edge:
        if "edge_geometry_valid" in table_edge:
            merged["edge_valid"] = bool(table_edge.get("edge_geometry_valid")) and not bool(merged.get("edge_obs_unavailable", False))
        else:
            merged["edge_valid"] = bool(merged.get("edge_found", False)) and not bool(merged.get("edge_obs_unavailable", False))
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
    if "edge_found" not in out and "edge_detected" in out:
        out["edge_found"] = bool(out.get("edge_detected"))
    if "edge_valid" not in out:
        if "edge_geometry_valid" in out:
            out["edge_valid"] = bool(out.get("edge_geometry_valid")) and not out.get("edge_obs_unavailable", False)
        else:
            out["edge_valid"] = bool(out.get("edge_found", False) and not out.get("edge_obs_unavailable", False))
    if "valid_for_control" not in out and "edge_trusted" in out:
        out["valid_for_control"] = bool(out.get("edge_trusted"))
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
