#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import time
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
        "table_roi_depth_valid": False,
        "table_roi_depth_p10": None,
        "table_roi_depth_median": None,
        "table_roi_depth_valid_ratio": 0.0,
        "table_roi_depth_sample_count": 0,
        "table_roi_depth_bbox": None,
        "table_roi_depth_bbox_norm": None,
        "table_roi_depth_coord_space": "depth_frame_xyxy",
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


def _edge_semantic_present(obs: Dict[str, object]) -> bool:
    if not isinstance(obs, dict):
        return False
    for key in (
        "edge_found",
        "edge_valid",
        "edge_trusted",
        "candidate_line_present",
        "detector_candidate_line_present",
        "edge_candidate_found",
    ):
        if bool(obs.get(key, False)):
            return True
    for key in ("point_count", "table_point_count", "valid_edge_points", "support_count", "inlier_count"):
        try:
            if int(obs.get(key, 0) or 0) > 0:
                return True
        except Exception:
            pass
    return False


_last_current_check_log_ts = 0.0
_last_current_enough_state = None


def check_edge_current_enough(
    edge_frame_id: Optional[int],
    local_frame_id: Optional[int],
    edge_ts: Optional[float],
    local_ts: Optional[float],
) -> tuple:
    global _last_current_check_log_ts, _last_current_enough_state
    
    # Coerce to int/float if possible
    def safe_int(val):
        if val is None:
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    def safe_float(val):
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    edge_fid = safe_int(edge_frame_id)
    local_fid = safe_int(local_frame_id)
    edge_t = safe_float(edge_ts)
    local_t = safe_float(local_ts)

    frame_id_delta = None
    if edge_fid is not None and local_fid is not None:
        frame_id_delta = edge_fid - local_fid

    ts_delta_ms = None
    if edge_t is not None and local_t is not None:
        ts_delta_ms = abs(edge_t - local_t) * 1000.0

    threshold = 0.25
    try:
        from common.config_loader import get_config
        cfg = get_config()
        if cfg is not None and hasattr(cfg, "vision") and hasattr(cfg.vision, "table_edge") and hasattr(cfg.vision.table_edge, "edge_sync_threshold_s"):
            threshold = float(cfg.vision.table_edge.edge_sync_threshold_s)
    except Exception:
        pass

    edge_current_enough = False
    if edge_t is not None and local_t is not None:
        edge_current_enough = abs(edge_t - local_t) <= threshold
    elif edge_fid is not None and local_fid is not None:
        edge_current_enough = abs(edge_fid - local_fid) <= 1

    now = time.time()
    state_changed = (edge_current_enough != _last_current_enough_state)
    if state_changed or (now - _last_current_check_log_ts >= 0.5):
        _last_current_check_log_ts = now
        _last_current_enough_state = edge_current_enough
        logger = logging.getLogger("vision.stage.search")
        logger.info(
            "[EDGE_CURRENT_CHECK] edge_frame_id=%s local_frame_id=%s frame_id_delta=%s "
            "edge_ts=%s local_ts=%s ts_delta_ms=%s edge_current_enough=%s",
            edge_fid,
            local_fid,
            frame_id_delta,
            edge_t,
            local_t,
            f"{ts_delta_ms:.2f}" if ts_delta_ms is not None else None,
            str(edge_current_enough).lower(),
        )

    details = {
        "edge_frame_id": edge_fid,
        "local_frame_id": local_fid,
        "frame_id_delta": frame_id_delta,
        "edge_ts": edge_t,
        "local_ts": local_t,
        "ts_delta_ms": ts_delta_ms,
        "edge_current_enough": edge_current_enough,
    }
    return edge_current_enough, details


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
    protected_edge = bool(
        str(out.get("selected_source") or out.get("source") or "").strip().lower() == "results"
        and _edge_semantic_present(out)
    )
    protected_values = {
        key: out.get(key)
        for key in (
            "edge_found",
            "edge_valid",
            "edge_trusted",
            "valid_for_control",
            "edge_control_allowed",
            "point_count",
            "table_point_count",
            "valid_edge_points",
            "support_count",
            "inlier_count",
            "edge_inlier_count",
            "yaw_err_rad",
            "yaw_err",
            "yaw",
            "dist_err_m",
            "dist_err",
            "dist",
            "lateral_err_m",
            "lateral",
            "reason",
            "reject_reason",
            "edge_reject_for_control_reason",
            "edge_control_block_reason",
            "source",
            "selected_source",
            "edge_obs_unavailable",
            "frame_id",
            "seq",
            "obs_ts",
            "ts",
            "age_ms",
            "is_stale",
        )
        if key in out
    }
    if bbox is None and not _has_local_table_bbox_signal(local_perception):
        if not bool(out.get("table_bbox_current_found", False)):
            out.update({
                "table_roi_depth_valid": False,
                "table_roi_depth_p10": None,
                "table_roi_depth_median": None,
                "table_roi_depth_valid_ratio": 0.0,
                "table_roi_depth_sample_count": 0,
                "table_roi_depth_bbox": None,
                "table_roi_depth_bbox_norm": None,
            })
        return out
    obs_ts = _local_obs_ts(local_perception, tick_ts)
    frame_id = _local_frame_id(local_perception)

    edge_frame_id = out.get("frame_id") or out.get("frame_seq") or out.get("seq")
    
    edge_ts_val = None
    for k in ("obs_ts", "ts"):
        v = out.get(k)
        if v is not None:
            try:
                edge_ts_val = float(v)
                break
            except (ValueError, TypeError):
                pass

    local_ts_val = None
    for k in ("obs_ts", "ts", "frame_capture_ts"):
        v = local_perception.get(k)
        if v is not None:
            try:
                local_ts_val = float(v)
                break
            except (ValueError, TypeError):
                pass
    if local_ts_val is None:
        local_ts_val = tick_ts

    is_current_frame, current_details = check_edge_current_enough(
        edge_frame_id=edge_frame_id,
        local_frame_id=frame_id,
        edge_ts=edge_ts_val,
        local_ts=local_ts_val,
    )

    was_valid_edge = bool(out.get("edge_found")) or bool(out.get("edge_valid")) or bool(out.get("edge_trusted"))
    if was_valid_edge and not is_current_frame:
        logger = logging.getLogger("vision.stage.search")
        logger.warning(
            "[EDGE_DROPPED_STALE] reason=frame_or_ts_mismatch edge_frame_id=%s local_frame_id=%s "
            "frame_id_delta=%s edge_ts=%s local_ts=%s ts_delta_ms=%s edge_current_enough=%s",
            current_details["edge_frame_id"],
            current_details["local_frame_id"],
            current_details["frame_id_delta"],
            current_details["edge_ts"],
            current_details["local_ts"],
            current_details["ts_delta_ms"],
            str(current_details["edge_current_enough"]).lower(),
        )

    if bbox is None:
        if protected_edge:
            edge_present = bool(out.get("edge_found", False))
            edge_valid = bool(out.get("edge_valid", False))
            edge_trusted = bool(out.get("edge_trusted", False))
            reason = out.get("reason") or ("no_current_table_bbox_edge_candidate_present" if edge_present else "no_current_table_bbox")
        elif is_current_frame:
            edge_present = bool(out.get("edge_found", False))
            edge_valid = bool(out.get("edge_valid", False))
            edge_trusted = bool(out.get("edge_trusted", False))
            reason = out.get("reason") or ("no_current_table_bbox_edge_candidate_present" if edge_present else "no_current_table_bbox")
        else:
            edge_present = False
            edge_valid = False
            edge_trusted = False
            out["edge_found"] = False
            out["edge_valid"] = False
            out["edge_trusted"] = False
            out["point_count"] = 0
            out["table_point_count"] = 0
            reason = "no_current_table_bbox"

        out.update(
            {
                "table_found": bool(edge_present and edge_valid),
                "edge_found": edge_present,
                "edge_valid": edge_valid,
                "edge_trusted": edge_trusted,
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
                "table_roi_depth_valid": False,
                "table_roi_depth_p10": None,
                "table_roi_depth_median": None,
                "table_roi_depth_valid_ratio": 0.0,
                "table_roi_depth_sample_count": 0,
                "table_roi_depth_bbox": None,
                "table_roi_depth_bbox_norm": None,
                "reason": reason,
                "source": out.get("source") if protected_edge else "local_perception_no_table_bbox",
            }
        )
        if protected_edge:
            out.update(protected_values)
            out["table_confirmed_by_yolo"] = False
            out["table_bbox_current_found"] = False
            out["table_bbox_control_valid"] = False
            out["table_bbox_found"] = False
            out["table_bbox_xyxy"] = None
            out["table_bbox"] = None
            out["table_bbox_source"] = "none"
            out["yolo_table_bbox"] = None
            out["yolo_reliable"] = False
            out["yolo_valid_reason"] = ""
            out["yolo_invalid_reason"] = "no_current_table_bbox"
            out["docking_enabled_by_yolo"] = False
            out["yolo_table_control_valid"] = False
            out["yolo_table_visible"] = False
            out["yolo_table_fresh"] = False
            out["yolo_table_age_ms"] = None
        return out

    if protected_edge:
        edge_present = bool(out.get("edge_found", False))
        edge_valid = bool(out.get("edge_valid", False))
        edge_trusted = bool(out.get("edge_trusted", False))
        edge_obs_unavailable = bool(out.get("edge_obs_unavailable", False))
        point_count = int(out.get("point_count", 0) or 0)
        table_point_count = int(out.get("table_point_count", 0) or point_count)
        reason = out.get("reason") or (
            "edge_trusted" if edge_trusted else ("edge_valid" if edge_valid else ("edge_candidate_found" if edge_present else "edge_result_with_local_perception_table_bbox"))
        )
    elif is_current_frame:
        edge_present = bool(out.get("edge_found", False))
        edge_valid = bool(out.get("edge_valid", False))
        edge_trusted = bool(out.get("edge_trusted", False))
        edge_obs_unavailable = bool(out.get("edge_obs_unavailable", False))
        point_count = int(out.get("point_count", 0) or 0)
        table_point_count = int(out.get("table_point_count", 0) or 0)
        reason = out.get("reason") or ("edge_result_with_local_perception_table_bbox" if edge_present else "table_bbox_from_local_perception_no_edge_result")
    else:
        edge_present = False
        edge_valid = False
        edge_trusted = False
        edge_obs_unavailable = True
        point_count = 0
        table_point_count = 0
        reason = "table_bbox_from_local_perception_no_edge_result"
        out["edge_found"] = False
        out["edge_valid"] = False
        out["edge_trusted"] = False
        out["point_count"] = 0
        out["table_point_count"] = 0

    metrics = _bbox_metrics(bbox, local_perception.get("rgb_shape"))
    source = str(local_perception.get("table_roi_source") or "local_perception_table_bbox")
    conf = (
        local_perception.get("yolo_table_conf")
        or local_perception.get("table_conf")
        or local_perception.get("table_bbox_conf_raw")
        or _bbox_conf(local_perception.get("detected_table_bbox"))
        or _bbox_conf(local_perception.get("table_bbox"))
    )

    out.update(
        {
            "table_found": True,
            "edge_found": edge_present,
            "edge_valid": edge_valid,
            "edge_trusted": edge_trusted,
            "edge_obs_unavailable": edge_obs_unavailable,
            "point_count": point_count,
            "table_point_count": table_point_count,
            "obs_ts": out.get("obs_ts") if protected_edge else float(obs_ts),
            "ts": out.get("ts") if protected_edge else float(obs_ts),
            "age_ms": out.get("age_ms") if protected_edge else 0.0,
            "frame_id": out.get("frame_id") if protected_edge else frame_id,
            "seq": out.get("seq") if protected_edge else frame_id,
            "is_stale": bool(out.get("is_stale", False)) if protected_edge else False,
            "reason": reason,
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
            "edge_control_allowed": edge_trusted,
            "edge_control_block_reason": "" if edge_trusted else (out.get("edge_reject_for_control_reason") or out.get("edge_control_block_reason") or "edge_not_trusted"),
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
            "source": out.get("source") if protected_edge else "local_perception_table_bbox",
        }
    )
    if protected_edge:
        out.update(protected_values)
        out.update(
            {
                "table_found": True,
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
            }
        )

    if is_current_frame:
        out.setdefault("depth_valid", True)
    else:
        out["depth_valid"] = False

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
    merged["debug_publish_found"] = bool(merged.get("edge_found", merged.get("edge_detected", False)))
    merged["debug_publish_valid"] = bool(merged.get("edge_valid", merged.get("edge_geometry_valid", False)))
    merged["debug_publish_trusted"] = bool(merged.get("edge_trusted", False))
    merged["debug_publish_point_count"] = int(merged.get("point_count", 0) or 0)
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
    
    # Ensure aliases and defaults are populated first
    out.setdefault("seq", out.get("frame_seq", out.get("frame_id")))
    out.setdefault("frame_id", out.get("frame_seq", out.get("seq")))
    
    # Detect candidate
    edge_candidate = bool(
        out.get("detector_candidate_line_present")
        or out.get("candidate")
        or out.get("candidate_line_present")
        or out.get("edge_candidate_found")
        or (out.get("support_count", 0) or 0) > 0
        or (out.get("inlier_count", 0) or 0) > 0
    )
    out["edge_candidate_found"] = edge_candidate
    out["candidate_line_present"] = edge_candidate
    out["detector_candidate_line_present"] = edge_candidate

    edge_obs_unavailable = bool(
        out.get("edge_obs_unavailable", False)
        or out.get("reason") in {"depth_unavailable", "depth_frame_missing", "depth_frame_not_2d", "detector_unavailable"}
    )
    out["edge_obs_unavailable"] = edge_obs_unavailable

    # Get raw status flags
    raw_edge_found = bool(out.get("edge_found", out.get("edge_detected", False)))
    raw_edge_valid = bool(out.get("edge_valid", out.get("edge_geometry_valid", False)))
    raw_edge_trusted = bool(out.get("edge_trusted", out.get("valid_for_control", out.get("edge_control_allowed", False))))

    # Prioritize and set final fields if this is a fresh results source
    if source == "results":
        if raw_edge_trusted and not edge_obs_unavailable:
            out["reason"] = "edge_trusted"
            out["edge_found"] = True
            out["edge_valid"] = True
            out["edge_trusted"] = True
            out["valid_for_control"] = True
            out["edge_control_allowed"] = True
            out["reject_reason"] = ""
            out["edge_reject_for_control_reason"] = ""
            out["edge_control_block_reason"] = ""
        elif raw_edge_valid and not edge_obs_unavailable:
            out["reason"] = "edge_valid"
            out["edge_found"] = True
            out["edge_valid"] = True
            out["edge_trusted"] = False
            out["valid_for_control"] = False
            out["edge_control_allowed"] = False
        elif edge_candidate and not edge_obs_unavailable:
            out["reason"] = "edge_candidate_rejected"
            out["edge_found"] = True
            out["edge_valid"] = False
            out["edge_trusted"] = False
            out["valid_for_control"] = False
            out["edge_control_allowed"] = False
        else:
            out["reason"] = out.get("reason") or "table_bbox_from_local_perception_no_edge_result"
            out["edge_found"] = False
            out["edge_valid"] = False
            out["edge_trusted"] = False
            out["valid_for_control"] = False
            out["edge_control_allowed"] = False
            out["point_count"] = 0
            out["table_point_count"] = 0
            if not out.get("reject_reason"):
                out["reject_reason"] = out["reason"]
    else:
        out["edge_found"] = raw_edge_found
        out["edge_valid"] = raw_edge_valid
        out["edge_trusted"] = raw_edge_trusted
        out["valid_for_control"] = raw_edge_trusted
        out["edge_control_allowed"] = raw_edge_trusted

    # Populate final unified keys
    out["edge_detected"] = out["edge_found"]
    out["edge_geometry_valid"] = out["edge_valid"]

    # Align point count fields
    point_count = int(out.get("point_count", 0) or 0)
    out["point_count"] = point_count
    out["table_point_count"] = int(out.get("table_point_count") or point_count)
    out["valid_edge_points"] = int(out.get("inlier_count", 0) or 0)

    support_count = int(out.get("support_count", out.get("fast_support_point_count", 0)) or 0)
    out["support_count"] = support_count
    out["fast_support_point_count"] = support_count

    inlier_count = int(out.get("inlier_count", out.get("edge_inlier_count", 0)) or 0)
    out["inlier_count"] = inlier_count
    out["edge_inlier_count"] = inlier_count

    edge_trusted = bool(out.get("edge_trusted", False))
    out["valid_for_control"] = edge_trusted
    out["edge_control_allowed"] = edge_trusted

    # Align error fields
    yaw = out.get("yaw_err_rad", out.get("yaw_err", out.get("yaw")))
    out["yaw_err_rad"] = yaw
    out["yaw_err"] = yaw
    out["yaw"] = yaw

    dist = out.get("dist_err_m", out.get("dist_err", out.get("dist")))
    out["dist_err_m"] = dist
    out["dist_err"] = dist
    out["dist"] = dist

    lateral = out.get("lateral", out.get("lateral_err_m", dist))
    out["lateral"] = lateral
    out["lateral_err_m"] = lateral

    out["edge_conf"] = float(out.get("edge_conf", out.get("confidence", 0.0)) or 0.0)

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
