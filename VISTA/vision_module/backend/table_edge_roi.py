#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure ROI selection helpers for table-edge perception."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from ..utils.table_roi import bbox_center_norm, bbox_to_quadrant, find_table_bbox, quadrant_to_roi, table_detection_debug


ROI_PRESETS = {
    "full_frame": (0.00, 0.00, 1.00, 1.00),
    "center_mid": (0.25, 0.35, 0.75, 0.65),
    "center_lower": (0.25, 0.50, 0.75, 0.85),
    "full_width_lower": (0.00, 0.50, 1.00, 0.95),
    "right_lower": (0.50, 0.50, 1.00, 0.95),
}


def _parse_shape(shape: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        return None
    try:
        height = int(shape[0])
        width = int(shape[1])
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return height, width


def _parse_bbox(value: Any) -> Optional[list[int]]:
    if isinstance(value, dict):
        value = value.get("bbox") or value.get("xyxy") or value.get("box")
    if isinstance(value, str):
        value = [part.strip() for part in value.replace(";", ",").split(",")]
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value[:4]]
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _clip_bbox(bbox: Sequence[int], shape: Any) -> list[int]:
    parsed_shape = _parse_shape(shape)
    if parsed_shape is None:
        return [int(v) for v in bbox[:4]]
    height, width = parsed_shape
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return [x1, y1, x2, y2]


def _normalize_quadrant(quadrant: Any) -> Optional[str]:
    text = str(quadrant or "").strip().upper()
    aliases = {
        "TOP_LEFT": "LT",
        "TOP_RIGHT": "RT",
        "BOTTOM_LEFT": "LB",
        "BOTTOM_RIGHT": "RB",
        "TL": "LT",
        "TR": "RT",
        "BL": "LB",
        "BR": "RB",
    }
    text = aliases.get(text, text)
    return text if text in {"LT", "RT", "LB", "RB"} else None


def normalize_table_bbox(value: Any, image_shape: Any = None) -> Optional[list[int]]:
    """Normalize a table bbox to xyxy ints, clipped when an image shape is known."""
    parsed = _parse_bbox(value)
    if parsed is None:
        return None
    return _clip_bbox(parsed, image_shape) if _parse_shape(image_shape) is not None else parsed


def choose_table_quadrant(table_bbox: Any, rgb_shape: Any = None, fallback_quadrant: Any = None) -> Optional[str]:
    """Choose LT/RT/LB/RB from bbox center, with an explicit fallback quadrant."""
    bbox = normalize_table_bbox(table_bbox, rgb_shape)
    quadrant = bbox_to_quadrant(bbox, rgb_shape) if bbox is not None else None
    return quadrant or _normalize_quadrant(fallback_quadrant)


def quadrant_to_depth_roi(quadrant: Any, depth_shape: Any, fallback_depth_roi: Any = None) -> Optional[list[int]]:
    """Map LT/RT/LB/RB quadrant to depth-frame xyxy ROI."""
    shape = _parse_shape(depth_shape)
    if shape is not None:
        height, width = shape
        roi = quadrant_to_roi(quadrant, width, height)
        if roi is not None:
            return roi
    return normalize_table_bbox(fallback_depth_roi, depth_shape)


def preset_to_roi(preset: Any, image_shape: Any) -> Optional[list[int]]:
    name = str(preset or "").strip().lower()
    ratios = ROI_PRESETS.get(name)
    shape = _parse_shape(image_shape)
    if ratios is None or shape is None:
        return None
    height, width = shape
    x1 = int(round(float(ratios[0]) * width))
    y1 = int(round(float(ratios[1]) * height))
    x2 = int(round(float(ratios[2]) * width))
    y2 = int(round(float(ratios[3]) * height))
    return _clip_bbox([x1, y1, x2, y2], image_shape)


def _bbox_area_ratio(bbox: Sequence[int], image_shape: Any) -> Optional[float]:
    shape = _parse_shape(image_shape)
    if shape is None:
        return None
    height, width = shape
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area / max(1.0, float(width * height))


def _shift_roi_center_x(roi: Sequence[int], center_x: float, image_shape: Any) -> Optional[list[int]]:
    shape = _parse_shape(image_shape)
    parsed = _parse_bbox(roi)
    if shape is None or parsed is None:
        return None
    height, width = shape
    x1, y1, x2, y2 = parsed
    roi_w = max(1, int(x2) - int(x1))
    cx = max(0.0, min(float(width), float(center_x)))
    new_x1 = int(round(cx - roi_w * 0.5))
    return _clip_bbox([new_x1, y1, new_x1 + roi_w, y2], image_shape)


def compute_dynamic_table_roi_from_yolo_bbox(
    image_shape: Any,
    table_bbox_xyxy: Any,
    current_static_roi: Any,
    roi_width: Any = None,
    roi_height: Any = None,
    mode: str = "",
    distance_hint: Any = None,
    edge_stability: Any = None,
    *,
    bbox_score: Any = None,
    class_id: Any = None,
    table_class_id: int = 0,
    conf_min: float = 0.25,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.90,
    smoothed_center_x: Any = None,
    bbox_image_shape: Any = None,
) -> Dict[str, Any]:
    """Return a center-following ROI that preserves the current static ROI size."""
    static_roi = normalize_table_bbox(current_static_roi, image_shape)
    if static_roi is None:
        return {
            "dynamic_roi": None,
            "roi_source": "static_default",
            "roi_reason": "static_roi_unavailable",
            "bbox_valid": False,
            "bbox_reject_reason": "static_roi_unavailable",
        }

    mode_text = str(mode or "").strip().lower()
    if mode_text in {"near", "near_distance", "edge_only"}:
        return {
            "dynamic_roi": static_roi,
            "roi_source": "static_bottom_near",
            "roi_reason": "near_distance_prefers_bottom_roi",
            "bbox_valid": False,
            "bbox_reject_reason": "near_distance",
        }

    bbox_shape = bbox_image_shape or image_shape
    bbox = normalize_table_bbox(table_bbox_xyxy, bbox_shape)
    if bbox is None:
        return {
            "dynamic_roi": static_roi,
            "roi_source": "static_bottom",
            "roi_reason": "table_bbox_unavailable",
            "bbox_valid": False,
            "bbox_reject_reason": "table_bbox_unavailable",
        }
    try:
        cid_ok = class_id is None or int(float(class_id)) == int(table_class_id)
    except (TypeError, ValueError):
        cid_ok = False
    try:
        score = float(bbox_score) if bbox_score is not None else None
    except (TypeError, ValueError):
        score = None
    if not cid_ok:
        reason = "class_id_mismatch"
    elif score is not None and score < float(conf_min):
        reason = "confidence_too_low"
    else:
        reason = ""
    area_ratio = _bbox_area_ratio(bbox, bbox_shape)
    if not reason and (area_ratio is None or area_ratio < float(min_area_ratio)):
        reason = "bbox_area_too_small"
    if not reason and area_ratio is not None and area_ratio > float(max_area_ratio):
        reason = "bbox_area_too_large"
    if reason:
        return {
            "dynamic_roi": static_roi,
            "roi_source": "static_bottom",
            "roi_reason": reason,
            "bbox_valid": False,
            "bbox_reject_reason": reason,
            "yolo_bbox_area_ratio": area_ratio,
            "yolo_table_conf": score,
        }

    shape = _parse_shape(image_shape)
    bbox_hw = _parse_shape(bbox_shape)
    height, width = shape if shape is not None else (0, 0)
    bbox_width = bbox_hw[1] if bbox_hw is not None else width
    raw_center_x = (float(bbox[0]) + float(bbox[2])) * 0.5
    raw_center_norm = raw_center_x / max(1.0, float(bbox_width))
    raw_target_center_x = raw_center_norm * max(1.0, float(width))
    try:
        center_x = float(smoothed_center_x) if smoothed_center_x is not None else raw_target_center_x
    except (TypeError, ValueError):
        center_x = raw_target_center_x
    roi = _shift_roi_center_x(static_roi, center_x, image_shape) or static_roi
    return {
        "dynamic_roi": roi,
        "roi_source": "yolo_table_bbox",
        "roi_reason": "table_bbox_center_follow",
        "bbox_valid": True,
        "bbox_reject_reason": "",
        "yolo_bbox_area_ratio": area_ratio,
        "yolo_table_conf": score,
        "yolo_table_class_id": int(table_class_id),
        "yolo_bbox_center_x": raw_center_x,
        "yolo_bbox_center_x_norm": raw_center_norm,
        "yolo_roi_center_x": (float(roi[0]) + float(roi[2])) * 0.5,
        "yolo_roi_center_x_norm": ((float(roi[0]) + float(roi[2])) * 0.5) / max(1.0, float(width)),
        "edge_stability": edge_stability,
        "distance_hint": distance_hint,
        "roi_width": int(roi_width) if roi_width is not None else int(static_roi[2] - static_roi[0]),
        "roi_height": int(roi_height) if roi_height is not None else int(static_roi[3] - static_roi[1]),
    }


def choose_depth_roi(
    local_perception: Any,
    rgb_shape: Any = None,
    depth_shape: Any = None,
    fallback_depth_roi: Any = None,
    *,
    last_valid_table_bbox: Any = None,
    last_valid_table_center_norm: Any = None,
    last_valid_quadrant: Any = None,
    last_valid_age_s: Optional[float] = None,
    last_valid_ttl_s: float = 1.0,
    manual_static: bool = False,
    roi_preset: Any = "",
    yolo_dynamic_enable: bool = False,
    yolo_table_class_id: int = 0,
    yolo_table_conf_min: float = 0.25,
    yolo_min_area_ratio: float = 0.01,
    yolo_max_area_ratio: float = 0.90,
    smoothed_table_center_x: Any = None,
    near_distance: bool = False,
    edge_stable: bool = False,
) -> Dict[str, Any]:
    """Choose table-edge ROI from current detection, recent history, or static fallback."""
    local = dict(local_perception or {}) if isinstance(local_perception, dict) else {}
    resolved_rgb_shape = rgb_shape or local.get("rgb_shape")
    table_bbox = normalize_table_bbox(find_table_bbox(local), resolved_rgb_shape)
    table_center = bbox_center_norm(table_bbox, resolved_rgb_shape) if table_bbox is not None else None
    quadrant = choose_table_quadrant(table_bbox, resolved_rgb_shape, local.get("table_quadrant"))
    fallback_roi = normalize_table_bbox(fallback_depth_roi, depth_shape)
    rgb_search_roi = None
    depth_edge_roi = None
    roi_source = "static_fallback"
    roi_reason = "table_bbox_unavailable"
    preset_name = str(roi_preset or "").strip().lower()
    preset_roi = preset_to_roi(preset_name, depth_shape)
    if preset_roi is not None:
        fallback_roi = preset_roi

    if manual_static:
        depth_edge_roi = fallback_roi
        roi_source = "manual_static"
        roi_reason = "manual_static_roi_enabled"
    elif yolo_dynamic_enable and fallback_roi is not None:
        det = table_detection_debug(local, resolved_rgb_shape, min_conf=yolo_table_conf_min)
        dyn = compute_dynamic_table_roi_from_yolo_bbox(
            depth_shape,
            table_bbox,
            fallback_roi,
            mode="near" if near_distance else "",
            edge_stability="stable" if edge_stable else "unstable",
            bbox_score=det.get("conf"),
            class_id=yolo_table_class_id if det.get("found") else None,
            table_class_id=yolo_table_class_id,
            conf_min=yolo_table_conf_min,
            min_area_ratio=yolo_min_area_ratio,
            max_area_ratio=yolo_max_area_ratio,
            smoothed_center_x=smoothed_table_center_x,
            bbox_image_shape=resolved_rgb_shape,
        )
        depth_edge_roi = dyn.get("dynamic_roi") or fallback_roi
        roi_source = str(dyn.get("roi_source") or "static_bottom")
        roi_reason = str(dyn.get("roi_reason") or "")
    elif quadrant is not None and table_bbox is not None:
        depth_edge_roi = quadrant_to_depth_roi(quadrant, depth_shape, fallback_roi)
        rgb_hw = _parse_shape(resolved_rgb_shape)
        if rgb_hw is not None:
            rgb_search_roi = quadrant_to_roi(quadrant, rgb_hw[1], rgb_hw[0])
        roi_source = "local_perception_table_bbox"
        roi_reason = "table_bbox_detected"
    else:
        age_ok = last_valid_age_s is not None and float(last_valid_age_s) <= float(last_valid_ttl_s)
        history_quadrant = _normalize_quadrant(last_valid_quadrant)
        if history_quadrant and age_ok:
            quadrant = history_quadrant
            table_bbox = normalize_table_bbox(last_valid_table_bbox, resolved_rgb_shape)
            table_center = last_valid_table_center_norm
            depth_edge_roi = quadrant_to_depth_roi(quadrant, depth_shape, fallback_roi)
            roi_source = "last_valid_table_bbox"
            roi_reason = "table_bbox_lost_using_history"
        else:
            depth_edge_roi = fallback_roi

    return {
        "table_bbox": table_bbox,
        "table_center_norm": table_center,
        "table_quadrant": quadrant,
        "rgb_search_roi": rgb_search_roi,
        "depth_edge_roi": depth_edge_roi,
        "table_edge_roi": depth_edge_roi,
        "edge_roi": depth_edge_roi,
        "roi_source": roi_source,
        "roi_reason": roi_reason,
        "roi_preset": preset_name if preset_roi is not None else None,
        "roi_format": "xyxy",
        **(dyn if "dyn" in locals() and isinstance(dyn, dict) else {}),
    }
