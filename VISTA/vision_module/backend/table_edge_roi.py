#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure ROI selection helpers for table-edge perception."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from ..utils.table_roi import bbox_center_norm, bbox_to_quadrant, find_table_bbox, quadrant_to_roi


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
        depth_edge_roi = preset_roi
        roi_source = f"preset:{preset_name}"
        roi_reason = "debug_roi_preset"
    elif manual_static:
        depth_edge_roi = fallback_roi
        roi_source = "manual_static"
        roi_reason = "manual_static_roi_enabled"
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
    }
