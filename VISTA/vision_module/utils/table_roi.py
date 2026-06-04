#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple


TABLE_CLASS_NAMES = {"table", "desk", "diningtable", "dining table", "table1"}
TABLE_CLASS_IDS = {0, 60}


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


def _normalize_class_name(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def table_class_available(local_perception: Any) -> bool:
    local = dict(local_perception or {}) if isinstance(local_perception, dict) else {}
    names = local.get("class_names")
    if isinstance(names, (list, tuple)):
        normalized = {_normalize_class_name(name) for name in names}
        if normalized & TABLE_CLASS_NAMES:
            return True
    return not isinstance(names, (list, tuple)) or len(names) > max(TABLE_CLASS_IDS)


def _row_conf(row: Any) -> float:
    try:
        if isinstance(row, dict):
            return float(row.get("score", row.get("conf", row.get("confidence", 0.0))) or 0.0)
        if isinstance(row, (list, tuple)) and len(row) >= 5:
            return float(row[4] or 0.0)
    except Exception:
        return 0.0
    return 0.0


def _row_class(row: Any, class_names: Any = None) -> tuple[Optional[int], str]:
    class_id = None
    class_name = ""
    if isinstance(row, dict):
        raw = row.get("cls_id", row.get("class_id", row.get("cls", row.get("class"))))
        class_name = _normalize_class_name(row.get("class_name", row.get("name", row.get("label", ""))))
    elif isinstance(row, (list, tuple)):
        raw = row[5] if len(row) > 5 else None
        class_name = _normalize_class_name(row[6]) if len(row) > 6 else ""
    else:
        raw = None
    try:
        class_id = int(float(raw))
    except (TypeError, ValueError):
        class_id = None
    if not class_name and class_id is not None and isinstance(class_names, (list, tuple)) and 0 <= class_id < len(class_names):
        class_name = _normalize_class_name(class_names[class_id])
    return class_id, class_name


def _is_table_row(row: Any, class_names: Any = None) -> bool:
    class_id, class_name = _row_class(row, class_names=class_names)
    return class_name in TABLE_CLASS_NAMES or class_id in TABLE_CLASS_IDS


def bbox_center_norm(bbox: Any, image_shape: Any) -> Optional[list[float]]:
    parsed = _parse_bbox(bbox)
    shape = _parse_shape(image_shape)
    if parsed is None or shape is None:
        return None
    height, width = shape
    x1, y1, x2, y2 = parsed
    cx = ((float(x1) + float(x2)) * 0.5) / float(width)
    cy = ((float(y1) + float(y2)) * 0.5) / float(height)
    return [max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))]


def _clip_roi(roi: Sequence[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in roi[:4]]
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
    q = str(quadrant or "").strip().upper()
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
    q = aliases.get(q, q)
    return q if q in {"LT", "RT", "LB", "RB"} else None


def find_table_bbox(local_perception: Any) -> Optional[list[int]]:
    """Return the first table-like bbox in predictor coordinates."""
    local = dict(local_perception or {}) if isinstance(local_perception, dict) else {}
    for key in ("table_bbox", "desk_bbox", "mock_table_bbox"):
        parsed = _parse_bbox(local.get(key))
        if parsed is not None:
            return parsed
    boxes = local.get("infer_boxes")
    if not isinstance(boxes, list):
        return None
    class_names = local.get("class_names")
    for row in boxes:
        if _is_table_row(row, class_names=class_names):
            return _parse_bbox(row)
    return None


def table_detection_debug(local_perception: Any, image_shape: Any = None, min_conf: float = 0.25, center_tol: float = 0.12) -> Dict[str, Any]:
    local = dict(local_perception or {}) if isinstance(local_perception, dict) else {}
    shape = image_shape or local.get("rgb_shape")
    boxes = local.get("infer_boxes")
    class_names = local.get("class_names")
    table_class_ok = table_class_available(local)
    if not table_class_ok:
        return {"found": False, "no_table_class": True, "reason": "no_table_class"}
    best = None
    best_conf = -1.0
    for row in boxes if isinstance(boxes, list) else []:
        if not _is_table_row(row, class_names=class_names):
            continue
        conf = _row_conf(row)
        if conf > best_conf:
            best = row
            best_conf = conf
    bbox = _parse_bbox(best)
    center = bbox_center_norm(bbox, shape) if bbox is not None else None
    found = bool(bbox is not None and best_conf >= float(min_conf))
    direction = "unknown"
    if center is not None:
        dx = float(center[0]) - 0.5
        tol = max(0.0, float(center_tol))
        if dx < -tol:
            direction = "left"
        elif dx > tol:
            direction = "right"
        else:
            direction = "center"
    return {
        "found": found,
        "no_table_class": False,
        "bbox": bbox,
        "cx": center[0] if center is not None else None,
        "cy": center[1] if center is not None else None,
        "conf": best_conf if best_conf >= 0.0 else None,
        "direction": direction,
        "reason": "ok" if found else "not_found_or_low_conf",
    }


def table_bbox_meta(local_perception: Any, image_shape: Any) -> Dict[str, Any]:
    table_bbox = find_table_bbox(local_perception)
    return {
        "table_bbox": table_bbox,
        "table_center_norm": bbox_center_norm(table_bbox, image_shape),
        "table_quadrant": bbox_to_quadrant(table_bbox, image_shape) if table_bbox is not None else None,
    }


def bbox_to_quadrant(bbox: Any, image_shape: Any) -> Optional[str]:
    parsed = _parse_bbox(bbox)
    shape = _parse_shape(image_shape)
    if parsed is None or shape is None:
        return None
    height, width = shape
    x1, y1, x2, y2 = parsed
    cx = (float(x1) + float(x2)) * 0.5
    cy = (float(y1) + float(y2)) * 0.5
    horizontal = "L" if cx < width * 0.5 else "R"
    vertical = "T" if cy < height * 0.5 else "B"
    return f"{horizontal}{vertical}"


def quadrant_to_roi(quadrant: Any, width: int, height: int) -> Optional[list[int]]:
    try:
        w = int(width)
        h = int(height)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    q = _normalize_quadrant(quadrant)
    if q is None:
        return None
    mid_x = w // 2
    mid_y = h // 2
    x1, x2 = (0, mid_x) if q.startswith("L") else (mid_x, w)
    y1, y2 = (0, mid_y) if q.endswith("T") else (mid_y, h)
    return _clip_roi([x1, y1, x2, y2], w, h)


def build_table_roi(
    local_perception: Any,
    rgb_shape: Any,
    depth_shape: Any,
    fallback_depth_roi: Any = None,
) -> Dict[str, Any]:
    """Build RGB/depth quadrant ROI metadata from a table bbox when available."""
    local = dict(local_perception or {}) if isinstance(local_perception, dict) else {}
    table_bbox = find_table_bbox(local)
    rgb_hw = _parse_shape(rgb_shape or local.get("rgb_shape"))
    depth_hw = _parse_shape(depth_shape)
    table_quadrant = bbox_to_quadrant(table_bbox, rgb_hw) if table_bbox is not None else None
    table_center = bbox_center_norm(table_bbox, rgb_hw) if table_bbox is not None else None
    if table_quadrant is None:
        table_quadrant = _normalize_quadrant(local.get("table_quadrant"))
    rgb_search_roi = None
    depth_edge_roi = None
    if table_quadrant is not None:
        if rgb_hw is not None:
            rgb_search_roi = quadrant_to_roi(table_quadrant, rgb_hw[1], rgb_hw[0])
        if depth_hw is not None:
            depth_edge_roi = quadrant_to_roi(table_quadrant, depth_hw[1], depth_hw[0])
    if depth_edge_roi is None:
        depth_edge_roi = _parse_bbox(fallback_depth_roi)
    source = "yolo_table_bbox" if table_bbox is not None else "fallback"
    return {
        "table_bbox": table_bbox,
        "table_center_norm": table_center,
        "table_quadrant": table_quadrant,
        "rgb_search_roi": rgb_search_roi,
        "depth_edge_roi": depth_edge_roi,
        "table_edge_roi": depth_edge_roi,
        "edge_roi": depth_edge_roi,
        "roi_source": source,
        "roi_format": "xyxy",
        "table_roi_source": source,
    }
