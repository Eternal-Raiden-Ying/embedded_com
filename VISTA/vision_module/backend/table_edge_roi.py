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


def _bbox_touch_flags(bbox: Sequence[int], image_shape: Any, edge_margin_norm: float = 0.03) -> Dict[str, bool]:
    shape = _parse_shape(image_shape)
    parsed = _parse_bbox(bbox)
    if shape is None or parsed is None:
        return {
            "table_bbox_touch_left": False,
            "table_bbox_touch_right": False,
            "table_bbox_touch_bottom": False,
        }
    height, width = shape
    x1, y1, x2, y2 = _clip_bbox(parsed, image_shape)
    margin_x = max(1.0, float(width) * float(edge_margin_norm))
    margin_y = max(1.0, float(height) * float(edge_margin_norm))
    return {
        "table_bbox_touch_left": bool(float(x1) <= margin_x),
        "table_bbox_touch_right": bool(float(x2) >= float(width) - margin_x),
        "table_bbox_touch_bottom": bool(float(y2) >= float(height) - margin_y),
    }


def _roi_from_center(center_x: float, center_y: float, roi_width: int, roi_height: int, image_shape: Any) -> Optional[list[int]]:
    shape = _parse_shape(image_shape)
    if shape is None:
        return None
    height, width = shape
    cx = max(0.0, min(float(width), float(center_x)))
    cy = max(0.0, min(float(height), float(center_y)))
    x1 = int(round(cx - float(max(1, int(roi_width))) * 0.5))
    y1 = int(round(cy - float(max(1, int(roi_height))) * 0.5))
    return _clip_bbox([x1, y1, x1 + int(roi_width), y1 + int(roi_height)], image_shape)




def _extend_roi_to_depth_boundaries(
    roi: Any,
    depth_shape: Any,
    *,
    touch_left: bool = False,
    touch_right: bool = False,
    touch_bottom: bool = False,
) -> Optional[list[int]]:
    """Extend ROI outward to depth-frame boundaries for RGB-boundary table bboxes.

    This is intentionally simple: when the table bbox touches an RGB boundary,
    the corresponding side of the depth ROI is extended to the depth image
    boundary.  The opposite side remains unchanged.
    """
    parsed = _parse_bbox(roi)
    shape = _parse_shape(depth_shape)
    if parsed is None or shape is None:
        return None
    height, width = shape
    x1, y1, x2, y2 = [int(v) for v in parsed[:4]]
    if bool(touch_left):
        x1 = 0
    if bool(touch_right):
        x2 = int(width)
    if bool(touch_bottom):
        y2 = int(height)
    extended = _clip_bbox([x1, y1, x2, y2], depth_shape)
    return extended if extended != parsed else None


def _crop_rect_for_mapping(rgb_native_shape: Any, rgb_crop_rect: Any) -> Optional[list[float]]:
    native = _parse_shape(rgb_native_shape)
    if native is None:
        return None
    native_h, native_w = native
    rect = rgb_crop_rect
    if isinstance(rect, str):
        rect = [part.strip() for part in rect.replace(";", ",").split(",")]
    if not isinstance(rect, (list, tuple)) or len(rect) < 4:
        return [0.0, 0.0, float(native_w), float(native_h)]
    try:
        x = float(rect[0])
        y = float(rect[1])
        w = float(rect[2])
        h = float(rect[3])
    except (TypeError, ValueError):
        return [0.0, 0.0, float(native_w), float(native_h)]
    if w <= 0.0:
        w = float(native_w)
    if h <= 0.0:
        h = float(native_h)
    x = max(0.0, min(float(native_w), x))
    y = max(0.0, min(float(native_h), y))
    w = max(1.0, min(float(native_w) - x, w))
    h = max(1.0, min(float(native_h) - y, h))
    return [x, y, w, h]



def _safe_float(value: Any, default: float = 0.0, *, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    try:
        v = float(value)
    except Exception:
        v = float(default)
    if lo is not None:
        v = max(float(lo), v)
    if hi is not None:
        v = min(float(hi), v)
    return v


def map_rgb_bbox_to_depth_roi(
    bbox_rgb_xyxy: Any,
    rgb_output_shape: Any,
    rgb_native_shape: Any,
    rgb_crop_rect: Any,
    depth_shape: Any,
    roi_width: Any,
    roi_height: Any,
    *,
    use_rgb_depth_mapping: bool = True,
    scale_x: float = 0.50,
    scale_y: float = 0.50,
    rgb_depth_mapping_mode: str = "centered_scale",
    rgb_fov_in_depth_scale_x: float = 0.75,
    rgb_fov_in_depth_scale_y: float = 0.75,
    rgb_depth_center_offset_x: float = 0.0,
    rgb_depth_center_offset_y: float = 0.0,
    boundary_extend_enable: bool = True,
    boundary_margin_norm: float = 0.03,
    **_compat: Any,
) -> Tuple[Optional[list[int]], Dict[str, Any]]:
    """Map a YOLO table bbox from RGB output coordinates into a depth ROI.

    The ROI model is intentionally simple and centered:

    1. Convert RGB bbox points to normalized RGB output coordinates.
    2. Map normalized RGB coordinates to normalized depth coordinates.
       The default `centered_scale` mapping assumes RGB and depth centers match,
       but RGB only covers a centered sub-FOV of depth:

           depth_norm = 0.5 + offset + (rgb_norm - 0.5) * rgb_fov_in_depth_scale

       For example, scale=0.75 means the full RGB image corresponds to the
       central 75% of the depth FOV.  Use scale=1.0 when FOVs match.
    3. Use the mapped bbox center as ROI center.
    4. Set ROI width/height = mapped_bbox_width/height * scale_x/scale_y.

    There is deliberately no expand/boundary-extension/min-size user logic here;
    `scale_x` and `scale_y` are the only ROI-size knobs.
    """
    bbox = normalize_table_bbox(bbox_rgb_xyxy, rgb_output_shape)
    rgb_out_hw = _parse_shape(rgb_output_shape)
    depth_hw = _parse_shape(depth_shape)
    if bbox is None or rgb_out_hw is None or depth_hw is None:
        return None, {
            "roi_mapping_mode": "unavailable",
            "roi_clamped": False,
            "table_bbox_rgb_xyxy": bbox,
            "mapped_depth_bbox_xyxy": None,
        }

    rgb_out_h, rgb_out_w = rgb_out_hw
    depth_h, depth_w = depth_hw
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]

    roi_scale_x = _safe_float(scale_x, 0.50, lo=0.05, hi=2.0)
    roi_scale_y = _safe_float(scale_y, 0.50, lo=0.05, hi=2.0)
    fov_scale_x = _safe_float(rgb_fov_in_depth_scale_x, 0.75, lo=0.05, hi=2.0)
    fov_scale_y = _safe_float(rgb_fov_in_depth_scale_y, 0.75, lo=0.05, hi=2.0)
    center_off_x = _safe_float(rgb_depth_center_offset_x, 0.0, lo=-1.0, hi=1.0)
    center_off_y = _safe_float(rgb_depth_center_offset_y, 0.0, lo=-1.0, hi=1.0)
    mapping_mode = str(rgb_depth_mapping_mode or "centered_scale").strip().lower()
    if mapping_mode not in {"centered_scale", "identity"}:
        mapping_mode = "centered_scale"

    def map_point(px: float, py: float) -> tuple[float, float, float, float, float, float]:
        rgb_norm_x = max(0.0, min(1.0, px / max(1.0, float(rgb_out_w))))
        rgb_norm_y = max(0.0, min(1.0, py / max(1.0, float(rgb_out_h))))
        if bool(use_rgb_depth_mapping) and mapping_mode == "centered_scale":
            depth_norm_x = 0.5 + center_off_x + (rgb_norm_x - 0.5) * fov_scale_x
            depth_norm_y = 0.5 + center_off_y + (rgb_norm_y - 0.5) * fov_scale_y
        else:
            depth_norm_x = rgb_norm_x
            depth_norm_y = rgb_norm_y
        return (
            depth_norm_x * float(depth_w),
            depth_norm_y * float(depth_h),
            rgb_norm_x,
            rgb_norm_y,
            depth_norm_x,
            depth_norm_y,
        )

    d1x, d1y, _, _, _, _ = map_point(x1, y1)
    d2x, d2y, _, _, _, _ = map_point(x2, y2)
    dx1, dx2 = sorted((d1x, d2x))
    dy1, dy2 = sorted((d1y, d2y))
    mapped_bbox_unclipped = [int(round(dx1)), int(round(dy1)), int(round(dx2)), int(round(dy2))]
    mapped_bbox = _clip_bbox(mapped_bbox_unclipped, depth_shape)

    cx_out = (x1 + x2) * 0.5
    cy_out = (y1 + y2) * 0.5
    depth_cx, depth_cy, cx_norm_out, cy_norm_out, depth_cx_norm, depth_cy_norm = map_point(cx_out, cy_out)

    mapped_w = max(1.0, dx2 - dx1)
    mapped_h = max(1.0, dy2 - dy1)
    roi_w = max(1.0, mapped_w * roi_scale_x)
    roi_h = max(1.0, mapped_h * roi_scale_y)
    unclamped = [
        int(round(depth_cx - roi_w * 0.5)),
        int(round(depth_cy - roi_h * 0.5)),
        int(round(depth_cx + roi_w * 0.5)),
        int(round(depth_cy + roi_h * 0.5)),
    ]
    roi = _clip_bbox(unclamped, depth_shape)

    debug = {
        "rgb_shape": list(rgb_output_shape[:2]) if isinstance(rgb_output_shape, (list, tuple)) and len(rgb_output_shape) >= 2 else rgb_output_shape,
        "depth_shape": list(depth_shape[:2]) if isinstance(depth_shape, (list, tuple)) and len(depth_shape) >= 2 else depth_shape,
        "table_bbox_rgb_xyxy": [int(v) for v in bbox],
        "table_bbox_rgb_center": [float(cx_out), float(cy_out)],
        "table_bbox_rgb_center_norm": [float(cx_norm_out), float(cy_norm_out)],
        "mapped_depth_center": [float(depth_cx), float(depth_cy)],
        "mapped_depth_center_norm": [float(depth_cx_norm), float(depth_cy_norm)],
        "mapped_depth_bbox_unclipped_xyxy": mapped_bbox_unclipped,
        "mapped_depth_bbox_xyxy": mapped_bbox,
        "table_edge_roi": roi,
        "roi_mapping_mode": "rgb_depth_centered_scale" if bool(use_rgb_depth_mapping) and mapping_mode == "centered_scale" else "rgb_depth_identity",
        "roi_clamped": bool(roi != unclamped),
        "roi_scale_x": float(roi_scale_x),
        "roi_scale_y": float(roi_scale_y),
        "rgb_depth_mapping_mode": mapping_mode,
        "rgb_fov_in_depth_scale_x": float(fov_scale_x),
        "rgb_fov_in_depth_scale_y": float(fov_scale_y),
        "rgb_depth_center_offset_x": float(center_off_x),
        "rgb_depth_center_offset_y": float(center_off_y),
    }
    return roi, debug

def _follow_table_lower_band_roi(
    static_roi: Sequence[int],
    center_x: float,
    bbox: Sequence[int],
    image_shape: Any,
    bbox_shape: Any,
    *,
    bottom_align: bool = False,
) -> Optional[list[int]]:
    shape = _parse_shape(image_shape)
    bbox_hw = _parse_shape(bbox_shape)
    parsed_roi = _parse_bbox(static_roi)
    parsed_bbox = _parse_bbox(bbox)
    if shape is None or bbox_hw is None or parsed_roi is None or parsed_bbox is None:
        return None
    height, width = shape
    bbox_h, bbox_w = bbox_hw
    roi_w = max(1, int(parsed_roi[2]) - int(parsed_roi[0]))
    roi_h = max(1, int(parsed_roi[3]) - int(parsed_roi[1]))
    x1, y1, x2, y2 = [float(v) for v in parsed_bbox[:4]]
    y1 = max(0.0, min(float(bbox_h), y1))
    y2 = max(0.0, min(float(bbox_h), y2))
    if y2 < y1:
        y1, y2 = y2, y1
    bbox_bottom_depth = (y2 / max(1.0, float(bbox_h))) * float(height)
    bbox_lower_center_depth = ((y1 + (y2 - y1) * 0.75) / max(1.0, float(bbox_h))) * float(height)
    center_y = bbox_bottom_depth - float(roi_h) * 0.5 if bottom_align else bbox_lower_center_depth
    new_x1 = int(round(max(0.0, min(float(width), float(center_x))) - roi_w * 0.5))
    new_y1 = int(round(center_y - roi_h * 0.5))
    return _clip_bbox([new_x1, new_y1, new_x1 + roi_w, new_y1 + roi_h], image_shape)


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
    smoothed_center_x: Any = None,
    bbox_image_shape: Any = None,
    yolo_near_bottom_norm: float = 0.60,
    rgb_native_shape: Any = None,
    rgb_crop_rect: Any = None,
    use_rgb_depth_mapping: bool = True,
    roi_mode: str = "centered_bbox_scale",
    roi_scale_x: float = 0.50,
    roi_scale_y: float = 0.50,
    rgb_depth_mapping_mode: str = "centered_scale",
    rgb_fov_in_depth_scale_x: float = 0.75,
    rgb_fov_in_depth_scale_y: float = 0.75,
    rgb_depth_center_offset_x: float = 0.0,
    rgb_depth_center_offset_y: float = 0.0,
    boundary_extend_enable: bool = True,
    boundary_margin_norm: float = 0.03,
    **_compat: Any,
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
    bbox_shape = bbox_image_shape or image_shape
    bbox = normalize_table_bbox(table_bbox_xyxy, bbox_shape)
    if bbox is None:
        return {
            "dynamic_roi": static_roi,
            "roi_source": "static_no_yolo_fallback",
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
    touch = _bbox_touch_flags(bbox, bbox_shape, edge_margin_norm=_safe_float(boundary_margin_norm, 0.03, lo=0.0, hi=0.25))
    touch_bottom = bool(touch.get("table_bbox_touch_bottom", False))
    touch_left = bool(touch.get("table_bbox_touch_left", False))
    touch_right = bool(touch.get("table_bbox_touch_right", False))
    bbox_shape_hw = _parse_shape(bbox_shape)
    bbox_h = bbox_shape_hw[0] if bbox_shape_hw is not None else 0
    try:
        bbox_y1_norm = max(0.0, min(1.0, float(bbox[1]) / max(1.0, float(bbox_h))))
        bbox_y2_norm = max(0.0, min(1.0, float(bbox[3]) / max(1.0, float(bbox_h))))
    except Exception:
        bbox_y1_norm = None
        bbox_y2_norm = None
    bbox_bottom_norm = bbox_y2_norm
    if reason:
        return {
            "dynamic_roi": static_roi,
            "roi_source": "static_no_yolo_fallback",
            "roi_reason": reason,
            "bbox_valid": False,
            "bbox_reject_reason": reason,
            "yolo_bbox_area_ratio": area_ratio,
            "yolo_table_conf": score,
            **touch,
            "table_bbox_boundary_allowed": False,
            "yolo_table_roi_valid": False,
            "bbox_y1_norm": bbox_y1_norm,
            "bbox_y2_norm": bbox_y2_norm,
            "bbox_bottom_norm": bbox_bottom_norm,
            "bbox_touch_bottom": touch_bottom,
        }

    shape = _parse_shape(image_shape)
    bbox_hw = _parse_shape(bbox_shape)
    height, width = shape if shape is not None else (0, 0)
    bbox_width = bbox_hw[1] if bbox_hw is not None else width
    bbox_height = bbox_hw[0] if bbox_hw is not None else height
    raw_center_x = (float(bbox[0]) + float(bbox[2])) * 0.5
    raw_center_y = (float(bbox[1]) + float(bbox[3])) * 0.5
    raw_center_norm = raw_center_x / max(1.0, float(bbox_width))
    raw_center_y_norm = raw_center_y / max(1.0, float(bbox_height))
    roi_w = int(roi_width) if roi_width is not None else int(static_roi[2] - static_roi[0])
    roi_h = int(roi_height) if roi_height is not None else int(static_roi[3] - static_roi[1])
    roi, mapping_debug = map_rgb_bbox_to_depth_roi(
        bbox,
        bbox_shape,
        rgb_native_shape,
        rgb_crop_rect,
        image_shape,
        roi_w,
        roi_h,
        use_rgb_depth_mapping=use_rgb_depth_mapping,
        scale_x=roi_scale_x,
        scale_y=roi_scale_y,
        rgb_depth_mapping_mode=rgb_depth_mapping_mode,
        rgb_fov_in_depth_scale_x=rgb_fov_in_depth_scale_x,
        rgb_fov_in_depth_scale_y=rgb_fov_in_depth_scale_y,
        rgb_depth_center_offset_x=rgb_depth_center_offset_x,
        rgb_depth_center_offset_y=rgb_depth_center_offset_y,
    )
    roi = roi or static_roi
    if smoothed_center_x is not None and shape is not None:
        # Keep the bbox-size-driven ROI width/height.  Older code reused the
        # static ROI size here, which made ROI height appear fixed even when the
        # mapped YOLO bbox changed.
        try:
            center_y = (float(roi[1]) + float(roi[3])) * 0.5
            cur_w = max(1, int(round(float(roi[2]) - float(roi[0]))))
            cur_h = max(1, int(round(float(roi[3]) - float(roi[1]))))
            roi = _roi_from_center(float(smoothed_center_x), center_y, cur_w, cur_h, image_shape) or roi
            mapping_debug["mapped_depth_center"] = [
                float((float(roi[0]) + float(roi[2])) * 0.5),
                float(center_y),
            ]
            mapping_debug["table_edge_roi"] = roi
            mapping_debug["roi_center_x_smoothed"] = True
        except Exception:
            pass
    boundary_extended_roi = _extend_roi_to_depth_boundaries(
        roi,
        image_shape,
        touch_left=touch_left,
        touch_right=touch_right,
        touch_bottom=touch_bottom,
    ) if bool(boundary_extend_enable) else None
    boundary_axes = []
    if touch_left:
        boundary_axes.append("left")
    if touch_right:
        boundary_axes.append("right")
    if touch_bottom:
        boundary_axes.append("bottom")

    roi_source = "yolo_table_bbox_mapped"
    roi_reason = "table_bbox_rgb_depth_mapped_bbox_size_driven"
    roi_phase = "edge_fusion" if str(edge_stability or "").lower() == "stable" else "far_yolo_guided"
    if mode_text in {"near", "near_distance", "edge_lost_yolo", "edge_lost_yolo_assist", "near_yolo_assist"}:
        roi_phase = "near_yolo_assist"
    roi_y_strategy = "bbox_center"
    roi_center_y = ((float(roi[1]) + float(roi[3])) * 0.5) if roi is not None else None
    return {
        "dynamic_roi": roi,
        "roi_source": roi_source,
        "roi_reason": roi_reason,
        "roi_phase": roi_phase,
        "bbox_valid": True,
        "bbox_reject_reason": "",
        "yolo_bbox_area_ratio": area_ratio,
        "yolo_table_conf": score,
        **touch,
        "table_bbox_boundary_allowed": bool(touch_left or touch_right or touch_bottom),
        "boundary_extend_enabled": bool(boundary_extend_enable),
        "boundary_extend_candidate": bool(boundary_extended_roi is not None),
        "boundary_extended_roi": boundary_extended_roi,
        "boundary_extend_touch_axes": boundary_axes,
        "boundary_margin_norm": float(_safe_float(boundary_margin_norm, 0.03, lo=0.0, hi=0.25)),
        "yolo_table_roi_valid": True,
        "bbox_y1_norm": bbox_y1_norm,
        "bbox_y2_norm": bbox_y2_norm,
        "bbox_bottom_norm": bbox_bottom_norm,
        "bbox_touch_bottom": touch_bottom,
        "roi_center_y_from_yolo": roi_center_y,
        "roi_y_strategy": roi_y_strategy,
        **mapping_debug,
        "yolo_table_class_id": int(table_class_id),
        "yolo_bbox_center_x": raw_center_x,
        "yolo_bbox_center_x_norm": raw_center_norm,
        "yolo_bbox_center_y": raw_center_y,
        "yolo_bbox_center_y_norm": raw_center_y_norm,
        "yolo_roi_center_x": (float(roi[0]) + float(roi[2])) * 0.5,
        "yolo_roi_center_x_norm": ((float(roi[0]) + float(roi[2])) * 0.5) / max(1.0, float(width)),
        "edge_stability": edge_stability,
        "distance_hint": distance_hint,
        "roi_width": int(roi[2] - roi[0]) if roi is not None else int(static_roi[2] - static_roi[0]),
        "roi_height": int(roi[3] - roi[1]) if roi is not None else int(static_roi[3] - static_roi[1]),
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
    smoothed_table_center_x: Any = None,
    near_distance: bool = False,
    yolo_near_bottom_norm: float = 0.60,
    edge_stable: bool = False,
    rgb_native_shape: Any = None,
    rgb_crop_rect: Any = None,
    yolo_roi_use_rgb_depth_mapping: bool = True,
    yolo_roi_mode: str = "centered_bbox_scale",
    yolo_roi_scale_x: float = 0.50,
    yolo_roi_scale_y: float = 0.50,
    rgb_depth_mapping_mode: str = "centered_scale",
    rgb_fov_in_depth_scale_x: float = 0.75,
    rgb_fov_in_depth_scale_y: float = 0.75,
    rgb_depth_center_offset_x: float = 0.0,
    rgb_depth_center_offset_y: float = 0.0,
    yolo_table_roi_boundary_extend_enable: bool = True,
    yolo_table_roi_boundary_margin_norm: float = 0.03,
    last_valid_depth_roi: Any = None,
    yolo_table_bbox_hold_enable: bool = True,
    yolo_table_bbox_hold_frames: int = 8,
    table_bbox_hold_age_frames: int = 0,
    yolo_table_roi_hold_enable: bool = True,
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
    elif preset_roi is not None:
        depth_edge_roi = preset_roi
        roi_source = f"preset:{preset_name}"
        roi_reason = "debug_roi_preset"
    elif yolo_dynamic_enable and fallback_roi is not None and table_bbox is not None:
        det = table_detection_debug(local, resolved_rgb_shape, min_conf=-1.0)
        dyn = compute_dynamic_table_roi_from_yolo_bbox(
            depth_shape,
            table_bbox,
            fallback_roi,
            mode="near" if near_distance else "",
            edge_stability="stable" if edge_stable else "unstable",
            bbox_score=det.get("conf"),
            class_id=None,
            table_class_id=yolo_table_class_id,
            conf_min=-1.0,
            smoothed_center_x=smoothed_table_center_x,
            bbox_image_shape=resolved_rgb_shape,
            yolo_near_bottom_norm=yolo_near_bottom_norm,
            rgb_native_shape=rgb_native_shape or local.get("rgb_native_shape"),
            rgb_crop_rect=rgb_crop_rect or local.get("rgb_crop_rect"),
            use_rgb_depth_mapping=yolo_roi_use_rgb_depth_mapping,
            roi_mode=yolo_roi_mode,
            roi_scale_x=yolo_roi_scale_x,
            roi_scale_y=yolo_roi_scale_y,
            rgb_depth_mapping_mode=rgb_depth_mapping_mode,
            rgb_fov_in_depth_scale_x=rgb_fov_in_depth_scale_x,
            rgb_fov_in_depth_scale_y=rgb_fov_in_depth_scale_y,
            rgb_depth_center_offset_x=rgb_depth_center_offset_x,
            rgb_depth_center_offset_y=rgb_depth_center_offset_y,
            boundary_extend_enable=yolo_table_roi_boundary_extend_enable,
            boundary_margin_norm=yolo_table_roi_boundary_margin_norm,
        )
        depth_edge_roi = dyn.get("dynamic_roi") or fallback_roi
        roi_source = str(dyn.get("roi_source") or "static_no_yolo_fallback")
        roi_reason = str(dyn.get("roi_reason") or "")
        if not bool(dyn.get("bbox_valid", False)) and preset_roi is not None:
            depth_edge_roi = preset_roi
            roi_source = f"preset:{preset_name}"
            roi_reason = "debug_roi_preset"
    elif (
        yolo_dynamic_enable
        and bool(yolo_table_bbox_hold_enable)
        and bool(yolo_table_roi_hold_enable)
        and table_bbox is None
        and int(table_bbox_hold_age_frames or 0) < max(1, int(yolo_table_bbox_hold_frames or 1))
        and normalize_table_bbox(last_valid_depth_roi, depth_shape) is not None
    ):
        # Short-term hold: keep the last YOLO-driven depth ROI steady when YOLO
        # drops for a few frames.  This prevents the ROI from jumping to a
        # quadrant/static fallback during momentary detector loss.
        depth_edge_roi = normalize_table_bbox(last_valid_depth_roi, depth_shape)
        table_bbox = normalize_table_bbox(last_valid_table_bbox, resolved_rgb_shape)
        table_center = last_valid_table_center_norm
        quadrant = _normalize_quadrant(last_valid_quadrant)
        roi_source = "yolo_table_bbox_hold"
        roi_reason = "table_bbox_lost_short_hold_last_yolo_roi"
    elif yolo_dynamic_enable and table_bbox is None:
        # Hold expired or no previous YOLO ROI: do not fall back to a static or
        # quadrant ROI for docking.  With no table bbox, the higher-level system
        # should enter/search table mode instead of trusting edge geometry.
        depth_edge_roi = None
        rgb_search_roi = None
        roi_source = "disabled_no_table_bbox"
        roi_reason = "table_bbox_lost_hold_expired"
    elif quadrant is not None and table_bbox is not None:
        depth_edge_roi = quadrant_to_depth_roi(quadrant, depth_shape, fallback_roi)
        rgb_hw = _parse_shape(resolved_rgb_shape)
        if rgb_hw is not None:
            rgb_search_roi = quadrant_to_roi(quadrant, rgb_hw[1], rgb_hw[0])
        roi_source = "local_perception_table_bbox"
        roi_reason = "table_bbox_detected"
    else:
        depth_edge_roi = fallback_roi
        if preset_roi is not None:
            roi_source = f"preset:{preset_name}"
            roi_reason = "debug_roi_preset"
        elif table_bbox is None:
            roi_source = "static_no_yolo_fallback"
            roi_reason = "table_bbox_unavailable"
        else:
            roi_source = "static_far_fallback"
            roi_reason = roi_reason or "static_depth_roi_fallback"

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
        "roi_phase": "static_fallback",
        "table_bbox_current_found": bool(normalize_table_bbox(find_table_bbox(local), resolved_rgb_shape) is not None),
        "table_bbox_control_valid": bool(table_bbox is not None or roi_source == "yolo_table_bbox_hold"),
        "table_bbox_hold_active": bool(roi_source == "yolo_table_bbox_hold"),
        "table_bbox_hold_age_frames": int(table_bbox_hold_age_frames or 0) if roi_source == "yolo_table_bbox_hold" else 0,
        "roi_hold_active": bool(roi_source == "yolo_table_bbox_hold"),
        "roi_hold_age_frames": int(table_bbox_hold_age_frames or 0) if roi_source == "yolo_table_bbox_hold" else 0,
        "last_valid_table_edge_roi": normalize_table_bbox(last_valid_depth_roi, depth_shape),
        "yolo_table_bbox_hold_enable": bool(yolo_table_bbox_hold_enable),
        "yolo_table_bbox_hold_frames": int(yolo_table_bbox_hold_frames or 0),
        "yolo_table_roi_hold_enable": bool(yolo_table_roi_hold_enable),
        "yolo_table_roi_scale_x": float(yolo_roi_scale_x or 1.0),
        "yolo_table_roi_scale_y": float(yolo_roi_scale_y or 1.0),
        "rgb_depth_mapping_mode": str(rgb_depth_mapping_mode or "centered_scale"),
        "rgb_fov_in_depth_scale_x": float(rgb_fov_in_depth_scale_x or 0.75),
        "rgb_fov_in_depth_scale_y": float(rgb_fov_in_depth_scale_y or 0.75),
        "rgb_depth_center_offset_x": float(rgb_depth_center_offset_x or 0.0),
        "rgb_depth_center_offset_y": float(rgb_depth_center_offset_y or 0.0),
        "yolo_table_roi_boundary_extend_enable": bool(yolo_table_roi_boundary_extend_enable),
        "yolo_table_roi_boundary_margin_norm": float(yolo_table_roi_boundary_margin_norm or 0.0),
        **(dyn if "dyn" in locals() and isinstance(dyn, dict) else {}),
    }
