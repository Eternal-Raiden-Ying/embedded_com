#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np

from ..config.data import COCO80_CLASSES, normalize_class_name, normalize_class_names


def _center_priority(x1, y1, x2, y2, w, h):
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    dx = abs(cx - (w / 2.0)) / max(1.0, w / 2.0)
    dy = abs(cy - (h / 2.0)) / max(1.0, h / 2.0)
    dist = min(1.0, (dx * dx + dy * dy) ** 0.5)
    return 1.0 - dist


def _resolve_class_names(class_names) -> tuple:
    normalized = normalize_class_names(class_names)
    if normalized:
        return normalized
    return COCO80_CLASSES


def resolve_target_classes(target: str, class_names=None) -> set:
    """Resolve only catalog targets; unknown or retired names are rejected."""
    from common.target_catalog import resolve_target
    spec = resolve_target(target)
    if spec is None:
        return set()
    available = set(_resolve_class_names(class_names))
    return {spec.canonical_name} if spec.canonical_name in available else set()


def resolve_target_class_id(target: str):
    from common.target_catalog import target_to_class_id
    return target_to_class_id(target)

def _target_min_conf() -> float:
    try:
        return max(0.0, float(os.getenv("VISTA_TARGET_MIN_CONF", "0.25") or 0.25))
    except Exception:
        return 0.25


def _target_min_bbox_area() -> float:
    try:
        return max(0.0, float(os.getenv("VISTA_TARGET_MIN_BBOX_AREA", "0.001") or 0.001))
    except Exception:
        return 0.001


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _detection_rows(frame_shape, det_pred, class_names=None):
    h, w = frame_shape[:2]
    resolved_class_names = _resolve_class_names(class_names)
    detections = []
    rows = [] if det_pred is None else det_pred
    for original_rank, row in enumerate(rows, start=1):
        try:
            if len(row) < 6:
                continue
            x1, y1, x2, y2 = [float(v) for v in row[:4]]
            conf = float(row[4])
            cls_id = int(float(row[5]))
        except Exception:
            continue
        cls_name = ""
        try:
            cls_name = str(row[6]).strip() if len(row) > 6 else ""
        except Exception:
            cls_name = ""
        if not cls_name:
            cls_name = resolved_class_names[cls_id] if 0 <= cls_id < len(resolved_class_names) else str(cls_id)
        cls_name = normalize_class_name(cls_name)
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        area_norm = area / max(1.0, w * h)
        center_pri = _center_priority(x1, y1, x2, y2, w, h)
        detections.append(
            {
                "original_rank": int(original_rank),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "conf": conf,
                "cls_id": cls_id,
                "cls_name": cls_name,
                "area_norm": area_norm,
                "center_priority": center_pri,
            }
        )
    return sorted(
        detections,
        key=lambda item: (
            round(float(item["conf"]), 6),
            round(float(item["center_priority"]), 6),
            round(float(item["area_norm"]), 6),
        ),
        reverse=True,
    )


def _bbox_validity(x1: float, y1: float, x2: float, y2: float, w: float, h: float) -> tuple:
    values = (x1, y1, x2, y2, w, h)
    if not all(np.isfinite(float(v)) for v in values):
        return False, "bbox_non_finite"
    if x2 <= x1 or y2 <= y1:
        return False, "bbox_non_positive_area"
    if x1 < 0.0 or y1 < 0.0 or x2 > float(w) or y2 > float(h):
        return False, "bbox_out_of_frame"
    return True, ""


def _select_best_target_candidate(candidates):
    """Select one canonical target strictly by confidence, then bbox area."""
    return max(
        candidates,
        key=lambda candidate: (
            float(candidate["conf"]),
            float(candidate["area_norm"]),
        ),
    )


def compute_target_obs(
    frame_shape,
    target: str,
    det_pred,
    class_names=None,
    target_min_conf=None,
    min_bbox_area=None,
    use_roi_filter=None,
    prefer_center_region=None,
):
    h, w = frame_shape[:2]
    if det_pred is None or len(det_pred) == 0:
        return None
    valid_names = resolve_target_classes(target, class_names=class_names)
    if not valid_names:
        return None

    detections = _detection_rows(frame_shape, det_pred, class_names=class_names)
    if not detections:
        return None

    min_conf = _target_min_conf() if target_min_conf is None else float(target_min_conf)
    min_area = _target_min_bbox_area() if min_bbox_area is None else float(min_bbox_area)
    _ = prefer_center_region
    _ = use_roi_filter if use_roi_filter is not None else _env_bool("VISTA_TARGET_USE_ROI_FILTER", False)

    for rank, det in enumerate(detections, start=1):
        det["rank_in_all_boxes"] = int(rank)
    best = detections[0]
    candidates = [
        det
        for det in detections
        if det["cls_name"] in valid_names
        and float(det["conf"]) >= float(min_conf)
        and float(det["area_norm"]) >= float(min_area)
    ]

    if not candidates:
        return None

    matched = _select_best_target_candidate(candidates)
    x1 = float(matched["x1"])
    y1 = float(matched["y1"])
    x2 = float(matched["x2"])
    y2 = float(matched["y2"])
    conf = float(matched["conf"])
    center_pri = float(matched["center_priority"])
    area_norm = float(matched["area_norm"])
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    full_cx_norm = float(np.clip(cx / max(1.0, float(w)), 0.0, 1.0))
    full_cy_norm = float(np.clip(cy / max(1.0, float(h)), 0.0, 1.0))
    # Keep the historic control convention: positive means target is left of center.
    cx_offset_norm = float(np.clip((w / 2.0 - cx) / max(1.0, w / 2.0), -1.0, 1.0))
    cy_offset_norm = float(np.clip((h / 2.0 - cy) / max(1.0, h / 2.0), -1.0, 1.0))
    size_norm = ((x2 - x1) * (y2 - y1)) / float(max(1, w * h))
    bbox = [int(x1), int(y1), int(x2), int(y2)]
    bbox_valid, bbox_invalid_reason = _bbox_validity(x1, y1, x2, y2, float(w), float(h))
    matched_center_full_norm = {
        "cx": full_cx_norm,
        "cy": full_cy_norm,
    }
    matched_center_offset_norm = {
        "dx": cx_offset_norm,
        "dy": cy_offset_norm,
    }
    matched_center = {
        "cx": full_cx_norm,
        "cy": full_cy_norm,
        "x_norm": full_cx_norm,
        "y_norm": full_cy_norm,
        "cx_norm": full_cx_norm,
        "cy_norm": full_cy_norm,
    }
    all_candidate_classes = []
    for det in detections:
        name = str(det.get("cls_name") or "")
        if name and name not in all_candidate_classes:
            all_candidate_classes.append(name)
    from common.target_catalog import resolve_target
    target_spec = resolve_target(target)
    return {
        "target": target_spec.canonical_name if target_spec else str(target),
        "canonical_target": target_spec.canonical_name if target_spec else "",
        "model_class_name": matched["cls_name"],
        "class_id": int(matched["cls_id"]),
        "target_found": True,
        "matched_cls": matched["cls_name"],
        "matched_class_id": int(matched["cls_id"]),
        "matched_conf": float(conf),
        "matched_bbox": bbox,
        "matched_center": matched_center,
        "matched_center_full_norm": matched_center_full_norm,
        "matched_center_offset_norm": matched_center_offset_norm,
        "matched_area": float(np.clip(area_norm, 0.0, 1.0)),
        "matched_rank_in_all_boxes": int(matched["rank_in_all_boxes"]),
        "num_target_candidates": int(len(candidates)),
        "target_candidate_confidences": [float(item["conf"]) for item in candidates],
        "all_candidate_classes": all_candidate_classes,
        "best_cls": best["cls_name"],
        "best_class_id": int(best["cls_id"]),
        "best_conf": float(best["conf"]),
        "confidence": float(conf),
        "x_norm": full_cx_norm,
        "y_norm": full_cy_norm,
        "cx_norm": cx_offset_norm,
        "cy_norm": full_cy_norm,
        "size_norm": float(np.clip(size_norm, 0.0, 1.0)),
        "bbox": bbox,
        "bbox_valid": bool(bbox_valid),
        "bbox_invalid_reason": bbox_invalid_reason or None,
        "center_priority": float(np.clip(center_pri, 0.0, 1.0)),
        "area_norm": float(np.clip(area_norm, 0.0, 1.0)),
    }
