#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from ..config.data import ASR_VOCAB_MAP, COCO80_CLASSES, normalize_class_name, normalize_class_names


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


def compute_target_obs(frame_shape, target: str, det_pred, class_names=None):
    h, w = frame_shape[:2]
    if det_pred is None or len(det_pred) == 0:
        return None
    normalized_target = normalize_class_name(target)
    valid_names = set(ASR_VOCAB_MAP.get(normalized_target, set()))
    if not valid_names:
        return None

    resolved_class_names = _resolve_class_names(class_names)
    candidates = []
    for row in det_pred:
        x1, y1, x2, y2 = [float(v) for v in row[:4]]
        conf = float(row[4])
        cls_id = int(row[5])
        cls_name = resolved_class_names[cls_id] if 0 <= cls_id < len(resolved_class_names) else str(cls_id)
        if normalize_class_name(cls_name) not in valid_names:
            continue
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        area_norm = area / max(1.0, w * h)
        center_pri = _center_priority(x1, y1, x2, y2, w, h)
        rank_key = (round(conf, 6), round(center_pri, 6), round(area_norm, 6))
        candidates.append((rank_key, x1, y1, x2, y2, conf, center_pri, area_norm))

    if not candidates:
        return None

    _, x1, y1, x2, y2, conf, center_pri, area_norm = max(candidates, key=lambda item: item[0])
    cx = (x1 + x2) / 2.0
    cx_norm = (w / 2.0 - cx) / (w / 2.0)
    cx_norm = float(np.clip(cx_norm, -1.0, 1.0))
    size_norm = ((x2 - x1) * (y2 - y1)) / float(max(1, w * h))
    return {
        "target": target,
        "confidence": float(conf),
        "cx_norm": float(cx_norm),
        "size_norm": float(np.clip(size_norm, 0.0, 1.0)),
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "center_priority": float(np.clip(center_pri, 0.0, 1.0)),
        "area_norm": float(np.clip(area_norm, 0.0, 1.0)),
    }
