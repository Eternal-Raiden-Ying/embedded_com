#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    import aidcv as cv2
except ImportError:
    import cv2
import numpy as np

from ..config.data import coco80, normalize_class_names


DEFAULT_CLASSES = normalize_class_names(coco80)
COLORS = {i: (0, int(i * (255 / max(1, len(DEFAULT_CLASSES)))), int(255 - i * (255 / max(1, len(DEFAULT_CLASSES))))) for i in range(len(DEFAULT_CLASSES))}


def _resolve_classes(class_names):
    resolved = normalize_class_names(class_names)
    if resolved:
        return resolved
    return DEFAULT_CLASSES


def draw_detect_res_fast(img_bgr, det_pred, masks, class_names=None):
    if det_pred is None or len(det_pred) == 0:
        return img_bgr

    resolved_classes = _resolve_classes(class_names)
    has_masks = isinstance(masks, list) and len(masks) >= len(det_pred)

    for i in range(len(det_pred)):
        x1, y1, x2, y2 = [int(t) for t in det_pred[i][:4]]
        cls_id = int(det_pred[i][5])
        color = COLORS.get(cls_id, (0, 255, 0))
        label = resolved_classes[cls_id] if 0 <= cls_id < len(resolved_classes) else str(cls_id)

        if has_masks:
            try:
                mask = np.asarray(masks[i]).astype(bool)
                img_bgr[mask] = img_bgr[mask] * 0.5 + np.array(color) * 0.5
            except Exception:
                pass

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness=2)
        cv2.putText(
            img_bgr,
            f"{label} {float(det_pred[i][4]):.2f}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return img_bgr
