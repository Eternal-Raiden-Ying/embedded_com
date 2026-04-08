import cv2
import numpy as np
from ultralytics import YOLO, settings


def load_yolo_model(model_name_or_path, weights_dir=None):
    if weights_dir:
        settings.update({'weights_dir': weights_dir})
    return YOLO(model_name_or_path)


def expand_bbox_xyxy(x1, y1, x2, y2, image_width, image_height, scale=2.0):
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    width = max(1.0, float(x2 - x1))
    height = max(1.0, float(y2 - y1))
    new_width = width * scale
    new_height = height * scale

    new_x1 = int(np.floor(cx - new_width / 2.0))
    new_x2 = int(np.ceil(cx + new_width / 2.0))
    new_y1 = int(np.floor(cy - new_height / 2.0))
    new_y2 = int(np.ceil(cy + new_height / 2.0))

    new_x1 = int(np.clip(new_x1, 0, image_width))
    new_x2 = int(np.clip(new_x2, 0, image_width))
    new_y1 = int(np.clip(new_y1, 0, image_height))
    new_y2 = int(np.clip(new_y2, 0, image_height))
    return new_x1, new_y1, new_x2, new_y2


def mask_to_bbox_mask(mask, scale=2.0):
    bbox_mask = np.zeros_like(mask, dtype=np.uint8)
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return bbox_mask, None

    h, w = mask.shape[:2]
    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    x1, y1, x2, y2 = expand_bbox_xyxy(x1, y1, x2, y2, w, h, scale=scale)
    bbox_mask[y1:y2, x1:x2] = 1
    return bbox_mask, (x1, y1, x2, y2)


def predict_target_masks(model, bgr_image, class_id, conf=0.25, iou=0.7, bbox_scale=2.0):
    results = model.predict(
        source=bgr_image,
        classes=[class_id],
        conf=conf,
        iou=iou,
        retina_masks=True,
        save=False,
        verbose=False,
    )

    result = results[0]
    h, w = result.orig_img.shape[:2]
    seg_mask = np.zeros((h, w), dtype=np.uint8)
    bbox_mask = np.zeros((h, w), dtype=np.uint8)
    overlay_img = result.plot(boxes=True, labels=True)
    info = {
        'found': False,
        'confidence': None,
        'bbox': None,
        'class_id': class_id,
    }

    if result.boxes is None or len(result.boxes) == 0:
        return seg_mask, bbox_mask, overlay_img, info

    confs = result.boxes.conf.detach().cpu().numpy()
    best_idx = int(np.argmax(confs))
    xyxy = result.boxes.xyxy[best_idx].detach().cpu().numpy()
    x1, y1, x2, y2 = np.round(xyxy).astype(int)
    x1 = int(np.clip(x1, 0, w))
    x2 = int(np.clip(x2, 0, w))
    y1 = int(np.clip(y1, 0, h))
    y2 = int(np.clip(y2, 0, h))
    x1, y1, x2, y2 = expand_bbox_xyxy(x1, y1, x2, y2, w, h, scale=bbox_scale)
    if x2 > x1 and y2 > y1:
        bbox_mask[y1:y2, x1:x2] = 1

    if result.masks is not None and best_idx < len(result.masks.xy):
        pts = np.array(result.masks.xy[best_idx], dtype=np.int32)
        if pts.size > 0:
            cv2.fillPoly(seg_mask, [pts], 1)

    if seg_mask.sum() == 0:
        seg_mask = bbox_mask.copy()

    info.update({
        'found': True,
        'confidence': float(confs[best_idx]),
        'bbox': (x1, y1, x2, y2),
    })
    return seg_mask, bbox_mask, overlay_img, info
