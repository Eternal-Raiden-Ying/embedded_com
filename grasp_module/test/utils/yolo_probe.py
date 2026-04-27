import cv2
import numpy as np

from grasp_module.backend.utils.yolo_utils import expand_bbox_xyxy, load_yolo_model


def load_probe_model(model_name_or_path, weights_dir=None):
    return load_yolo_model(model_name_or_path, weights_dir)


def _predict_class_candidates(model, bgr_image, class_id, conf, iou):
    results = model.predict(
        source=bgr_image,
        classes=[int(class_id)],
        conf=conf,
        iou=iou,
        retina_masks=True,
        save=False,
        verbose=False,
    )
    return results[0]


def probe_single_class(model, bgr_image, class_id, conf=0.25, iou=0.7, bbox_scale=2.0):
    result = _predict_class_candidates(model, bgr_image, class_id, conf, iou)
    h, w = result.orig_img.shape[:2]
    seg_mask = np.zeros((h, w), dtype=np.uint8)
    bbox_mask = np.zeros((h, w), dtype=np.uint8)
    overlay_img = result.plot(boxes=True, labels=True)

    info = {
        "found": False,
        "class_id": int(class_id),
        "confidence": None,
        "bbox": None,
        "count": 0,
        "multiple_detections": False,
    }

    if result.boxes is None or len(result.boxes) == 0:
        return {
            "class_id": int(class_id),
            "count": 0,
            "best_conf": None,
            "multiple_detections": False,
            "seg_mask": seg_mask,
            "bbox_mask": bbox_mask,
            "overlay_img": overlay_img,
            "info": info,
        }

    confs = result.boxes.conf.detach().cpu().numpy()
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    best_idx = int(np.argmax(confs))
    x1, y1, x2, y2 = np.round(xyxy[best_idx]).astype(int)
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

    best_conf = float(confs[best_idx])
    info.update(
        {
            "found": True,
            "confidence": best_conf,
            "bbox": (x1, y1, x2, y2),
            "count": int(len(confs)),
            "multiple_detections": int(len(confs)) > 1,
        }
    )
    return {
        "class_id": int(class_id),
        "count": int(len(confs)),
        "best_conf": best_conf,
        "multiple_detections": int(len(confs)) > 1,
        "seg_mask": seg_mask,
        "bbox_mask": bbox_mask,
        "overlay_img": overlay_img,
        "info": info,
    }


def resolve_detection_route(model, bgr_image, primary_class_id, fallback_class_ids=None, conf=0.10, iou=0.7, bbox_scale=2.0):
    fallback_class_ids = fallback_class_ids or []
    candidate_ids = [int(primary_class_id)]
    for class_id in fallback_class_ids:
        if int(class_id) not in candidate_ids:
            candidate_ids.append(int(class_id))

    inspection = []
    for class_id in candidate_ids:
        probe = probe_single_class(model, bgr_image, class_id, conf=conf, iou=iou, bbox_scale=bbox_scale)
        inspection.append(
            {
                "class_id": int(class_id),
                "count": probe["count"],
                "best_conf": probe["best_conf"],
                "multiple_detections": probe["multiple_detections"],
            }
        )
        if probe["count"] > 0:
            return {
                "resolved_class_id": int(class_id),
                "resolved_conf": float(conf),
                "used_fallback": int(class_id) != int(primary_class_id),
                "candidate_inspection": inspection,
                "multiple_detections": probe["multiple_detections"],
                "detection_count": probe["count"],
                "best_conf": probe["best_conf"],
                "seg_mask": probe["seg_mask"],
                "bbox_mask": probe["bbox_mask"],
                "overlay_img": probe["overlay_img"],
                "info": probe["info"],
            }

    return {
        "resolved_class_id": int(primary_class_id),
        "resolved_conf": float(conf),
        "used_fallback": False,
        "candidate_inspection": inspection,
        "multiple_detections": False,
        "detection_count": 0,
        "best_conf": None,
        "seg_mask": None,
        "bbox_mask": None,
        "overlay_img": bgr_image.copy(),
        "info": {
            "found": False,
            "class_id": int(primary_class_id),
            "confidence": None,
            "bbox": None,
            "count": 0,
            "multiple_detections": False,
        },
    }


def choose_detection_frame(probed_frames):
    single_detection = next((item for item in probed_frames if item["route"]["detection_count"] == 1), None)
    if single_detection is not None:
        return single_detection, "single_detection"

    multi_detection = [item for item in probed_frames if item["route"]["detection_count"] > 1]
    if multi_detection:
        best_multi = max(multi_detection, key=lambda item: item["route"]["best_conf"] or -1.0)
        return best_multi, "best_multi_detection"

    return None, "no_detection"
