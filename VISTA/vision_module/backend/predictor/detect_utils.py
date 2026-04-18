#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Iterable, Optional, Tuple

import cv2
import numpy as np


def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=False,
    scale_fill=False,
    scaleup=True,
    stride=32,
):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto and stride:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def preprocess_img(img, target_shape: Tuple[int, int]):
    processed = letterbox(img, target_shape, stride=None, auto=False)[0]
    processed = processed.astype(np.float32) / 255.0
    processed = processed[None, :]
    return np.ascontiguousarray(processed.astype(np.float32))


def clip_coords(boxes, img_shape):
    boxes[:, 0].clip(0, img_shape[1], out=boxes[:, 0])
    boxes[:, 1].clip(0, img_shape[0], out=boxes[:, 1])
    boxes[:, 2].clip(0, img_shape[1], out=boxes[:, 2])
    boxes[:, 3].clip(0, img_shape[0], out=boxes[:, 3])


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2,
        )
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> Iterable[int]:
    if len(boxes) <= 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1 + 1) * np.maximum(0.0, y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-6)
        remaining = np.where(iou <= float(iou_thres))[0]
        order = order[remaining + 1]
    return keep


def detect_postprocess(
    prediction: np.ndarray,
    image_shape: Tuple[int, ...],
    input_shape: Tuple[int, int],
    conf_thres: float,
    iou_thres: float,
) -> np.ndarray:
    if prediction.ndim == 3:
        prediction = prediction[0]
    if prediction.size <= 0:
        return np.empty((0, 6), dtype=np.float32)

    obj_scores = prediction[:, 4]
    candidates = prediction[obj_scores > float(conf_thres)]
    if candidates.size <= 0:
        return np.empty((0, 6), dtype=np.float32)

    class_scores = candidates[:, 5:]
    class_ids = np.argmax(class_scores, axis=1).astype(np.float32)
    best_cls_scores = class_scores[np.arange(class_scores.shape[0]), class_ids.astype(np.int32)]
    scores = obj_scores[obj_scores > float(conf_thres)] * best_cls_scores
    valid = scores > float(conf_thres)
    if not np.any(valid):
        return np.empty((0, 6), dtype=np.float32)

    boxes = xywh2xyxy(candidates[valid, :4])
    scores = scores[valid]
    class_ids = class_ids[valid]

    offsets = class_ids.reshape(-1, 1) * 4096.0
    keep = list(nms_indices(boxes + np.concatenate([offsets, offsets, offsets, offsets], axis=1), scores, iou_thres))
    if not keep:
        return np.empty((0, 6), dtype=np.float32)

    detections = np.concatenate([boxes, scores.reshape(-1, 1), class_ids.reshape(-1, 1)], axis=1)
    detections = detections[np.array(keep, dtype=np.int32)]
    detections[:, :4] = scale_coords(input_shape, detections[:, :4], image_shape[:2]).round()
    return detections.astype(np.float32)


def default_yolov7_anchors() -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    return (
        (12, 16, 19, 36, 40, 28),
        (36, 75, 76, 55, 72, 146),
        (142, 110, 192, 243, 459, 401),
    )


def default_yolov7_strides() -> Tuple[int, int, int]:
    return (8, 16, 32)


def normalize_anchors(raw: Optional[tuple]) -> Tuple[Tuple[int, ...], ...]:
    if not raw:
        return default_yolov7_anchors()
    out = []
    for level in tuple(raw):
        if isinstance(level, np.ndarray):
            level = level.tolist()
        out.append(tuple(int(v) for v in tuple(level)))
    return tuple(out)


def normalize_strides(raw: Optional[tuple]) -> Tuple[int, ...]:
    if not raw:
        return default_yolov7_strides()
    return tuple(int(v) for v in tuple(raw))
