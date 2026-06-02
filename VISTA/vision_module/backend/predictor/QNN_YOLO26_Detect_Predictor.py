#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import threading
from typing import List, Tuple

import aidlite
import cv2
import numpy as np

from .base import IPredictor


logger = logging.getLogger("vision.inference")


def _yolo26s_preprocess(image: np.ndarray, input_size: int = 640) -> Tuple[np.ndarray, float]:
    """Square-fill preprocess for yolo26s.

    Returns (input_tensor, scale) where scale = max(h, w) / input_size.
    Short-circuits when the image is already ``input_size × input_size``.
    """
    height, width = image.shape[:2]
    length = max(height, width)
    scale = length / float(input_size)

    # Short-circuit: already square at target size — skip canvas + resize
    if height == width == input_size:
        canvas = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if canvas.dtype != np.float32:
            canvas = canvas.astype(np.float32) / 255.0
        return canvas[None, :], scale

    canvas = np.zeros((length, length, 3), dtype=np.uint8)
    canvas[:height, :width] = image
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    canvas = cv2.resize(canvas, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    return (canvas.astype(np.float32) / 255.0)[None, :], scale


def _yolo26s_merge_outputs(bbox_out: np.ndarray, class_out: np.ndarray, class_num: int = 80) -> np.ndarray:
    """Merge the two cutoff-model outputs into a single (8400, 4+class_num) array.

    bbox_out:  flat (33600,) or [1, 4, 8400]  bounding-box deltas (cx, cy, w, h)
    class_out: flat (672000,) or [1, N, 8400]  class logits for N classes
    Returns:   [8400, 4+N]                     merged prediction rows
    """
    bbox = np.asarray(bbox_out, dtype=np.float32)
    cls = np.asarray(class_out, dtype=np.float32)
    # Reshape flat tensors to (1, channels, 8400) then take batch[0]
    if bbox.ndim == 1:
        bbox = bbox.reshape(1, 4, 8400)
    if cls.ndim == 1:
        cls = cls.reshape(1, class_num, 8400)
    # (1, C, 8400) → (8400, C)
    bbox = bbox[0].T
    cls = cls[0].T
    return np.hstack([bbox, cls])


def _yolo26s_nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest]); yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest]); yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-12)
        order = rest[iou <= iou_thres]
    return keep


def _yolo26s_postprocess(
    prediction: np.ndarray,
    original_shape: Tuple[int, ...],
    scale: float,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
) -> np.ndarray:
    """Return N×6 [x1, y1, x2, y2, score, class_id].

    prediction: (8400, 4+class_num) — merged bbox + class scores.
    """
    prediction = np.asarray(prediction)
    if prediction.ndim != 2:
        raise ValueError(f"Expected 2D prediction, got shape {prediction.shape}")

    class_conf = prediction[:, 4:]          # (8400, class_num)
    class_ids = class_conf.argmax(axis=1)    # (8400,)
    scores = class_conf[np.arange(class_conf.shape[0]), class_ids]
    mask = scores >= conf_thres
    if not np.any(mask):
        return np.empty((0, 6), dtype=np.float32)

    boxes = prediction[mask, :4].copy()      # cx cy w h
    # cx cy w h → x1 y1 x2 y2
    boxes[:, 0] -= boxes[:, 2] / 2.0
    boxes[:, 1] -= boxes[:, 3] / 2.0
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    boxes *= scale
    boxes[:, 0] = boxes[:, 0].clip(0, original_shape[1])
    boxes[:, 1] = boxes[:, 1].clip(0, original_shape[0])
    boxes[:, 2] = boxes[:, 2].clip(0, original_shape[1])
    boxes[:, 3] = boxes[:, 3].clip(0, original_shape[0])

    scores = scores[mask]
    class_ids = class_ids[mask]

    detections: List[np.ndarray] = []
    for class_id in np.unique(class_ids):
        class_mask = class_ids == class_id
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        keep = _yolo26s_nms(class_boxes, class_scores, iou_thres)[:max_det]
        if keep:
            rows = np.column_stack((
                class_boxes[keep],
                class_scores[keep],
                np.full(len(keep), class_id, dtype=np.float32),
            ))
            detections.append(rows.astype(np.float32))

    if not detections:
        return np.empty((0, 6), dtype=np.float32)
    output = np.concatenate(detections, axis=0)
    output = output[output[:, 4].argsort()[::-1]]
    return output[:max_det]


def _resolve_post_process_path(target_model: str) -> str:
    """Return the companion post_process.onnx path next to the QNN model."""
    model_dir = os.path.dirname(os.path.abspath(str(target_model)))
    candidate = os.path.join(model_dir, "post_process.onnx")
    if os.path.isfile(candidate):
        return candidate
    return ""


class QNN_YOLO26_Detect_Predictor(IPredictor):
    """QNN YOLO26s detector predictor (anchor-free, cutoff model with dual outputs).

    The cutoff model emits two tensors:
      - output[0]: [1, 4, 8400]   bounding-box deltas (cx, cy, w, h)
      - output[1]: [1, N, 8400]    class logits for N classes

    These are merged and post-processed into the standard N×6 format:
      [x1, y1, x2, y2, score, class_id]

    Runtime contract (same as yoloV7):
    - input:  BGR frame as HWC numpy array
    - output: N×6 numpy array in [x1, y1, x2, y2, score, class_id]
    """

    def __init__(self, args) -> None:
        self._lock = threading.RLock()
        self.interpreter = None
        self._post_proc_onnx: str = ""

        backend_name = str(getattr(args, "model_backend", "qnn") or "qnn").strip().lower()
        config = aidlite.Config.create_instance()
        config.implement_type = aidlite.ImplementType.TYPE_LOCAL
        if backend_name in {"snpe", "snpe2"}:
            config.framework_type = aidlite.FrameworkType.TYPE_SNPE2
        else:
            config.framework_type = aidlite.FrameworkType.TYPE_QNN
        config.accelerate_type = aidlite.AccelerateType.TYPE_DSP
        config.is_quantify_model = 1

        target_model = str(getattr(args, "target_model", "") or "")
        model = aidlite.Model.create_instance(target_model)

        self.conf = float(getattr(args, "conf_thres", 0.25))
        self.iou = float(getattr(args, "iou_thres", 0.45))
        self.width = int(getattr(args, "width", 640))
        self.height = int(getattr(args, "height", 640))
        self.class_num = int(getattr(args, "class_num", 80))

        # Cutoff model: two output tensors
        #   output[0]: boxes  [1, 4, 8400]
        #   output[1]: scores [1, class_num, 8400]
        self.input_shape = [[1, self.height, self.width, 3]]
        self.output_shapes = [
            [1, 4, 8400],
            [1, self.class_num, 8400],
        ]

        model.set_model_properties(
            self.input_shape,
            aidlite.DataType.TYPE_FLOAT32,
            self.output_shapes,
            aidlite.DataType.TYPE_FLOAT32,
        )
        interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(model, config)
        interpreter.init()
        interpreter.load_model()
        self.interpreter = interpreter
        logger.info(
            "qnn yolo26s detect predictor loaded (cutoff, %d outputs): %s | backend=%s",
            len(self.output_shapes),
            target_model,
            backend_name,
        )

        # Resolve companion post_process.onnx path (optional, for diagnostics)
        self._post_proc_onnx = _resolve_post_process_path(target_model)
        if self._post_proc_onnx:
            logger.info("yolo26s post_process.onnx found: %s", self._post_proc_onnx)

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def is_ready(self) -> bool:
        with self._lock:
            return self.interpreter is not None

    def release(self) -> None:
        with self._lock:
            interpreter = getattr(self, "interpreter", None)
            if interpreter is None:
                return
            logger.info("releasing qnn yolo26s detect predictor resources")
            try:
                interpreter.destory()
            except Exception as exc:
                logger.warning("yolo26s detect predictor release failed: %s", exc)
            finally:
                self.interpreter = None
            logger.info("qnn yolo26s detect predictor released")

    def predict_frame(self, orig_img_bgr: np.ndarray):
        with self._lock:
            interpreter = self.interpreter
            if interpreter is None:
                return [], []

            input_tensor, scale = _yolo26s_preprocess(orig_img_bgr, input_size=self.width)
            interpreter.set_input_tensor(0, input_tensor)
            interpreter.invoke()

            bbox_out = interpreter.get_output_tensor(0)
            class_out = interpreter.get_output_tensor(1)

        merged = _yolo26s_merge_outputs(bbox_out, class_out, class_num=self.class_num)
        detections = _yolo26s_postprocess(
            merged,
            original_shape=orig_img_bgr.shape,
            scale=scale,
            conf_thres=self.conf,
            iou_thres=self.iou,
        )
        return detections, []
