#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import aidlite
import cv2
import numpy as np

from .base import IPredictor


logger = logging.getLogger("vision.inference")


def _shape_hw(shape: Any) -> Tuple[int, int]:
    if isinstance(shape, np.ndarray):
        shape = shape.shape
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        raise ValueError(f"Expected image/model shape with at least 2 dims, got {shape!r}")
    values = [int(v) for v in shape]
    if len(values) >= 4:
        return values[1], values[2]
    return values[0], values[1]


_DECODE_MODE_ALIASES = {
    "cxcywh_norm_crop": ("cxcywh", True, "crop"),
    "cxcywh_norm_model_square": ("cxcywh", True, "model"),
    "xyxy_norm_crop": ("xyxy", True, "crop"),
    "xyxy_norm_model_square": ("xyxy", True, "model"),
    "xywh_norm_crop": ("xywh", True, "crop"),
    "xywh_norm_model_square": ("xywh", True, "model"),
    "cxcywh_pixel_model_square": ("cxcywh", False, "model"),
    "xyxy_pixel_model_square": ("xyxy", False, "model"),
}


def _normalize_decode_mode(value: Any) -> str:
    mode = str(value or "xyxy_pixel_model_square").strip().lower()
    return mode if mode in _DECODE_MODE_ALIASES else "xyxy_pixel_model_square"


def _decode_mode_parts(decode_mode: str) -> Tuple[str, bool, str]:
    return _DECODE_MODE_ALIASES[_normalize_decode_mode(decode_mode)]


def _normalize_preprocess_mode(value: Any) -> str:
    mode = str(value or "square_fill_top_left").strip().lower()
    return mode if mode in {"square_fill_top_left", "direct_resize", "letterbox_center"} else "square_fill_top_left"


def _yolo26s_preprocess(
    image: np.ndarray,
    input_size: int = 640,
    preprocess_mode: str = "square_fill_top_left",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Square-fill preprocess for yolo26s.

    Returns (input_tensor, preprocess_meta). The camera layer already returns
    the RGB crop/output frame consumed by preview, so this meta maps model-space
    boxes back into that predictor-input/crop-local image by default.
    Short-circuits when the image is already ``input_size × input_size``.
    """
    height, width = image.shape[:2]
    mode = _normalize_preprocess_mode(preprocess_mode)
    length = max(height, width)
    resize_scale = float(input_size) / float(length) if length > 0 else 1.0
    meta = {
        "src_w": int(width),
        "src_h": int(height),
        "crop_x0": 0,
        "crop_y0": 0,
        "crop_w": int(width),
        "crop_h": int(height),
        "model_w": int(input_size),
        "model_h": int(input_size),
        "resize_scale": float(resize_scale),
        "resize_scale_x": float(resize_scale),
        "resize_scale_y": float(resize_scale),
        "scale": float(length) / float(input_size) if input_size > 0 else 1.0,
        "pad_x": 0.0,
        "pad_y": 0.0,
        "draw_space": "crop",
        "preprocess_mode": mode,
    }

    if mode == "direct_resize":
        meta["resize_scale_x"] = float(input_size) / float(width) if width > 0 else 1.0
        meta["resize_scale_y"] = float(input_size) / float(height) if height > 0 else 1.0
        meta["resize_scale"] = float(meta["resize_scale_x"])
        canvas = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        return (canvas.astype(np.float32) / 255.0)[None, :], meta

    if mode == "letterbox_center":
        gain = min(float(input_size) / float(height), float(input_size) / float(width)) if height > 0 and width > 0 else 1.0
        new_w = max(1, int(round(width * gain)))
        new_h = max(1, int(round(height * gain)))
        pad_x = (float(input_size) - float(new_w)) / 2.0
        pad_y = (float(input_size) - float(new_h)) / 2.0
        left = int(round(pad_x - 0.1))
        top = int(round(pad_y - 0.1))
        canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas[top : top + new_h, left : left + new_w] = resized
        meta.update(
            {
                "resize_scale": float(gain),
                "resize_scale_x": float(gain),
                "resize_scale_y": float(gain),
                "scale": 1.0 / float(gain) if gain > 0 else 1.0,
                "pad_x": float(left),
                "pad_y": float(top),
            }
        )
        return (canvas.astype(np.float32) / 255.0)[None, :], meta

    # Short-circuit: already square at target size, skip canvas + resize.
    if height == width == input_size:
        canvas = image
        if canvas.dtype != np.float32:
            canvas = canvas.astype(np.float32) / 255.0
        return canvas[None, :], meta

    canvas = np.zeros((length, length, 3), dtype=np.uint8)
    canvas[:height, :width] = image
    canvas = cv2.resize(canvas, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    return (canvas.astype(np.float32) / 255.0)[None, :], meta


def _bbox_is_normalized(boxes: np.ndarray) -> bool:
    if boxes.size <= 0:
        return False
    finite = boxes[np.isfinite(boxes)]
    if finite.size <= 0:
        return False
    return float(np.nanmax(np.abs(finite))) <= 2.0


def map_bbox_from_model_to_image(
    bbox: Any,
    model_input_shape: Any,
    preprocess_meta: Dict[str, Any],
    draw_image_shape: Any = None,
    output_format: str = "cxcywh",
    normalized: Optional[bool] = None,
    normalized_basis: str = "model",
    draw_space: str = "crop",
    clip: bool = True,
) -> np.ndarray:
    """Map one model-output bbox into image xyxy coordinates.

    Supports cxcywh/xyxy, normalized/model-pixel coordinates, square/letterbox
    padding removal, resize inverse transform, optional crop offset, and clamp.
    """
    arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if arr.size < 4:
        raise ValueError(f"Expected bbox with 4 values, got {arr}")
    box = arr[:4].astype(np.float32, copy=True)
    model_h, model_w = _shape_hw(model_input_shape)
    fmt = str(output_format or "cxcywh").strip().lower()
    if fmt in {"center_xywh"}:
        fmt = "cxcywh"
    if fmt not in {"cxcywh", "xyxy", "xywh"}:
        raise ValueError(f"Unsupported bbox output_format={output_format!r}")
    target_space = str(draw_space or preprocess_meta.get("draw_space") or "crop").strip().lower()
    basis = str(normalized_basis or preprocess_meta.get("bbox_normalized_basis") or "model").strip().lower()
    if basis in {"image", "draw"}:
        basis = "crop"
    if basis not in {"crop", "model"}:
        basis = "model"
    if normalized is None:
        normalized = _bbox_is_normalized(box)
    skip_model_inverse = False
    if normalized:
        if basis == "crop":
            if draw_image_shape is not None:
                image_h, image_w = _shape_hw(draw_image_shape)
            else:
                image_h = int(preprocess_meta.get("crop_h", preprocess_meta.get("src_h", model_h)) or model_h)
                image_w = int(preprocess_meta.get("crop_w", preprocess_meta.get("src_w", model_w)) or model_w)
            crop_w = float(preprocess_meta.get("crop_w", image_w) or image_w)
            crop_h = float(preprocess_meta.get("crop_h", image_h) or image_h)
            box[[0, 2]] *= crop_w
            box[[1, 3]] *= crop_h
            skip_model_inverse = True
        else:
            box[[0, 2]] *= float(model_w)
            box[[1, 3]] *= float(model_h)

    if fmt == "cxcywh":
        cx, cy, bw, bh = [float(v) for v in box[:4]]
        xyxy = np.array([cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], dtype=np.float32)
    elif fmt == "xywh":
        x, y, bw, bh = [float(v) for v in box[:4]]
        xyxy = np.array([x, y, x + bw, y + bh], dtype=np.float32)
    else:
        xyxy = box[:4].astype(np.float32, copy=True)

    if not skip_model_inverse:
        pad_x = float(preprocess_meta.get("pad_x", 0.0) or 0.0)
        pad_y = float(preprocess_meta.get("pad_y", 0.0) or 0.0)
        resize_scale_x = float(preprocess_meta.get("resize_scale_x", preprocess_meta.get("resize_scale", 0.0)) or 0.0)
        resize_scale_y = float(preprocess_meta.get("resize_scale_y", preprocess_meta.get("resize_scale", 0.0)) or 0.0)
        if resize_scale_x <= 0 or resize_scale_y <= 0:
            inv_scale = float(preprocess_meta.get("scale", 1.0) or 1.0)
            fallback = 1.0 / inv_scale if inv_scale > 0 else 1.0
            resize_scale_x = resize_scale_x if resize_scale_x > 0 else fallback
            resize_scale_y = resize_scale_y if resize_scale_y > 0 else fallback

        xyxy[[0, 2]] = (xyxy[[0, 2]] - pad_x) / resize_scale_x
        xyxy[[1, 3]] = (xyxy[[1, 3]] - pad_y) / resize_scale_y

    if target_space in {"full", "full_frame", "original"}:
        xyxy[[0, 2]] += float(preprocess_meta.get("crop_x0", 0.0) or 0.0)
        xyxy[[1, 3]] += float(preprocess_meta.get("crop_y0", 0.0) or 0.0)

    if draw_image_shape is None:
        if target_space in {"full", "full_frame", "original"}:
            draw_image_shape = (
                int(preprocess_meta.get("full_h", preprocess_meta.get("src_h", 0)) or 0),
                int(preprocess_meta.get("full_w", preprocess_meta.get("src_w", 0)) or 0),
            )
        else:
            draw_image_shape = (
                int(preprocess_meta.get("crop_h", preprocess_meta.get("src_h", 0)) or 0),
                int(preprocess_meta.get("crop_w", preprocess_meta.get("src_w", 0)) or 0),
            )
    draw_h, draw_w = _shape_hw(draw_image_shape)
    if not clip:
        return xyxy.astype(np.float32)
    xyxy[0] = np.clip(xyxy[0], 0.0, float(draw_w))
    xyxy[2] = np.clip(xyxy[2], 0.0, float(draw_w))
    xyxy[1] = np.clip(xyxy[1], 0.0, float(draw_h))
    xyxy[3] = np.clip(xyxy[3], 0.0, float(draw_h))
    return xyxy.astype(np.float32)


def decode_bbox_with_mode(
    bbox: Any,
    decode_mode: str,
    model_input_shape: Any,
    preprocess_meta: Dict[str, Any],
    draw_image_shape: Any,
    draw_space: str = "crop",
    clip: bool = True,
) -> np.ndarray:
    fmt, normalized, basis = _decode_mode_parts(decode_mode)
    return map_bbox_from_model_to_image(
        bbox,
        model_input_shape=model_input_shape,
        preprocess_meta=preprocess_meta,
        draw_image_shape=draw_image_shape,
        output_format=fmt,
        normalized=normalized,
        normalized_basis=basis,
        draw_space=draw_space,
        clip=clip,
    )


def _clip_ratio(unclipped: np.ndarray, clipped: np.ndarray, image_shape: Any) -> float:
    h, w = _shape_hw(image_shape)
    raw_w = max(1e-6, float(unclipped[2] - unclipped[0]))
    raw_h = max(1e-6, float(unclipped[3] - unclipped[1]))
    clipped_w = max(0.0, float(clipped[2] - clipped[0]))
    clipped_h = max(0.0, float(clipped[3] - clipped[1]))
    raw_area = raw_w * raw_h
    clipped_area = clipped_w * clipped_h
    outside = max(0.0, raw_area - clipped_area)
    edge_flag = float(
        unclipped[0] < 0.0
        or unclipped[1] < 0.0
        or unclipped[2] > float(w)
        or unclipped[3] > float(h)
    )
    return max(edge_flag, min(1.0, outside / max(raw_area, 1e-6)))


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
    preprocess_meta: Dict[str, Any],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
    output_format: str = "xyxy",
    normalized: Optional[bool] = False,
    normalized_basis: str = "model",
    decode_mode: Optional[str] = "xyxy_pixel_model_square",
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

    raw_boxes = prediction[mask, :4].copy()
    use_normalized = _bbox_is_normalized(raw_boxes) if normalized is None else bool(normalized)
    model_shape = (
        int(preprocess_meta.get("model_h", original_shape[0] if len(original_shape) > 0 else 640) or 640),
        int(preprocess_meta.get("model_w", original_shape[1] if len(original_shape) > 1 else 640) or 640),
    )
    active_decode_mode = _normalize_decode_mode(decode_mode) if decode_mode else ""
    if active_decode_mode:
        boxes = np.vstack(
            [
                decode_bbox_with_mode(
                    row,
                    active_decode_mode,
                    model_input_shape=model_shape,
                    preprocess_meta=preprocess_meta,
                    draw_image_shape=original_shape,
                    draw_space=str(preprocess_meta.get("draw_space") or "crop"),
                    clip=True,
                )
                for row in raw_boxes
            ]
        )
    else:
        boxes = np.vstack(
            [
                map_bbox_from_model_to_image(
                    row,
                    model_input_shape=model_shape,
                    preprocess_meta=preprocess_meta,
                    draw_image_shape=original_shape,
                    output_format=output_format,
                    normalized=use_normalized,
                    normalized_basis=normalized_basis,
                    draw_space=str(preprocess_meta.get("draw_space") or "crop"),
                )
                for row in raw_boxes
            ]
        )

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
        self._debug_frames_remaining = max(0, int(os.getenv("VISTA_YOLO26_BBOX_DEBUG_FRAMES", "3") or "0"))
        self._debug_topk = max(1, int(os.getenv("VISTA_YOLO26_BBOX_DEBUG_TOPK", "5") or "5"))
        self.decode_mode = _normalize_decode_mode(
            os.getenv("VISTA_YOLO26_BBOX_DECODE_MODE", getattr(args, "bbox_decode_mode", "xyxy_pixel_model_square"))
        )
        self.bbox_format, mode_normalized, self.bbox_normalized_basis = _decode_mode_parts(self.decode_mode)
        self.bbox_coord_type = "normalized" if mode_normalized else "pixel"
        self.preprocess_mode = _normalize_preprocess_mode(
            os.getenv("VISTA_YOLO26_PREPROCESS_MODE", getattr(args, "preprocess_mode", "square_fill_top_left"))
        )

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
            "qnn yolo26s detect predictor loaded (cutoff, %d outputs): %s | backend=%s class_num=%d input_shape=%s output_shapes=%s decode_mode=%s preprocess_mode=%s bbox_format=%s bbox_coord_type=%s bbox_normalized_basis=%s",
            len(self.output_shapes),
            target_model,
            backend_name,
            self.class_num,
            self.input_shape,
            self.output_shapes,
            self.decode_mode,
            self.preprocess_mode,
            self.bbox_format,
            self.bbox_coord_type,
            self.bbox_normalized_basis,
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

            input_tensor, preprocess_meta = _yolo26s_preprocess(
                orig_img_bgr,
                input_size=self.width,
                preprocess_mode=self.preprocess_mode,
            )
            interpreter.set_input_tensor(0, input_tensor)
            interpreter.invoke()

            bbox_out = interpreter.get_output_tensor(0)
            class_out = interpreter.get_output_tensor(1)

        merged = _yolo26s_merge_outputs(bbox_out, class_out, class_num=self.class_num)
        detections = _yolo26s_postprocess(
            merged,
            original_shape=orig_img_bgr.shape,
            preprocess_meta=preprocess_meta,
            conf_thres=self.conf,
            iou_thres=self.iou,
            decode_mode=self.decode_mode,
        )
        if self._debug_frames_remaining > 0:
            self._debug_frames_remaining -= 1
            self._log_bbox_debug(
                orig_img_bgr.shape,
                input_tensor,
                bbox_out,
                class_out,
                merged,
                preprocess_meta,
                detections,
            )
        return detections, []

    def _candidate_debug_row(
        self,
        *,
        anchor_idx: int,
        raw_bbox: np.ndarray,
        table_score: float,
        max_score: float,
        max_class_id: int,
        model_shape: Tuple[int, int],
        image_shape: Tuple[int, ...],
        preprocess_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        mode_boxes = {}
        for mode in (
            "cxcywh_norm_crop",
            "cxcywh_norm_model_square",
            "xyxy_norm_crop",
            "xyxy_norm_model_square",
            "xywh_norm_crop",
            "cxcywh_pixel_model_square",
            "xyxy_pixel_model_square",
        ):
            try:
                mode_boxes[mode] = decode_bbox_with_mode(
                    raw_bbox,
                    mode,
                    model_input_shape=model_shape,
                    preprocess_meta=preprocess_meta,
                    draw_image_shape=image_shape,
                    draw_space=str(preprocess_meta.get("draw_space") or "crop"),
                    clip=False,
                ).round(3).tolist()
            except Exception as exc:
                mode_boxes[mode] = f"error:{exc}"
        final_unclipped = decode_bbox_with_mode(
            raw_bbox,
            self.decode_mode,
            model_input_shape=model_shape,
            preprocess_meta=preprocess_meta,
            draw_image_shape=image_shape,
            draw_space=str(preprocess_meta.get("draw_space") or "crop"),
            clip=False,
        )
        final_clipped = decode_bbox_with_mode(
            raw_bbox,
            self.decode_mode,
            model_input_shape=model_shape,
            preprocess_meta=preprocess_meta,
            draw_image_shape=image_shape,
            draw_space=str(preprocess_meta.get("draw_space") or "crop"),
            clip=True,
        )
        h, w = _shape_hw(image_shape)
        box_w = max(0.0, float(final_clipped[2] - final_clipped[0]))
        box_h = max(0.0, float(final_clipped[3] - final_clipped[1]))
        return {
            "anchor": int(anchor_idx),
            "raw_bbox": np.asarray(raw_bbox, dtype=np.float32).round(6).tolist(),
            "table1_score": round(float(table_score), 6),
            "max_score": round(float(max_score), 6),
            "max_class_id": int(max_class_id),
            "decode_unclipped": mode_boxes,
            "final_unclipped": final_unclipped.round(3).tolist(),
            "final_clipped": final_clipped.round(3).tolist(),
            "clip_ratio": round(float(_clip_ratio(final_unclipped, final_clipped, image_shape)), 6),
            "clipped_by_edge": bool(_clip_ratio(final_unclipped, final_clipped, image_shape) > 0.0),
            "box_w_over_image_w": round(box_w / max(1.0, float(w)), 6),
            "box_h_over_image_h": round(box_h / max(1.0, float(h)), 6),
        }

    def _log_bbox_debug(
        self,
        image_shape: Tuple[int, ...],
        input_tensor: np.ndarray,
        bbox_out: Any,
        class_out: Any,
        merged: np.ndarray,
        preprocess_meta: Dict[str, Any],
        detections: np.ndarray,
    ) -> None:
        bbox = np.asarray(bbox_out, dtype=np.float32).reshape(1, 4, 8400)[0].T
        scores_tensor = np.asarray(class_out, dtype=np.float32).reshape(1, self.class_num, 8400)[0].T
        table_scores = scores_tensor[:, 0] if self.class_num > 0 else np.zeros((scores_tensor.shape[0],), dtype=np.float32)
        max_ids = scores_tensor.argmax(axis=1)
        max_scores = scores_tensor[np.arange(scores_tensor.shape[0]), max_ids]
        topk = min(int(self._debug_topk), int(bbox.shape[0]))
        max_indices = max_scores.argsort()[::-1][:topk]
        model_shape = (
            int(preprocess_meta.get("model_h", self.height) or self.height),
            int(preprocess_meta.get("model_w", self.width) or self.width),
        )
        rows = []
        for idx in max_indices:
            idx = int(idx)
            raw_bbox = bbox[idx]
            mapped = decode_bbox_with_mode(
                raw_bbox,
                self.decode_mode,
                model_input_shape=model_shape,
                preprocess_meta=preprocess_meta,
                draw_image_shape=image_shape,
                draw_space=str(preprocess_meta.get("draw_space") or "crop"),
                clip=True,
            )
            rows.append(
                {
                    "anchor": idx,
                    "raw_bbox": np.asarray(raw_bbox, dtype=np.float32).round(3).tolist(),
                    "mapped_bbox": mapped.round(3).tolist(),
                    "class_id": int(max_ids[idx]),
                    "score": round(float(max_scores[idx]), 6),
                }
            )
        logger.info(
            "yolo26s bbox debug | image_shape=%s preprocess_mode=%s decode_mode=%s bbox_out_minmax=(%.4f,%.4f) score_minmax=(%.4f,%.4f) detections_first3=%s top_mapped=%s",
            tuple(image_shape),
            self.preprocess_mode,
            self.decode_mode,
            float(np.nanmin(bbox)) if bbox.size else 0.0,
            float(np.nanmax(bbox)) if bbox.size else 0.0,
            float(np.nanmin(scores_tensor)) if scores_tensor.size else 0.0,
            float(np.nanmax(scores_tensor)) if scores_tensor.size else 0.0,
            detections[:3].tolist(),
            rows,
        )
