#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-image probe for YOLO26s finetune QNN bbox decoding.

This script is intentionally outside the robot-stack runtime. It loads one
image, runs the finetune cutoff model once, then writes visual and JSON outputs
for comparing bbox decode modes by eye.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
VISTA_ROOT = REPO_ROOT / "VISTA"
DEFAULT_MODEL = VISTA_ROOT / "vision_module/model/yolo26s/models/finetune/yolo26s-cutoff-bgr_qcs6490_w8a8.qnn236.ctx.bin"
DEFAULT_CLASSES = VISTA_ROOT / "vision_module/model/yolo26s/models/finetune/classes.txt"

REQUESTED_DECODE_MODES: Tuple[str, ...] = (
    "xyxy_pixel_model_square",
    "xyxy_pixel_direct_resize",
    "xyxy_pixel_letterbox_center",
    "cxcywh_pixel_model_square",
    "cxcywh_pixel_direct_resize",
    "cxcywh_norm_crop",
    "cxcywh_norm_model",
    "xyxy_norm_crop",
    "xyxy_norm_model",
    "xywh_norm_crop",
    "xywh_norm_model",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe YOLO26s finetune QNN model on one image.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--save-dir", default="logs/yolo26s_probe/single_test", help="Output directory.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="QNN ctx.bin path.")
    parser.add_argument("--classes", default=str(DEFAULT_CLASSES), help="classes.txt path.")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--width", type=int, default=640, help="Model input width.")
    parser.add_argument("--height", type=int, default=640, help="Model input height.")
    parser.add_argument("--topk", type=int, default=20, help="Top candidate count.")
    parser.add_argument("--max-det", type=int, default=300, help="Max detections per mode.")
    parser.add_argument(
        "--decode-mode",
        default="xyxy_pixel_model_square",
        choices=REQUESTED_DECODE_MODES,
        help="Decode mode for result_current.jpg.",
    )
    parser.add_argument(
        "--all-modes",
        action="store_true",
        help="Also print and emphasize all decode modes. result_all_modes.jpg is always written.",
    )
    parser.add_argument(
        "--preprocess-mode",
        default="square_fill_top_left",
        choices=("square_fill_top_left", "direct_resize", "letterbox_center"),
        help="Preprocess mode used before inference.",
    )
    parser.add_argument("--backend", default="qnn", choices=("qnn", "snpe", "snpe2"), help="AidLite backend.")
    return parser.parse_args()


def ensure_import_path() -> None:
    vista = str(VISTA_ROOT)
    if vista not in sys.path:
        sys.path.insert(0, vista)


def load_classes(path: Path) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def as_jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(as_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def shape_list(value: Any) -> List[int]:
    try:
        return [int(v) for v in value.shape]
    except Exception:
        pass
    try:
        return [int(v) for v in list(value)]
    except Exception:
        return []


def build_aidlite_model(args: argparse.Namespace, class_num: int):
    import aidlite

    config = aidlite.Config.create_instance()
    config.implement_type = aidlite.ImplementType.TYPE_LOCAL
    backend = str(args.backend or "qnn").lower()
    config.framework_type = aidlite.FrameworkType.TYPE_QNN if backend == "qnn" else aidlite.FrameworkType.TYPE_SNPE2
    config.accelerate_type = aidlite.AccelerateType.TYPE_DSP
    config.is_quantify_model = 1

    model = aidlite.Model.create_instance(str(args.model))
    model.set_model_properties(
        [[1, int(args.height), int(args.width), 3]],
        aidlite.DataType.TYPE_FLOAT32,
        [[1, 4, 8400], [1, int(class_num), 8400]],
        aidlite.DataType.TYPE_FLOAT32,
    )
    interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(model, config)
    interpreter.init()
    interpreter.load_model()
    return interpreter


def box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def is_large_box(box: Sequence[float], image_shape: Sequence[int]) -> Tuple[bool, bool, float, float]:
    h = max(1.0, float(image_shape[0]))
    w = max(1.0, float(image_shape[1]))
    bw = max(0.0, float(box[2]) - float(box[0]))
    bh = max(0.0, float(box[3]) - float(box[1]))
    wr = bw / w
    hr = bh / h
    large = bool(wr >= 0.70 or hr >= 0.70 or (wr * hr) >= 0.50)
    near_full = bool(wr >= 0.90 and hr >= 0.90)
    return large, near_full, wr, hr


def clip_xyxy(box, image_shape: Sequence[int]):
    import numpy as np

    out = np.asarray(box, dtype=np.float32).copy()
    h = float(image_shape[0])
    w = float(image_shape[1])
    out[0] = np.clip(out[0], 0.0, w)
    out[2] = np.clip(out[2], 0.0, w)
    out[1] = np.clip(out[1], 0.0, h)
    out[3] = np.clip(out[3], 0.0, h)
    return out


def clip_ratio(unclipped, clipped, image_shape: Sequence[int]) -> float:
    _ = image_shape
    raw_area = box_area(unclipped)
    clipped_area = box_area(clipped)
    if raw_area <= 1e-6:
        return 0.0
    outside = max(0.0, raw_area - clipped_area)
    edge = any(abs(float(a) - float(b)) > 1e-3 for a, b in zip(unclipped, clipped))
    return max(float(edge), min(1.0, outside / raw_area))


def cxcywh_to_xyxy(raw):
    import numpy as np

    cx, cy, bw, bh = [float(v) for v in raw[:4]]
    return np.array([cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], dtype=np.float32)


def xywh_to_xyxy(raw):
    import numpy as np

    x, y, bw, bh = [float(v) for v in raw[:4]]
    return np.array([x, y, x + bw, y + bh], dtype=np.float32)


def decode_bbox(mode: str, raw_bbox, image_shape: Sequence[int], model_shape: Tuple[int, int], clip: bool = True):
    import numpy as np

    mode = str(mode or "xyxy_pixel_model_square").strip().lower()
    raw = np.asarray(raw_bbox, dtype=np.float32).reshape(-1)[:4]
    src_h = float(image_shape[0])
    src_w = float(image_shape[1])
    model_h = float(model_shape[0])
    model_w = float(model_shape[1])

    if mode.startswith("cxcywh"):
        if "norm" in mode:
            box = raw.copy()
            if "crop" in mode:
                box[[0, 2]] *= src_w
                box[[1, 3]] *= src_h
                xyxy = cxcywh_to_xyxy(box)
            else:
                box[[0, 2]] *= model_w
                box[[1, 3]] *= model_h
                xyxy = cxcywh_to_xyxy(box)
        else:
            xyxy = cxcywh_to_xyxy(raw)
    elif mode.startswith("xywh"):
        if "norm" in mode:
            box = raw.copy()
            if "crop" in mode:
                box[[0, 2]] *= src_w
                box[[1, 3]] *= src_h
            else:
                box[[0, 2]] *= model_w
                box[[1, 3]] *= model_h
            xyxy = xywh_to_xyxy(box)
        else:
            xyxy = xywh_to_xyxy(raw)
    else:
        if "norm" in mode:
            xyxy = raw.copy()
            if "crop" in mode:
                xyxy[[0, 2]] *= src_w
                xyxy[[1, 3]] *= src_h
            else:
                xyxy[[0, 2]] *= model_w
                xyxy[[1, 3]] *= model_h
        else:
            xyxy = raw.copy()

    if mode.endswith("_direct_resize"):
        xyxy[[0, 2]] *= src_w / max(1.0, model_w)
        xyxy[[1, 3]] *= src_h / max(1.0, model_h)
    elif mode.endswith("_letterbox_center"):
        gain = min(model_h / max(1.0, src_h), model_w / max(1.0, src_w))
        new_w = round(src_w * gain)
        new_h = round(src_h * gain)
        pad_x = (model_w - new_w) / 2.0
        pad_y = (model_h - new_h) / 2.0
        xyxy[[0, 2]] = (xyxy[[0, 2]] - pad_x) / max(1e-6, gain)
        xyxy[[1, 3]] = (xyxy[[1, 3]] - pad_y) / max(1e-6, gain)
    elif mode.endswith("_model_square") or mode.endswith("_model"):
        scale = max(src_h, src_w) / max(1.0, max(model_h, model_w))
        xyxy *= float(scale)

    return clip_xyxy(xyxy, image_shape) if clip else xyxy.astype(np.float32)


def nms_indices(boxes, scores, iou_thres: float, nms_fn) -> List[int]:
    if len(boxes) <= 0:
        return []
    return list(nms_fn(boxes, scores, float(iou_thres)))


def detections_for_mode(
    *,
    mode: str,
    merged,
    scores,
    class_ids,
    image_shape: Sequence[int],
    preprocess_meta: Dict[str, Any],
    model_shape: Tuple[int, int],
    conf: float,
    iou: float,
    max_det: int,
    nms_fn,
) -> List[Dict[str, Any]]:
    import numpy as np

    mask = scores >= float(conf)
    if not bool(np.any(mask)):
        return []
    selected_indices = np.where(mask)[0]
    raw_boxes = merged[selected_indices, :4]
    decoded = np.vstack(
        [
            decode_bbox(
                mode,
                row,
                image_shape=image_shape,
                model_shape=model_shape,
                clip=True,
            )
            for row in raw_boxes
        ]
    ).astype(np.float32)
    selected_scores = scores[selected_indices].astype(np.float32)
    selected_classes = class_ids[selected_indices].astype(np.int32)

    output: List[Dict[str, Any]] = []
    for cid in np.unique(selected_classes):
        class_mask = selected_classes == cid
        class_boxes = decoded[class_mask]
        class_scores = selected_scores[class_mask]
        class_indices = selected_indices[class_mask]
        keep = nms_indices(class_boxes, class_scores, iou, nms_fn)[:max_det]
        for local_idx in keep:
            box = class_boxes[int(local_idx)].astype(float).tolist()
            large, near_full, wr, hr = is_large_box(box, image_shape)
            output.append(
                {
                    "decode_mode": mode,
                    "anchor_index": int(class_indices[int(local_idx)]),
                    "bbox": box,
                    "score": float(class_scores[int(local_idx)]),
                    "class_id": int(cid),
                    "large_box": large,
                    "near_full_image": near_full,
                    "box_w_ratio": wr,
                    "box_h_ratio": hr,
                }
            )
    output.sort(key=lambda row: float(row["score"]), reverse=True)
    return output[:max_det]


def candidate_rows(
    *,
    modes: Iterable[str],
    merged,
    scores,
    class_ids,
    classes: Sequence[str],
    image_shape: Sequence[int],
    preprocess_meta: Dict[str, Any],
    model_shape: Tuple[int, int],
    topk: int,
) -> List[Dict[str, Any]]:
    import numpy as np

    top_indices = scores.argsort()[::-1][: max(1, int(topk))]
    rows: List[Dict[str, Any]] = []
    for anchor in top_indices:
        anchor = int(anchor)
        cid = int(class_ids[anchor])
        class_name = str(classes[cid]) if 0 <= cid < len(classes) else str(cid)
        raw_bbox = merged[anchor, :4].astype(np.float32)
        for mode in modes:
            unclipped = decode_bbox(
                mode,
                raw_bbox,
                image_shape=image_shape,
                model_shape=model_shape,
                clip=False,
            )
            clipped = decode_bbox(
                mode,
                raw_bbox,
                image_shape=image_shape,
                model_shape=model_shape,
                clip=True,
            )
            large, near_full, wr, hr = is_large_box(clipped, image_shape)
            clipped_flag = bool(np.any(np.abs(unclipped - clipped) > 1e-3))
            rows.append(
                {
                    "anchor_index": anchor,
                    "raw_bbox": raw_bbox.astype(float).tolist(),
                    "class_id": cid,
                    "class_name": class_name,
                    "score": float(scores[anchor]),
                    "decode_mode": mode,
                    "unclipped_bbox": unclipped.astype(float).tolist(),
                    "clipped_bbox": clipped.astype(float).tolist(),
                    "box_w_ratio": float(wr),
                    "box_h_ratio": float(hr),
                    "large_box": bool(large),
                    "near_full_image": bool(near_full),
                    "clipped": clipped_flag,
                    "clip_ratio": float(clip_ratio(unclipped, clipped, image_shape)),
                }
            )
    return rows


def top_raw_scores(scores_arr, classes: Sequence[str], limit: int = 10) -> List[Dict[str, Any]]:
    import numpy as np

    flat = np.asarray(scores_arr, dtype=np.float32).reshape(-1)
    finite_mask = np.isfinite(flat)
    finite_indices = np.where(finite_mask)[0]
    if finite_indices.size <= 0:
        return []
    order = finite_indices[np.argsort(flat[finite_indices])[::-1][: int(limit)]]
    class_num = int(scores_arr.shape[1])
    anchor_num = int(scores_arr.shape[2])
    rows = []
    for flat_idx in order:
        local = int(flat_idx)
        class_id = (local // anchor_num) % class_num
        anchor = local % anchor_num
        rows.append(
            {
                "anchor_index": int(anchor),
                "class_id": int(class_id),
                "class_name": str(classes[class_id]) if 0 <= class_id < len(classes) else str(class_id),
                "value": float(flat[flat_idx]),
            }
        )
    return rows


def top_merged_scores(class_scores, classes: Sequence[str], limit: int = 10) -> List[Dict[str, Any]]:
    import numpy as np

    arr = np.asarray(class_scores, dtype=np.float32)
    flat = arr.reshape(-1)
    finite_indices = np.where(np.isfinite(flat))[0]
    if finite_indices.size <= 0:
        return []
    order = finite_indices[np.argsort(flat[finite_indices])[::-1][: int(limit)]]
    class_num = int(arr.shape[1])
    rows = []
    for flat_idx in order:
        anchor = int(flat_idx) // class_num
        class_id = int(flat_idx) % class_num
        rows.append(
            {
                "anchor_index": int(anchor),
                "class_id": int(class_id),
                "class_name": str(classes[class_id]) if 0 <= class_id < len(classes) else str(class_id),
                "value": float(flat[flat_idx]),
            }
        )
    return rows


def summarize_modes(detections_by_mode: Dict[str, List[Dict[str, Any]]], classes: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for mode, detections in detections_by_mode.items():
        count = len(detections)
        large_count = sum(1 for det in detections if bool(det.get("large_box")))
        near_full_count = sum(1 for det in detections if bool(det.get("near_full_image")))
        avg_w = sum(float(det.get("box_w_ratio", 0.0) or 0.0) for det in detections) / max(1, count)
        avg_h = sum(float(det.get("box_h_ratio", 0.0) or 0.0) for det in detections) / max(1, count)
        top = detections[0] if detections else None
        if top is not None:
            cid = int(top.get("class_id", -1))
            top_payload = {
                "class_id": cid,
                "class_name": str(classes[cid]) if 0 <= cid < len(classes) else str(cid),
                "score": float(top.get("score", 0.0) or 0.0),
                "bbox": top.get("bbox"),
            }
        else:
            top_payload = None
        summary[mode] = {
            "detection_count": int(count),
            "large_box_count": int(large_count),
            "near_full_image_count": int(near_full_count),
            "avg_box_w_ratio": float(avg_w),
            "avg_box_h_ratio": float(avg_h),
            "top_detection": top_payload,
        }
    return summary


def write_summary_txt(path: Path, mode_summary: Dict[str, Dict[str, Any]], score_summary: Dict[str, Any]) -> None:
    lines = []
    lines.append("YOLO26s single image probe summary")
    lines.append("")
    lines.append("score_sanity:")
    for key in (
        "raw_score_nonfinite_count",
        "raw_score_abs_gt_10_count",
        "sanitized_score_nonfinite_or_abs_gt_10_count",
    ):
        lines.append(f"  {key}: {score_summary.get(key)}")
    lines.append("")
    lines.append("decode_modes:")
    for mode, row in mode_summary.items():
        top = row.get("top_detection") or {}
        top_text = "none"
        if top:
            top_text = (
                f"{top.get('class_name')} score={float(top.get('score', 0.0)):.4f} "
                f"bbox={top.get('bbox')}"
            )
        lines.append(
            f"  {mode}: det={row['detection_count']} large={row['large_box_count']} "
            f"near_full={row['near_full_image_count']} "
            f"avg_w={row['avg_box_w_ratio']:.4f} avg_h={row['avg_box_h_ratio']:.4f} "
            f"top={top_text}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_detections(image, detections: Sequence[Dict[str, Any]], classes: Sequence[str], title: str):
    import cv2
    import numpy as np

    out = image.copy()
    h, w = out.shape[:2]
    palette = [
        (70, 220, 255),
        (90, 180, 255),
        (80, 255, 120),
        (255, 190, 70),
        (220, 120, 255),
        (120, 220, 120),
        (255, 120, 120),
    ]
    cv2.putText(out, title[:80], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    if not detections:
        cv2.putText(out, "no detections", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 255), 2, cv2.LINE_AA)
        return out
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = [int(round(float(v))) for v in det["bbox"][:4]]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        cid = int(det["class_id"])
        name = str(classes[cid]) if 0 <= cid < len(classes) else str(cid)
        color = palette[idx % len(palette)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = (
            f"{name} {float(det['score']):.3f} {det['decode_mode']} "
            f"[{x1},{y1},{x2},{y2}]"
        )
        y_text = max(18, y1 - 6)
        cv2.rectangle(out, (x1, max(0, y_text - 16)), (min(w, x1 + 8 + len(label) * 8), y_text + 4), (0, 0, 0), -1)
        cv2.putText(out, label[:120], (x1 + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return out


def make_contact_sheet(images: Sequence[Any], labels: Sequence[str]):
    import cv2
    import numpy as np

    if not images:
        return np.zeros((320, 480, 3), dtype=np.uint8)
    thumbs = []
    target_w, target_h = 480, 360
    for img, label in zip(images, labels):
        h, w = img.shape[:2]
        scale = min(target_w / max(1, w), target_h / max(1, h))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:] = (18, 22, 28)
        x0, y0 = (target_w - nw) // 2, (target_h - nh) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
        cv2.putText(canvas, label[:58], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(canvas)
    cols = min(3, len(thumbs))
    rows = int(math.ceil(len(thumbs) / float(cols)))
    blank = np.zeros_like(thumbs[0])
    grid = []
    for r in range(rows):
        row_imgs = list(thumbs[r * cols : (r + 1) * cols])
        while len(row_imgs) < cols:
            row_imgs.append(blank.copy())
        grid.append(np.hstack(row_imgs))
    return np.vstack(grid)


def main() -> int:
    args = parse_args()
    ensure_import_path()

    import cv2
    import numpy as np
    from vision_module.backend.predictor.QNN_YOLO26_Detect_Predictor import (
        _yolo26s_merge_outputs,
        _yolo26s_nms,
        _yolo26s_preprocess,
    )

    image_path = Path(args.image)
    save_dir = Path(args.save_dir)
    model_path = Path(args.model)
    classes_path = Path(args.classes)
    save_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, save_dir / "input.jpg")

    classes = load_classes(classes_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not classes:
        raise RuntimeError(f"No classes loaded from {classes_path}")

    input_tensor, preprocess_meta = _yolo26s_preprocess(
        image,
        input_size=int(args.width),
        preprocess_mode=str(args.preprocess_mode),
    )

    start = time.perf_counter()
    interpreter = build_aidlite_model(args, len(classes))
    interpreter.set_input_tensor(0, input_tensor)
    interpreter.invoke()
    infer_ms = (time.perf_counter() - start) * 1000.0
    bbox_out = interpreter.get_output_tensor(0)
    scores_out = interpreter.get_output_tensor(1)
    try:
        interpreter.destory()
    except Exception:
        pass

    bbox_arr = np.asarray(bbox_out, dtype=np.float32).reshape(1, 4, 8400)
    scores_arr = np.asarray(scores_out, dtype=np.float32).reshape(1, len(classes), 8400)
    merged = _yolo26s_merge_outputs(bbox_arr, scores_arr, class_num=len(classes))
    raw_class_scores = merged[:, 4:].astype(np.float32)
    raw_scores_flat = np.asarray(scores_arr, dtype=np.float32).reshape(-1)
    raw_score_nonfinite_count = int(np.size(raw_scores_flat) - int(np.isfinite(raw_scores_flat).sum()))
    raw_score_abs_gt_10_count = int(np.sum(np.isfinite(raw_scores_flat) & (np.abs(raw_scores_flat) > 10.0)))
    valid_score_mask = np.isfinite(raw_class_scores) & (np.abs(raw_class_scores) <= 10.0)
    sanitized_score_bad_count = int(raw_class_scores.size - int(valid_score_mask.sum()))
    class_scores = np.where(valid_score_mask, raw_class_scores, -np.inf).astype(np.float32)
    if not bool(np.isfinite(class_scores).any()):
        class_scores = np.zeros_like(raw_class_scores, dtype=np.float32)
    class_ids = class_scores.argmax(axis=1).astype(np.int32)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    top_idx = int(scores.argmax()) if scores.size else -1
    top_class_id = int(class_ids[top_idx]) if top_idx >= 0 else -1
    top_class_name = str(classes[top_class_id]) if 0 <= top_class_id < len(classes) else ""
    model_shape = (int(args.height), int(args.width))

    modes = list(REQUESTED_DECODE_MODES)
    detections_by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for mode in modes:
        detections_by_mode[mode] = detections_for_mode(
            mode=mode,
            merged=merged,
            scores=scores,
            class_ids=class_ids,
            image_shape=image.shape,
            preprocess_meta=preprocess_meta,
            model_shape=model_shape,
            conf=float(args.conf),
            iou=float(args.iou),
            max_det=int(args.max_det),
            nms_fn=_yolo26s_nms,
        )

    current_dets = detections_by_mode[str(args.decode_mode)]
    result_current = draw_detections(image, current_dets, classes, f"current: {args.decode_mode}")
    cv2.imwrite(str(save_dir / "result_current.jpg"), result_current)

    mode_images = []
    for mode in modes:
        title = f"{mode} det={len(detections_by_mode[mode])}"
        mode_images.append(draw_detections(image, detections_by_mode[mode], classes, title))
    contact = make_contact_sheet(mode_images, [f"{mode} det={len(detections_by_mode[mode])}" for mode in modes])
    cv2.imwrite(str(save_dir / "result_all_modes.jpg"), contact)

    candidates = candidate_rows(
        modes=modes,
        merged=merged,
        scores=scores,
        class_ids=class_ids,
        classes=classes,
        image_shape=image.shape,
        preprocess_meta=preprocess_meta,
        model_shape=model_shape,
        topk=int(args.topk),
    )
    with (save_dir / "candidates.jsonl").open("w", encoding="utf-8") as fh:
        for row in candidates:
            fh.write(json.dumps(as_jsonable(row), ensure_ascii=False) + "\n")

    write_json(
        save_dir / "raw_outputs_summary.json",
        {
            "image": str(image_path),
            "model": str(model_path),
            "classes": str(classes_path),
            "class_count": len(classes),
            "decode_mode": str(args.decode_mode),
            "all_modes": bool(args.all_modes),
            "preprocess_mode": str(args.preprocess_mode),
            "preprocess_meta": preprocess_meta,
            "input_tensor_shape": shape_list(input_tensor),
            "input_tensor_dtype": str(input_tensor.dtype),
            "bbox_tensor_shape": shape_list(bbox_arr),
            "scores_tensor_shape": shape_list(scores_arr),
            "bbox_min": float(np.nanmin(bbox_arr)) if bbox_arr.size else None,
            "bbox_max": float(np.nanmax(bbox_arr)) if bbox_arr.size else None,
            "score_min": float(np.nanmin(scores_arr)) if scores_arr.size else None,
            "score_max": float(np.nanmax(scores_arr)) if scores_arr.size else None,
            "raw_score_nonfinite_count": raw_score_nonfinite_count,
            "raw_score_abs_gt_10_count": raw_score_abs_gt_10_count,
            "sanitized_score_nonfinite_or_abs_gt_10_count": sanitized_score_bad_count,
            "top10_raw_scores": top_raw_scores(scores_arr, classes, limit=10),
            "top10_merged_scores": top_merged_scores(raw_class_scores, classes, limit=10),
            "top10_sanitized_scores": top_merged_scores(class_scores, classes, limit=10),
            "top_anchor_index": top_idx,
            "top_score": float(scores[top_idx]) if top_idx >= 0 else None,
            "top_class_id": top_class_id,
            "top_class_name": top_class_name,
            "infer_ms": float(infer_ms),
        },
    )
    write_json(
        save_dir / "detections.json",
        {
            "current_decode_mode": str(args.decode_mode),
            "conf": float(args.conf),
            "iou": float(args.iou),
            "classes": classes,
            "detections_by_mode": detections_by_mode,
        },
    )
    mode_summary = summarize_modes(detections_by_mode, classes)
    score_summary = {
        "raw_score_nonfinite_count": raw_score_nonfinite_count,
        "raw_score_abs_gt_10_count": raw_score_abs_gt_10_count,
        "sanitized_score_nonfinite_or_abs_gt_10_count": sanitized_score_bad_count,
    }
    write_summary_txt(save_dir / "summary.txt", mode_summary, score_summary)

    print(f"[YOLO26S_PROBE] image={image_path}")
    print(f"[YOLO26S_PROBE] save_dir={save_dir}")
    print(f"[YOLO26S_PROBE] current_mode={args.decode_mode} current_detections={len(current_dets)}")
    print(f"[YOLO26S_PROBE] top_score={float(scores[top_idx]) if top_idx >= 0 else None} top_class={top_class_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
