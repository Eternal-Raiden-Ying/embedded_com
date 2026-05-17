#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Camera + YOLO pipeline smoke test — direct camera and predictor, no engine.

Mock backend support via --backend auto (default) or --backend mock.
"""

import argparse
import sys
import time
from typing import Tuple

import numpy as np

try:
    import aidcv as cv2
except ImportError:
    try:
        import cv2
    except ImportError:
        cv2 = None

from test_support import (
    EXIT_FAIL,
    EXIT_INTERRUPT,
    EXIT_OK,
    EXIT_USAGE,
    add_camera_args,
    add_common_backend_args,
    add_model_args,
    describe_frame,
    import_camera_classes,
    import_predictor_class,
    make_model_profile,
    print_header,
    print_step,
    print_summary,
    safe_release,
    try_with_backends,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VISTA pipeline backend smoke test")
    add_common_backend_args(parser)
    add_camera_args(parser)
    add_model_args(parser)
    parser.add_argument("--debug-window", action="store_true",
                        help="Show CV debug window with YOLO detection boxes")
    return parser


COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def _class_name(cls_id: int) -> str:
    idx = int(cls_id)
    if 0 <= idx < len(COCO80_NAMES):
        return COCO80_NAMES[idx]
    return f"cls#{idx}"


def camera_kwargs(args: argparse.Namespace) -> dict:
    return {
        "device": args.rgb_device,
        "in_w": args.rgb_in_w,
        "in_h": args.rgb_in_h,
        "out_w": args.rgb_out_w,
        "out_h": args.rgb_out_h,
        "fps": args.rgb_fps,
        "format": "BGR",
        "in_format": "YUY2",
        "auto_exposure": True,
    }


def draw_yolo_boxes(display: np.ndarray, boxes) -> np.ndarray:
    """Draw raw YOLO detection boxes onto a BGR frame.

    ``boxes`` are already in pixel coordinates — ``detect_postprocess()`` calls
    ``scale_coords()`` which maps from model-input space back to the original
    frame dimensions.  No further normalisation is needed here.
    """
    if boxes is None or (isinstance(boxes, np.ndarray) and boxes.size == 0):
        return display
    h, w = display.shape[:2]
    for row in boxes:
        if len(row) < 6:
            continue
        x1, y1, x2, y2, conf, cls_id = row[:6]
        pt1 = (max(0, int(x1)), max(0, int(y1)))
        pt2 = (min(w, int(x2)), min(h, int(y2)))
        cv2.rectangle(display, pt1, pt2, (0, 255, 0), 2)

        label = f"{_class_name(cls_id)} {float(conf):.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = pt1[1] - th - baseline if pt1[1] - th - baseline > 0 else pt2[1]
        cv2.rectangle(display, (pt1[0], label_y - 2),
                      (pt1[0] + tw + 4, label_y + th + 2), (0, 255, 0), -1)
        cv2.putText(display, label, (pt1[0] + 2, label_y + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return display


# ── Backend resolution ─────────────────────────────────────────────────

def resolve_camera(requested: str, args: argparse.Namespace):
    """Return (class or None, AttemptResult) — camera is opened just to verify."""
    kw = camera_kwargs(args)

    def real_factory():
        color_cls, _, _ = import_camera_classes("real")
        cam = color_cls(**kw)
        frame = cam.read_frame()
        if frame is None or getattr(frame, "size", 0) == 0:
            safe_release(cam)
            raise RuntimeError("camera opened but returned empty frame")
        safe_release(cam)
        return color_cls

    def mock_factory():
        color_cls, _, _ = import_camera_classes("mock")
        return color_cls  # mock camera always returns frames

    return try_with_backends(requested, real_factory, mock_factory)


def resolve_predictor(requested: str, args: argparse.Namespace):
    profile = make_model_profile(args)

    def real_factory():
        if not args.model_path:
            raise RuntimeError("--model-path is required for real predictor")
        predictor_cls = import_predictor_class("real")
        inst = predictor_cls(profile)
        if not inst.is_ready():
            safe_release(inst)
            raise RuntimeError("predictor not ready")
        safe_release(inst)
        return predictor_cls

    def mock_factory():
        return import_predictor_class("mock")

    return try_with_backends(requested, real_factory, mock_factory)


# ── Unified test phase ─────────────────────────────────────────────────

def run_pipeline(
    camera_cls,
    predictor_cls,
    camera_backend: str,
    predictor_backend: str,
    args: argparse.Namespace,
    debug_window: bool,
) -> Tuple[bool, str]:
    """Open camera + predictor directly, run inference loop.

    Predictor is initialised first to avoid static TLS conflicts between
    QNN runtime and GL dispatch on ARM boards.
    """

    # 1. Predictor (before camera — TLS ordering).
    print_step("predictor_init", "INFO", f"backend={predictor_backend}")
    profile = make_model_profile(args)
    predictor = predictor_cls(profile)
    if not predictor.is_ready():
        safe_release(predictor)
        return False, "predictor not ready"
    print_step("predictor_init", "PASS", type(predictor).__name__)

    # 2. Camera.
    print_step("camera_init", "INFO", f"backend={camera_backend}")
    kw = camera_kwargs(args)
    camera = camera_cls(**kw)
    frame = camera.read_frame()
    if frame is None or getattr(frame, "size", 0) == 0:
        safe_release(camera)
        safe_release(predictor)
        return False, "camera returned empty frame"
    print_step("camera_init", "PASS", describe_frame(frame))

    # 3. Inference loop — infinite when debug window is active (stop with ESC / Ctrl+C).
    show_window = bool(debug_window and cv2 is not None)
    if debug_window and cv2 is None:
        print_step("debug_window", "WARN", "cv2/aidcv not available")

    if show_window:
        cv2.namedWindow("pipeline debug (ESC to exit)")
        print_step("debug_window", "INFO", "ESC to close window, Ctrl+C to exit")

    t0 = time.time()
    frame_count = 0
    infer_count = 0
    window_active = show_window
    live_mode = show_window  # infinite loop for debug, limited for headless
    try:
        while live_mode or frame_count < max(args.iterations, 100):
            frame = camera.read_frame()
            if frame is None or frame.size == 0:
                if window_active:
                    cv2.waitKey(5)
                continue
            frame_count += 1

            boxes, _masks = predictor.predict_frame(frame)
            has_boxes = isinstance(boxes, np.ndarray) and boxes.size > 0
            if has_boxes:
                infer_count += 1

            if window_active:
                display = draw_yolo_boxes(frame.copy(), boxes)
                fps = frame_count / max(0.001, time.time() - t0)
                cv2.putText(display, f"fps={fps:.1f} frames={frame_count} infer={infer_count}",
                            (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("pipeline debug (ESC to exit)", display)
                if cv2.waitKey(1) & 0xFF == 27:
                    window_active = False
                    live_mode = False
                    cv2.destroyWindow("pipeline debug (ESC to exit)")

        elapsed = time.time() - t0
        fps = frame_count / max(0.001, elapsed)
        ok = frame_count > 0
        detail = f"frames={frame_count} infer={infer_count} time={elapsed:.1f}s fps={fps:.1f}"
        return (True, detail) if ok else (False, detail)
    except KeyboardInterrupt:
        elapsed = time.time() - t0
        fps = frame_count / max(0.001, elapsed)
        print_step("pipeline", "STOP", f"Ctrl+C after frames={frame_count} infer={infer_count} time={elapsed:.1f}s fps={fps:.1f}")
        return (True, f"frames={frame_count} infer={infer_count} time={elapsed:.1f}s fps={fps:.1f}")
    finally:
        safe_release(camera)
        safe_release(predictor)
        if show_window:
            cv2.destroyAllWindows()


# ── Main ───────────────────────────────────────────────────────────────

def main() -> int:
    args = build_parser().parse_args()
    print_header("VISTA Pipeline Backend Test", args)

    # Resolve backends (predictor first for TLS ordering).
    predictor_cls, predictor_result = resolve_predictor(args.backend, args)
    camera_cls, camera_result = resolve_camera(args.backend, args)

    camera_backend = camera_result.resolved
    predictor_backend = predictor_result.resolved

    if camera_cls is None or predictor_cls is None:
        print_step("camera_backend", "PASS" if camera_cls else "FAIL", camera_result.detail)
        print_step("predictor_backend", "PASS" if predictor_cls else "FAIL", predictor_result.detail)
        print_summary(args.backend, "none", "FAIL",
                      [("camera", camera_result.detail), ("predictor", predictor_result.detail)])
        return EXIT_USAGE if ("model-path" in predictor_result.detail) else EXIT_FAIL

    print_step("camera_backend", "PASS", camera_result.detail)
    print_step("predictor_backend", "PASS", predictor_result.detail)

    ok, detail = run_pipeline(camera_cls, predictor_cls,
                              camera_backend, predictor_backend,
                              args, debug_window=args.debug_window)

    overall = "PASS" if ok else "FAIL"
    print_summary(args.backend,
                  f"camera={camera_backend} predictor={predictor_backend}",
                  overall, [("pipeline", detail)])
    return EXIT_OK if ok else EXIT_FAIL


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(EXIT_INTERRUPT)
