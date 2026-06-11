#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


TEST_DIR = Path(__file__).resolve().parent
VISION_ROOT = TEST_DIR.parent
VISTA_ROOT = VISION_ROOT.parent
CAMERA_BACKEND_DIR = VISION_ROOT / "backend" / "camera"
if str(VISTA_ROOT) not in sys.path:
    sys.path.insert(0, str(VISTA_ROOT))
if str(VISION_ROOT) not in sys.path:
    sys.path.insert(0, str(VISION_ROOT))
if str(CAMERA_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(CAMERA_BACKEND_DIR))

if platform.machine().lower() != "aarch64":
    pytest.skip("VISTA camera inference e2e requires aarch64 target hardware", allow_module_level=True)
if importlib.util.find_spec("fast_cam") is None:
    pytest.skip("VISTA camera inference e2e requires fast_cam native module", allow_module_level=True)

from vision_module.backend.camera.ColorCamera import ColorCamera
from vision_module.backend.predictor.QNN_YOLO_Detect_Predictor import QNN_YOLO_Detect_Predictor


DEFAULT_MODEL_PATH = (
    "/home/aidlux/embedded_com/VISTA/vision_module/model/qnn216/"
    "model_farm_yolov7_qcs6490_qnn2.16_int8_aidlite/models/"
    "cutoff_yolov7_w8a8.qnn216.ctx.bin"
)
DEFAULT_ANCHORS = (
    (12, 16, 19, 36, 40, 28),
    (36, 75, 76, 55, 72, 146),
    (142, 110, 192, 243, 459, 401),
)
DEFAULT_STRIDES = (8, 16, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VISTA real hardware end-to-end test: RGB camera + QNN216 detector"
    )
    parser.add_argument("--device", default="/dev/video6")
    parser.add_argument("--in-w", type=int, default=1280)
    parser.add_argument("--in-h", type=int, default=720)
    parser.add_argument("--out-w", type=int, default=640)
    parser.add_argument("--out-h", type=int, default=640)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--loops", type=int, default=20)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--class-num", type=int, default=80)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    return parser.parse_args()


def make_predictor_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        target_model=args.model_path,
        conf_thres=float(args.conf_thres),
        iou_thres=float(args.iou_thres),
        width=int(args.out_w),
        height=int(args.out_h),
        class_num=int(args.class_num),
        model_backend="qnn",
        anchors=DEFAULT_ANCHORS,
        strides=DEFAULT_STRIDES,
    )


def main() -> int:
    args = parse_args()
    predictor = None
    camera = None
    loops = max(1, int(args.loops))
    empty_count = 0
    infer_ok = 0
    infer_ms = []

    print("=" * 72)
    print("VISTA Camera + QNN216 E2E Test")
    print("=" * 72)
    print(f"device={args.device} out={args.out_w}x{args.out_h} fps={args.fps}")
    print(f"model={args.model_path}")
    print(f"loops={loops} class_num={args.class_num}")
    print("=" * 72)

    try:
        # Load predictor first to avoid TLS import-order issues on some images.
        predictor = QNN_YOLO_Detect_Predictor(make_predictor_args(args))
        if not predictor.is_ready():
            print("FAIL     | predictor not ready")
            return 1
        print("PASS     | predictor_init")

        camera = ColorCamera(
            device=args.device,
            in_w=int(args.in_w),
            in_h=int(args.in_h),
            out_w=int(args.out_w),
            out_h=int(args.out_h),
            fps=int(args.fps),
            in_format="YUY2",
            format="RGB",
            crop_x=0,
            crop_y=0,
            crop_w=0,
            crop_h=0,
        )
        print("PASS     | camera_init")

        for i in range(loops):
            frame = camera.read_frame()
            if frame is None or getattr(frame, "size", 0) == 0:
                empty_count += 1
                print(f"WARN     | iter={i} empty frame")
                continue

            start = time.perf_counter()
            boxes, _ = predictor.predict_frame(frame)
            dt_ms = (time.perf_counter() - start) * 1000.0
            infer_ms.append(dt_ms)
            infer_ok += 1
            print(
                "PASS     | "
                f"iter={i} shape={getattr(frame, 'shape', None)} boxes={len(boxes)} infer_ms={dt_ms:.2f}"
            )

        if infer_ok <= 0:
            print(f"FAIL     | no successful inference; empty_frames={empty_count}")
            return 1

        avg_ms = sum(infer_ms) / len(infer_ms)
        print(
            "PASS     | summary | "
            f"infer_ok={infer_ok}/{loops} empty_frames={empty_count} "
            f"avg_ms={avg_ms:.2f} min_ms={min(infer_ms):.2f} max_ms={max(infer_ms):.2f}"
        )
        return 0
    except Exception as exc:
        print(f"FAIL     | error | {exc}")
        return 1
    finally:
        if camera is not None:
            try:
                camera.release()
            except Exception:
                pass
        if predictor is not None:
            try:
                predictor.release()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
