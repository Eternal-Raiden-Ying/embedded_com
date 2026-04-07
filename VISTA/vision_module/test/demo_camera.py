#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import aidcv as cv2
except ImportError:
    import cv2


TEST_DIR = Path(__file__).resolve().parent
VISION_ROOT = TEST_DIR.parent
VISTA_ROOT = VISION_ROOT.parent
if str(VISTA_ROOT) not in sys.path:
    sys.path.insert(0, str(VISTA_ROOT))
if str(VISION_ROOT) not in sys.path:
    sys.path.insert(0, str(VISION_ROOT))

from vision_module.config.board_config import CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive VISTA camera/model demo")
    parser.add_argument("--stream", choices=["rgb", "depth", "ir"], default="")
    parser.add_argument("--backend", choices=["auto", "mock", "real"], default="auto")
    parser.add_argument("--model", default="none", help="none | active | model profile name")
    parser.add_argument("--headless", action="store_true", help="run without opening a window")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames; 0 means unlimited")
    return parser.parse_args()


def available_model_names() -> List[str]:
    return sorted(CONFIG.model.profiles.keys())


def choose_stream(args: argparse.Namespace) -> str:
    if args.stream:
        return args.stream
    print("=" * 60)
    print("Select stream")
    print("  1. rgb   (/dev/video6)")
    print("  2. depth (/dev/video2)")
    print("  3. ir    (/dev/video4)")
    print("=" * 60)
    choice = input("Enter 1/2/3: ").strip()
    mapping = {"1": "rgb", "2": "depth", "3": "ir"}
    return mapping.get(choice, "rgb")


def resolve_model_name(requested: str) -> str:
    requested = str(requested or "none").strip().lower()
    if requested in {"", "none", "off"}:
        return "none"
    if requested == "active":
        return str(CONFIG.model.active_model)
    for name in available_model_names():
        if requested == name.lower():
            return name
    raise SystemExit(f"Unknown model: {requested}. Available: none, active, {', '.join(available_model_names())}")


def import_camera_classes(backend: str):
    if backend == "mock":
        module = importlib.import_module("vision_module.backend.camera.mock")
        return module.MockCamera, module.MockCamera
    hw = importlib.import_module("vision_module.backend.camera.HardwareCamera")
    depth = importlib.import_module("vision_module.backend.camera.RealSenseDepthCamera")
    return hw.HardwareCamera, depth.RealSenseDepthCamera


def import_predictor_class(backend: str):
    if backend == "mock":
        module = importlib.import_module("vision_module.backend.predictor.mock")
        return module.MockPredictor
    module = importlib.import_module("vision_module.backend.predictor.QNNPredictor")
    return module.QNN_YOLO_Segment_Predictor


def try_backend_order(requested: str) -> List[str]:
    if requested == "real":
        return ["real"]
    if requested == "mock":
        return ["mock"]
    return ["real", "mock"]


def build_camera_kwargs(stream: str) -> Dict[str, object]:
    if stream == "rgb":
        return {
            "device": "/dev/video6",
            "in_w": 1280,
            "in_h": 720,
            "out_w": 640,
            "out_h": 640,
            "fps": 30,
            "in_format": "YUY2",
            "format": "RGB",
            "crop_x": 280,
            "crop_y": 0,
            "crop_w": 720,
            "crop_h": 720,
            "auto_exposure": False,
            "exposure": 166,
            "brightness": 0,
        }
    if stream == "depth":
        return {"width": 424, "height": 240, "fps": 15}
    return {
        "device": "/dev/video4",
        "in_w": 640,
        "in_h": 480,
        "out_w": 640,
        "out_h": 480,
        "fps": 30,
        "format": "GRAY8",
    }


def open_camera(stream: str, requested_backend: str):
    kwargs = build_camera_kwargs(stream)
    errors: List[str] = []
    for backend in try_backend_order(requested_backend):
        try:
            hardware_cls, depth_cls = import_camera_classes(backend)
            cls = depth_cls if stream == "depth" else hardware_cls
            camera = cls(**kwargs)
            return camera, backend
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
            if requested_backend != "auto":
                break
            print(f"[WARN] camera backend {backend} unavailable: {exc}")
    raise RuntimeError("camera init failed | " + " | ".join(errors))


def open_predictor(model_name: str, requested_backend: str):
    if model_name == "none":
        return None, "none"
    profile = CONFIG.model.profiles[model_name]
    errors: List[str] = []
    for backend in try_backend_order(requested_backend):
        try:
            predictor_cls = import_predictor_class(backend)
            predictor = predictor_cls(profile)
            return predictor, backend
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
            if requested_backend != "auto":
                break
            print(f"[WARN] predictor backend {backend} unavailable: {exc}")
    raise RuntimeError("predictor init failed | " + " | ".join(errors))


def safe_release(obj) -> None:
    if obj is None:
        return
    try:
        obj.release()
    except Exception:
        pass


def depth_to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame.copy()
    if frame.dtype != np.uint8:
        norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        norm = norm.astype(np.uint8)
    else:
        norm = frame
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def ir_to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame.copy()
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return ir_to_bgr(frame)


def frame_to_display(stream: str, frame: np.ndarray) -> np.ndarray:
    if stream == "rgb":
        return rgb_to_bgr(frame)
    if stream == "depth":
        return depth_to_bgr(frame)
    return ir_to_bgr(frame)


def prepare_predictor_input(display_bgr: np.ndarray, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    profile = CONFIG.model.profiles[model_name]
    resized_bgr = cv2.resize(display_bgr, (profile.width, profile.height))
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    return resized_bgr, resized_rgb


def draw_detections(image_bgr: np.ndarray, boxes, masks, class_names) -> np.ndarray:
    if boxes is None or len(boxes) == 0:
        return image_bgr
    out = image_bgr.copy()
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = [int(v) for v in boxes[idx][:4]]
        score = float(boxes[idx][4]) if len(boxes[idx]) > 4 else 0.0
        cls_id = int(boxes[idx][5]) if len(boxes[idx]) > 5 else -1
        color = (0, 255, 0) if cls_id < 0 else (0, int((cls_id * 40) % 255), int((255 - cls_id * 30) % 255))
        if masks is not None and idx < len(masks):
            mask = np.asarray(masks[idx]).astype(bool)
            if mask.shape[:2] == out.shape[:2]:
                overlay = np.zeros_like(out)
                overlay[:, :] = color
                out[mask] = (out[mask] * 0.5 + overlay[mask] * 0.5).astype(np.uint8)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = class_names[cls_id] if class_names and 0 <= cls_id < len(class_names) else f"cls{cls_id}"
        cv2.putText(out, f"{label} {score:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def overlay_status(
    image_bgr: np.ndarray,
    stream: str,
    camera_backend: str,
    model_name: str,
    predictor_backend: str,
    fps: float,
    exposure: int,
    brightness: int,
) -> np.ndarray:
    lines = [
        f"stream={stream} camera_backend={camera_backend}",
        f"model={model_name} predictor_backend={predictor_backend}",
        f"fps={fps:.1f}",
    ]
    if stream == "rgb":
        lines.append(f"exposure={exposure} brightness={brightness}")
        lines.append("keys: A/D exposure W/S brightness N next-model M toggle-model")
    else:
        lines.append("keys: N next-model M toggle-model")
    lines.append("keys: ESC/Q quit")
    y = 25
    for line in lines:
        cv2.putText(image_bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25
    return image_bgr


def next_model_name(current: str) -> str:
    names = available_model_names()
    if not names:
        return "none"
    if current == "none":
        return names[0]
    idx = names.index(current) if current in names else 0
    return names[(idx + 1) % len(names)]


def print_demo_info(stream: str, requested_backend: str, model_name: str) -> None:
    print("=" * 72)
    print("VISTA demo_camera")
    print("=" * 72)
    print(f"stream={stream}")
    print(f"requested_backend={requested_backend}")
    print(f"requested_model={model_name}")
    print(f"available_models={', '.join(available_model_names()) or 'none'}")
    print("=" * 72)


def main() -> int:
    args = parse_args()
    stream = choose_stream(args)
    model_name = resolve_model_name(args.model)
    print_demo_info(stream, args.backend, model_name)

    predictor = None
    camera = None
    predictor_backend = "none"
    camera_backend = "none"
    current_model = model_name

    try:
        camera, camera_backend = open_camera(stream, args.backend)
        if current_model != "none":
            predictor, predictor_backend = open_predictor(current_model, args.backend)
        print(f"[INFO] camera_backend={camera_backend} predictor_backend={predictor_backend}")

        window_name = "VISTA Demo Camera"
        if not args.headless:
            cv2.namedWindow(window_name)

        prev_time = time.time()
        frame_count = 0
        exposure = 166
        brightness = 0

        while True:
            frame = camera.read_frame()
            if frame is None or getattr(frame, "size", 0) == 0:
                if args.headless and args.max_frames and frame_count >= args.max_frames:
                    break
                if not args.headless:
                    cv2.waitKey(5)
                continue

            display_bgr = frame_to_display(stream, frame)
            vis_bgr = display_bgr

            if predictor is not None and predictor.is_ready() and current_model != "none":
                predictor_bgr, predictor_rgb = prepare_predictor_input(display_bgr, current_model)
                boxes, masks = predictor.predict_frame(predictor_rgb)
                class_names = CONFIG.model.profiles[current_model].classes
                vis_bgr = draw_detections(predictor_bgr, boxes, masks, class_names)

            current_time = time.time()
            fps = 1.0 / max(1e-6, current_time - prev_time)
            prev_time = current_time
            vis_bgr = overlay_status(vis_bgr, stream, camera_backend, current_model, predictor_backend, fps, exposure, brightness)

            if not args.headless:
                cv2.imshow(window_name, vis_bgr)

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break

            key = -1
            if not args.headless:
                key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break
            if key == ord("n"):
                next_model = next_model_name(current_model)
                safe_release(predictor)
                predictor = None
                predictor_backend = "none"
                current_model = next_model
                if current_model != "none":
                    predictor, predictor_backend = open_predictor(current_model, args.backend)
                print(f"[INFO] switched model -> {current_model} ({predictor_backend})")
            elif key == ord("m"):
                if current_model == "none":
                    current_model = resolve_model_name(args.model if args.model != "none" else "active")
                    predictor, predictor_backend = open_predictor(current_model, args.backend)
                else:
                    safe_release(predictor)
                    predictor = None
                    predictor_backend = "none"
                    current_model = "none"
                print(f"[INFO] model toggle -> {current_model}")
            elif stream == "rgb" and key == ord("a") and hasattr(camera, "set_exposure"):
                exposure = max(1, exposure - 10)
                camera.set_exposure(exposure)
            elif stream == "rgb" and key == ord("d") and hasattr(camera, "set_exposure"):
                exposure = min(10000, exposure + 10)
                camera.set_exposure(exposure)
            elif stream == "rgb" and key == ord("s") and hasattr(camera, "set_brightness"):
                brightness = max(-64, brightness - 5)
                camera.set_brightness(brightness)
            elif stream == "rgb" and key == ord("w") and hasattr(camera, "set_brightness"):
                brightness = min(64, brightness + 5)
                camera.set_brightness(brightness)

        return 0
    finally:
        safe_release(predictor)
        safe_release(camera)
        if not args.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
