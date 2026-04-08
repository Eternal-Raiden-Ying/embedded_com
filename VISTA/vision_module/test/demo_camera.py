#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Dict, List

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive VISTA camera demo")
    parser.add_argument("--stream", choices=["rgb", "depth", "ir"], default="")
    parser.add_argument("--backend", choices=["auto", "mock", "real"], default="real")
    parser.add_argument("--headless", action="store_true", help="run without opening a window")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames; 0 means unlimited")
    return parser.parse_args()


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


def import_camera_classes():
    color = importlib.import_module("vision_module.backend.camera.ColorCamera")
    ir = importlib.import_module("vision_module.backend.camera.IRCamera")
    depth = importlib.import_module("vision_module.backend.camera.RealSenseDepthCamera")
    return color.ColorCamera, ir.IRCamera, depth.RealSenseDepthCamera


def build_camera_kwargs(stream: str) -> Dict[str, object]:
    if stream == "rgb":
        return {
            "device": "/dev/video6",
            "in_w": 1280,
            "in_h": 720,
            "out_w": 1280,
            "out_h": 720,
            "fps": 30,
            "in_format": "YUY2",
            "format": "RGB",
            "crop_x": 0,
            "crop_y": 0,
            "crop_w": 0,
            "crop_h": 0,
            "auto_exposure": True,
        }
    if stream == "depth":
        return {"width": 640, "height": 480, "fps": 30}
    return {
        "device": "/dev/video4",
        "in_w": 1280,
        "in_h": 720,
        "out_w": 1280,
        "out_h": 720,
        "fps": 30,
        "in_format": "GRAY8",
        "format": "BGR",
    }


def open_camera(stream: str):
    kwargs = build_camera_kwargs(stream)
    color_cls, ir_cls, depth_cls = import_camera_classes()
    if stream == "depth":
        return depth_cls(**kwargs)
    if stream == "ir":
        return ir_cls(**kwargs)
    return color_cls(**kwargs)


def safe_release(obj) -> None:
    if obj is None:
        return
    try:
        obj.release()
    except Exception:
        pass


def depth_to_bgr(frame):
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame.copy()
    if str(frame.dtype) == "uint16":
        vis = cv2.convertScaleAbs(frame, alpha=0.03)
    elif str(frame.dtype) != "uint8":
        vis = frame.astype("uint8")
    else:
        vis = frame
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)


def ir_to_bgr(frame):
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame.copy()
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def rgb_to_bgr(frame):
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return ir_to_bgr(frame)


def frame_to_display(stream: str, frame):
    if stream == "rgb":
        return rgb_to_bgr(frame)
    if stream == "depth":
        return depth_to_bgr(frame)
    return ir_to_bgr(frame)


def overlay_status(image_bgr, stream: str, fps: float):
    title = f"{stream.upper()} CAMERA | {fps:.1f} FPS"
    cv2.putText(image_bgr, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return image_bgr


def print_demo_info(stream: str, requested_backend: str) -> None:
    print("=" * 72)
    print("VISTA demo_camera")
    print("=" * 72)
    print(f"stream={stream}")
    print(f"requested_backend={requested_backend}")
    print("=" * 72)


def main() -> int:
    args = parse_args()
    if args.backend != "real":
        raise SystemExit("demo_camera only supports --backend real")
    stream = choose_stream(args)
    print_demo_info(stream, args.backend)

    camera = None
    try:
        camera = open_camera(stream)
        window_name = "VISTA Demo Camera"
        if not args.headless:
            cv2.namedWindow(window_name)

        prev_time = time.time()
        frame_count = 0

        while True:
            frame = camera.read_frame()
            if frame is None or getattr(frame, "size", 0) == 0:
                if args.headless and args.max_frames and frame_count >= args.max_frames:
                    break
                if not args.headless:
                    cv2.waitKey(5)
                continue

            vis_bgr = frame_to_display(stream, frame)
            current_time = time.time()
            fps = 1.0 / max(1e-6, current_time - prev_time)
            prev_time = current_time
            vis_bgr = overlay_status(vis_bgr, stream, fps)

            if not args.headless:
                cv2.imshow(window_name, vis_bgr)

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break

            if not args.headless:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

        return 0
    finally:
        safe_release(camera)
        if not args.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
