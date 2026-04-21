#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Tuple

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

from test_support import EXIT_FAIL, EXIT_INTERRUPT, EXIT_OK, describe_frame, safe_release


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real color camera exposure/brightness control test")
    parser.add_argument("--device", default="/dev/video6")
    parser.add_argument("--in-w", type=int, default=1280)
    parser.add_argument("--in-h", type=int, default=720)
    parser.add_argument("--out-w", type=int, default=1280)
    parser.add_argument("--out-h", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--step-secs", type=float, default=2.0)
    parser.add_argument("--start-exposure", type=int, default=80)
    parser.add_argument("--min-exposure", type=int, default=20)
    parser.add_argument("--max-exposure", type=int, default=300)
    parser.add_argument("--exposure-step", type=int, default=20)
    parser.add_argument("--start-brightness", type=int, default=0)
    parser.add_argument("--min-brightness", type=int, default=-32)
    parser.add_argument("--max-brightness", type=int, default=32)
    parser.add_argument("--brightness-step", type=int, default=8)
    parser.add_argument("--headless", action="store_true")
    return parser


def import_color_camera():
    module = importlib.import_module("vision_module.backend.camera.ColorCamera")
    return module.ColorCamera


def fit_preview(image_bgr, max_width: int = 1280, max_height: int = 720):
    h, w = image_bgr.shape[:2]
    scale = min(float(max_width) / float(w), float(max_height) / float(h))
    if scale <= 0 or abs(scale - 1.0) < 0.05:
        return image_bgr
    return cv2.resize(image_bgr, (max(1, int(w * scale)), max(1, int(h * scale))))


def overlay_status(image_bgr, fps: float, exposure: int, brightness: int):
    cv2.putText(image_bgr, f"COLOR CAMERA | {fps:.1f} FPS", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(image_bgr, f"exposure={exposure} brightness={brightness}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return image_bgr


def bounce(value: int, step: int, minimum: int, maximum: int) -> Tuple[int, int]:
    candidate = value + step
    if candidate > maximum or candidate < minimum:
        step = -step
        candidate = value + step
    return candidate, step


def main() -> int:
    args = build_parser().parse_args()
    ColorCamera = import_color_camera()
    camera = None
    window_name = "VISTA Color Controls"
    exposure = args.start_exposure
    brightness = args.start_brightness
    exposure_step = args.exposure_step
    brightness_step = args.brightness_step

    print("=" * 72)
    print("VISTA Color Camera Controls Test")
    print("=" * 72)
    print(f"device={args.device} resolution={args.out_w}x{args.out_h} fps={args.fps}")
    print(f"duration={args.duration}s step_secs={args.step_secs}s")
    print("=" * 72)

    try:
        camera = ColorCamera(
            device=args.device,
            in_w=args.in_w,
            in_h=args.in_h,
            out_w=args.out_w,
            out_h=args.out_h,
            fps=args.fps,
            in_format="YUY2",
            format="BGR",
            auto_exposure=False,
            exposure=exposure,
            brightness=brightness,
        )
        print(f"PASS     | init | {type(camera).__name__}")

        if not args.headless:
            cv2.namedWindow(window_name)

        started = time.time()
        last_tick = started
        prev_time = started
        frame_count = 0

        while True:
            frame = camera.read_frame()
            if frame is None or getattr(frame, "size", 0) == 0:
                print("FAIL     | frame | empty frame")
                return EXIT_FAIL

            bgr = frame.copy()
            now = time.time()
            fps = 1.0 / max(1e-6, now - prev_time)
            prev_time = now
            bgr = overlay_status(bgr, fps, exposure, brightness)
            bgr = fit_preview(bgr)

            if not args.headless:
                cv2.imshow(window_name, bgr)

            if now - last_tick >= args.step_secs:
                exposure, exposure_step = bounce(exposure, exposure_step, args.min_exposure, args.max_exposure)
                brightness, brightness_step = bounce(brightness, brightness_step, args.min_brightness, args.max_brightness)
                camera.set_exposure(exposure)
                camera.set_brightness(brightness)
                print(f"PASS     | adjust | exposure={exposure} brightness={brightness}")
                last_tick = now

            frame_count += 1
            if now - started >= args.duration:
                break
            if not args.headless:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

        print(f"PASS     | summary | frames={frame_count} final_exposure={exposure} final_brightness={brightness}")
        return EXIT_OK
    except KeyboardInterrupt:
        print("\nInterrupted")
        return EXIT_INTERRUPT
    except Exception as exc:
        print(f"FAIL     | error | {exc}")
        return EXIT_FAIL
    finally:
        safe_release(camera)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
