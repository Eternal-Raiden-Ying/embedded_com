#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从当前接入的 RealSense 实时抓取一帧 depth/color，用于离线算法验证")
    parser.add_argument("--output-dir", type=Path, default=Path("test_data/live_capture"))
    parser.add_argument("--depth-width", type=int, default=424)
    parser.add_argument("--depth-height", type=int, default=240)
    parser.add_argument("--depth-fps", type=int, default=15)
    parser.add_argument("--color-backend", choices=["v4l2", "rs"], default="v4l2", help="RGB 抓取后端；板端默认建议走 /dev/video6")
    parser.add_argument("--color-device", default="/dev/video6", help="当 color-backend=v4l2 时使用的设备节点")
    parser.add_argument("--color-width", type=int, default=640)
    parser.add_argument("--color-height", type=int, default=360)
    parser.add_argument("--color-fps", type=int, default=15)
    parser.add_argument("--enable-color", action="store_true")
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--preview", action="store_true", help="有显示环境时弹出预览窗口")
    parser.add_argument("--save-preview", action="store_true", help="保存伪彩预览图")
    return parser


def save_depth_png(path: Path, depth_frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if depth_frame.dtype != np.uint16:
        depth_frame = depth_frame.astype(np.uint16)
    cv2.imwrite(str(path), depth_frame)


def save_color_png(path: Path, color_frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def make_depth_preview(depth_frame: np.ndarray, color_frame: Optional[np.ndarray] = None) -> np.ndarray:
    depth_vis = cv2.convertScaleAbs(depth_frame, alpha=0.03)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    if color_frame is None:
        return depth_vis
    color_bgr = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
    h = depth_vis.shape[0]
    w = int(color_bgr.shape[1] * h / max(1, color_bgr.shape[0]))
    color_thumb = cv2.resize(color_bgr, (w, h))
    return np.hstack([depth_vis, color_thumb])


def save_preview_png(path: Path, preview: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), preview)


def main() -> None:
    args = build_parser().parse_args()
    script_dir = Path(__file__).resolve().parent
    output_dir = (script_dir / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gui_enabled = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    preview_enabled = bool(args.preview and gui_enabled)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.depth_width, args.depth_height, rs.format.z16, args.depth_fps)
    if args.enable_color and args.color_backend == "rs":
        config.enable_stream(rs.stream.color, args.color_width, args.color_height, rs.format.rgb8, args.color_fps)

    print("Starting live snapshot capture:")
    print("  output_dir :", output_dir)
    print("  depth      :", f"{args.depth_width}x{args.depth_height}@{args.depth_fps}")
    print("  color      :", ("enabled via " + args.color_backend) if args.enable_color else "disabled")
    print("  gui_enabled:", gui_enabled)
    if args.preview and not gui_enabled:
        print("  preview disabled: no DISPLAY/WAYLAND_DISPLAY detected; will save preview file instead")

    profile = pipeline.start(config)
    device = profile.get_device()
    print("Device:", device)
    print("Sensors:", [sensor.get_info(rs.camera_info.name) for sensor in device.query_sensors()])

    align = rs.align(rs.stream.color) if args.enable_color and args.color_backend == "rs" else None
    depth_np: Optional[np.ndarray] = None
    color_np: Optional[np.ndarray] = None
    color_cap = None

    if args.enable_color and args.color_backend == "v4l2":
        backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 0
        color_cap = cv2.VideoCapture(args.color_device, backend)
        if not color_cap.isOpened():
            pipeline.stop()
            raise RuntimeError(f"failed to open color device: {args.color_device}")
        color_cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.color_width))
        color_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.color_height))
        color_cap.set(cv2.CAP_PROP_FPS, float(args.color_fps))

    try:
        for idx in range(max(1, int(args.warmup_frames))):
            frames = pipeline.wait_for_frames(int(args.timeout_ms))
            if align is not None:
                frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame() if args.enable_color and args.color_backend == "rs" else None
            if not depth_frame:
                continue

            depth_np = np.asanyarray(depth_frame.get_data())
            if color_frame:
                color_np = np.asanyarray(color_frame.get_data())
            elif color_cap is not None:
                ok, color_bgr = color_cap.read()
                if ok and color_bgr is not None and color_bgr.size > 0:
                    color_np = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

            if preview_enabled:
                preview = make_depth_preview(depth_np, color_np)
                cv2.putText(
                    preview,
                    f"warmup={idx + 1}/{args.warmup_frames}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("RealSense Live Snapshot", preview)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        pipeline.stop()
        if color_cap is not None:
            color_cap.release()
        if preview_enabled:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    if depth_np is None:
        raise RuntimeError("depth frame was not captured; camera may be occupied or stream config unsupported")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    depth_path = output_dir / f"depth_{stamp}.png"
    save_depth_png(depth_path, depth_np)

    color_path = None
    if color_np is not None:
        color_path = output_dir / f"color_{stamp}.png"
        save_color_png(color_path, color_np)

    preview_path = None
    if args.save_preview or args.preview:
        preview_path = output_dir / f"preview_{stamp}.png"
        preview = make_depth_preview(depth_np, color_np)
        save_preview_png(preview_path, preview)

    summary = {
        "depth_path": str(depth_path),
        "color_path": str(color_path) if color_path else None,
        "preview_path": str(preview_path) if preview_path else None,
        "depth_shape": list(depth_np.shape),
        "color_shape": list(color_np.shape) if color_np is not None else None,
        "depth_min": int(depth_np.min()),
        "depth_max": int(depth_np.max()),
        "depth_nonzero": int(np.count_nonzero(depth_np)),
        "color_backend": (args.color_backend if args.enable_color else None),
        "color_device": (args.color_device if args.enable_color and args.color_backend == "v4l2" else None),
        "gui_enabled": gui_enabled,
        "preview_enabled": preview_enabled,
    }
    summary_path = output_dir / f"capture_{stamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("Capture finished:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
