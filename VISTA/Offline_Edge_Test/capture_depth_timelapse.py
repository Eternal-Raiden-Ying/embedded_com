#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按固定时间间隔连续抓取 RealSense 深度图，可选同时抓取 RGB")
    parser.add_argument("--output-dir", type=Path, default=Path("test_data/timelapse_capture"))
    parser.add_argument("--duration", type=float, default=30.0, help="总采集时长，单位秒")
    parser.add_argument("--interval", type=float, default=1.0, help="两次保存之间的间隔，单位秒")
    parser.add_argument("--depth-width", type=int, default=424)
    parser.add_argument("--depth-height", type=int, default=240)
    parser.add_argument("--depth-fps", type=int, default=15)
    parser.add_argument("--warmup-seconds", type=float, default=2.0, help="启动后预热时长，单位秒")
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--save-preview", action="store_true", help="同时保存伪彩深度预览图")
    parser.add_argument("--enable-color", action="store_true", help="同时抓取 RGB")
    parser.add_argument("--color-backend", choices=["v4l2", "rs"], default="v4l2", help="RGB 抓取后端；板端推荐 v4l2")
    parser.add_argument("--color-device", default="/dev/video6", help="当 color-backend=v4l2 时使用的设备节点")
    parser.add_argument("--color-width", type=int, default=640)
    parser.add_argument("--color-height", type=int, default=360)
    parser.add_argument("--color-fps", type=int, default=15)
    return parser


def save_depth_png(path: Path, depth_frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if depth_frame.dtype != np.uint16:
        depth_frame = depth_frame.astype(np.uint16)
    cv2.imwrite(str(path), depth_frame)


def save_color_png(path: Path, color_frame_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    color_bgr = cv2.cvtColor(color_frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), color_bgr)


def make_depth_preview(depth_frame: np.ndarray) -> np.ndarray:
    depth_vis = cv2.convertScaleAbs(depth_frame, alpha=0.06)
    return cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)


def make_combined_preview(depth_frame: np.ndarray, color_frame_rgb: Optional[np.ndarray] = None) -> np.ndarray:
    depth_vis = make_depth_preview(depth_frame)
    if color_frame_rgb is None:
        return depth_vis
    color_bgr = cv2.cvtColor(color_frame_rgb, cv2.COLOR_RGB2BGR)
    target_h = depth_vis.shape[0]
    target_w = int(color_bgr.shape[1] * target_h / max(1, color_bgr.shape[0]))
    color_panel = cv2.resize(color_bgr, (target_w, target_h))
    gap = 12
    canvas = cv2.copyMakeBorder(
        depth_vis,
        0,
        0,
        0,
        gap + color_panel.shape[1],
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    canvas[:, depth_vis.shape[1]:depth_vis.shape[1] + gap] = 0
    canvas[:, depth_vis.shape[1] + gap:depth_vis.shape[1] + gap + color_panel.shape[1]] = color_panel
    cv2.putText(canvas, "DEPTH", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "RGB", (depth_vis.shape[1] + gap + 20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def save_preview_png(path: Path, preview: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), preview)


def main() -> None:
    args = build_parser().parse_args()
    script_dir = Path(__file__).resolve().parent
    output_dir = (script_dir / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.depth_width, args.depth_height, rs.format.z16, args.depth_fps)
    if args.enable_color and args.color_backend == "rs":
        config.enable_stream(rs.stream.color, args.color_width, args.color_height, rs.format.rgb8, args.color_fps)

    print("Starting depth timelapse capture:")
    print("  output_dir     :", output_dir)
    print("  duration_s     :", args.duration)
    print("  interval_s     :", args.interval)
    print("  depth          :", f"{args.depth_width}x{args.depth_height}@{args.depth_fps}")
    print("  warmup_seconds :", args.warmup_seconds)
    print("  color          :", ("enabled via " + args.color_backend) if args.enable_color else "disabled")

    profile = pipeline.start(config)
    device = profile.get_device()
    print("Device:", device)
    print("Sensors:", [sensor.get_info(rs.camera_info.name) for sensor in device.query_sensors()])

    align = rs.align(rs.stream.color) if args.enable_color and args.color_backend == "rs" else None
    color_cap = None
    if args.enable_color and args.color_backend == "v4l2":
        backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 0
        color_cap = cv2.VideoCapture(args.color_device, backend)
        if not color_cap.isOpened():
            pipeline.stop()
            raise RuntimeError("failed to open color device: %s" % args.color_device)
        color_cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.color_width))
        color_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.color_height))
        color_cap.set(cv2.CAP_PROP_FPS, float(args.color_fps))

    warmup_deadline = time.time() + max(0.0, float(args.warmup_seconds))
    end_deadline = warmup_deadline + max(0.0, float(args.duration))
    next_capture_ts = warmup_deadline

    captures = []
    frame_counter = 0

    try:
        while time.time() < end_deadline:
            frames = pipeline.wait_for_frames(int(args.timeout_ms))
            if align is not None:
                frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame() if args.enable_color and args.color_backend == "rs" else None
            if not depth_frame:
                continue

            depth_np = np.asanyarray(depth_frame.get_data())
            color_np = None
            if color_frame:
                color_np = np.asanyarray(color_frame.get_data())
            elif color_cap is not None:
                ok, color_bgr = color_cap.read()
                if ok and color_bgr is not None and color_bgr.size > 0:
                    color_np = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

            now = time.time()
            if now < next_capture_ts:
                continue

            frame_counter += 1
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
            millis = int((now - int(now)) * 1000.0)
            stem = "capture_%s_%03d" % (stamp, millis)

            depth_path = output_dir / ("%s_depth.png" % stem)
            save_depth_png(depth_path, depth_np)

            color_path = None
            if color_np is not None:
                color_path = output_dir / ("%s_color.png" % stem)
                save_color_png(color_path, color_np)

            preview_path = None
            if args.save_preview:
                preview_path = output_dir / ("%s_preview.png" % stem)
                preview = make_combined_preview(depth_np, color_np)
                save_preview_png(preview_path, preview)

            item = {
                "index": frame_counter,
                "ts": now,
                "ts_ms_sensor": float(frames.get_timestamp()),
                "depth_path": str(depth_path),
                "color_path": str(color_path) if color_path is not None else None,
                "preview_path": str(preview_path) if preview_path is not None else None,
                "depth_shape": list(depth_np.shape),
                "color_shape": list(color_np.shape) if color_np is not None else None,
                "depth_min": int(depth_np.min()),
                "depth_max": int(depth_np.max()),
                "depth_nonzero": int(np.count_nonzero(depth_np)),
            }
            captures.append(item)
            print(
                "[%02d] saved depth=%s color=%s nonzero=%d max=%d" % (
                    frame_counter,
                    depth_path.name,
                    color_path.name if color_path is not None else "None",
                    item["depth_nonzero"],
                    item["depth_max"],
                )
            )
            next_capture_ts += max(0.05, float(args.interval))
    finally:
        pipeline.stop()
        if color_cap is not None:
            color_cap.release()

    summary = {
        "output_dir": str(output_dir),
        "duration_s": float(args.duration),
        "interval_s": float(args.interval),
        "depth_width": int(args.depth_width),
        "depth_height": int(args.depth_height),
        "depth_fps": int(args.depth_fps),
        "enable_color": bool(args.enable_color),
        "color_backend": args.color_backend if args.enable_color else None,
        "color_device": args.color_device if args.enable_color and args.color_backend == "v4l2" else None,
        "capture_count": len(captures),
        "captures": captures,
    }
    summary_path = output_dir / "timelapse_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("Finished timelapse capture")
    print("  capture_count:", len(captures))
    print("  summary      :", summary_path)


if __name__ == "__main__":
    main()
