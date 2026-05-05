#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="录制包含 RGB + Depth 的 RealSense .bag")
    parser.add_argument("--output", type=Path, default=Path("record_rgb_depth.bag"))
    parser.add_argument("--duration", type=float, default=10.0, help="录制时长，单位秒")
    parser.add_argument("--depth-width", type=int, default=424)
    parser.add_argument("--depth-height", type=int, default=240)
    parser.add_argument("--depth-fps", type=int, default=15)
    parser.add_argument("--color-width", type=int, default=640)
    parser.add_argument("--color-height", type=int, default=360)
    parser.add_argument("--color-fps", type=int, default=15)
    parser.add_argument("--preview", action="store_true", help="有显示环境时显示 RGB 和 Depth 预览")
    parser.add_argument("--save-preview", action="store_true", help="保存录制期间最后一帧的预览图")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    script_dir = Path(__file__).resolve().parent
    output_path = (script_dir / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gui_enabled = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    preview_enabled = bool(args.preview and gui_enabled)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.depth_width, args.depth_height, rs.format.z16, args.depth_fps)
    config.enable_stream(rs.stream.color, args.color_width, args.color_height, rs.format.rgb8, args.color_fps)
    config.enable_record_to_file(str(output_path))

    print("Start recording:")
    print("  output :", output_path)
    print("  depth  :", f"{args.depth_width}x{args.depth_height}@{args.depth_fps}")
    print("  color  :", f"{args.color_width}x{args.color_height}@{args.color_fps}")
    print("  duration_s :", args.duration)
    print("  gui_enabled :", gui_enabled)
    if args.preview and not gui_enabled:
        print("  preview disabled: no DISPLAY/WAYLAND_DISPLAY detected")

    profile = pipeline.start(config)
    device = profile.get_device()
    print("Device:", device)
    print("Sensors:", [sensor.get_info(rs.camera_info.name) for sensor in device.query_sensors()])

    align = rs.align(rs.stream.color)
    start_ts = time.time()
    frame_count = 0
    last_preview = None

    try:
        while True:
            now = time.time()
            if now - start_ts >= args.duration:
                break

            frames = pipeline.wait_for_frames(5000)
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            frame_count += 1
            depth_np = np.asanyarray(depth_frame.get_data())
            color_np = np.asanyarray(color_frame.get_data())
            color_bgr = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)
            depth_vis = cv2.convertScaleAbs(depth_np, alpha=0.03)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            preview = np.hstack([color_bgr, depth_vis])
            remain = max(0.0, args.duration - (now - start_ts))
            cv2.putText(preview, f"frames={frame_count} remain={remain:4.1f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            last_preview = preview

            if preview_enabled:
                cv2.imshow("RealSense Recorder", preview)
                if cv2.waitKey(1) & 0xFF == 27:
                    print("ESC pressed, stop recording early")
                    break
    finally:
        pipeline.stop()
        if preview_enabled:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    preview_path = None
    if last_preview is not None and (args.save_preview or args.preview):
        preview_path = output_path.with_name(output_path.stem + "_preview.png")
        cv2.imwrite(str(preview_path), last_preview)

    print("Finished recording")
    print("  frames:", frame_count)
    print("  saved :", output_path)
    if preview_path is not None:
        print("  preview:", preview_path)


if __name__ == "__main__":
    main()
