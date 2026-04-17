#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pyrealsense2 as rs


def load_calib(json_path: Path):
    from TableEdgeDetector import CameraCalib

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    calib = CameraCalib(
        fx=data["fx"],
        fy=data["fy"],
        cx=data["cx"],
        cy=data["cy"],
        depth_scale=data["depth_scale"],
    )
    return calib, float(data["target_dist_m"])


def frame_to_array(frame) -> np.ndarray:
    return np.asanyarray(frame.get_data())


def save_depth_png(path: Path, depth_frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if depth_frame.dtype != np.uint16:
        depth_frame = depth_frame.astype(np.uint16)
    cv2.imwrite(str(path), depth_frame)


def save_color_png(path: Path, color_frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if color_frame.ndim == 3 and color_frame.shape[2] == 3:
        bgr = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
    else:
        bgr = color_frame
    cv2.imwrite(str(path), bgr)


def save_preview_png(path: Path, preview: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), preview)


def profile_summary(profile) -> Dict:
    stream = profile.as_video_stream_profile()
    return {
        "stream_name": str(profile.stream_type()),
        "format": str(profile.format()),
        "width": int(stream.width()),
        "height": int(stream.height()),
        "fps": int(stream.fps()),
    }


def run_table_edge_detector(depth_png_path: Path, calib_json_path: Path) -> Dict:
    from TableEdgeDetector import TableEdgeDetector

    calib, target_dist = load_calib(calib_json_path)
    detector = TableEdgeDetector(calib, target_dist_m=target_dist)
    depth_raw = cv2.imread(str(depth_png_path), cv2.IMREAD_ANYDEPTH)
    if depth_raw is None:
        raise RuntimeError(f"failed to load depth png: {depth_png_path}")
    res, _, pc_all, pc_table = detector.process_offline(depth_raw)
    return {
        "depth_png": str(depth_png_path),
        "edge_found": bool(res.edge_found),
        "yaw_err_rad": float(res.yaw_err_rad),
        "dist_err_m": float(res.dist_err_m),
        "edge_confidence": float(res.edge_confidence),
        "all_point_count": int(len(pc_all)) if pc_all is not None else 0,
        "table_point_count": int(len(pc_table)) if pc_table is not None else 0,
    }


def read_bag(
    bag_path: Path,
    max_frames: int,
    save_dir: Optional[Path],
    save_index: int,
    preview: bool,
    run_detector: bool,
    calib_json: Optional[Path],
) -> Dict:
    if not bag_path.exists():
        raise FileNotFoundError(f"bag file not found: {bag_path}")

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(bag_path), repeat_playback=False)

    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

    sensor_names = [sensor.get_info(rs.camera_info.name) for sensor in device.query_sensors()]
    stream_profiles = [profile_summary(p) for p in profile.get_streams()]
    has_depth_stream = any(item["stream_name"] == "stream.depth" for item in stream_profiles)
    has_color_stream = any(item["stream_name"] == "stream.color" for item in stream_profiles)

    print("Opened bag:", bag_path)
    print("Device sensors:", sensor_names)
    print("Available recorded streams:")
    for item in stream_profiles:
        print("  -", json.dumps(item, ensure_ascii=False))

    gui_enabled = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    preview_enabled = bool(preview and gui_enabled)
    align = rs.align(rs.stream.color)

    frame_count = 0
    saved_depth = None
    saved_color = None
    saved_preview = None
    detector_result = None
    detector_skipped_reason = None
    last_depth_shape = None
    last_color_shape = None
    first_ts_ms = None
    last_ts_ms = None
    last_preview = None

    try:
        while frame_count < max_frames:
            try:
                frames = pipeline.wait_for_frames(3000)
            except RuntimeError as exc:
                print("wait_for_frames stopped:", exc)
                break

            if not frames:
                break

            try:
                aligned = align.process(frames)
            except Exception:
                aligned = frames

            depth_frame = aligned.get_depth_frame() or frames.get_depth_frame()
            color_frame = aligned.get_color_frame() or frames.get_color_frame()
            if not depth_frame and not color_frame:
                continue

            frame_count += 1
            ts_ms = frames.get_timestamp()
            if first_ts_ms is None:
                first_ts_ms = ts_ms
            last_ts_ms = ts_ms

            depth_np = None
            color_np = None
            if depth_frame:
                depth_np = frame_to_array(depth_frame)
                last_depth_shape = tuple(depth_np.shape)
                if frame_count == save_index and save_dir is not None:
                    saved_depth = save_dir / f"depth_frame_{frame_count:04d}.png"
                    save_depth_png(saved_depth, depth_np)

            if color_frame:
                color_np = frame_to_array(color_frame)
                last_color_shape = tuple(color_np.shape)
                if frame_count == save_index and save_dir is not None:
                    saved_color = save_dir / f"color_frame_{frame_count:04d}.png"
                    save_color_png(saved_color, color_np)

            if depth_np is not None:
                depth_vis = cv2.convertScaleAbs(depth_np, alpha=0.03)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                if color_np is not None:
                    color_bgr = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)
                    h = depth_vis.shape[0]
                    w = int(color_bgr.shape[1] * h / max(1, color_bgr.shape[0]))
                    color_thumb = cv2.resize(color_bgr, (w, h))
                    last_preview = np.hstack([depth_vis, color_thumb])
                else:
                    last_preview = depth_vis

            if preview_enabled and last_preview is not None:
                cv2.putText(last_preview, f"frame={frame_count} ts_ms={ts_ms:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("RealSense Bag Preview", last_preview)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        pipeline.stop()
        if preview_enabled:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    if save_dir is not None and last_preview is not None:
        saved_preview = save_dir / f"preview_frame_{max(1, frame_count):04d}.png"
        save_preview_png(saved_preview, last_preview)

    if run_detector:
        if saved_depth is None:
            detector_skipped_reason = "no saved depth frame available; this bag likely does not contain a depth stream"
        elif calib_json is None:
            detector_skipped_reason = "calib json not provided"
        else:
            try:
                detector_result = run_table_edge_detector(saved_depth, calib_json)
            except Exception as exc:
                detector_skipped_reason = f"failed to run TableEdgeDetector: {exc}"

    diagnosis = []
    if not has_depth_stream and has_color_stream:
        diagnosis.append("recorded bag contains color stream only")
    if "Stereo Module" in sensor_names and not has_depth_stream:
        diagnosis.append("device has depth sensor, but depth frames were not recorded into this bag")
        diagnosis.append("most likely depth stream was not enabled in RealSense Viewer before pressing Record")

    return {
        "bag_path": str(bag_path),
        "device_sensors": sensor_names,
        "stream_profiles": stream_profiles,
        "frame_count": frame_count,
        "first_ts_ms": first_ts_ms,
        "last_ts_ms": last_ts_ms,
        "has_depth_stream": has_depth_stream,
        "has_color_stream": has_color_stream,
        "last_depth_shape": last_depth_shape,
        "last_color_shape": last_color_shape,
        "saved_depth": str(saved_depth) if saved_depth else None,
        "saved_color": str(saved_color) if saved_color else None,
        "saved_preview": str(saved_preview) if saved_preview else None,
        "detector_result": detector_result,
        "detector_skipped_reason": detector_skipped_reason,
        "diagnosis": diagnosis,
        "gui_enabled": gui_enabled,
        "preview_enabled": preview_enabled,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="读取 realsense-viewer 录制的 .bag 文件，导出样例帧，并在有 depth 时直接跑 TableEdgeDetector")
    parser.add_argument("--bag", type=Path, default=Path("../../desk_scene.bag"))
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--save-dir", type=Path, default=Path("bag_extract"))
    parser.add_argument("--save-index", type=int, default=1, help="保存第几帧")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--run-detector", action="store_true", help="如果导出了 depth png，就直接调用 TableEdgeDetector")
    parser.add_argument("--calib-json", type=Path, default=Path("calib.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    script_dir = Path(__file__).resolve().parent
    bag_path = (script_dir / args.bag).resolve() if not args.bag.is_absolute() else args.bag.resolve()
    save_dir = (script_dir / args.save_dir).resolve() if not args.save_dir.is_absolute() else args.save_dir.resolve()
    calib_json = (script_dir / args.calib_json).resolve() if not args.calib_json.is_absolute() else args.calib_json.resolve()

    t0 = time.time()
    result = read_bag(
        bag_path=bag_path,
        max_frames=max(1, int(args.max_frames)),
        save_dir=save_dir,
        save_index=max(1, int(args.save_index)),
        preview=bool(args.preview),
        run_detector=bool(args.run_detector),
        calib_json=calib_json if args.calib_json else None,
    )
    result["elapsed_s"] = round(time.time() - t0, 3)
    print("\nSummary:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
