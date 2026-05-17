#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
VISTA_ROOT = HERE.parents[2]
for item in (REPO_ROOT, VISTA_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from VISTA.vision_module.backend.table_edge_roi import ROI_PRESETS, choose_depth_roi


cv2 = None
np = None
OnlineTableEdgeDetector = None
load_calib = None
DetectorConfig = None
PreviewFrame = None
PreviewOverlay = None
OpenCVPreviewSink = None


DEFAULT_BAG = VISTA_ROOT / "20260516_161436.bag"
DEFAULT_CALIB = VISTA_ROOT / "Offline_Edge_Test" / "calib.json"


def _load_runtime_deps() -> None:
    global cv2, np, OnlineTableEdgeDetector, load_calib, DetectorConfig, PreviewFrame, PreviewOverlay, OpenCVPreviewSink
    if np is not None:
        return
    import numpy as _np

    try:
        import aidcv as _cv2
    except ImportError:
        import cv2 as _cv2

    from VISTA.Online_Edge_Detect.detector import OnlineTableEdgeDetector as _OnlineTableEdgeDetector
    from VISTA.Online_Edge_Detect.detector import load_calib as _load_calib
    from VISTA.Online_Edge_Detect.schema import DetectorConfig as _DetectorConfig
    from VISTA.vision_module.backend.preview.base import PreviewFrame as _PreviewFrame
    from VISTA.vision_module.backend.preview.base import PreviewOverlay as _PreviewOverlay
    from VISTA.vision_module.backend.preview.opencv_sink import OpenCVPreviewSink as _OpenCVPreviewSink

    cv2 = _cv2
    np = _np
    OnlineTableEdgeDetector = _OnlineTableEdgeDetector
    load_calib = _load_calib
    DetectorConfig = _DetectorConfig
    PreviewFrame = _PreviewFrame
    PreviewOverlay = _PreviewOverlay
    OpenCVPreviewSink = _OpenCVPreviewSink


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a RealSense RGB+Depth bag through the table-edge detector.")
    parser.add_argument("--bag", type=Path, default=DEFAULT_BAG, help="RealSense .bag file path.")
    parser.add_argument("--stride", type=int, default=10, help="Process one frame every N recorded frames.")
    parser.add_argument("--start-frame", type=int, default=0, help="Skip frames before this 0-based frame index.")
    parser.add_argument("--max-frames", type=int, default=500, help="Maximum recorded frames to read.")
    parser.add_argument(
        "--roi-preset",
        default="center_lower",
        choices=sorted(ROI_PRESETS.keys()),
        help="Depth ROI preset from vision_module.backend.table_edge_roi.",
    )
    parser.add_argument("--show", action="store_true", help="Show the reused OpenCV preview overlay.")
    parser.add_argument("--save-dir", type=Path, default=None, help="Optional directory for sampled preview PNGs.")
    parser.add_argument("--calib-json", type=Path, default=DEFAULT_CALIB, help="Camera calibration JSON.")
    return parser


def _profile_summary(profile: Any) -> Dict[str, Any]:
    stream = profile.as_video_stream_profile()
    return {
        "stream_name": str(profile.stream_type()),
        "format": str(profile.format()),
        "width": int(stream.width()),
        "height": int(stream.height()),
        "fps": int(stream.fps()),
    }


def iter_bag_frames(bag_path: Path, max_frames: int) -> Iterator[Dict[str, Any]]:
    _load_runtime_deps()
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 is required to read RealSense .bag files") from exc

    if not bag_path.exists():
        raise FileNotFoundError(f"bag file not found: {bag_path}")

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(bag_path.expanduser().resolve()), repeat_playback=False)
    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)
    streams = [_profile_summary(p) for p in profile.get_streams()]
    print("[BAG_EDGE] opened bag=", bag_path)
    for item in streams:
        print("[BAG_EDGE] stream=", item)

    align = rs.align(rs.stream.color)
    frame_count = 0
    try:
        while max_frames <= 0 or frame_count < max_frames:
            try:
                frames = pipeline.wait_for_frames(3000)
            except RuntimeError as exc:
                print(f"[BAG_EDGE] playback stopped: {exc}")
                break
            if not frames:
                break
            frame_count += 1
            try:
                aligned = align.process(frames)
            except Exception:
                aligned = frames
            depth_frame = aligned.get_depth_frame() or frames.get_depth_frame()
            color_frame = aligned.get_color_frame() or frames.get_color_frame()
            if not depth_frame:
                continue
            yield {
                "frame": frame_count - 1,
                "ts_ms": float(frames.get_timestamp()),
                "depth": np.asanyarray(depth_frame.get_data()),
                "rgb": np.asanyarray(color_frame.get_data()) if color_frame else None,
            }
    finally:
        pipeline.stop()


def _fallback_roi(cfg: DetectorConfig) -> list[int]:
    return [int(cfg.roi_x0), int(cfg.roi_y0), int(cfg.roi_x1), int(cfg.roi_y1)]


def _table_edge_payload(
    result: Any,
    debug: Dict[str, Any],
    roi_meta: Dict[str, Any],
    frame_id: int,
    age_ms: float,
    target_dist_m: float,
    cfg: DetectorConfig,
) -> Dict[str, Any]:
    roi_box = debug.get("roi_box") if isinstance(debug, dict) else None
    roi = [int(v) for v in roi_box] if roi_box is not None else roi_meta.get("depth_edge_roi")
    edge_found = bool(getattr(result, "edge_found", False))
    yaw = float(getattr(result, "yaw_err_rad", 0.0)) if edge_found else None
    dist = float(getattr(result, "dist_err_m", 0.0)) if edge_found else None
    table_points = int(getattr(result, "table_point_count", 0) or 0)
    all_points = int(getattr(result, "point_count", 0) or 0)
    payload = {
        "table_found": bool(table_points > 0),
        "edge_found": edge_found,
        "edge_valid": edge_found,
        "valid": edge_found,
        "confidence": float(getattr(result, "edge_confidence", 0.0) or 0.0),
        "edge_conf": float(getattr(result, "edge_confidence", 0.0) or 0.0),
        "yaw_err_rad": yaw,
        "yaw_err": yaw,
        "dist_err_m": dist,
        "dist_err": dist,
        "edge_k": getattr(result, "line_k", None),
        "edge_b": getattr(result, "line_b", None),
        "depth_valid": True,
        "point_count": all_points,
        "valid_edge_points": all_points,
        "table_point_count": table_points,
        "edge_inlier_count": table_points,
        "selected_edge": edge_found,
        "near_edge": edge_found,
        "frame_id": int(frame_id),
        "frame_seq": int(frame_id),
        "source": "offline_bag_edge_debug",
        "reason": "" if edge_found else ("roi_empty" if all_points <= 0 and table_points <= 0 else "no_valid_edge"),
        "target_dist_m": float(target_dist_m),
        "age_ms": float(age_ms),
        "depth_z_min_m": float(cfg.z_min),
        "depth_z_max_m": float(cfg.z_max),
        "table_y_min_m": float(cfg.table_y_min),
        "table_y_max_m": float(cfg.table_y_max),
        "type": "table_edge_obs",
    }
    payload.update(dict(roi_meta or {}))
    payload.update({"depth_edge_roi": roi, "table_edge_roi": roi, "edge_roi": roi, "roi_format": "xyxy"})
    return payload


def _make_preview_canvas(sink: OpenCVPreviewSink, rgb: Any, depth: np.ndarray, table_edge: Dict[str, Any], frame_id: int) -> np.ndarray:
    panel_w = max(320, sink.canvas_w // 2)
    panel_h = max(220, sink.canvas_h // 2)
    panel_size = (panel_w, panel_h)
    metadata = {
        "preview_layout": "rgb_depth_edge",
        "runtime_status": {"stage": "OFFLINE_BAG", "mode": "TABLE_EDGE_PERCEPTION", "epoch": 0},
        "local_perception": {"rgb_shape": getattr(rgb, "shape", None), "box_count": 0},
        "table_edge_obs": table_edge,
        "target_obs": {},
        "source_cameras": ["rgb", "depth"] if isinstance(rgb, np.ndarray) else ["depth"],
        "window_id": sink.window_id,
        "show_age_ms": True,
        "frame_age_s": 0.0,
    }
    frame = PreviewFrame(
        ts=time.time(),
        image={"rgb": rgb, "depth": depth},
        stage="OFFLINE_BAG",
        mode="TABLE_EDGE_PERCEPTION",
        overlay=PreviewOverlay(title="Offline Bag Edge Debug", metadata=metadata),
    )
    rgb_panel = sink._make_rgb_panel(rgb, metadata, panel_size)
    depth_panel = sink._make_depth_panel(depth, table_edge, panel_size)
    edge_panel = sink._make_edge_panel(depth, table_edge, panel_size)
    info_panel = sink._make_info_panel(frame, metadata, table_edge, {}, panel_size)
    canvas = np.vstack([np.hstack([rgb_panel, depth_panel]), np.hstack([edge_panel, info_panel])])
    cv2.putText(canvas, f"bag_frame={frame_id}", (18, canvas.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 235, 240), 1)
    if canvas.shape[:2] != (sink.canvas_h, sink.canvas_w):
        canvas = cv2.resize(canvas, (sink.canvas_w, sink.canvas_h), interpolation=cv2.INTER_AREA)
    return canvas


def _preview_frame(rgb: Any, depth: np.ndarray, table_edge: Dict[str, Any], frame_id: int) -> PreviewFrame:
    metadata = {
        "preview_layout": "rgb_depth_edge",
        "runtime_status": {"stage": "OFFLINE_BAG", "mode": "TABLE_EDGE_PERCEPTION", "epoch": 0},
        "local_perception": {"rgb_shape": getattr(rgb, "shape", None), "box_count": 0},
        "table_edge_obs": table_edge,
        "target_obs": {},
        "source_cameras": ["rgb", "depth"] if isinstance(rgb, np.ndarray) else ["depth"],
        "show_age_ms": True,
        "frame_age_s": 0.0,
    }
    lines = [
        "stage=OFFLINE_BAG",
        "mode=TABLE_EDGE_PERCEPTION",
        f"frame={frame_id}",
        f"valid={int(bool(table_edge.get('edge_valid')))}",
        f"yaw={table_edge.get('yaw_err_rad')}",
        f"dist={table_edge.get('dist_err_m')}",
        f"age_ms={table_edge.get('age_ms')}",
    ]
    return PreviewFrame(time.time(), {"rgb": rgb, "depth": depth}, "OFFLINE_BAG", "TABLE_EDGE_PERCEPTION", PreviewOverlay("Offline Bag Edge Debug", lines, metadata=metadata))


def main() -> None:
    args = build_parser().parse_args()
    _load_runtime_deps()
    stride = max(1, int(args.stride))
    start_frame = max(0, int(args.start_frame))
    max_frames = int(args.max_frames)
    save_dir: Optional[Path] = args.save_dir
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    cfg = DetectorConfig()
    calib, target_dist = load_calib(args.calib_json.expanduser().resolve())
    if float(cfg.target_dist_m_override) > 0:
        target_dist = float(cfg.target_dist_m_override)
    detector = OnlineTableEdgeDetector(calib, cfg, target_dist)
    sink = OpenCVPreviewSink("Offline Bag Edge Debug")
    if args.show:
        sink.open()

    processed = 0
    try:
        for pack in iter_bag_frames(args.bag, max_frames=max_frames):
            frame_id = int(pack["frame"])
            if frame_id < start_frame or ((frame_id - start_frame) % stride) != 0:
                continue
            depth = pack["depth"]
            rgb = pack.get("rgb")
            t0 = time.perf_counter()
            roi_meta = choose_depth_roi(
                {},
                getattr(rgb, "shape", None),
                getattr(depth, "shape", None),
                _fallback_roi(cfg),
                manual_static=False,
                roi_preset=args.roi_preset,
            )
            result, debug = detector.process_depth(depth, roi_override=roi_meta.get("depth_edge_roi"))
            age_ms = (time.perf_counter() - t0) * 1000.0
            table_edge = _table_edge_payload(result, debug, roi_meta, frame_id, age_ms, target_dist, cfg)
            processed += 1

            print(
                "[BAG_EDGE] "
                f"frame={frame_id} valid={int(bool(table_edge.get('edge_valid')))} "
                f"dist={table_edge.get('dist_err_m')} yaw={table_edge.get('yaw_err_rad')} "
                f"roi={table_edge.get('depth_edge_roi')} preset={args.roi_preset} age_ms={age_ms:.1f}"
            )

            if save_dir is not None:
                canvas = _make_preview_canvas(sink, rgb, depth, table_edge, frame_id)
                cv2.imwrite(str(save_dir / f"bag_edge_{frame_id:06d}_{args.roi_preset}.png"), canvas)

            if args.show:
                keep_running = sink.render(_preview_frame(rgb, depth, table_edge, frame_id))
                key = cv2.waitKey(1) & 0xFF
                if not keep_running or key in (ord("q"), ord("Q")):
                    break
    finally:
        if args.show:
            sink.close()
    print(f"[BAG_EDGE] processed={processed} stride={stride} start_frame={start_frame} max_frames={max_frames}")


if __name__ == "__main__":
    main()
