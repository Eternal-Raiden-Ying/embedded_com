#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterator, List, Optional


HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
VISTA_ROOT = HERE.parents[2]
for item in (REPO_ROOT, VISTA_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a RealSense bag through the online table-plane processing path.")
    parser.add_argument("--bag", type=Path, required=True, help="RealSense .bag file path.")
    parser.add_argument("--config", type=Path, default=VISTA_ROOT / "configs" / "vision_params.yaml", help="Vision params YAML.")
    parser.add_argument("--output", type=Path, required=True, help="Output run directory.")
    parser.add_argument("--mode", choices=("benchmark", "fixed-hz", "realtime"), default="benchmark")
    parser.add_argument("--hz", type=float, default=5.0, help="Input cadence for fixed-hz/realtime modes.")
    parser.add_argument("--stride", type=int, default=1, help="Process one frame every N recorded frames.")
    parser.add_argument("--start-frame", type=int, default=0, help="Skip recorded frames before this 0-based index.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum recorded frames to read, 0 means all.")
    parser.add_argument("--roi-preset", default="", help="Optional table plane ROI preset override.")
    preview = parser.add_mutually_exclusive_group()
    preview.add_argument("--preview", action="store_true", help="Show the same RGB/depth/plane preview layout.")
    preview.add_argument("--no-preview", action="store_true", help="Disable preview.")
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
    import numpy as np
    import pyrealsense2 as rs

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
    print(f"[BAG_TABLE_PLANE] opened bag={bag_path}")
    for item in streams:
        print(f"[BAG_TABLE_PLANE] stream={item}")

    align = rs.align(rs.stream.color)
    frame_count = 0
    try:
        while max_frames <= 0 or frame_count < max_frames:
            try:
                frames = pipeline.wait_for_frames(3000)
            except RuntimeError as exc:
                print(f"[BAG_TABLE_PLANE] playback stopped: {exc}")
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
                "timestamp_ms": float(frames.get_timestamp()),
                "depth": np.asanyarray(depth_frame.get_data()),
                "rgb": np.asanyarray(color_frame.get_data()) if color_frame else None,
            }
    finally:
        pipeline.stop()


def _json_ready(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if np is not None and isinstance(value, np.ndarray):
        if value.size > 4096:
            return {"shape": list(value.shape), "dtype": str(value.dtype), "omitted": True}
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return str(value)


def _finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _percentiles(values: List[float], include_max: bool = True) -> Dict[str, Optional[float]]:
    vals = sorted(v for v in (_finite_float(v) for v in values) if v is not None)
    if not vals:
        out: Dict[str, Optional[float]] = {"p50": None, "p90": None}
        if include_max:
            out["max"] = None
        return out

    def pick(q: float) -> float:
        if len(vals) == 1:
            return float(vals[0])
        pos = q * float(len(vals) - 1)
        lo = int(pos)
        hi = min(len(vals) - 1, lo + 1)
        frac = pos - float(lo)
        return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)

    out = {"p50": pick(0.50), "p90": pick(0.90)}
    if include_max:
        out["max"] = float(vals[-1])
    return out


def _metrics_summary(frames_total: int, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    obs_total = len(observations)
    roi_sources = Counter(str(obs.get("roi_source") or "unknown") for obs in observations)
    found_count = sum(1 for obs in observations if bool(obs.get("table_found") or obs.get("plane_found") or obs.get("edge_found")))
    control_valid_count = sum(1 for obs in observations if bool(obs.get("valid_for_control") or obs.get("edge_valid")))
    stale_count = sum(1 for obs in observations if bool(obs.get("is_stale")))
    return {
        "frames_total": int(frames_total),
        "obs_total": int(obs_total),
        "valid_obs_ratio": float(control_valid_count) / float(obs_total) if obs_total else 0.0,
        "process_ms": _percentiles([obs.get("process_ms", obs.get("vision_process_ms")) for obs in observations]),
        "obs_total_age_ms": _percentiles([obs.get("obs_total_age_ms") for obs in observations]),
        "update_interval_ms": _percentiles([obs.get("update_interval_ms", obs.get("edge_update_interval_ms")) for obs in observations], include_max=False),
        "stale_ratio": float(stale_count) / float(obs_total) if obs_total else 0.0,
        "roi_source": dict(sorted(roi_sources.items())),
        "found_ratio": float(found_count) / float(obs_total) if obs_total else 0.0,
        "control_valid_ratio": float(control_valid_count) / float(obs_total) if obs_total else 0.0,
    }


def _make_preview_frame(rgb: Any, depth: Any, obs: Dict[str, Any], frame_seq: int):
    from VISTA.vision_module.backend.preview.base import PreviewFrame, PreviewOverlay

    metadata = {
        "preview_layout": "rgb_depth_edge",
        "runtime_status": {"stage": "OFFLINE_BAG", "mode": "TABLE_EDGE_PERCEPTION", "epoch": 0},
        "local_perception": {"rgb_shape": getattr(rgb, "shape", None), "box_count": 0},
        # Legacy route name retained so online/offline use the same preview sink.
        "table_edge_obs": obs,
        "target_obs": {},
        "source_cameras": ["rgb", "depth"] if rgb is not None else ["depth"],
        "show_age_ms": True,
        "show_yolo_boxes": False,
        "frame_age_s": 0.0,
    }
    lines = [
        "stage=OFFLINE_BAG",
        "mode=TABLE_EDGE_PERCEPTION",
        f"frame={frame_seq}",
        f"valid={int(bool(obs.get('valid_for_control') or obs.get('edge_valid')))}",
        f"roi={obs.get('roi_source')}",
        f"process_ms={obs.get('process_ms')}",
    ]
    return PreviewFrame(time.time(), {"rgb": rgb, "depth": depth}, "OFFLINE_BAG", "TABLE_EDGE_PERCEPTION", PreviewOverlay("Bag Table Plane Replay", lines, metadata=metadata))


def _sleep_for_mode(mode: str, hz: float, next_deadline: Optional[float]) -> Optional[float]:
    if mode == "benchmark":
        return None
    period_s = 1.0 / max(0.1, float(hz or 0.0))
    now = time.perf_counter()
    if next_deadline is None:
        return now + period_s
    remaining = next_deadline - now
    if remaining > 0:
        time.sleep(remaining)
    return max(next_deadline + period_s, time.perf_counter())


def main() -> None:
    args = build_parser().parse_args()
    os.environ["VISION_PARAMS_FILE"] = str(args.config.expanduser().resolve())

    from VISTA.vision_module.config.board_config import CONFIG
    from VISTA.vision_module.backend.table_edge_manager import TableEdgeManager

    if args.roi_preset:
        CONFIG.table_edge.roi_preset = str(args.roi_preset).strip().lower()

    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    obs_path = output_dir / "table_edge_obs.jsonl"
    metrics_path = output_dir / "metrics_summary.json"

    processor = TableEdgeManager(cfg=CONFIG)
    preview_sink = None
    if args.preview and not args.no_preview:
        from VISTA.vision_module.backend.preview.opencv_sink import OpenCVPreviewSink

        preview_sink = OpenCVPreviewSink("Bag Table Plane Replay")
        preview_sink.open()

    frames_total = 0
    observations: List[Dict[str, Any]] = []
    stride = max(1, int(args.stride))
    start_frame = max(0, int(args.start_frame))
    next_deadline: Optional[float] = None

    try:
        with obs_path.open("w", encoding="utf-8") as fp:
            for pack in iter_bag_frames(args.bag, int(args.max_frames)):
                frame_seq = int(pack["frame"])
                frames_total += 1
                if frame_seq < start_frame or ((frame_seq - start_frame) % stride) != 0:
                    continue
                next_deadline = _sleep_for_mode(args.mode, args.hz, next_deadline)
                capture_ts = time.time()
                frames = {
                    "rgb": pack.get("rgb"),
                    "depth": pack.get("depth"),
                    "frame_capture_ts": capture_ts,
                    "timestamp_ms": pack.get("timestamp_ms"),
                }
                obs = processor.process_camera_frame(
                    frames,
                    frame_seq=frame_seq,
                    frame_slot={"seq": frame_seq, "ts": capture_ts, "payload": frames},
                    local_perception={"has_infer": False, "box_count": 0, "infer_boxes": [], "rgb_shape": getattr(pack.get("rgb"), "shape", None)},
                    runtime_status={"stage": "OFFLINE_BAG", "mode": "TABLE_EDGE_PERCEPTION"},
                    source_mode="OFFLINE_BAG",
                    count_dropped=False,
                )
                obs["source"] = "offline_bag_table_plane_replay"
                obs["bag_timestamp_ms"] = float(pack.get("timestamp_ms", 0.0) or 0.0)
                clean_obs = _json_ready(obs)
                fp.write(json.dumps(clean_obs, ensure_ascii=False, sort_keys=True) + "\n")
                observations.append(clean_obs)
                print(
                    "[BAG_TABLE_PLANE] "
                    f"frame_seq={frame_seq} valid={int(bool(obs.get('valid_for_control') or obs.get('edge_valid')))} "
                    f"found={int(bool(obs.get('table_found') or obs.get('plane_found') or obs.get('edge_found')))} "
                    f"roi_source={obs.get('roi_source')} roi={obs.get('plane_roi') or obs.get('depth_edge_roi')} "
                    f"process_ms={float(obs.get('process_ms') or 0.0):.1f} "
                    f"obs_total_age_ms={float(obs.get('obs_total_age_ms') or 0.0):.1f} "
                    f"update_interval_ms={obs.get('update_interval_ms')}"
                )
                if preview_sink is not None:
                    keep_running = preview_sink.render(_make_preview_frame(pack.get("rgb"), pack.get("depth"), obs, frame_seq))
                    try:
                        import cv2

                        key = cv2.waitKey(1) & 0xFF
                    except Exception:
                        key = 0
                    if not keep_running or key in (ord("q"), ord("Q")):
                        break
    finally:
        if preview_sink is not None:
            preview_sink.close()
        processor.release_all()

    summary = _metrics_summary(frames_total, observations)
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[BAG_TABLE_PLANE] wrote obs={obs_path} metrics={metrics_path}")


if __name__ == "__main__":
    main()
