#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple


HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
VISTA_ROOT = HERE.parents[2]
for item in (REPO_ROOT, VISTA_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))


@dataclass(frozen=True)
class ModePreset:
    mode: str
    purpose: str
    selection_strategy: str
    frame_stride: Optional[int] = None
    target_hz: Optional[float] = None
    preview_every: int = 0
    preview_max: Optional[int] = None
    contact_sheet: bool = False
    system_monitor: bool = False


MODE_PRESETS: Dict[str, ModePreset] = {
    "scan": ModePreset(
        mode="scan",
        purpose="quick sparse visual inspection",
        selection_strategy="frame_stride",
        frame_stride=100,
        preview_every=1,
        preview_max=None,
        contact_sheet=True,
        system_monitor=False,
    ),
    "eval": ModePreset(
        mode="eval",
        purpose="5Hz control usability evaluation",
        selection_strategy="bag_timestamp_hz",
        target_hz=5.0,
        preview_every=25,
        preview_max=40,
        contact_sheet=False,
        system_monitor=True,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process a RealSense bag through the online table-plane path.")
    parser.add_argument("--bag", type=Path, required=True, help="RealSense .bag file path.")
    parser.add_argument("--mode", choices=sorted(MODE_PRESETS.keys()), required=True, help="Fixed processing mode preset.")
    parser.add_argument("--output", type=Path, default=None, help="Output run directory. Default: VISTA/runs/bag_table_plane_<bag>_<mode>_<time>.")
    advanced = parser.add_argument_group("advanced")
    advanced.add_argument("--config", type=Path, default=VISTA_ROOT / "configs" / "vision_params.yaml", help="Advanced: vision params YAML.")
    advanced.add_argument("--roi-preset", default="", help="Advanced: optional table plane ROI preset override.")
    advanced.add_argument("--detector-mode", choices=("full", "fast_plane_only"), default="", help="Advanced: override table plane detector mode.")
    advanced.add_argument("--no-system-monitor", action="store_true", help="Advanced: disable eval system_monitor.csv.")
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


TIMING_FIELDS = (
    "frame_prepare_ms",
    "roi_extract_ms",
    "point_build_ms",
    "candidate_select_ms",
    "plane_fit_ms",
    "residual_eval_ms",
    "mask_build_ms",
    "obs_build_ms",
    "json_write_ms",
    "preview_render_ms",
    "preview_save_ms",
    "loop_total_ms",
)


COMPACT_OBS_FIELDS = (
    "type",
    "source",
    "source_mode",
    "timestamp",
    "ts",
    "obs_ts",
    "bag_timestamp_ms",
    "frame_capture_ts",
    "frame_seq",
    "frame_id",
    "seq",
    "detector_mode",
    "fast_plane_stride",
    "table_found",
    "edge_found",
    "edge_valid",
    "valid",
    "valid_for_control",
    "confidence",
    "edge_conf",
    "yaw_err_rad",
    "yaw_err",
    "dist_err_m",
    "dist_err",
    "edge_k",
    "edge_b",
    "plane_found",
    "plane_confidence",
    "plane_cx_norm",
    "plane_width_norm",
    "plane_area_ratio",
    "plane_touch_left",
    "plane_touch_right",
    "plane_touch_top",
    "plane_touch_bottom",
    "plane_yaw_err_rad",
    "plane_dist_err_m",
    "plane_x_span_m",
    "plane_residual_mean",
    "plane_residual_max",
    "plane_mask_status",
    "fast_fit_attempted",
    "fast_raw_yaw_err_rad",
    "fast_raw_dist_err_m",
    "fast_raw_plane_cx_norm",
    "fast_raw_plane_width_norm",
    "fast_raw_plane_x_span_m",
    "fast_raw_residual_mean",
    "fast_raw_residual_p90",
    "fast_candidate_point_count",
    "fast_support_point_count",
    "fast_rep_count",
    "fast_rep_inlier_count",
    "fast_rep_outlier_count",
    "fast_rep_cluster_count",
    "fast_selected_cluster_index",
    "fast_selected_cluster_y_center",
    "fast_selected_cluster_x_span_m",
    "fast_selected_cluster_support",
    "fast_selected_cluster_score",
    "fast_background_rep_count",
    "fast_front_rep_count",
    "fast_candidate_x_span_m",
    "fast_support_x_span_m",
    "fast_rep_x_span_m",
    "fast_fit_inlier_x_span_m",
    "fast_residual_mean",
    "fast_residual_p90",
    "fast_support_mode",
    "fast_fit_line_source",
    "fast_line_source",
    "fast_line_score",
    "fast_frontness_score",
    "fast_edge_consistency_score",
    "fast_background_penalty",
    "fast_edge_candidate_count",
    "fast_edge_inlier_count",
    "fast_edge_x_span_m",
    "fast_edge_y_median_px",
    "fast_edge_residual",
    "fast_edge_line_yaw_rad",
    "fast_edge_line_dist_m",
    "fast_edge_support_score",
    "fast_local_band_support_count",
    "fast_local_band_x_span_m",
    "fast_local_band_edge_support",
    "fast_local_band_residual_mean",
    "fast_background_blocked",
    "fast_near_stage_far_jump",
    "fast_selected_dist_source",
    "fast_prev_dist_used",
    "fast_raw_confidence",
    "fast_raw_inlier_count",
    "fast_raw_candidate_count",
    "fast_raw_sampled_point_count",
    "fast_raw_reject_reason",
    "fast_gate_reject_reason",
    "fast_gate_reason",
    "fast_coord_frame",
    "fast_camera_pitch_deg",
    "fast_camera_height_m",
    "fast_table_height_m",
    "fast_robot_z_p10",
    "fast_robot_z_p50",
    "fast_robot_z_p90",
    "fast_robot_z_min",
    "fast_robot_z_max",
    "candidate_robot_z_min",
    "candidate_robot_z_p50",
    "candidate_robot_z_max",
    "fast_ground_like_count",
    "fast_table_height_like_count",
    "fast_front_face_rep_count",
    "fast_front_face_support_point_count",
    "fast_representative_inlier_count",
    "fast_support_inlier_count",
    "fast_vertical_support_score",
    "fast_accepted_column_score",
    "fast_z_span_score",
    "fast_x_span_score",
    "fast_z_span_m_p50",
    "fast_z_span_m_max",
    "fast_birdview_fit_residual_mean",
    "fast_confidence_version",
    "fast_distance_stage",
    "fast_control_level",
    "fast_score_inlier",
    "fast_score_abs_inlier",
    "fast_score_inlier_ratio",
    "fast_score_evidence",
    "fast_score_residual",
    "fast_score_span",
    "fast_score_area",
    "fast_score_coverage",
    "fast_score_geometry",
    "fast_score_support_geometry",
    "fast_score_temporal",
    "fast_score_temporal_available",
    "fast_temporal_stable_count",
    "fast_temporal_jump",
    "fast_temporal_yaw_delta",
    "fast_temporal_dist_delta",
    "fast_temporal_cx_delta",
    "fast_score_final",
    "fast_candidate_pixel_count",
    "fast_support_pixel_count",
    "fast_front_face_rep_pixel_count",
    "fast_inlier_pixel_count",
    "fast_outlier_pixel_count",
    "fast_sampled_pixels",
    "fast_candidate_pixels",
    "fast_support_pixels",
    "fast_front_face_rep_pixels",
    "fast_inlier_pixels",
    "fast_outlier_pixels",
    "pose_source",
    "final_pose_source",
    "selected_line_type",
    "control_level",
    "control_reject_reason",
    "usable_for_approach",
    "usable_for_alignment",
    "usable_for_stop",
    "reason",
    "reject_reason",
    "depth_valid",
    "edge_obs_unavailable",
    "is_stale",
    "roi_source",
    "roi_reason",
    "roi_preset",
    "depth_edge_roi",
    "plane_roi",
    "table_edge_roi",
    "edge_roi",
    "roi_format",
    "table_bbox",
    "table_center_norm",
    "table_quadrant",
    "rgb_search_roi",
    "process_ms",
    "vision_process_ms",
    "obs_total_age_ms",
    "age_ms",
    "update_interval_ms",
    "bag_update_interval_ms",
    "edge_update_interval_ms",
    "frame_age_ms",
    "depth_frame_fetch_ms",
    "latest_frame_lag_ms",
    *TIMING_FIELDS,
    "payload_size_bytes",
    "point_count",
    "sampled_point_count",
    "candidate_count",
    "table_point_count",
    "inlier_count",
    "edge_inlier_count",
    "target_dist_m",
    "plane_only_mode",
    "enable_crease_line",
    "table_geometry_score",
    "front_plane_score",
    "line_score",
    "plane_line_consistency_score",
    "depth_z_min_m",
    "depth_z_max_m",
    "table_y_min_m",
    "table_y_max_m",
)


def _compact_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in COMPACT_OBS_FIELDS:
        if key not in obs:
            continue
        value = _json_ready(obs.get(key))
        if value is None or value == [] or value == {}:
            continue
        compact[key] = value
    compact.setdefault("type", "table_edge_obs")
    return compact


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


def _percentiles_p10_p50_p90(values: List[float]) -> Dict[str, Optional[float]]:
    vals = sorted(v for v in (_finite_float(v) for v in values) if v is not None)
    if not vals:
        return {"p10": None, "p50": None, "p90": None}

    def pick(q: float) -> float:
        if len(vals) == 1:
            return float(vals[0])
        pos = q * float(len(vals) - 1)
        lo = int(pos)
        hi = min(len(vals) - 1, lo + 1)
        frac = pos - float(lo)
        return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)

    return {"p10": pick(0.10), "p50": pick(0.50), "p90": pick(0.90)}


def _ratio(observations: List[Dict[str, Any]], key: str) -> float:
    total = len(observations)
    return float(sum(1 for obs in observations if bool(obs.get(key)))) / float(total) if total else 0.0


def _metrics_summary(
    *,
    preset: ModePreset,
    bag_frame_count: int,
    observations: List[Dict[str, Any]],
    preview_saved_count: int,
    system_monitor_path: Optional[Path],
) -> Dict[str, Any]:
    obs_total = len(observations)
    roi_sources = Counter(str(obs.get("roi_source") or "unknown") for obs in observations)
    detector_modes = Counter(str(obs.get("detector_mode") or "unknown") for obs in observations)
    reject_reasons = Counter(str(obs.get("reject_reason") or obs.get("reason") or "none") for obs in observations)
    fast_gate_reasons = Counter(str(obs.get("fast_gate_reason") or obs.get("fast_gate_reject_reason") or obs.get("fast_raw_reject_reason") or "none") for obs in observations)
    control_levels = Counter(str(obs.get("control_level") or obs.get("fast_control_level") or "none") for obs in observations)
    distance_stages = Counter(str(obs.get("fast_distance_stage") or "unknown") for obs in observations)
    line_sources = Counter(str(obs.get("fast_line_source") or obs.get("fast_fit_line_source") or "none") for obs in observations)
    timing = {field: _percentiles([obs.get(field) for obs in observations]) for field in TIMING_FIELDS}
    fast_score_keys = (
        "fast_score_inlier",
        "fast_score_abs_inlier",
        "fast_score_inlier_ratio",
        "fast_score_evidence",
        "fast_score_residual",
        "fast_score_span",
        "fast_score_area",
        "fast_score_coverage",
        "fast_score_geometry",
        "fast_score_temporal",
        "fast_score_final",
    )
    return {
        "mode": str(preset.mode),
        "purpose": str(preset.purpose),
        "selection_strategy": str(preset.selection_strategy),
        "bag_frame_count": int(bag_frame_count),
        "frames_total": int(bag_frame_count),
        "selected_frame_count": int(obs_total),
        "obs_total": int(obs_total),
        "frame_stride": preset.frame_stride,
        "target_hz": preset.target_hz,
        "process_ms": _percentiles([obs.get("process_ms", obs.get("vision_process_ms")) for obs in observations]),
        "detector_process_ms": _percentiles([obs.get("process_ms", obs.get("vision_process_ms")) for obs in observations]),
        "timing_ms": timing,
        "timing_notes": {
            "frame_prepare_ms": "bag frame packaging plus ROI/gate preparation before detector call",
            "roi_extract_ms": "detector depth preprocessing and ROI extraction",
            "candidate_select_ms": "table candidate selection from generated point cloud",
            "residual_eval_ms": "crease-line estimation plus fused pose, geometry, and control scoring",
            "json_write_ms": "compact obs serialization before writing jsonl",
        },
        "payload_size_bytes": _percentiles([obs.get("payload_size_bytes") for obs in observations]),
        "sampled_point_count": _percentiles([obs.get("sampled_point_count") for obs in observations]),
        "candidate_count": _percentiles([obs.get("candidate_count") for obs in observations]),
        "inlier_count": _percentiles([obs.get("inlier_count", obs.get("edge_inlier_count")) for obs in observations]),
        "plane_residual_mean": _percentiles([obs.get("plane_residual_mean") for obs in observations]),
        "fast_raw_confidence": _percentiles_p10_p50_p90([obs.get("fast_raw_confidence") for obs in observations]),
        "fast_raw_inlier_count": _percentiles_p10_p50_p90([obs.get("fast_raw_inlier_count") for obs in observations]),
        "fast_raw_candidate_count": _percentiles_p10_p50_p90([obs.get("fast_raw_candidate_count") for obs in observations]),
        "fast_raw_plane_width_norm": _percentiles_p10_p50_p90([obs.get("fast_raw_plane_width_norm") for obs in observations]),
        "fast_candidate_x_span_m": _percentiles_p10_p50_p90([obs.get("fast_candidate_x_span_m") for obs in observations]),
        "fast_support_x_span_m": _percentiles_p10_p50_p90([obs.get("fast_support_x_span_m") for obs in observations]),
        "fast_rep_x_span_m": _percentiles_p10_p50_p90([obs.get("fast_rep_x_span_m") for obs in observations]),
        "fast_fit_inlier_x_span_m": _percentiles_p10_p50_p90([obs.get("fast_fit_inlier_x_span_m", obs.get("fast_raw_plane_x_span_m")) for obs in observations]),
        "fast_raw_plane_x_span_m": _percentiles_p10_p50_p90([obs.get("fast_raw_plane_x_span_m") for obs in observations]),
        "fast_raw_residual_mean": _percentiles([obs.get("fast_raw_residual_mean") for obs in observations], include_max=False),
        "fast_edge_candidate_count": _percentiles_p10_p50_p90([obs.get("fast_edge_candidate_count") for obs in observations]),
        "fast_edge_inlier_count": _percentiles_p10_p50_p90([obs.get("fast_edge_inlier_count") for obs in observations]),
        "fast_local_band_support_count": _percentiles_p10_p50_p90([obs.get("fast_local_band_support_count") for obs in observations]),
        "fast_local_band_x_span_m": _percentiles_p10_p50_p90([obs.get("fast_local_band_x_span_m") for obs in observations]),
        "fast_score_components_p50": {
            key: _percentiles([obs.get(key) for obs in observations], include_max=False).get("p50")
            for key in fast_score_keys
        },
        "fast_score_components": {
            key: _percentiles_p10_p50_p90([obs.get(key) for obs in observations])
            for key in fast_score_keys
        },
        "obs_total_age_ms": _percentiles([obs.get("obs_total_age_ms") for obs in observations]),
        "update_interval_ms": _percentiles([obs.get("bag_update_interval_ms", obs.get("update_interval_ms")) for obs in observations], include_max=False),
        "roi_source": dict(sorted(roi_sources.items())),
        "detector_mode": dict(sorted(detector_modes.items())),
        "reject_reason": dict(sorted(reject_reasons.items())),
        "fast_gate_reject_reason": dict(sorted(fast_gate_reasons.items())),
        "fast_gate_reason": dict(sorted(fast_gate_reasons.items())),
        "control_level": dict(sorted(control_levels.items())),
        "distance_stage": dict(sorted(distance_stages.items())),
        "line_source": dict(sorted(line_sources.items())),
        "background_blocked_count": int(sum(1 for obs in observations if bool(obs.get("fast_background_blocked")))),
        "near_stage_far_jump_count": int(sum(1 for obs in observations if bool(obs.get("fast_near_stage_far_jump")))),
        "plane_found_ratio": _ratio(observations, "plane_found"),
        "usable_for_approach_ratio": _ratio(observations, "usable_for_approach"),
        "valid_for_control_ratio": float(
            sum(1 for obs in observations if bool(obs.get("valid_for_control") or obs.get("edge_valid")))
        ) / float(obs_total) if obs_total else 0.0,
        "preview_saved_count": int(preview_saved_count),
        "system_monitor_csv": str(system_monitor_path) if system_monitor_path is not None else None,
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


def _make_preview_canvas(sink: Any, rgb: Any, depth: Any, obs: Dict[str, Any], frame_seq: int):
    import numpy as np

    frame = _make_preview_frame(rgb, depth, obs, frame_seq)
    metadata = dict(frame.overlay.metadata or {})
    panel_w = max(320, int(getattr(sink, "canvas_w", 1280)) // 2)
    panel_h = max(220, int(getattr(sink, "canvas_h", 720)) // 2)
    panel_size = (panel_w, panel_h)
    rgb_panel = sink._make_rgb_panel(rgb, metadata, panel_size)
    depth_panel = sink._make_depth_panel(depth, obs, panel_size)
    plane_panel = sink._make_edge_panel(depth, obs, panel_size)
    info_panel = sink._make_info_panel(frame, metadata, obs, {}, panel_size)
    canvas = np.vstack([np.hstack([rgb_panel, depth_panel]), np.hstack([plane_panel, info_panel])])
    if canvas.shape[:2] != (sink.canvas_h, sink.canvas_w):
        try:
            import cv2

            canvas = cv2.resize(canvas, (sink.canvas_w, sink.canvas_h), interpolation=cv2.INTER_AREA)
        except Exception:
            pass
    return canvas


def _should_process_by_sample_hz(timestamp_ms: float, sample_hz: float, last_sample_ms: Optional[float]) -> bool:
    if sample_hz <= 0.0 or last_sample_ms is None:
        return True
    return float(timestamp_ms) - float(last_sample_ms) >= (1000.0 / max(0.1, float(sample_hz))) - 1e-6


def _select_frame(preset: ModePreset, frame_seq: int, bag_ts_ms: float, last_selected_ms: Optional[float]) -> Tuple[bool, Optional[float]]:
    if preset.selection_strategy == "frame_stride":
        stride = max(1, int(preset.frame_stride or 1))
        selected = (int(frame_seq) % stride) == 0
        return selected, bag_ts_ms if selected else last_selected_ms
    if preset.selection_strategy == "bag_timestamp_hz":
        target_hz = max(0.1, float(preset.target_hz or 5.0))
        selected = _should_process_by_sample_hz(float(bag_ts_ms), target_hz, last_selected_ms)
        return selected, bag_ts_ms if selected else last_selected_ms
    return True, bag_ts_ms


class SystemMonitor:
    def __init__(self, path: Optional[Path], enabled: bool):
        self.path = path
        self.enabled = bool(enabled and path is not None)
        self._fp = None
        self._writer = None
        self._psutil = None
        self._process = None
        self._start = time.perf_counter()

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            import psutil  # type: ignore

            self._psutil = psutil
            self._process = psutil.Process(os.getpid())
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fp = self.path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._fp,
                fieldnames=("wall_time", "process_cpu_percent", "system_cpu_percent", "rss_mb", "num_threads"),
            )
            self._writer.writeheader()
            self._process.cpu_percent(None)
            psutil.cpu_percent(None)
        except Exception as exc:
            print(f"[BAG_TABLE_PLANE] system_monitor_disabled error={exc}")
            self.enabled = False
            self.close()
        return self

    def sample(self) -> None:
        if not self.enabled or self._writer is None or self._psutil is None or self._process is None:
            return
        try:
            mem = self._process.memory_info()
            self._writer.writerow(
                {
                    "wall_time": max(0.0, time.perf_counter() - self._start),
                    "process_cpu_percent": self._process.cpu_percent(None),
                    "system_cpu_percent": self._psutil.cpu_percent(None),
                    "rss_mb": float(getattr(mem, "rss", 0) or 0) / (1024.0 * 1024.0),
                    "num_threads": self._process.num_threads(),
                }
            )
        except Exception:
            pass

    def close(self) -> None:
        if self._fp is not None:
            try:
                self._fp.close()
            except Exception:
                pass
        self._fp = None

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _write_contact_sheet(image_paths: List[Path], output_path: Path, *, thumb_w: int = 320, cols: int = 4) -> bool:
    if not image_paths:
        return False
    try:
        import cv2
        import numpy as np

        thumbs = []
        for path in image_paths:
            img = cv2.imread(str(path))
            if img is None:
                continue
            scale = float(thumb_w) / max(1.0, float(img.shape[1]))
            thumb_h = max(1, int(round(float(img.shape[0]) * scale)))
            thumbs.append(cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))
        if not thumbs:
            return False
        thumb_h = max(img.shape[0] for img in thumbs)
        padded = []
        for img in thumbs:
            if img.shape[0] < thumb_h:
                pad = np.zeros((thumb_h - img.shape[0], img.shape[1], 3), dtype=img.dtype)
                img = np.vstack([img, pad])
            padded.append(img)
        rows = []
        for start in range(0, len(padded), cols):
            row = padded[start : start + cols]
            while len(row) < cols:
                row.append(np.zeros((thumb_h, thumb_w, 3), dtype=padded[0].dtype))
            rows.append(np.hstack(row))
        sheet = np.vstack(rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(output_path), sheet))
    except Exception as exc:
        print(f"[BAG_TABLE_PLANE] contact_sheet_failed error={exc}")
        return False


FAST_DEBUG_COLUMNS = (
    "frame_seq",
    "bag_timestamp_ms",
    "table_found",
    "edge_found",
    "plane_found",
    "valid_for_control",
    "edge_valid",
    "usable_for_approach",
    "usable_for_alignment",
    "usable_for_stop",
    "control_level",
    "reject_reason",
    "control_reject_reason",
    "fast_fit_attempted",
    "fast_raw_yaw_err_rad",
    "fast_raw_dist_err_m",
    "fast_raw_plane_cx_norm",
    "fast_raw_plane_width_norm",
    "fast_raw_plane_x_span_m",
    "fast_raw_residual_mean",
    "fast_raw_residual_p90",
    "fast_candidate_point_count",
    "fast_support_point_count",
    "fast_rep_count",
    "fast_rep_inlier_count",
    "fast_rep_outlier_count",
    "fast_rep_cluster_count",
    "fast_selected_cluster_index",
    "fast_selected_cluster_y_center",
    "fast_selected_cluster_x_span_m",
    "fast_selected_cluster_support",
    "fast_selected_cluster_score",
    "fast_background_rep_count",
    "fast_front_rep_count",
    "fast_candidate_x_span_m",
    "fast_support_x_span_m",
    "fast_rep_x_span_m",
    "fast_fit_inlier_x_span_m",
    "fast_residual_mean",
    "fast_residual_p90",
    "fast_support_mode",
    "fast_fit_line_source",
    "fast_line_source",
    "fast_line_score",
    "fast_frontness_score",
    "fast_edge_consistency_score",
    "fast_background_penalty",
    "fast_edge_candidate_count",
    "fast_edge_inlier_count",
    "fast_edge_x_span_m",
    "fast_edge_y_median_px",
    "fast_edge_residual",
    "fast_edge_line_yaw_rad",
    "fast_edge_line_dist_m",
    "fast_edge_support_score",
    "fast_local_band_support_count",
    "fast_local_band_x_span_m",
    "fast_local_band_edge_support",
    "fast_local_band_residual_mean",
    "fast_background_blocked",
    "fast_near_stage_far_jump",
    "fast_selected_dist_source",
    "fast_prev_dist_used",
    "fast_raw_confidence",
    "fast_raw_inlier_count",
    "fast_raw_candidate_count",
    "fast_raw_sampled_point_count",
    "fast_raw_reject_reason",
    "fast_gate_reject_reason",
    "fast_gate_reason",
    "fast_confidence_version",
    "fast_distance_stage",
    "fast_control_level",
    "fast_score_inlier",
    "fast_score_abs_inlier",
    "fast_score_inlier_ratio",
    "fast_score_evidence",
    "fast_score_residual",
    "fast_score_span",
    "fast_score_area",
    "fast_score_coverage",
    "fast_score_geometry",
    "fast_score_temporal",
    "fast_score_temporal_available",
    "fast_temporal_stable_count",
    "fast_temporal_jump",
    "fast_temporal_yaw_delta",
    "fast_temporal_dist_delta",
    "fast_temporal_cx_delta",
    "fast_score_final",
    "process_ms",
    "plane_fit_ms",
)


def _write_csv_rows(path: Path, rows: List[Dict[str, Any]], columns: Tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for obs in rows:
            writer.writerow({key: _json_ready(obs.get(key)) for key in columns})


def _write_fast_debug_csvs(observations: List[Dict[str, Any]], output_dir: Path) -> Dict[str, Optional[str]]:
    fast_rows = [obs for obs in observations if "fast_fit_attempted" in obs or str(obs.get("detector_mode") or "") == "fast_plane_only"]
    if not fast_rows:
        return {"fast_debug_csv": None, "fast_key_frames_csv": None}
    debug_path = output_dir / "fast_debug_frames.csv"
    _write_csv_rows(debug_path, fast_rows, FAST_DEBUG_COLUMNS)
    key_targets = (144, 294, 444, 594, 636, 642, 648, 744)
    key_rows = []
    for obs in fast_rows:
        try:
            frame_seq = int(obs.get("frame_seq"))
        except Exception:
            continue
        near_target = any(abs(frame_seq - target) <= 12 for target in key_targets)
        in_problem_range = 504 <= frame_seq <= 624
        if near_target or in_problem_range:
            key_rows.append(obs)
    key_path = output_dir / "fast_debug_around_294_444_744_594.csv"
    _write_csv_rows(key_path, key_rows, FAST_DEBUG_COLUMNS)
    return {"fast_debug_csv": str(debug_path), "fast_key_frames_csv": str(key_path)}


def main() -> None:
    args = build_parser().parse_args()
    preset = MODE_PRESETS[str(args.mode)]
    os.environ["VISION_PARAMS_FILE"] = str(args.config.expanduser().resolve())

    from VISTA.vision_module.config.board_config import CONFIG
    from VISTA.vision_module.backend.table_edge_manager import TableEdgeManager

    if args.roi_preset:
        CONFIG.table_edge.roi_preset = str(args.roi_preset).strip().lower()
    if args.detector_mode:
        CONFIG.table_edge.detector_mode = str(args.detector_mode).strip().lower()

    if args.output is None:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        output_dir = (VISTA_ROOT / "runs" / f"bag_table_plane_{args.bag.stem}_{preset.mode}_{run_id}").resolve()
    else:
        output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    obs_path = output_dir / "table_edge_obs.jsonl"
    metrics_path = output_dir / "metrics_summary.json"
    monitor_path = output_dir / "system_monitor.csv" if preset.system_monitor and not args.no_system_monitor else None

    processor = TableEdgeManager(cfg=CONFIG)
    preview_sink = None
    preview_dir = None
    preview_paths: List[Path] = []
    preview_saved_count = 0
    if int(preset.preview_every or 0) > 0:
        from VISTA.vision_module.backend.preview.opencv_sink import OpenCVPreviewSink

        preview_sink = OpenCVPreviewSink("Bag Table Plane Replay")
        preview_dir = output_dir / "preview_frames"
        preview_dir.mkdir(parents=True, exist_ok=True)

    frames_total = 0
    observations: List[Dict[str, Any]] = []
    last_selected_ms: Optional[float] = None
    print(
        "[BAG_TABLE_PLANE] "
        f"mode={preset.mode} strategy={preset.selection_strategy} "
        f"detector_mode={getattr(CONFIG.table_edge, 'detector_mode', 'full')} "
        f"frame_stride={preset.frame_stride} target_hz={preset.target_hz} "
        f"preview_every={preset.preview_every} preview_max={preset.preview_max}"
    )

    try:
        with obs_path.open("w", encoding="utf-8") as fp, SystemMonitor(monitor_path, monitor_path is not None) as monitor:
            for pack in iter_bag_frames(args.bag, 0):
                frame_seq = int(pack["frame"])
                frames_total += 1
                bag_ts_ms = float(pack.get("timestamp_ms", 0.0) or 0.0)
                previous_selected_ms = last_selected_ms
                selected, last_selected_ms = _select_frame(preset, frame_seq, bag_ts_ms, last_selected_ms)
                if not selected:
                    continue
                bag_update_interval_ms = None if previous_selected_ms is None else max(0.0, bag_ts_ms - float(previous_selected_ms))
                loop_total_start = time.perf_counter()
                frame_prepare_start = time.perf_counter()
                capture_ts = time.time()
                frames = {
                    "rgb": pack.get("rgb"),
                    "depth": pack.get("depth"),
                    "frame_capture_ts": capture_ts,
                    "timestamp_ms": pack.get("timestamp_ms"),
                }
                bag_frame_prepare_ms = (time.perf_counter() - frame_prepare_start) * 1000.0
                obs = processor.process_camera_frame(
                    frames,
                    frame_seq=frame_seq,
                    frame_slot={"seq": frame_seq, "ts": capture_ts, "payload": frames},
                    local_perception={"has_infer": False, "box_count": 0, "infer_boxes": [], "rgb_shape": getattr(pack.get("rgb"), "shape", None)},
                    runtime_status={"stage": "OFFLINE_BAG", "mode": "TABLE_EDGE_PERCEPTION"},
                    source_mode="OFFLINE_BAG",
                    count_dropped=False,
                )
                obs["frame_prepare_ms"] = float(obs.get("frame_prepare_ms") or 0.0) + float(bag_frame_prepare_ms)
                obs["source"] = f"bag_table_plane_{preset.mode}"
                obs["bag_timestamp_ms"] = bag_ts_ms
                if bag_update_interval_ms is not None:
                    obs["bag_update_interval_ms"] = bag_update_interval_ms
                    obs["update_interval_ms"] = bag_update_interval_ms
                    obs["edge_update_interval_ms"] = bag_update_interval_ms
                obs_index = len(observations) + 1
                can_save_more = preset.preview_max is None or preview_saved_count < int(preset.preview_max)
                if (
                    preview_sink is not None
                    and preview_dir is not None
                    and can_save_more
                    and int(preset.preview_every or 0) > 0
                    and (obs_index % int(preset.preview_every)) == 0
                ):
                    preview_render_start = time.perf_counter()
                    canvas = _make_preview_canvas(preview_sink, pack.get("rgb"), pack.get("depth"), obs, frame_seq)
                    obs["preview_render_ms"] = (time.perf_counter() - preview_render_start) * 1000.0
                    try:
                        import cv2

                        out_path = preview_dir / f"table_plane_preview_{frame_seq:06d}.png"
                        preview_save_start = time.perf_counter()
                        if cv2.imwrite(str(out_path), canvas):
                            preview_saved_count += 1
                            preview_paths.append(out_path)
                        obs["preview_save_ms"] = (time.perf_counter() - preview_save_start) * 1000.0
                    except Exception as exc:
                        obs["preview_save_ms"] = 0.0
                        print(f"[BAG_TABLE_PLANE] preview_save_failed frame_seq={frame_seq} error={exc}")
                json_start = time.perf_counter()
                clean_obs = _compact_obs(obs)
                line = json.dumps(clean_obs, ensure_ascii=False, sort_keys=True)
                obs["json_write_ms"] = (time.perf_counter() - json_start) * 1000.0
                obs["loop_total_ms"] = (time.perf_counter() - loop_total_start) * 1000.0
                clean_obs = _compact_obs(obs)
                line = json.dumps(clean_obs, ensure_ascii=False, sort_keys=True)
                clean_obs["payload_size_bytes"] = len(line.encode("utf-8"))
                line = json.dumps(clean_obs, ensure_ascii=False, sort_keys=True)
                clean_obs["payload_size_bytes"] = len(line.encode("utf-8"))
                line = json.dumps(clean_obs, ensure_ascii=False, sort_keys=True)
                fp.write(line + "\n")
                observations.append(clean_obs)
                monitor.sample()
                print(
                    "[BAG_TABLE_PLANE] "
                    f"mode={preset.mode} frame_seq={frame_seq} "
                    f"valid={int(bool(obs.get('valid_for_control') or obs.get('edge_valid')))} "
                    f"plane={int(bool(obs.get('plane_found')))} "
                    f"detector_mode={obs.get('detector_mode')} "
                    f"roi_source={obs.get('roi_source')} roi={obs.get('plane_roi') or obs.get('depth_edge_roi')} "
                    f"process_ms={float(obs.get('process_ms') or 0.0):.1f} "
                    f"obs_total_age_ms={float(obs.get('obs_total_age_ms') or 0.0):.1f} "
                    f"bag_update_interval_ms={obs.get('bag_update_interval_ms')}"
                )
    finally:
        processor.release_all()

    contact_sheet_path = None
    if preset.contact_sheet and preview_paths:
        contact_sheet_path = output_dir / "preview_contact_sheet.jpg"
        if not _write_contact_sheet(preview_paths, contact_sheet_path):
            contact_sheet_path = None
    summary = _metrics_summary(
        preset=preset,
        bag_frame_count=frames_total,
        observations=observations,
        preview_saved_count=preview_saved_count,
        system_monitor_path=monitor_path if monitor_path is not None and monitor_path.exists() else None,
    )
    summary.update(_write_fast_debug_csvs(observations, output_dir))
    summary["preview_dir"] = str(preview_dir) if preview_dir is not None else None
    summary["contact_sheet"] = str(contact_sheet_path) if contact_sheet_path is not None else None
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[BAG_TABLE_PLANE] wrote obs={obs_path} metrics={metrics_path}")


if __name__ == "__main__":
    main()
