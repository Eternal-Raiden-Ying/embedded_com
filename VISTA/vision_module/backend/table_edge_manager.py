#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from .depth_calibration import DepthIntrinsics, depth_intrinsics_from_dict
from .table_edge_roi import choose_depth_roi
from .vision_semantics import standardize_table_edge_payload
from ..config.schema import VisionServiceConfig
from ..utils.table_roi import table_detection_debug


CapabilitySink = Optional[Callable[[str, Dict[str, Any]], None]]


class TableEdgeManager:
    """Own depth-based table-edge perception and publish summarized results."""

    def __init__(
        self,
        cfg: Optional[VisionServiceConfig] = None,
        logger: Optional[logging.Logger] = None,
        capability_sink: CapabilitySink = None,
    ):
        self.cfg = cfg or VisionServiceConfig()
        self.log = logger or logging.getLogger("vision.table_edge_manager")
        self._capability_sink = capability_sink
        self._scheduler = None
        self._generation_getter = lambda: 0
        self._runtime_running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_stop = threading.Event()
        self._detector_mode = "lightweight"
        self._edge_update_hz = 5.0
        self._worker_interval_s = 1.0 / max(1.0, self._edge_update_hz)
        self._default_interval_s = self._worker_interval_s
        self._light_stride = 4
        self._fast_plane_stride = 4
        self._require_yolo_confirm = True
        self._static_roi_enabled = False
        self._camera_pitch_deg = 15.0
        self._camera_height_m = 0.70
        self._camera_roll_deg = 0.0
        self._camera_yaw_deg = 0.0
        self._table_height_m = 0.40
        self._front_face_z_min_m = 0.03
        self._front_face_z_max_m = 0.43
        self._min_vertical_z_span_m = 0.12
        self._min_vertical_support_points = 3
        self._x_bin_width_m = 0.04
        self._y_cluster_bin_m = 0.04
        self._min_front_face_columns = 3
        self._min_front_face_x_span_m = 0.07
        self._front_cluster_gap_m = 0.10
        self._max_yaw_abs_rad = 0.75
        self._enable_yolo_in_plane_only = False
        self._yolo_table_min_conf = 0.25
        self._fast_candidate_point_cap = 1800
        self._fast_front_edge_col_step = 2
        self._fast_front_edge_row_step = 2
        # debug knobs — read from global config, not per-mode
        table_edge_cfg = getattr(self.cfg, "table_edge", None)
        self._profile_log_interval_s = float(getattr(table_edge_cfg, "profile_log_interval_s", 2.0) or 2.0)
        self._save_debug_frames = bool(getattr(table_edge_cfg, "save_debug_frames", False))
        self._fast_candidate_point_cap = max(0, int(getattr(table_edge_cfg, "fast_candidate_point_cap", self._fast_candidate_point_cap) or 0))
        self._fast_front_edge_col_step = max(1, int(getattr(table_edge_cfg, "fast_front_edge_col_step", self._fast_front_edge_col_step) or 1))
        self._fast_front_edge_row_step = max(1, int(getattr(table_edge_cfg, "fast_front_edge_row_step", self._fast_front_edge_row_step) or 1))
        self._last_camera_generation = 0
        self._last_camera_seq = 0
        self._last_camera_frame_capture_ts = 0.0
        self._last_camera_frame_interval_ms: Optional[float] = None
        self._last_worker_loop_ts = 0.0
        self._last_worker_interval_ms: Optional[float] = None
        self._last_process_start_ts = 0.0
        self._last_table_edge_process_interval_ms: Optional[float] = None
        self._last_publish_interval_ms: Optional[float] = None
        self._last_scheduler_read_ms = 0.0
        self._last_scheduler_publish_ms = 0.0
        self._table_edge_no_new_frame_count = 0
        self._last_obs_seq = 0
        self._last_publish_ts = 0.0
        self._last_obs_ts = 0.0
        self._last_depth_frame_fetch_ms = 0.0
        self._last_process_ms = 0.0
        self._last_update_interval_ms = None
        self._last_edge_dbg_ts = 0.0
        self._last_profile_log_ts = 0.0
        self._frame_id = 0
        self._dropped_frame_count = 0
        self._processed_frame_count = 0
        self._processing_busy = False
        self._detector = None
        self._detector_cfg = None
        self._detector_error = ""
        self._fallback_calib = None
        self._frame_calib_payload: Dict[str, Any] = {}
        self._last_calib_log_key = ""
        self._target_dist_m = 0.5
        self._source_mode_override: Optional[str] = None
        self._local_perception_override: Optional[Dict[str, Any]] = None
        self._runtime_status_override: Optional[Dict[str, Any]] = None
        self._last_valid_quadrant: Optional[str] = None
        self._last_valid_quadrant_ts = 0.0
        self._last_valid_quadrant_ttl_s = 1.0
        self._last_valid_table_bbox = None
        self._last_valid_table_center_norm = None
        self._yolo_table_roi_center_x_ema: Optional[float] = None
        self._edge_stable_count = 0
        self._edge_stability_prev: Dict[str, Any] = {}
        self._last_measured_edge_dist_m: Optional[float] = None
        self._fast_temporal_state: Dict[str, Any] = {}
        self._load_detector()

    @staticmethod
    def _normalize_detector_mode(value: Any) -> str:
        mode = str(value or "full").strip().lower().replace("-", "_")
        return mode if mode in {"full", "fast_plane_only"} else "full"

    def _emit(self, action: str, **fields: Any) -> None:
        if self._capability_sink is None:
            return
        try:
            payload = {"action": str(action or "updated").strip().lower()}
            payload.update(dict(fields or {}))
            self._capability_sink("table_edge_detector", payload)
        except Exception:
            pass

    def _load_detector(self) -> None:
        try:
            from .edge_detect.board_config import CONFIG as edge_cfg
            from .edge_detect.detector import OnlineTableEdgeDetector, load_calib
        except ImportError:
            try:
                from vision_module.backend.edge_detect.board_config import CONFIG as edge_cfg  # type: ignore[assignment]
                from vision_module.backend.edge_detect.detector import OnlineTableEdgeDetector, load_calib  # type: ignore[assignment]
            except ImportError as exc2:
                self._detector = None
                self._detector_cfg = None
                self._detector_error = str(exc2 or "detector_unavailable")
                self._emit("load_failed", error=self._detector_error)
                return
        try:
            calib_path = Path(str(edge_cfg.detector.calib_json)).expanduser()
            calib, target_dist = load_calib(calib_path)
            if float(edge_cfg.detector.target_dist_m_override) > 0:
                target_dist = float(edge_cfg.detector.target_dist_m_override)
            self._detector = OnlineTableEdgeDetector(calib, edge_cfg.detector, target_dist)
            self._detector_cfg = edge_cfg.detector
            self._fallback_calib = calib
            self._target_dist_m = float(target_dist)
            self._detector_error = ""
            self._emit(
                "loaded",
                calib_json=str(calib_path),
                target_dist_m=float(self._target_dist_m),
            )
        except Exception as exc:
            self._detector = None
            self._detector_cfg = None
            self._fallback_calib = None
            self._detector_error = str(exc or "detector_unavailable")
            self._emit("load_failed", error=self._detector_error)

    def bind_runtime(self, scheduler, generation_getter=None) -> None:
        self._scheduler = scheduler
        if callable(generation_getter):
            self._generation_getter = generation_getter

    def start_runtime(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._runtime_running = True
        self._worker_stop.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, name="table_edge_manager.loop", daemon=True)
        self._worker_thread.start()

    def stop_runtime(self) -> None:
        self._runtime_running = False
        self._worker_stop.set()
        thread = self._worker_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._worker_thread = None

    def _publish_result(self, route: str, payload: Any) -> None:
        scheduler = self._scheduler
        if scheduler is None:
            return
        try:
            generation = int(self._generation_getter())
        except Exception:
            generation = 0
        try:
            if isinstance(payload, dict):
                publish_start = time.perf_counter()
                now = time.time()
                payload["table_edge_publish_interval_ms"] = (
                    (now - float(self._last_publish_ts)) * 1000.0
                    if float(self._last_publish_ts or 0.0) > 0.0
                    else None
                )
                self._last_publish_interval_ms = payload.get("table_edge_publish_interval_ms")
                payload["obs_publish_ts"] = float(time.time())
                payload["vision_publish_ts_ms"] = self._epoch_ms(payload["obs_publish_ts"])
                done_ts = float(payload.get("vision_done_ts") or payload["obs_publish_ts"])
                payload["publish_delay_ms"] = max(0.0, (payload["obs_publish_ts"] - done_ts) * 1000.0)
                profile = payload.get("edge_profile")
                if isinstance(profile, dict):
                    profile["table_edge_publish_interval_ms"] = payload.get("table_edge_publish_interval_ms")
                    profile["vision_publish_ts_ms"] = payload.get("vision_publish_ts_ms")
            scheduler.publish_result(route, payload, generation=generation)
            if isinstance(payload, dict):
                self._last_scheduler_publish_ms = self._ms_since(publish_start)
                payload["scheduler_publish_ms"] = float(self._last_scheduler_publish_ms)
                profile = payload.get("edge_profile")
                if isinstance(profile, dict):
                    profile["scheduler_publish_ms"] = float(self._last_scheduler_publish_ms)
            self._last_publish_ts = time.time()
        except Exception:
            pass

    def configure(self, payload: Dict[str, Any]) -> None:
        mode = str(payload.get("detector_mode") or "full")
        self._detector_mode = mode if mode in {"lightweight", "full", "fast_plane_only"} else "full"
        self._edge_update_hz = float(payload.get("update_hz", self._edge_update_hz) or self._edge_update_hz)
        if self._edge_update_hz > 0:
            self._worker_interval_s = 1.0 / max(1.0, self._edge_update_hz)
        if "light_stride" in payload:
            self._light_stride = max(1, int(payload.get("light_stride")))
        if "fast_plane_stride" in payload:
            self._fast_plane_stride = max(1, int(payload.get("fast_plane_stride")))
        if "require_yolo_confirm" in payload:
            self._require_yolo_confirm = bool(payload.get("require_yolo_confirm"))
        if "static_roi_enabled" in payload:
            self._static_roi_enabled = bool(payload.get("static_roi_enabled"))
        if "camera_pitch_deg" in payload:
            self._camera_pitch_deg = float(payload.get("camera_pitch_deg"))
        if "camera_height_m" in payload:
            self._camera_height_m = float(payload.get("camera_height_m"))
        if "camera_roll_deg" in payload:
            self._camera_roll_deg = float(payload.get("camera_roll_deg"))
        if "camera_yaw_deg" in payload:
            self._camera_yaw_deg = float(payload.get("camera_yaw_deg"))
        if "table_height_m" in payload:
            self._table_height_m = float(payload.get("table_height_m"))
        if "front_face_z_min_m" in payload:
            self._front_face_z_min_m = float(payload.get("front_face_z_min_m"))
        if "front_face_z_max_m" in payload:
            self._front_face_z_max_m = float(payload.get("front_face_z_max_m"))
        if "min_vertical_z_span_m" in payload:
            self._min_vertical_z_span_m = float(payload.get("min_vertical_z_span_m"))
        if "min_vertical_support_points" in payload:
            self._min_vertical_support_points = max(1, int(payload.get("min_vertical_support_points")))
        if "x_bin_width_m" in payload:
            self._x_bin_width_m = float(payload.get("x_bin_width_m"))
        if "y_cluster_bin_m" in payload:
            self._y_cluster_bin_m = float(payload.get("y_cluster_bin_m"))
        if "min_front_face_columns" in payload:
            self._min_front_face_columns = max(2, int(payload.get("min_front_face_columns")))
        if "min_front_face_x_span_m" in payload:
            self._min_front_face_x_span_m = float(payload.get("min_front_face_x_span_m"))
        if "front_cluster_gap_m" in payload:
            self._front_cluster_gap_m = float(payload.get("front_cluster_gap_m"))
        if "max_yaw_abs_rad" in payload:
            self._max_yaw_abs_rad = float(payload.get("max_yaw_abs_rad"))
        if "enable_yolo_in_plane_only" in payload:
            self._enable_yolo_in_plane_only = bool(payload.get("enable_yolo_in_plane_only"))
        if "yolo_table_min_conf" in payload:
            self._yolo_table_min_conf = float(payload.get("yolo_table_min_conf"))
        if "fast_candidate_point_cap" in payload:
            self._fast_candidate_point_cap = max(0, int(payload.get("fast_candidate_point_cap") or 0))
        if "fast_front_edge_col_step" in payload:
            self._fast_front_edge_col_step = max(1, int(payload.get("fast_front_edge_col_step") or 1))
        if "fast_front_edge_row_step" in payload:
            self._fast_front_edge_row_step = max(1, int(payload.get("fast_front_edge_row_step") or 1))

    @staticmethod
    def _pick_frame_capture_ts(frame_slot: Dict[str, Any], frames: Dict[str, Any]) -> float:
        for source in (frames, frame_slot):
            for key in ("frame_capture_ts", "capture_ts", "frame_ts", "ts"):
                value = source.get(key) if isinstance(source, dict) else None
                if value is None:
                    continue
                try:
                    ts = float(value)
                    if ts > 0.0:
                        if ts > 1e12:
                            return ts / 1000.0
                        return ts
                except Exception:
                    continue
        return time.time()

    def _with_freshness(
        self,
        payload: Dict[str, Any],
        *,
        frame_capture_ts: float,
        vision_start_ts: float,
        vision_done_ts: float,
        latest_frame_lag_ms: float,
    ) -> Dict[str, Any]:
        out = dict(payload or {})
        obs_ts = float(vision_done_ts or time.time())
        previous_obs_ts = float(self._last_obs_ts or 0.0)
        update_interval_ms = ((obs_ts - previous_obs_ts) * 1000.0) if previous_obs_ts > 0.0 else None
        self._last_obs_ts = float(obs_ts)
        self._last_update_interval_ms = update_interval_ms
        obs_hz = 1000.0 / float(update_interval_ms) if update_interval_ms and update_interval_ms > 0.0 else 0.0
        out["ts"] = float(obs_ts)
        out["obs_ts"] = float(obs_ts)
        out["frame_capture_ts"] = float(frame_capture_ts)
        out["vision_start_ts"] = float(vision_start_ts)
        out["vision_done_ts"] = float(vision_done_ts)
        out["obs_seq"] = int(out.get("obs_seq") or self._last_obs_seq or 0)
        out["camera_frame_ts_ms"] = self._epoch_ms(frame_capture_ts)
        out["vision_process_start_ts_ms"] = self._epoch_ms(vision_start_ts)
        out["vision_process_end_ts_ms"] = self._epoch_ms(vision_done_ts)
        out.setdefault("vision_publish_ts_ms", None)
        out.setdefault("obs_out_send_ts_ms", None)
        out.setdefault("orchestrator_recv_ts_ms", None)
        out.setdefault("state_machine_consume_ts_ms", None)
        out.setdefault("cmd_publish_ts_ms", None)
        out["frame_age_ms"] = max(0.0, (float(vision_start_ts) - float(frame_capture_ts)) * 1000.0)
        out["vision_process_ms"] = max(0.0, (float(vision_done_ts) - float(vision_start_ts)) * 1000.0)
        out["obs_total_age_ms"] = max(0.0, (float(obs_ts) - float(frame_capture_ts)) * 1000.0)
        out["age_ms"] = float(out["obs_total_age_ms"])
        out["edge_update_interval_ms"] = update_interval_ms
        out["update_interval_ms"] = update_interval_ms
        out["process_ms"] = float(out["vision_process_ms"])
        out["edge_process_ms"] = float(out["vision_process_ms"])
        out["total_edge_process_ms"] = float(out["vision_process_ms"])
        out.setdefault("depth_frame_fetch_ms", float(self._last_depth_frame_fetch_ms))
        out["dropped_frame_count"] = int(self._dropped_frame_count)
        out["processed_frame_count"] = int(self._processed_frame_count)
        out["latest_frame_lag_ms"] = float(latest_frame_lag_ms)
        out["camera_frame_seq"] = out.get("camera_frame_seq", out.get("frame_seq", out.get("seq")))
        out["camera_frame_age_ms"] = float(latest_frame_lag_ms)
        out["camera_frame_interval_ms"] = self._last_camera_frame_interval_ms
        out["camera_frame_hz"] = (
            1000.0 / float(self._last_camera_frame_interval_ms)
            if self._last_camera_frame_interval_ms and self._last_camera_frame_interval_ms > 0.0
            else 0.0
        )
        out["camera_frames_hz"] = out["camera_frame_hz"]
        out["table_edge_worker_interval_ms"] = self._last_worker_interval_ms
        out["table_edge_no_new_frame_count"] = int(self._table_edge_no_new_frame_count)
        out["table_edge_process_interval_ms"] = self._last_table_edge_process_interval_ms
        out["vision_process_interval_ms"] = self._last_table_edge_process_interval_ms
        out["table_edge_publish_interval_ms"] = self._last_publish_interval_ms
        out["vision_publish_interval_ms"] = self._last_publish_interval_ms
        out["table_edge_obs_hz"] = float(obs_hz)
        out["scheduler_read_ms"] = float(self._last_scheduler_read_ms)
        out["scheduler_publish_ms"] = float(self._last_scheduler_publish_ms)
        profile = out.get("edge_profile")
        if isinstance(profile, dict):
            for key in (
                "camera_frame_seq",
                "obs_seq",
                "camera_frame_ts_ms",
                "vision_process_start_ts_ms",
                "vision_process_end_ts_ms",
                "camera_frame_age_ms",
                "camera_frame_interval_ms",
                "camera_frame_hz",
                "camera_frames_hz",
                "table_edge_worker_interval_ms",
                "table_edge_no_new_frame_count",
                "table_edge_process_interval_ms",
                "vision_process_interval_ms",
                "table_edge_publish_interval_ms",
                "vision_publish_interval_ms",
                "table_edge_obs_hz",
                "scheduler_read_ms",
                "scheduler_publish_ms",
            ):
                profile[key] = out.get(key)
        unavailable = bool(out.get("edge_obs_unavailable", False))
        out["is_stale"] = bool(out.get("is_stale", False) or unavailable)
        out["source_mode"] = self._source_mode_override if self._source_mode_override is not None else self._detector_mode
        out.setdefault("timestamp", float(obs_ts))
        out.setdefault("seq", out.get("frame_seq", out.get("frame_id")))
        out.setdefault("frame_id", out.get("frame_seq", out.get("seq")))
        out.setdefault("edge_conf", out.get("confidence"))
        out.setdefault("edge_detected", bool(out.get("edge_found", False)))
        out.setdefault("edge_geometry_valid", bool(out.get("edge_found", False)) and not unavailable)
        out.setdefault("edge_valid", bool(out.get("edge_geometry_valid", False)))
        out.setdefault("yaw_err", out.get("yaw_err_rad"))
        out.setdefault("dist_err", out.get("dist_err_m"))
        if not bool(out.get("table_bbox_found", False)):
            for key in ("depth_edge_roi", "table_edge_roi", "edge_roi", "plane_roi"):
                out[key] = None
            out["roi_source"] = "disabled_no_table_bbox"
            out["roi_phase"] = "disabled_no_table_bbox"
            out["table_bbox_current_found"] = False
            out["table_bbox_control_valid"] = False
            out["edge_control_allowed"] = False
            out["docking_enabled_by_yolo"] = False
            out["edge_trusted"] = False
            out["valid_for_control"] = False
            out["edge_control_block_reason"] = out.get("edge_control_block_reason") or "table_bbox_unavailable"
            out["edge_reject_for_control_reason"] = out.get("edge_reject_for_control_reason") or "table_bbox_unavailable"
        # Compatibility aliases retained for the state machine and legacy logs while
        # table plane fields become the canonical semantics.
        if out.get("plane_roi") is None and out.get("depth_edge_roi") is not None:
            out["plane_roi"] = out.get("depth_edge_roi")
        cfg = getattr(self.cfg, "table_edge", None)
        max_residual = getattr(cfg, "edge_trusted_max_residual", 0.0) if cfg is not None else 0.0
        try:
            max_residual = float(max_residual or 0.0)
        except Exception:
            max_residual = 0.0
        out = standardize_table_edge_payload(
            out,
            edge_stable_required_frames=max(1, int(getattr(cfg, "yolo_table_edge_stable_frames", 5) if cfg is not None else 5)),
            edge_trusted_min_conf=float(getattr(cfg, "edge_trusted_min_conf", 0.60) if cfg is not None else 0.60),
            edge_trusted_max_residual=max_residual if max_residual > 0.0 else None,
        )
        return out

    def _log_profile_if_due(self, payload: Dict[str, Any]) -> None:
        interval_s = float(self._profile_log_interval_s)
        if interval_s <= 0.0:
            return
        now = time.time()
        if now - float(self._last_profile_log_ts or 0.0) < interval_s:
            return
        self._last_profile_log_ts = now
        self.log.info(
            "[TABLE_EDGE_PROFILE] obs_total_age_ms=%.1f vision_process_ms=%.1f edge_update_interval_ms=%s dropped=%d processed=%d latest_frame_lag_ms=%.1f depth_shape=%s fx=%.3f fy=%.3f cx=%.3f cy=%.3f depth_scale=%.6f calib_source=%s warning=%s",
            float(payload.get("obs_total_age_ms") or 0.0),
            float(payload.get("vision_process_ms") or 0.0),
            "None" if payload.get("edge_update_interval_ms") is None else f"{float(payload.get('edge_update_interval_ms')):.1f}",
            int(payload.get("dropped_frame_count") or 0),
            int(payload.get("processed_frame_count") or 0),
            float(payload.get("latest_frame_lag_ms") or 0.0),
            payload.get("depth_shape"),
            float(payload.get("fx") or 0.0),
            float(payload.get("fy") or 0.0),
            float(payload.get("cx") or 0.0),
            float(payload.get("cy") or 0.0),
            float(payload.get("depth_scale") or 0.0),
            payload.get("calib_source"),
            payload.get("calib_mismatch_warning") or "none",
        )

    @staticmethod
    def _ms_since(start_ts: float) -> float:
        return max(0.0, (time.perf_counter() - float(start_ts)) * 1000.0)

    @staticmethod
    def _epoch_ms(ts: Any) -> Optional[int]:
        try:
            value = float(ts)
            if value <= 0.0:
                return None
            return int(round(value * 1000.0))
        except Exception:
            return None

    def _calib_from_depth_intrinsics(self, intr: DepthIntrinsics):
        try:
            from .edge_detect.detector import CameraCalib
        except ImportError:
            from vision_module.backend.edge_detect.detector import CameraCalib  # type: ignore
        return CameraCalib(
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.cx),
            cy=float(intr.cy),
            depth_scale=float(intr.depth_scale),
            width=int(intr.width),
            height=int(intr.height),
            source=str(intr.source or "runtime_profile"),
            profile_info=str(intr.profile_info or ""),
        )

    def _resolve_frame_calib(self, frames: Dict[str, Any], depth: Any) -> Any:
        fallback = self._fallback_calib or getattr(self._detector, "calib", None)
        intr = depth_intrinsics_from_dict((frames or {}).get("depth_intrinsics"))
        if intr is not None:
            return self._calib_from_depth_intrinsics(intr)
        if fallback is not None:
            try:
                fallback.source = "calib_json_fallback"
            except Exception:
                pass
            return fallback
        return None

    @staticmethod
    def _depth_shape_payload(depth: Any) -> tuple[Optional[int], Optional[int], Optional[list[int]]]:
        shape = getattr(depth, "shape", None)
        if not isinstance(shape, tuple) or len(shape) < 2:
            return None, None, None
        try:
            h = int(shape[0])
            w = int(shape[1])
        except Exception:
            return None, None, None
        if h <= 0 or w <= 0:
            return None, None, None
        return h, w, [h, w]

    def _build_calib_payload(self, calib: Any, depth: Any) -> Dict[str, Any]:
        h, w, depth_shape = self._depth_shape_payload(depth)
        if calib is None:
            return {
                "depth_shape": depth_shape,
                "calib_width": None,
                "calib_height": None,
                "fx": None,
                "fy": None,
                "cx": None,
                "cy": None,
                "depth_scale": None,
                "calib_source": "unavailable",
                "calib_mismatch_warning": "CALIB_UNAVAILABLE",
            }
        calib_w = int(getattr(calib, "width", 0) or 0) or None
        calib_h = int(getattr(calib, "height", 0) or 0) or None
        fx = float(getattr(calib, "fx", 0.0) or 0.0)
        fy = float(getattr(calib, "fy", 0.0) or 0.0)
        cx = float(getattr(calib, "cx", 0.0) or 0.0)
        cy = float(getattr(calib, "cy", 0.0) or 0.0)
        source = str(getattr(calib, "source", "") or "calib_json_fallback")
        warnings = []
        if w is not None and h is not None:
            if not (0.0 <= cx <= float(w)) or not (0.0 <= cy <= float(h)):
                warnings.append(f"CALIB_MISMATCH: depth={w}x{h} cx={cx:.3f} cy={cy:.3f}")
            if calib_w is not None and calib_h is not None and (int(calib_w) != int(w) or int(calib_h) != int(h)):
                warnings.append(f"CALIB_MISMATCH: depth={w}x{h} calib={calib_w}x{calib_h}")
        if source == "calib_json_fallback":
            warnings.append("CALIB_FALLBACK: calib_json_fallback")
        warning = " | ".join(dict.fromkeys(warnings))
        payload = {
            "depth_shape": depth_shape,
            "calib_width": calib_w,
            "calib_height": calib_h,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "depth_scale": float(getattr(calib, "depth_scale", 0.001) or 0.001),
            "calib_source": source,
            "calib_profile_info": str(getattr(calib, "profile_info", "") or ""),
            "calib_mismatch_warning": warning,
        }
        log_key = f"{source}|{w}x{h}|{calib_w}x{calib_h}|{fx:.2f}|{fy:.2f}|{cx:.2f}|{cy:.2f}|{warning}"
        if log_key != self._last_calib_log_key:
            self._last_calib_log_key = log_key
            log_fn = self.log.warning if warning else self.log.info
            log_fn(
                "[TABLE_EDGE_CALIB] depth_shape=%s calib=%sx%s fx=%.3f fy=%.3f cx=%.3f cy=%.3f depth_scale=%.6f calib_source=%s warning=%s",
                depth_shape,
                calib_w,
                calib_h,
                fx,
                fy,
                cx,
                cy,
                float(payload["depth_scale"]),
                source,
                warning or "none",
            )
        return payload

    @staticmethod
    def _shape_hw(shape: Any) -> Optional[tuple[int, int]]:
        if not isinstance(shape, (list, tuple)) or len(shape) < 2:
            return None
        try:
            h = int(shape[0])
            w = int(shape[1])
        except Exception:
            return None
        if w <= 0 or h <= 0:
            return None
        return h, w

    @classmethod
    def _bbox_view_metrics(
        cls,
        bbox: Any,
        shape: Any,
        *,
        edge_margin_norm: float = 0.03,
        max_reliable_area: float = 0.90,
    ) -> Dict[str, Any]:
        hw = cls._shape_hw(shape)
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4 or hw is None:
            return {
                "cx_norm": None,
                "area_norm": None,
                "touch_left": False,
                "touch_right": False,
                "touch_top": False,
                "touch_bottom": False,
                "touch_boundary": False,
                "boundary_allowed": False,
                "reliable": False,
            }
        h, w = hw
        try:
            x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
        except Exception:
            return {
                "cx_norm": None,
                "area_norm": None,
                "touch_left": False,
                "touch_right": False,
                "touch_top": False,
                "touch_bottom": False,
                "touch_boundary": False,
                "boundary_allowed": False,
                "reliable": False,
            }
        x0, x1 = sorted((max(0.0, min(float(w), x0)), max(0.0, min(float(w), x1))))
        y0, y1 = sorted((max(0.0, min(float(h), y0)), max(0.0, min(float(h), y1))))
        bw = max(0.0, x1 - x0)
        bh = max(0.0, y1 - y0)
        area = (bw * bh) / max(1.0, float(w * h))
        margin_x = max(1.0, float(w) * float(edge_margin_norm))
        margin_y = max(1.0, float(h) * float(edge_margin_norm))
        touch_left = x0 <= margin_x
        touch_right = x1 >= float(w) - margin_x
        touch_top = y0 <= margin_y
        touch_bottom = y1 >= float(h) - margin_y
        cx_norm = (((x0 + x1) * 0.5) / max(1.0, float(w)) - 0.5) * 2.0
        touch_boundary = bool(touch_left or touch_right or touch_top or touch_bottom)
        return {
            "cx_norm": max(-1.0, min(1.0, float(cx_norm))),
            "area_norm": max(0.0, min(1.0, float(area))),
            "touch_left": bool(touch_left),
            "touch_right": bool(touch_right),
            "touch_top": bool(touch_top),
            "touch_bottom": bool(touch_bottom),
            "touch_boundary": touch_boundary,
            "boundary_allowed": bool(touch_left or touch_right or touch_bottom),
            "reliable": bool(area > 0.0 and area <= float(max_reliable_area)),
        }

    @classmethod
    def _plane_view_from_bbox(cls, bbox: Any, shape: Any, *, area_ratio: Optional[float] = None) -> Dict[str, Any]:
        metrics = cls._bbox_view_metrics(bbox, shape, max_reliable_area=1.0)
        width_norm = None
        hw = cls._shape_hw(shape)
        if hw is not None and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                width_norm = abs(float(bbox[2]) - float(bbox[0])) / max(1.0, float(hw[1]))
            except Exception:
                width_norm = None
        return {
            "plane_cx_norm": metrics["cx_norm"],
            "plane_width_norm": width_norm,
            "plane_area_ratio": area_ratio,
            "plane_touch_left": bool(metrics["touch_left"]),
            "plane_touch_right": bool(metrics["touch_right"]),
            "plane_touch_top": bool(metrics["touch_top"]),
            "plane_touch_bottom": bool(metrics["touch_bottom"]),
        }

    @staticmethod
    def _profile_template() -> Dict[str, float]:
        return {
            "depth_frame_fetch_ms": 0.0,
            "depth_preprocess_ms": 0.0,
            "roi_crop_ms": 0.0,
            "plane_or_edge_fit_ms": 0.0,
            "mask_or_binary_debug_ms": 0.0,
            "top_view_build_ms": 0.0,
            "preview_overlay_ms": 0.0,
            "total_edge_process_ms": 0.0,
            "frame_prepare_ms": 0.0,
            "roi_extract_ms": 0.0,
            "point_build_ms": 0.0,
            "candidate_select_ms": 0.0,
            "plane_fit_ms": 0.0,
            "residual_eval_ms": 0.0,
            "mask_build_ms": 0.0,
            "obs_build_ms": 0.0,
            "json_write_ms": 0.0,
            "preview_render_ms": 0.0,
            "preview_save_ms": 0.0,
            "loop_total_ms": 0.0,
            "payload_size_bytes": 0.0,
            "camera_frame_age_ms": 0.0,
            "camera_frame_interval_ms": 0.0,
            "camera_frames_hz": 0.0,
            "table_edge_worker_interval_ms": 0.0,
            "table_edge_no_new_frame_count": 0.0,
            "table_edge_process_interval_ms": 0.0,
            "table_edge_publish_interval_ms": 0.0,
            "table_edge_obs_hz": 0.0,
            "scheduler_read_ms": 0.0,
            "scheduler_publish_ms": 0.0,
            "fast_roi_extract_ms": 0.0,
            "fast_depth_valid_ms": 0.0,
            "fast_projection_ms": 0.0,
            
            "fast_height_filter_ms": 0.0,
            "fast_rep_select_ms": 0.0,
            "fast_front_cluster_fit_ms": 0.0,
            "fast_front_edge_ms": 0.0,
            "fast_local_band_ms": 0.0,
            "fast_background_protect_ms": 0.0,
            "fast_control_gate_ms": 0.0,
            "fast_debug_payload_ms": 0.0,
            "fast_total_ms": 0.0,
        }

    def _detector_mode_payload(self) -> Dict[str, Any]:
        return {
            "detector_mode": str(self._detector_mode),
            "fast_plane_stride": int(self._fast_plane_stride),
        }

    @staticmethod
    def _clip01(value: Any) -> float:
        try:
            out = float(value)
        except Exception:
            return 0.0
        if not math.isfinite(out):
            return 0.0
        return max(0.0, min(1.0, out))

    def _fast_temporal_score(self, *, yaw: float, dist: float, cx_norm: Optional[float]) -> Dict[str, Any]:
        prev = dict(self._fast_temporal_state or {})
        if not prev:
            self._fast_temporal_state = {
                "yaw": float(yaw),
                "dist": float(dist),
                "cx": cx_norm,
                "stable_count": 0,
            }
            return {
                "score": 0.0,
                "available": False,
                "stable_count": 0,
                "jump": False,
                "yaw_delta": None,
                "dist_delta": None,
                "cx_delta": None,
            }
        prev_cx = prev.get("cx")
        yaw_delta = abs(float(yaw) - float(prev.get("yaw", yaw)))
        dist_delta = abs(float(dist) - float(prev.get("dist", dist)))
        cx_delta = abs(float(cx_norm) - float(prev_cx)) if cx_norm is not None and prev_cx is not None else 0.0
        yaw_score = self._clip01(1.0 - yaw_delta / 0.30)
        dist_score = self._clip01(1.0 - dist_delta / 0.16)
        cx_score = self._clip01(1.0 - cx_delta / 0.16)
        stable_hits = int(yaw_delta < 0.15) + int(dist_delta < 0.08) + int(cx_delta < 0.08)
        prev_stable = int(prev.get("stable_count", 0) or 0)
        stable_count = prev_stable + 1 if stable_hits >= 2 else 0
        jump = bool(yaw_delta > 0.35 or dist_delta > 0.20 or cx_delta > 0.20)
        continuity = (yaw_score + dist_score + cx_score) / 3.0
        score = 0.0 if jump else self._clip01(0.80 * continuity + 0.20 * min(1.0, float(stable_count) / 3.0))
        self._fast_temporal_state = {
            "yaw": float(yaw),
            "dist": float(dist),
            "cx": cx_norm,
            "stable_count": int(stable_count),
        }
        return {
            "score": float(score),
            "available": True,
            "stable_count": int(stable_count),
            "jump": bool(jump),
            "yaw_delta": float(yaw_delta),
            "dist_delta": float(dist_delta),
            "cx_delta": float(cx_delta),
        }

    def _fast_debug_pixels_enabled(self) -> bool:
        cfg = getattr(self.cfg, "table_edge", None)
        if not bool(getattr(cfg, "fast_debug_pixels", True)):
            return False
        mode = str(self._source_mode_override or "").upper()
        is_offline = "OFFLINE" in mode or "BAG" in mode
        if is_offline:
            return bool(getattr(cfg, "fast_debug_pixels_offline", True))
        return bool(getattr(cfg, "fast_debug_pixels_online", False))

    def _fast_debug_pixel_cap(self) -> int:
        cfg = getattr(self.cfg, "table_edge", None)
        try:
            return max(0, int(float(getattr(cfg, "fast_debug_pixel_cap", 300) or 300)))
        except Exception:
            return 300

    @staticmethod
    def _sparse_pixel_pairs(px: Any, py: Any, *, cap: int) -> list:
        px_arr = np.asarray(px, dtype=np.int32)
        py_arr = np.asarray(py, dtype=np.int32)
        n = int(min(len(px_arr), len(py_arr)))
        if n <= 0 or int(cap) <= 0:
            return []
        cap_i = int(cap)
        step = max(1, int(math.ceil(float(n) / float(cap_i))))
        return [[int(x), int(y)] for x, y in zip(px_arr[:n][::step][:cap_i].tolist(), py_arr[:n][::step][:cap_i].tolist())]

    @staticmethod
    def _pixel_payload(enabled: bool, **items: Any) -> Dict[str, Any]:
        if not bool(enabled):
            return {}
        return {str(k): v for k, v in items.items()}

    @staticmethod
    def _empty_fast_edge_cue() -> Dict[str, Any]:
        empty_f = np.asarray([], dtype=np.float32)
        empty_i = np.asarray([], dtype=np.int32)
        return {
            "count": 0,
            "inlier_count": 0,
            "x_span_m": 0.0,
            "median_py": None,
            "residual_mean": 0.0,
            "k": 0.0,
            "b": 0.0,
            "yaw": 0.0,
            "dist": 0.0,
            "score": 0.0,
            "x": empty_f,
            "y": empty_f,
            "z": empty_f,
            "y_median_m": None,
            "px": empty_i,
            "py": empty_i,
            "inlier": np.asarray([], dtype=bool),
        }

    def _fast_edge_debug_payload(
        self,
        edge_cue: Dict[str, Any],
        *,
        debug_pixels_enabled: bool,
        debug_cap: int,
        skipped: bool,
        skip_reason: str,
    ) -> Dict[str, Any]:
        edge_px = np.asarray(edge_cue.get("px", []), dtype=np.int32)
        edge_py = np.asarray(edge_cue.get("py", []), dtype=np.int32)
        edge_inlier_mask = np.asarray(edge_cue.get("inlier", []), dtype=bool)
        if edge_inlier_mask.size == edge_px.size and int(edge_inlier_mask.sum()) > 0:
            edge_draw_px = edge_px[edge_inlier_mask]
            edge_draw_py = edge_py[edge_inlier_mask]
        else:
            edge_draw_px = edge_px
            edge_draw_py = edge_py
        out = {
            "fast_front_edge_skipped": bool(skipped),
            "fast_front_edge_skip_reason": str(skip_reason or ""),
            "fast_edge_candidate_count": int(edge_cue.get("count", 0) or 0),
            "fast_edge_inlier_count": int(edge_cue.get("inlier_count", 0) or 0),
            "fast_edge_x_span_m": float(edge_cue.get("x_span_m", 0.0) or 0.0),
            "fast_edge_y_median_px": edge_cue.get("median_py"),
            "fast_edge_residual": float(edge_cue.get("residual_mean", 0.0) or 0.0),
            "fast_edge_line_yaw_rad": edge_cue.get("yaw") if int(edge_cue.get("inlier_count", 0) or 0) > 0 else None,
            "fast_edge_line_dist_m": edge_cue.get("dist") if int(edge_cue.get("inlier_count", 0) or 0) > 0 else None,
            "fast_edge_support_score": float(edge_cue.get("score", 0.0) or 0.0),
            "fast_edge_col_step": int(edge_cue.get("col_step", self._fast_front_edge_col_step) or self._fast_front_edge_col_step),
            "fast_edge_row_step": int(edge_cue.get("row_step", self._fast_front_edge_row_step) or self._fast_front_edge_row_step),
            "fast_edge_pixel_count": int(min(len(edge_draw_px), max(0, int(debug_cap)))) if debug_pixels_enabled else 0,
        }
        out.update(self._pixel_payload(
            debug_pixels_enabled,
            fast_edge_pixels=self._sparse_pixel_pairs(edge_draw_px, edge_draw_py, cap=debug_cap),
        ))
        return out

    def _should_run_fast_verify(
        self,
        *,
        distance_stage: str,
        representative_inlier_count: int,
        support_inlier_count: int,
        selected_cluster_support: int,
        fit_inlier_x_span_m: float,
        residual_mean: float,
        residual_threshold: float,
        yaw_abs: float,
        max_yaw: float,
        selected_cluster_index: int,
        selected_cluster_score: float,
        background_rep_count: int,
        background_penalty_seed: float,
        temporal: Dict[str, Any],
        min_front_face_columns: int,
        min_vertical_support: int,
        min_front_face_x_span: float,
    ) -> tuple:
        if bool(temporal.get("jump", False)):
            return True, "temporal_jump"
        if not bool(temporal.get("available", False)):
            return True, "previous_obs_missing"
        min_support = int(min_front_face_columns) * int(min_vertical_support)
        if int(representative_inlier_count) < max(int(min_front_face_columns) * 2, 8):
            return True, "rep_inlier_weak"
        if int(support_inlier_count) < max(min_support * 2, 18):
            return True, "support_weak"
        if float(fit_inlier_x_span_m) < max(float(min_front_face_x_span) * 2.2, 0.28):
            return True, "fit_span_weak"
        if float(residual_mean) > max(1e-6, float(residual_threshold)) * 0.70:
            return True, "residual_high"
        if float(yaw_abs) > max(0.40, float(max_yaw) * 0.70):
            return True, "yaw_high"
        if int(selected_cluster_index) > 0:
            return True, "background_risk"
        if float(background_penalty_seed) > 0.05:
            return True, "background_penalty"
        if float(selected_cluster_score) < 0.55:
            return True, "cluster_score_low"
        stage = str(distance_stage or "unknown").strip().lower()
        return False, "strong_stable_near_core" if stage == "near" else "strong_stable_far_core"

    @staticmethod
    def _finite_percentiles(values: Any, qs=(10, 50, 90)) -> Dict[str, Optional[float]]:
        try:
            arr = np.asarray(values, dtype=np.float32)
            arr = arr[np.isfinite(arr)]
        except Exception:
            arr = np.asarray([], dtype=np.float32)
        if arr.size <= 0:
            return {f"p{int(q)}": None for q in qs}
        return {f"p{int(q)}": float(np.percentile(arr, q)) for q in qs}

    @staticmethod
    def _camera_points_to_robot(
        x_cam: Any,
        y_cam_down: Any,
        z_cam_forward: Any,
        *,
        pitch_deg: float,
        camera_height_m: float,
    ) -> tuple:
        """Convert RealSense camera coordinates to robot frame.

        Camera convention here is X right, Y down, Z forward. Robot convention is
        X lateral, Y forward along ground, Z upward from ground. Positive pitch_deg
        means the optical axis is pitched downward. With pitch=0, Y_robot=Z_cam and
        Z_robot=camera_height-Y_cam.
        """
        theta = math.radians(float(pitch_deg))
        c, s = math.cos(theta), math.sin(theta)
        x_r = np.asarray(x_cam, dtype=np.float32)
        y_down = np.asarray(y_cam_down, dtype=np.float32)
        z_fwd = np.asarray(z_cam_forward, dtype=np.float32)
        y_r = z_fwd * c - y_down * s
        z_r = float(camera_height_m) - (z_fwd * s + y_down * c)
        return x_r, y_r.astype(np.float32, copy=False), z_r.astype(np.float32, copy=False)

    @staticmethod
    def _fast_quantile_span(values: Any, low_q: float = 0.10, high_q: float = 0.90) -> float:
        arr = np.asarray(values)
        n = int(arr.size)
        if n <= 2:
            return 0.0
        lo = max(0, min(n - 1, int(round(float(low_q) * float(n - 1)))))
        hi = max(0, min(n - 1, int(round(float(high_q) * float(n - 1)))))
        if hi < lo:
            lo, hi = hi, lo
        part = np.partition(arr, (lo, hi))
        return float(part[hi] - part[lo])

    @staticmethod
    def _fast_median_value(values: Any) -> float:
        arr = np.asarray(values)
        n = int(arr.size)
        if n <= 0:
            return 0.0
        mid = n // 2
        if n % 2:
            return float(np.partition(arr, mid)[mid])
        part = np.partition(arr, (mid - 1, mid))
        return float((part[mid - 1] + part[mid]) * 0.5)

    def _select_fast_front_face_representatives(
        self,
        *,
        x_robot: Any,
        y_robot: Any,
        z_robot: Any,
        px: Any,
        py: Any,
        x_bin_width_m: float,
        y_cluster_bin_m: float,
        min_support_points: int,
        min_z_span_m: float,
    ) -> Dict[str, Any]:
        x_arr = np.asarray(x_robot, dtype=np.float32)
        y_arr = np.asarray(y_robot, dtype=np.float32)
        z_arr = np.asarray(z_robot, dtype=np.float32)
        px_arr = np.asarray(px, dtype=np.int32)
        py_arr = np.asarray(py, dtype=np.int32)
        n = int(min(len(x_arr), len(y_arr), len(z_arr), len(px_arr), len(py_arr)))
        if n <= 0:
            return {"count": 0}
        x_arr, y_arr, z_arr, px_arr, py_arr = x_arr[:n], y_arr[:n], z_arr[:n], px_arr[:n], py_arr[:n]
        x_bins = np.floor(x_arr / max(1e-6, float(x_bin_width_m))).astype(np.int32)
        reps = []
        x_order = np.argsort(x_bins, kind="mergesort")
        x_bins_sorted = x_bins[x_order]
        x_change = np.flatnonzero(np.diff(x_bins_sorted)) + 1
        x_starts = np.concatenate((np.asarray([0], dtype=np.int32), x_change.astype(np.int32, copy=False)))
        x_ends = np.concatenate((x_change.astype(np.int32, copy=False), np.asarray([len(x_bins_sorted)], dtype=np.int32)))
        y_radius_bins = max(1, int(math.ceil(0.08 / max(1e-6, float(y_cluster_bin_m)))))
        for start, end in zip(x_starts, x_ends):
            if int(end - start) < int(min_support_points):
                continue
            idxs = x_order[int(start):int(end)]
            y_bins = np.floor(y_arr[idxs] / max(1e-6, float(y_cluster_bin_m))).astype(np.int32)
            y_order = np.argsort(y_bins, kind="mergesort")
            y_bins_sorted = y_bins[y_order]
            idxs_sorted = idxs[y_order]
            best = None
            for yb in np.unique(y_bins_sorted):
                left = int(np.searchsorted(y_bins_sorted, int(yb) - y_radius_bins, side="left"))
                right = int(np.searchsorted(y_bins_sorted, int(yb) + y_radius_bins, side="right"))
                local = idxs_sorted[left:right]
                support = int(len(local))
                if support < int(min_support_points):
                    continue
                local_z = z_arr[local]
                z_span = float(np.max(local_z) - np.min(local_z)) if support > 1 else 0.0
                if z_span < float(min_z_span_m):
                    continue
                local_y = y_arr[local]
                y_spread = self._fast_quantile_span(local_y) if support > 2 else 0.0
                if y_spread > max(0.16, float(y_cluster_bin_m) * float(2 * y_radius_bins + 1)):
                    continue
                local_x = x_arr[local]
                local_px = px_arr[local]
                local_py = py_arr[local]
                score = float(support) * float(z_span) / max(0.04, y_spread + 0.02)
                item = {
                    "score": score,
                    "support": support,
                    "z_span": z_span,
                    "x": self._fast_median_value(local_x),
                    "y": self._fast_median_value(local_y),
                    "z": self._fast_median_value(local_z),
                    "px": int(self._fast_median_value(local_px)),
                    "py": int(self._fast_median_value(local_py)),
                    "y_spread": y_spread,
                    "support_px": local_px.astype(np.int32, copy=False),
                    "support_py": local_py.astype(np.int32, copy=False),
                    "support_x": local_x.astype(np.float32, copy=False),
                    "support_y": local_y.astype(np.float32, copy=False),
                    "support_z": local_z.astype(np.float32, copy=False),
                }
                if best is None or item["score"] > best["score"]:
                    best = item
            if best is not None:
                reps.append(best)
        if not reps:
            return {"count": 0}
        reps.sort(key=lambda item: item["x"])
        support_rep_index = [
            np.full(len(r["support_px"]), idx, dtype=np.int32)
            for idx, r in enumerate(reps)
            if len(r["support_px"]) > 0
        ]
        return {
            "count": int(len(reps)),
            "x": np.asarray([r["x"] for r in reps], dtype=np.float32),
            "y": np.asarray([r["y"] for r in reps], dtype=np.float32),
            "z": np.asarray([r["z"] for r in reps], dtype=np.float32),
            "px": np.asarray([r["px"] for r in reps], dtype=np.int32),
            "py": np.asarray([r["py"] for r in reps], dtype=np.int32),
            "support": np.asarray([r["support"] for r in reps], dtype=np.int32),
            "z_span": np.asarray([r["z_span"] for r in reps], dtype=np.float32),
            "y_spread": np.asarray([r["y_spread"] for r in reps], dtype=np.float32),
            "support_total": int(sum(int(r["support"]) for r in reps)),
            "support_px": np.concatenate([r["support_px"] for r in reps]).astype(np.int32, copy=False),
            "support_py": np.concatenate([r["support_py"] for r in reps]).astype(np.int32, copy=False),
            "support_x": np.concatenate([r["support_x"] for r in reps]).astype(np.float32, copy=False),
            "support_y": np.concatenate([r["support_y"] for r in reps]).astype(np.float32, copy=False),
            "support_z": np.concatenate([r["support_z"] for r in reps]).astype(np.float32, copy=False),
            "support_rep_index": np.concatenate(support_rep_index).astype(np.int32, copy=False) if support_rep_index else np.asarray([], dtype=np.int32),
        }

    @staticmethod
    def _weighted_line_fit(x: Any, y: Any, weights: Any = None) -> tuple:
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        if weights is None:
            k, b = np.polyfit(x_arr, y_arr, 1)
            return float(k), float(b)
        w_arr = np.asarray(weights, dtype=np.float32)
        n = int(min(len(x_arr), len(y_arr), len(w_arr)))
        if n <= 0:
            k, b = np.polyfit(x_arr, y_arr, 1)
            return float(k), float(b)
        w_arr = np.clip(w_arr[:n], 0.2, 3.0)
        if not np.all(np.isfinite(w_arr)) or float(np.max(w_arr)) <= 0.0:
            k, b = np.polyfit(x_arr[:n], y_arr[:n], 1)
        else:
            k, b = np.polyfit(x_arr[:n], y_arr[:n], 1, w=w_arr)
        return float(k), float(b)

    @staticmethod
    def _representative_neighbor_mask(x_values: Any, *, max_gap_m: float) -> np.ndarray:
        x_arr = np.asarray(x_values, dtype=np.float32)
        n = int(len(x_arr))
        if n < 4:
            return np.ones(n, dtype=bool)
        out = np.ones(n, dtype=bool)
        for idx in range(n):
            left = abs(float(x_arr[idx] - x_arr[idx - 1])) if idx > 0 else float("inf")
            right = abs(float(x_arr[idx + 1] - x_arr[idx])) if idx < n - 1 else float("inf")
            out[idx] = min(left, right) <= float(max_gap_m)
        return out

    def _extract_fast_front_edge_cue(
        self,
        *,
        depth_m: Any,
        x0: int,
        y0: int,
        stride: int,
        calib: Any,
        pitch_deg: float,
        camera_height_m: float,
        z_min: float,
        z_max: float,
        target_dist_m: float,
        min_x_span_m: float,
        max_yaw: float,
    ) -> Dict[str, Any]:
        # Current implementation intentionally keeps the legacy column/row scan
        # semantics. This method is the replacement point for a future vectorized
        # NumPy/OpenCV front-edge cue; keep callers and output keys stable.
        arr = np.asarray(depth_m, dtype=np.float32)
        if arr.ndim != 2 or arr.size <= 0:
            return {"count": 0, "inlier_count": 0, "score": 0.0}
        h, w = int(arr.shape[0]), int(arr.shape[1])
        if h < 7 or w < 4:
            return {"count": 0, "inlier_count": 0, "score": 0.0}
        candidates = []
        col_step = max(1, int(self._fast_front_edge_col_step or 1))
        row_step = max(1, int(self._fast_front_edge_row_step or 1))
        for col in range(0, w, col_step):
            prof = arr[:, col]
            valid = np.isfinite(prof) & (prof > float(z_min)) & (prof < float(z_max))
            if int(valid.sum()) < 7:
                continue
            filled = prof.astype(np.float32, copy=True)
            finite_idx = np.flatnonzero(valid)
            if finite_idx.size < 7:
                continue
            filled[~valid] = np.interp(np.flatnonzero(~valid), finite_idx, prof[finite_idx]).astype(np.float32) if int((~valid).sum()) else filled[~valid]
            smooth = np.convolve(filled, np.asarray([0.25, 0.50, 0.25], dtype=np.float32), mode="same")
            best = None
            for row in range(3, h - 3, row_step):
                if not bool(valid[row]):
                    continue
                if not (smooth[row] < smooth[row - 1] and smooth[row] <= smooth[row + 1]):
                    continue
                before = float(np.percentile(smooth[max(0, row - 6):row], 70))
                after = float(np.percentile(smooth[row + 1:min(h, row + 7)], 70))
                valley = float(smooth[row])
                # A front edge in pitched camera depth often appears as far -> near -> far along image v.
                prominence = min(before - valley, after - valley)
                if prominence < 0.025:
                    continue
                if row < 4 or row > h - 5:
                    continue
                item = (float(prominence), int(col), int(row), float(prof[row]))
                if best is None or item[0] > best[0]:
                    best = item
            if best is not None:
                candidates.append(best)
        if len(candidates) < 3:
            return {"count": int(len(candidates)), "inlier_count": 0, "score": 0.0}

        candidates.sort(key=lambda item: item[1])
        groups = []
        current = [candidates[0]]
        for item in candidates[1:]:
            if int(item[1]) - int(current[-1][1]) <= 2:
                current.append(item)
            else:
                groups.append(current)
                current = [item]
        groups.append(current)
        group = max(groups, key=lambda g: (len(g), sum(float(v[0]) for v in g)))
        if len(group) < 3:
            return {"count": int(len(candidates)), "inlier_count": 0, "score": 0.0}

        cols = np.asarray([g[1] for g in group], dtype=np.float32)
        rows = np.asarray([g[2] for g in group], dtype=np.float32)
        depths = np.asarray([g[3] for g in group], dtype=np.float32)
        px = (float(x0) + cols * float(stride)).astype(np.int32)
        py = (float(y0) + rows * float(stride)).astype(np.int32)
        u = px.astype(np.float32)
        v = py.astype(np.float32)
        x_c = (u - float(calib.cx)) * depths / float(calib.fx)
        y_c = (v - float(calib.cy)) * depths / float(calib.fy)
        x_r, y_r, z_r = self._camera_points_to_robot(
            x_c,
            y_c,
            depths,
            pitch_deg=pitch_deg,
            camera_height_m=camera_height_m,
        )
        try:
            k, b = self._weighted_line_fit(x_r, y_r, np.ones_like(x_r, dtype=np.float32))
            residual = np.abs(y_r - (float(k) * x_r + float(b)))
            threshold = 0.055
            inlier = residual <= threshold
            if int(inlier.sum()) >= 3:
                k, b = self._weighted_line_fit(x_r[inlier], y_r[inlier], np.ones(int(inlier.sum()), dtype=np.float32))
                residual = np.abs(y_r - (float(k) * x_r + float(b)))
                inlier = residual <= threshold
            inlier_count = int(inlier.sum())
            inlier_x = x_r[inlier] if inlier_count > 0 else np.asarray([], dtype=np.float32)
            x_span = float(np.max(inlier_x) - np.min(inlier_x)) if inlier_x.size > 1 else 0.0
            residual_mean = float(np.mean(residual[inlier])) if inlier_count > 0 else float(np.mean(residual))
            yaw = math.atan(float(k))
            dist = float(b) - float(target_dist_m)
            span_score = self._clip01(x_span / max(1e-6, float(min_x_span_m) * 2.0))
            count_score = self._clip01(float(inlier_count) / 10.0)
            residual_score = max(0.0, 1.0 - residual_mean / threshold)
            yaw_score = self._clip01(1.0 - abs(float(yaw)) / max(1e-6, float(max_yaw)))
            score = float(0.35 * span_score + 0.25 * count_score + 0.25 * residual_score + 0.15 * yaw_score)
        except Exception:
            k, b, yaw, dist = 0.0, 0.0, 0.0, 0.0
            residual = np.zeros(len(px), dtype=np.float32)
            inlier = np.zeros(len(px), dtype=bool)
            inlier_count, x_span, residual_mean, score = 0, 0.0, 0.0, 0.0
        return {
            "count": int(len(candidates)),
            "inlier_count": int(inlier_count),
            "x_span_m": float(x_span),
            "median_py": float(np.median(py)) if len(py) else None,
            "residual_mean": float(residual_mean),
            "k": float(k),
            "b": float(b),
            "yaw": float(yaw),
            "dist": float(dist),
            "score": float(score),
            "x": x_r.astype(np.float32, copy=False),
            "y": y_r.astype(np.float32, copy=False),
            "z": z_r.astype(np.float32, copy=False),
            "y_median_m": float(np.median(y_r)) if len(y_r) else None,
            "px": px.astype(np.int32, copy=False),
            "py": py.astype(np.int32, copy=False),
            "inlier": inlier.astype(bool, copy=False),
            "col_step": int(col_step),
            "row_step": int(row_step),
        }

    @staticmethod
    def _fast_local_band_stats(x_values: Any, y_values: Any, edge_x: Any, edge_y: Any, *, k: float, b: float, band_m: float) -> Dict[str, Any]:
        x_arr = np.asarray(x_values, dtype=np.float32)
        y_arr = np.asarray(y_values, dtype=np.float32)
        n = int(min(len(x_arr), len(y_arr)))
        if n <= 0:
            return {"count": 0, "x_span_m": 0.0, "edge_support": 0, "residual_mean": 0.0}
        residual = np.abs(y_arr[:n] - (float(k) * x_arr[:n] + float(b)))
        mask = residual <= float(band_m)
        xs = x_arr[:n][mask]
        edge_x_arr = np.asarray(edge_x, dtype=np.float32)
        edge_y_arr = np.asarray(edge_y, dtype=np.float32)
        if len(edge_x_arr) and len(edge_y_arr):
            en = int(min(len(edge_x_arr), len(edge_y_arr)))
            edge_res = np.abs(edge_y_arr[:en] - (float(k) * edge_x_arr[:en] + float(b)))
            edge_support = int(np.sum(edge_res <= float(band_m)))
        else:
            edge_support = 0
        return {
            "count": int(mask.sum()),
            "x_span_m": float(np.max(xs) - np.min(xs)) if xs.size > 1 else 0.0,
            "edge_support": int(edge_support),
            "residual_mean": float(np.mean(residual[mask])) if int(mask.sum()) > 0 else float(np.mean(residual)),
        }

    def _fit_fast_front_cluster_line(
        self,
        *,
        x_t: Any,
        y_t: Any,
        rep_support: Any,
        rep_z_span: Any,
        min_front_face_columns: int,
        min_front_face_x_span: float,
        min_vertical_support: int,
        min_vertical_z_span: float,
        residual_threshold: float,
        max_yaw: float,
        x_bin_width_m: float,
        front_cluster_gap_m: float,
        target_dist_m: float,
    ) -> Dict[str, Any]:
        x_arr = np.asarray(x_t, dtype=np.float32)
        y_arr = np.asarray(y_t, dtype=np.float32)
        support_arr = np.asarray(rep_support, dtype=np.int32)
        z_span_arr = np.asarray(rep_z_span, dtype=np.float32)
        n = int(min(len(x_arr), len(y_arr), len(support_arr)))
        if n <= 0:
            return {"selected": None, "clusters": [], "reject_reason": "no_front_cluster"}
        x_arr, y_arr, support_arr = x_arr[:n], y_arr[:n], support_arr[:n]
        z_span_arr = z_span_arr[:n] if len(z_span_arr) >= n else np.zeros(n, dtype=np.float32)

        # Robot Y is forward distance along the ground; smaller Y is nearer/front-most.
        order_y = np.argsort(y_arr, kind="mergesort")
        clusters: list = []
        current = [int(order_y[0])]
        for raw_idx in order_y[1:]:
            idx = int(raw_idx)
            prev = current[-1]
            if float(y_arr[idx] - y_arr[prev]) > float(front_cluster_gap_m):
                clusters.append(current)
                current = [idx]
            else:
                current.append(idx)
        clusters.append(current)

        cluster_infos = []
        min_support_sum = int(min_front_face_columns) * int(min_vertical_support)
        hard_span_min = max(float(min_front_face_x_span), float(min_front_face_x_span) * 0.90)
        for cluster_index, idx_list in enumerate(clusters):
            idx = np.asarray(sorted(idx_list, key=lambda i: float(x_arr[i])), dtype=np.int32)
            cx = x_arr[idx]
            cy = y_arr[idx]
            cs = support_arr[idx]
            cz_span = z_span_arr[idx]
            rep_count = int(len(idx))
            support_sum = int(np.sum(cs)) if cs.size else 0
            x_span = float(np.max(cx) - np.min(cx)) if cx.size > 1 else 0.0
            y_center = float(np.median(cy)) if cy.size else None
            y_min = float(np.min(cy)) if cy.size else None
            y_max = float(np.max(cy)) if cy.size else None
            z_span_p50 = float(np.percentile(cz_span, 50)) if cz_span.size else 0.0
            z_span_max = float(np.max(cz_span)) if cz_span.size else 0.0
            info: Dict[str, Any] = {
                "index": int(cluster_index),
                "rep_indices": idx,
                "rep_count": rep_count,
                "support_sum": support_sum,
                "x_span_m": x_span,
                "y_center": y_center,
                "y_min": y_min,
                "y_max": y_max,
                "z_span_p50": z_span_p50,
                "z_span_max": z_span_max,
                "valid": False,
                "invalid_reason": "",
                "score": 0.0,
            }
            if rep_count < int(min_front_face_columns):
                info["invalid_reason"] = "front_cluster_weak"
                cluster_infos.append(info)
                continue
            if x_span < hard_span_min:
                info["invalid_reason"] = "front_face_x_span_low"
                cluster_infos.append(info)
                continue
            if support_sum < min_support_sum:
                info["invalid_reason"] = "vertical_support_low"
                cluster_infos.append(info)
                continue
            try:
                support_w = np.sqrt(np.clip(cs.astype(np.float32), 1.0, 36.0))
                z_w = np.clip(cz_span / max(1e-6, float(min_vertical_z_span)), 0.5, 1.5) if cz_span.size else 1.0
                weights = np.clip(support_w * z_w, 0.5, 3.0)
                k, b = self._weighted_line_fit(cx, cy, weights)
                residual = np.abs(cy - (float(k) * cx + float(b)))
                neighbor_mask = self._representative_neighbor_mask(cx, max_gap_m=max(0.18, float(x_bin_width_m) * 5.0))
                inlier_local = (residual <= float(residual_threshold)) & neighbor_mask
                if int(inlier_local.sum()) >= int(min_front_face_columns):
                    k, b = self._weighted_line_fit(cx[inlier_local], cy[inlier_local], weights[inlier_local])
                    residual = np.abs(cy - (float(k) * cx + float(b)))
                    inlier_local = (residual <= float(residual_threshold)) & neighbor_mask
                inlier_count = int(inlier_local.sum())
                inlier_x = cx[inlier_local] if inlier_count > 0 else np.asarray([], dtype=np.float32)
                inlier_x_span = float(np.max(inlier_x) - np.min(inlier_x)) if inlier_x.size > 1 else 0.0
                residual_mean = float(np.mean(residual[inlier_local])) if inlier_count > 0 else float(np.mean(residual))
                residual_p90 = float(np.percentile(residual[inlier_local], 90)) if inlier_count > 0 else float(np.percentile(residual, 90))
                yaw = math.atan(float(k))
                dist = float(b) - float(target_dist_m)
                inlier_support = int(np.sum(cs[inlier_local])) if inlier_count > 0 else 0
            except Exception as exc:
                info["invalid_reason"] = f"cluster_fit_failed:{exc}"
                cluster_infos.append(info)
                continue

            info.update({
                "k": float(k),
                "b": float(b),
                "yaw": float(yaw),
                "dist": float(dist),
                "residual": residual.astype(np.float32, copy=False),
                "inlier_local": inlier_local.astype(bool, copy=False),
                "inlier_count": int(inlier_count),
                "inlier_support": int(inlier_support),
                "inlier_x_span_m": float(inlier_x_span),
                "residual_mean": float(residual_mean),
                "residual_p90": float(residual_p90),
            })
            strict_for_farther = cluster_index > 0
            if inlier_count < int(min_front_face_columns):
                info["invalid_reason"] = "front_cluster_weak"
            elif inlier_support < min_support_sum:
                info["invalid_reason"] = "vertical_support_low"
            elif inlier_x_span < hard_span_min:
                info["invalid_reason"] = "front_face_x_span_low"
            elif residual_mean > float(residual_threshold):
                info["invalid_reason"] = "residual_too_large"
            elif abs(float(yaw)) > float(max_yaw):
                info["invalid_reason"] = "yaw_out_of_range"
            elif strict_for_farther and (
                inlier_count < max(int(min_front_face_columns) + 1, 4)
                or inlier_x_span < max(float(min_front_face_x_span) * 1.5, 0.15)
                or inlier_support < min_support_sum * 2
                or residual_mean > float(residual_threshold) * 0.80
            ):
                info["invalid_reason"] = "selected_cluster_invalid"
            else:
                info["valid"] = True
                frontness_score = max(0.0, 1.0 - 0.35 * float(cluster_index))
                span_score = self._clip01(inlier_x_span / max(1e-6, float(min_front_face_x_span) * 2.0))
                support_score = self._clip01(float(inlier_support) / float(max(1, min_support_sum * 3)))
                residual_score = max(0.0, 1.0 - residual_mean / max(1e-6, float(residual_threshold)))
                info["score"] = float(5.0 * frontness_score + 1.2 * span_score + 1.0 * support_score + 1.0 * residual_score)
            cluster_infos.append(info)

        selected = None
        for info in cluster_infos:
            if bool(info.get("valid")):
                selected = info
                break
        if selected is None:
            reject = "no_front_cluster" if not cluster_infos else str(cluster_infos[0].get("invalid_reason") or "front_cluster_weak")
        else:
            reject = "none"
        return {"selected": selected, "clusters": cluster_infos, "reject_reason": reject}

    @staticmethod
    def _sparse_pixel_sample(xs: Any, ys: Any, x0: int, y0: int, stride: int, *, cap: int = 1000) -> list:
        try:
            xs_arr = np.asarray(xs)
            ys_arr = np.asarray(ys)
            n = int(min(len(xs_arr), len(ys_arr)))
        except Exception:
            return []
        if n <= 0:
            return []
        cap = max(1, int(cap))
        step = max(1, int(math.ceil(float(n) / float(cap))))
        out = []
        for ix, iy in zip(xs_arr[:n:step], ys_arr[:n:step]):
            out.append([int(x0 + int(ix) * int(stride)), int(y0 + int(iy) * int(stride))])
            if len(out) >= cap:
                break
        return out

    @staticmethod
    def _mask_rle_payload(mask: Any, roi_box: Any) -> Optional[Dict[str, Any]]:
        try:
            arr = np.asarray(mask).astype(bool)
        except Exception:
            return None
        if arr.ndim != 2 or arr.size <= 0 or int(arr.sum()) <= 0:
            return None
        roi = None
        try:
            if isinstance(roi_box, (list, tuple)) and len(roi_box) >= 4:
                roi = [int(round(float(v))) for v in roi_box[:4]]
        except Exception:
            roi = None
        flat = np.ascontiguousarray(arr.reshape(-1).astype(np.uint8))
        padded = np.concatenate(([0], flat, [0]))
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        counts = []
        for start, end in zip(changes[0::2], changes[1::2]):
            counts.extend([int(start), int(end - start)])
        return {
            "encoding": "rle",
            "coord": "roi" if roi is not None else "full",
            "shape": [int(arr.shape[0]), int(arr.shape[1])],
            "roi": roi,
            "counts": counts,
            "sum": int(arr.sum()),
        }

    def _plane_debug_payload(self, debug: Any, roi_box: Any) -> Dict[str, Any]:
        if not isinstance(debug, dict):
            return {"plane_mask_status": "missing"}
        front_plane = debug.get("front_plane") if isinstance(debug.get("front_plane"), dict) else {}
        out: Dict[str, Any] = {
            "front_plane_candidate_pixels": debug.get("front_plane_candidate_pixels") or [],
            "crease_candidate_pixels": debug.get("crease_candidate_pixels") or [],
            "crease_inlier_pixels": debug.get("crease_inlier_pixels") or [],
            "upper_line_candidate_pixels": debug.get("upper_line_candidate_pixels") or [],
            "upper_line_inlier_pixels": debug.get("upper_line_inlier_pixels") or [],
            "lower_line_candidate_pixels": debug.get("lower_line_candidate_pixels") or [],
            "lower_line_inlier_pixels": debug.get("lower_line_inlier_pixels") or [],
        }
        candidate = self._mask_rle_payload(front_plane.get("candidate_mask"), roi_box)
        inlier = self._mask_rle_payload(front_plane.get("inlier_mask"), roi_box)
        if candidate is not None:
            out["front_plane_candidate_mask"] = candidate
        if inlier is not None:
            out["front_plane_inlier_mask"] = inlier
        out["plane_mask_status"] = "present" if inlier is not None or candidate is not None or bool(out["front_plane_candidate_pixels"]) else "missing"
        return out

    def _attach_profile(self, payload: Dict[str, Any], profile: Dict[str, Any], *, path: str) -> Dict[str, Any]:
        out = dict(payload or {})
        self._update_edge_stability(out)
        if out.get("fast_debug_pixels_enabled") is False:
            for key in (
                "fast_sampled_pixels",
                "fast_candidate_pixels",
                "fast_support_pixels",
                "fast_front_face_rep_pixels",
                "fast_inlier_pixels",
                "fast_outlier_pixels",
                "fast_background_pixels",
                "fast_rep_background_pixels",
                "fast_weak_pixels",
                "fast_edge_pixels",
            ):
                out.pop(key, None)
        prof = self._profile_template()
        prof.update({k: float(v) for k, v in dict(profile or {}).items() if isinstance(v, (int, float))})
        prof["total_edge_process_ms"] = float(prof.get("total_edge_process_ms", 0.0) or 0.0)
        if str(path or "").startswith("fast_plane_only") and float(prof.get("fast_total_ms", 0.0) or 0.0) <= 0.0:
            prof["fast_total_ms"] = float(prof.get("total_edge_process_ms", 0.0) or 0.0)
        out.update(prof)
        out.update(dict(self._frame_calib_payload or {}))
        out.setdefault("preview_enabled", False)
        out.setdefault("preview_skipped_reason", "disabled")
        out["edge_profile"] = dict(prof)
        out["edge_profile"].update(dict(self._frame_calib_payload or {}))
        out["edge_profile"]["preview_enabled"] = bool(out.get("preview_enabled", False))
        out["edge_profile"]["preview_skipped_reason"] = str(out.get("preview_skipped_reason") or "")
        for key in (
            "fast_front_edge_skipped",
            "fast_front_edge_skip_reason",
            "fast_local_band_skipped",
            "fast_local_band_skip_reason",
            "fast_debug_pixels_enabled",
            "fast_debug_pixel_cap",
        ):
            if key in out:
                out["edge_profile"][key] = out.get(key)
        out["edge_process_path"] = str(path or "")
        cfg = getattr(self.cfg, "table_edge", None)
        max_residual = getattr(cfg, "edge_trusted_max_residual", 0.0) if cfg is not None else 0.0
        try:
            max_residual = float(max_residual or 0.0)
        except Exception:
            max_residual = 0.0
        out = standardize_table_edge_payload(
            out,
            edge_stable_required_frames=max(1, int(getattr(cfg, "yolo_table_edge_stable_frames", 5) if cfg is not None else 5)),
            edge_trusted_min_conf=float(getattr(cfg, "edge_trusted_min_conf", 0.60) if cfg is not None else 0.60),
            edge_trusted_max_residual=max_residual if max_residual > 0.0 else None,
        )
        return out

    def _update_edge_stability(self, payload: Dict[str, Any]) -> None:
        cfg = getattr(self.cfg, "table_edge", None)
        required = max(1, int(getattr(cfg, "yolo_table_edge_stable_frames", 5) or 5))
        found = bool(payload.get("edge_found", False))
        conf = float(payload.get("confidence") or payload.get("edge_conf") or 0.0)
        yaw = payload.get("yaw_err_rad")
        dist = payload.get("dist_err_m")
        stable = False
        if found and yaw is not None and dist is not None and conf >= float(getattr(cfg, "yolo_table_conf_min", 0.25) or 0.25):
            prev = dict(self._edge_stability_prev or {})
            try:
                yaw_delta = abs(float(yaw) - float(prev.get("yaw", yaw)))
                dist_delta = abs(float(dist) - float(prev.get("dist", dist)))
            except Exception:
                yaw_delta = 0.0
                dist_delta = 0.0
            stable = bool(yaw_delta <= 0.15 and dist_delta <= 0.08)
        self._edge_stable_count = int(self._edge_stable_count + 1) if stable else (1 if found else 0)
        self._edge_stability_prev = {"yaw": yaw, "dist": dist, "conf": conf, "found": found}
        if dist is not None:
            try:
                self._last_measured_edge_dist_m = float(payload.get("target_dist_m") or self._target_dist_m) + float(dist)
            except Exception:
                self._last_measured_edge_dist_m = None
        payload["edge_stable_count"] = int(self._edge_stable_count)
        payload["edge_stable_required_frames"] = int(required)
        payload["edge_stable_for_yolo_blend"] = bool(self._edge_stable_count >= required)

    def _default_result(
        self,
        *,
        depth_valid: bool,
        reason: str,
        frame_seq: int,
        roi_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        roi = self._roi_payload(roi_meta=roi_meta)
        reason_text = str(reason or "")
        edge_unavailable = (not bool(depth_valid)) or reason_text in {
            "depth_unavailable",
            "depth_frame_missing",
            "depth_frame_not_2d",
            "detector_unavailable",
        } or reason_text.startswith("detect_failed:")
        return {
            "table_found": False,
            "edge_found": False,
            "edge_detected": False,
            "edge_geometry_valid": False,
            "edge_valid": False,
            "edge_stable": False,
            "edge_trusted": False,
            "edge_quality": {},
            "edge_trust_reason": "",
            "edge_reject_for_control_reason": str(reason or ""),
            "confidence": 0.0,
            "edge_conf": 0.0,
            "yaw_err_rad": None,
            "yaw_err": None,
            "dist_err_m": None,
            "dist_err": None,
            "edge_k": None,
            "edge_b": None,
            "depth_valid": bool(depth_valid),
            "edge_obs_unavailable": bool(edge_unavailable),
            "point_count": 0,
            "table_point_count": 0,
            "frame_id": int(self._frame_id),
            "frame_seq": int(frame_seq),
            "source": "vision_table_edge_manager",
            "reason": str(reason or ""),
            "reject_reason": str(reason or ""),
            "target_dist_m": float(self._target_dist_m),
            "plane_only_mode": bool(getattr(self._detector_cfg, "plane_only_mode", False)),
            "enable_crease_line": bool(getattr(self._detector_cfg, "enable_crease_line", True)),
            **self._detector_mode_payload(),
            "table_confirmed_by_yolo": False,
            "table_bbox_current_found": False,
            "table_bbox_control_valid": False,
            "table_bbox_found": False,
            "table_bbox_xyxy": None,
            "table_bbox_source": "none",
            "table_bbox_area_ratio": None,
            "table_bbox_conf_raw": None,
            "table_bbox_conf_used_for_gate": False,
            "yolo_table_conf": None,
            "yolo_gate_reason": str(reason or ""),
            "yolo_reliable": False,
            "yolo_gate_open": False,
            "yolo_valid_reason": "",
            "yolo_invalid_reason": str(reason or ""),
            "docking_enabled_by_yolo": False,
            "edge_control_allowed": False,
            "edge_control_block_reason": str(reason or "table_bbox_unavailable"),
            "yolo_bbox_area_norm": None,
            "yolo_bbox_touch_left": False,
            "yolo_bbox_touch_right": False,
            "yolo_bbox_touch_bottom": False,
            "yolo_bbox_touch_boundary": False,
            "table_bbox_touch_left": False,
            "table_bbox_touch_right": False,
            "table_bbox_touch_bottom": False,
            "table_bbox_boundary_allowed": False,
            "yolo_table_control_valid": False,
            "yolo_table_roi_valid": False,
            "roi_phase": "static_bottom_fallback",
            "plane_cx_norm": None,
            "plane_width_norm": None,
            "plane_area_ratio": None,
            "plane_touch_left": False,
            "plane_touch_right": False,
            "plane_touch_top": False,
            "plane_touch_bottom": False,
            "view_err_norm": None,
            "view_source": "none",
            "view_reliable": False,
            "fov_guard_active": False,
            "valid_for_control": False,
            "usable_for_approach": False,
            "usable_for_alignment": False,
            "usable_for_stop": False,
            "control_level": "none",
            "control_reject_reason": str(reason or ""),
            **roi,
            "type": "table_edge_obs",
        }

    def _yolo_table_confirmation(self, local: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        require_yolo = self._require_yolo_confirm
        plane_only = bool(getattr(self._detector_cfg, "plane_only_mode", False))
        local_payload = dict(local if local is not None else self._local_perception())
        min_conf = float(self._yolo_table_min_conf)
        table_bbox = table_detection_debug(local_payload, local_payload.get("rgb_shape"), min_conf=-1.0).get("bbox")
        det = table_detection_debug(local_payload, local_payload.get("rgb_shape"), min_conf=min_conf)
        source = str(local_payload.get("table_roi_source") or "").strip()
        bbox_found = table_bbox is not None
        confirmed = bool(bbox_found)
        reason = "table_bbox_found" if bbox_found else "table_bbox_unavailable"
        if not bbox_found and source:
            reason = f"table_source_{source}"
        bbox_metrics = self._bbox_view_metrics(table_bbox, local_payload.get("rgb_shape"))
        bbox_conf = det.get("conf")
        yolo_reliable = bool(confirmed)
        gate_open = bool(confirmed)
        if not gate_open and not require_yolo and plane_only and self._enable_yolo_in_plane_only:
            gate_open = False
        return {
            "table_confirmed_by_yolo": confirmed,
            "table_bbox_current_found": bool(bbox_found),
            "table_bbox_control_valid": bool(bbox_found),
            "table_bbox_found": bool(bbox_found),
            "table_bbox_xyxy": table_bbox,
            "table_bbox_source": "yolo_table_bbox" if bbox_found else (source or "none"),
            "table_bbox_area_ratio": bbox_metrics.get("area_norm"),
            "table_bbox_conf_raw": bbox_conf,
            "table_bbox_conf_used_for_gate": False,
            "yolo_table_conf": bbox_conf,
            "yolo_gate_reason": reason,
            "yolo_table_bbox": table_bbox,
            "yolo_reliable": yolo_reliable,
            "yolo_gate_open": bool(gate_open),
            "yolo_valid_reason": "table_bbox_found" if bbox_found else "",
            "yolo_invalid_reason": "" if bbox_found else reason,
            "docking_enabled_by_yolo": bool(bbox_found),
            "edge_control_allowed": bool(bbox_found),
            "edge_control_block_reason": "" if bbox_found else "table_bbox_unavailable",
            "yolo_bbox_area_norm": bbox_metrics.get("area_norm"),
            "yolo_bbox_touch_left": bool(bbox_metrics.get("touch_left", False)),
            "yolo_bbox_touch_right": bool(bbox_metrics.get("touch_right", False)),
            "yolo_bbox_touch_bottom": bool(bbox_metrics.get("touch_bottom", False)),
            "yolo_bbox_touch_boundary": bool(bbox_metrics.get("touch_boundary", False)),
            "table_bbox_touch_left": bool(bbox_metrics.get("touch_left", False)),
            "table_bbox_touch_right": bool(bbox_metrics.get("touch_right", False)),
            "table_bbox_touch_bottom": bool(bbox_metrics.get("touch_bottom", False)),
            "table_bbox_boundary_allowed": bool(bbox_metrics.get("boundary_allowed", False)),
            "yolo_table_control_valid": bool(bbox_found),
            "table_cx_norm": bbox_metrics.get("cx_norm"),
            "table_size_norm": bbox_metrics.get("area_norm"),
        }

    def _static_roi(self) -> Optional[list[int]]:
        cfg = self._detector_cfg
        if cfg is None:
            return None
        return [
            int(getattr(cfg, "roi_x0", 0) or 0),
            int(getattr(cfg, "roi_y0", 0) or 0),
            int(getattr(cfg, "roi_x1", 0) or 0),
            int(getattr(cfg, "roi_y1", 0) or 0),
        ]

    def _manual_static_roi_enabled(self) -> bool:
        return bool(self._static_roi_enabled)

    def _debug_roi_preset(self) -> str:
        return str(getattr(getattr(self.cfg, "table_edge", None), "roi_preset", "") or "").strip().lower()

    def _local_perception(self) -> Dict[str, Any]:
        if self._local_perception_override is not None:
            return dict(self._local_perception_override)
        scheduler = self._scheduler
        if scheduler is None:
            return {}
        try:
            value = scheduler.read_result("local_perception", default={}) or {}
        except Exception:
            return {}
        return dict(value) if isinstance(value, dict) else {}

    def _runtime_status(self) -> Dict[str, Any]:
        if self._runtime_status_override is not None:
            return dict(self._runtime_status_override)
        scheduler = self._scheduler
        if scheduler is None:
            return {}
        try:
            value = scheduler.read_result("runtime_status", default={}) or {}
        except Exception:
            return {}
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _locked_roi_meta(runtime_status: Dict[str, Any], depth_shape: Optional[object]) -> Optional[Dict[str, Any]]:
        mode = str((runtime_status or {}).get("mode") or "").strip().upper()
        if mode != "FIND_OBJECT":
            return None
        roi_raw = (runtime_status or {}).get("locked_roi")
        if not isinstance(roi_raw, (list, tuple)) or len(roi_raw) != 4:
            return None
        try:
            x0, y0, x1, y1 = [int(round(float(v))) for v in roi_raw]
        except Exception:
            return None
        if isinstance(depth_shape, tuple) and len(depth_shape) >= 2:
            h = int(depth_shape[0] or 0)
            w = int(depth_shape[1] or 0)
            if w > 0 and h > 0:
                x0 = max(0, min(w - 1, x0))
                x1 = max(0, min(w, x1))
                y0 = max(0, min(h - 1, y0))
                y1 = max(0, min(h, y1))
        if x1 <= x0 or y1 <= y0:
            return None
        roi = [x0, y0, x1, y1]
        return {
            "depth_edge_roi": roi,
            "table_edge_roi": roi,
            "edge_roi": roi,
            "roi_source": "locked_edge_roi",
            "roi_reason": "track_local_locked_edge",
            "locked_edge_id": runtime_status.get("locked_edge_id"),
            "locked_edge_line": runtime_status.get("locked_edge_line"),
            "locked_roi": list(roi),
            "locked_yaw_err": runtime_status.get("locked_yaw_err"),
            "locked_dist_err": runtime_status.get("locked_dist_err"),
            "locked_edge_conf": runtime_status.get("locked_edge_conf"),
            "locked_obs_seq": runtime_status.get("locked_obs_seq"),
        }

    def _select_roi(self, depth_frame: Optional[np.ndarray]) -> Dict[str, Any]:
        fallback = self._static_roi()
        runtime_status = self._runtime_status()
        depth_shape = getattr(depth_frame, "shape", None)
        locked_meta = self._locked_roi_meta(runtime_status, depth_shape)
        if locked_meta is not None:
            return locked_meta
        local = self._local_perception()
        manual_static = self._manual_static_roi_enabled()
        roi_preset = self._debug_roi_preset()
        table_edge_cfg = getattr(self.cfg, "table_edge", None)
        dynamic_enable = bool(getattr(table_edge_cfg, "yolo_table_roi_enable", True))
        stable_required = max(1, int(getattr(table_edge_cfg, "yolo_table_edge_stable_frames", 5) or 5))
        edge_stable = bool(self._edge_stable_count >= stable_required)
        near_dist_m = float(getattr(table_edge_cfg, "yolo_table_near_dist_m", 0.45) or 0.45)
        near_distance = bool(
            self._last_measured_edge_dist_m is not None
            and float(self._last_measured_edge_dist_m) <= near_dist_m
        )
        smoothed_center_x = None
        table_center = local.get("table_center_norm")
        depth_hw = self._shape_hw(depth_shape)
        if dynamic_enable and depth_hw is not None and isinstance(table_center, (list, tuple)) and table_center:
            try:
                raw_center_x = float(table_center[0]) * float(depth_hw[1])
                alpha = max(0.0, min(1.0, float(getattr(table_edge_cfg, "yolo_table_roi_ema_alpha", 0.4) or 0.4)))
                if self._yolo_table_roi_center_x_ema is None:
                    self._yolo_table_roi_center_x_ema = raw_center_x
                else:
                    self._yolo_table_roi_center_x_ema = (1.0 - alpha) * self._yolo_table_roi_center_x_ema + alpha * raw_center_x
                smoothed_center_x = self._yolo_table_roi_center_x_ema
            except Exception:
                smoothed_center_x = None
        last_age_s = None
        if self._last_valid_quadrant_ts:
            last_age_s = time.time() - float(self._last_valid_quadrant_ts or 0.0)
        roi_meta = choose_depth_roi(
            local,
            local.get("rgb_shape"),
            depth_shape,
            fallback,
            last_valid_table_bbox=self._last_valid_table_bbox,
            last_valid_table_center_norm=self._last_valid_table_center_norm,
            last_valid_quadrant=self._last_valid_quadrant,
            last_valid_age_s=last_age_s,
            last_valid_ttl_s=self._last_valid_quadrant_ttl_s,
            manual_static=manual_static,
            roi_preset=roi_preset,
            yolo_dynamic_enable=dynamic_enable,
            yolo_table_class_id=int(getattr(table_edge_cfg, "yolo_table_class_id", 0) or 0),
            yolo_table_conf_min=float(getattr(table_edge_cfg, "yolo_table_conf_min", self._yolo_table_min_conf) or self._yolo_table_min_conf),
            yolo_min_area_ratio=float(getattr(table_edge_cfg, "yolo_table_roi_min_area_ratio", 0.01) or 0.01),
            yolo_max_area_ratio=float(getattr(table_edge_cfg, "yolo_table_roi_max_area_ratio", 0.90) or 0.90),
            smoothed_table_center_x=smoothed_center_x,
            near_distance=near_distance,
            yolo_near_bottom_norm=float(getattr(table_edge_cfg, "yolo_table_near_bottom_norm", 0.60) or 0.60),
            edge_stable=edge_stable,
            rgb_native_shape=local.get("rgb_native_shape"),
            rgb_crop_rect=local.get("rgb_crop_rect"),
            yolo_roi_anchor=str(getattr(table_edge_cfg, "yolo_table_roi_anchor", "center") or "center"),
            yolo_roi_lower_ratio=float(getattr(table_edge_cfg, "yolo_table_roi_lower_ratio", 0.75) or 0.75),
            yolo_roi_use_rgb_depth_mapping=bool(getattr(table_edge_cfg, "yolo_table_roi_use_rgb_depth_mapping", True)),
        )
        quadrant = roi_meta.get("table_quadrant")
        table_bbox = roi_meta.get("table_bbox")
        if quadrant and roi_meta.get("roi_source") in {"local_perception_table_bbox", "yolo_table_bbox", "yolo_table_mapped_center"}:
            self._last_valid_quadrant = str(quadrant).strip().upper()
            self._last_valid_quadrant_ts = time.time()
            self._last_valid_table_bbox = table_bbox
            self._last_valid_table_center_norm = roi_meta.get("table_center_norm")
        roi_meta["yolo_table_roi_enable"] = bool(dynamic_enable)
        roi_meta["yolo_table_roi_ema_alpha"] = float(getattr(table_edge_cfg, "yolo_table_roi_ema_alpha", 0.4) or 0.4)
        roi_meta["yolo_table_roi_anchor"] = str(getattr(table_edge_cfg, "yolo_table_roi_anchor", "center") or "center")
        roi_meta["yolo_table_roi_lower_ratio"] = float(getattr(table_edge_cfg, "yolo_table_roi_lower_ratio", 0.75) or 0.75)
        roi_meta["yolo_table_roi_use_rgb_depth_mapping"] = bool(getattr(table_edge_cfg, "yolo_table_roi_use_rgb_depth_mapping", True))
        roi_meta["yolo_table_roi_center_x_ema"] = self._yolo_table_roi_center_x_ema
        roi_meta["yolo_table_edge_stable_count"] = int(self._edge_stable_count)
        roi_meta["yolo_table_edge_stable_required"] = int(stable_required)
        roi_meta["yolo_table_near_distance"] = bool(near_distance)
        roi_meta["yolo_table_near_dist_m"] = float(near_dist_m)
        roi_meta["yolo_table_measured_edge_dist_m"] = self._last_measured_edge_dist_m
        return roi_meta

    def _roi_payload(self, roi_box=None, roi_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg = self._detector_cfg
        if roi_box is None:
            roi_box = (roi_meta or {}).get("depth_edge_roi") or self._static_roi()
        roi = [int(v) for v in roi_box] if roi_box is not None else None
        meta = dict(roi_meta or {})
        payload: Dict[str, Any] = {
            "table_bbox": meta.get("table_bbox"),
            "table_center_norm": meta.get("table_center_norm"),
            "table_quadrant": meta.get("table_quadrant"),
            "rgb_search_roi": meta.get("rgb_search_roi"),
            "depth_edge_roi": roi,
            "table_edge_roi": roi,
            "edge_roi": roi,
            "roi_source": meta.get("roi_source") or "static_fallback",
            "roi_reason": meta.get("roi_reason") or "",
            "roi_preset": meta.get("roi_preset"),
            "roi_format": "xyxy",
        }
        for key in (
            "locked_edge_id",
            "locked_edge_line",
            "locked_roi",
            "locked_yaw_err",
            "locked_dist_err",
            "locked_edge_conf",
            "locked_obs_seq",
        ):
            if key in meta:
                payload[key] = meta.get(key)
        for key in (
            "dynamic_roi",
            "bbox_valid",
            "bbox_reject_reason",
            "yolo_bbox_area_ratio",
            "yolo_table_conf",
            "yolo_table_class_id",
            "yolo_bbox_center_x",
            "yolo_bbox_center_x_norm",
            "yolo_roi_center_x",
            "yolo_roi_center_x_norm",
            "table_bbox_touch_left",
            "table_bbox_touch_right",
            "table_bbox_touch_bottom",
            "table_bbox_boundary_allowed",
            "yolo_table_roi_valid",
            "roi_phase",
            "yolo_table_roi_enable",
            "yolo_table_roi_ema_alpha",
            "yolo_table_roi_anchor",
            "yolo_table_roi_lower_ratio",
            "yolo_table_roi_use_rgb_depth_mapping",
            "yolo_table_roi_center_x_ema",
            "yolo_table_edge_stable_count",
            "yolo_table_edge_stable_required",
            "yolo_table_near_distance",
            "yolo_table_near_dist_m",
            "yolo_table_measured_edge_dist_m",
            "bbox_y1_norm",
            "bbox_y2_norm",
            "bbox_bottom_norm",
            "bbox_touch_bottom",
            "roi_center_y_from_yolo",
            "roi_y_strategy",
            "rgb_shape",
            "depth_shape",
            "rgb_native_shape",
            "rgb_crop_rect",
            "table_bbox_rgb_xyxy",
            "table_bbox_rgb_center",
            "table_bbox_rgb_center_norm",
            "mapped_depth_center",
            "roi_anchor",
            "roi_mapping_mode",
            "roi_clamped",
            "yolo_bbox_center_y",
            "yolo_bbox_center_y_norm",
            "edge_stability",
            "distance_hint",
        ):
            if key in meta:
                payload[key] = meta.get(key)
        if cfg is not None:
            payload.update(
                {
                    "depth_z_min_m": float(getattr(cfg, "z_min", 0.0) or 0.0),
                    "depth_z_max_m": float(getattr(cfg, "z_max", 0.0) or 0.0),
                    "table_y_min_m": float(getattr(cfg, "table_y_min", 0.0) or 0.0),
                    "table_y_max_m": float(getattr(cfg, "table_y_max", 0.0) or 0.0),
                }
            )
        return payload

    def _process_depth(self, depth_frame: np.ndarray, frame_seq: int) -> Dict[str, Any]:
        if self._detector_mode == "fast_plane_only":
            return self._process_depth_fast_plane_only(depth_frame, frame_seq)
        if self._detector_mode == "lightweight":
            return self._process_depth_lightweight(depth_frame, frame_seq)
        total_start = time.perf_counter()
        profile = self._profile_template()
        profile["depth_frame_fetch_ms"] = float(self._last_depth_frame_fetch_ms)
        frame_prepare_start = time.perf_counter()
        roi_meta = self._select_roi(depth_frame)
        yolo_gate = self._yolo_table_confirmation()
        profile["frame_prepare_ms"] = self._ms_since(frame_prepare_start)
        if not bool(yolo_gate.get("yolo_gate_open", yolo_gate.get("table_confirmed_by_yolo", False))):
            payload = self._default_result(
                depth_valid=True,
                reason="waiting_yolo_table_confirm",
                frame_seq=frame_seq,
                roi_meta=roi_meta,
            )
            payload.update(yolo_gate)
            profile["total_edge_process_ms"] = self._ms_since(total_start)
            return self._attach_profile(payload, profile, path="yolo_gate_wait")
        roi_override = roi_meta.get("depth_edge_roi") if roi_meta.get("roi_source") != "static_fallback" else None
        if self._detector is None:
            payload = self._default_result(
                depth_valid=True,
                reason=self._detector_error or "detector_unavailable",
                frame_seq=frame_seq,
                roi_meta=roi_meta,
            )
            profile["total_edge_process_ms"] = self._ms_since(total_start)
            return self._attach_profile(payload, profile, path="full_unavailable")
        try:
            detect_start = time.perf_counter()
            result, _debug = self._detector.process_depth(depth_frame, roi_override=roi_override)
            profile["plane_or_edge_fit_ms"] = self._ms_since(detect_start)
            if isinstance(_debug, dict) and isinstance(_debug.get("timing"), dict):
                for key, value in _debug.get("timing", {}).items():
                    if key in profile and isinstance(value, (int, float)):
                        profile[key] = float(value)
                profile["roi_crop_ms"] = float(profile.get("roi_extract_ms", 0.0) or 0.0)
                profile["depth_preprocess_ms"] = float(profile.get("roi_extract_ms", 0.0) or 0.0)
        except Exception as exc:
            self.log.debug("table edge detect failed | error=%s", exc)
            payload = self._default_result(depth_valid=True, reason=f"detect_failed:{exc}", frame_seq=frame_seq, roi_meta=roi_meta)
            profile["total_edge_process_ms"] = self._ms_since(total_start)
            return self._attach_profile(payload, profile, path="full_detect_failed")
        obs_build_start = time.perf_counter()
        roi_box = None
        front_plane = None
        if isinstance(_debug, dict):
            roi_box = _debug.get("roi_box")
            front_plane = _debug.get("front_plane") if isinstance(_debug.get("front_plane"), dict) else None
        roi_payload = self._roi_payload(roi_box, roi_meta)
        plane_bbox = None
        if isinstance(front_plane, dict) and bool(front_plane.get("found", False)):
            try:
                ix0 = front_plane.get("image_x_min")
                ix1 = front_plane.get("image_x_max")
                iy0 = front_plane.get("image_y_min")
                iy1 = front_plane.get("image_y_max")
                if ix0 is not None and ix1 is not None and iy0 is not None and iy1 is not None:
                    plane_bbox = [int(ix0), int(iy0), int(ix1) + 1, int(iy1) + 1]
            except Exception:
                plane_bbox = None
        plane_view = self._plane_view_from_bbox(
            plane_bbox,
            getattr(depth_frame, "shape", None),
            area_ratio=front_plane.get("area_ratio") if isinstance(front_plane, dict) else None,
        )
        table_points = int(getattr(result, "table_point_count", 0) or 0)
        all_points = int(getattr(result, "point_count", 0) or 0)
        edge_found = bool(getattr(result, "edge_found", False))
        reason = ""
        if not edge_found:
            reason = "roi_empty" if all_points <= 0 and table_points <= 0 else "no_valid_edge"
        valid_for_control = bool(getattr(result, "valid_for_control", edge_found))
        reject_reason = getattr(result, "reject_reason", "") or reason
        yaw_err = float(getattr(result, "yaw_err_rad", 0.0)) if edge_found else None
        dist_err = float(getattr(result, "dist_err_m", 0.0)) if edge_found else None
        edge_conf = float(getattr(result, "edge_confidence", 0.0) or 0.0)
        payload = {
            "table_found": bool(table_points > 0),
            "edge_found": edge_found,
            "edge_detected": bool(edge_found),
            "edge_geometry_valid": bool(edge_found),
            "edge_valid": bool(edge_found),
            "valid_for_control": valid_for_control,
            "confidence": edge_conf,
            "edge_conf": edge_conf,
            "yaw_err_rad": yaw_err,
            "yaw_err": yaw_err,
            "dist_err_m": dist_err,
            "dist_err": dist_err,
            "edge_k": getattr(result, "line_k", None),
            "edge_b": getattr(result, "line_b", None),
            "image_line_k": getattr(result, "image_line_k", None),
            "image_line_b": getattr(result, "image_line_b", None),
            "depth_valid": True,
            "edge_obs_unavailable": False,
            "point_count": all_points,
            "valid_edge_points": all_points,
            "table_point_count": table_points,
            "edge_inlier_count": int(getattr(result, "inlier_count", 0) or 0),
            "selected_edge": edge_found,
            "near_edge": valid_for_control,
            **plane_view,
            "view_err_norm": plane_view.get("plane_cx_norm") if edge_found else yolo_gate.get("table_cx_norm"),
            "view_source": "plane" if edge_found else ("yolo" if yolo_gate.get("yolo_reliable") else "none"),
            "view_reliable": bool(
                (edge_found and plane_view.get("plane_cx_norm") is not None)
                or yolo_gate.get("yolo_reliable", False)
            ),
            "fov_guard_active": False,
            "frame_id": int(self._frame_id),
            "frame_seq": int(frame_seq),
            "source": "vision_table_edge_manager",
            "reason": reject_reason,
            "reject_reason": reject_reason,
            "target_dist_m": float(self._target_dist_m),
            "plane_only_mode": bool(getattr(self._detector_cfg, "plane_only_mode", False)),
            "enable_crease_line": bool(getattr(self._detector_cfg, "enable_crease_line", True)),
            **self._detector_mode_payload(),
            **yolo_gate,
            **roi_payload,
            **self._plane_debug_payload(_debug, roi_payload.get("edge_roi") or roi_box),
            "type": "table_edge_obs",
        }
        for key in (
            "raw_found",
            "pose_found",
            "pose_source",
            "plane_found",
            "line_found",
            "plane_confidence",
            "line_confidence",
            "plane_residual_mean",
            "line_residual_mean",
            "plane_x_span_m",
            "line_x_span_m",
            "candidate_count",
            "inlier_count",
            "stable_count",
            "front_face_area_ratio",
            "plane_yaw_err_rad",
            "plane_dist_err_m",
            "line_yaw_err_rad",
            "line_dist_err_m",
            "plane_k",
            "plane_b",
            "upper_line_found",
            "upper_line_confidence",
            "upper_line_candidate_count",
            "upper_line_inlier_count",
            "upper_line_residual_mean",
            "upper_line_x_span_m",
            "upper_line_y_norm_mean",
            "upper_line_k",
            "upper_line_b",
            "upper_line_yaw_err_rad",
            "upper_line_dist_err_m",
            "lower_line_found",
            "lower_line_confidence",
            "lower_line_candidate_count",
            "lower_line_inlier_count",
            "lower_line_residual_mean",
            "lower_line_x_span_m",
            "lower_line_y_norm_mean",
            "lower_line_k",
            "lower_line_b",
            "lower_line_yaw_err_rad",
            "lower_line_dist_err_m",
            "selected_line_type",
            "table_geometry_score",
            "front_plane_score",
            "line_score",
            "plane_line_consistency_score",
            "roi_boundary_score",
            "temporal_score",
            "geometry_reject_reason",
            "usable_for_approach",
            "usable_for_alignment",
            "usable_for_stop",
            "control_level",
            "control_reject_reason",
            "selected_line_plane_boundary_dist",
            "selected_line_plane_consistency",
            "line_reject_reason",
            "line_drift_rejected",
            "object_like_line_score",
            "final_pose_source",
        ):
            payload[key] = getattr(result, key, None)
        profile["obs_build_ms"] = float(profile.get("obs_build_ms", 0.0) or 0.0) + self._ms_since(obs_build_start)
        profile["total_edge_process_ms"] = self._ms_since(total_start)
        return self._attach_profile(payload, profile, path="full")

    def _process_depth_fast_plane_only(self, depth_frame: np.ndarray, frame_seq: int) -> Dict[str, Any]:
        total_start = time.perf_counter()
        profile = self._profile_template()
        profile["depth_frame_fetch_ms"] = float(self._last_depth_frame_fetch_ms)
        cfg = self._detector_cfg
        debug_pixels_enabled = self._fast_debug_pixels_enabled()
        debug_cap = self._fast_debug_pixel_cap()
        edge_cue = self._empty_fast_edge_cue()
        edge_debug_payload = self._fast_edge_debug_payload(
            edge_cue,
            debug_pixels_enabled=debug_pixels_enabled,
            debug_cap=debug_cap,
            skipped=True,
            skip_reason="not_run_yet",
        )

        frame_prepare_start = time.perf_counter()
        roi_meta = self._select_roi(depth_frame)
        yolo_gate = self._yolo_table_confirmation()
        profile["frame_prepare_ms"] = self._ms_since(frame_prepare_start)
        if not bool(yolo_gate.get("yolo_gate_open", yolo_gate.get("table_confirmed_by_yolo", False))):
            payload = self._default_result(
                depth_valid=True,
                reason="waiting_yolo_table_confirm",
                frame_seq=frame_seq,
                roi_meta=roi_meta,
            )
            payload.update(yolo_gate)
            payload.update(self._detector_mode_payload())
            payload.update({
                "fast_fit_attempted": False,
                "fast_gate_reject_reason": "waiting_yolo_table_confirm",
                "fast_gate_reason": "waiting_yolo_table_confirm",
                "fast_raw_reject_reason": "waiting_yolo_table_confirm",
            })
            profile["total_edge_process_ms"] = self._ms_since(total_start)
            return self._attach_profile(payload, profile, path="fast_plane_only_yolo_gate_wait")

        roi_start = time.perf_counter()
        roi_box = roi_meta.get("depth_edge_roi") if roi_meta.get("roi_source") != "static_fallback" else self._static_roi()
        if roi_box is None:
            roi_box = self._static_roi()
        try:
            roi_box = self._detector._resolve_roi(depth_frame, roi_override=roi_box) if self._detector is not None else tuple(int(v) for v in roi_box)
        except Exception:
            roi_box = tuple(int(v) for v in self._static_roi() or (0, 0, depth_frame.shape[1], depth_frame.shape[0]))
        x0, y0, x1, y1 = [int(v) for v in roi_box]
        stride = max(1, int(self._fast_plane_stride))
        depth_roi = depth_frame[y0:y1:stride, x0:x1:stride]
        profile["fast_roi_extract_ms"] = self._ms_since(roi_start)
        profile["roi_extract_ms"] = float(profile["fast_roi_extract_ms"])
        profile["roi_crop_ms"] = float(profile["fast_roi_extract_ms"])
        roi_payload = self._roi_payload(roi_box, roi_meta)
        fast_debug_base: Dict[str, Any] = {
            "fast_debug_pixels_enabled": bool(debug_pixels_enabled),
            "fast_debug_pixel_cap": int(debug_cap),
            "fast_fit_attempted": False,
            "fast_raw_sampled_point_count": 0,
            "fast_raw_candidate_count": 0,
            "fast_raw_inlier_count": 0,
            "fast_raw_confidence": 0.0,
            "fast_score_inlier": 0.0,
            "fast_score_abs_inlier": 0.0,
            "fast_score_inlier_ratio": 0.0,
            "fast_score_evidence": 0.0,
            "fast_score_residual": 0.0,
            "fast_score_span": 0.0,
            "fast_score_area": 0.0,
            "fast_score_coverage": 0.0,
            "fast_score_geometry": 0.0,
            "fast_score_temporal": 0.0,
            "fast_score_temporal_available": False,
            "fast_score_final": 0.0,
            "fast_confidence_version": "v3",
            "fast_distance_stage": "unknown",
            "fast_control_level": "none",
            "fast_rep_cluster_count": 0,
            "fast_selected_cluster_index": None,
            "fast_selected_cluster_y_center": None,
            "fast_selected_cluster_x_span_m": 0.0,
            "fast_selected_cluster_support": 0,
            "fast_selected_cluster_score": 0.0,
            "fast_background_rep_count": 0,
            "fast_front_rep_count": 0,
            "fast_line_source": "none",
            "fast_line_score": 0.0,
            "fast_frontness_score": 0.0,
            "fast_edge_consistency_score": 0.0,
            "fast_background_penalty": 0.0,
            "fast_edge_candidate_count": 0,
            "fast_edge_inlier_count": 0,
            "fast_edge_x_span_m": 0.0,
            "fast_edge_y_median_px": None,
            "fast_edge_residual": 0.0,
            "fast_edge_line_yaw_rad": None,
            "fast_edge_line_dist_m": None,
            "fast_edge_support_score": 0.0,
            "fast_front_edge_skipped": True,
            "fast_front_edge_skip_reason": "not_run_yet",
            "fast_local_band_support_count": 0,
            "fast_local_band_x_span_m": 0.0,
            "fast_local_band_edge_support": 0,
            "fast_local_band_residual_mean": 0.0,
            "fast_local_band_skipped": True,
            "fast_local_band_skip_reason": "not_run_yet",
            "fast_background_blocked": False,
            "fast_near_stage_far_jump": False,
            "fast_selected_dist_source": "none",
            "fast_prev_dist_used": None,
        }

        point_start = time.perf_counter()
        if depth_roi.size <= 0:
            payload = self._default_result(depth_valid=False, reason="roi_empty", frame_seq=frame_seq, roi_meta=roi_meta)
            payload.update(roi_payload)
            payload.update(self._detector_mode_payload())
            payload.update(fast_debug_base)
            payload.update({"fast_gate_reject_reason": "roi_empty", "fast_gate_reason": "roi_empty", "fast_raw_reject_reason": "roi_empty"})
            profile["point_build_ms"] = self._ms_since(point_start)
            profile["total_edge_process_ms"] = self._ms_since(total_start)
            return self._attach_profile(payload, profile, path="fast_plane_only_roi_empty")
        depth_m = depth_roi.astype(np.float32, copy=False)
        if depth_roi.dtype != np.float32:
            scale = float(getattr(self._detector.calib, "depth_scale", 0.001) if self._detector is not None else 0.001)
            depth_m = depth_m * scale
        z_min = float(getattr(cfg, "z_min", 0.2) if cfg is not None else 0.2)
        z_max = float(getattr(cfg, "z_max", 2.0) if cfg is not None else 2.0)
        valid_mask = (depth_m > z_min) & (depth_m < z_max)
        yy, xx = np.nonzero(valid_mask)
        sampled_count = int(depth_roi.size)
        point_count = int(len(xx))
        profile["fast_depth_valid_ms"] = self._ms_since(point_start)
        profile["point_build_ms"] = float(profile["fast_depth_valid_ms"])

        min_all = max(60, int(getattr(cfg, "min_all_points", 1000) if cfg is not None else 1000) // max(1, stride * stride))
        min_table = max(45, int(getattr(cfg, "plane_min_inliers", 220) if cfg is not None else 220) // max(1, stride))
        if self._detector is None or point_count < min_all:
            payload = self._default_result(
                depth_valid=True,
                reason=self._detector_error or "not_enough_points",
                frame_seq=frame_seq,
                roi_meta=roi_meta,
            )
            payload.update(roi_payload)
            payload.update(self._detector_mode_payload())
            payload.update({"sampled_point_count": sampled_count, "point_count": point_count, "candidate_count": 0})
            payload.update(fast_debug_base)
            payload.update(edge_debug_payload)
            payload.update({
                "fast_raw_sampled_point_count": sampled_count,
                "fast_gate_reject_reason": payload.get("reason") or "not_enough_points",
                "fast_gate_reason": payload.get("reason") or "not_enough_points",
                "fast_raw_reject_reason": payload.get("reason") or "not_enough_points",
            })
            profile["total_edge_process_ms"] = self._ms_since(total_start)
            return self._attach_profile(payload, profile, path="fast_plane_only_not_enough_points")

        select_start = time.perf_counter()
        projection_start = time.perf_counter()
        z = depth_m[valid_mask]
        calib = self._detector.calib
        u = (float(x0) + xx.astype(np.float32) * float(stride))
        v = (float(y0) + yy.astype(np.float32) * float(stride))
        x_c = (u - float(calib.cx)) * z / float(calib.fx)
        y_c = (v - float(calib.cy)) * z / float(calib.fy)
        pitch_deg = float(self._camera_pitch_deg)
        camera_height_m = float(self._camera_height_m)
        table_height_m = float(self._table_height_m)
        front_face_z_min = float(self._front_face_z_min_m)
        front_face_z_max = float(self._front_face_z_max_m)
        min_vertical_z_span = float(self._min_vertical_z_span_m)
        min_vertical_support = max(1, int(self._min_vertical_support_points))
        x_bin_width_m = float(self._x_bin_width_m)
        y_cluster_bin_m = float(self._y_cluster_bin_m)
        min_front_face_columns = max(2, int(self._min_front_face_columns))
        min_front_face_x_span = float(self._min_front_face_x_span_m)
        max_yaw_cfg = float(self._max_yaw_abs_rad)
        x_robot, y_robot, z_robot = self._camera_points_to_robot(
            x_c,
            y_c,
            z,
            pitch_deg=pitch_deg,
            camera_height_m=camera_height_m,
        )
        height_filter_start = time.perf_counter()
        robot_z_pct = self._finite_percentiles(z_robot)
        ground_like_count = int(np.sum((z_robot >= -0.04) & (z_robot <= 0.06)))
        table_height_like_count = int(np.sum(np.abs(z_robot - table_height_m) <= 0.06))
        height_mask = (z_robot > front_face_z_min) & (z_robot < front_face_z_max)
        candidate_count = int(height_mask.sum())
        height_x = x_robot[height_mask]
        height_y = y_robot[height_mask]
        height_z = z_robot[height_mask]
        height_px = (x0 + xx[height_mask].astype(np.int32) * int(stride)).astype(np.int32)
        height_py = (y0 + yy[height_mask].astype(np.int32) * int(stride)).astype(np.int32)
        raw_candidate_count = int(candidate_count)
        candidate_cap = max(0, int(self._fast_candidate_point_cap or 0))
        if candidate_cap > 0 and candidate_count > candidate_cap:
            cap_idx = np.linspace(0, candidate_count - 1, candidate_cap).astype(np.int32)
            height_x = height_x[cap_idx]
            height_y = height_y[cap_idx]
            height_z = height_z[cap_idx]
            height_px = height_px[cap_idx]
            height_py = height_py[cap_idx]
            candidate_count = int(candidate_cap)
        profile["fast_candidate_raw_count"] = int(raw_candidate_count)
        profile["fast_candidate_fit_count"] = int(candidate_count)
        profile["fast_candidate_point_cap"] = int(candidate_cap)
        candidate_z_pct = self._finite_percentiles(height_z)
        candidate_x_span = float(np.max(height_x) - np.min(height_x)) if len(height_x) > 1 else 0.0
        profile["fast_height_filter_ms"] = self._ms_since(height_filter_start)
        raw_sample_pixels = self._sparse_pixel_sample(xx, yy, x0, y0, stride, cap=debug_cap) if debug_pixels_enabled else []
        fast_candidate_pixels = self._sparse_pixel_pairs(height_px, height_py, cap=debug_cap) if debug_pixels_enabled else []
        profile["candidate_select_ms"] = self._ms_since(select_start)
        if point_count <= 0:
            payload = self._default_result(depth_valid=True, reason="no_robot_points", frame_seq=frame_seq, roi_meta=roi_meta)
            payload.update(roi_payload)
            payload.update(self._detector_mode_payload())
            payload.update({"sampled_point_count": sampled_count, "point_count": point_count, "candidate_count": 0})
            payload.update(fast_debug_base)
            payload.update({
                "fast_coord_frame": "robot_xyz",
                "fast_camera_pitch_deg": pitch_deg,
                "fast_camera_height_m": camera_height_m,
                "fast_table_height_m": table_height_m,
                "fast_raw_sampled_point_count": sampled_count,
                "fast_gate_reject_reason": "no_robot_points",
                "fast_gate_reason": "no_robot_points",
                "fast_raw_reject_reason": "no_robot_points",
                "fast_sampled_pixels": raw_sample_pixels,
                "fast_candidate_pixels": [],
                "fast_support_pixels": [],
                "fast_front_face_rep_pixels": [],
                "fast_inlier_pixels": [],
                "fast_outlier_pixels": [],
            })
            profile["total_edge_process_ms"] = self._ms_since(total_start)
            return self._attach_profile(payload, profile, path="fast_plane_only_no_robot_points")
        if candidate_count < max(min_vertical_support, min_front_face_columns):
            payload = self._default_result(depth_valid=True, reason="height_filter_empty", frame_seq=frame_seq, roi_meta=roi_meta)
            payload.update(roi_payload)
            payload.update(self._detector_mode_payload())
            payload.update({"sampled_point_count": sampled_count, "point_count": point_count, "candidate_count": candidate_count})
            payload.update(fast_debug_base)
            payload.update(edge_debug_payload)
            payload.update({
                "fast_coord_frame": "robot_xyz",
                "fast_camera_pitch_deg": pitch_deg,
                "fast_camera_height_m": camera_height_m,
                "fast_table_height_m": table_height_m,
                "fast_robot_z_p10": robot_z_pct.get("p10"),
                "fast_robot_z_p50": robot_z_pct.get("p50"),
                "fast_robot_z_p90": robot_z_pct.get("p90"),
                "fast_robot_z_min": float(np.min(z_robot)) if len(z_robot) else None,
                "fast_robot_z_max": float(np.max(z_robot)) if len(z_robot) else None,
                "fast_ground_like_count": ground_like_count,
                "fast_table_height_like_count": table_height_like_count,
                "candidate_robot_z_min": float(np.min(height_z)) if len(height_z) else None,
                "candidate_robot_z_p50": candidate_z_pct.get("p50"),
                "candidate_robot_z_max": float(np.max(height_z)) if len(height_z) else None,
                "fast_raw_sampled_point_count": sampled_count,
                "fast_raw_candidate_count": raw_candidate_count,
                "fast_candidate_point_count": candidate_count,
                "fast_candidate_point_cap": int(candidate_cap),
                "fast_candidate_downsampled": bool(raw_candidate_count > candidate_count),
                "fast_candidate_x_span_m": candidate_x_span,
                "fast_score_area": float(candidate_count) / float(max(1, sampled_count)),
                "fast_score_coverage": float(candidate_count) / float(max(1, sampled_count)),
                "fast_gate_reject_reason": "height_filter_empty",
                "fast_gate_reason": "height_filter_empty",
                "fast_raw_reject_reason": "height_filter_empty",
                "fast_sampled_pixels": raw_sample_pixels,
                "fast_candidate_pixel_count": int(len(fast_candidate_pixels)),
                "fast_support_pixel_count": 0,
                "fast_front_face_rep_pixel_count": 0,
                "fast_inlier_pixel_count": 0,
                "fast_outlier_pixel_count": 0,
                "fast_candidate_pixels": fast_candidate_pixels,
                "fast_support_pixels": [],
                "fast_front_face_rep_pixels": [],
                "fast_inlier_pixels": [],
                "fast_outlier_pixels": [],
            })
            profile["total_edge_process_ms"] = self._ms_since(total_start)
            return self._attach_profile(payload, profile, path="fast_plane_only_height_filter_empty")

        fit_start = time.perf_counter()
        reps = self._select_fast_front_face_representatives(
            x_robot=height_x,
            y_robot=height_y,
            z_robot=height_z,
            px=height_px,
            py=height_py,
            x_bin_width_m=x_bin_width_m,
            y_cluster_bin_m=y_cluster_bin_m,
            min_support_points=min_vertical_support,
            min_z_span_m=min_vertical_z_span,
        )
        profile["fast_rep_select_ms"] = self._ms_since(fit_start)
        rep_count = int(reps.get("count", 0) or 0)
        support_total = int(reps.get("support_total", 0) or 0)
        z_span_arr = np.asarray(reps.get("z_span", []), dtype=np.float32)
        z_span_p50 = float(np.percentile(z_span_arr, 50)) if z_span_arr.size else 0.0
        z_span_max = float(np.max(z_span_arr)) if z_span_arr.size else 0.0
        support_x_pre = np.asarray(reps.get("support_x", []), dtype=np.float32)
        rep_x_pre = np.asarray(reps.get("x", []), dtype=np.float32)
        support_x_span_pre = float(np.max(support_x_pre) - np.min(support_x_pre)) if support_x_pre.size > 1 else 0.0
        rep_x_span_pre = float(np.max(rep_x_pre) - np.min(rep_x_pre)) if rep_x_pre.size > 1 else 0.0
        if rep_count < min_front_face_columns:
            front_edge_start = time.perf_counter()
            edge_cue = self._extract_fast_front_edge_cue(
                depth_m=depth_m,
                x0=x0,
                y0=y0,
                stride=stride,
                calib=calib,
                pitch_deg=pitch_deg,
                camera_height_m=camera_height_m,
                z_min=z_min,
                z_max=z_max,
                target_dist_m=float(self._target_dist_m),
                min_x_span_m=float(min_front_face_x_span),
                max_yaw=float(max_yaw_cfg),
            )
            profile["fast_front_edge_ms"] = self._ms_since(front_edge_start)
            edge_debug_payload = self._fast_edge_debug_payload(
                edge_cue,
                debug_pixels_enabled=debug_pixels_enabled,
                debug_cap=debug_cap,
                skipped=False,
                skip_reason="vertical_support_failed",
            )
            edge_px = np.asarray(edge_cue.get("px", []), dtype=np.int32)
            edge_py = np.asarray(edge_cue.get("py", []), dtype=np.int32)
            edge_inlier_mask = np.asarray(edge_cue.get("inlier", []), dtype=bool)
            if edge_inlier_mask.size == edge_px.size and int(edge_inlier_mask.sum()) > 0:
                edge_draw_px = edge_px[edge_inlier_mask]
                edge_draw_py = edge_py[edge_inlier_mask]
            else:
                edge_draw_px = edge_px
                edge_draw_py = edge_py
            low_reason = "vertical_support_low" if rep_count <= 0 else "front_face_columns_low"
            rep_px_pre = np.asarray(reps.get("px", []), dtype=np.int32)
            rep_py_pre = np.asarray(reps.get("py", []), dtype=np.int32)
            support_px_pre = np.asarray(reps.get("support_px", []), dtype=np.int32)
            support_py_pre = np.asarray(reps.get("support_py", []), dtype=np.int32)
            fast_support_pixels = self._sparse_pixel_pairs(support_px_pre, support_py_pre, cap=debug_cap) if debug_pixels_enabled else []
            fast_rep_pixels = self._sparse_pixel_pairs(rep_px_pre, rep_py_pre, cap=debug_cap) if debug_pixels_enabled else []
            edge_enough = (
                int(edge_cue.get("inlier_count", 0) or 0) >= max(4, min_front_face_columns)
                and float(edge_cue.get("x_span_m", 0.0) or 0.0) >= max(0.15, float(min_front_face_x_span) * 1.5)
                and float(edge_cue.get("score", 0.0) or 0.0) >= 0.45
                and abs(float(edge_cue.get("yaw", 0.0) or 0.0)) <= max_yaw_cfg
            )
            if edge_enough:
                edge_yaw = float(edge_cue.get("yaw", 0.0) or 0.0)
                edge_dist = float(edge_cue.get("dist", 0.0) or 0.0)
                edge_conf = self._clip01(0.35 + 0.55 * float(edge_cue.get("score", 0.0) or 0.0))
                edge_control = "rotate_only" if abs(edge_yaw) >= 0.55 else "approach_slow"
                edge_bbox = None
                if len(edge_draw_px) > 0 and len(edge_draw_py) > 0:
                    edge_bbox = [int(np.min(edge_draw_px)), int(np.min(edge_draw_py)), int(np.max(edge_draw_px)) + stride, int(np.max(edge_draw_py)) + stride]
                edge_view = self._plane_view_from_bbox(edge_bbox or roi_box, getattr(depth_frame, "shape", None), area_ratio=None)
                payload = {
                    "table_found": bool(candidate_count > 0),
                    "edge_found": True,
                    "edge_valid": False,
                    "valid_for_control": False,
                    "confidence": float(edge_conf),
                    "edge_conf": float(edge_conf),
                    "yaw_err_rad": float(edge_yaw),
                    "yaw_err": float(edge_yaw),
                    "dist_err_m": float(edge_dist),
                    "dist_err": float(edge_dist),
                    "edge_k": float(edge_cue.get("k", 0.0) or 0.0),
                    "edge_b": float(edge_cue.get("b", 0.0) or 0.0),
                    "depth_valid": True,
                    "point_count": int(point_count),
                    "sampled_point_count": int(sampled_count),
                    "candidate_count": int(candidate_count),
                    "table_point_count": int(candidate_count),
                    "inlier_count": int(edge_cue.get("inlier_count", 0) or 0),
                    "edge_inlier_count": int(edge_cue.get("inlier_count", 0) or 0),
                    "representative_inlier_count": 0,
                    "support_point_count": 0,
                    "selected_edge": True,
                    "near_edge": False,
                    **edge_view,
                    "plane_found": True,
                    "plane_confidence": float(edge_conf),
                    "plane_residual_mean": float(edge_cue.get("residual_mean", 0.0) or 0.0),
                    "plane_residual_max": float(edge_cue.get("residual_mean", 0.0) or 0.0),
                    "plane_x_span_m": float(edge_cue.get("x_span_m", 0.0) or 0.0),
                    "plane_yaw_err_rad": float(edge_yaw),
                    "plane_dist_err_m": float(edge_dist),
                    "plane_mask_status": "fast_sparse",
                    **fast_debug_base,
                    **edge_debug_payload,
                    "fast_fit_attempted": False,
                    "fast_raw_yaw_err_rad": float(edge_yaw),
                    "fast_raw_dist_err_m": float(edge_dist),
                    "fast_raw_plane_cx_norm": edge_view.get("plane_cx_norm"),
                    "fast_raw_plane_width_norm": edge_view.get("plane_width_norm"),
                    "fast_raw_plane_x_span_m": float(edge_cue.get("x_span_m", 0.0) or 0.0),
                    "fast_raw_residual_mean": float(edge_cue.get("residual_mean", 0.0) or 0.0),
                    "fast_raw_residual_p90": float(edge_cue.get("residual_mean", 0.0) or 0.0),
                    "fast_candidate_point_count": int(candidate_count),
                    "fast_support_point_count": int(support_total),
                    "fast_rep_count": int(rep_count),
                    "fast_rep_inlier_count": 0,
                    "fast_rep_outlier_count": int(rep_count),
                    "fast_candidate_x_span_m": float(candidate_x_span),
                    "fast_support_x_span_m": float(support_x_span_pre),
                    "fast_rep_x_span_m": float(rep_x_span_pre),
                    "fast_fit_inlier_x_span_m": float(edge_cue.get("x_span_m", 0.0) or 0.0),
                    "fast_residual_mean": float(edge_cue.get("residual_mean", 0.0) or 0.0),
                    "fast_residual_p90": float(edge_cue.get("residual_mean", 0.0) or 0.0),
                    "fast_support_mode": "edge",
                    "fast_fit_line_source": "front_edge",
                    "fast_line_source": "edge",
                    "fast_line_score": float(edge_conf),
                    "fast_frontness_score": 1.0,
                    "fast_edge_consistency_score": 1.0,
                    "fast_background_penalty": 0.0,
                    "fast_selected_dist_source": "edge",
                    "fast_raw_confidence": float(edge_conf),
                    "fast_raw_inlier_count": int(edge_cue.get("inlier_count", 0) or 0),
                    "fast_raw_candidate_count": int(raw_candidate_count),
                    "fast_candidate_point_cap": int(candidate_cap),
                    "fast_candidate_downsampled": bool(raw_candidate_count > candidate_count),
                    "fast_raw_sampled_point_count": int(sampled_count),
                    "fast_raw_reject_reason": "front_line_weak",
                    "fast_gate_reject_reason": "front_line_weak",
                    "fast_gate_reason": "front_line_weak",
                    "fast_coord_frame": "robot_xyz",
                    "fast_camera_pitch_deg": float(pitch_deg),
                    "fast_camera_height_m": float(camera_height_m),
                    "fast_table_height_m": float(table_height_m),
                    "fast_distance_stage": "near" if edge_dist <= 0.25 else "middle" if edge_dist <= 0.60 else "far",
                    "fast_control_level": edge_control,
                    "fast_score_final": float(edge_conf),
                    "fast_candidate_pixel_count": int(len(fast_candidate_pixels)),
                    "fast_support_pixel_count": int(len(fast_support_pixels)),
                    "fast_front_face_rep_pixel_count": int(len(fast_rep_pixels)),
                    "fast_inlier_pixel_count": 0,
                    "fast_outlier_pixel_count": int(len(fast_rep_pixels)),
                    "fast_sampled_pixels": raw_sample_pixels,
                    "fast_candidate_pixels": fast_candidate_pixels,
                    "fast_support_pixels": fast_support_pixels,
                    "fast_front_face_rep_pixels": fast_rep_pixels,
                    "fast_inlier_pixels": [],
                    "fast_outlier_pixels": fast_rep_pixels,
                    "fast_background_pixels": [],
                    "fast_rep_background_pixels": [],
                    "fast_weak_pixels": fast_rep_pixels,
                    "pose_source": "fast_plane_only",
                    "final_pose_source": "fast_plane_only",
                    "view_err_norm": edge_view.get("plane_cx_norm"),
                    "view_source": "plane",
                    "view_reliable": True,
                    "frame_id": int(self._frame_id),
                    "frame_seq": int(frame_seq),
                    "source": "vision_table_edge_manager",
                    "reason": "front_line_weak",
                    "reject_reason": "front_line_weak",
                    "target_dist_m": float(self._target_dist_m),
                    "plane_only_mode": True,
                    "enable_crease_line": False,
                    "usable_for_approach": True,
                    "usable_for_alignment": False,
                    "usable_for_stop": False,
                    "control_level": edge_control,
                    "control_reject_reason": "front_line_weak",
                    **self._detector_mode_payload(),
                    **yolo_gate,
                    **roi_payload,
                    "type": "table_edge_obs",
                }
                profile["plane_fit_ms"] = self._ms_since(fit_start)
                profile["total_edge_process_ms"] = self._ms_since(total_start)
                return self._attach_profile(payload, profile, path="fast_plane_only_front_edge_fallback")
            payload = self._default_result(depth_valid=True, reason=low_reason, frame_seq=frame_seq, roi_meta=roi_meta)
            payload.update(roi_payload)
            payload.update(self._detector_mode_payload())
            payload.update({"sampled_point_count": sampled_count, "point_count": point_count, "candidate_count": candidate_count})
            payload.update(fast_debug_base)
            payload.update(edge_debug_payload)
            payload.update({
                "fast_fit_attempted": False,
                "fast_coord_frame": "robot_xyz",
                "fast_camera_pitch_deg": pitch_deg,
                "fast_camera_height_m": camera_height_m,
                "fast_table_height_m": table_height_m,
                "fast_robot_z_p10": robot_z_pct.get("p10"),
                "fast_robot_z_p50": robot_z_pct.get("p50"),
                "fast_robot_z_p90": robot_z_pct.get("p90"),
                "fast_robot_z_min": float(np.min(z_robot)) if len(z_robot) else None,
                "fast_robot_z_max": float(np.max(z_robot)) if len(z_robot) else None,
                "candidate_robot_z_min": float(np.min(height_z)) if len(height_z) else None,
                "candidate_robot_z_p50": candidate_z_pct.get("p50"),
                "candidate_robot_z_max": float(np.max(height_z)) if len(height_z) else None,
                "fast_ground_like_count": ground_like_count,
                "fast_table_height_like_count": table_height_like_count,
                "fast_raw_sampled_point_count": sampled_count,
                "fast_raw_candidate_count": raw_candidate_count,
                "fast_candidate_point_count": candidate_count,
                "fast_candidate_point_cap": int(candidate_cap),
                "fast_candidate_downsampled": bool(raw_candidate_count > candidate_count),
                "fast_support_point_count": support_total,
                "fast_rep_count": rep_count,
                "fast_rep_inlier_count": 0,
                "fast_rep_outlier_count": rep_count,
                "fast_candidate_x_span_m": candidate_x_span,
                "fast_support_x_span_m": support_x_span_pre,
                "fast_rep_x_span_m": rep_x_span_pre,
                "fast_fit_inlier_x_span_m": 0.0,
                "fast_support_mode": "none",
                "fast_fit_line_source": "none",
                "fast_front_face_rep_count": rep_count,
                "fast_front_face_support_point_count": support_total,
                "fast_z_span_m_p50": z_span_p50,
                "fast_z_span_m_max": z_span_max,
                "fast_gate_reject_reason": low_reason,
                "fast_gate_reason": low_reason,
                "fast_raw_reject_reason": low_reason,
                "fast_sampled_pixels": raw_sample_pixels,
                "fast_candidate_pixel_count": int(len(fast_candidate_pixels)),
                "fast_support_pixel_count": int(len(fast_support_pixels)),
                "fast_front_face_rep_pixel_count": int(len(fast_rep_pixels)),
                "fast_inlier_pixel_count": 0,
                "fast_outlier_pixel_count": int(len(fast_rep_pixels)),
                "fast_candidate_pixels": fast_candidate_pixels,
                "fast_support_pixels": fast_support_pixels,
                "fast_front_face_rep_pixels": fast_rep_pixels,
                "fast_inlier_pixels": [],
                "fast_outlier_pixels": fast_rep_pixels,
                "fast_background_pixels": [],
                "fast_rep_background_pixels": [],
                "fast_weak_pixels": fast_rep_pixels,
            })
            profile["plane_fit_ms"] = self._ms_since(fit_start)
            profile["total_edge_process_ms"] = self._ms_since(total_start)
            return self._attach_profile(payload, profile, path=f"fast_plane_only_{low_reason}")
        x_t = np.asarray(reps.get("x"), dtype=np.float32)
        y_t = np.asarray(reps.get("y"), dtype=np.float32)
        rep_support = np.asarray(reps.get("support"), dtype=np.int32)
        rep_z_span = np.asarray(reps.get("z_span", []), dtype=np.float32)
        support_x = np.asarray(reps.get("support_x", []), dtype=np.float32)
        support_x_span = float(np.max(support_x) - np.min(support_x)) if support_x.size > 1 else 0.0
        rep_x_span = float(np.max(x_t) - np.min(x_t)) if x_t.size > 1 else 0.0
        residual_threshold = max(0.035, float(getattr(cfg, "front_plane_max_residual_m", 0.035) if cfg is not None else 0.035) * 1.5)
        front_cluster_gap_m = float(self._front_cluster_gap_m)
        front_cluster_start = time.perf_counter()
        try:
            cluster_fit = self._fit_fast_front_cluster_line(
                x_t=x_t,
                y_t=y_t,
                rep_support=rep_support,
                rep_z_span=rep_z_span,
                min_front_face_columns=min_front_face_columns,
                min_front_face_x_span=min_front_face_x_span,
                min_vertical_support=min_vertical_support,
                min_vertical_z_span=min_vertical_z_span,
                residual_threshold=residual_threshold,
                max_yaw=max_yaw_cfg,
                x_bin_width_m=x_bin_width_m,
                front_cluster_gap_m=front_cluster_gap_m,
                target_dist_m=float(self._target_dist_m),
            )
        except Exception as exc:
            cluster_fit = {"selected": None, "clusters": [], "reject_reason": f"birdview_fit_failed:{exc}"}
        profile["fast_front_cluster_fit_ms"] = self._ms_since(front_cluster_start)
        clusters = list(cluster_fit.get("clusters") or [])
        selected_cluster = cluster_fit.get("selected")
        rep_px = np.asarray(reps.get("px"), dtype=np.int32)
        rep_py = np.asarray(reps.get("py"), dtype=np.int32)
        support_px_all = np.asarray(reps.get("support_px", []), dtype=np.int32)
        support_py_all = np.asarray(reps.get("support_py", []), dtype=np.int32)
        support_rep_index = np.asarray(reps.get("support_rep_index", []), dtype=np.int32)

        if selected_cluster is None:
            cluster_reason = str(cluster_fit.get("reject_reason") or "no_front_cluster")
            if cluster_reason.startswith("birdview_fit_failed:"):
                low_reason = cluster_reason
            elif clusters:
                low_reason = "front_cluster_weak" if cluster_reason in {"front_cluster_weak", "front_face_columns_low", "vertical_support_low", "front_face_x_span_low"} else cluster_reason
            else:
                low_reason = "no_front_cluster"
            fast_support_pixels = self._sparse_pixel_pairs(support_px_all, support_py_all, cap=debug_cap) if debug_pixels_enabled else []
            fast_rep_pixels = self._sparse_pixel_pairs(rep_px, rep_py, cap=debug_cap) if debug_pixels_enabled else []
            payload = self._default_result(depth_valid=True, reason=low_reason, frame_seq=frame_seq, roi_meta=roi_meta)
            payload.update(roi_payload)
            payload.update(self._detector_mode_payload())
            payload.update({"sampled_point_count": sampled_count, "point_count": point_count, "candidate_count": candidate_count})
            payload.update(fast_debug_base)
            payload.update(edge_debug_payload)
            payload.update({
                "fast_fit_attempted": bool(rep_count > 0),
                "fast_coord_frame": "robot_xyz",
                "fast_camera_pitch_deg": pitch_deg,
                "fast_camera_height_m": camera_height_m,
                "fast_table_height_m": table_height_m,
                "fast_raw_sampled_point_count": sampled_count,
                "fast_raw_candidate_count": raw_candidate_count,
                "fast_candidate_point_count": candidate_count,
                "fast_candidate_point_cap": int(candidate_cap),
                "fast_candidate_downsampled": bool(raw_candidate_count > candidate_count),
                "fast_support_point_count": 0,
                "fast_rep_count": rep_count,
                "fast_rep_inlier_count": 0,
                "fast_rep_outlier_count": 0,
                "fast_background_rep_count": rep_count,
                "fast_front_rep_count": 0,
                "fast_rep_cluster_count": int(len(clusters)),
                "fast_selected_cluster_index": None,
                "fast_selected_cluster_y_center": None,
                "fast_selected_cluster_x_span_m": 0.0,
                "fast_selected_cluster_support": 0,
                "fast_selected_cluster_score": 0.0,
                "fast_candidate_x_span_m": candidate_x_span,
                "fast_support_x_span_m": support_x_span,
                "fast_rep_x_span_m": rep_x_span,
                "fast_fit_inlier_x_span_m": 0.0,
                "fast_support_mode": "none",
                "fast_fit_line_source": "front_cluster",
                "fast_front_face_rep_count": rep_count,
                "fast_front_face_support_point_count": support_total,
                "fast_gate_reject_reason": low_reason,
                "fast_gate_reason": low_reason,
                "fast_raw_reject_reason": low_reason,
                "fast_sampled_pixels": raw_sample_pixels,
                "fast_candidate_pixel_count": int(len(fast_candidate_pixels)),
                "fast_support_pixel_count": int(len(fast_support_pixels)),
                "fast_front_face_rep_pixel_count": int(len(fast_rep_pixels)),
                "fast_inlier_pixel_count": 0,
                "fast_outlier_pixel_count": 0,
                "fast_background_pixel_count": int(len(fast_rep_pixels)),
                "fast_candidate_pixels": fast_candidate_pixels,
                "fast_support_pixels": fast_support_pixels,
                "fast_front_face_rep_pixels": fast_rep_pixels,
                "fast_inlier_pixels": [],
                "fast_outlier_pixels": [],
                "fast_background_pixels": fast_rep_pixels,
                "fast_rep_background_pixels": fast_rep_pixels,
                "fast_weak_pixels": fast_rep_pixels,
            })
            profile["plane_fit_ms"] = self._ms_since(fit_start)
            profile["total_edge_process_ms"] = self._ms_since(total_start)
            return self._attach_profile(payload, profile, path=f"fast_plane_only_{low_reason.split(':', 1)[0]}")

        selected_cluster_index = int(selected_cluster.get("index", 0))
        selected_rep_indices = np.asarray(selected_cluster.get("rep_indices", []), dtype=np.int32)
        selected_mask = np.zeros(rep_count, dtype=bool)
        selected_mask[selected_rep_indices] = True
        local_inlier = np.asarray(selected_cluster.get("inlier_local", []), dtype=bool)
        inlier = np.zeros(rep_count, dtype=bool)
        if local_inlier.size == selected_rep_indices.size:
            inlier[selected_rep_indices] = local_inlier
        outlier = selected_mask & ~inlier
        background_mask = np.zeros(rep_count, dtype=bool)
        weak_mask = np.zeros(rep_count, dtype=bool)
        for info in clusters:
            idx = np.asarray(info.get("rep_indices", []), dtype=np.int32)
            if idx.size <= 0 or int(info.get("index", -1)) == selected_cluster_index:
                continue
            if int(info.get("index", -1)) > selected_cluster_index:
                background_mask[idx] = True
            else:
                weak_mask[idx] = True
        k = float(selected_cluster.get("k", 0.0))
        b = float(selected_cluster.get("b", 0.0))
        residual = np.full(rep_count, np.nan, dtype=np.float32)
        selected_residual = np.asarray(selected_cluster.get("residual", []), dtype=np.float32)
        if selected_residual.size == selected_rep_indices.size:
            residual[selected_rep_indices] = selected_residual
        residual_mean = float(selected_cluster.get("residual_mean", 0.0) or 0.0)
        residual_p90 = float(selected_cluster.get("residual_p90", 0.0) or 0.0)
        residual_max = float(np.nanmax(residual[inlier])) if int(inlier.sum()) > 0 else float(np.nanmax(selected_residual)) if selected_residual.size else 0.0
        representative_inlier_count = int(inlier.sum())
        inlier_count = representative_inlier_count
        representative_outlier_count = int(outlier.sum())
        support_inlier_count = int(np.sum(rep_support[inlier])) if representative_inlier_count > 0 and rep_support.size else 0
        selected_cluster_support = int(selected_cluster.get("support_sum", support_inlier_count) or 0)
        selected_cluster_x_span = float(selected_cluster.get("x_span_m", 0.0) or 0.0)
        selected_cluster_y_center = selected_cluster.get("y_center")
        selected_cluster_score = float(selected_cluster.get("score", 0.0) or 0.0)
        background_rep_count = int(background_mask.sum())
        front_rep_count = int(selected_mask.sum())
        selected_support_mask = np.isin(support_rep_index, selected_rep_indices) if support_rep_index.size else np.asarray([], dtype=bool)
        selected_support_x = support_x[selected_support_mask] if support_x.size and selected_support_mask.size == support_x.size else np.asarray([], dtype=np.float32)
        selected_support_x_span = float(np.max(selected_support_x) - np.min(selected_support_x)) if selected_support_x.size > 1 else selected_cluster_x_span
        profile["plane_fit_ms"] = self._ms_since(fit_start)
        profile["plane_or_edge_fit_ms"] = float(profile["plane_fit_ms"])

        eval_start = time.perf_counter()
        inlier_x = x_t[inlier] if inlier_count > 0 else np.asarray([], dtype=np.float32)
        x_span = float(np.max(inlier_x) - np.min(inlier_x)) if inlier_x.size > 1 else 0.0
        yaw_err = math.atan(float(k))
        dist_err = float(b) - float(self._target_dist_m)
        inlier_ratio = float(representative_inlier_count) / float(max(1, rep_count))
        support_ratio = float(support_inlier_count) / float(max(1, selected_cluster_support))
        span_min = float(min_front_face_x_span)
        max_yaw = float(max_yaw_cfg)
        residual_score = max(0.0, 1.0 - residual_mean / max(1e-6, residual_threshold))
        span_score = self._clip01(x_span / max(1e-6, span_min * 1.8))
        area_score = float(candidate_count) / float(max(1, sampled_count))
        accepted_column_score = self._clip01(float(front_rep_count) / float(max(1, min_front_face_columns * 2)))
        vertical_support_score = self._clip01(float(selected_cluster_support) / float(max(1, min_front_face_columns * min_vertical_support * 2)))
        z_span_score = self._clip01(z_span_p50 / max(1e-6, min_vertical_z_span * 1.5))
        x_span_score = self._clip01(x_span / max(1e-6, min_front_face_x_span * 1.5))
        residual_p90 = float(np.percentile(residual[inlier], 90)) if inlier_count > 0 else float(residual_p90)
        if representative_inlier_count >= 8 and x_span >= 0.30:
            support_mode = "vertical"
        elif representative_inlier_count >= 4 and x_span >= 0.15:
            support_mode = "partial"
        elif representative_inlier_count > 0:
            support_mode = "edge"
        else:
            support_mode = "none"
        profile["residual_eval_ms"] = self._ms_since(eval_start)

        obs_start = time.perf_counter()
        plane_bbox = None
        rep_px = np.asarray(reps.get("px"), dtype=np.int32)
        rep_py = np.asarray(reps.get("py"), dtype=np.int32)
        rep_inlier_px = rep_px[inlier] if representative_inlier_count > 0 else np.asarray([], dtype=np.int32)
        rep_inlier_py = rep_py[inlier] if representative_inlier_count > 0 else np.asarray([], dtype=np.int32)
        rep_outlier_px = rep_px[outlier] if representative_outlier_count > 0 else np.asarray([], dtype=np.int32)
        rep_outlier_py = rep_py[outlier] if representative_outlier_count > 0 else np.asarray([], dtype=np.int32)
        raw_bbox = None
        if representative_inlier_count > 0 and len(rep_inlier_px) > 0 and len(rep_inlier_py) > 0:
            px0 = int(np.min(rep_inlier_px))
            px1 = int(np.max(rep_inlier_px))
            py0 = int(np.min(rep_inlier_py))
            py1 = int(np.max(rep_inlier_py))
            raw_bbox = [px0, py0, px1 + stride, py1 + stride]
        roi_area = max(1.0, float(max(1, x1 - x0) * max(1, y1 - y0)) / float(max(1, stride * stride)))
        raw_plane_view = self._plane_view_from_bbox(
            raw_bbox or roi_box,
            getattr(depth_frame, "shape", None),
            area_ratio=float(support_inlier_count) / roi_area if representative_inlier_count > 0 else None,
        )
        raw_width_norm = raw_plane_view.get("plane_width_norm")
        raw_cx_norm = raw_plane_view.get("plane_cx_norm")
        width_score = self._clip01((float(raw_width_norm or 0.0) - 0.12) / 0.30)
        coverage_score = self._clip01(0.30 * width_score + 0.35 * x_span_score + 0.20 * accepted_column_score + 0.15 * self._clip01(area_score / 0.70))
        abs_inlier_score = self._clip01(float(representative_inlier_count) / float(max(1, min_front_face_columns * 2)))
        ratio_inlier_score = self._clip01(inlier_ratio / 0.70)
        support_inlier_score = self._clip01(support_ratio / 0.70)
        evidence_score = self._clip01(0.40 * abs_inlier_score + 0.25 * ratio_inlier_score + 0.35 * support_inlier_score)
        yaw_abs = abs(float(yaw_err))
        geometry_score = self._clip01(1.0 - max(0.0, yaw_abs - 0.10) / max(1e-6, max_yaw - 0.10))
        support_geometry_score = self._clip01(
            0.35 * vertical_support_score
            + 0.25 * accepted_column_score
            + 0.20 * z_span_score
            + 0.20 * x_span_score
        )
        temporal = self._fast_temporal_score(yaw=float(yaw_err), dist=float(dist_err), cx_norm=raw_cx_norm)
        temporal_score = float(temporal.get("score", 0.0) or 0.0)
        temporal_available = bool(temporal.get("available", False))
        if dist_err > 0.60:
            distance_stage = "far"
        elif dist_err > 0.25:
            distance_stage = "middle"
        else:
            distance_stage = "near"
        frontness_score = max(0.0, 1.0 - 0.35 * float(selected_cluster_index))
        background_penalty_seed = 0.45 if int(selected_cluster_index) > 0 else 0.0
        run_verify, verify_reason = self._should_run_fast_verify(
            distance_stage=distance_stage,
            representative_inlier_count=representative_inlier_count,
            support_inlier_count=support_inlier_count,
            selected_cluster_support=selected_cluster_support,
            fit_inlier_x_span_m=x_span,
            residual_mean=residual_mean,
            residual_threshold=residual_threshold,
            yaw_abs=yaw_abs,
            max_yaw=max_yaw,
            selected_cluster_index=selected_cluster_index,
            selected_cluster_score=selected_cluster_score,
            background_rep_count=background_rep_count,
            background_penalty_seed=background_penalty_seed,
            temporal=temporal,
            min_front_face_columns=min_front_face_columns,
            min_vertical_support=min_vertical_support,
            min_front_face_x_span=min_front_face_x_span,
        )
        if run_verify:
            front_edge_start = time.perf_counter()
            edge_cue = self._extract_fast_front_edge_cue(
                depth_m=depth_m,
                x0=x0,
                y0=y0,
                stride=stride,
                calib=calib,
                pitch_deg=pitch_deg,
                camera_height_m=camera_height_m,
                z_min=z_min,
                z_max=z_max,
                target_dist_m=float(self._target_dist_m),
                min_x_span_m=float(min_front_face_x_span),
                max_yaw=float(max_yaw_cfg),
            )
            profile["fast_front_edge_ms"] = self._ms_since(front_edge_start)
            edge_debug_payload = self._fast_edge_debug_payload(
                edge_cue,
                debug_pixels_enabled=debug_pixels_enabled,
                debug_cap=debug_cap,
                skipped=False,
                skip_reason=verify_reason,
            )
        else:
            edge_debug_payload = self._fast_edge_debug_payload(
                edge_cue,
                debug_pixels_enabled=debug_pixels_enabled,
                debug_cap=debug_cap,
                skipped=True,
                skip_reason=verify_reason,
            )
        edge_x = np.asarray(edge_cue.get("x", []), dtype=np.float32)
        edge_y = np.asarray(edge_cue.get("y", []), dtype=np.float32)
        edge_inlier_full = np.asarray(edge_cue.get("inlier", []), dtype=bool)
        if edge_inlier_full.size == edge_x.size and int(edge_inlier_full.sum()) > 0:
            edge_x_eval = edge_x[edge_inlier_full]
            edge_y_eval = edge_y[edge_inlier_full]
        else:
            edge_x_eval = edge_x
            edge_y_eval = edge_y
        if len(edge_x_eval) and len(edge_y_eval):
            edge_res_to_selected = np.abs(edge_y_eval - (float(k) * edge_x_eval + float(b)))
            edge_consistency_score = max(0.0, 1.0 - float(np.mean(edge_res_to_selected)) / 0.08)
            edge_support_on_selected = int(np.sum(edge_res_to_selected <= 0.06))
        else:
            edge_consistency_score = 0.0
            edge_support_on_selected = 0
        selected_y_center_num = None if selected_cluster_y_center is None else float(selected_cluster_y_center)
        edge_y_median_m = edge_cue.get("y_median_m")
        front_y_gap = None
        if selected_y_center_num is not None and edge_y_median_m is not None:
            front_y_gap = float(selected_y_center_num) - float(edge_y_median_m)
        background_penalty = float(background_penalty_seed)
        if front_y_gap is not None:
            background_penalty = max(background_penalty, self._clip01((front_y_gap - 0.12) / 0.25))
        if run_verify:
            local_band_start = time.perf_counter()
            local_band = self._fast_local_band_stats(
                height_x,
                height_y,
                edge_x_eval,
                edge_y_eval,
                k=float(k),
                b=float(b),
                band_m=max(0.055, float(residual_threshold)),
            )
            profile["fast_local_band_ms"] = self._ms_since(local_band_start)
            local_band_support_count = int(local_band.get("count", 0) or 0)
            local_band_x_span = float(local_band.get("x_span_m", 0.0) or 0.0)
            local_band_edge_support = int(local_band.get("edge_support", 0) or edge_support_on_selected)
            local_band_residual_mean = float(local_band.get("residual_mean", 0.0) or 0.0)
            local_band_skipped = False
            local_band_skip_reason = ""
        else:
            local_band_support_count = int(support_inlier_count)
            local_band_x_span = float(x_span)
            local_band_edge_support = 0
            local_band_residual_mean = float(residual_mean)
            local_band_skipped = True
            local_band_skip_reason = str(verify_reason)
        line_source = "hybrid" if edge_consistency_score >= 0.55 and local_band_edge_support >= 3 else "vertical"
        line_score = self._clip01(
            0.30 * evidence_score
            + 0.20 * support_geometry_score
            + 0.18 * residual_score
            + 0.14 * frontness_score
            + 0.12 * edge_consistency_score
            + 0.06 * temporal_score
            - 0.25 * background_penalty
        )
        if temporal_available:
            confidence = self._clip01(
                0.25 * evidence_score
                + 0.15 * residual_score
                + 0.20 * coverage_score
                + 0.15 * geometry_score
                + 0.15 * support_geometry_score
                + 0.10 * temporal_score
            )
        else:
            confidence = self._clip01(
                0.30 * evidence_score
                + 0.15 * residual_score
                + 0.25 * coverage_score
                + 0.15 * geometry_score
                + 0.15 * support_geometry_score
            )
        prev_dist_used = None
        dist_delta = temporal.get("dist_delta")
        if dist_delta is not None:
            try:
                prev_dist_used = float(dist_err) - float(dist_delta)
            except Exception:
                prev_dist_used = None
        previous_near = prev_dist_used is not None and float(prev_dist_used) <= 0.25
        background_protect_start = time.perf_counter()
        near_stage_far_jump = bool(previous_near and (float(dist_err) - float(prev_dist_used) > 0.22) and edge_consistency_score < 0.65)
        background_blocked = bool(
            (front_y_gap is not None and front_y_gap > 0.16 and int(edge_cue.get("inlier_count", 0) or 0) >= 3)
            or (selected_cluster_index > 0 and distance_stage == "near")
            or near_stage_far_jump
        )
        profile["fast_background_protect_ms"] = self._ms_since(background_protect_start)

        control_gate_start = time.perf_counter()
        reject_reason = ""
        control_level = "none"
        width_min = max(0.06, float(min_front_face_x_span) * 0.90)
        hard_span_min = max(float(min_front_face_x_span), span_min * 0.90)
        local_band_min = max(20, min_front_face_columns * min_vertical_support * 2)
        if near_stage_far_jump:
            reject_reason = "near_stage_far_jump"
        elif background_blocked and background_penalty >= 0.60:
            reject_reason = "far_background_selected_blocked"
        elif selected_cluster_index > 0 and distance_stage == "near":
            reject_reason = "background_only"
        elif representative_inlier_count < min_front_face_columns:
            reject_reason = "vertical_support_low"
        elif support_inlier_count < min_front_face_columns * min_vertical_support:
            reject_reason = "vertical_support_low"
        elif rep_count < min_front_face_columns:
            reject_reason = "front_face_columns_low"
        elif residual_mean > residual_threshold:
            reject_reason = "residual_too_large"
        elif yaw_abs > max_yaw:
            reject_reason = "yaw_out_of_range"
        elif float(raw_width_norm or 0.0) < width_min:
            reject_reason = "width_too_small"
        elif x_span < hard_span_min:
            reject_reason = "front_face_x_span_low"
        elif local_band_support_count < local_band_min and local_band_edge_support < 3:
            reject_reason = "front_line_weak"
        elif int(edge_cue.get("inlier_count", 0) or 0) >= 4 and edge_consistency_score < 0.20 and background_penalty > 0.30:
            reject_reason = "edge_inconsistent"
        elif bool(temporal.get("jump", False)) and confidence < 0.62:
            reject_reason = "temporal_jump"
        else:
            if distance_stage == "near":
                usable_min, align_min = 0.40, 0.52
            elif distance_stage == "middle":
                usable_min, align_min = 0.44, 0.58
            else:
                usable_min, align_min = 0.38, 0.66
            if confidence < usable_min:
                reject_reason = "confidence_too_low"
            elif yaw_abs >= 0.55:
                control_level = "rotate_only"
            elif distance_stage == "near":
                if abs(dist_err) <= 0.12 and yaw_abs < 0.30 and confidence >= align_min:
                    control_level = "stop_ready"
                elif yaw_abs < 0.45 and confidence >= align_min:
                    control_level = "align"
                else:
                    control_level = "approach_slow"
            elif distance_stage == "middle":
                middle_span_ok = x_span >= max(span_min * 1.35, 0.28) and float(raw_width_norm or 0.0) >= 0.28
                if yaw_abs < 0.40 and confidence >= align_min and middle_span_ok:
                    control_level = "align"
                elif yaw_abs >= 0.45:
                    control_level = "rotate_only"
                else:
                    control_level = "approach_slow"
            else:
                if yaw_abs < 0.25 and confidence >= align_min:
                    control_level = "align"
                elif yaw_abs >= 0.45:
                    control_level = "rotate_only"
                else:
                    control_level = "approach_slow"
        if not reject_reason:
            if yaw_abs > max_yaw:
                reject_reason = "yaw_out_of_range"
                control_level = "none"
            elif background_blocked:
                reject_reason = "far_background_selected_blocked"
                control_level = "none"
            elif representative_inlier_count <= 3 or x_span < 0.15:
                if control_level in {"stop_ready", "align"}:
                    control_level = "approach_slow"
                support_mode = "edge" if representative_inlier_count > 0 else "none"
            elif (4 <= representative_inlier_count <= 7) or (0.15 <= x_span < 0.30):
                if control_level == "stop_ready":
                    control_level = "align"
                support_mode = "partial"
            if control_level == "stop_ready" and (
                line_source == "vertical"
                and edge_consistency_score < 0.25
                and int(edge_cue.get("inlier_count", 0) or 0) >= 4
            ):
                control_level = "align"
            if control_level == "stop_ready" and (
                local_band_support_count < max(40, local_band_min * 2)
                or local_band_x_span < 0.30
                or background_penalty > 0.0
            ):
                control_level = "align"
        profile["fast_control_gate_ms"] = self._ms_since(control_gate_start)
        plane_usable = control_level != "none"
        valid_for_control = control_level in {"align", "stop_ready"}
        debug_payload_start = time.perf_counter()
        support_px = support_px_all[selected_support_mask] if selected_support_mask.size == support_px_all.size else np.asarray([], dtype=np.int32)
        support_py = support_py_all[selected_support_mask] if selected_support_mask.size == support_py_all.size else np.asarray([], dtype=np.int32)
        fast_support_pixels = self._sparse_pixel_pairs(support_px, support_py, cap=debug_cap) if debug_pixels_enabled else []
        fast_rep_pixels = self._sparse_pixel_pairs(rep_px, rep_py, cap=debug_cap) if debug_pixels_enabled else []
        fast_inlier_pixels = self._sparse_pixel_pairs(rep_inlier_px, rep_inlier_py, cap=debug_cap) if debug_pixels_enabled else []
        fast_outlier_pixels = self._sparse_pixel_pairs(rep_outlier_px, rep_outlier_py, cap=debug_cap) if debug_pixels_enabled else []
        rep_background_px = rep_px[background_mask] if background_rep_count > 0 else np.asarray([], dtype=np.int32)
        rep_background_py = rep_py[background_mask] if background_rep_count > 0 else np.asarray([], dtype=np.int32)
        rep_weak_px = rep_px[weak_mask] if int(weak_mask.sum()) > 0 else np.asarray([], dtype=np.int32)
        rep_weak_py = rep_py[weak_mask] if int(weak_mask.sum()) > 0 else np.asarray([], dtype=np.int32)
        fast_background_pixels = self._sparse_pixel_pairs(rep_background_px, rep_background_py, cap=debug_cap) if debug_pixels_enabled else []
        fast_weak_pixels = self._sparse_pixel_pairs(rep_weak_px, rep_weak_py, cap=debug_cap) if debug_pixels_enabled else []
        profile["fast_debug_payload_ms"] = self._ms_since(debug_payload_start)
        edge_found = bool(plane_usable)
        if edge_found:
            plane_bbox = raw_bbox
        plane_view = self._plane_view_from_bbox(
            plane_bbox or roi_box,
            getattr(depth_frame, "shape", None),
            area_ratio=float(support_inlier_count) / roi_area if edge_found else None,
        )
        payload = {
            "table_found": bool(candidate_count > 0),
            "edge_found": bool(edge_found),
            "edge_detected": bool(edge_found),
            "edge_geometry_valid": bool(edge_found),
            "edge_valid": bool(edge_found),
            "valid_for_control": valid_for_control,
            "confidence": float(confidence if edge_found else 0.0),
            "edge_conf": float(confidence if edge_found else 0.0),
            "yaw_err_rad": float(yaw_err) if edge_found else None,
            "yaw_err": float(yaw_err) if edge_found else None,
            "dist_err_m": float(dist_err) if edge_found else None,
            "dist_err": float(dist_err) if edge_found else None,
            "edge_k": float(k) if edge_found else None,
            "edge_b": float(b) if edge_found else None,
            "depth_valid": True,
            "edge_obs_unavailable": False,
            "point_count": int(point_count),
            "sampled_point_count": int(sampled_count),
            "candidate_count": int(candidate_count),
            "table_point_count": int(candidate_count),
            "inlier_count": int(support_inlier_count),
            "edge_inlier_count": int(support_inlier_count),
            "representative_inlier_count": int(representative_inlier_count),
            "support_point_count": int(support_inlier_count),
            "valid_edge_points": int(point_count),
            "selected_edge": bool(edge_found),
            "near_edge": valid_for_control,
            **plane_view,
            "plane_found": bool(edge_found),
            "plane_confidence": float(confidence if edge_found else 0.0),
            "plane_residual_mean": float(residual_mean),
            "plane_residual_max": float(residual_max),
            "plane_x_span_m": float(x_span),
            "plane_yaw_err_rad": float(yaw_err) if edge_found else None,
            "plane_dist_err_m": float(dist_err) if edge_found else None,
            "plane_mask_status": "fast_sparse",
            "fast_debug_pixels_enabled": bool(debug_pixels_enabled),
            "fast_debug_pixel_cap": int(debug_cap),
            "fast_fit_attempted": True,
            "fast_raw_yaw_err_rad": float(yaw_err),
            "fast_raw_dist_err_m": float(dist_err),
            "fast_raw_plane_cx_norm": raw_plane_view.get("plane_cx_norm"),
            "fast_raw_plane_width_norm": raw_plane_view.get("plane_width_norm"),
            "fast_raw_plane_x_span_m": float(x_span),
            "fast_raw_residual_mean": float(residual_mean),
            "fast_raw_residual_p90": float(residual_p90),
            "fast_candidate_point_count": int(candidate_count),
            "fast_support_point_count": int(selected_cluster_support),
            "fast_all_support_point_count": int(support_total),
            "fast_rep_count": int(rep_count),
            "fast_rep_inlier_count": int(representative_inlier_count),
            "fast_rep_outlier_count": int(representative_outlier_count),
            "fast_background_rep_count": int(background_rep_count),
            "fast_front_rep_count": int(front_rep_count),
            "fast_rep_cluster_count": int(len(clusters)),
            "fast_selected_cluster_index": int(selected_cluster_index),
            "fast_selected_cluster_y_center": selected_cluster_y_center,
            "fast_selected_cluster_x_span_m": float(selected_cluster_x_span),
            "fast_selected_cluster_support": int(selected_cluster_support),
            "fast_selected_cluster_score": float(selected_cluster_score),
            "fast_candidate_x_span_m": float(candidate_x_span),
            "fast_support_x_span_m": float(selected_support_x_span),
            "fast_all_support_x_span_m": float(support_x_span),
            "fast_rep_x_span_m": float(rep_x_span),
            "fast_fit_inlier_x_span_m": float(x_span),
            "fast_residual_mean": float(residual_mean),
            "fast_residual_p90": float(residual_p90),
            "fast_support_mode": str(support_mode),
            "fast_fit_line_source": str(line_source),
            "fast_line_source": str(line_source),
            "fast_line_score": float(line_score),
            "fast_frontness_score": float(frontness_score),
            "fast_edge_consistency_score": float(edge_consistency_score),
            "fast_background_penalty": float(background_penalty),
            "fast_local_band_support_count": int(local_band_support_count),
            "fast_local_band_x_span_m": float(local_band_x_span),
            "fast_local_band_edge_support": int(local_band_edge_support),
            "fast_local_band_residual_mean": float(local_band_residual_mean),
            "fast_local_band_skipped": bool(local_band_skipped),
            "fast_local_band_skip_reason": str(local_band_skip_reason),
            "fast_background_blocked": bool(background_blocked),
            "fast_near_stage_far_jump": bool(near_stage_far_jump),
            "fast_selected_dist_source": str(line_source),
            "fast_prev_dist_used": prev_dist_used,
            "fast_raw_confidence": float(confidence),
            "fast_raw_inlier_count": int(support_inlier_count),
            "fast_raw_candidate_count": int(raw_candidate_count),
            "fast_candidate_point_cap": int(candidate_cap),
            "fast_candidate_downsampled": bool(raw_candidate_count > candidate_count),
            "fast_raw_sampled_point_count": int(sampled_count),
            "fast_raw_reject_reason": reject_reason or "none",
            "fast_gate_reject_reason": reject_reason or "none",
            "fast_gate_reason": reject_reason or "none",
            "fast_coord_frame": "robot_xyz",
            "fast_camera_pitch_deg": float(pitch_deg),
            "fast_camera_height_m": float(camera_height_m),
            "fast_table_height_m": float(table_height_m),
            "fast_robot_z_p10": robot_z_pct.get("p10"),
            "fast_robot_z_p50": robot_z_pct.get("p50"),
            "fast_robot_z_p90": robot_z_pct.get("p90"),
            "fast_robot_z_min": float(np.min(z_robot)) if len(z_robot) else None,
            "fast_robot_z_max": float(np.max(z_robot)) if len(z_robot) else None,
            "candidate_robot_z_min": float(np.min(height_z)) if len(height_z) else None,
            "candidate_robot_z_p50": candidate_z_pct.get("p50"),
            "candidate_robot_z_max": float(np.max(height_z)) if len(height_z) else None,
            "fast_ground_like_count": int(ground_like_count),
            "fast_table_height_like_count": int(table_height_like_count),
            "fast_front_face_rep_count": int(rep_count),
            "fast_front_face_support_point_count": int(selected_cluster_support),
            "fast_representative_inlier_count": int(representative_inlier_count),
            "fast_support_inlier_count": int(support_inlier_count),
            "fast_vertical_support_score": float(vertical_support_score),
            "fast_accepted_column_score": float(accepted_column_score),
            "fast_z_span_score": float(z_span_score),
            "fast_x_span_score": float(x_span_score),
            "fast_z_span_m_p50": float(z_span_p50),
            "fast_z_span_m_max": float(z_span_max),
            "fast_birdview_fit_residual_mean": float(residual_mean),
            "fast_confidence_version": "v3",
            "fast_distance_stage": distance_stage,
            "fast_control_level": control_level,
            "fast_score_inlier": float(inlier_ratio),
            "fast_score_abs_inlier": float(abs_inlier_score),
            "fast_score_inlier_ratio": float(ratio_inlier_score),
            "fast_score_evidence": float(evidence_score),
            "fast_score_residual": float(residual_score),
            "fast_score_span": float(span_score),
            "fast_score_area": float(area_score),
            "fast_score_coverage": float(coverage_score),
            "fast_score_geometry": float(geometry_score),
            "fast_score_support_geometry": float(support_geometry_score),
            "fast_score_temporal": float(temporal_score),
            "fast_score_temporal_available": bool(temporal_available),
            "fast_temporal_stable_count": int(temporal.get("stable_count", 0) or 0),
            "fast_temporal_jump": bool(temporal.get("jump", False)),
            "fast_temporal_yaw_delta": temporal.get("yaw_delta"),
            "fast_temporal_dist_delta": temporal.get("dist_delta"),
            "fast_temporal_cx_delta": temporal.get("cx_delta"),
            "fast_score_final": float(confidence),
            **edge_debug_payload,
            "fast_candidate_pixel_count": int(len(fast_candidate_pixels)),
            "fast_support_pixel_count": int(len(fast_support_pixels)),
            "fast_front_face_rep_pixel_count": int(len(fast_rep_pixels)),
            "fast_inlier_pixel_count": int(len(fast_inlier_pixels)),
            "fast_outlier_pixel_count": int(len(fast_outlier_pixels)),
            "fast_background_pixel_count": int(len(fast_background_pixels)),
            "fast_weak_pixel_count": int(len(fast_weak_pixels)),
            "fast_sampled_pixels": raw_sample_pixels,
            "fast_candidate_pixels": fast_candidate_pixels,
            "fast_support_pixels": fast_support_pixels,
            "fast_front_face_rep_pixels": fast_rep_pixels,
            "fast_inlier_pixels": fast_inlier_pixels,
            "fast_outlier_pixels": fast_outlier_pixels,
            "fast_background_pixels": fast_background_pixels,
            "fast_rep_background_pixels": fast_background_pixels,
            "fast_weak_pixels": fast_weak_pixels,
            "pose_source": "fast_plane_only" if edge_found else "none",
            "final_pose_source": "fast_plane_only" if edge_found else "none",
            "view_err_norm": plane_view.get("plane_cx_norm") if edge_found else None,
            "view_source": "plane" if edge_found else "none",
            "view_reliable": bool(edge_found),
            "fov_guard_active": False,
            "frame_id": int(self._frame_id),
            "frame_seq": int(frame_seq),
            "source": "vision_table_edge_manager",
            "reason": reject_reason,
            "reject_reason": reject_reason,
            "target_dist_m": float(self._target_dist_m),
            "plane_only_mode": True,
            "enable_crease_line": False,
            "usable_for_approach": bool(plane_usable),
            "usable_for_alignment": bool(control_level in {"align", "stop_ready"}),
            "usable_for_stop": bool(control_level == "stop_ready"),
            "control_level": control_level,
            "control_reject_reason": "" if plane_usable else reject_reason,
            **self._detector_mode_payload(),
            **yolo_gate,
            **roi_payload,
            "type": "table_edge_obs",
        }
        profile["obs_build_ms"] = self._ms_since(obs_start)
        profile["total_edge_process_ms"] = self._ms_since(total_start)
        return self._attach_profile(payload, profile, path="fast_plane_only")

    def _process_depth_lightweight(self, depth_frame: np.ndarray, frame_seq: int) -> Dict[str, Any]:
        total_start = time.perf_counter()
        profile = self._profile_template()
        profile["depth_frame_fetch_ms"] = float(self._last_depth_frame_fetch_ms)
        roi_select_start = time.perf_counter()
        roi_meta = self._select_roi(depth_frame)
        yolo_gate = self._yolo_table_confirmation()
        roi_box = roi_meta.get("depth_edge_roi") if roi_meta.get("roi_source") != "static_fallback" else self._static_roi()
        if roi_box is None:
            roi_box = self._static_roi()
        try:
            roi_box = self._detector._resolve_roi(depth_frame, roi_override=roi_box) if self._detector is not None else tuple(int(v) for v in roi_box)
        except Exception:
            roi_box = tuple(int(v) for v in self._static_roi() or (0, 0, depth_frame.shape[1], depth_frame.shape[0]))
        x0, y0, x1, y1 = [int(v) for v in roi_box]
        stride = max(1, int(self._light_stride))
        depth_roi = depth_frame[y0:y1:stride, x0:x1:stride]
        profile["roi_crop_ms"] = self._ms_since(roi_select_start)

        prep_start = time.perf_counter()
        if depth_roi.size <= 0:
            payload = self._default_result(depth_valid=False, reason="roi_empty", frame_seq=frame_seq, roi_meta=roi_meta)
            profile["depth_preprocess_ms"] = self._ms_since(prep_start)
            profile["total_edge_process_ms"] = self._ms_since(total_start)
            return self._attach_profile(payload, profile, path="light_roi_empty")
        if depth_roi.dtype != np.float32:
            depth_m = depth_roi.astype(np.float32) * float(self._target_dist_m * 0.0 + getattr(self._detector.calib, "depth_scale", 0.001) if self._detector is not None else 0.001)
        else:
            depth_m = depth_roi
        cfg = self._detector_cfg
        z_min = float(getattr(cfg, "z_min", 0.2) if cfg is not None else 0.2)
        z_max = float(getattr(cfg, "z_max", 2.0) if cfg is not None else 2.0)
        valid_mask = (depth_m > z_min) & (depth_m < z_max)
        valid_count = int(valid_mask.sum())
        profile["depth_preprocess_ms"] = self._ms_since(prep_start)

        fit_start = time.perf_counter()
        min_all = max(40, int(getattr(cfg, "min_all_points", 1000) if cfg is not None else 1000) // max(1, stride * stride))
        min_table = max(25, int(getattr(cfg, "min_table_points", 500) if cfg is not None else 500) // max(1, stride * stride))
        roi_payload = self._roi_payload(roi_box, roi_meta)
        if self._detector is None or valid_count < min_all:
            payload = self._default_result(
                depth_valid=True,
                reason=self._detector_error or "not_enough_points",
                frame_seq=frame_seq,
                roi_meta=roi_meta,
            )
            payload.update(roi_payload)
            payload["point_count"] = int(valid_count)
            profile["plane_or_edge_fit_ms"] = self._ms_since(fit_start)
            profile["total_edge_process_ms"] = self._ms_since(total_start)
            return self._attach_profile(payload, profile, path="light_not_enough_points")

        calib = self._detector.calib
        yy, xx = np.nonzero(valid_mask)
        z = depth_m[valid_mask]
        u = (x0 + xx.astype(np.float32) * float(stride))
        v = (y0 + yy.astype(np.float32) * float(stride))
        x_c = (u - float(calib.cx)) * z / float(calib.fx)
        y_c = (v - float(calib.cy)) * z / float(calib.fy)
        table_mask = (y_c > float(getattr(cfg, "table_y_min", -0.2))) & (y_c < float(getattr(cfg, "table_y_max", 0.2)))
        table_count = int(table_mask.sum())
        if table_count < min_table:
            payload = self._default_result(depth_valid=True, reason="no_valid_edge", frame_seq=frame_seq, roi_meta=roi_meta)
            payload.update(roi_payload)
            payload["point_count"] = int(valid_count)
            payload["table_point_count"] = int(table_count)
            profile["plane_or_edge_fit_ms"] = self._ms_since(fit_start)
            profile["total_edge_process_ms"] = self._ms_since(total_start)
            return self._attach_profile(payload, profile, path="light_no_table_points")

        x_t = x_c[table_mask]
        z_t = z[table_mask]
        try:
            k, b = np.polyfit(x_t, z_t, 1)
            residual = np.abs(z_t - (float(k) * x_t + float(b)))
            threshold = float(getattr(cfg, "residual_threshold_m", 0.05))
            inlier = residual <= threshold
            inlier_count = int(inlier.sum())
            if inlier_count >= min_table:
                k, b = np.polyfit(x_t[inlier], z_t[inlier], 1)
            else:
                inlier_count = table_count
            yaw_err = math.atan(float(k))
            dist_err = float(b) - float(self._target_dist_m)
            edge_conf = float(inlier_count) / float(max(1, table_count))
            edge_found = bool(inlier_count >= min_table)
            if edge_found and inlier_count >= min_table:
                yy_plane = yy[table_mask][inlier]
                xx_plane = xx[table_mask][inlier]
            else:
                yy_plane = yy[table_mask]
                xx_plane = xx[table_mask]
        except Exception as exc:
            payload = self._default_result(depth_valid=True, reason=f"light_fit_failed:{exc}", frame_seq=frame_seq, roi_meta=roi_meta)
            payload.update(roi_payload)
            payload["point_count"] = int(valid_count)
            payload["table_point_count"] = int(table_count)
            profile["plane_or_edge_fit_ms"] = self._ms_since(fit_start)
            profile["total_edge_process_ms"] = self._ms_since(total_start)
            return self._attach_profile(payload, profile, path="light_fit_failed")
        profile["plane_or_edge_fit_ms"] = self._ms_since(fit_start)
        plane_bbox = None
        if edge_found and len(xx_plane) > 0 and len(yy_plane) > 0:
            px0 = x0 + int(np.min(xx_plane)) * stride
            px1 = x0 + int(np.max(xx_plane)) * stride
            py0 = y0 + int(np.min(yy_plane)) * stride
            py1 = y0 + int(np.max(yy_plane)) * stride
            plane_bbox = [px0, py0, px1 + stride, py1 + stride]
        roi_area = max(1.0, float(max(1, x1 - x0) * max(1, y1 - y0)) / float(max(1, stride * stride)))
        plane_view = self._plane_view_from_bbox(
            plane_bbox or roi_box,
            getattr(depth_frame, "shape", None),
            area_ratio=float(inlier_count) / roi_area if edge_found else None,
        )

        payload = {
            "table_found": bool(table_count > 0),
            "edge_found": edge_found,
            "edge_detected": bool(edge_found),
            "edge_geometry_valid": bool(edge_found),
            "edge_valid": bool(edge_found),
            "valid_for_control": bool(edge_found),
            "confidence": float(edge_conf),
            "edge_conf": float(edge_conf),
            "yaw_err_rad": float(yaw_err) if edge_found else None,
            "yaw_err": float(yaw_err) if edge_found else None,
            "dist_err_m": float(dist_err) if edge_found else None,
            "dist_err": float(dist_err) if edge_found else None,
            "edge_k": float(k) if edge_found else None,
            "edge_b": float(b) if edge_found else None,
            "depth_valid": True,
            "edge_obs_unavailable": False,
            "point_count": int(valid_count),
            "valid_edge_points": int(valid_count),
            "table_point_count": int(table_count),
            "edge_inlier_count": int(inlier_count),
            "selected_edge": edge_found,
            "near_edge": edge_found,
            **plane_view,
            "view_err_norm": plane_view.get("plane_cx_norm") if edge_found else None,
            "view_source": "plane" if edge_found else "none",
            "view_reliable": bool(edge_found),
            "fov_guard_active": False,
            "frame_id": int(self._frame_id),
            "frame_seq": int(frame_seq),
            "source": "vision_table_edge_manager",
            "reason": "" if edge_found else "no_valid_edge",
            "target_dist_m": float(self._target_dist_m),
            "lightweight": True,
            "sample_stride": int(stride),
            **yolo_gate,
            **roi_payload,
            "type": "table_edge_obs",
        }
        profile["total_edge_process_ms"] = self._ms_since(total_start)
        return self._attach_profile(payload, profile, path="light")

    def process_camera_frame(
        self,
        frames: Dict[str, Any],
        *,
        frame_seq: int,
        frame_slot: Optional[Dict[str, Any]] = None,
        local_perception: Optional[Dict[str, Any]] = None,
        runtime_status: Optional[Dict[str, Any]] = None,
        source_mode: Optional[str] = None,
        depth_frame_fetch_ms: float = 0.0,
        count_dropped: bool = True,
    ) -> Dict[str, Any]:
        """Process one RGB/depth frame pack through the online table-plane path.

        The emitted type and legacy field aliases stay `table_edge_obs` for current
        state-machine compatibility; plane_* fields carry the unified semantics.
        """
        if not isinstance(frames, dict):
            frames = {}
        seq = int(frame_seq)
        if count_dropped and seq > self._last_camera_seq + 1 and self._last_camera_seq > 0:
            self._dropped_frame_count += int(seq - self._last_camera_seq - 1)
        if count_dropped:
            self._last_camera_seq = seq
        self._frame_id += 1
        self._last_obs_seq += 1
        slot = dict(frame_slot or {})
        if "seq" not in slot:
            slot["seq"] = seq
        if "payload" not in slot:
            slot["payload"] = frames
        self._last_depth_frame_fetch_ms = float(depth_frame_fetch_ms or 0.0)
        frame_capture_ts = self._pick_frame_capture_ts(slot, frames)
        if float(self._last_camera_frame_capture_ts or 0.0) > 0.0:
            self._last_camera_frame_interval_ms = max(0.0, (float(frame_capture_ts) - float(self._last_camera_frame_capture_ts)) * 1000.0)
        self._last_camera_frame_capture_ts = float(frame_capture_ts)
        latest_frame_lag_ms = max(0.0, (time.time() - float(frame_capture_ts)) * 1000.0)
        vision_start_ts = time.time()
        if float(self._last_process_start_ts or 0.0) > 0.0:
            self._last_table_edge_process_interval_ms = max(0.0, (float(vision_start_ts) - float(self._last_process_start_ts)) * 1000.0)
        self._last_process_start_ts = float(vision_start_ts)
        prev_source_mode = self._source_mode_override
        prev_local = self._local_perception_override
        prev_runtime = self._runtime_status_override
        self._source_mode_override = source_mode if source_mode is not None else prev_source_mode
        self._local_perception_override = dict(local_perception) if local_perception is not None else prev_local
        self._runtime_status_override = dict(runtime_status) if runtime_status is not None else prev_runtime
        self._processing_busy = True
        depth = frames.get("depth")
        frame_calib = self._resolve_frame_calib(frames, depth)
        self._frame_calib_payload = self._build_calib_payload(frame_calib, depth)
        if self._detector is not None and frame_calib is not None:
            self._detector.calib = frame_calib
        try:
            if not isinstance(depth, np.ndarray) or depth.size <= 0:
                payload = self._default_result(
                    depth_valid=False,
                    reason="depth_unavailable",
                    frame_seq=seq,
                    roi_meta=self._select_roi(None),
                )
            elif depth.ndim != 2:
                payload = self._default_result(
                    depth_valid=False,
                    reason="depth_frame_not_2d",
                    frame_seq=seq,
                    roi_meta=self._select_roi(depth),
                )
            else:
                payload = self._process_depth(depth, seq)
        finally:
            self._processing_busy = False
        vision_done_ts = time.time()
        if isinstance(payload, dict):
            payload["obs_seq"] = int(self._last_obs_seq)
            payload["camera_frame_seq"] = int(seq)
        self._processed_frame_count += 1
        self._last_process_ms = max(0.0, (vision_done_ts - vision_start_ts) * 1000.0)
        try:
            return self._with_freshness(
                payload,
                frame_capture_ts=frame_capture_ts,
                vision_start_ts=vision_start_ts,
                vision_done_ts=vision_done_ts,
                latest_frame_lag_ms=latest_frame_lag_ms,
            )
        finally:
            self._source_mode_override = prev_source_mode
            self._local_perception_override = prev_local
            self._runtime_status_override = prev_runtime

    def _worker_error_result(
        self,
        frames: Any,
        *,
        frame_seq: int,
        runtime_status: Optional[Dict[str, Any]] = None,
        error: Exception,
    ) -> Dict[str, Any]:
        reason = f"table_edge_worker_error:{type(error).__name__}"
        depth_frame = frames.get("depth") if isinstance(frames, dict) else None
        roi_meta: Dict[str, Any] = {}
        if isinstance(depth_frame, np.ndarray):
            try:
                roi_meta = self._select_roi(depth_frame)
            except Exception:
                roi_meta = {}
        payload = self._default_result(
            depth_valid=isinstance(depth_frame, np.ndarray),
            reason=reason,
            frame_seq=frame_seq,
            roi_meta=roi_meta,
        )
        payload.update(self._detector_mode_payload())
        payload.update(
            {
                "table_edge_worker_error": str(error),
                "worker_error": str(error),
                "reject_reason": reason,
                "control_reject_reason": reason,
                "source_mode": str((runtime_status or {}).get("mode") or "").strip().upper(),
            }
        )
        if isinstance(depth_frame, np.ndarray):
            try:
                payload["depth_shape"] = [int(v) for v in depth_frame.shape[:2]]
            except Exception:
                pass
        return payload

    def _worker_loop(self) -> None:
        while self._runtime_running and not self._worker_stop.is_set():
            loop_start = time.time()
            if float(self._last_worker_loop_ts or 0.0) > 0.0:
                self._last_worker_interval_ms = max(0.0, (float(loop_start) - float(self._last_worker_loop_ts)) * 1000.0)
            self._last_worker_loop_ts = float(loop_start)
            scheduler = self._scheduler
            interval_s = self._worker_interval_s
            if scheduler is None:
                self._worker_stop.wait(timeout=interval_s)
                continue
            fetch_start = time.perf_counter()
            frame_slot = scheduler.read_slot("camera_frames")
            self._last_depth_frame_fetch_ms = self._ms_since(fetch_start)
            self._last_scheduler_read_ms = float(self._last_depth_frame_fetch_ms)
            if not isinstance(frame_slot, dict):
                self._table_edge_no_new_frame_count += 1
                self._worker_stop.wait(timeout=interval_s)
                continue
            generation = int(frame_slot.get("generation", 0) or 0)
            if generation != self._last_camera_generation:
                self._last_camera_generation = generation
                self._last_camera_seq = 0
            seq = int(frame_slot.get("seq", 0) or 0)
            frames = frame_slot.get("payload")
            if seq <= self._last_camera_seq or not isinstance(frames, dict):
                self._table_edge_no_new_frame_count += 1
                self._worker_stop.wait(timeout=interval_s)
                continue
            if seq > self._last_camera_seq + 1 and self._last_camera_seq > 0:
                self._dropped_frame_count += int(seq - self._last_camera_seq - 1)
            self._last_camera_seq = seq
            runtime_status = self._runtime_status()
            runtime_mode = str(runtime_status.get("mode") or "").strip().upper()
            try:
                payload = self.process_camera_frame(
                    frames,
                    frame_seq=seq,
                    frame_slot=frame_slot,
                    runtime_status=runtime_status,
                    source_mode=runtime_mode if runtime_mode else None,
                    depth_frame_fetch_ms=self._last_depth_frame_fetch_ms,
                    count_dropped=False,
                )
            except Exception as exc:
                self.log.exception("table_edge_manager.loop detector failed | frame_seq=%s error=%s", seq, exc)
                payload = self._worker_error_result(
                    frames,
                    frame_seq=seq,
                    runtime_status=runtime_status,
                    error=exc,
                )
            self._emit_edge_debug(payload)
            self._log_profile_if_due(payload)
            self._publish_result("table_edge_obs", payload)
            elapsed_s = max(0.0, time.time() - loop_start)
            self._worker_stop.wait(timeout=max(0.0, interval_s - elapsed_s))

    def _emit_edge_debug(self, payload: Dict[str, Any]) -> None:
        debug_cfg = getattr(self.cfg, "debug", None)
        if not bool(getattr(debug_cfg, "edge_debug_enabled", False)):
            return
        now = time.time()
        period_s = float(getattr(debug_cfg, "edge_debug_period_s", 1.0) or 1.0)
        if now - float(self._last_edge_dbg_ts or 0.0) < max(0.1, period_s):
            return
        self._last_edge_dbg_ts = now
        valid = bool(payload.get("edge_valid", payload.get("edge_found", False)))
        self.log.info(
            "[EDGE_DBG] valid=%s dist=%s yaw=%s age_ms=%s roi=%s",
            int(valid),
            payload.get("dist_err_m"),
            payload.get("yaw_err_rad"),
            payload.get("age_ms"),
            payload.get("roi_preset") or payload.get("roi_source") or payload.get("edge_roi"),
        )

    def release_all(self) -> None:
        self.stop_runtime()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "detector_ready": bool(self._detector is not None),
            "detector_error": str(self._detector_error or ""),
            "runtime_running": bool(self._runtime_running),
            "last_camera_generation": int(self._last_camera_generation),
            "last_camera_seq": int(self._last_camera_seq),
            "last_publish_ts": float(self._last_publish_ts),
            "last_obs_ts": float(self._last_obs_ts),
            "last_process_ms": float(self._last_process_ms),
            "last_update_interval_ms": self._last_update_interval_ms,
            "frame_id": int(self._frame_id),
            "dropped_frame_count": int(self._dropped_frame_count),
            "processed_frame_count": int(self._processed_frame_count),
            "processing_busy": bool(self._processing_busy),
            "target_dist_m": float(self._target_dist_m),
            "last_valid_quadrant": self._last_valid_quadrant,
            "default_update_hz": 1.0 / max(1e-6, float(self._default_interval_s)),
            "detector_mode": self._detector_mode,
            "edge_update_hz": self._edge_update_hz,
            "detector_mode": str(self._detector_mode),
            "fast_plane_stride": int(self._fast_plane_stride),
        }
