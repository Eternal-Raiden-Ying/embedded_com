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

from .table_edge_roi import choose_depth_roi
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
        table_edge_cfg = getattr(self.cfg, "table_edge", None)
        edge_hz = float(getattr(table_edge_cfg, "update_hz", 10.0) or 10.0)
        self._worker_interval_s = 1.0 / max(1.0, edge_hz)
        track_edge_hz = float(getattr(table_edge_cfg, "track_local_update_hz", 5.0) or 5.0)
        self._track_local_interval_s = 1.0 / max(5.0, track_edge_hz)
        self._track_local_lightweight = bool(getattr(table_edge_cfg, "track_local_light_edge", True))
        self._light_stride = max(1, int(float(getattr(table_edge_cfg, "track_local_edge_stride", 4) or 4)))
        self._default_interval_s = self._worker_interval_s
        self._last_camera_generation = 0
        self._last_camera_seq = 0
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
        self._target_dist_m = 0.5
        self._last_valid_quadrant: Optional[str] = None
        self._last_valid_quadrant_ts = 0.0
        self._last_valid_quadrant_ttl_s = 1.0
        self._last_valid_table_bbox = None
        self._last_valid_table_center_norm = None
        self._load_detector()

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
        candidates = [
            ("Online_Edge_Detect.board_config", "Online_Edge_Detect.detector"),
            ("VISTA.Online_Edge_Detect.board_config", "VISTA.Online_Edge_Detect.detector"),
        ]
        last_error = ""
        for cfg_mod_name, det_mod_name in candidates:
            try:
                cfg_mod = __import__(cfg_mod_name, fromlist=["CONFIG"])
                det_mod = __import__(det_mod_name, fromlist=["OnlineTableEdgeDetector", "load_calib"])
                edge_cfg = getattr(cfg_mod, "CONFIG")
                load_calib = getattr(det_mod, "load_calib")
                detector_cls = getattr(det_mod, "OnlineTableEdgeDetector")
                calib_path = Path(str(edge_cfg.detector.calib_json)).expanduser()
                calib, target_dist = load_calib(calib_path)
                if float(edge_cfg.detector.target_dist_m_override) > 0:
                    target_dist = float(edge_cfg.detector.target_dist_m_override)
                self._detector = detector_cls(calib, edge_cfg.detector, target_dist)
                self._detector_cfg = edge_cfg.detector
                self._target_dist_m = float(target_dist)
                if bool(getattr(edge_cfg.detector, "plane_only_mode", False)):
                    self._track_local_lightweight = False
                self._detector_error = ""
                self._emit(
                    "loaded",
                    calib_json=str(calib_path),
                    target_dist_m=float(self._target_dist_m),
                )
                return
            except Exception as exc:
                last_error = str(exc)
        self._detector = None
        self._detector_cfg = None
        self._detector_error = str(last_error or "detector_unavailable")
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
                payload["obs_publish_ts"] = float(time.time())
                done_ts = float(payload.get("vision_done_ts") or payload["obs_publish_ts"])
                payload["publish_delay_ms"] = max(0.0, (payload["obs_publish_ts"] - done_ts) * 1000.0)
            scheduler.publish_result(route, payload, generation=generation)
            self._last_publish_ts = time.time()
        except Exception:
            pass

    def _active_mode(self) -> str:
        scheduler = self._scheduler
        if scheduler is None:
            return ""
        try:
            return str((scheduler.snapshot().get("active_mode") or "")).strip().upper()
        except Exception:
            return ""

    def _current_interval_s(self) -> float:
        if self._active_mode() == "TRACK_LOCAL":
            return min(float(self._default_interval_s), float(self._track_local_interval_s))
        return float(self._default_interval_s)

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
        out["ts"] = float(obs_ts)
        out["obs_ts"] = float(obs_ts)
        out["frame_capture_ts"] = float(frame_capture_ts)
        out["vision_start_ts"] = float(vision_start_ts)
        out["vision_done_ts"] = float(vision_done_ts)
        out["frame_age_ms"] = max(0.0, (float(vision_start_ts) - float(frame_capture_ts)) * 1000.0)
        out["vision_process_ms"] = max(0.0, (float(vision_done_ts) - float(vision_start_ts)) * 1000.0)
        out["obs_total_age_ms"] = max(0.0, (float(obs_ts) - float(frame_capture_ts)) * 1000.0)
        out["age_ms"] = float(out["obs_total_age_ms"])
        out["edge_update_interval_ms"] = update_interval_ms
        out["edge_process_ms"] = float(out["vision_process_ms"])
        out["total_edge_process_ms"] = float(out["vision_process_ms"])
        out.setdefault("depth_frame_fetch_ms", float(self._last_depth_frame_fetch_ms))
        out["dropped_frame_count"] = int(self._dropped_frame_count)
        out["processed_frame_count"] = int(self._processed_frame_count)
        out["latest_frame_lag_ms"] = float(latest_frame_lag_ms)
        unavailable = bool(out.get("edge_obs_unavailable", False))
        out["is_stale"] = bool(out.get("is_stale", False) or unavailable)
        out["source_mode"] = self._active_mode()
        out.setdefault("seq", out.get("frame_seq", out.get("frame_id")))
        out.setdefault("frame_id", out.get("frame_seq", out.get("seq")))
        out.setdefault("edge_conf", out.get("confidence"))
        out.setdefault("edge_valid", bool(out.get("edge_found", False)) and not unavailable)
        out.setdefault("yaw_err", out.get("yaw_err_rad"))
        out.setdefault("dist_err", out.get("dist_err_m"))
        return out

    def _log_profile_if_due(self, payload: Dict[str, Any]) -> None:
        interval_s = float(getattr(getattr(self.cfg, "table_edge", None), "profile_log_interval_s", 2.0) or 2.0)
        if interval_s <= 0.0:
            return
        now = time.time()
        if now - float(self._last_profile_log_ts or 0.0) < interval_s:
            return
        self._last_profile_log_ts = now
        self.log.info(
            "[TABLE_EDGE_PROFILE] obs_total_age_ms=%.1f vision_process_ms=%.1f edge_update_interval_ms=%s dropped=%d processed=%d latest_frame_lag_ms=%.1f",
            float(payload.get("obs_total_age_ms") or 0.0),
            float(payload.get("vision_process_ms") or 0.0),
            "None" if payload.get("edge_update_interval_ms") is None else f"{float(payload.get('edge_update_interval_ms')):.1f}",
            int(payload.get("dropped_frame_count") or 0),
            int(payload.get("processed_frame_count") or 0),
            float(payload.get("latest_frame_lag_ms") or 0.0),
        )

    @staticmethod
    def _ms_since(start_ts: float) -> float:
        return max(0.0, (time.perf_counter() - float(start_ts)) * 1000.0)

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
        max_reliable_area: float = 0.65,
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
            "reliable": bool(area > 0.0 and area <= float(max_reliable_area) and not touch_boundary),
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
        }

    def _attach_profile(self, payload: Dict[str, Any], profile: Dict[str, Any], *, path: str) -> Dict[str, Any]:
        out = dict(payload or {})
        prof = self._profile_template()
        prof.update({k: float(v) for k, v in dict(profile or {}).items() if isinstance(v, (int, float))})
        prof["total_edge_process_ms"] = float(prof.get("total_edge_process_ms", 0.0) or 0.0)
        out.update(prof)
        out["edge_profile"] = dict(prof)
        out["edge_process_path"] = str(path or "")
        return out

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
            "edge_valid": False,
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
            "target_dist_m": float(self._target_dist_m),
            "plane_only_mode": bool(getattr(self._detector_cfg, "plane_only_mode", False)),
            "enable_crease_line": bool(getattr(self._detector_cfg, "enable_crease_line", True)),
            "table_confirmed_by_yolo": False,
            "yolo_table_conf": None,
            "yolo_gate_reason": str(reason or ""),
            "yolo_reliable": False,
            "yolo_gate_open": False,
            "yolo_bbox_area_norm": None,
            "yolo_bbox_touch_left": False,
            "yolo_bbox_touch_right": False,
            "yolo_bbox_touch_boundary": False,
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
        table_edge_cfg = getattr(self.cfg, "table_edge", None)
        require_yolo = bool(getattr(table_edge_cfg, "require_yolo_table_confirm", True))
        plane_only = bool(getattr(self._detector_cfg, "plane_only_mode", False))
        if plane_only and not bool(getattr(table_edge_cfg, "enable_yolo_in_plane_only", False)):
            require_yolo = False
        if not require_yolo:
            return {
                "table_confirmed_by_yolo": False,
                "yolo_table_conf": None,
                "yolo_gate_reason": "not_required_plane_only",
                "yolo_reliable": False,
                "yolo_gate_open": True,
            }
        local_payload = dict(local if local is not None else self._local_perception())
        min_conf = float(getattr(table_edge_cfg, "yolo_table_min_conf", 0.25) or 0.25)
        det = table_detection_debug(local_payload, local_payload.get("rgb_shape"), min_conf=min_conf)
        source = str(local_payload.get("table_roi_source") or "").strip()
        confirmed = bool(det.get("found")) and source == "yolo_table_bbox"
        reason = "yolo_table_confirmed" if confirmed else str(det.get("reason") or "waiting_yolo_table_confirm")
        if source and source != "yolo_table_bbox":
            reason = f"table_source_{source}"
        bbox_metrics = self._bbox_view_metrics(det.get("bbox"), local_payload.get("rgb_shape"))
        bbox_conf = det.get("conf")
        yolo_reliable = bool(
            confirmed
            and bbox_metrics.get("reliable", False)
            and bbox_conf is not None
            and float(bbox_conf) >= min_conf
        )
        return {
            "table_confirmed_by_yolo": confirmed,
            "yolo_table_conf": det.get("conf"),
            "yolo_gate_reason": reason,
            "yolo_table_bbox": det.get("bbox"),
            "yolo_reliable": yolo_reliable,
            "yolo_gate_open": bool(confirmed),
            "yolo_bbox_area_norm": bbox_metrics.get("area_norm"),
            "yolo_bbox_touch_left": bool(bbox_metrics.get("touch_left", False)),
            "yolo_bbox_touch_right": bool(bbox_metrics.get("touch_right", False)),
            "yolo_bbox_touch_boundary": bool(bbox_metrics.get("touch_boundary", False)),
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
        return bool(getattr(getattr(self.cfg, "table_edge", None), "static_roi_enabled", False))

    def _debug_roi_preset(self) -> str:
        return str(getattr(getattr(self.cfg, "table_edge", None), "roi_preset", "") or "").strip().lower()

    def _local_perception(self) -> Dict[str, Any]:
        scheduler = self._scheduler
        if scheduler is None:
            return {}
        try:
            value = scheduler.read_result("local_perception", default={}) or {}
        except Exception:
            return {}
        return dict(value) if isinstance(value, dict) else {}

    def _runtime_status(self) -> Dict[str, Any]:
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
        if mode != "TRACK_LOCAL":
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
        )
        quadrant = roi_meta.get("table_quadrant")
        table_bbox = roi_meta.get("table_bbox")
        if quadrant and roi_meta.get("roi_source") == "local_perception_table_bbox":
            self._last_valid_quadrant = str(quadrant).strip().upper()
            self._last_valid_quadrant_ts = time.time()
            self._last_valid_table_bbox = table_bbox
            self._last_valid_table_center_norm = roi_meta.get("table_center_norm")
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
        if (
            self._active_mode() == "TRACK_LOCAL"
            and self._track_local_lightweight
            and not bool(getattr(self._detector_cfg, "plane_only_mode", False))
        ):
            return self._process_depth_lightweight(depth_frame, frame_seq)
        total_start = time.perf_counter()
        profile = self._profile_template()
        profile["depth_frame_fetch_ms"] = float(self._last_depth_frame_fetch_ms)
        roi_meta = self._select_roi(depth_frame)
        yolo_gate = self._yolo_table_confirmation()
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
        except Exception as exc:
            self.log.debug("table edge detect failed | error=%s", exc)
            payload = self._default_result(depth_valid=True, reason=f"detect_failed:{exc}", frame_seq=frame_seq, roi_meta=roi_meta)
            profile["total_edge_process_ms"] = self._ms_since(total_start)
            return self._attach_profile(payload, profile, path="full_detect_failed")
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
            "edge_valid": valid_for_control,
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
            **yolo_gate,
            **roi_payload,
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
        profile["total_edge_process_ms"] = self._ms_since(total_start)
        return self._attach_profile(payload, profile, path="full")

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
            "edge_valid": edge_found,
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

    def _worker_loop(self) -> None:
        while self._runtime_running and not self._worker_stop.is_set():
            loop_start = time.time()
            scheduler = self._scheduler
            interval_s = self._current_interval_s()
            if scheduler is None:
                self._worker_stop.wait(timeout=interval_s)
                continue
            fetch_start = time.perf_counter()
            frame_slot = scheduler.read_slot("camera_frames")
            self._last_depth_frame_fetch_ms = self._ms_since(fetch_start)
            if not isinstance(frame_slot, dict):
                self._worker_stop.wait(timeout=interval_s)
                continue
            generation = int(frame_slot.get("generation", 0) or 0)
            if generation != self._last_camera_generation:
                self._last_camera_generation = generation
                self._last_camera_seq = 0
            seq = int(frame_slot.get("seq", 0) or 0)
            frames = frame_slot.get("payload")
            if seq <= self._last_camera_seq or not isinstance(frames, dict):
                self._worker_stop.wait(timeout=interval_s)
                continue
            if seq > self._last_camera_seq + 1 and self._last_camera_seq > 0:
                self._dropped_frame_count += int(seq - self._last_camera_seq - 1)
            self._last_camera_seq = seq
            self._frame_id += 1
            frame_capture_ts = self._pick_frame_capture_ts(frame_slot, frames)
            latest_frame_lag_ms = max(0.0, (time.time() - float(frame_capture_ts)) * 1000.0)
            vision_start_ts = time.time()
            self._processing_busy = True
            depth = frames.get("depth")
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
            self._processed_frame_count += 1
            self._last_process_ms = max(0.0, (vision_done_ts - vision_start_ts) * 1000.0)
            payload = self._with_freshness(
                payload,
                frame_capture_ts=frame_capture_ts,
                vision_start_ts=vision_start_ts,
                vision_done_ts=vision_done_ts,
                latest_frame_lag_ms=latest_frame_lag_ms,
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
            "track_local_update_hz": 1.0 / max(1e-6, float(self._track_local_interval_s)),
        }
