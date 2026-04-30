#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from .table_edge_roi import choose_depth_roi


CapabilitySink = Optional[Callable[[str, Dict[str, Any]], None]]


class TableEdgeManager:
    """Own depth-based table-edge perception and publish summarized results."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        capability_sink: CapabilitySink = None,
    ):
        self.log = logger or logging.getLogger("vision.table_edge_manager")
        self._capability_sink = capability_sink
        self._scheduler = None
        self._generation_getter = lambda: 0
        self._runtime_running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_stop = threading.Event()
        edge_hz = float(os.getenv("VISTA_TABLE_EDGE_HZ", "10") or 10.0)
        self._worker_interval_s = 1.0 / max(1.0, edge_hz)
        self._last_camera_seq = 0
        self._last_publish_ts = 0.0
        self._frame_id = 0
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

    def _with_freshness(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(payload or {})
        obs_ts = time.time()
        out["ts"] = float(obs_ts)
        out["obs_ts"] = float(obs_ts)
        out["age_ms"] = 0.0
        out["is_stale"] = False
        out["source_mode"] = self._active_mode()
        out.setdefault("seq", out.get("frame_seq", out.get("frame_id")))
        out.setdefault("edge_conf", out.get("confidence"))
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
        return {
            "table_found": False,
            "edge_found": False,
            "confidence": 0.0,
            "yaw_err_rad": None,
            "dist_err_m": None,
            "edge_k": None,
            "edge_b": None,
            "depth_valid": bool(depth_valid),
            "point_count": 0,
            "table_point_count": 0,
            "frame_id": int(self._frame_id),
            "frame_seq": int(frame_seq),
            "source": "vision_table_edge_manager",
            "reason": str(reason or ""),
            "target_dist_m": float(self._target_dist_m),
            **roi,
            "type": "table_edge_obs",
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

    @staticmethod
    def _manual_static_roi_enabled() -> bool:
        raw = os.getenv("VISTA_TABLE_EDGE_STATIC_ROI", os.getenv("VISTA_FORCE_STATIC_EDGE_ROI", "0"))
        return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}

    def _local_perception(self) -> Dict[str, Any]:
        scheduler = self._scheduler
        if scheduler is None:
            return {}
        try:
            value = scheduler.read_result("local_perception", default={}) or {}
        except Exception:
            return {}
        return dict(value) if isinstance(value, dict) else {}

    def _select_roi(self, depth_frame: Optional[np.ndarray]) -> Dict[str, Any]:
        fallback = self._static_roi()
        local = self._local_perception()
        depth_shape = getattr(depth_frame, "shape", None)
        manual_static = self._manual_static_roi_enabled()
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
            "roi_format": "xyxy",
        }
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
        roi_meta = self._select_roi(depth_frame)
        roi_override = roi_meta.get("depth_edge_roi") if roi_meta.get("roi_source") != "static_fallback" else None
        if self._detector is None:
            return self._default_result(
                depth_valid=True,
                reason=self._detector_error or "detector_unavailable",
                frame_seq=frame_seq,
                roi_meta=roi_meta,
            )
        try:
            result, _debug = self._detector.process_depth(depth_frame, roi_override=roi_override)
        except Exception as exc:
            self.log.debug("table edge detect failed | error=%s", exc)
            return self._default_result(depth_valid=True, reason=f"detect_failed:{exc}", frame_seq=frame_seq, roi_meta=roi_meta)
        roi_box = None
        if isinstance(_debug, dict):
            roi_box = _debug.get("roi_box")
        roi_payload = self._roi_payload(roi_box, roi_meta)
        table_points = int(getattr(result, "table_point_count", 0) or 0)
        all_points = int(getattr(result, "point_count", 0) or 0)
        return {
            "table_found": bool(table_points > 0),
            "edge_found": bool(getattr(result, "edge_found", False)),
            "confidence": float(getattr(result, "edge_confidence", 0.0) or 0.0),
            "yaw_err_rad": float(getattr(result, "yaw_err_rad", 0.0)) if bool(getattr(result, "edge_found", False)) else None,
            "dist_err_m": float(getattr(result, "dist_err_m", 0.0)) if bool(getattr(result, "edge_found", False)) else None,
            "edge_k": getattr(result, "line_k", None),
            "edge_b": getattr(result, "line_b", None),
            "depth_valid": True,
            "point_count": all_points,
            "valid_edge_points": all_points,
            "table_point_count": table_points,
            "edge_inlier_count": table_points,
            "selected_edge": bool(getattr(result, "edge_found", False)),
            "near_edge": bool(getattr(result, "edge_found", False)),
            "frame_id": int(self._frame_id),
            "frame_seq": int(frame_seq),
            "source": "vision_table_edge_manager",
            "target_dist_m": float(self._target_dist_m),
            **roi_payload,
            "type": "table_edge_obs",
        }

    def _worker_loop(self) -> None:
        while self._runtime_running and not self._worker_stop.is_set():
            scheduler = self._scheduler
            if scheduler is None:
                self._worker_stop.wait(timeout=self._worker_interval_s)
                continue
            frame_slot = scheduler.read_slot("camera_frames")
            if not isinstance(frame_slot, dict):
                self._worker_stop.wait(timeout=self._worker_interval_s)
                continue
            seq = int(frame_slot.get("seq", 0) or 0)
            frames = frame_slot.get("payload")
            if seq <= self._last_camera_seq or not isinstance(frames, dict):
                self._worker_stop.wait(timeout=self._worker_interval_s)
                continue
            self._last_camera_seq = seq
            self._frame_id += 1
            depth = frames.get("depth")
            if not isinstance(depth, np.ndarray) or depth.size <= 0:
                payload = self._default_result(
                    depth_valid=False,
                    reason="depth_frame_missing",
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
            self._publish_result("table_edge_obs", self._with_freshness(payload))
            self._worker_stop.wait(timeout=self._worker_interval_s)

    def release_all(self) -> None:
        self.stop_runtime()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "detector_ready": bool(self._detector is not None),
            "detector_error": str(self._detector_error or ""),
            "runtime_running": bool(self._runtime_running),
            "last_camera_seq": int(self._last_camera_seq),
            "last_publish_ts": float(self._last_publish_ts),
            "frame_id": int(self._frame_id),
            "target_dist_m": float(self._target_dist_m),
            "last_valid_quadrant": self._last_valid_quadrant,
        }
