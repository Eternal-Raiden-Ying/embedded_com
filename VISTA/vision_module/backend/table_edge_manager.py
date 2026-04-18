#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np


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
        self._worker_interval_s = 0.05
        self._last_camera_seq = 0
        self._last_publish_ts = 0.0
        self._frame_id = 0
        self._detector = None
        self._detector_cfg = None
        self._detector_error = ""
        self._target_dist_m = 0.5
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

    def _default_result(self, *, depth_valid: bool, reason: str, frame_seq: int) -> Dict[str, Any]:
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
            "type": "table_edge_obs",
        }

    def _process_depth(self, depth_frame: np.ndarray, frame_seq: int) -> Dict[str, Any]:
        if self._detector is None:
            return self._default_result(depth_valid=True, reason=self._detector_error or "detector_unavailable", frame_seq=frame_seq)
        try:
            result, _debug = self._detector.process_depth(depth_frame)
        except Exception as exc:
            self.log.debug("table edge detect failed | error=%s", exc)
            return self._default_result(depth_valid=True, reason=f"detect_failed:{exc}", frame_seq=frame_seq)
        return {
            "table_found": bool(getattr(result, "table_point_count", 0) > 0),
            "edge_found": bool(getattr(result, "edge_found", False)),
            "confidence": float(getattr(result, "edge_confidence", 0.0) or 0.0),
            "yaw_err_rad": float(getattr(result, "yaw_err_rad", 0.0)) if bool(getattr(result, "edge_found", False)) else None,
            "dist_err_m": float(getattr(result, "dist_err_m", 0.0)) if bool(getattr(result, "edge_found", False)) else None,
            "edge_k": getattr(result, "line_k", None),
            "edge_b": getattr(result, "line_b", None),
            "depth_valid": True,
            "point_count": int(getattr(result, "point_count", 0) or 0),
            "table_point_count": int(getattr(result, "table_point_count", 0) or 0),
            "frame_id": int(self._frame_id),
            "frame_seq": int(frame_seq),
            "source": "vision_table_edge_manager",
            "target_dist_m": float(self._target_dist_m),
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
                payload = self._default_result(depth_valid=False, reason="depth_frame_missing", frame_seq=seq)
            elif depth.ndim != 2:
                payload = self._default_result(depth_valid=False, reason="depth_frame_not_2d", frame_seq=seq)
            else:
                payload = self._process_depth(depth, seq)
            self._publish_result("table_edge_obs", payload)
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
        }
