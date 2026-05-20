#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import math
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
        self._default_interval_s = self._worker_interval_s
        self._edge_path = "full"
        self._edge_update_hz = edge_hz
        self._last_camera_generation = 0
        self._last_camera_seq = 0
        self._last_publish_ts = 0.0
        self._last_obs_ts = 0.0
        self._last_depth_frame_fetch_ms = 0.0
        self._last_process_ms = 0.0
        self._last_update_interval_ms = None
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
            scheduler.publish_result(route, payload, generation=generation)
            self._last_publish_ts = time.time()
        except Exception:
            pass

    def configure(self, payload: Dict[str, Any]) -> None:
        path = str(payload.get("path") or "full").strip().lower()
        self._edge_path = path if path in {"lightweight", "full"} else "full"
        self._edge_update_hz = float(payload.get("update_hz", self._edge_update_hz) or self._edge_update_hz)
        if self._edge_update_hz > 0:
            self._worker_interval_s = 1.0 / max(1.0, self._edge_update_hz)

    def _with_freshness(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(payload or {})
        obs_ts = time.time()
        previous_obs_ts = float(self._last_obs_ts or 0.0)
        update_interval_ms = ((obs_ts - previous_obs_ts) * 1000.0) if previous_obs_ts > 0.0 else None
        self._last_obs_ts = float(obs_ts)
        self._last_update_interval_ms = update_interval_ms
        out["ts"] = float(obs_ts)
        out["obs_ts"] = float(obs_ts)
        out["age_ms"] = 0.0
        out["edge_update_interval_ms"] = update_interval_ms
        out["edge_process_ms"] = float(self._last_process_ms)
        out["total_edge_process_ms"] = float(self._last_process_ms)
        out.setdefault("depth_frame_fetch_ms", float(self._last_depth_frame_fetch_ms))
        unavailable = bool(out.get("edge_obs_unavailable", False))
        out["is_stale"] = bool(out.get("is_stale", False) or unavailable)
        out["source_mode"] = self._edge_path
        out.setdefault("seq", out.get("frame_seq", out.get("frame_id")))
        out.setdefault("frame_id", out.get("frame_seq", out.get("seq")))
        out.setdefault("edge_conf", out.get("confidence"))
        out.setdefault("edge_valid", bool(out.get("edge_found", False)) and not unavailable)
        out.setdefault("yaw_err", out.get("yaw_err_rad"))
        out.setdefault("dist_err", out.get("dist_err_m"))
        return out

    @staticmethod
    def _ms_since(start_ts: float) -> float:
        return max(0.0, (time.perf_counter() - float(start_ts)) * 1000.0)

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
        if self._edge_path == "lightweight":
            return self._process_depth_lightweight(depth_frame, frame_seq)
        total_start = time.perf_counter()
        profile = self._profile_template()
        profile["depth_frame_fetch_ms"] = float(self._last_depth_frame_fetch_ms)
        roi_meta = self._select_roi(depth_frame)
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
        if isinstance(_debug, dict):
            roi_box = _debug.get("roi_box")
        roi_payload = self._roi_payload(roi_box, roi_meta)
        table_points = int(getattr(result, "table_point_count", 0) or 0)
        all_points = int(getattr(result, "point_count", 0) or 0)
        edge_found = bool(getattr(result, "edge_found", False))
        reason = ""
        if not edge_found:
            reason = "roi_empty" if all_points <= 0 and table_points <= 0 else "no_valid_edge"
        yaw_err = float(getattr(result, "yaw_err_rad", 0.0)) if edge_found else None
        dist_err = float(getattr(result, "dist_err_m", 0.0)) if edge_found else None
        edge_conf = float(getattr(result, "edge_confidence", 0.0) or 0.0)
        payload = {
            "table_found": bool(table_points > 0),
            "edge_found": edge_found,
            "edge_valid": edge_found,
            "confidence": edge_conf,
            "edge_conf": edge_conf,
            "yaw_err_rad": yaw_err,
            "yaw_err": yaw_err,
            "dist_err_m": dist_err,
            "dist_err": dist_err,
            "edge_k": getattr(result, "line_k", None),
            "edge_b": getattr(result, "line_b", None),
            "depth_valid": True,
            "edge_obs_unavailable": False,
            "point_count": all_points,
            "valid_edge_points": all_points,
            "table_point_count": table_points,
            "edge_inlier_count": table_points,
            "selected_edge": edge_found,
            "near_edge": edge_found,
            "frame_id": int(self._frame_id),
            "frame_seq": int(frame_seq),
            "source": "vision_table_edge_manager",
            "reason": reason,
            "target_dist_m": float(self._target_dist_m),
            **roi_payload,
            "type": "table_edge_obs",
        }
        profile["total_edge_process_ms"] = self._ms_since(total_start)
        return self._attach_profile(payload, profile, path="full")

    def _process_depth_lightweight(self, depth_frame: np.ndarray, frame_seq: int) -> Dict[str, Any]:
        total_start = time.perf_counter()
        profile = self._profile_template()
        profile["depth_frame_fetch_ms"] = float(self._last_depth_frame_fetch_ms)
        roi_select_start = time.perf_counter()
        roi_meta = self._select_roi(depth_frame)
        roi_box = roi_meta.get("depth_edge_roi") if roi_meta.get("roi_source") != "static_fallback" else self._static_roi()
        if roi_box is None:
            roi_box = self._static_roi()
        try:
            roi_box = self._detector._resolve_roi(depth_frame, roi_override=roi_box) if self._detector is not None else tuple(int(v) for v in roi_box)
        except Exception:
            roi_box = tuple(int(v) for v in self._static_roi() or (0, 0, depth_frame.shape[1], depth_frame.shape[0]))
        x0, y0, x1, y1 = [int(v) for v in roi_box]
        stride = max(1, 4)
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
        except Exception as exc:
            payload = self._default_result(depth_valid=True, reason=f"light_fit_failed:{exc}", frame_seq=frame_seq, roi_meta=roi_meta)
            payload.update(roi_payload)
            payload["point_count"] = int(valid_count)
            payload["table_point_count"] = int(table_count)
            profile["plane_or_edge_fit_ms"] = self._ms_since(fit_start)
            profile["total_edge_process_ms"] = self._ms_since(total_start)
            return self._attach_profile(payload, profile, path="light_fit_failed")
        profile["plane_or_edge_fit_ms"] = self._ms_since(fit_start)

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
            "frame_id": int(self._frame_id),
            "frame_seq": int(frame_seq),
            "source": "vision_table_edge_manager",
            "reason": "" if edge_found else "no_valid_edge",
            "target_dist_m": float(self._target_dist_m),
            "lightweight": True,
            "sample_stride": int(stride),
            **roi_payload,
            "type": "table_edge_obs",
        }
        profile["total_edge_process_ms"] = self._ms_since(total_start)
        return self._attach_profile(payload, profile, path="light")

    def _worker_loop(self) -> None:
        while self._runtime_running and not self._worker_stop.is_set():
            loop_start = time.time()
            scheduler = self._scheduler
            interval_s = self._worker_interval_s
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
            self._last_camera_seq = seq
            self._frame_id += 1
            depth = frames.get("depth")
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
            self._last_process_ms = max(0.0, (time.time() - loop_start) * 1000.0)
            self._publish_result("table_edge_obs", self._with_freshness(payload))
            elapsed_s = max(0.0, time.time() - loop_start)
            self._worker_stop.wait(timeout=max(0.0, interval_s - elapsed_s))

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
            "target_dist_m": float(self._target_dist_m),
            "last_valid_quadrant": self._last_valid_quadrant,
            "default_update_hz": 1.0 / max(1e-6, float(self._default_interval_s)),
            "edge_update_hz": self._edge_update_hz,
        }
