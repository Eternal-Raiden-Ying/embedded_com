#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

try:
    import aidcv as cv2
except ImportError:
    try:
        import cv2
    except ImportError:
        cv2 = None  # type: ignore

from .client import RemoteGraspClient
from .protocol import (
    DEFAULT_REMOTE_ROBOT_ID,
    RemoteMetadata,
    RemotePredictRequest,
    RemotePredictResponse,
    image_encoding_info,
    normalize_image_encoding,
)


def _cfg_bool(cfg: Dict[str, Any], key: str, default: bool) -> bool:
    return bool(dict(cfg or {}).get(key, default))


def _cfg_float(cfg: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float(dict(cfg or {}).get(key, default))
    except Exception:
        return float(default)


def prepare_remote_rgb(rgb, cfg: Dict[str, Any]):
    """Prepare an HxWx3 RGB uint8 frame for remote detection/upload."""
    arr = np.asarray(rgb)
    input_dtype = str(getattr(arr.dtype, "name", arr.dtype))
    info: Dict[str, Any] = {
        "rgb_correction_enable": _cfg_bool(cfg, "remote_rgb_correction_enable", True),
        "rgb_correction_mode": str(dict(cfg or {}).get("remote_rgb_correction_mode") or "auto_level_gamma"),
        "rgb_gamma": _cfg_float(cfg, "remote_rgb_gamma", 1.8),
        "rgb_percentile_low": _cfg_float(cfg, "remote_rgb_percentile_low", 1.0),
        "rgb_percentile_high": _cfg_float(cfg, "remote_rgb_percentile_high", 99.0),
        "rgb_input_dtype": input_dtype,
        "rgb_gain_applied": 1.0,
    }
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"remote rgb must be HxWx3, got shape={getattr(arr, 'shape', None)}")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            finite = np.nan_to_num(arr[:, :, :3], nan=0.0, posinf=255.0, neginf=0.0)
            if float(np.nanmax(finite)) <= 1.0:
                finite = finite * 255.0
            arr_u8 = np.clip(finite, 0.0, 255.0).astype(np.uint8)
        else:
            arr_u8 = np.clip(arr[:, :, :3], 0, 255).astype(np.uint8)
    else:
        arr_u8 = arr[:, :, :3].copy()

    raw_mean = float(np.mean(arr_u8)) if arr_u8.size else 0.0
    info["rgb_raw_mean"] = raw_mean
    if cv2 is None or not info["rgb_correction_enable"] or info["rgb_correction_mode"] != "auto_level_gamma":
        info.update(
            {
                "rgb_y_p_low": None,
                "rgb_y_p50": None,
                "rgb_y_p_high": None,
                "rgb_corrected_mean": raw_mean,
            }
        )
        return arr_u8, info

    ycc = cv2.cvtColor(arr_u8, cv2.COLOR_RGB2YCrCb)
    y = ycc[:, :, 0].astype(np.float32)
    p_low = max(0.0, min(100.0, float(info["rgb_percentile_low"])))
    p_high = max(0.0, min(100.0, float(info["rgb_percentile_high"])))
    if p_high <= p_low:
        p_low, p_high = 1.0, 99.0
    y_low = float(np.percentile(y, p_low))
    y_p50 = float(np.percentile(y, 50.0))
    y_high = float(np.percentile(y, p_high))
    info.update(
        {
            "rgb_y_p_low": y_low,
            "rgb_y_p50": y_p50,
            "rgb_y_p_high": y_high,
        }
    )
    raw_bright_enough = bool(raw_mean >= 115.0 or y_p50 >= 125.0 or y_high >= 245.0)
    if raw_bright_enough or y_high - y_low < 8.0:
        corrected = arr_u8
    else:
        y_norm = y / 255.0
        stretched = np.clip((y - y_low) / max(1.0, y_high - y_low), 0.0, 1.0)
        gamma = max(1.0, float(info["rgb_gamma"]))
        corrected_y = np.power(stretched, 1.0 / gamma)
        max_gain = max(1.0, _cfg_float(cfg, "remote_rgb_max_gain", 3.0))
        corrected_y = np.minimum(corrected_y, np.clip(y_norm * max_gain, 0.0, 1.0))
        y_out = np.clip(np.round(corrected_y * 255.0), 0.0, 255.0).astype(np.uint8)
        ycc[:, :, 0] = y_out
        corrected = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)
        raw_y_mean = float(np.mean(y)) if y.size else 0.0
        corrected_y_mean = float(np.mean(y_out)) if y_out.size else raw_y_mean
        if raw_y_mean > 1e-6:
            info["rgb_gain_applied"] = min(max_gain, corrected_y_mean / raw_y_mean)
    info["rgb_corrected_mean"] = float(np.mean(corrected)) if corrected.size else raw_mean
    return corrected, info


class RemoteManager:
    """Own remote client lifecycle and remote request orchestration."""

    def __init__(
        self,
        client: Optional[RemoteGraspClient] = None,
        logger=None,
        capability_sink: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        archive_root: Optional[str] = None,
        archive_enable: bool = True,
        archive_max_keep: int = 20,
        run_id: str = "",
        rgb_correction_config: Optional[Dict[str, Any]] = None,
    ):
        self.client = client
        self.logger = logger
        self._capability_sink = capability_sink
        self._archive_root = Path(archive_root) if archive_root else None
        self._archive_enable = bool(archive_enable)
        self._archive_max_keep = max(0, int(archive_max_keep or 0))
        self._run_id = str(run_id or "")
        self._rgb_correction_config = {
            "remote_rgb_correction_enable": True,
            "remote_rgb_correction_mode": "auto_level_gamma",
            "remote_rgb_gamma": 1.8,
            "remote_rgb_percentile_low": 1.0,
            "remote_rgb_percentile_high": 99.0,
            "remote_rgb_max_gain": 3.0,
            "remote_rgb_save_raw": True,
        }
        self._rgb_correction_config.update(dict(rgb_correction_config or {}))
        self.enabled = False
        self._scheduler = None
        self._generation_getter = lambda: 0
        self._runtime_running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_stop = threading.Event()
        self._worker_interval_s = 0.05
        self._sequence = 0
        self._runtime_profile: Dict[str, Any] = {
            "kind": "loop",
            "action": "",
            "max_retries": 1,
            "base_url": None,
            "command": "predict",
            "require_depth": False,
            "timeout_s": 10.0,
            "robot_id": DEFAULT_REMOTE_ROBOT_ID,
            "metadata": {},
            "rgb_encoding": "jpeg",
            "depth_encoding": "png",
            "rgb_quality": 90,
            "depth_compression": 3,
            "capture_warmup_frames": 5,
            "capture_warmup_timeout_s": 1.0,
            "expected_rgb_shape": [720, 1280],
            "expected_depth_shape": [720, 1280],
            **self._rgb_correction_config,
        }
        self._service_init_state = "uninitialized"
        self._service_init_confirmed = False
        self._service_init_attempts = 0
        self._service_init_last_error = ""
        self._service_init_last_ok = False
        self._service_init_last_ts: Optional[float] = None
        self._service_init_pending = False
        self._service_init_inflight = False
        self._warmup_keepalive_until = 0.0
        self._last_result: Dict[str, Any] = {
            "enabled": False,
            "state": "disabled",
            "last_action": "init",
            "last_ok": True,
            "last_error": "",
            "status_code": None,
            "has_result": False,
            "result": None,
            "request_id": None,
            "sequence": 0,
            "ts": 0.0,
            "init_confirmed": False,
            "service_init_state": "uninitialized",
            "service_init_confirmed": False,
            "service_init_attempts": 0,
            "service_init_last_error": "",
            "service_init_last_ok": False,
            "service_init_last_ts": None,
        }
        self._latest_grasp_remote_context: Dict[str, Any] = {}

    def _emit(self, action: str, **fields: Any) -> None:
        if self._capability_sink is None:
            return
        try:
            payload = {"action": str(action or "updated").strip().lower()}
            payload.update(dict(fields or {}))
            self._capability_sink("remote", payload)
        except Exception:
            pass

    def _log(self, level: str, message: str, **fields: Any) -> None:
        if self.logger is None:
            return
        extra = fields or None
        text = message if not extra else f"{message} | {extra}"
        fn = getattr(self.logger, level, None)
        if callable(fn):
            fn(text)

    def _service_has_base_url(self) -> bool:
        return bool(str(self._runtime_profile.get("base_url") or "").strip())

    def _reset_service_init_state(
        self,
        *,
        state: str = "uninitialized",
        confirmed: bool = False,
        attempts: int = 0,
        last_error: str = "",
        last_ok: bool = False,
        last_ts: Optional[float] = None,
        pending: bool = False,
    ) -> None:
        self._service_init_state = str(state or "uninitialized")
        self._service_init_confirmed = bool(confirmed)
        self._service_init_attempts = int(attempts or 0)
        self._service_init_last_error = str(last_error or "")
        self._service_init_last_ok = bool(last_ok)
        self._service_init_last_ts = last_ts
        self._service_init_pending = bool(pending)
        self._service_init_inflight = False

    def _service_init_fields(self) -> Dict[str, Any]:
        return {
            "init_confirmed": bool(self._service_init_confirmed),
            "service_init_state": str(self._service_init_state or "uninitialized"),
            "service_init_confirmed": bool(self._service_init_confirmed),
            "service_init_attempts": int(self._service_init_attempts),
            "service_init_last_error": str(self._service_init_last_error or ""),
            "service_init_last_ok": bool(self._service_init_last_ok),
            "service_init_last_ts": self._service_init_last_ts,
        }

    def _schedule_service_init(self) -> None:
        if not self.enabled or not self._service_has_base_url():
            return
        if self._service_init_confirmed or self._service_init_inflight:
            return
        self._service_init_pending = True

    def _effective_runtime_field(self, cmd: Dict[str, Any], key: str, default=None):
        if key in cmd and cmd.get(key) is not None:
            return cmd.get(key)
        if key in self._runtime_profile:
            return self._runtime_profile.get(key, default)
        return default

    def _context_from_runtime_status(self, status: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(status, dict):
            return {}
        metadata = dict(status.get("remote_metadata") or {}) if isinstance(status.get("remote_metadata"), dict) else {}
        class_id = status.get("class_id", status.get("remote_class_id", metadata.get("class_id")))
        context = {
            "class_id": class_id,
            "target": status.get("target", metadata.get("target")),
            "request_id": status.get("request_id") or status.get("req_id") or status.get("remote_request_id") or metadata.get("request_id"),
            "session_id": status.get("session_id", metadata.get("session_id")),
            "robot_id": status.get("robot_id") or status.get("remote_robot_id") or metadata.get("robot_id"),
            "need_depth": status.get("need_depth", metadata.get("need_depth")),
            "timeout_s": status.get("timeout_s") or status.get("remote_timeout_s"),
            "metadata": metadata,
        }
        return {key: value for key, value in context.items() if value is not None}

    def _update_latest_grasp_remote_context(self, context: Dict[str, Any], *, source: str) -> Dict[str, Any]:
        clean = {key: value for key, value in dict(context or {}).items() if value is not None}
        if clean:
            clean["context_source"] = str(source or "unknown")
            merged = dict(self._latest_grasp_remote_context or {})
            merged.update(clean)
            self._latest_grasp_remote_context = merged
        return dict(self._latest_grasp_remote_context or {})

    def _wait_runtime_status_context(self, timeout_s: float) -> Dict[str, Any]:
        deadline = time.time() + max(0.0, float(timeout_s))
        last_status: Dict[str, Any] = {}
        while time.time() <= deadline:
            status = self._runtime_status_payload()
            if status:
                last_status = status
                context = self._context_from_runtime_status(status)
                self._update_latest_grasp_remote_context(context, source="runtime_status")
                if self._resolve_class_id(context.get("class_id")) is not None:
                    return status
            if timeout_s <= 0.0:
                break
            self._worker_stop.wait(timeout=0.05)
        return last_status

    def _predict_context(self, runtime_status: Dict[str, Any]) -> Dict[str, Any]:
        profile_metadata = dict(self._runtime_profile.get("metadata") or {})
        context: Dict[str, Any] = {}
        context.update(profile_metadata)
        context.update(dict(self._latest_grasp_remote_context or {}))
        runtime_context = self._context_from_runtime_status(runtime_status)
        context.update(runtime_context)
        self._update_latest_grasp_remote_context(runtime_context, source="runtime_status")
        metadata = {}
        if isinstance(profile_metadata, dict):
            metadata.update(profile_metadata)
        latest_metadata = self._latest_grasp_remote_context.get("metadata")
        if isinstance(latest_metadata, dict):
            metadata.update(latest_metadata)
        runtime_metadata = runtime_context.get("metadata")
        if isinstance(runtime_metadata, dict):
            metadata.update(runtime_metadata)
        metadata.setdefault("target", context.get("target"))
        metadata.setdefault("request_id", context.get("request_id"))
        metadata.setdefault("session_id", context.get("session_id"))
        context["metadata"] = metadata
        return context

    def set_client(self, client: RemoteGraspClient) -> None:
        self.client = client

    def configure_runtime(self, payload: Dict[str, Any]) -> None:
        previous_base_url = str(self._runtime_profile.get("base_url") or "").strip()
        profile = dict(payload or {})
        next_profile = dict(self._runtime_profile)
        next_profile.update(
            {
                "kind": str(profile.get("kind") or "loop").strip().lower() or "loop",
                "action": str(profile.get("action") or "").strip().lower(),
                "max_retries": max(1, int(profile.get("max_retries", 1) or 1)),
                "base_url": str(profile.get("base_url") or "").strip() or None,
                "command": str(profile.get("command") or "predict").strip() or "predict",
                "require_depth": bool(profile.get("require_depth", False)),
                "timeout_s": float(profile.get("timeout_s", 10.0) or 10.0),
                "robot_id": str(profile.get("robot_id") or "").strip() or DEFAULT_REMOTE_ROBOT_ID,
                "metadata": dict(profile.get("metadata") or {}) if isinstance(profile.get("metadata"), dict) else {},
                "rgb_encoding": normalize_image_encoding(profile.get("rgb_encoding"), default="jpeg"),
                "depth_encoding": normalize_image_encoding(profile.get("depth_encoding"), default="png"),
                "rgb_quality": int(profile.get("rgb_quality", 90) or 90),
                "depth_compression": int(profile.get("depth_compression", 3) or 3),
                "capture_warmup_frames": max(1, int(profile.get("capture_warmup_frames", 5) or 5)),
                "capture_warmup_timeout_s": max(0.1, float(profile.get("capture_warmup_timeout_s", 1.0) or 1.0)),
                "expected_rgb_shape": self._shape_hw_list(profile.get("expected_rgb_shape"), default=[720, 1280]),
                "expected_depth_shape": self._shape_hw_list(profile.get("expected_depth_shape"), default=[720, 1280]),
                "remote_rgb_correction_enable": bool(profile.get("remote_rgb_correction_enable", next_profile.get("remote_rgb_correction_enable", True))),
                "remote_rgb_correction_mode": str(profile.get("remote_rgb_correction_mode", next_profile.get("remote_rgb_correction_mode", "auto_level_gamma")) or "auto_level_gamma"),
                "remote_rgb_gamma": float(profile.get("remote_rgb_gamma", next_profile.get("remote_rgb_gamma", 1.8)) or 1.8),
                "remote_rgb_percentile_low": float(profile.get("remote_rgb_percentile_low", next_profile.get("remote_rgb_percentile_low", 1.0)) or 1.0),
                "remote_rgb_percentile_high": float(profile.get("remote_rgb_percentile_high", next_profile.get("remote_rgb_percentile_high", 99.0)) or 99.0),
                "remote_rgb_max_gain": float(profile.get("remote_rgb_max_gain", next_profile.get("remote_rgb_max_gain", 3.0)) or 3.0),
                "remote_rgb_save_raw": bool(profile.get("remote_rgb_save_raw", next_profile.get("remote_rgb_save_raw", True))),
            }
        )
        self._runtime_profile = next_profile
        next_base_url = str(next_profile.get("base_url") or "").strip()
        if self.client is not None and next_profile.get("base_url"):
            try:
                self.client.configure(next_profile["base_url"])
            except Exception:
                pass
        if not next_base_url:
            self._reset_service_init_state()
        elif next_base_url != previous_base_url:
            self._reset_service_init_state(pending=self.enabled)
        elif self.enabled and not self._service_init_confirmed and self._service_init_attempts <= 0:
            self._schedule_service_init()

    def bind_runtime(self, scheduler, generation_getter=None) -> None:
        self._scheduler = scheduler
        if callable(generation_getter):
            self._generation_getter = generation_getter

    def start_runtime(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._runtime_running = True
        self._worker_stop.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, name="remote_manager.loop", daemon=True)
        self._worker_thread.start()

    def start_init_warmup(self, *, min_interval_s: float = 30.0, source: str = "task_start_warmup") -> Dict[str, Any]:
        now = time.time()
        min_interval_s = max(0.0, float(min_interval_s or 0.0))
        base_url = self._runtime_base_url()
        remote_request_id = f"ri_{int(now * 1000)}"
        if self._service_init_confirmed and self._service_init_last_ts is not None and now - float(self._service_init_last_ts) < min_interval_s:
            payload = {
                "op": "INIT",
                "ok": True,
                "reason": "already_ready",
                "base_url": base_url,
                "endpoint": "/api/v1/init",
                "elapsed_ms": 0,
                "source": source,
                "request_id": remote_request_id,
            }
            self._log("info", "[GRASP_REMOTE][INIT_SKIP] already_ready", base_url=base_url, age_s=now - float(self._service_init_last_ts))
            self._publish_result("remote_init_status", {**self._task_payload("init"), **payload})
            return payload
        if self._service_init_inflight:
            payload = {
                "op": "INIT",
                "ok": False,
                "reason": "already_inflight",
                "base_url": base_url,
                "endpoint": "/api/v1/init",
                "elapsed_ms": 0,
                "source": source,
                "request_id": remote_request_id,
            }
            self._log("info", "[GRASP_REMOTE][INIT_SKIP] already_inflight", base_url=base_url)
            return payload

        def _runner() -> None:
            self._warmup_keepalive_until = time.time() + max(5.0, min_interval_s)
            self._log("info", "[GRASP_REMOTE][INIT_WARMUP] start", base_url=self._runtime_base_url(), request_id=remote_request_id)
            result = self._run_service_init(timeout_s=float(self._runtime_profile.get("timeout_s", 10.0) or 10.0), source=source, request_id=remote_request_id)
            payload = {**self._task_payload("init"), **dict(result or {})}
            self._publish_result("remote_init_status", payload)
            if bool(result.get("ok")):
                self._log("info", "[GRASP_REMOTE][INIT_READY]", base_url=self._runtime_base_url(), elapsed_ms=result.get("elapsed_ms"), request_id=remote_request_id)
            else:
                self._log("error", "[GRASP_REMOTE][INIT_FAILED]", base_url=self._runtime_base_url(), status_code=result.get("status_code"), elapsed_ms=result.get("elapsed_ms"), error_message=result.get("error_message") or result.get("reason"), request_id=remote_request_id)

        thread = threading.Thread(target=_runner, name="remote.init_warmup", daemon=True)
        thread.start()
        return {
            "op": "INIT",
            "ok": None,
            "reason": "started",
            "base_url": base_url,
            "endpoint": "/api/v1/init",
            "source": source,
            "request_id": remote_request_id,
        }

    def keep_remote_warm(self) -> bool:
        return bool(self._service_init_inflight or (self._service_init_confirmed and time.time() < float(self._warmup_keepalive_until or 0.0)))

    def stop_runtime(self) -> None:
        self._runtime_running = False
        self._worker_stop.set()
        thread = self._worker_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
            if thread.is_alive():
                self._log("warn", "remote task worker still alive after 1s join, orphaning")
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
        except Exception as exc:
            if str(route) == "remote_init_status":
                fallback = dict(payload or {}) if isinstance(payload, dict) else {"payload": payload}
                fallback["route_missing"] = True
                fallback["route_error"] = str(exc)
                try:
                    scheduler.publish_result("remote_result", fallback, generation=generation)
                except Exception:
                    pass

    def _resolve_class_id(self, explicit_class_id: Any = None) -> Optional[int]:
        if explicit_class_id is None:
            return None
        try:
            return int(explicit_class_id)
        except Exception:
            return None

    def _encode_frame(self, encoding: str, frame, *, quality: int = 90, compression: int = 3) -> Optional[bytes]:
        if frame is None:
            return None
        if cv2 is None:
            return None
        ext, _ = image_encoding_info(encoding, default="png")
        params = []
        if ext == ".jpg":
            params = [int(cv2.IMWRITE_JPEG_QUALITY), max(0, min(100, int(quality)))]
        elif ext == ".png":
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), max(0, min(9, int(compression)))]
        try:
            ok, encoded = cv2.imencode(ext, frame, params)
        except Exception:
            return None
        if not ok:
            return None
        try:
            return encoded.tobytes()
        except Exception:
            return None

    def _encode_rgb_frame(self, encoding: str, rgb, *, quality: int = 90) -> Optional[bytes]:
        if rgb is None:
            return None
        arr = np.asarray(rgb)
        if cv2 is not None and arr.ndim == 3 and arr.shape[2] >= 3:
            try:
                return self._encode_frame(encoding, cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR), quality=quality)
            except Exception:
                return None
        return self._encode_frame(encoding, rgb, quality=quality)

    @staticmethod
    def _remote_rgb_cfg_from_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "remote_rgb_correction_enable": bool(profile.get("remote_rgb_correction_enable", True)),
            "remote_rgb_correction_mode": str(profile.get("remote_rgb_correction_mode") or "auto_level_gamma"),
            "remote_rgb_gamma": float(profile.get("remote_rgb_gamma", 1.8) or 1.8),
            "remote_rgb_percentile_low": float(profile.get("remote_rgb_percentile_low", 1.0) or 1.0),
            "remote_rgb_percentile_high": float(profile.get("remote_rgb_percentile_high", 99.0) or 99.0),
            "remote_rgb_max_gain": float(profile.get("remote_rgb_max_gain", 3.0) or 3.0),
            "remote_rgb_save_raw": bool(profile.get("remote_rgb_save_raw", True)),
        }

    @staticmethod
    def _rgb_upload_order(frames: Dict[str, Any]) -> str:
        explicit = str((frames or {}).get("rgb_channel_order") or "").strip().upper()
        if explicit in {"RGB", "BGR"}:
            return explicit
        return "BGR"

    @staticmethod
    def _frame_to_rgb(frame, channel_order: str):
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[2] >= 3 and str(channel_order).upper() == "BGR" and cv2 is not None:
            return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)
        return arr[:, :, :3].copy() if arr.ndim == 3 and arr.shape[2] >= 3 else arr

    @staticmethod
    def _shape_hw_list(value: Any, default=None):
        if isinstance(value, str):
            parts = [part.strip() for part in value.replace("x", ",").replace("X", ",").split(",") if part.strip()]
        elif isinstance(value, (list, tuple)):
            parts = list(value)
        else:
            parts = []
        if len(parts) >= 2:
            try:
                shape = [int(parts[0]), int(parts[1])]
                if shape[0] > 0 and shape[1] > 0:
                    return shape
            except Exception:
                pass
        return list(default or []) if default is not None else None

    @staticmethod
    def _frame_shape(frame) -> Optional[list]:
        shape = getattr(frame, "shape", None)
        if not isinstance(shape, tuple) or len(shape) < 2:
            return None
        try:
            return [int(v) for v in shape]
        except Exception:
            return None

    @staticmethod
    def _resize_frame_hw(frame, shape_hw, *, nearest: bool = False):
        if frame is None or cv2 is None:
            return frame
        shape = RemoteManager._shape_hw_list(shape_hw, default=None)
        if not shape:
            return frame
        current = RemoteManager._frame_shape(frame)
        if current is not None and current[:2] == shape[:2]:
            return frame
        try:
            interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
            return cv2.resize(frame, (int(shape[1]), int(shape[0])), interpolation=interpolation)
        except Exception:
            return frame

    @staticmethod
    def _depth_stats(depth) -> Dict[str, Any]:
        if depth is None or not hasattr(depth, "size") or int(depth.size) <= 0:
            return {
                "depth_min": None,
                "depth_max": None,
                "depth_valid_count": 0,
                "depth_valid_ratio": 0.0,
            }
        arr = np.asarray(depth)
        valid = np.isfinite(arr)
        if np.issubdtype(arr.dtype, np.integer):
            valid = valid & (arr > 0)
        valid_count = int(np.count_nonzero(valid))
        total = int(arr.size)
        if valid_count <= 0:
            return {
                "depth_min": None,
                "depth_max": None,
                "depth_valid_count": 0,
                "depth_valid_ratio": 0.0,
            }
        valid_values = arr[valid]
        return {
            "depth_min": float(np.min(valid_values)),
            "depth_max": float(np.max(valid_values)),
            "depth_valid_count": valid_count,
            "depth_valid_ratio": float(valid_count) / float(total or 1),
        }

    def _capture_metadata(self, frames: Dict[str, Any], frame_slot: Dict[str, Any]) -> Dict[str, Any]:
        rgb = frames.get("rgb")
        depth = frames.get("depth")
        frame_seq, frame_seq_source = self._frame_seq_from_slot(frame_slot, frames)
        timestamp_ms = frames.get("camera_frame_ts_ms")
        if timestamp_ms is None:
            timestamp_ms = int(round(float(frames.get("frame_capture_ts") or time.time()) * 1000.0))
        capture = {
            "rgb_shape": self._frame_shape(rgb),
            "depth_shape": self._frame_shape(depth),
            "rgb_dtype": str(getattr(getattr(rgb, "dtype", None), "name", getattr(rgb, "dtype", "")) or ""),
            "depth_dtype": str(getattr(getattr(depth, "dtype", None), "name", getattr(depth, "dtype", "")) or ""),
            "depth_unit": str(frames.get("depth_unit") or "raw_uint16"),
            "depth_scale": frames.get("depth_scale"),
            "depth_aligned_to_color": frames.get("depth_aligned_to_color", "best_effort_true"),
            "color_intrinsics": frames.get("color_intrinsics"),
            "depth_intrinsics": frames.get("depth_intrinsics"),
            "frame_seq": int(frame_seq),
            "frame_seq_source": str(frame_seq_source),
            "timestamp_ms": int(timestamp_ms),
        }
        capture.update(self._depth_stats(depth))
        return capture

    def _archive_predict_payload(
        self,
        request: RemotePredictRequest,
        frames: Dict[str, Any],
    ) -> Optional[Path]:
        if not self._archive_enable or self._archive_root is None or cv2 is None:
            return None
        request_id = str(request.metadata.request_id or f"rr_{int(time.time() * 1000)}")
        archive_dir = self._archive_root / request_id
        try:
            archive_dir.mkdir(parents=True, exist_ok=True)
            metadata = dict(request.metadata.to_metadata_payload())
            capture = dict(metadata.get("capture") or {})
            metadata.update(
                {
                    "run_id": self._run_id,
                    "request_id": request_id,
                    "rgb_shape": capture.get("rgb_shape"),
                    "rgb_dtype": capture.get("rgb_dtype"),
                    "depth_shape": capture.get("depth_shape"),
                    "depth_dtype": capture.get("depth_dtype"),
                    "depth_scale": capture.get("depth_scale"),
                    "depth_intrinsics": capture.get("depth_intrinsics"),
                    "url": f"{self._runtime_base_url().rstrip('/')}/api/v1/predict",
                }
            )
            (archive_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            if bool(metadata.get("remote_rgb_save_raw", True)) and request.rgb_raw_bytes is not None:
                (archive_dir / "rgb_raw.jpg").write_bytes(request.rgb_raw_bytes)
            if request.rgb_bytes is not None:
                (archive_dir / "rgb.jpg").write_bytes(request.rgb_bytes)
            depth = frames.get("depth") if isinstance(frames, dict) else None
            if depth is not None:
                cv2.imwrite(str(archive_dir / "depth.png"), depth)
            self._prune_payload_archive()
            return archive_dir
        except Exception as exc:
            self._log("warn", "remote_payload_archive_failed", request_id=request_id, error=str(exc))
            return None

    def _archive_predict_response(self, archive_dir: Optional[Path], response: Optional[RemotePredictResponse]) -> None:
        if archive_dir is None:
            return
        payload = {
            "ok": bool(response is not None and response.ok),
            "status_code": getattr(response, "status_code", None),
            "elapsed_ms": getattr(response, "elapsed_ms", None),
            "error": str(getattr(response, "error", "") or ""),
            "payload": getattr(response, "payload", None),
        }
        try:
            (archive_dir / "response.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            self._log("warn", "remote_payload_response_archive_failed", archive_dir=str(archive_dir), error=str(exc))

    def _prune_payload_archive(self) -> None:
        if self._archive_root is None or self._archive_max_keep <= 0:
            return
        try:
            dirs = [path for path in self._archive_root.iterdir() if path.is_dir()]
            dirs.sort(key=lambda path: path.stat().st_mtime)
            for old in dirs[:-self._archive_max_keep]:
                shutil.rmtree(old, ignore_errors=True)
        except Exception as exc:
            self._log("warn", "remote_payload_archive_prune_failed", error=str(exc))

    def _precheck_capture_shapes(
        self,
        *,
        frames: Dict[str, Any],
        frame_slot: Dict[str, Any],
        request_id: str,
        require_depth: bool,
    ) -> Optional[Dict[str, Any]]:
        capture = self._capture_metadata(frames, frame_slot)
        expected_rgb = self._shape_hw_list(self._runtime_profile.get("expected_rgb_shape"), default=[720, 1280])
        expected_depth = self._shape_hw_list(self._runtime_profile.get("expected_depth_shape"), default=[720, 1280])
        actual_rgb = (capture.get("rgb_shape") or [])[:2]
        actual_depth = (capture.get("depth_shape") or [])[:2]
        mismatch = bool(expected_rgb and actual_rgb != expected_rgb)
        mismatch = mismatch or bool(require_depth and expected_depth and actual_depth != expected_depth)
        if not mismatch:
            return capture
        detail = {
            "reason": "capture_shape_mismatch",
            "remote_error": "capture_shape_mismatch",
            "expected_rgb_shape": expected_rgb,
            "actual_rgb_shape": actual_rgb,
            "expected_depth_shape": expected_depth,
            "actual_depth_shape": actual_depth,
            "capture": capture,
            "request_id": request_id,
        }
        self._log("error", "remote_predict_precheck_failed", **detail)
        self._update_result(
            action="predict",
            state="predict_failed",
            ok=False,
            error="capture_shape_mismatch",
            result=detail,
            request_id=request_id,
        )
        return None

    def _frame_seq_from_slot(self, frame_slot: Dict[str, Any], frames: Dict[str, Any]) -> tuple:
        candidates = []
        if isinstance(frame_slot, dict):
            candidates.extend(
                [
                    ("slot.seq", frame_slot.get("seq")),
                    ("slot.frame_seq", frame_slot.get("frame_seq")),
                    ("slot.camera_frame_seq", frame_slot.get("camera_frame_seq")),
                ]
            )
            payload = frame_slot.get("payload")
            if isinstance(payload, dict):
                candidates.extend(
                    [
                        ("slot.payload.seq", payload.get("seq")),
                        ("slot.payload.frame_seq", payload.get("frame_seq")),
                        ("slot.payload.camera_frame_seq", payload.get("camera_frame_seq")),
                    ]
                )
        if isinstance(frames, dict):
            candidates.extend(
                [
                    ("frames.seq", frames.get("seq")),
                    ("frames.frame_seq", frames.get("frame_seq")),
                    ("frames.camera_frame_seq", frames.get("camera_frame_seq")),
                ]
            )
        for source, value in candidates:
            if value is None:
                continue
            try:
                return int(value), source
            except Exception:
                continue
        return 0, "fallback_0"

    def _build_predict_request(
        self,
        cmd: Dict[str, Any],
        frames: Dict[str, Any] = None,
        frame_slot: Dict[str, Any] = None,
    ) -> Optional[RemotePredictRequest]:
        frame_slot = dict(frame_slot or {})
        if frames is None:
            scheduler = self._scheduler
            if scheduler is None:
                return None
            frame_slot = scheduler.read_slot("camera_frames")
            frames = frame_slot.get("payload") if isinstance(frame_slot, dict) else None
        request_id = str(cmd.get("request_id") or "").strip()
        request_id_source = "runtime_status"
        if not request_id:
            request_id = f"rr_{int(time.time() * 1000)}"
            request_id_source = "generated"
        session_id = str(cmd.get("session_id") or "")
        if not isinstance(frames, dict):
            self._log("error", "remote_predict_precheck_failed", reason="missing_camera_frames", request_id=request_id)
            self._update_result(
                action="predict",
                state="predict_failed",
                ok=False,
                error="missing_camera_frames",
                request_id=request_id,
            )
            return None

        rgb = frames.get("rgb")
        depth = frames.get("depth")
        require_depth = bool(cmd.get("need_depth", self._runtime_profile.get("require_depth", False)))
        if rgb is None:
            self._log("error", "remote_predict_precheck_failed", reason="missing_rgb_frame", request_id=request_id)
            self._update_result(
                action="predict",
                state="predict_failed",
                ok=False,
                error="missing_rgb_frame",
                request_id=request_id,
            )
            return None
        if require_depth and depth is None:
            self._log("error", "remote_predict_precheck_failed", reason="missing_depth_frame", request_id=request_id)
            self._update_result(
                action="predict",
                state="predict_failed",
                ok=False,
                error="missing_depth_frame",
                request_id=request_id,
            )
            return None

        class_id = self._resolve_class_id(cmd.get("class_id"))
        if class_id is None:
            self._log("error", "remote_predict_precheck_failed", reason="missing_class_id", request_id=request_id)
            self._update_result(
                action="predict",
                state="predict_failed",
                ok=False,
                error="missing_class_id",
                request_id=request_id,
            )
            return None
        if cv2 is None:
            self._log("error", "remote_predict_precheck_failed", reason="opencv_unavailable", request_id=request_id)
            self._update_result(
                action="predict",
                state="predict_failed",
                ok=False,
                error="opencv_unavailable",
                request_id=cmd.get("request_id"),
            )
            return None

        expected_rgb = self._shape_hw_list(self._runtime_profile.get("expected_rgb_shape"), default=[720, 1280])
        expected_depth = self._shape_hw_list(self._runtime_profile.get("expected_depth_shape"), default=[720, 1280])
        raw_rgb_shape = self._frame_shape(rgb)
        raw_depth_shape = self._frame_shape(depth)
        upload_frames = dict(frames)
        rgb_upload = self._resize_frame_hw(rgb, expected_rgb, nearest=False)
        depth_upload = self._resize_frame_hw(depth, expected_depth, nearest=True) if depth is not None else None
        upload_frames["rgb"] = rgb_upload
        if depth_upload is not None:
            upload_frames["depth"] = depth_upload
        rgb = rgb_upload
        depth = depth_upload
        capture = self._precheck_capture_shapes(
            frames=upload_frames,
            frame_slot=frame_slot,
            request_id=request_id,
            require_depth=require_depth,
        )
        if capture is None:
            return None
        capture["raw_rgb_shape"] = raw_rgb_shape
        capture["raw_depth_shape"] = raw_depth_shape
        capture["rgb_upload_resized"] = bool(raw_rgb_shape is not None and self._frame_shape(rgb) is not None and raw_rgb_shape[:2] != self._frame_shape(rgb)[:2])
        capture["depth_upload_resized"] = bool(raw_depth_shape is not None and self._frame_shape(depth) is not None and raw_depth_shape[:2] != self._frame_shape(depth)[:2])

        rgb_encoding = normalize_image_encoding(self._runtime_profile.get("rgb_encoding", "jpeg"), default="jpeg")
        depth_encoding = normalize_image_encoding(self._runtime_profile.get("depth_encoding", "png"), default="png")
        rgb_channel_order = self._rgb_upload_order(upload_frames)
        rgb_rgb = self._frame_to_rgb(rgb, rgb_channel_order)
        remote_rgb_cfg = self._remote_rgb_cfg_from_profile(self._runtime_profile)
        try:
            corrected_rgb, correction_info = prepare_remote_rgb(rgb_rgb, remote_rgb_cfg)
        except Exception as exc:
            self._log("error", "remote_predict_precheck_failed", reason="rgb_prepare_failed", request_id=request_id, error=str(exc))
            self._update_result(
                action="predict",
                state="predict_failed",
                ok=False,
                error="rgb_prepare_failed",
                request_id=request_id,
            )
            return None
        raw_rgb_bytes = self._encode_rgb_frame(
            rgb_encoding,
            rgb_rgb,
            quality=int(self._runtime_profile.get("rgb_quality", 90) or 90),
        )
        upload_rgb = corrected_rgb if bool(correction_info.get("rgb_correction_enable", True)) else rgb_rgb
        rgb_bytes = self._encode_rgb_frame(
            rgb_encoding,
            upload_rgb,
            quality=int(self._runtime_profile.get("rgb_quality", 90) or 90),
        )
        depth_bytes = None
        if depth is not None:
            depth_bytes = self._encode_frame(
                depth_encoding,
                depth,
                compression=int(self._runtime_profile.get("depth_compression", 3) or 3),
            )
        if rgb_bytes is None:
            self._log("error", "remote_predict_precheck_failed", reason="rgb_encode_failed", request_id=request_id)
            self._update_result(
                action="predict",
                state="predict_failed",
                ok=False,
                error="rgb_encode_failed",
                request_id=request_id,
            )
            return None
        if raw_rgb_bytes is None:
            raw_rgb_bytes = rgb_bytes
        if require_depth and depth_bytes is None:
            self._log("error", "remote_predict_precheck_failed", reason="depth_encode_failed", request_id=request_id)
            self._update_result(
                action="predict",
                state="predict_failed",
                ok=False,
                error="depth_encode_failed",
                request_id=request_id,
            )
            return None

        profile_metadata = dict(self._runtime_profile.get("metadata") or {})
        request_metadata = dict(cmd.get("metadata") or {}) if isinstance(cmd.get("metadata"), dict) else {}
        extras = dict(profile_metadata)
        extras.update(request_metadata)
        capture["rgb_channel_order"] = "RGB"
        capture["rgb_source_channel_order"] = rgb_channel_order
        capture["rgb_save_backend"] = "cv2"
        capture["rgb_uploaded_corrected"] = bool(correction_info.get("rgb_correction_enable", True))
        capture["rgb_raw_archive_name"] = "rgb_raw.jpg"
        capture["rgb_upload_archive_name"] = "rgb.jpg"
        capture["depth_archive_name"] = "depth.png"
        correction_info["rgb_channel_order"] = "RGB"
        correction_info["rgb_source_channel_order"] = rgb_channel_order
        correction_info["rgb_save_backend"] = "cv2"
        extras["capture"] = capture
        extras["correction_info"] = dict(correction_info)
        extras.update(dict(correction_info))
        target = str(cmd.get("target") or extras.get("target") or "")
        for key in (
            "task_id",
            "raw_target",
            "canonical_target",
            "class_name",
            "local_target_bbox_xyxy",
            "local_target_conf",
            "local_target_frame_id",
            "epoch",
        ):
            if key in cmd and cmd.get(key) is not None:
                extras[key] = cmd.get(key)
            elif key in request_metadata and request_metadata.get(key) is not None:
                extras[key] = request_metadata.get(key)
        extras.setdefault("class_name", request_metadata.get("class_name") or target)
        extras.setdefault("canonical_target", request_metadata.get("canonical_target") or target)
        extras.setdefault("raw_target", request_metadata.get("raw_target") or target)
        extras["class_id"] = int(class_id)
        cmd_robot_id = str(cmd.get("robot_id") or "").strip()
        if cmd_robot_id == "arm_001":
            cmd_robot_id = ""
        robot_id = str(
            cmd_robot_id
            or self._runtime_profile.get("robot_id")
            or profile_metadata.get("robot_id")
            or DEFAULT_REMOTE_ROBOT_ID
        ).strip() or DEFAULT_REMOTE_ROBOT_ID
        command = "predict"
        frame_seq = int(capture.get("frame_seq") or 0)
        frame_seq_source = str(capture.get("frame_seq_source") or "fallback")
        camera_names = sorted(str(name) for name in frames.keys() if str(name) in {"rgb", "depth"})
        extras["request_id_source"] = request_id_source
        extras["session_id_source"] = "runtime_status" if session_id else "empty"
        metadata = RemoteMetadata(
            robot_id=robot_id,
            cmd="predict",
            command=command,
            request_id=request_id,
            session_id=session_id,
            target=target,
            class_id=class_id,
            frame_seq=frame_seq,
            frame_seq_source=frame_seq_source,
            timestamp_ms=int(capture.get("timestamp_ms") or 0) or None,
            camera_names=camera_names,
            extras=extras,
        )
        self._log(
            "info",
            "[GRASP_REMOTE][RGB_CORRECT]",
            request_id=request_id,
            raw_mean=round(float(correction_info.get("rgb_raw_mean") or 0.0), 3),
            corrected_mean=round(float(correction_info.get("rgb_corrected_mean") or 0.0), 3),
            gamma=correction_info.get("rgb_gamma"),
            gain=round(float(correction_info.get("rgb_gain_applied") or 1.0), 3),
        )
        return RemotePredictRequest(
            rgb_bytes=rgb_bytes,
            rgb_raw_bytes=raw_rgb_bytes,
            depth_bytes=depth_bytes,
            class_id=class_id,
            metadata=metadata,
            timeout_s=float(self._effective_runtime_field(cmd, "timeout_s", self._runtime_profile.get("timeout_s", 10.0)) or 10.0),
            rgb_encoding=rgb_encoding,
            depth_encoding=depth_encoding,
        )

    def _runtime_base_url(self) -> str:
        return str(self._runtime_profile.get("base_url") or "").strip()

    def _record_service_init_result(self, response: Optional[RemotePredictResponse]) -> Optional[RemotePredictResponse]:
        now = time.time()
        ok = bool(response is not None and response.ok)
        error = str((response.error if response is not None else "") or "")
        payload = response.payload if isinstance(getattr(response, "payload", None), dict) else {}
        status_code = getattr(response, "status_code", None)
        elapsed_ms = getattr(response, "elapsed_ms", None)
        self._service_init_confirmed = bool(ok)
        self._service_init_state = "ready" if ok else "failed"
        self._service_init_last_error = "" if ok else (error or "init_failed")
        self._service_init_last_ok = bool(ok)
        self._service_init_last_ts = now
        self._service_init_pending = False
        self._service_init_inflight = False
        self._update_result(
            action="init",
            state="init_ok" if ok else "init_failed",
            ok=bool(ok),
            error="" if ok else self._service_init_last_error,
            status_code=status_code,
            elapsed_ms=elapsed_ms,
            result=payload if payload else None,
            request_id=None,
        )
        if not ok:
            detail = {
                "base_url": self._runtime_base_url(),
                "endpoint": "/api/v1/init",
                "status_code": status_code,
                "elapsed_ms": elapsed_ms,
                "error_message": self._service_init_last_error,
            }
            self._log("error", "remote_init_failed", **detail)
        return response

    def _service_init_unavailable(self, *, timeout_s: float, reason: str) -> Dict[str, Any]:
        now = time.time()
        self._service_init_attempts = int(self._service_init_attempts) + 1
        self._service_init_confirmed = False
        self._service_init_state = "failed"
        self._service_init_last_error = str(reason or "init_unavailable")
        self._service_init_last_ok = False
        self._service_init_last_ts = now
        self._service_init_pending = False
        self._service_init_inflight = False
        self._update_result(
            action="init",
            state="init_failed",
            ok=False,
            error=self._service_init_last_error,
            elapsed_ms=0,
            request_id=None,
        )
        self._log(
            "error",
            "remote_init_failed",
            base_url=self._runtime_base_url(),
            endpoint="/api/v1/init",
            status_code=None,
            elapsed_ms=0,
            error_message=self._service_init_last_error,
        )
        return {
            "op": "INIT",
            "ok": False,
            "reason": self._service_init_last_error,
            "error_message": self._service_init_last_error,
            "status_code": None,
            "elapsed_ms": 0,
            "request_id": None,
            "timeout_s": float(timeout_s),
        }

    def _run_service_init(self, *, timeout_s: float, source: str = "service", request_id: Optional[str] = None) -> Dict[str, Any]:
        client = self.client
        base_url = self._runtime_base_url()
        if not self.enabled or client is None:
            return self._service_init_unavailable(timeout_s=timeout_s, reason="remote_disabled")
        if not base_url:
            return self._service_init_unavailable(timeout_s=timeout_s, reason="missing_base_url")
        try:
            client.configure(base_url)
        except Exception:
            pass
        self._service_init_pending = False
        self._service_init_inflight = True
        self._service_init_state = "initializing"
        self._service_init_attempts = int(self._service_init_attempts) + 1
        try:
            response = self.init_server(timeout_s=timeout_s)
        except Exception as exc:
            response = RemotePredictResponse(ok=False, error=str(exc), status_code=None)
        self._record_service_init_result(response)
        return {
            "op": "INIT",
            "ok": bool(response is not None and response.ok),
            "reason": str((response.error if response is not None else "") or ""),
            "error_message": str((response.error if response is not None else "") or ""),
            "status_code": getattr(response, "status_code", None),
            "elapsed_ms": getattr(response, "elapsed_ms", None),
            "request_id": request_id,
            "source": str(source or "service"),
        }

    def _ensure_service_ready_for_predict(self, *, timeout_s: float) -> bool:
        if self._service_init_confirmed or str(self._service_init_state or "").strip().lower() == "ready":
            self._service_init_confirmed = True
            return True
        self._log(
            "warn",
            "remote_predict_init_not_confirmed",
            service_init_state=self._service_init_state,
            service_init_confirmed=self._service_init_confirmed,
            service_init_last_error=self._service_init_last_error,
        )
        result = self._run_service_init(timeout_s=timeout_s, source="predict_preflight")
        if bool(result.get("ok")) or self._service_init_confirmed:
            return True
        self._update_result(
            action="predict",
            state="predict_failed",
            ok=False,
            error="init_not_confirmed",
            status_code=result.get("status_code"),
            elapsed_ms=result.get("elapsed_ms"),
        )
        return False

    def _release_service_quiet(self, *, timeout_s: float, source: str = "predict_finally") -> None:
        if not self.enabled or self.client is None:
            return
        try:
            self.client.release_server(timeout_s=max(0.1, float(timeout_s)))
        except Exception as exc:
            self._log("warn", "remote_release_failed", source=source, error=str(exc))
        finally:
            self._reset_service_init_state()

    def _release_service_quiet_async(self, *, timeout_s: float, source: str = "predict_finally_async") -> None:
        def _runner() -> None:
            try:
                self._release_service_quiet(timeout_s=timeout_s, source=source)
            except Exception as exc:
                self._log("warn", "remote_release_async_failed", source=source, error=str(exc))

        thread = threading.Thread(target=_runner, name="remote.release.quiet", daemon=True)
        thread.start()

    def _release_service_if_ready(self, timeout_s: float = 5.0) -> None:
        if not self.enabled or self.client is None or not self._service_init_confirmed:
            return
        try:
            self.release_server(timeout_s=timeout_s)
        except Exception:
            pass
        self._reset_service_init_state()

    def _worker_loop(self) -> None:
        kind = str(self._runtime_profile.get("kind") or "loop").strip().lower()
        action = str(self._runtime_profile.get("action") or "").strip().lower()
        max_retries = max(1, int(self._runtime_profile.get("max_retries", 1) or 1))

        if kind == "task" and action:
            # ── task worker: execute action once, publish to action-specific route, exit ──
            self._run_task(action=action, max_retries=max_retries)
            self._publish_result(self._task_route(action), self._task_payload(action))
            self._runtime_running = False
            return

        # ── loop worker: no longer used (effects channel removed) ──
        if self._service_init_pending:
            timeout_s = float(self._runtime_profile.get("timeout_s", 10.0) or 10.0)
            self._run_service_init(timeout_s=timeout_s, source="loop_init_compat")
            self._publish_result("remote_result", dict(self._last_result))
        self.logger.warning("remote loop worker started but no effects producer exists; idling")
        self._worker_stop.wait(timeout=1.0)

    def _runtime_status_payload(self) -> Dict[str, Any]:
        scheduler = self._scheduler
        if scheduler is None:
            return {}
        try:
            slot = scheduler.read_slot("runtime_status")
        except Exception:
            return {}
        if not isinstance(slot, dict):
            return {}
        payload = slot.get("payload")
        return dict(payload) if isinstance(payload, dict) else {}

    def _wait_for_warm_capture(
        self,
        *,
        expected_gen: int,
        require_depth: bool,
        frame_wait_timeout_s: float,
        request_id: Optional[str],
    ):
        warmup_frames_requested = max(1, int(self._runtime_profile.get("capture_warmup_frames", 5) or 5))
        warmup_timeout_s = max(0.1, float(self._runtime_profile.get("capture_warmup_timeout_s", 1.0) or 1.0))
        deadline = time.time() + max(0.1, float(frame_wait_timeout_s))
        warmup_deadline = time.time() + warmup_timeout_s
        wait_start = time.time()
        frames = None
        selected_frame_slot = None
        collected = 0
        seen_seq = set()
        last_slot = None
        self._log(
            "info",
            "grasp_capture_warmup_start",
            expected_generation=expected_gen,
            require_depth=require_depth,
            warmup_frames_requested=warmup_frames_requested,
            warmup_timeout_s=warmup_timeout_s,
            request_id=request_id,
        )
        while time.time() < deadline:
            if self._worker_stop.is_set():
                return None, None
            frame_slot = self._scheduler.read_slot("camera_frames") if self._scheduler else None
            last_slot = frame_slot
            if isinstance(frame_slot, dict):
                slot_gen = int(frame_slot.get("generation", 0) or 0)
                payload = frame_slot.get("payload")
                has_rgb = isinstance(payload, dict) and payload.get("rgb") is not None
                has_depth = isinstance(payload, dict) and payload.get("depth") is not None
                if slot_gen == expected_gen and isinstance(payload, dict) and has_rgb and (has_depth or not require_depth):
                    frame_seq, frame_seq_source = self._frame_seq_from_slot(frame_slot, payload)
                    seq_key = (slot_gen, frame_seq)
                    if seq_key not in seen_seq:
                        seen_seq.add(seq_key)
                        collected += 1
                    frames = payload
                    selected_frame_slot = frame_slot
                    if collected >= warmup_frames_requested:
                        break
                    if time.time() >= warmup_deadline:
                        self._log(
                            "warn",
                            "grasp_capture_warmup_timeout",
                            warmup_frames_requested=warmup_frames_requested,
                            warmup_frames_collected=collected,
                            warmup_elapsed_s=round(time.time() - wait_start, 3),
                            warmup_timeout=True,
                            final_rgb_shape=self._frame_shape(payload.get("rgb")),
                            final_depth_shape=self._frame_shape(payload.get("depth")),
                            frame_seq=frame_seq,
                            frame_seq_source=frame_seq_source,
                            request_id=request_id,
                        )
                        break
            self._worker_stop.wait(timeout=0.05)
        if frames is not None:
            self._log(
                "info",
                "grasp_capture_warmup_done",
                warmup_frames_requested=warmup_frames_requested,
                warmup_frames_collected=collected,
                warmup_elapsed_s=round(time.time() - wait_start, 3),
                warmup_timeout=bool(collected < warmup_frames_requested),
                final_rgb_shape=self._frame_shape(frames.get("rgb")),
                final_depth_shape=self._frame_shape(frames.get("depth")),
                request_id=request_id,
            )
            return frames, selected_frame_slot
        slot_gen = None
        has_rgb = False
        has_depth = False
        frame_seq, frame_seq_source = 0, "fallback_0"
        if isinstance(last_slot, dict):
            slot_gen = last_slot.get("generation")
            payload = last_slot.get("payload")
            if isinstance(payload, dict):
                has_rgb = payload.get("rgb") is not None
                has_depth = payload.get("depth") is not None
            frame_seq, frame_seq_source = self._frame_seq_from_slot(last_slot, payload if isinstance(payload, dict) else {})
        reason = "missing_camera_frames"
        if not has_rgb:
            reason = "missing_rgb_frame"
        elif require_depth and not has_depth:
            reason = "missing_depth_frame"
        self._log(
            "error",
            "remote_predict_wait_camera_timeout",
            expected_generation=expected_gen,
            slot_generation=slot_gen,
            wait_ms=int(round((time.time() - wait_start) * 1000.0)),
            has_rgb=has_rgb,
            has_depth=has_depth,
            require_depth=require_depth,
            frame_seq=frame_seq,
            frame_seq_source=frame_seq_source,
            reason=reason,
            warmup_frames_requested=warmup_frames_requested,
            warmup_frames_collected=collected,
            warmup_timeout=True,
            request_id=request_id,
        )
        self._update_result(action="predict", state="predict_failed", ok=False, error=reason, request_id=request_id)
        return None, None

    def _run_task(self, *, action: str, max_retries: int) -> None:
        """Execute a finite task action (init / predict / release).

        For ``init``: retry up to *max_retries* times.
        For ``predict``: issue one /predict after init is confirmed.
        For ``release``: issue one /release unconditionally.
        """
        action = str(action or "").strip().lower()
        if action not in {"init", "predict", "release"}:
            self._update_result(action=action, state="bad_action", ok=False, error=f"unknown task action: {action}")
            return

        timeout_s = float(self._runtime_profile.get("timeout_s", 10.0) or 10.0)

        if action == "init":
            for attempt in range(1, max_retries + 1):
                if self._worker_stop.is_set():
                    self._update_result(action="init", state="init_cancelled", ok=False, error="stopped")
                    return
                self._run_service_init(timeout_s=timeout_s, source="task_init")
                if self._service_init_confirmed:
                    return
            self._update_result(action="init", state="init_exhausted", ok=False,
                                error=f"init failed after {max_retries} retries")
            return

        if action == "predict":
            if not self._ensure_service_ready_for_predict(timeout_s=min(timeout_s, 5.0)):
                self._release_service_quiet(timeout_s=min(2.0, timeout_s), source="init_not_confirmed")
                return
            require_depth = bool(self._runtime_profile.get("require_depth", False))
            frame_wait_timeout_s = float(
                self._runtime_profile.get("remote_predict_frame_wait_timeout_s")
                or (self._runtime_profile.get("metadata") or {}).get("remote_predict_frame_wait_timeout_s")
                or 2.0
            )
            runtime_status = self._wait_runtime_status_context(timeout_s=min(frame_wait_timeout_s, 2.0))
            predict_context = self._predict_context(runtime_status)
            runtime_class_id = predict_context.get("class_id")
            if runtime_class_id is None:
                self._log(
                    "error",
                    "remote_predict_precheck_failed",
                    reason="missing_class_id",
                    sources_checked=["command_payload", "latest_context", "runtime_status", "mode_profile"],
                    latest_context=dict(self._latest_grasp_remote_context or {}),
                    runtime_status=runtime_status,
                )
                self._update_result(
                    action="predict",
                    state="predict_failed",
                    ok=False,
                    error="missing_class_id",
                    request_id=predict_context.get("request_id"),
                )
                self._release_service_quiet(timeout_s=min(2.0, timeout_s), source="predict_precheck_failed")
                return
            runtime_class_id = self._resolve_class_id(runtime_class_id)
            if runtime_class_id is None:
                self._log(
                    "error",
                    "remote_predict_precheck_failed",
                    reason="invalid_class_id",
                    sources_checked=["command_payload", "latest_context", "runtime_status", "mode_profile"],
                    latest_context=dict(self._latest_grasp_remote_context or {}),
                    runtime_status=runtime_status,
                )
                self._update_result(
                    action="predict",
                    state="predict_failed",
                    ok=False,
                    error="invalid_class_id",
                    request_id=predict_context.get("request_id"),
                )
                self._release_service_quiet(timeout_s=min(2.0, timeout_s), source="predict_precheck_failed")
                return
            cmd = {
                **dict(self._runtime_profile.get("metadata") or {}),
                "need_depth": require_depth,
                "class_id": runtime_class_id,
                "robot_id": predict_context.get("robot_id") or self._runtime_profile.get("robot_id"),
                "timeout_s": predict_context.get("timeout_s") or timeout_s,
                "target": predict_context.get("target"),
                "request_id": predict_context.get("request_id"),
                "session_id": predict_context.get("session_id"),
                "metadata": predict_context.get("metadata"),
            }
            # Wait for a fresh camera frame matching current generation.
            # Mode switch clears scheduler slots, so the first frame after
            # camera threads restart may not be published yet.
            if self._scheduler is None:
                self._log("error", "remote_predict_precheck_failed", reason="scheduler_unavailable", runtime_status=runtime_status)
                self._update_result(action="predict", state="predict_failed", ok=False, error="scheduler_unavailable")
                self._release_service_quiet(timeout_s=min(2.0, timeout_s), source="scheduler_unavailable")
                return
            expected_gen = int(self._generation_getter())
            self._log(
                "info",
                "remote_predict_wait_camera_start",
                expected_generation=expected_gen,
                require_depth=require_depth,
            )
            frames, selected_frame_slot = self._wait_for_warm_capture(
                expected_gen=expected_gen,
                require_depth=require_depth,
                frame_wait_timeout_s=frame_wait_timeout_s,
                request_id=cmd.get("request_id"),
            )
            if frames is None:
                self._release_service_quiet(timeout_s=min(2.0, timeout_s), source="missing_camera_frames")
                return
            request = self._build_predict_request(cmd, frames=frames, frame_slot=selected_frame_slot)
            if request is not None:
                metadata_payload = request.metadata.to_metadata_payload()
                self._log(
                    "info",
                    "[GRASP_REMOTE][PREDICT_SEND]",
                    request_id=request.metadata.request_id,
                    target=metadata_payload.get("target"),
                    class_name=metadata_payload.get("class_name"),
                    class_id=metadata_payload.get("class_id"),
                )
                archive_dir = self._archive_predict_payload(request, frames)
                response = self.predict(request, request_id=request.metadata.request_id)
                self._archive_predict_response(archive_dir, response)
                payload = response.payload if isinstance(getattr(response, "payload", None), dict) else {}
                server_status = str(payload.get("status") or "").strip().lower()
                if response is not None and response.ok and server_status not in {"failure", "failed", "error"}:
                    self._log("info", "[GRASP_REMOTE][PREDICT_OK]", request_id=request.metadata.request_id, status_code=response.status_code, elapsed_ms=response.elapsed_ms)
                else:
                    reason = str(payload.get("reason") or payload.get("message") or getattr(response, "error", "") or "predict_failed")
                    local_bbox = metadata_payload.get("local_target_bbox_xyxy")
                    remote_no_detection = "detect" in reason.lower() or "no_detection" in reason.lower()
                    if remote_no_detection and local_bbox:
                        result = dict(self._last_result.get("result") or {})
                        result.update(
                            {
                                "remote_no_detection_but_local_target_present": True,
                                "local_target_bbox_xyxy": local_bbox,
                                "local_target_conf": metadata_payload.get("local_target_conf"),
                                "remote_reason": reason,
                            }
                        )
                        self._last_result["result"] = result
                    self._log("error", "[GRASP_REMOTE][PREDICT_FAILED]", request_id=request.metadata.request_id, status=response.status_code if response is not None else None, reason=reason)
                self._publish_result("remote_result", dict(self._last_result))
                self._release_service_quiet_async(timeout_s=min(2.0, timeout_s), source="predict_done")
            else:
                self._release_service_quiet(timeout_s=min(2.0, timeout_s), source="predict_request_not_built")
            return

        if action == "release":
            self._release_service_if_ready(timeout_s=timeout_s)

    @staticmethod
    def _task_route(action: str) -> str:
        _ROUTES = {"init": "remote_init_status", "predict": "remote_result", "release": "remote_result"}
        return _ROUTES.get(str(action or "").strip().lower(), "remote_result")

    def _task_payload(self, action: str) -> dict:
        action = str(action or "").strip().lower()
        if action == "init":
            return {
                "service_init_state": str(self._service_init_state or "uninitialized"),
                "service_init_confirmed": bool(self._service_init_confirmed),
                "service_init_attempts": int(self._service_init_attempts),
                "service_init_last_error": str(self._service_init_last_error or ""),
                "service_init_last_ok": bool(self._service_init_last_ok),
                "service_init_last_ts": self._service_init_last_ts,
                "base_url": self._runtime_base_url(),
                "endpoint": "/api/v1/init",
                "status_code": self._last_result.get("status_code"),
                "elapsed_ms": self._last_result.get("elapsed_ms"),
                "error_message": str(self._service_init_last_error or ""),
                "ts": time.time(),
            }
        # predict / release
        return {
            "last_action": str(self._last_result.get("last_action") or action),
            "last_ok": bool(self._last_result.get("last_ok", False)),
            "last_error": str(self._last_result.get("last_error") or ""),
            "status_code": self._last_result.get("status_code"),
            "elapsed_ms": self._last_result.get("elapsed_ms"),
            "has_result": bool(self._last_result.get("has_result", False)),
            "result": self._last_result.get("result"),
            "request_id": self._last_result.get("request_id"),
            "sequence": int(self._last_result.get("sequence", 0) or 0),
            "ts": float(self._last_result.get("ts", 0.0) or 0.0),
        }

    def _update_result(
        self,
        *,
        action: str,
        state: str,
        ok: bool,
        error: str = "",
        status_code: Optional[int] = None,
        elapsed_ms: Optional[int] = None,
        result: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> None:
        self._sequence += 1
        self._last_result = {
            "enabled": bool(self.enabled),
            "state": str(state or "idle"),
            "last_action": str(action or "update"),
            "last_ok": bool(ok),
            "last_error": str(error or ""),
            "status_code": status_code,
            "elapsed_ms": elapsed_ms,
            "has_result": result is not None,
            "result": dict(result or {}) if isinstance(result, dict) else result,
            "request_id": request_id,
            "sequence": int(self._sequence),
            "ts": time.time(),
            **self._service_init_fields(),
        }

    def enable(self) -> bool:
        if self.enabled:
            return False
        self.enabled = True
        if self.client is not None:
            base_url = str(self._runtime_profile.get("base_url") or "").strip()
            if base_url:
                try:
                    self.client.configure(base_url)
                except Exception:
                    pass
            self.client.open()
        self._schedule_service_init()
        self._update_result(action="enable", state="enabled", ok=True)
        self._emit("enabled", enabled=True)
        return True

    def disable(self) -> bool:
        if not self.enabled:
            return False
        self._release_service_if_ready(timeout_s=float(self._runtime_profile.get("timeout_s", 5.0) or 5.0))
        self.enabled = False
        self._reset_service_init_state()
        if self.client is not None:
            self.client.close()
        self._update_result(action="disable", state="disabled", ok=True)
        self._emit("disabled", enabled=False)
        return True

    def _record_response(
        self,
        action: str,
        response: Optional[RemotePredictResponse],
        request_id: Optional[str] = None,
    ) -> Optional[RemotePredictResponse]:
        if response is None:
            self._update_result(
                action=action,
                state=f"{action}_skipped",
                ok=False,
                error="no_response",
                elapsed_ms=0,
                request_id=request_id,
            )
            return None
        payload = response.payload if isinstance(response.payload, dict) else {"value": response.payload}
        self._update_result(
            action=action,
            state=f"{action}_{'ok' if response.ok else 'failed'}",
            ok=bool(response.ok),
            error=str(response.error or ""),
            status_code=response.status_code,
            elapsed_ms=getattr(response, "elapsed_ms", None),
            result=payload,
            request_id=request_id,
        )
        return response

    def init_server(self, timeout_s: float = 15.0, request_id: Optional[str] = None) -> Optional[RemotePredictResponse]:
        _ = request_id
        if not self.enabled or self.client is None:
            return None
        return self._record_response("init", self.client.init_server(timeout_s=timeout_s), request_id=None)

    def predict(
        self,
        request: Optional[RemotePredictRequest],
        request_id: Optional[str] = None,
    ) -> Optional[RemotePredictResponse]:
        if request is None or not self.enabled or self.client is None:
            return None
        return self._record_response("predict", self.client.predict(request), request_id=request_id)

    def release_server(self, timeout_s: float = 5.0, request_id: Optional[str] = None) -> Optional[RemotePredictResponse]:
        _ = request_id
        if not self.enabled or self.client is None:
            return None
        return self._record_response("release", self.client.release_server(timeout_s=timeout_s), request_id=None)

    def result_summary(self) -> Dict[str, Any]:
        payload = dict(self._last_result or {})
        payload["enabled"] = bool(self.enabled)
        payload.update(self._service_init_fields())
        return payload

    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "client": self.client.snapshot() if self.client is not None else None,
            "result_summary": self.result_summary(),
            "runtime_running": bool(self._runtime_running),
            "runtime_profile": dict(self._runtime_profile or {}),
            "service_init": self._service_init_fields(),
        }
