#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import time
from typing import Any, Callable, Dict, Optional

try:
    import aidcv as cv2
except ImportError:
    try:
        import cv2
    except ImportError:
        cv2 = None  # type: ignore

from .client import RemoteGraspClient
from .protocol import RemoteMetadata, RemotePredictRequest, RemotePredictResponse, image_encoding_info, normalize_image_encoding


class RemoteManager:
    """Own remote client lifecycle and remote request orchestration."""

    def __init__(
        self,
        client: Optional[RemoteGraspClient] = None,
        logger=None,
        capability_sink: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.client = client
        self.logger = logger
        self._capability_sink = capability_sink
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
            "metadata": {},
            "rgb_encoding": "jpeg",
            "depth_encoding": "png",
            "rgb_quality": 90,
            "depth_compression": 3,
        }
        self._service_init_state = "uninitialized"
        self._service_init_confirmed = False
        self._service_init_attempts = 0
        self._service_init_last_error = ""
        self._service_init_last_ok = False
        self._service_init_last_ts: Optional[float] = None
        self._service_init_pending = False
        self._service_init_inflight = False
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
                "metadata": dict(profile.get("metadata") or {}) if isinstance(profile.get("metadata"), dict) else {},
                "rgb_encoding": normalize_image_encoding(profile.get("rgb_encoding"), default="jpeg"),
                "depth_encoding": normalize_image_encoding(profile.get("depth_encoding"), default="png"),
                "rgb_quality": int(profile.get("rgb_quality", 90) or 90),
                "depth_compression": int(profile.get("depth_compression", 3) or 3),
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
        except Exception:
            pass

    def _publish_event(self, route: str, payload: Any) -> None:
        scheduler = self._scheduler
        if scheduler is None:
            return
        try:
            generation = int(self._generation_getter())
        except Exception:
            generation = 0
        try:
            scheduler.publish_event(route, payload, generation=generation)
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

    def _build_predict_request(self, cmd: Dict[str, Any]) -> Optional[RemotePredictRequest]:
        scheduler = self._scheduler
        if scheduler is None:
            return None
        frame_slot = scheduler.read_slot("camera_frames")
        frames = frame_slot.get("payload") if isinstance(frame_slot, dict) else None
        request_id = cmd.get("request_id")
        if not isinstance(frames, dict):
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
            self._update_result(
                action="predict",
                state="predict_failed",
                ok=False,
                error="missing_rgb_frame",
                request_id=request_id,
            )
            return None
        if require_depth and depth is None:
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
            self._update_result(
                action="predict",
                state="predict_failed",
                ok=False,
                error="missing_class_id",
                request_id=request_id,
            )
            return None
        if cv2 is None:
            self._update_result(
                action="predict",
                state="predict_failed",
                ok=False,
                error="opencv_unavailable",
                request_id=cmd.get("request_id"),
            )
            return None

        rgb_encoding = normalize_image_encoding(self._runtime_profile.get("rgb_encoding", "jpeg"), default="jpeg")
        depth_encoding = normalize_image_encoding(self._runtime_profile.get("depth_encoding", "png"), default="png")
        rgb_bytes = self._encode_frame(
            rgb_encoding,
            rgb,
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
            self._update_result(
                action="predict",
                state="predict_failed",
                ok=False,
                error="rgb_encode_failed",
                request_id=request_id,
            )
            return None
        if require_depth and depth_bytes is None:
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
        extras.setdefault("target", cmd.get("target"))
        extras.setdefault("request_id", request_id)
        extras.setdefault("frame_seq", int(frame_slot.get("seq", 0) or 0))
        extras.setdefault("camera_names", sorted(frames.keys()))
        metadata = RemoteMetadata(
            robot_id=str(cmd.get("robot_id") or "arm_001"),
            command=str(self._effective_runtime_field(cmd, "command", self._runtime_profile.get("command", "predict")) or "predict"),
            class_id=class_id,
            extras=extras,
        )
        return RemotePredictRequest(
            rgb_bytes=rgb_bytes,
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
            result=payload if payload else None,
            request_id=None,
        )
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
            request_id=None,
        )
        return {
            "op": "INIT",
            "ok": False,
            "reason": self._service_init_last_error,
            "status_code": None,
            "request_id": None,
            "timeout_s": float(timeout_s),
        }

    def _run_service_init(self, *, timeout_s: float, source: str = "service") -> Dict[str, Any]:
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
            "status_code": getattr(response, "status_code", None),
            "request_id": None,
            "source": str(source or "service"),
        }

    def _release_service_if_ready(self, timeout_s: float = 5.0) -> None:
        if not self.enabled or self.client is None or not self._service_init_confirmed:
            return
        try:
            self.release_server(timeout_s=timeout_s)
        except Exception:
            pass
        self._reset_service_init_state()

    def _handle_command(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        op = str(cmd.get("op") or "").strip().upper()
        timeout_s = float(self._effective_runtime_field(cmd, "timeout_s", self._runtime_profile.get("timeout_s", 5.0)) or 5.0)
        request_id = cmd.get("request_id")
        client = self.client
        base_url = self._runtime_base_url()
        if client is not None and base_url:
            try:
                client.configure(base_url)
            except Exception:
                pass
        ack = {"op": op, "ok": False, "reason": "unsupported", "request_id": request_id}
        try:
            if op == "INIT":
                ack = self._run_service_init(timeout_s=timeout_s, source="command")
            elif op == "PREDICT":
                if not self._service_init_confirmed:
                    self._update_result(
                        action="predict",
                        state="predict_failed",
                        ok=False,
                        error="init_not_confirmed",
                        request_id=request_id,
                    )
                    return {
                        "op": op,
                        "ok": False,
                        "reason": "init_not_confirmed",
                        "request_id": request_id,
                    }
                request = self._build_predict_request(cmd)
                resp = self.predict(request, request_id=request_id) if request is not None else None
                ack = {
                    "op": op,
                    "ok": bool(resp is not None and resp.ok),
                    "reason": str((resp.error if resp is not None else self._last_result.get("last_error")) or ""),
                    "status_code": getattr(resp, "status_code", None),
                    "request_id": request_id,
                }
            elif op == "RELEASE":
                resp = self.release_server(timeout_s=timeout_s)
                self._reset_service_init_state()
                ack = {
                    "op": op,
                    "ok": bool(resp is not None and resp.ok),
                    "reason": str((resp.error if resp is not None else "") or ""),
                    "status_code": getattr(resp, "status_code", None),
                    "request_id": request_id,
                }
        except Exception as exc:
            ack = {"op": op, "ok": False, "reason": str(exc), "request_id": request_id}
            self._update_result(
                action=op.lower() or "remote",
                state=f"{op.lower()}_failed" if op else "remote_failed",
                ok=False,
                error=str(exc),
                request_id=request_id,
            )
        return ack

    def _worker_loop(self) -> None:
        kind = str(self._runtime_profile.get("kind") or "loop").strip().lower()
        action = str(self._runtime_profile.get("action") or "").strip().lower()
        max_retries = max(1, int(self._runtime_profile.get("max_retries", 1) or 1))

        if kind == "task" and action:
            # ── task worker: execute action once, publish result, exit ──
            self._run_task(action=action, max_retries=max_retries)
            self._publish_result("remote_result", self.result_summary())
            self._runtime_running = False
            return

        # ── loop worker: continuous event-driven ──
        while self._runtime_running and not self._worker_stop.is_set():
            if self._service_init_pending and not self._service_init_inflight:
                timeout_s = float(self._runtime_profile.get("timeout_s", 10.0) or 10.0)
                self._run_service_init(timeout_s=timeout_s, source="startup")
            scheduler = self._scheduler
            if scheduler is not None:
                cmd = scheduler.consume_event("remote_cmd")
                if isinstance(cmd, dict):
                    ack = self._handle_command(cmd)
                    self._publish_event("remote_ack", ack)
            self._publish_result("remote_result", self.result_summary())
            self._worker_stop.wait(timeout=self._worker_interval_s)

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
            if not self._service_init_confirmed:
                self._update_result(action="predict", state="predict_failed", ok=False, error="init_not_confirmed")
                return
            if self._worker_stop.is_set():
                return
            frame_slot = (self._scheduler.read_slot("camera_frames") if self._scheduler else None)
            frames = frame_slot.get("payload") if isinstance(frame_slot, dict) else {}
            require_depth = bool(self._runtime_profile.get("require_depth", False))
            cmd = {
                "need_depth": require_depth,
                "class_id": None,  # supplied via plan metadata if needed
                **dict(self._runtime_profile.get("metadata") or {}),
            }
            request = self._build_predict_request(cmd)
            if request is not None:
                resp = self.predict(request)
                self._record_response("predict", resp)
            return

        if action == "release":
            self._release_service_if_ready(timeout_s=timeout_s)

    def _update_result(
        self,
        *,
        action: str,
        state: str,
        ok: bool,
        error: str = "",
        status_code: Optional[int] = None,
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
