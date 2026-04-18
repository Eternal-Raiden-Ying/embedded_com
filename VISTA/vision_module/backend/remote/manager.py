#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import time
from typing import Any, Callable, Dict, Optional

try:
    import aidcv as cv2
except ImportError:
    import cv2

from ...config.data import ASR_VOCAB_MAP, TARGET_CLASSES
from .client import RemoteGraspClient
from .protocol import RemoteMetadata, RemotePredictRequest, RemotePredictResponse


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

    def set_client(self, client: RemoteGraspClient) -> None:
        self.client = client

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

    def _resolve_class_id(self, target: Optional[str], explicit_class_id: Any = None) -> Optional[int]:
        if explicit_class_id is not None:
            try:
                return int(explicit_class_id)
            except Exception:
                return None
        target_name = str(target or "").strip()
        if not target_name:
            return None
        if target_name in TARGET_CLASSES:
            return TARGET_CLASSES.index(target_name)
        valid_names = ASR_VOCAB_MAP.get(target_name, set())
        for class_name in sorted(valid_names):
            if class_name in TARGET_CLASSES:
                return TARGET_CLASSES.index(class_name)
        return None

    def _encode_frame(self, ext: str, frame, params=None) -> Optional[bytes]:
        if frame is None:
            return None
        try:
            ok, encoded = cv2.imencode(ext, frame, list(params or ()))
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
        if not isinstance(frames, dict):
            self._update_result(
                action="predict",
                state="predict_failed",
                ok=False,
                error="missing_camera_frames",
                request_id=cmd.get("request_id"),
            )
            return None

        rgb = frames.get("rgb")
        depth = frames.get("depth")
        if rgb is None:
            self._update_result(
                action="predict",
                state="predict_failed",
                ok=False,
                error="missing_rgb_frame",
                request_id=cmd.get("request_id"),
            )
            return None
        if bool(cmd.get("need_depth", False)) and depth is None:
            self._update_result(
                action="predict",
                state="predict_failed",
                ok=False,
                error="missing_depth_frame",
                request_id=cmd.get("request_id"),
            )
            return None

        class_id = self._resolve_class_id(cmd.get("target"), cmd.get("class_id"))
        if class_id is None:
            self._update_result(
                action="predict",
                state="predict_failed",
                ok=False,
                error="missing_class_id",
                request_id=cmd.get("request_id"),
            )
            return None
        rgb_bytes = self._encode_frame(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        depth_bytes = self._encode_frame(".png", depth, [int(cv2.IMWRITE_PNG_COMPRESSION), 3]) if depth is not None else None
        if rgb_bytes is None:
            self._update_result(
                action="predict",
                state="predict_failed",
                ok=False,
                error="rgb_encode_failed",
                request_id=cmd.get("request_id"),
            )
            return None

        metadata = RemoteMetadata(
            robot_id=str(cmd.get("robot_id") or "arm_001"),
            command=str(cmd.get("command") or "predict"),
            class_id=class_id,
            extras=dict(cmd.get("metadata") or {}) if isinstance(cmd.get("metadata"), dict) else {},
        )
        metadata.extras.setdefault("target", cmd.get("target"))
        metadata.extras.setdefault("request_id", cmd.get("request_id"))
        return RemotePredictRequest(
            rgb_bytes=rgb_bytes,
            depth_bytes=depth_bytes,
            seg_bytes=None,
            class_id=class_id,
            metadata=metadata,
            timeout_s=float(cmd.get("timeout_s", 10.0) or 10.0),
        )

    def _handle_command(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        op = str(cmd.get("op") or "").strip().upper()
        timeout_s = float(cmd.get("timeout_s", 5.0) or 5.0)
        request_id = cmd.get("request_id")
        client = self.client
        base_url = str(cmd.get("base_url") or "").strip()
        if client is not None and base_url:
            try:
                client.configure(base_url)
            except Exception:
                pass
        ack = {"op": op, "ok": False, "reason": "unsupported", "request_id": request_id}
        try:
            if op == "INIT":
                resp = self.init_server(timeout_s=timeout_s, request_id=request_id)
                ack = {
                    "op": op,
                    "ok": bool(resp is not None and resp.ok),
                    "reason": str((resp.error if resp is not None else "") or ""),
                    "status_code": getattr(resp, "status_code", None),
                    "request_id": request_id,
                }
            elif op == "PREDICT":
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
                resp = self.release_server(timeout_s=timeout_s, request_id=request_id)
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
        while self._runtime_running and not self._worker_stop.is_set():
            self._publish_result("remote_result", self.result_summary())
            scheduler = self._scheduler
            if scheduler is not None:
                cmd = scheduler.consume_event("remote_cmd")
                if isinstance(cmd, dict):
                    ack = self._handle_command(cmd)
                    self._publish_event("remote_ack", ack)
            self._worker_stop.wait(timeout=self._worker_interval_s)

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
        }

    def enable(self) -> bool:
        if self.enabled:
            return False
        self.enabled = True
        if self.client is not None:
            self.client.open()
        self._update_result(action="enable", state="enabled", ok=True)
        self._emit("enabled", enabled=True)
        return True

    def disable(self) -> bool:
        if not self.enabled:
            return False
        self.enabled = False
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
        if not self.enabled or self.client is None:
            return None
        return self._record_response("init", self.client.init_server(timeout_s=timeout_s), request_id=request_id)

    def predict(
        self,
        request: Optional[RemotePredictRequest],
        request_id: Optional[str] = None,
    ) -> Optional[RemotePredictResponse]:
        if request is None or not self.enabled or self.client is None:
            return None
        return self._record_response("predict", self.client.predict(request), request_id=request_id)

    def release_server(self, timeout_s: float = 5.0, request_id: Optional[str] = None) -> Optional[RemotePredictResponse]:
        if not self.enabled or self.client is None:
            return None
        return self._record_response("release", self.client.release_server(timeout_s=timeout_s), request_id=request_id)

    def result_summary(self) -> Dict[str, Any]:
        payload = dict(self._last_result or {})
        payload["enabled"] = bool(self.enabled)
        return payload

    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "client": self.client.snapshot() if self.client is not None else None,
            "result_summary": self.result_summary(),
            "runtime_running": bool(self._runtime_running),
        }
