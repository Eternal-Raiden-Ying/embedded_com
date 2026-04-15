#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import time
from typing import Any, Callable, Dict, Optional

from .client import RemoteGraspClient
from .protocol import RemotePredictRequest, RemotePredictResponse


class RemoteManager:
    """Own remote client lifecycle and remote request orchestration.

    This manager is the capability-facing wrapper used by mode control and
    GRASP stage logic. It should hide HTTP details from the rest of VISTA.
    """

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

    def set_client(self, client: RemoteGraspClient) -> None:
        """Replace the underlying remote client implementation."""
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

    def _worker_loop(self) -> None:
        while self._runtime_running and not self._worker_stop.is_set():
            self._publish_result("remote_result", self.result_summary())
            scheduler = self._scheduler
            if scheduler is not None:
                cmd = scheduler.consume_event("remote_cmd")
                if isinstance(cmd, dict):
                    op = str(cmd.get("op") or "").strip().upper()
                    timeout_s = float(cmd.get("timeout_s", 5.0) or 5.0)
                    ack = {"op": op, "ok": False, "reason": "unsupported"}
                    try:
                        if op == "INIT":
                            resp = self.init_server(timeout_s=timeout_s)
                            ack = {"op": op, "ok": bool(resp is not None and resp.ok)}
                        elif op == "RELEASE":
                            resp = self.release_server(timeout_s=timeout_s)
                            ack = {"op": op, "ok": bool(resp is not None and resp.ok)}
                    except Exception as exc:
                        ack = {"op": op, "ok": False, "reason": str(exc)}
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
            "sequence": int(self._sequence),
            "ts": time.time(),
        }

    def enable(self) -> bool:
        """Enable remote capability and prepare the client session."""
        if self.enabled:
            return False
        self.enabled = True
        if self.client is not None:
            self.client.open()
        self._update_result(action="enable", state="enabled", ok=True)
        self._emit("enabled", enabled=True)
        return True

    def disable(self) -> bool:
        """Disable remote capability and close the client session."""
        if not self.enabled:
            return False
        self.enabled = False
        if self.client is not None:
            self.client.close()
        self._update_result(action="disable", state="disabled", ok=True)
        self._emit("disabled", enabled=False)
        return True

    def _record_response(self, action: str, response: Optional[RemotePredictResponse]) -> Optional[RemotePredictResponse]:
        if response is None:
            self._update_result(action=action, state=f"{action}_skipped", ok=False, error="no_response")
            return None
        payload = response.payload if isinstance(response.payload, dict) else {"value": response.payload}
        self._update_result(
            action=action,
            state=f"{action}_{'ok' if response.ok else 'failed'}",
            ok=bool(response.ok),
            error=str(response.error or ""),
            status_code=response.status_code,
            result=payload,
        )
        return response

    def init_server(self, timeout_s: float = 15.0) -> Optional[RemotePredictResponse]:
        """Initialize the remote service before GRASP_REMOTE begins."""
        if not self.enabled or self.client is None:
            return None
        return self._record_response("init", self.client.init_server(timeout_s=timeout_s))

    def predict(self, request: RemotePredictRequest) -> Optional[RemotePredictResponse]:
        """Send one remote predict request assembled from synchronized inputs."""
        if not self.enabled or self.client is None:
            return None
        return self._record_response("predict", self.client.predict(request))

    def release_server(self, timeout_s: float = 5.0) -> Optional[RemotePredictResponse]:
        """Release remote service resources after remote work completes."""
        if not self.enabled or self.client is None:
            return None
        return self._record_response("release", self.client.release_server(timeout_s=timeout_s))

    def result_summary(self) -> Dict[str, Any]:
        payload = dict(self._last_result or {})
        payload["enabled"] = bool(self.enabled)
        return payload

    def snapshot(self) -> Dict[str, Any]:
        """Expose remote manager and client state for diagnostics."""
        return {
            "enabled": self.enabled,
            "client": self.client.snapshot() if self.client is not None else None,
            "result_summary": self.result_summary(),
            "runtime_running": bool(self._runtime_running),
        }
