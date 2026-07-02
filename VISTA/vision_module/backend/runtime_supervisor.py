#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .preview import NullPreviewSink, OpenCVPreviewSink


class RuntimeSupervisor:
    """Own manager runtime lifecycle based on the active mode plan."""

    def __init__(
        self,
        scheduler,
        camera_manager=None,
        predictor_manager=None,
        remote_manager=None,
        table_edge_manager=None,
        preview_manager=None,
        logger: Optional[logging.Logger] = None,
        backend_event_sink=None,
    ):
        self._scheduler = scheduler
        self.camera_manager = camera_manager
        self.predictor_manager = predictor_manager
        self.remote_manager = remote_manager
        self.table_edge_manager = table_edge_manager
        self.preview_manager = preview_manager
        self.logger = logger or logging.getLogger("vision.runtime_supervisor")
        self._backend_event_sink = backend_event_sink
        self._runtime_running = False
        self._active_generation = 0
        self._active_plan: Optional[Dict[str, Any]] = None
        self._last_apply_result: Dict[str, Any] = {
            "ok": True,
            "reason": "init",
            "mode": "IDLE",
            "generation": 0,
        }
        for mgr in (
            self.camera_manager,
            self.predictor_manager,
            self.remote_manager,
            self.table_edge_manager,
            self.preview_manager,
        ):
            binder = getattr(mgr, "bind_runtime", None)
            if callable(binder):
                try:
                    binder(self._scheduler, self.current_generation)
                except Exception:
                    pass

    def current_generation(self) -> int:
        return int(self._active_generation)

    def _emit_backend_event(self, event_name: str, **fields: Any) -> None:
        if self._backend_event_sink is None:
            return
        try:
            self._backend_event_sink(str(event_name or "").strip().upper(), **dict(fields or {}))
        except Exception:
            pass

    def _log(self, message: str, **fields: Any) -> None:
        extra = fields or None
        self.logger.info("%s%s", message, f" | {extra}" if extra else "")

    def start_runtime(self) -> None:
        self._runtime_running = True
        if self._active_plan is not None:
            self._apply_plan(self._active_plan, self._active_generation)

    def stop_runtime(self) -> None:
        self._runtime_running = False
        self._stop_preview()
        self._stop_table_edge()
        self._stop_remote()
        self._stop_predictor()
        self._stop_camera()
        self._last_apply_result = {
            "ok": True,
            "reason": "runtime_stopped",
            "mode": str((self._active_plan or {}).get("mode") or "IDLE").strip().upper() or "IDLE",
            "generation": int(self._active_generation),
        }

    def reconcile(self, plan: Dict[str, Any], generation: int) -> bool:
        self._active_plan = dict(plan or {})
        self._active_generation = int(generation)
        if not self._runtime_running:
            self._last_apply_result = {
                "ok": True,
                "reason": "plan_staged",
                "mode": str((self._active_plan or {}).get("mode") or "IDLE").strip().upper() or "IDLE",
                "generation": int(self._active_generation),
            }
            return True
        return self._apply_plan(self._active_plan, self._active_generation)

    def _stop_camera(self) -> None:
        if self.camera_manager is None:
            return
        releaser = getattr(self.camera_manager, "release_all", None)
        if callable(releaser):
            try:
                releaser()
            except Exception:
                pass
            return
        stopper = getattr(self.camera_manager, "stop_runtime", None)
        if callable(stopper):
            try:
                stopper()
            except Exception:
                pass

    def _stop_predictor(self) -> None:
        if self.predictor_manager is None:
            return
        setter = getattr(self.predictor_manager, "set_inference_enabled", None)
        if callable(setter):
            try:
                setter(False)
            except Exception:
                pass
        releaser = getattr(self.predictor_manager, "release_all", None)
        if callable(releaser):
            try:
                releaser()
            except Exception:
                pass
            return
        stopper = getattr(self.predictor_manager, "stop_runtime", None)
        if callable(stopper):
            try:
                stopper()
            except Exception:
                pass

    def _stop_remote(self) -> None:
        if self.remote_manager is None:
            return
        stopper = getattr(self.remote_manager, "stop_runtime", None)
        if callable(stopper):
            try:
                stopper()
            except Exception:
                pass
        disabler = getattr(self.remote_manager, "disable", None)
        if callable(disabler):
            try:
                disabler()
            except Exception:
                pass

    def _stop_table_edge(self) -> None:
        if self.table_edge_manager is None:
            return
        stopper = getattr(self.table_edge_manager, "stop_runtime", None)
        if callable(stopper):
            try:
                stopper()
            except Exception:
                pass
        releaser = getattr(self.table_edge_manager, "release_all", None)
        if callable(releaser):
            try:
                releaser()
            except Exception:
                pass

    def _stop_preview(self) -> None:
        if self.preview_manager is None:
            return
        stopper = getattr(self.preview_manager, "stop_runtime", None)
        if callable(stopper):
            try:
                stopper()
            except Exception:
                pass
        disabler = getattr(self.preview_manager, "disable", None)
        if callable(disabler):
            try:
                disabler()
            except Exception:
                pass

    def _configure_camera(self, payload: Dict[str, Any]) -> bool:
        if self.camera_manager is None:
            return True
        enabled_cameras = list(payload.get("enabled_cameras") or [])
        overrides = dict(payload.get("camera_overrides") or {})
        ok = True
        active = set()
        try:
            active = set(self.camera_manager.active_names())
        except Exception:
            active = set()
        desired = set(str(name) for name in enabled_cameras)
        for camera_name in sorted(desired):
            try:
                applied = self.camera_manager.ensure_camera(camera_name, override=overrides.get(camera_name))
                ok = bool(applied or camera_name in self.camera_manager.active_names()) and ok
            except Exception:
                ok = False
        for camera_name in sorted(active - desired):
            try:
                self.camera_manager.disable_camera(camera_name)
            except Exception:
                ok = False
        if desired:
            try:
                self.camera_manager.start_runtime()
            except Exception:
                ok = False
        else:
            try:
                self.camera_manager.stop_runtime()
            except Exception:
                ok = False
        return ok

    def _configure_predictor(self, payload: Dict[str, Any]) -> bool:
        if self.predictor_manager is None:
            return True
        enabled = bool(payload.get("enabled", False))
        model_name = payload.get("model_name")
        keep_loaded = bool(payload.get("keep_loaded", False))
        ok = True
        setter = getattr(self.predictor_manager, "set_inference_enabled", None)
        if callable(setter):
            try:
                setter(enabled)
            except Exception:
                ok = False
        if enabled and model_name:
            try:
                prepared = self.predictor_manager.ensure_model(str(model_name))
                ok = bool(prepared or self.predictor_manager.is_ready()) and ok
            except Exception:
                ok = False
            try:
                self.predictor_manager.start_runtime()
            except Exception:
                ok = False
        elif keep_loaded and model_name:
            try:
                prepared = self.predictor_manager.ensure_model(str(model_name))
                ok = bool(prepared or self.predictor_manager.is_ready()) and ok
            except Exception:
                ok = False
            try:
                self.predictor_manager.stop_runtime()
            except Exception:
                ok = False
        else:
            try:
                self.predictor_manager.stop_runtime()
            except Exception:
                ok = False
            try:
                self.predictor_manager.disable_model()
            except Exception:
                ok = False
        return ok

    def _configure_remote(self, payload: Dict[str, Any]) -> bool:
        if self.remote_manager is None:
            return True
        enabled = bool(payload.get("enabled", False))
        ok = True
        configurator = getattr(self.remote_manager, "configure_runtime", None)
        if callable(configurator):
            try:
                configurator(dict(payload or {}))
            except Exception:
                ok = False
        client = getattr(self.remote_manager, "client", None)
        base_url = str(payload.get("base_url") or "").strip()
        if client is not None and base_url:
            try:
                client.configure(base_url)
            except Exception:
                ok = False
        service_available = bool(base_url)
        if service_available:
            try:
                self.remote_manager.enable()
                self.remote_manager.start_runtime()
            except Exception:
                ok = False
        else:
            keep_warm = False
            try:
                keep_warm = bool(getattr(self.remote_manager, "keep_remote_warm", lambda: False)())
            except Exception:
                keep_warm = False
            if keep_warm:
                return ok
            try:
                self.remote_manager.stop_runtime()
            except Exception:
                ok = False
            try:
                self.remote_manager.disable()
            except Exception:
                ok = False
        if enabled and not service_available:
            ok = False
        return ok

    def _configure_table_edge(self, payload: Dict[str, Any]) -> bool:
        if self.table_edge_manager is None:
            return True
        enabled = bool(payload.get("enabled", False))
        ok = True
        configurator = getattr(self.table_edge_manager, "configure", None)
        if callable(configurator):
            try:
                configurator(dict(payload or {}))
            except Exception:
                ok = False
        if enabled:
            try:
                self.table_edge_manager.start_runtime()
            except Exception:
                ok = False
        else:
            try:
                self.table_edge_manager.stop_runtime()
            except Exception:
                ok = False
            releaser = getattr(self.table_edge_manager, "release_all", None)
            if callable(releaser):
                try:
                    releaser()
                except Exception:
                    ok = False
        return ok

    def _resolve_preview_sink(self, payload: Dict[str, Any]):
        sink_name = str(payload.get("sink_name") or "null").strip().lower()
        window_name = str(payload.get("window_name") or "VISTA App Dashboard")
        if sink_name == "opencv":
            return OpenCVPreviewSink(window_name=window_name)
        return NullPreviewSink()

    def _configure_preview(self, payload: Dict[str, Any], mode_name: str = "IDLE") -> bool:
        if self.preview_manager is None:
            return True
        enabled = bool(payload.get("enabled", False))
        ok = True
        sink = self._resolve_preview_sink(payload)
        try:
            current_sink = getattr(self.preview_manager, "sink", None)
            current_name = getattr(current_sink, "sink_name", "null") if current_sink is not None else "null"
            current_window = getattr(current_sink, "window_name", "")
            requested_window = getattr(sink, "window_name", "")
            if getattr(sink, "sink_name", "null") != current_name or (
                getattr(sink, "sink_name", "null") == "opencv" and requested_window and requested_window != current_window
            ):
                self.preview_manager.set_sink(sink)
        except Exception:
            ok = False
        configurator = getattr(self.preview_manager, "configure_preview_mode", None)
        if callable(configurator):
            try:
                configurator(
                    str(mode_name or "IDLE").strip().upper() or "IDLE",
                    metadata=dict(payload.get("metadata") or {}),
                    reason="mode_switch",
                )
            except Exception:
                ok = False
        if enabled:
            try:
                self.preview_manager.enable()
                self.preview_manager.start_runtime()
            except Exception:
                ok = False
        else:
            try:
                self.preview_manager.stop_runtime()
            except Exception:
                ok = False
            try:
                self.preview_manager.disable()
            except Exception:
                ok = False
        return ok

    def _apply_plan(self, plan: Dict[str, Any], generation: int) -> bool:
        mode_name = str((plan or {}).get("mode") or "IDLE").strip().upper() or "IDLE"
        capabilities = dict((plan or {}).get("capabilities") or {})
        ok = True
        ok = self._configure_camera(dict(capabilities.get("camera") or {})) and ok
        ok = self._configure_predictor(dict(capabilities.get("predictor") or {})) and ok
        ok = self._configure_remote(dict(capabilities.get("remote") or {})) and ok
        ok = self._configure_table_edge(dict(capabilities.get("table_edge") or {})) and ok
        ok = self._configure_preview(dict(capabilities.get("preview") or {}), mode_name=mode_name) and ok
        self._last_apply_result = {
            "ok": bool(ok),
            "reason": "reconciled" if ok else "apply_failed",
            "mode": mode_name,
            "generation": int(generation),
        }
        level = "info" if ok else "error"
        self._emit_backend_event(
            "BACKEND_RUNTIME_RECONCILED",
            level=level,
            mode=mode_name,
            generation=int(generation),
            ok=bool(ok),
            yolo26_enabled=bool(
                str(((capabilities.get("predictor") or {}).get("model_name") or "")).strip().lower().startswith("yolo26")
            ),
            yolo_table_search_enabled=bool(mode_name == "FIND_TABLE" and (capabilities.get("predictor") or {}).get("enabled", False)),
            current_vision_mode=mode_name,
            current_preview_layout=str(
                (((capabilities.get("preview") or {}).get("metadata") or {}).get("layout") or "")
            ),
            request_source="state_machine",
        )
        if not ok:
            self._log("runtime supervisor apply failed", mode=mode_name, generation=int(generation))
        return bool(ok)

    def snapshot(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "runtime_running": bool(self._runtime_running),
            "active_generation": int(self._active_generation),
            "active_mode": str((self._active_plan or {}).get("mode") or "IDLE").strip().upper() or "IDLE",
            "last_apply_result": dict(self._last_apply_result or {}),
        }
        if self.camera_manager is not None:
            try:
                payload["camera"] = self.camera_manager.snapshot()
            except Exception:
                payload["camera"] = {"error": "snapshot_failed"}
        if self.predictor_manager is not None:
            try:
                payload["predictor"] = self.predictor_manager.snapshot()
            except Exception:
                payload["predictor"] = {"error": "snapshot_failed"}
        if self.remote_manager is not None:
            try:
                payload["remote"] = self.remote_manager.snapshot()
            except Exception:
                payload["remote"] = {"error": "snapshot_failed"}
        if self.table_edge_manager is not None:
            try:
                payload["table_edge"] = self.table_edge_manager.snapshot()
            except Exception:
                payload["table_edge"] = {"error": "snapshot_failed"}
        if self.preview_manager is not None:
            try:
                payload["preview"] = self.preview_manager.snapshot()
            except Exception:
                payload["preview"] = {"error": "snapshot_failed"}
        return payload
