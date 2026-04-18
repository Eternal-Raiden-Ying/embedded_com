#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

from .camera_manager import CameraManager
from .mode_controller import ModeController
from .predictor_manager import PredictorManager
from .preview import NullPreviewSink
from .preview.manager import PreviewManager
from .remote.client import RemoteGraspClient
from .remote.manager import RemoteManager
from .table_edge_manager import TableEdgeManager
from ..config.mode_defaults import build_default_mode_profiles
from ..config.schema import VisionServiceConfig


class VisionEngine:
    """Runtime facade: assemble controllers/managers and expose control-plane APIs."""

    def __init__(
        self,
        cfg: VisionServiceConfig,
        logger: Optional[logging.Logger] = None,
        event_sink: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.cfg = cfg
        self.log = logger or logging.getLogger("vision.engine")
        self._event_sink = event_sink
        self.running = False
        self.lock = threading.RLock()
        self.current_mode = "IDLE"
        self._external_mode_control_enabled = True
        self._last_frame_seq_seen = 0
        self.infer_enabled = False

        self.camera_manager = CameraManager(
            cfg=self.cfg,
            logger=self.log,
            capability_sink=self._on_camera_capability_change,
        )
        self.predictor_manager = PredictorManager(
            cfg=self.cfg,
            logger=self.log,
            capability_sink=self._on_predictor_capability_change,
        )
        self.remote_manager = RemoteManager(
            client=RemoteGraspClient(logger=self.log),
            logger=self.log,
            capability_sink=self._on_simple_capability_change,
        )
        self.table_edge_manager = TableEdgeManager(
            logger=self.log,
            capability_sink=self._on_simple_capability_change,
        )
        self.preview_manager = PreviewManager(
            sink=NullPreviewSink(),
            logger=self.log,
            capability_sink=self._on_simple_capability_change,
        )

        self.mode_controller = ModeController(
            camera_manager=self.camera_manager,
            predictor_manager=self.predictor_manager,
            remote_manager=self.remote_manager,
            table_edge_manager=self.table_edge_manager,
            preview_manager=self.preview_manager,
            logger=self.log,
            backend_event_sink=self._emit_event,
        )
        self.mode_controller.bind_runtime_controls(
            set_inference=self.set_inference_enabled,
            preview_allowed=bool(self.cfg.debug.preview),
        )
        self.mode_controller.register_profiles(build_default_mode_profiles(self.cfg.model.active_model).values())

        # Backward-compatible aliases used by test scripts.
        self.cams = self.camera_manager.cams
        self.predictor = self.predictor_manager.predictor
        self.active_model_name = self.predictor_manager.active_model_name
        self.remote_enabled = bool(self.remote_manager.enabled)
        self.preview_enabled = bool(self.preview_manager.enabled)
        self.table_edge_enabled = True

    def _sync_aliases(self) -> None:
        self.predictor = self.predictor_manager.predictor
        self.active_model_name = self.predictor_manager.active_model_name
        self.remote_enabled = bool(self.remote_manager.enabled)
        self.preview_enabled = bool(self.preview_manager.enabled)
        self.infer_enabled = bool(self.predictor_manager.inference_enabled)
        self.table_edge_enabled = True

    def _emit_event(self, event_name: str, **fields: Any) -> None:
        if self._event_sink is None:
            return
        try:
            self._event_sink(event_name, fields)
        except Exception:
            pass

    def _emit_backend_lifecycle(self, action: str, **fields: Any) -> None:
        self._emit_event("BACKEND_LIFECYCLE_CHANGED", action=str(action or "updated").strip().lower(), **fields)

    def _emit_capability_change(self, capability: str, action: str, **fields: Any) -> None:
        self._emit_event(
            "CAPABILITY_CHANGED",
            capability=str(capability or "unknown").strip().lower(),
            action=str(action or "updated").strip().lower(),
            **fields,
        )

    def _emit_backend_failure(self, failure_type: str, **fields: Any) -> None:
        self._emit_event(
            "BACKEND_FAILURE",
            level="error",
            failure_type=str(failure_type or "backend_failure").strip().lower(),
            **fields,
        )

    def _on_camera_capability_change(self, action: str, resource_name: str, fields: Dict[str, Any]) -> None:
        payload = dict(fields or {})
        payload["resource_name"] = resource_name
        self._emit_capability_change("camera", action, **payload)

    def _on_predictor_capability_change(self, action: str, capability_name: str, fields: Dict[str, Any]) -> None:
        payload = dict(fields or {})
        self._emit_capability_change(capability_name, action, **payload)

    def _on_simple_capability_change(self, capability: str, payload: Dict[str, Any]) -> None:
        fields = dict(payload or {})
        action = str(fields.pop("action", "updated")).strip().lower()
        self._emit_capability_change(capability, action, **fields)

    def init(self) -> None:
        self.log.info("engine init: system ready")
        self.set_mode("IDLE", reason="engine_init", force=True, source="engine")
        self._emit_backend_lifecycle(
            "initialized",
            registered_modes=sorted(self.mode_controller.snapshot().get("registered_modes", [])),
        )

    def tick(self, now_ts: Optional[float] = None) -> None:
        self.mode_controller.tick(now_ts=now_ts)
        self.current_mode = self.mode_controller.current_mode()
        self._sync_aliases()

    def set_external_mode_control(self, enabled: bool) -> None:
        self._external_mode_control_enabled = bool(enabled)

    def preview_exit_requested(self) -> bool:
        return bool(self.mode_controller.preview_exit_requested())

    def push_stage_signals(self, signals: Dict[str, object]) -> None:
        self.mode_controller.push_stage_signals(dict(signals or {}))

    def set_mode(self, name: str, reason: str = "", force: bool = False, source: str = "external") -> bool:
        requested = str(name or "IDLE").strip().upper() or "IDLE"
        source_text = str(source or "external").strip().lower() or "external"
        if source_text == "external" and not self._external_mode_control_enabled:
            self.log.warning("reject external set_mode: %s", requested)
            self._emit_backend_failure(
                "external_mode_set_blocked",
                requested_mode=requested,
                reason=str(reason or ""),
            )
            return False
        if not force and requested == self.current_mode:
            return True
        profile = self.mode_controller.switch_mode(requested, reason=reason, force=force)
        if profile is None:
            return False
        self.current_mode = str(profile.name or "IDLE").strip().upper() or "IDLE"
        self.reset_runtime_state()
        self._sync_aliases()
        return True

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.mode_controller.start_runtime()
        self.log.info("vision engine runtime started")
        self._emit_backend_lifecycle("started")

    def reset_runtime_state(self) -> None:
        self._last_frame_seq_seen = 0

    def set_inference_enabled(self, enable: bool) -> None:
        before = bool(self.predictor_manager.inference_enabled)
        self.predictor_manager.set_inference_enabled(bool(enable))
        self.infer_enabled = bool(self.predictor_manager.inference_enabled)
        if before != self.infer_enabled:
            self.log.info("inference %s", "enabled" if self.infer_enabled else "disabled")
            self._emit_capability_change(
                "inference",
                "enabled" if self.infer_enabled else "disabled",
                enabled=bool(self.infer_enabled),
            )

    def set_camera(self, name: str, enable: bool, cfg: Optional[dict] = None) -> None:
        if enable:
            self.camera_manager.ensure_camera(name, override=cfg)
        else:
            self.camera_manager.disable_camera(name)

    def set_model(self, name: str, enable: bool) -> None:
        if enable:
            self.predictor_manager.ensure_model(name)
        else:
            self.predictor_manager.disable_model()
        self._sync_aliases()

    def stop(self) -> None:
        self.running = False
        self.mode_controller.stop_runtime()
        self.mode_controller.close()
        self.camera_manager.release_all()
        self.predictor_manager.release_all()
        self.remote_manager.disable()
        self.table_edge_manager.release_all()
        self.preview_manager.disable()
        self.set_inference_enabled(False)
        self.current_mode = "IDLE"
        self.reset_runtime_state()
        self._sync_aliases()
        self.log.info("runtime stopped")
        self._emit_backend_lifecycle("stopped")

    def poll_runtime_results(self) -> None:
        """Refresh low-frequency status slots owned by control-plane facade."""
        self.tick(now_ts=time.time())

    def get_new_data(self) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, list]]]:
        """Compatibility API kept for tests and low-level debug tooling."""
        self.poll_runtime_results()
        frame_slot = self.mode_controller.read_slot("camera_frames")
        if not isinstance(frame_slot, dict):
            return None, None
        seq = int(frame_slot.get("seq", 0) or 0)
        if seq <= self._last_frame_seq_seen:
            return None, None
        self._last_frame_seq_seen = seq
        frames = frame_slot.get("payload")
        if not isinstance(frames, dict):
            return None, None
        local = dict(self.mode_controller.read_result("local_perception", default={}) or {})
        boxes = list(local.get("infer_boxes", []) or [])
        masks = list(local.get("infer_masks", []) or [])
        infer_res = None
        if boxes or masks or bool(local.get("has_infer")):
            infer_res = {"boxes": boxes, "masks": masks}
        return frames, infer_res

    def collect_tick_input(self, ts: float):
        return self.mode_controller.collect_tick_input(ts=ts)

    def snapshot(self) -> Dict[str, Any]:
        self._sync_aliases()
        frame_slot = self.mode_controller.read_slot("camera_frames") or {}
        current_frame_seq = int(frame_slot.get("seq", 0) or 0)
        has_new_data = current_frame_seq > int(self._last_frame_seq_seen)
        return {
            "current_mode": self.current_mode,
            "mode_controller": self.mode_controller.snapshot(),
            "external_mode_control_enabled": bool(self._external_mode_control_enabled),
            "enabled_cameras": sorted(self.cams.keys()),
            "active_model_name": self.active_model_name,
            "inference_enabled": bool(self.infer_enabled),
            "remote_enabled": bool(self.remote_enabled),
            "preview_enabled": bool(self.preview_enabled),
            "preview_exit_requested": bool(self.mode_controller.preview_exit_requested()),
            "has_new_data": bool(has_new_data),
            "capabilities": {
                "camera": self.camera_manager.snapshot(),
                "predictor": self.predictor_manager.snapshot(),
                "remote": self.remote_manager.snapshot(),
                "table_edge": self.table_edge_manager.snapshot(),
                "preview": self.preview_manager.snapshot(),
            },
        }


class ModeControlPort:
    """Narrow stage-facing mode control surface."""

    def __init__(self, mode_controller):
        self._mode_controller = mode_controller

    def current_mode(self) -> str:
        return self._mode_controller.current_mode()

    def switch_mode(self, name: str, reason: str = "", force: bool = False):
        return self._mode_controller.switch_mode(name=name, reason=reason, force=force)

    def tick_runtime(self, ts: Optional[float] = None) -> None:
        self._mode_controller.tick(now_ts=ts)

    def collect_tick_input(self, ts: float):
        return self._mode_controller.collect_tick_input(ts=ts)

    def push_stage_signals(self, signals: Dict[str, object]) -> None:
        self._mode_controller.push_stage_signals(dict(signals or {}))

    def preview_exit_requested(self) -> bool:
        return bool(self._mode_controller.preview_exit_requested())

    def snapshot(self) -> Dict[str, Any]:
        return dict(self._mode_controller.snapshot() or {})


class VisionRuntimeService:
    """App-facing runtime facade that hides engine internals."""

    def __init__(
        self,
        cfg: VisionServiceConfig,
        logger: Optional[logging.Logger] = None,
        event_sink: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self._engine = VisionEngine(cfg=cfg, logger=logger, event_sink=event_sink)
        self._mode_port = ModeControlPort(self._engine.mode_controller)

    def mode_control_port(self):
        return self._mode_port

    def set_external_mode_control(self, enabled: bool) -> None:
        self._engine.set_external_mode_control(enabled)

    def init(self) -> None:
        self._engine.init()

    def start(self) -> None:
        self._engine.start()

    def stop(self) -> None:
        self._engine.stop()

    def poll_runtime_results(self) -> None:
        self._engine.poll_runtime_results()

    def collect_tick_input(self, ts: float):
        return self._engine.collect_tick_input(ts=ts)

    def push_stage_signals(self, signals: Dict[str, object]) -> None:
        self._engine.push_stage_signals(signals)

    def preview_exit_requested(self) -> bool:
        return self._engine.preview_exit_requested()

    def snapshot(self) -> Dict[str, Any]:
        return self._engine.snapshot()
