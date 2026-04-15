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
        self._preview_context: Dict[str, Any] = {
            "stage": "IDLE",
            "mode": "IDLE",
            "session_id": None,
            "req_id": None,
            "epoch": 0,
        }
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
        self.preview_manager = PreviewManager(
            sink=NullPreviewSink(),
            logger=self.log,
            capability_sink=self._on_simple_capability_change,
        )

        self.mode_controller = ModeController(
            camera_manager=self.camera_manager,
            predictor_manager=self.predictor_manager,
            remote_manager=self.remote_manager,
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

    def _sync_aliases(self) -> None:
        self.predictor = self.predictor_manager.predictor
        self.active_model_name = self.predictor_manager.active_model_name
        self.remote_enabled = bool(self.remote_manager.enabled)
        self.preview_enabled = bool(self.preview_manager.enabled)
        self.infer_enabled = bool(self.predictor_manager.inference_enabled)

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

    def set_preview_context(
        self,
        *,
        stage: str,
        mode: str,
        session_id: Optional[str] = None,
        req_id: Optional[str] = None,
        epoch: int = 0,
    ) -> None:
        with self.lock:
            self._preview_context = {
                "stage": str(stage or "IDLE").strip().upper() or "IDLE",
                "mode": str(mode or "IDLE").strip().upper() or "IDLE",
                "session_id": session_id,
                "req_id": req_id,
                "epoch": int(epoch),
            }

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
        self.preview_manager.disable()
        self.set_inference_enabled(False)
        self.current_mode = "IDLE"
        self.reset_runtime_state()
        self._sync_aliases()
        self.log.info("runtime stopped")
        self._emit_backend_lifecycle("stopped")

    def _runtime_status_payload(self) -> Dict[str, Any]:
        with self.lock:
            ctx = dict(self._preview_context or {})
        return {
            "stage": str(ctx.get("stage") or "IDLE").strip().upper() or "IDLE",
            "mode": str(ctx.get("mode") or self.current_mode).strip().upper() or "IDLE",
            "session_id": ctx.get("session_id"),
            "req_id": ctx.get("req_id"),
            "epoch": int(ctx.get("epoch", 0) or 0),
            "ts": time.time(),
        }

    def poll_runtime_results(self) -> None:
        """Refresh low-frequency status slots owned by control-plane facade."""
        self.tick(now_ts=time.time())
        generation = self.mode_controller.current_generation()
        self.mode_controller.publish_result("runtime_status", self._runtime_status_payload(), generation=generation)
        self.mode_controller.publish_result("remote_result", self.remote_manager.result_summary(), generation=generation)

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
                "preview": self.preview_manager.snapshot(),
            },
        }
