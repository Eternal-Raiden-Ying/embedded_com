#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import threading
from typing import Any, Callable, Dict, Optional

from .camera_manager import CameraManager
from .predictor_manager import PredictorManager
from .preview import NullPreviewSink
from .preview.manager import PreviewManager
from .remote.client import RemoteGraspClient
from .remote.manager import RemoteManager
from .runtime_supervisor import RuntimeSupervisor
from .scheduler import Scheduler
from .table_edge_manager import TableEdgeManager
from ..config.schema import VisionServiceConfig


class VisionEngine:
    """Runtime assembly and execution root for the VISTA backend."""

    def __init__(
        self,
        cfg: VisionServiceConfig,
        logger: Optional[logging.Logger] = None,
        event_sink: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.cfg = cfg
        self.log = logger or logging.getLogger("vision.engine")
        self._event_sink = event_sink
        self.lock = threading.RLock()
        self.running = False
        self._active_runtime_plan: Optional[Dict[str, Any]] = None
        self._active_runtime_generation = 0

        self.scheduler = Scheduler()
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
        self.runtime_supervisor = RuntimeSupervisor(
            scheduler=self.scheduler,
            camera_manager=self.camera_manager,
            predictor_manager=self.predictor_manager,
            remote_manager=self.remote_manager,
            table_edge_manager=self.table_edge_manager,
            preview_manager=self.preview_manager,
            logger=self.log,
            backend_event_sink=self._emit_event,
        )

        self.cams = self.camera_manager.cams
        self.predictor = self.predictor_manager.predictor
        self.active_model_name = self.predictor_manager.active_model_name
        self.remote_enabled = bool(self.remote_manager.enabled)
        self.preview_enabled = bool(self.preview_manager.enabled)
        self.infer_enabled = bool(self.predictor_manager.inference_enabled)
        self.table_edge_enabled = bool(self.runtime_supervisor.snapshot().get("table_edge", {}).get("runtime_running", False))

    def _sync_aliases(self) -> None:
        self.predictor = self.predictor_manager.predictor
        self.active_model_name = self.predictor_manager.active_model_name
        self.remote_enabled = bool(self.remote_manager.enabled)
        self.preview_enabled = bool(self.preview_manager.enabled)
        self.infer_enabled = bool(self.predictor_manager.inference_enabled)
        self.table_edge_enabled = bool(self.runtime_supervisor.snapshot().get("table_edge", {}).get("runtime_running", False))

    def _emit_event(self, event_name: str, **fields: Any) -> None:
        if self._event_sink is None:
            return
        try:
            self._event_sink(str(event_name or "").strip().upper(), fields)
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
        self.log.info("engine init: runtime service ready")
        self._emit_backend_lifecycle(
            "initialized",
            registered_modes=[],
        )

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.scheduler.start_runtime()
        if self._active_runtime_plan is not None:
            self.scheduler.configure(plan=self._active_runtime_plan, generation=self._active_runtime_generation)
        self.runtime_supervisor.start_runtime()
        self._sync_aliases()
        self.log.info("vision runtime started")
        self._emit_backend_lifecycle(
            "started",
            active_runtime_mode=str((self._active_runtime_plan or {}).get("mode") or "IDLE").strip().upper() or "IDLE",
            generation=int(self._active_runtime_generation),
        )

    def stop(self) -> None:
        if not self.running and not self.runtime_supervisor.snapshot().get("runtime_running"):
            return
        self.running = False
        self.runtime_supervisor.stop_runtime()
        self.scheduler.stop_runtime()
        self._sync_aliases()
        self.log.info("runtime stopped")
        self._emit_backend_lifecycle("stopped")

    def apply_mode_plan(self, plan: Dict[str, Any], generation: int) -> bool:
        plan_payload = dict(plan or {})
        target_generation = int(generation)
        if not self.running:
            self._active_runtime_plan = plan_payload
            self._active_runtime_generation = target_generation
            return bool(self.runtime_supervisor.reconcile(plan=plan_payload, generation=target_generation))

        previous_plan = self._active_runtime_plan
        previous_generation = self._active_runtime_generation
        self.scheduler.configure(plan=plan_payload, generation=target_generation)
        ok = bool(self.runtime_supervisor.reconcile(plan=plan_payload, generation=target_generation))
        if not ok:
            if previous_plan is not None:
                self.scheduler.configure(plan=previous_plan, generation=previous_generation)
            self._emit_backend_failure(
                "mode_apply_incomplete",
                requested_mode=str(plan_payload.get("mode") or "IDLE").strip().upper() or "IDLE",
                previous_mode=str((self._active_runtime_plan or {}).get("mode") or "IDLE").strip().upper() or "IDLE",
                capability_snapshot=self.runtime_supervisor.snapshot(),
            )
            return False

        self._active_runtime_plan = plan_payload
        self._active_runtime_generation = target_generation
        self._sync_aliases()
        return True

    def collect_tick_input(self, ts: float, route_filter: set = None):
        return self.scheduler.collect_tick_input(ts=ts, route_filter=route_filter)

    def push_stage_signals(self, signals: Dict[str, object]) -> None:
        self.scheduler.push_stage_signals(dict(signals or {}))

    def publish_result(self, route: str, payload) -> bool:
        return self.scheduler.publish_result(route, payload, generation=self._active_runtime_generation)

    def publish_event(self, route: str, payload) -> bool:
        return self.scheduler.publish_event(route, payload, generation=self._active_runtime_generation)

    def active_runtime_plan(self) -> Optional[Dict[str, Any]]:
        if self._active_runtime_plan is None:
            return None
        return dict(self._active_runtime_plan)

    def active_runtime_generation(self) -> int:
        return int(self._active_runtime_generation)

    def runtime_snapshot(self) -> Dict[str, Any]:
        self._sync_aliases()
        return {
            "runtime_running": bool(self.running),
            "active_runtime_generation": int(self._active_runtime_generation),
            "active_runtime_plan": dict(self._active_runtime_plan or {}),
            "active_runtime_mode": str((self._active_runtime_plan or {}).get("mode") or "IDLE").strip().upper() or "IDLE",
            "scheduler": self.scheduler.snapshot(),
            "runtime_supervisor": self.runtime_supervisor.snapshot(),
            "active_model_name": self.active_model_name,
            "inference_enabled": bool(self.infer_enabled),
            "remote_enabled": bool(self.remote_enabled),
            "preview_enabled": bool(self.preview_enabled),
            "capabilities": {
                "camera": self.camera_manager.snapshot(),
                "predictor": self.predictor_manager.snapshot(),
                "remote": self.remote_manager.snapshot(),
                "table_edge": self.table_edge_manager.snapshot(),
                "preview": self.preview_manager.snapshot(),
            },
        }

    def snapshot(self) -> Dict[str, Any]:
        return self.runtime_snapshot()
