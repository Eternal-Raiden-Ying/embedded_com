#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from typing import Any, Dict, Iterable, Optional

from .mode_profiles import ModeProfile


class ModeController:
    """Pure control-plane owner for mode profiles, scheduler, and runtime supervisor."""

    def __init__(
        self,
        scheduler,
        supervisor,
        logger=None,
        backend_event_sink=None,
        preview_allowed: bool = True,
    ):
        self.scheduler = scheduler
        self.supervisor = supervisor
        self.logger = logger
        self._backend_event_sink = backend_event_sink
        self._preview_allowed = bool(preview_allowed)
        self._profiles: Dict[str, ModeProfile] = {}
        self._current_mode: str = "SILENT"
        self._target_mode: Optional[str] = None
        self._last_switch_ts: float = 0.0
        self._generation = 0
        self._active_plan: Optional[Dict[str, Any]] = None
        self._last_switch_result: Dict[str, Any] = {
            "ok": True,
            "reason": "init",
            "requested_mode": "SILENT",
            "active_mode": "SILENT",
            "generation": 0,
        }

    def register_profile(self, profile: ModeProfile) -> None:
        self._profiles[str(profile.name or "").upper()] = profile

    def register_profiles(self, profiles: Iterable[ModeProfile]) -> None:
        for profile in profiles:
            self.register_profile(profile)

    def resolve_profile(self, name: str) -> Optional[ModeProfile]:
        return self._profiles.get(str(name or "").upper()) or self._profiles.get(str(name or ""))

    def current_mode(self) -> str:
        return self._current_mode

    def current_profile(self) -> Optional[ModeProfile]:
        return self.resolve_profile(self._current_mode)

    def current_generation(self) -> int:
        return int(self._generation)

    def current_plan(self) -> Optional[Dict[str, Any]]:
        if self._active_plan is None:
            return None
        return dict(self._active_plan)

    def switch_mode(
        self,
        name: str,
        reason: str = "",
        force: bool = False,
    ) -> Optional[ModeProfile]:
        switch_state = self.prepare_switch(name=name, reason=reason, force=force)
        if switch_state is None:
            return None
        if bool(switch_state.get("noop", False)):
            return self.commit_switch(switch_state, reason=reason or "mode_unchanged")

        plan = dict(switch_state.get("plan") or {})
        generation = int(switch_state.get("next_generation", self._generation + 1))

        previous_plan = self._active_plan
        previous_generation = self._generation
        self.scheduler.configure(plan=plan, generation=generation)
        ok = bool(self.supervisor.reconcile(plan=plan, generation=generation))
        if not ok:
            if previous_plan is not None:
                self.scheduler.configure(plan=previous_plan, generation=previous_generation)
            self.record_switch_failure(switch_state, reason="runtime_apply_failed")
            return None

        return self.commit_switch(switch_state, reason=reason)

    def _emit_backend_event(self, event_name: str, **fields: Any) -> None:
        if self._backend_event_sink is None:
            return
        try:
            self._backend_event_sink(str(event_name or "").strip().upper(), **dict(fields or {}))
        except Exception:
            pass

    def _log(self, message: str, **fields: Any) -> None:
        if self.logger is None:
            return
        extra = fields or None
        self.logger.info("%s%s", message, f" | {extra}" if extra else "")

    def _compile_plan(self, profile: ModeProfile) -> Dict[str, Any]:
        preview_enabled = bool(profile.preview.enabled and self._preview_allowed)
        remote_profile = profile.remote
        mode_name = str(profile.name or "SILENT").strip().upper() or "SILENT"
        # INIT / GRASP_REMOTE_INIT modes need remote_init_status for server health tracking
        routes = {
            "camera_frames": {"policy": "slot", "scope": "backend"},
            "frame_meta": {"policy": "slot", "scope": "stage"},
            "local_perception": {"policy": "slot", "scope": "stage"},
            "table_edge_obs": {"policy": "slot", "scope": "stage"},
            "remote_result": {"policy": "slot", "scope": "stage"},
            "runtime_status": {"policy": "slot", "scope": "backend"},
        }
        if mode_name in {"INIT", "GRASP_REMOTE_INIT"}:
            routes["remote_init_status"] = {"policy": "slot", "scope": "stage"}
        return {
            "mode": mode_name,
            "contract": {
                "results": {
                    "stage": ["frame_meta", "local_perception", "table_edge_obs", "remote_result"],
                    "backend": ["camera_frames", "runtime_status"],
                },
                "signals": {
                    "transient_only": True,
                    "owner": "stage_controller",
                },
                "stage_state": {
                    "persistent_only": True,
                    "owner": "stage_plan",
                },
                "capability": dict((profile.metadata or {}).get("contract") or {}),
            },
            "routes": routes,
            "capabilities": {
                "camera": {
                    "enabled_cameras": list(profile.enabled_cameras or ()),
                    "camera_overrides": dict(profile.camera_overrides or {}),
                },
                "predictor": {
                    "enabled": bool(profile.predictor_enabled),
                    "model_name": profile.predictor_model,
                },
                "remote": {
                    "enabled": bool(remote_profile.enabled),
                    "kind": str(remote_profile.kind).strip().lower(),
                    "action": str(remote_profile.action).strip().lower(),
                    "max_retries": int(remote_profile.max_retries),
                    "base_url": remote_profile.base_url,
                    "command": remote_profile.command,
                    "require_depth": bool(remote_profile.require_depth),
                    "timeout_s": float(remote_profile.timeout_s),
                    "rgb_encoding": str(remote_profile.rgb_encoding).strip().lower(),
                    "depth_encoding": str(remote_profile.depth_encoding).strip().lower(),
                    "rgb_quality": int(remote_profile.rgb_quality),
                    "depth_compression": int(remote_profile.depth_compression),
                    "metadata": dict(remote_profile.metadata or {}),
                },
                "table_edge": {
                    "enabled": bool(profile.table_edge.enabled),
                    "detector_mode": str(profile.table_edge.detector_mode),
                    "update_hz": float(profile.table_edge.update_hz),
                    "light_stride": int(profile.table_edge.light_stride),
                    "fast_plane_stride": int(profile.table_edge.fast_plane_stride),
                    "require_yolo_confirm": bool(profile.table_edge.require_yolo_confirm),
                    "static_roi_enabled": bool(profile.table_edge.static_roi_enabled),
                    "camera_pitch_deg": float(profile.table_edge.camera_pitch_deg),
                    "camera_height_m": float(profile.table_edge.camera_height_m),
                    "camera_roll_deg": float(profile.table_edge.camera_roll_deg),
                    "camera_yaw_deg": float(profile.table_edge.camera_yaw_deg),
                    "table_height_m": float(profile.table_edge.table_height_m),
                    "front_face_z_min_m": float(profile.table_edge.front_face_z_min_m),
                    "front_face_z_max_m": float(profile.table_edge.front_face_z_max_m),
                    "min_vertical_z_span_m": float(profile.table_edge.min_vertical_z_span_m),
                    "min_vertical_support_points": int(profile.table_edge.min_vertical_support_points),
                    "x_bin_width_m": float(profile.table_edge.x_bin_width_m),
                    "y_cluster_bin_m": float(profile.table_edge.y_cluster_bin_m),
                    "min_front_face_columns": int(profile.table_edge.min_front_face_columns),
                    "min_front_face_x_span_m": float(profile.table_edge.min_front_face_x_span_m),
                    "front_cluster_gap_m": float(profile.table_edge.front_cluster_gap_m),
                    "max_yaw_abs_rad": float(profile.table_edge.max_yaw_abs_rad),
                    "enable_yolo_in_plane_only": bool(profile.table_edge.enable_yolo_in_plane_only),
                    "yolo_table_min_conf": float(profile.table_edge.yolo_table_min_conf),
                },
                "preview": {
                    "enabled": preview_enabled,
                    "sink_name": profile.preview.sink_name,
                    "overlay_enabled": bool(profile.preview.overlay_enabled),
                    "window_name": profile.preview.window_name,
                    "metadata": dict(profile.preview.metadata or {}),
                },
            },
            "loop_hz": profile.loop_hz,
            "send_hz": profile.send_hz,
        }

    def _verify_plan(self, plan: Dict[str, Any]) -> bool:
        mode_name = str((plan or {}).get("mode") or "").strip().upper()
        if not mode_name:
            return False
        routes = dict((plan or {}).get("routes") or {})
        required = {
            "camera_frames": ("slot", "backend"),
            "frame_meta": ("slot", "stage"),
            "local_perception": ("slot", "stage"),
            "table_edge_obs": ("slot", "stage"),
            "remote_result": ("slot", "stage"),
            "runtime_status": ("slot", "backend"),
        }
        for route_name, (policy, scope) in required.items():
            cfg = dict(routes.get(route_name) or {})
            if str(cfg.get("policy") or "").strip().lower() != policy:
                return False
            if str(cfg.get("scope") or "").strip().lower() != scope:
                return False
        return True

    def prepare_switch(self, name: str, reason: str = "", force: bool = False) -> Optional[Dict[str, Any]]:
        requested = str(name or "SILENT").strip().upper() or "SILENT"
        profile = self.resolve_profile(requested)
        if profile is None:
            self._log("mode switch failed", requested_mode=requested, reason=reason)
            self._last_switch_result = {
                "ok": False,
                "reason": "mode_not_registered",
                "requested_mode": requested,
                "active_mode": self._current_mode,
                "generation": int(self._generation),
            }
            self._emit_backend_event(
                "BACKEND_FAILURE",
                level="error",
                failure_type="mode_change_failed",
                requested_mode=requested,
                reason=str(reason or ""),
            )
            return None
        if not force and profile.name == self._current_mode:
            return {
                "noop": True,
                "profile": profile,
                "plan": dict(self._active_plan or self._compile_plan(profile)),
                "requested_mode": requested,
                "previous_mode": self._current_mode,
                "next_generation": int(self._generation),
            }
        plan = self._compile_plan(profile)
        if not self._verify_plan(plan):
            self._last_switch_result = {
                "ok": False,
                "reason": "mode_plan_invalid",
                "requested_mode": requested,
                "active_mode": self._current_mode,
                "generation": int(self._generation),
            }
            self._emit_backend_event(
                "BACKEND_FAILURE",
                level="error",
                failure_type="mode_plan_invalid",
                requested_mode=requested,
                current_mode=str(self._current_mode or "SILENT").strip().upper(),
                reason=str(reason or ""),
            )
            return None
        return {
            "noop": False,
            "profile": profile,
            "plan": plan,
            "requested_mode": requested,
            "previous_mode": self._current_mode,
            "next_generation": int(self._generation) + 1,
        }

    def commit_switch(self, switch_state: Dict[str, Any], reason: str = "") -> Optional[ModeProfile]:
        if not isinstance(switch_state, dict):
            return None
        profile = switch_state.get("profile")
        if profile is None:
            return None
        if bool(switch_state.get("noop", False)):
            self._target_mode = str(profile.name or self._current_mode).strip().upper() or self._current_mode
            self._last_switch_result = {
                "ok": True,
                "reason": str(reason or "mode_unchanged").strip() or "mode_unchanged",
                "requested_mode": str(switch_state.get("requested_mode") or self._current_mode).strip().upper(),
                "active_mode": self._current_mode,
                "generation": int(self._generation),
            }
            return profile

        previous_mode = str(switch_state.get("previous_mode") or self._current_mode).strip().upper() or "SILENT"
        next_generation = int(switch_state.get("next_generation", self._generation + 1))
        plan = dict(switch_state.get("plan") or {})
        self._active_plan = dict(plan or {})
        self._target_mode = str(profile.name or previous_mode).strip().upper() or previous_mode
        self._current_mode = self._target_mode
        self._generation = next_generation
        self._last_switch_ts = time.time()
        self._last_switch_result = {
            "ok": True,
            "reason": str(reason or "mode_apply").strip() or "mode_apply",
            "requested_mode": str(switch_state.get("requested_mode") or self._current_mode).strip().upper(),
            "active_mode": self._current_mode,
            "generation": int(self._generation),
        }
        self._log("mode switched", mode=self._current_mode, reason=reason)
        self._emit_backend_event(
            "BACKEND_MODE_CHANGED",
            previous_mode=previous_mode,
            current_mode=self._current_mode,
            reason=str(reason or "mode_apply").strip() or "mode_apply",
            contract=dict((self._active_plan or {}).get("contract") or {}),
        )
        return profile

    def record_switch_failure(self, switch_state: Optional[Dict[str, Any]], reason: str = "mode_apply_failed") -> None:
        payload = dict(switch_state or {})
        requested_mode = str(payload.get("requested_mode") or self._target_mode or self._current_mode or "SILENT").strip().upper() or "SILENT"
        previous_mode = str(payload.get("previous_mode") or self._current_mode or "SILENT").strip().upper() or "SILENT"
        next_generation = int(payload.get("next_generation", self._generation + 1) or (self._generation + 1))
        self._target_mode = previous_mode
        self._last_switch_result = {
            "ok": False,
            "reason": str(reason or "mode_apply_failed").strip() or "mode_apply_failed",
            "requested_mode": requested_mode,
            "active_mode": previous_mode,
            "generation": int(self._generation),
            "failed_generation": next_generation,
        }

    def start_runtime(self) -> None:
        self.scheduler.start_runtime()
        if self._active_plan is not None:
            self.scheduler.configure(plan=self._active_plan, generation=self._generation)
        self.supervisor.start_runtime()

    def stop_runtime(self) -> None:
        self.supervisor.stop_runtime()
        self.scheduler.stop_runtime()

    def runtime_snapshot(self) -> Dict[str, Any]:
        return {
            "runtime_running": True,
            "active_runtime_generation": int(self._generation),
            "active_runtime_plan": dict(self._active_plan or {}),
            "active_runtime_mode": str((self._active_plan or {}).get("mode") or "SILENT").strip().upper() or "SILENT",
            "scheduler": self.scheduler.snapshot(),
            "runtime_supervisor": self.supervisor.snapshot(),
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "current_mode": self._current_mode,
            "target_mode": self._target_mode,
            "registered_modes": sorted(self._profiles.keys()),
            "last_switch_ts": self._last_switch_ts,
            "generation": int(self._generation),
            "active_plan": dict(self._active_plan or {}),
            "last_switch_result": dict(self._last_switch_result),
        }

    def reset(self) -> None:
        self._current_mode = "SILENT"
        self._target_mode = None
        self._last_switch_ts = 0.0
        self._generation = 0
        self._active_plan = None
        self._last_switch_result = {
            "ok": True,
            "reason": "reset",
            "requested_mode": "SILENT",
            "active_mode": "SILENT",
            "generation": 0,
        }

    def close(self) -> None:
        self.reset()
