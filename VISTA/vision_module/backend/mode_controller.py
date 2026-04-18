#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from typing import Any, Dict, Iterable, Optional

from .mode_profiles import ModeProfile


class ModeController:
    """Pure control-plane owner for mode profiles and mode switch state."""

    def __init__(
        self,
        logger=None,
        backend_event_sink=None,
        preview_allowed: bool = True,
    ):
        self.logger = logger
        self._backend_event_sink = backend_event_sink
        self._preview_allowed = bool(preview_allowed)
        self._profiles: Dict[str, ModeProfile] = {}
        self._current_mode: str = "IDLE"
        self._target_mode: Optional[str] = None
        self._last_switch_ts: float = 0.0
        self._generation = 0
        self._active_plan: Optional[Dict[str, Any]] = None
        self._last_switch_result: Dict[str, Any] = {
            "ok": True,
            "reason": "init",
            "requested_mode": "IDLE",
            "active_mode": "IDLE",
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
        apply_mode_plan=None,
    ) -> Optional[ModeProfile]:
        switch_state = self.prepare_switch(name=name, reason=reason, force=force)
        if switch_state is None:
            return None
        if bool(switch_state.get("noop", False)):
            return self.commit_switch(switch_state, reason=reason or "mode_unchanged")

        plan = dict(switch_state.get("plan") or {})
        generation = int(switch_state.get("next_generation", self._generation + 1))
        if callable(apply_mode_plan):
            try:
                ok = bool(apply_mode_plan(plan=plan, generation=generation))
            except Exception:
                ok = False
            if not ok:
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
        return {
            "mode": str(profile.name or "IDLE").strip().upper() or "IDLE",
            "contract": {
                "results": {
                    "stage": ["frame_meta", "local_perception", "remote_result"],
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
            "routes": {
                "camera_frames": {"policy": "slot", "scope": "backend"},
                "frame_meta": {"policy": "slot", "scope": "stage"},
                "local_perception": {"policy": "slot", "scope": "stage"},
                "remote_result": {"policy": "slot", "scope": "stage"},
                "runtime_status": {"policy": "slot", "scope": "backend"},
                "remote_cmd": {"policy": "event", "scope": "backend"},
                "remote_ack": {"policy": "event", "scope": "backend"},
            },
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
                    "base_url": remote_profile.base_url,
                    "command": remote_profile.command,
                    "require_depth": bool(remote_profile.require_depth),
                    "require_segmentation": bool(remote_profile.require_segmentation),
                    "timeout_s": float(remote_profile.timeout_s),
                    "metadata": dict(remote_profile.metadata or {}),
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
            "remote_result": ("slot", "stage"),
            "runtime_status": ("slot", "backend"),
            "remote_cmd": ("event", "backend"),
            "remote_ack": ("event", "backend"),
        }
        for route_name, (policy, scope) in required.items():
            cfg = dict(routes.get(route_name) or {})
            if str(cfg.get("policy") or "").strip().lower() != policy:
                return False
            if str(cfg.get("scope") or "").strip().lower() != scope:
                return False
        return True

    def prepare_switch(self, name: str, reason: str = "", force: bool = False) -> Optional[Dict[str, Any]]:
        requested = str(name or "IDLE").strip().upper() or "IDLE"
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
                current_mode=str(self._current_mode or "IDLE").strip().upper(),
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

        previous_mode = str(switch_state.get("previous_mode") or self._current_mode).strip().upper() or "IDLE"
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
        requested_mode = str(payload.get("requested_mode") or self._target_mode or self._current_mode or "IDLE").strip().upper() or "IDLE"
        previous_mode = str(payload.get("previous_mode") or self._current_mode or "IDLE").strip().upper() or "IDLE"
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
        self._current_mode = "IDLE"
        self._target_mode = None
        self._last_switch_ts = 0.0
        self._generation = 0
        self._active_plan = None
        self._last_switch_result = {
            "ok": True,
            "reason": "reset",
            "requested_mode": "IDLE",
            "active_mode": "IDLE",
            "generation": 0,
        }

    def close(self) -> None:
        self.reset()
