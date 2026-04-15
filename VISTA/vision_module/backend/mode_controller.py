#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from typing import Dict, Iterable, Optional, Set

from .preview import NullPreviewSink, OpenCVPreviewSink
from .scheduler import Scheduler

from .mode_profiles import ModeProfile


class ModeController:
    """Apply mode profiles onto capability managers.

    The final implementation should perform diff-based capability updates,
    delayed resource release, and mode-level diagnostics.
    """

    def __init__(
        self,
        camera_manager=None,
        predictor_manager=None,
        remote_manager=None,
        preview_manager=None,
        logger=None,
        backend_event_sink=None,
    ):
        self.camera_manager = camera_manager
        self.predictor_manager = predictor_manager
        self.remote_manager = remote_manager
        self.preview_manager = preview_manager
        self.logger = logger
        self._backend_event_sink = backend_event_sink
        self._profiles: Dict[str, ModeProfile] = {}
        self._current_mode: str = "IDLE"
        self._target_mode: Optional[str] = None
        self._last_switch_ts: float = 0.0
        self._pending_camera_release: Dict[str, float] = {}
        self._pending_predictor_release_ts: Optional[float] = None
        self._inference_setter = None
        self._preview_allowed = True
        self._generation = 0
        self._runtime_running = False
        self._scheduler = Scheduler()
        for mgr in (self.camera_manager, self.predictor_manager, self.remote_manager, self.preview_manager):
            binder = getattr(mgr, "bind_runtime", None)
            if callable(binder):
                try:
                    binder(self._scheduler, self.current_generation)
                except Exception:
                    pass
        self._last_switch_result: Dict[str, object] = {
            "ok": True,
            "reason": "init",
            "requested_mode": "IDLE",
            "active_mode": "IDLE",
        }
        for mgr in (self.camera_manager, self.predictor_manager, self.preview_manager):
            binder = getattr(mgr, "bind_runtime", None)
            if callable(binder):
                try:
                    binder(self._scheduler, self.current_generation)
                except Exception:
                    pass

    def register_profile(self, profile: ModeProfile) -> None:
        """Register one named mode profile."""
        self._profiles[profile.name] = profile

    def register_profiles(self, profiles: Iterable[ModeProfile]) -> None:
        """Register multiple mode profiles during engine bootstrap."""
        for profile in profiles:
            self.register_profile(profile)

    def resolve_profile(self, name: str) -> Optional[ModeProfile]:
        """Resolve one mode profile by name."""
        return self._profiles.get(str(name or "").upper()) or self._profiles.get(str(name or ""))

    def current_mode(self) -> str:
        """Return the currently active mode name."""
        return self._current_mode

    def current_profile(self) -> Optional[ModeProfile]:
        """Return the currently active mode profile."""
        return self.resolve_profile(self._current_mode)

    def bind_runtime_controls(self, set_inference=None, preview_allowed: bool = True) -> None:
        """Bind runtime callbacks that live in VisionEngine."""
        self._inference_setter = set_inference
        self._preview_allowed = bool(preview_allowed)

    def bind_backend_event_sink(self, backend_event_sink) -> None:
        """Bind backend event sink from VisionEngine."""
        self._backend_event_sink = backend_event_sink

    def start_runtime(self) -> None:
        self._runtime_running = True
        self._scheduler.start_runtime()
        for mgr in (self.camera_manager, self.predictor_manager, self.remote_manager, self.preview_manager):
            starter = getattr(mgr, "start_runtime", None)
            if callable(starter):
                try:
                    starter()
                except Exception:
                    pass

    def stop_runtime(self) -> None:
        self._runtime_running = False
        for mgr in (self.preview_manager, self.remote_manager, self.predictor_manager, self.camera_manager):
            stopper = getattr(mgr, "stop_runtime", None)
            if callable(stopper):
                try:
                    stopper()
                except Exception:
                    pass
        self._scheduler.stop_runtime()

    def preview_exit_requested(self) -> bool:
        checker = getattr(self.preview_manager, "exit_requested", None)
        if not callable(checker):
            return False
        try:
            return bool(checker())
        except Exception:
            return False

    def current_generation(self) -> int:
        return int(self._generation)

    def publish_runtime_results(
        self,
        results: Dict[str, object],
        snapshot: Optional[Dict[str, object]] = None,
        generation: Optional[int] = None,
    ) -> None:
        self._scheduler.publish_results(
            dict(results or {}),
            snapshot=dict(snapshot or {}),
            generation=(self._generation if generation is None else int(generation)),
        )

    def collect_tick_input(self, ts: float):
        return self._scheduler.collect_tick_input(ts=ts)

    def push_stage_signals(self, signals: Dict[str, object]) -> None:
        self._scheduler.push_stage_signals(dict(signals or {}))

    def read_slot(self, route: str):
        return self._scheduler.read_slot(route)

    def read_result(self, route: str, default=None):
        return self._scheduler.read_result(route, default=default)

    def publish_result(self, route: str, payload, generation: Optional[int] = None) -> bool:
        target_generation = self._generation if generation is None else int(generation)
        return self._scheduler.publish_result(route, payload, generation=target_generation)

    def publish_event(self, route: str, payload, generation: Optional[int] = None) -> bool:
        target_generation = self._generation if generation is None else int(generation)
        return self._scheduler.publish_event(route, payload, generation=target_generation)

    def _emit_backend_event(self, event_name: str, **fields) -> None:
        if self._backend_event_sink is None:
            return
        try:
            self._backend_event_sink(str(event_name or "").strip().upper(), **dict(fields or {}))
        except Exception:
            pass

    def _log(self, message: str, **fields) -> None:
        if self.logger is None:
            return
        extra = fields or None
        self.logger.info("%s%s", message, f" | {extra}" if extra else "")

    def _desired_camera_set(self, profile: ModeProfile) -> Set[str]:
        return set(profile.enabled_cameras or ())

    def _apply_preview(self, profile: ModeProfile) -> None:
        if self.preview_manager is None:
            return
        enabled = bool(profile.preview.enabled and self._preview_allowed)
        sink_name = str(profile.preview.sink_name or "null").strip().lower() if enabled else "null"
        window_name = str(profile.preview.window_name or "VISTA App Dashboard")
        current_sink = getattr(self.preview_manager, "sink", None)
        current_name = getattr(current_sink, "sink_name", "null") if current_sink is not None else "null"
        if sink_name != current_name:
            if sink_name == "opencv":
                self.preview_manager.set_sink(OpenCVPreviewSink(window_name=window_name))
            else:
                self.preview_manager.set_sink(NullPreviewSink())

        if enabled:
            self.preview_manager.enable()
        else:
            self.preview_manager.disable()

    def _apply_remote(self, profile: ModeProfile) -> None:
        if self.remote_manager is None:
            return
        remote_profile = profile.remote
        if self.remote_manager.client is not None and remote_profile.base_url:
            self.remote_manager.client.configure(remote_profile.base_url)
        if remote_profile.enabled:
            self.remote_manager.enable()
        else:
            self.remote_manager.disable()

    def _apply_profile(self, profile: ModeProfile, force: bool = False) -> None:
        now = time.time()
        cooldown_s = max(0.0, float(profile.release_cooldown_s))
        desired_cameras = self._desired_camera_set(profile)

        if self.camera_manager is not None:
            for camera_name in sorted(desired_cameras):
                override = dict(profile.camera_overrides.get(camera_name, {})) if profile.camera_overrides else None
                self.camera_manager.ensure_camera(camera_name, override=override)
                self._pending_camera_release.pop(camera_name, None)

            current_cameras = self.camera_manager.active_names()
            removed = current_cameras - desired_cameras
            for camera_name in sorted(removed):
                if cooldown_s > 0.0 and not force:
                    self._pending_camera_release[camera_name] = now + cooldown_s
                else:
                    self.camera_manager.disable_camera(camera_name)
                    self._pending_camera_release.pop(camera_name, None)

        if self._inference_setter is not None:
            self._inference_setter(bool(profile.predictor_enabled))

        if self.predictor_manager is not None:
            if profile.predictor_enabled:
                model_name = profile.predictor_model or self.predictor_manager.active_model_name
                if model_name:
                    self.predictor_manager.ensure_model(str(model_name))
                self._pending_predictor_release_ts = None
            else:
                if cooldown_s > 0.0 and not force:
                    self._pending_predictor_release_ts = now + cooldown_s
                else:
                    self.predictor_manager.disable_model()
                    self._pending_predictor_release_ts = None

        self._apply_remote(profile)
        self._apply_preview(profile)

    def _profile_apply_satisfied(self, profile: ModeProfile) -> bool:
        desired_cameras = self._desired_camera_set(profile)

        if self.camera_manager is not None:
            active_cameras = set(self.camera_manager.active_names())
            missing = desired_cameras - active_cameras
            if missing:
                self._log("mode apply incomplete: missing cameras", mode=profile.name, missing=sorted(missing))
                return False

        if self.predictor_manager is not None and profile.predictor_enabled:
            if not self.predictor_manager.is_ready():
                self._log("mode apply incomplete: predictor not ready", mode=profile.name)
                return False
            required_model = str(profile.predictor_model or "").strip()
            if required_model and str(self.predictor_manager.active_model_name or "").strip() != required_model:
                self._log(
                    "mode apply incomplete: predictor model mismatch",
                    mode=profile.name,
                    expected_model=required_model,
                    active_model=self.predictor_manager.active_model_name,
                )
                return False

        if self.remote_manager is not None:
            expected_remote = bool(profile.remote.enabled)
            if bool(self.remote_manager.enabled) != expected_remote:
                self._log(
                    "mode apply incomplete: remote state mismatch",
                    mode=profile.name,
                    expected_remote=expected_remote,
                    current_remote=bool(self.remote_manager.enabled),
                )
                return False

        if self.preview_manager is not None:
            expected_preview = bool(profile.preview.enabled and self._preview_allowed)
            if bool(self.preview_manager.enabled) != expected_preview:
                self._log(
                    "mode apply incomplete: preview state mismatch",
                    mode=profile.name,
                    expected_preview=expected_preview,
                    current_preview=bool(self.preview_manager.enabled),
                )
                return False

        return True

    def _compile_plan(self, profile: ModeProfile) -> Dict[str, object]:
        return {
            "mode": str(profile.name or "IDLE").strip().upper() or "IDLE",
            "enabled_cameras": list(profile.enabled_cameras or ()),
            "camera_overrides": dict(profile.camera_overrides or {}),
            "predictor_enabled": bool(profile.predictor_enabled),
            "predictor_model": profile.predictor_model,
            "remote_enabled": bool(profile.remote.enabled),
            "preview_enabled": bool(profile.preview.enabled and self._preview_allowed),
            "loop_hz": profile.loop_hz,
            "send_hz": profile.send_hz,
            "release_cooldown_s": float(profile.release_cooldown_s),
            "contract": dict((profile.metadata or {}).get("contract") or {}),
            "routes": {
                "camera_frames": {"policy": "slot", "scope": "backend"},
                "frame_meta": {"policy": "slot", "scope": "stage"},
                "local_perception": {"policy": "slot", "scope": "stage"},
                "remote_result": {"policy": "slot", "scope": "stage"},
                "runtime_status": {"policy": "slot", "scope": "backend"},
                "preview_exit": {"policy": "event", "scope": "backend"},
            },
        }

    def _verify_plan(self, plan: Dict[str, object]) -> bool:
        mode_name = str((plan or {}).get("mode") or "").strip().upper()
        if not mode_name:
            return False
        cameras = (plan or {}).get("enabled_cameras")
        if cameras is not None and not isinstance(cameras, list):
            return False
        return True

    def _capability_snapshot(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "current_mode": self._current_mode,
            "target_mode": self._target_mode,
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
        if self.preview_manager is not None:
            try:
                payload["preview"] = self.preview_manager.snapshot()
            except Exception:
                payload["preview"] = {"error": "snapshot_failed"}
        return payload

    def switch_mode(self, name: str, reason: str = "", force: bool = False) -> Optional[ModeProfile]:
        """Request a mode transition.

        Later implementation should apply the profile to camera, predictor,
        remote, and preview managers here.
        """
        requested = str(name or "IDLE").strip().upper() or "IDLE"
        profile = self.resolve_profile(requested)
        if profile is None:
            self._log("mode switch failed", requested_mode=requested, reason=reason)
            self._last_switch_result = {
                "ok": False,
                "reason": "mode_not_registered",
                "requested_mode": requested,
                "active_mode": self._current_mode,
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
            return profile
        previous_mode = self._current_mode
        previous_profile = self.current_profile()
        self._apply_profile(profile, force=force)
        if not self._profile_apply_satisfied(profile):
            rollback_attempted = False
            rollback_verified = False
            rollback_target = previous_mode
            rollback_error = ""
            self._emit_backend_event(
                "BACKEND_FAILURE",
                level="error",
                failure_type="mode_apply_incomplete",
                requested_mode=requested,
                previous_mode=str(previous_mode or "IDLE").strip().upper(),
                reason=str(reason or ""),
                contract=(profile.metadata or {}).get("contract"),
            )
            if previous_profile is not None and previous_profile.name != profile.name:
                try:
                    rollback_attempted = True
                    rollback_target = previous_profile.name
                    self._apply_profile(previous_profile, force=True)
                    rollback_verified = self._profile_apply_satisfied(previous_profile)
                except Exception:
                    rollback_error = "rollback_apply_exception"
            else:
                rollback_verified = True
            if not rollback_verified:
                self._emit_backend_event(
                    "BACKEND_FAILURE",
                    level="error",
                    failure_type="mode_rollback_failed",
                    requested_mode=requested,
                    previous_mode=str(previous_mode or "IDLE").strip().upper(),
                    rollback_target=str(rollback_target or previous_mode or "IDLE").strip().upper(),
                    rollback_attempted=bool(rollback_attempted),
                    rollback_verified=bool(rollback_verified),
                    rollback_error=str(rollback_error or ""),
                    capability_snapshot=self._capability_snapshot(),
                )
            self._last_switch_result = {
                "ok": False,
                "reason": "mode_apply_incomplete",
                "requested_mode": requested,
                "active_mode": previous_mode,
                "rollback_attempted": bool(rollback_attempted),
                "rollback_verified": bool(rollback_verified),
                "rollback_target": str(rollback_target or previous_mode or "IDLE").strip().upper(),
                "rollback_error": str(rollback_error or ""),
            }
            self._target_mode = previous_mode
            return None
        self._target_mode = profile.name
        self._current_mode = profile.name
        self._last_switch_ts = time.time()
        self._generation += 1
        plan = self._compile_plan(profile)
        if not self._verify_plan(plan):
            self._emit_backend_event(
                "BACKEND_FAILURE",
                level="error",
                failure_type="mode_plan_invalid",
                requested_mode=requested,
                current_mode=str(self._current_mode or "IDLE").strip().upper(),
                reason=str(reason or ""),
            )
        else:
            self._scheduler.configure(plan=plan, generation=self._generation)
        self._last_switch_result = {
            "ok": True,
            "reason": str(reason or "mode_apply").strip() or "mode_apply",
            "requested_mode": requested,
            "active_mode": self._current_mode,
            "generation": int(self._generation),
        }
        self._log("mode switched", mode=self._current_mode, reason=reason, force=bool(force))
        self._emit_backend_event(
            "BACKEND_MODE_CHANGED",
            previous_mode=str(previous_mode or "IDLE").strip().upper(),
            current_mode=str(self._current_mode or "IDLE").strip().upper(),
            reason=str(reason or "mode_apply").strip() or "mode_apply",
            release_cooldown_s=float(profile.release_cooldown_s),
            contract=(profile.metadata or {}).get("contract"),
        )
        return profile

    def tick(self, now_ts: Optional[float] = None) -> None:
        """Advance cooldown bookkeeping and delayed capability release."""
        now = float(now_ts or time.time())
        profile = self.current_profile()
        desired_cameras = self._desired_camera_set(profile) if profile is not None else set()

        if self.camera_manager is not None and self._pending_camera_release:
            due_cameras = [name for name, due_ts in self._pending_camera_release.items() if now >= due_ts]
            for camera_name in due_cameras:
                if camera_name in desired_cameras:
                    self._pending_camera_release.pop(camera_name, None)
                    continue
                self.camera_manager.disable_camera(camera_name)
                self._pending_camera_release.pop(camera_name, None)

        if self.predictor_manager is not None and self._pending_predictor_release_ts is not None:
            if profile is not None and profile.predictor_enabled:
                self._pending_predictor_release_ts = None
            elif now >= float(self._pending_predictor_release_ts):
                self.predictor_manager.disable_model()
                self._pending_predictor_release_ts = None

    def snapshot(self) -> Dict[str, object]:
        """Return diagnostic state for logging and heartbeats."""
        return {
            "current_mode": self._current_mode,
            "target_mode": self._target_mode,
            "registered_modes": sorted(self._profiles.keys()),
            "last_switch_ts": self._last_switch_ts,
            "generation": int(self._generation),
            "runtime_running": bool(self._runtime_running),
            "preview_exit_requested": bool(getattr(self.preview_manager, "exit_requested", lambda: False)()),
            "scheduler": self._scheduler.snapshot(),
            "last_switch_result": dict(self._last_switch_result),
            "pending_camera_release": dict(self._pending_camera_release),
            "pending_predictor_release_ts": self._pending_predictor_release_ts,
        }

    def close(self) -> None:
        """Release controller-owned bookkeeping during engine shutdown."""
        self._target_mode = None
        self._pending_camera_release.clear()
        self._pending_predictor_release_ts = None
        self.stop_runtime()
