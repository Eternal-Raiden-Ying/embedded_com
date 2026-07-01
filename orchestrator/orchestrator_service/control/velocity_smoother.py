#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
import time
from typing import Any, Dict, Optional, Tuple

from ..ipc.protocol import CmdVel


@dataclass
class MotionSmoothingConfig:
    enabled: bool = True
    bypass_on_safety_stop: bool = True
    vx_accel_mps2: float = 0.35
    vx_decel_mps2: float = 0.70
    vy_accel_mps2: float = 0.20
    vy_decel_mps2: float = 0.35
    wz_accel_radps2: float = 0.90
    wz_decel_radps2: float = 1.40
    urgent_wz_accel_radps2: float = 2.20
    urgent_wz_decel_radps2: float = 2.80
    dt_min_s: float = 0.02
    dt_max_s: float = 0.20
    reset_gap_s: float = 0.50


class VelocitySmoother:
    """Final UART velocity slew-rate limiter.

    This class shapes only the final command sent to STM32. State handlers and
    motion arbitration keep owning the nominal command and safety decisions.
    """

    _ZERO_STATES = {
        "IDLE",
        "DONE",
        "ERROR",
        "ERROR_RECOVERY",
        "TARGET_CONFIRM",
        "TARGET_LOCKED",
        "FREEZE_BASE",
        "GRASP",
        "FINAL_LOCKED_STOP",
    }
    _STOP_MODES = {"STOP", "IDLE", "DONE", "ERROR", "ERROR_RECOVERY", "FINAL_LOCKED_STOP"}

    def __init__(self, cfg: Optional[MotionSmoothingConfig] = None):
        self.cfg = cfg or MotionSmoothingConfig()
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_wz = 0.0
        self.last_ts_monotonic: Optional[float] = None
        self.last_task_epoch: Optional[int] = None

    def reset_to_zero(self, now_monotonic: Optional[float] = None) -> None:
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_wz = 0.0
        self.last_ts_monotonic = time.monotonic() if now_monotonic is None else float(now_monotonic)

    def apply(
        self,
        cmd: CmdVel,
        *,
        state: str = "",
        summary: Optional[Dict[str, Any]] = None,
        task_epoch: Optional[int] = None,
        now_monotonic: Optional[float] = None,
    ) -> Tuple[CmdVel, Dict[str, Any]]:
        summary = dict(summary or {})
        now_mono = time.monotonic() if now_monotonic is None else float(now_monotonic)
        before = self._cmd_dict(cmd)
        state_u = str(state or "").strip().upper()
        mode_u = str(getattr(cmd, "mode", "") or "").strip().upper()
        urgent_wz = self._is_urgent_wz(state_u, summary)

        bypass_reason = self._bypass_reason(cmd, state_u, mode_u, summary)
        if bypass_reason:
            out = self._zero_cmd(cmd, hard=True)
            self.reset_to_zero(now_mono)
            meta = self._meta(
                before,
                self._cmd_dict(out),
                enabled=bool(self.cfg.enabled),
                applied=False,
                bypassed=True,
                bypass_reason=bypass_reason,
                profile="bypass",
                urgent=urgent_wz,
                dt_s=0.0,
            )
            return out, meta

        if task_epoch is not None and self.last_task_epoch is not None and int(task_epoch) != int(self.last_task_epoch):
            self.reset_to_zero(now_mono)
        if task_epoch is not None:
            self.last_task_epoch = int(task_epoch)

        raw_dt = None if self.last_ts_monotonic is None else now_mono - float(self.last_ts_monotonic)
        if raw_dt is not None and raw_dt > float(self.cfg.reset_gap_s):
            self.reset_to_zero(now_mono)
            raw_dt = None
        dt = self._clamp(
            float(self.cfg.dt_min_s) if raw_dt is None else float(raw_dt),
            float(self.cfg.dt_min_s),
            float(self.cfg.dt_max_s),
        )

        nominal_vx = float(getattr(cmd, "vx_mps", 0.0) or 0.0)
        nominal_vy = float(getattr(cmd, "vy_mps", 0.0) or 0.0)
        nominal_wz = float(getattr(cmd, "wz_radps", 0.0) or 0.0)
        axis_profile = "normal"
        if state_u == "EDGE_SLIDE_SEARCH" or mode_u == "EDGE_SLIDE_SEARCH":
            nominal_vx = 0.0
            nominal_wz = 0.0
            self.last_vx = 0.0
            self.last_wz = 0.0
            axis_profile = "edge_slide_vy_only"

        if not bool(self.cfg.enabled):
            out = CmdVel(
                ts=float(getattr(cmd, "ts", time.time()) or time.time()),
                mode=str(getattr(cmd, "mode", "") or state_u or "SEARCH"),
                vx_mps=nominal_vx,
                vy_mps=nominal_vy,
                wz_radps=nominal_wz,
                hold_ms=int(getattr(cmd, "hold_ms", 150) or 150),
                brake=bool(getattr(cmd, "brake", False)),
                base_freeze=bool(getattr(cmd, "base_freeze", False)),
            )
            self.last_vx, self.last_vy, self.last_wz = nominal_vx, nominal_vy, nominal_wz
            self.last_ts_monotonic = now_mono
            return out, self._meta(before, self._cmd_dict(out), enabled=False, applied=False, bypassed=True, bypass_reason="disabled", profile=axis_profile, urgent=urgent_wz, dt_s=dt)

        vx = self._slew(self.last_vx, nominal_vx, dt, self.cfg.vx_accel_mps2, self.cfg.vx_decel_mps2)
        vy = self._slew(self.last_vy, nominal_vy, dt, self.cfg.vy_accel_mps2, self.cfg.vy_decel_mps2)
        wz_accel = self.cfg.urgent_wz_accel_radps2 if urgent_wz else self.cfg.wz_accel_radps2
        wz_decel = self.cfg.urgent_wz_decel_radps2 if urgent_wz else self.cfg.wz_decel_radps2
        wz = self._slew(self.last_wz, nominal_wz, dt, wz_accel, wz_decel)

        self.last_vx, self.last_vy, self.last_wz = vx, vy, wz
        self.last_ts_monotonic = now_mono
        out = CmdVel(
            ts=float(getattr(cmd, "ts", time.time()) or time.time()),
            mode=str(getattr(cmd, "mode", "") or state_u or "SEARCH"),
            vx_mps=float(vx),
            vy_mps=float(vy),
            wz_radps=float(wz),
            hold_ms=int(getattr(cmd, "hold_ms", 150) or 150),
            brake=bool(getattr(cmd, "brake", False)),
            base_freeze=bool(getattr(cmd, "base_freeze", False)),
        )
        after = self._cmd_dict(out)
        applied = any(abs(after[k] - before[k]) > 1e-9 for k in ("vx_mps", "vy_mps", "wz_radps"))
        profile = "urgent_wz" if urgent_wz else axis_profile
        return out, self._meta(before, after, enabled=True, applied=applied, bypassed=False, bypass_reason="", profile=profile, urgent=urgent_wz, dt_s=dt)

    def _bypass_reason(self, cmd: CmdVel, state_u: str, mode_u: str, summary: Dict[str, Any]) -> str:
        if bool(getattr(cmd, "brake", False)):
            return "brake"
        if mode_u in self._STOP_MODES:
            return f"mode_{mode_u.lower()}"
        if state_u in self._ZERO_STATES:
            return f"state_{state_u.lower()}"
        stop_class = str(summary.get("stop_class") or "").strip().lower()
        motion_class = str(summary.get("motion_class") or "").strip().lower()
        emit_reason = str(summary.get("uart_emit_reason") or "").strip().lower()
        zero_reason = str(summary.get("zero_cmd_reason") or "").strip().lower()
        if summary.get("allow_uart_send") is False:
            return "allow_uart_send_false"
        if bool(self.cfg.bypass_on_safety_stop) and (
            stop_class in {"emergency", "safety"}
            or motion_class in {"emergency_stop", "safety_stop"}
            or emit_reason in {"hard_stale_stop", "explicit_stop", "arbiter_hard_stop"}
            or "hard_stale" in zero_reason
        ):
            return emit_reason or motion_class or stop_class or zero_reason or "hard_stop"
        for key in (
            "emergency_stop_active",
            "safety_stop_active",
            "car_estop",
            "estop_active",
            "depth_emergency",
            "obstacle_too_close",
            "uart_fault",
            "service_stopping",
            "final_locked",
            "final_depth_stop",
        ):
            if bool(summary.get(key, False)):
                return key
        reason_text = " ".join(
            str(summary.get(key) or "").strip().lower()
            for key in ("docking_action", "docking_reason", "final_lock_reason", "effective_block_reason")
        )
        if "final_depth_stop" in reason_text or "final_locked_stop" in reason_text:
            return "final_depth_stop"
        return ""

    def _is_urgent_wz(self, state_u: str, summary: Dict[str, Any]) -> bool:
        if state_u == "SEARCH_TABLE":
            return True
        action = str(summary.get("docking_action") or "").strip().upper()
        if action in {"BBOX_REACQUIRE_ROTATE", "CONTROL_RECOVERY_ROTATE"}:
            return True
        if str(summary.get("fov_guard_level") or summary.get("bbox_fov_guard_level") or "").strip().lower() == "hard":
            return True
        reason = str(summary.get("docking_reason") or summary.get("reason") or "").strip().lower()
        if any(token in reason for token in ("reacquire", "bbox_center_extreme", "target_lost")):
            return True
        return any(bool(summary.get(key, False)) for key in ("bbox_touch_left", "bbox_touch_right", "target_near_fov_edge"))

    def _slew(self, last: float, nominal: float, dt: float, accel: float, decel: float) -> float:
        delta = float(nominal) - float(last)
        rate = float(decel if abs(nominal) < abs(last) or self._sign_changed(last, nominal) else accel)
        max_delta = max(0.0, rate) * float(dt)
        return float(last) + self._clamp(delta, -max_delta, max_delta)

    @staticmethod
    def _sign_changed(a: float, b: float) -> bool:
        return abs(a) > 1e-9 and abs(b) > 1e-9 and ((a < 0.0 < b) or (b < 0.0 < a))

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(value)))

    @staticmethod
    def _cmd_dict(cmd: CmdVel) -> Dict[str, Any]:
        return {
            "ts": float(getattr(cmd, "ts", time.time()) or time.time()),
            "mode": str(getattr(cmd, "mode", "") or ""),
            "vx_mps": float(getattr(cmd, "vx_mps", 0.0) or 0.0),
            "vy_mps": float(getattr(cmd, "vy_mps", 0.0) or 0.0),
            "wz_radps": float(getattr(cmd, "wz_radps", 0.0) or 0.0),
            "vx": float(getattr(cmd, "vx_mps", 0.0) or 0.0),
            "vy": float(getattr(cmd, "vy_mps", 0.0) or 0.0),
            "wz": float(getattr(cmd, "wz_radps", 0.0) or 0.0),
            "hold_ms": int(getattr(cmd, "hold_ms", 150) or 150),
            "brake": bool(getattr(cmd, "brake", False)),
        }

    @staticmethod
    def _zero_cmd(cmd: CmdVel, hard: bool = False) -> CmdVel:
        return CmdVel(
            ts=float(getattr(cmd, "ts", time.time()) or time.time()),
            mode="STOP" if hard else str(getattr(cmd, "mode", "") or "STOP"),
            vx_mps=0.0,
            vy_mps=0.0,
            wz_radps=0.0,
            hold_ms=int(getattr(cmd, "hold_ms", 150) or 150),
            brake=True if hard else bool(getattr(cmd, "brake", False)),
            base_freeze=bool(getattr(cmd, "base_freeze", False)),
        )

    def _meta(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
        *,
        enabled: bool,
        applied: bool,
        bypassed: bool,
        bypass_reason: str,
        profile: str,
        urgent: bool,
        dt_s: float,
    ) -> Dict[str, Any]:
        return {
            "smoothing_enabled": bool(enabled),
            "smoothing_applied": bool(applied),
            "smoothing_bypassed": bool(bypassed),
            "smoothing_bypass_reason": str(bypass_reason or ""),
            "smoothing_profile": str(profile or ""),
            "smoothing_urgent": bool(urgent),
            "smoothing_dt_s": float(dt_s),
            "cmd_before_smoothing": dict(before),
            "cmd_after_smoothing": dict(after),
            "vx_before_smoothing": float(before["vx_mps"]),
            "vy_before_smoothing": float(before["vy_mps"]),
            "wz_before_smoothing": float(before["wz_radps"]),
            "vx_smoothed": float(after["vx_mps"]),
            "vy_smoothed": float(after["vy_mps"]),
            "wz_smoothed": float(after["wz_radps"]),
            "smoothing_delta_vx": float(after["vx_mps"] - before["vx_mps"]),
            "smoothing_delta_vy": float(after["vy_mps"] - before["vy_mps"]),
            "smoothing_delta_wz": float(after["wz_radps"] - before["wz_radps"]),
        }
