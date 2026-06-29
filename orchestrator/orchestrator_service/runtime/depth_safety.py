#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Close-range table docking depth safety gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .context import State
from .docking_model import DockingAction
from .motion_arbiter import ArbitrationResult


@dataclass(frozen=True)
class DepthSafetyGateResult:
    final_vx: float
    final_vy: float
    final_wz: float
    summary_update: Dict[str, Any]
    force_final_locked: bool = False
    force_action: str = ""
    reason: str = ""


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cfg_float(cfg: Any, *names: str, default: float) -> float:
    for name in names:
        value = _as_float(getattr(cfg, name, None))
        if value is not None:
            return value
    return float(default)


def _positive_cap(value: float, cap: float) -> float:
    if value > 0.0:
        return min(float(value), abs(float(cap)))
    return float(value)


def _state_value(ctx: Any) -> str:
    state = getattr(ctx, "state", "")
    return str(getattr(state, "value", state) or "")


def _current_obs_depth(obs: Any) -> Tuple[Optional[float], str]:
    if obs is None:
        return None, ""
    table_roi_valid = bool(getattr(obs, "table_roi_depth_valid", False))
    table_roi_p10 = _as_float(getattr(obs, "table_roi_depth_p10", None))
    if table_roi_valid and table_roi_p10 is not None:
        return table_roi_p10, "obs.table_roi_depth_p10"
    depth_p10 = _as_float(getattr(obs, "depth_p10", None))
    if depth_p10 is not None:
        return depth_p10, "obs.depth_p10"
    return None, ""


def resolve_best_depth_p10(ctx: Any, obs: Any, summary: Dict[str, Any], now_mono: float, depth_missing_hold_s: float) -> Dict[str, Any]:
    current_depth, current_source = _current_obs_depth(obs)
    current_valid = current_depth is not None
    depth_missing_age_s = 0.0

    if current_valid:
        try:
            ctx.last_valid_depth_p10_m = float(current_depth)
            ctx.last_valid_depth_p10_source = current_source
            ctx.last_valid_depth_p10_mono = float(now_mono)
            ctx.depth_missing_started_mono = 0.0
        except Exception:
            pass
        best_depth = current_depth
        best_source = current_source
        best_age = 0.0
    else:
        missing_started = float(getattr(ctx, "depth_missing_started_mono", 0.0) or 0.0)
        if missing_started <= 0.0:
            missing_started = float(now_mono)
            try:
                ctx.depth_missing_started_mono = missing_started
            except Exception:
                pass
        depth_missing_age_s = max(0.0, float(now_mono) - missing_started)
        best_depth = None
        best_source = ""
        best_age = None
        for key in ("table_roi_depth_p10", "roi_final_p10_m", "depth_p10"):
            value = _as_float(summary.get(key))
            if value is not None:
                best_depth = value
                best_source = f"summary.{key}"
                best_age = 0.0
                break
        if best_depth is None:
            last_depth = _as_float(getattr(ctx, "last_valid_depth_p10_m", None))
            last_mono = float(getattr(ctx, "last_valid_depth_p10_mono", 0.0) or 0.0)
            last_age = max(0.0, float(now_mono) - last_mono) if last_mono > 0.0 else None
            if last_depth is not None and last_age is not None and last_age <= float(depth_missing_hold_s):
                best_depth = last_depth
                best_source = str(getattr(ctx, "last_valid_depth_p10_source", "") or "ctx.last_valid_depth_p10_m")
                best_age = last_age

    last_mono = float(getattr(ctx, "last_valid_depth_p10_mono", 0.0) or 0.0)
    last_age = max(0.0, float(now_mono) - last_mono) if last_mono > 0.0 else None
    return {
        "best_depth_p10_m": best_depth,
        "best_depth_source": best_source,
        "best_depth_age_s": best_age,
        "depth_missing_age_s": depth_missing_age_s,
        "current_depth_p10_valid": bool(current_valid),
        "last_valid_depth_p10_m": _as_float(getattr(ctx, "last_valid_depth_p10_m", None)),
        "last_valid_depth_p10_source": str(getattr(ctx, "last_valid_depth_p10_source", "") or ""),
        "last_valid_depth_p10_age_s": last_age,
    }


def _close_range_or_final(ctx: Any, summary: Dict[str, Any]) -> bool:
    return bool(
        summary.get("close_range_latched", False)
        or summary.get("final_roi_mode_latched", False)
        or summary.get("final_edge_mode_latched", False)
        or summary.get("final_distance_servo_active", False)
        or getattr(ctx, "near_table_latched", False)
        or getattr(ctx, "final_depth_latched", False)
        or getattr(ctx, "final_locked", False)
        or _state_value(ctx) == State.FINAL_SLOW_STOP.value
    )


def _result_with_gate(result: ArbitrationResult, vx: float, vy: float, wz: float, summary: Dict[str, Any], reason: str, blocked_by: str = "") -> ArbitrationResult:
    return ArbitrationResult(
        final_vx=float(vx),
        final_vy=float(vy),
        final_wz=float(wz),
        motion_class=result.motion_class,
        stop_class=result.stop_class,
        blocked_by=str(blocked_by),
        reason=str(reason or result.reason),
        allow_uart_send=bool(result.allow_uart_send),
        service_may_override=False,
        summary=summary,
    )


def apply_close_range_depth_safety_gate(ctx: Any, obs: Any, result: ArbitrationResult, cfg: Any, now_mono: float) -> ArbitrationResult:
    summary = dict(result.summary or {})
    if not _close_range_or_final(ctx, summary):
        return result

    depth_stop_p10_m = _cfg_float(cfg, "roi_final_stop_p10_m", "depth_envelope_stop_p10_m", default=0.42)
    depth_slow_p10_m = _cfg_float(cfg, "roi_final_slow_p10_m", "depth_envelope_slow_p10_m", default=0.52)
    depth_missing_hold_s = max(0.0, _cfg_float(cfg, "roi_final_missing_hold_s", default=0.8))
    final_probe_vx_mps = abs(_cfg_float(cfg, "final_probe_vx_mps", "close_range_probe_vx_mps", "roi_final_probe_vx_mps", default=0.008))
    final_missing_probe_vx_mps = abs(_cfg_float(cfg, "final_missing_probe_vx_mps", "close_range_missing_probe_vx_mps", "roi_final_missing_probe_vx_mps", default=0.004))
    final_probe_timeout_s = max(0.0, _cfg_float(cfg, "final_probe_timeout_s", default=8.0))
    final_probe_distance_budget_m = max(0.0, _cfg_float(cfg, "final_probe_distance_budget_m", default=0.15))

    vx_raw = float(result.final_vx)
    vy_raw = float(result.final_vy)
    wz_raw = float(result.final_wz)

    start_mono = float(getattr(ctx, "close_range_probe_start_mono", 0.0) or 0.0)
    if start_mono <= 0.0:
        start_mono = float(now_mono)
        try:
            ctx.close_range_probe_start_mono = start_mono
            ctx.close_range_probe_distance_used_m = 0.0
        except Exception:
            pass
    last_mono = float(getattr(ctx, "close_range_probe_last_mono", 0.0) or 0.0)
    dt = max(0.0, float(now_mono) - last_mono) if last_mono > 0.0 else 0.0
    distance_used = float(getattr(ctx, "close_range_probe_distance_used_m", 0.0) or 0.0)
    distance_used += max(0.0, vx_raw) * dt
    try:
        ctx.close_range_probe_last_mono = float(now_mono)
        ctx.close_range_probe_distance_used_m = float(distance_used)
    except Exception:
        pass
    final_probe_elapsed_s = max(0.0, float(now_mono) - start_mono)

    depth_info = resolve_best_depth_p10(ctx, obs, summary, now_mono, depth_missing_hold_s)
    best_depth = _as_float(depth_info.get("best_depth_p10_m"))
    current_valid = bool(depth_info.get("current_depth_p10_valid", False))
    depth_missing_age_s = float(depth_info.get("depth_missing_age_s") or 0.0)
    last_valid_age = _as_float(depth_info.get("last_valid_depth_p10_age_s"))
    last_valid_fresh = bool(last_valid_age is not None and last_valid_age <= depth_missing_hold_s)
    already_locked = bool(summary.get("final_locked", False) or getattr(ctx, "final_locked", False))

    vx = _positive_cap(vx_raw, final_probe_vx_mps)
    vy = 0.0
    wz = 0.0
    state = "pass_or_probe_cap"
    reason = "depth_probe_cap"
    action = str(summary.get("docking_action") or "")
    if action == DockingAction.FINAL_LOCKED_STOP.value and not already_locked:
        action = DockingAction.FINAL_SLOW_PROBE.value if bool(summary.get("final_distance_servo_active", False)) else DockingAction.CLOSE_RANGE_PROBE.value
    blocked_by = ""
    final_locked = False
    allow_forward = bool(vx > 1e-9)

    if already_locked:
        vx = 0.0
        state = "final_locked_hold"
        reason = "final_locked_hold"
        action = DockingAction.FINAL_LOCKED_STOP.value
        final_locked = True
        allow_forward = False
        blocked_by = reason
    elif current_valid and best_depth is not None and best_depth <= depth_stop_p10_m:
        vx = 0.0
        try:
            ctx.depth_stop_stable_count = int(getattr(ctx, "depth_stop_stable_count", 0) or 0) + 1
        except Exception:
            pass
        if int(getattr(ctx, "depth_stop_stable_count", 0) or 0) >= 2:
            state = "hard_stop_locked"
            reason = "depth_hard_stop"
            action = DockingAction.FINAL_LOCKED_STOP.value
            final_locked = True
            allow_forward = False
            blocked_by = reason
            try:
                ctx.final_locked = True
                ctx.final_lock_reason = reason
            except Exception:
                pass
        else:
            state = "hard_stop_confirming"
            reason = "depth_hard_stop_confirming"
            action = DockingAction.DEPTH_SAFETY_HOLD.value
            final_locked = False
            allow_forward = False
            blocked_by = reason
    else:
        try:
            ctx.depth_stop_stable_count = 0
        except Exception:
            pass
        if not current_valid and depth_missing_age_s > depth_missing_hold_s:
            vx = 0.0
            state = "missing_hold"
            reason = "depth_missing_hold"
            action = DockingAction.DEPTH_SAFETY_HOLD.value
            allow_forward = False
            blocked_by = reason
        elif not current_valid and last_valid_fresh:
            vx = _positive_cap(vx_raw, final_missing_probe_vx_mps)
            state = "missing_short_probe"
            reason = "depth_missing_short_probe"
            allow_forward = bool(vx > 1e-9)
        elif best_depth is not None and best_depth <= depth_slow_p10_m:
            vx = _positive_cap(vx_raw, final_probe_vx_mps)
            state = "slow_cap"
            reason = "depth_slow_cap"
            action = DockingAction.FINAL_SLOW_PROBE.value
            allow_forward = bool(vx > 1e-9)
        elif final_probe_elapsed_s > final_probe_timeout_s:
            vx = 0.0
            state = "timeout_hold"
            reason = "final_probe_timeout_hold"
            action = DockingAction.DEPTH_SAFETY_HOLD.value
            allow_forward = False
            blocked_by = reason
        elif distance_used > final_probe_distance_budget_m:
            vx = 0.0
            state = "distance_budget_hold"
            reason = "final_probe_distance_budget_hold"
            action = DockingAction.DEPTH_SAFETY_HOLD.value
            allow_forward = False
            blocked_by = reason

    summary.update(depth_info)
    summary.update(
        {
            "depth_safety_applied": True,
            "depth_safety_state": state,
            "depth_safety_reason": reason,
            "vx_before_depth_gate": vx_raw,
            "vx_after_depth_gate": float(vx),
            "vy_before_depth_gate": vy_raw,
            "vy_after_depth_gate": float(vy),
            "wz_before_depth_gate": wz_raw,
            "wz_after_depth_gate": float(wz),
            "final_probe_elapsed_s": float(final_probe_elapsed_s),
            "final_probe_distance_used_m": float(distance_used),
            "depth_stop_p10_m": float(depth_stop_p10_m),
            "depth_slow_p10_m": float(depth_slow_p10_m),
            "depth_missing_hold_s": float(depth_missing_hold_s),
            "final_probe_timeout_s": float(final_probe_timeout_s),
            "final_probe_distance_budget_m": float(final_probe_distance_budget_m),
            "final_locked": bool(final_locked),
            "allow_forward": bool(allow_forward),
            "allow_rotate": False,
            "allow_lateral": False,
            "yaw_owner": "none",
            "lateral_owner": "none",
        }
    )
    if action:
        summary["docking_action"] = action
    if blocked_by:
        summary["forward_block_reason"] = blocked_by
        summary["effective_block_reason"] = blocked_by
    if final_locked:
        summary["final_lock_reason"] = reason
    try:
        ctx.depth_safety_state = state
        ctx.depth_safety_reason = reason
    except Exception:
        pass

    return _result_with_gate(result, vx, vy, wz, summary, reason=reason, blocked_by=blocked_by)
