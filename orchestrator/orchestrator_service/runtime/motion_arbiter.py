#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Central motion arbitration for table docking.

The table docking state machine and controller produce candidate motion intent.
This module owns the final vx/vy/wz decision and the STOP vocabulary that
service/uart layers are allowed to honor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class FovGuardLevel(str, Enum):
    NONE = "none"
    SOFT = "soft"
    HARD = "hard"


class StalePolicy(str, Enum):
    FRESH = "fresh"
    SOFT_STALE_HOLD = "soft_stale_hold"
    DROPOUT_HOLD = "dropout_hold"
    HARD_STOP = "hard_stop"


@dataclass(frozen=True)
class MotionIntent:
    intent_type: str
    desired_vx: float = 0.0
    desired_wz: float = 0.0
    yaw_owner: str = ""
    forward_allowed_by_behavior: bool = False
    rotate_allowed_by_behavior: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArbitrationResult:
    final_vx: float
    final_vy: float
    final_wz: float
    motion_class: str
    stop_class: str
    blocked_by: str
    reason: str
    allow_uart_send: bool
    service_may_override: bool
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["summary"] = dict(self.summary)
        return d


_EMERGENCY_KEYS = {
    "emergency_stop_active",
    "car_estop",
    "estop_active",
    "obstacle_active",
    "obstacle_stop_active",
    "base_depth_emergency_active",
}

_HARD_SAFETY_KEYS = {
    "base_depth_hard_safety",
    "base_depth_stop_active",
    "depth_hard_stop_active",
    "safety_stop_active",
}


def _state_value(ctx: Any) -> str:
    state = getattr(ctx, "state", "")
    return str(getattr(state, "value", state) or "").strip().upper()


def _as_bool(summary: Dict[str, Any], key: str) -> bool:
    return bool(summary.get(key, False))


def _float(summary: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(summary.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def _fov_level(summary: Dict[str, Any]) -> FovGuardLevel:
    raw = str(summary.get("bbox_fov_guard_level") or summary.get("fov_guard_level") or "none").strip().lower()
    if raw == "hard":
        return FovGuardLevel.HARD
    if raw == "soft":
        return FovGuardLevel.SOFT
    return FovGuardLevel.NONE


def _stale_policy(summary: Dict[str, Any]) -> StalePolicy:
    stale_hold = str(summary.get("stale_hold_policy") or "").strip().lower()
    stale_level = str(summary.get("stale_level") or "fresh").strip().lower()
    if bool(summary.get("perception_dropout_hold_active", False)) or stale_hold == "approach_commit_short_dropout":
        return StalePolicy.DROPOUT_HOLD
    if stale_level == "soft_stale":
        return StalePolicy.SOFT_STALE_HOLD
    if stale_level in {"hard_stale", "dead"}:
        return StalePolicy.HARD_STOP
    return StalePolicy.FRESH


def _search_wz(ctx: Any, summary: Dict[str, Any]) -> float:
    sign = int(getattr(ctx, "search_wz_sign_latched", 0) or getattr(ctx, "relocate_turn_sign", 1) or 1)
    if sign == 0:
        sign = 1
    magnitude = abs(_float(summary, "search_wz_radps", _float(summary, "search_table_wz_radps", 0.10)))
    if magnitude <= 1e-9:
        magnitude = abs(_float(summary, "desired_search_wz", 0.10))
    return magnitude * (1.0 if sign >= 0 else -1.0)


def _stop_result(
    *,
    ctx: Any,
    summary: Dict[str, Any],
    motion_class: str,
    stop_class: str,
    blocked_by: str,
    reason: str,
    service_may_override: bool = True,
) -> ArbitrationResult:
    del ctx
    out = dict(summary)
    out.update(
        {
            "motion_class": motion_class,
            "stop_class": stop_class,
            "blocked_by": blocked_by,
            "arbitration_reason": reason,
            "final_vx": 0.0,
            "final_vy": 0.0,
            "final_wz": 0.0,
            "vx_mps": 0.0,
            "vy_mps": 0.0,
            "wz_radps": 0.0,
        }
    )
    return ArbitrationResult(
        final_vx=0.0,
        final_vy=0.0,
        final_wz=0.0,
        motion_class=motion_class,
        stop_class=stop_class,
        blocked_by=blocked_by,
        reason=reason,
        allow_uart_send=True,
        service_may_override=service_may_override,
        summary=out,
    )


def _active_table_docking(ctx: Any) -> bool:
    return _state_value(ctx) in {"SEARCH_TABLE", "YOLO_ACQUIRE_ALIGN", "YOLO_APPROACH", "EDGE_ADJUST"}


def _edge_usable(obs: Any, summary: Dict[str, Any]) -> bool:
    return bool(
        summary.get("edge_found")
        or summary.get("edge_trusted")
        or summary.get("usable_for_approach")
        or (obs is not None and (getattr(obs, "edge_found", False) or getattr(obs, "usable_for_approach", False) or getattr(obs, "edge_trusted", False)))
    )


def _result(
    *,
    ctx: Any,
    intent: MotionIntent,
    summary: Dict[str, Any],
    final_vx: float,
    final_vy: float,
    final_wz: float,
    motion_class: str,
    stop_class: str,
    blocked_by: str,
    reason: str,
    service_may_override: bool = False,
) -> ArbitrationResult:
    out = dict(summary)
    out.update(
        {
            "motion_intent_type": str(intent.intent_type or ""),
            "yaw_owner": str(intent.yaw_owner or ""),
            "arbitration_reason": str(reason or ""),
            "motion_class": str(motion_class or "normal"),
            "stop_class": str(stop_class or "none"),
            "blocked_by": str(blocked_by or ""),
            "final_vx": float(final_vx),
            "final_vy": float(final_vy),
            "final_wz": float(final_wz),
            "vx_mps": float(final_vx),
            "vy_mps": float(final_vy),
            "wz_radps": float(final_wz),
            "allow_uart_send": True,
            "service_may_override": bool(service_may_override),
            "active_table_docking": bool(_active_table_docking(ctx)),
        }
    )
    return ArbitrationResult(
        final_vx=float(final_vx),
        final_vy=float(final_vy),
        final_wz=float(final_wz),
        motion_class=str(motion_class or "normal"),
        stop_class=str(stop_class or "none"),
        blocked_by=str(blocked_by or ""),
        reason=str(reason or ""),
        allow_uart_send=True,
        service_may_override=bool(service_may_override),
        summary=out,
    )


def arbitrate_table_docking_motion(
    ctx: Any,
    obs: Any,
    intent: MotionIntent,
    current_summary: Optional[Dict[str, Any]] = None,
) -> ArbitrationResult:
    """Return the single final table-docking motion command.

    Priority order:
    emergency/hard safety, final stop, dropout hold, committed approach,
    hard/soft FOV guard, acquire/search rotate, zero watchdog escape.
    """
    summary = dict(current_summary or {})
    state = _state_value(ctx)
    fov_level = _fov_level(summary)
    fov_reason = str(summary.get("bbox_fov_guard_reason") or summary.get("fov_guard_reason") or "")
    stale_policy = _stale_policy(summary)
    phase = str(summary.get("control_phase") or "").strip().upper()
    intent_type = str(intent.intent_type or summary.get("control_source") or "").strip().lower()
    desired_vx = float(intent.desired_vx or 0.0)
    desired_vy = _float(summary, "desired_vy", _float(summary, "vy_mps", 0.0))
    desired_wz = float(intent.desired_wz or 0.0)
    zero_escape_reason = ""

    blocked: List[str] = []
    emergency_active = bool(_as_bool(summary, "explicit_stop_active") or any(_as_bool(summary, key) for key in _EMERGENCY_KEYS))
    explicit_stop = bool(summary.get("explicit_stop_active", False))
    hard_safety = bool(any(_as_bool(summary, key) for key in _HARD_SAFETY_KEYS))

    if emergency_active or explicit_stop:
        reason = "explicit_stop" if explicit_stop else "emergency_or_obstacle"
        return _stop_result(ctx=ctx, summary=summary, motion_class="emergency_stop", stop_class="emergency", blocked_by=reason, reason=reason)

    if hard_safety:
        return _stop_result(ctx=ctx, summary=summary, motion_class="safety_stop", stop_class="safety", blocked_by="hard_safety", reason="hard_safety")

    final_stop = bool(
        summary.get("depth_roi_stop_ready", False)
        or phase == "DEPTH_FINAL_STOP"
        or state in {"FINAL_SLOW_STOP", "AT_TABLE_EDGE"}
        or str(summary.get("stop_source") or "").strip().lower() in {"roi_depth", "final_lock"}
    )
    if final_stop:
        return _stop_result(ctx=ctx, summary=summary, motion_class="safety_stop", stop_class="safety", blocked_by="final_depth_stop", reason="final_depth_stop")

    if stale_policy == StalePolicy.DROPOUT_HOLD:
        vx = max(abs(desired_vx), _float(summary, "forward_commit_vx", 0.020))
        wz = desired_wz if abs(desired_wz) > 1e-9 else _float(summary, "last_edge_yaw_cmd", 0.0)
        return _result(
            ctx=ctx,
            intent=intent,
            summary={**summary, "stale_policy": stale_policy.value, "perception_dropout_hold_active": True},
            final_vx=vx,
            final_vy=0.0,
            final_wz=wz,
            motion_class="recovery",
            stop_class="none",
            blocked_by="",
            reason="perception_dropout_hold",
        )

    if stale_policy == StalePolicy.HARD_STOP and state not in {"SEARCH_TABLE"} and not bool(summary.get("search_table_stale_gate_bypass", False)):
        return _stop_result(
            ctx=ctx,
            summary={**summary, "stale_policy": stale_policy.value},
            motion_class="recovery",
            stop_class="stale_recovery",
            blocked_by=str(summary.get("stale_hold_policy") or summary.get("stale_level") or "hard_stale"),
            reason="stale_recovery",
            service_may_override=False,
        )

    edge_block = str(summary.get("edge_guided_commit_block_reason") or "").strip().lower()
    if edge_block in {"explicit_stop", "base_safety"}:
        stop_class = "emergency" if edge_block == "explicit_stop" else "safety"
        motion_class = "emergency_stop" if stop_class == "emergency" else "safety_stop"
        return _stop_result(ctx=ctx, summary=summary, motion_class=motion_class, stop_class=stop_class, blocked_by=edge_block, reason=edge_block)
    if edge_block in {"hard_stale", "perception_dropout_hold_expired"}:
        return _stop_result(
            ctx=ctx,
            summary={**summary, "stale_policy": stale_policy.value},
            motion_class="recovery",
            stop_class="stale_recovery",
            blocked_by=edge_block,
            reason=edge_block,
            service_may_override=False,
        )
    if edge_block in {"depth_final_stop", "bbox_fov_guard_hard", "edge_yaw_too_large"}:
        return _stop_result(ctx=ctx, summary=summary, motion_class="safety_stop", stop_class="safety", blocked_by=edge_block, reason=edge_block)

    if fov_level == FovGuardLevel.HARD:
        return _stop_result(
            ctx=ctx,
            summary={**summary, "fov_guard_level": fov_level.value, "fov_guard_reason": fov_reason},
            motion_class="safety_stop",
            stop_class="safety",
            blocked_by="bbox_fov_guard_hard",
            reason=fov_reason or "bbox_fov_guard_hard",
        )

    if phase == "EDGE_GUIDED_APPROACH" and intent.forward_allowed_by_behavior:
        vx = desired_vx
        if bool(summary.get("approach_commit_active", False)) or _edge_usable(obs, summary):
            vx = max(abs(vx), _float(summary, "forward_commit_vx", 0.020))
        if fov_level == FovGuardLevel.SOFT:
            max_soft_vx = max(0.0, _float(summary, "forward_commit_vx", 0.020))
            vx = min(max(abs(vx), max_soft_vx), max_soft_vx)
        if abs(vx) > 1e-9:
            return _result(
                ctx=ctx,
                intent=intent,
                summary={
                    **summary,
                    "fov_guard_level": fov_level.value,
                    "fov_guard_reason": fov_reason,
                    "bbox_fov_soft_allowed_forward": bool(fov_level == FovGuardLevel.SOFT),
                    "stale_policy": stale_policy.value,
                },
                final_vx=vx,
                final_vy=0.0,
                final_wz=desired_wz,
                motion_class="normal" if stale_policy == StalePolicy.FRESH else "recovery",
                stop_class="none",
                blocked_by="",
                reason="edge_guided_approach_soft_fov" if fov_level == FovGuardLevel.SOFT else "edge_guided_approach",
            )
        blocked.append(str(summary.get("forward_block_reason") or "forward_not_available"))

    if phase in {"BBOX_ACQUIRE", "EDGE_HANDOFF_CONFIRM"} and intent.rotate_allowed_by_behavior:
        return _result(
            ctx=ctx,
            intent=intent,
            summary={**summary, "fov_guard_level": fov_level.value, "fov_guard_reason": fov_reason, "stale_policy": stale_policy.value},
            final_vx=0.0,
            final_vy=0.0,
            final_wz=desired_wz,
            motion_class="recovery" if stale_policy != StalePolicy.FRESH else "normal",
            stop_class="none",
            blocked_by="",
            reason="bbox_acquire_rotate",
        )

    if state == "SEARCH_TABLE" or intent_type in {"local_rotate_search", "search"} or phase == "SEARCH_SCAN":
        wz = desired_wz if abs(desired_wz) > 1e-9 else _search_wz(ctx, summary)
        return _result(
            ctx=ctx,
            intent=intent,
            summary={**summary, "stale_policy": stale_policy.value},
            final_vx=0.0,
            final_vy=0.0,
            final_wz=wz,
            motion_class="recovery" if stale_policy != StalePolicy.FRESH else "normal",
            stop_class="none",
            blocked_by="",
            reason="search_rotate",
        )

    if _active_table_docking(ctx) and abs(desired_vx) < 1e-9 and abs(desired_wz) < 1e-9:
        zero_age_ms = _float(summary, "zero_cmd_age_ms", 0.0)
        if zero_age_ms >= 800.0:
            if bool(summary.get("approach_commit_active", False)):
                zero_escape_reason = "forward_coast"
                return _result(
                    ctx=ctx,
                    intent=intent,
                    summary={**summary, "zero_escape_reason": zero_escape_reason, "stale_policy": stale_policy.value},
                    final_vx=_float(summary, "forward_commit_vx", 0.020),
                    final_vy=0.0,
                    final_wz=_float(summary, "last_edge_yaw_cmd", 0.0),
                    motion_class="recovery",
                    stop_class="none",
                    blocked_by="",
                    reason="zero_watchdog_forward_coast",
                )
            zero_escape_reason = "search_rotate"
            return _result(
                ctx=ctx,
                intent=intent,
                summary={**summary, "zero_escape_reason": zero_escape_reason, "stale_policy": stale_policy.value},
                final_vx=0.0,
                final_vy=0.0,
                final_wz=_search_wz(ctx, summary),
                motion_class="recovery",
                stop_class="none",
                blocked_by="",
                reason="zero_watchdog_search_rotate",
            )

    return _result(
        ctx=ctx,
        intent=intent,
        summary={
            **summary,
            "fov_guard_level": fov_level.value,
            "fov_guard_reason": fov_reason,
            "stale_policy": stale_policy.value,
            "zero_escape_reason": zero_escape_reason,
        },
        final_vx=desired_vx,
        final_vy=desired_vy,
        final_wz=desired_wz,
        motion_class="normal" if (abs(desired_vx) > 1e-9 or abs(desired_wz) > 1e-9) else "control_stop",
        stop_class="none" if (abs(desired_vx) > 1e-9 or abs(desired_wz) > 1e-9) else "control_recovery",
        blocked_by=";".join(item for item in blocked if item),
        reason=str(intent.reason or summary.get("reason") or "candidate_intent"),
    )
