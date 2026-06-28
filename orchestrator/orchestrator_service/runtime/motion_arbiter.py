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
import time
from typing import Any, Dict, List, Optional

from .docking_model import (
    DockingAction,
    DockingMotionResult,
    DockingStage,
    StopClass,
    build_docking_observation,
    update_docking_stage,
)


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
    desired_vy: float = 0.0
    desired_wz: float = 0.0
    yaw_owner: str = ""
    forward_owner: str = ""
    lateral_owner: str = "none"
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
            "forward_owner": str(intent.forward_owner or ""),
            "lateral_owner": str(intent.lateral_owner or "none"),
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


def _from_docking_result(result: DockingMotionResult) -> ArbitrationResult:
    summary = result.legacy_summary()
    return ArbitrationResult(
        final_vx=float(result.vx),
        final_vy=float(result.vy),
        final_wz=float(result.wz),
        motion_class=str(summary.get("motion_class") or result.motion_class),
        stop_class=result.stop_class.value,
        blocked_by=str(result.blocked_by or ""),
        reason=str(result.reason or ""),
        allow_uart_send=bool(result.allow_uart_send),
        service_may_override=bool(result.service_may_override),
        summary=summary,
    )


def _docking_result(
    *,
    action: DockingAction,
    stage: DockingStage,
    summary: Dict[str, Any],
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
    yaw_owner: str = "",
    forward_owner: str = "",
    lateral_owner: str = "",
    stop_class: StopClass = StopClass.NONE,
    safety_class: str = "",
    blocked_by: str = "",
    reason: str = "",
    service_may_override: bool = False,
    not_overridden_by: Optional[List[str]] = None,
) -> ArbitrationResult:
    safe_summary = dict(summary or {})
    safe_summary.setdefault("vy_cmd_raw", float(vy))
    safe_summary["vy_cmd_limited"] = float(vy)
    safe_summary["vy_enabled"] = bool(abs(float(vy)) > 1e-9)
    safe_summary.setdefault("vy_block_reason", "" if abs(float(vy)) > 1e-9 else "lateral_inactive")
    dock_obs = safe_summary.get("docking_observation")
    dock_obs = dock_obs if isinstance(dock_obs, dict) else {}

    def _field(name: str, default: Any = None) -> Any:
        value = safe_summary.get(name)
        if value is None:
            value = dock_obs.get(name, default)
        return value

    def _bool_field(name: str) -> bool:
        return bool(_field(name, False))

    def _optional_field_float(name: str) -> Optional[float]:
        value = _field(name)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    final_depth_latched = _bool_field("final_depth_latched")
    near_table_latched = _bool_field("near_table_latched")
    depth_p10 = _optional_field_float("table_roi_depth_p10")
    if depth_p10 is None:
        depth_p10 = _optional_field_float("depth_p10")
    vx_cap: Optional[float] = None
    envelope_reason = ""
    if final_depth_latched:
        vx_cap = 0.0
        envelope_reason = "final_depth_latched"
    elif near_table_latched:
        vx_cap = abs(_float(safe_summary, "near_slow_max_vx_mps", _float(safe_summary, "depth_envelope_slow_vx_mps", 0.008)))
        envelope_reason = "near_table_latched"
    elif depth_p10 is not None:
        stop_p10 = _float(safe_summary, "depth_envelope_stop_p10_m", 0.55)
        slow_p10 = _float(safe_summary, "depth_envelope_slow_p10_m", 0.65)
        mid_p10 = _float(safe_summary, "depth_envelope_mid_p10_m", 0.80)
        if depth_p10 <= stop_p10:
            vx_cap = 0.0
            envelope_reason = "depth_p10_stop"
        elif depth_p10 <= slow_p10:
            vx_cap = abs(_float(safe_summary, "depth_envelope_slow_vx_mps", 0.008))
            envelope_reason = "depth_p10_slow"
        elif depth_p10 <= mid_p10:
            vx_cap = abs(_float(safe_summary, "depth_envelope_mid_vx_mps", 0.012))
            envelope_reason = "depth_p10_mid"
    final_vx = float(vx)
    if vx_cap is not None and abs(final_vx) > vx_cap:
        final_vx = max(-vx_cap, min(vx_cap, final_vx))
    if near_table_latched and not final_depth_latched and action == DockingAction.EDGE_APPROACH_FORWARD:
        action = DockingAction.NEAR_EDGE_FORWARD
        stage = DockingStage.NEAR_EDGE_APPROACH
        forward_owner = str(forward_owner or "near_depth")
        safe_summary["docking_reason"] = str(safe_summary.get("docking_reason") or "near_table_latched")
    if final_depth_latched and action not in {DockingAction.FINAL_LOCKED_STOP, DockingAction.FINAL_YAW_ALIGN}:
        action = DockingAction.FINAL_LOCKED_STOP
        stage = DockingStage.FINAL_DISTANCE_HOLD
        forward_owner = "none"
        if str(safe_summary.get("docking_reason") or "") in {"", "near_table_latched", "near_edge_forward", "near_hold"}:
            safe_summary["docking_reason"] = "final_depth_latched"
    if envelope_reason:
        safe_summary["depth_speed_envelope_reason"] = envelope_reason
        safe_summary["depth_speed_envelope_vx_cap"] = float(vx_cap if vx_cap is not None else 0.0)
    edge_score = _float(safe_summary, "edge_readiness_score", 0.0)
    edge_enter = _float(safe_summary, "edge_readiness_enter_score", 0.65)
    edge_ready_for_approach = bool(_bool_field("edge_trusted") or bool(safe_summary.get("edge_handoff_complete", False)) or edge_score >= edge_enter)
    safe_summary.setdefault("edge_ready_for_approach", edge_ready_for_approach)
    safe_summary.setdefault("edge_ready_for_final", bool(_bool_field("edge_valid") or _bool_field("edge_trusted")))
    actual_block = str(blocked_by or safe_summary.get("effective_block_reason") or safe_summary.get("forward_block_reason") or safe_summary.get("effective_forward_block_reason") or "")
    moving = bool(abs(float(final_vx)) > 1e-9 or abs(float(vy)) > 1e-9 or abs(float(wz)) > 1e-9)
    safe_summary["effective_block_reason"] = "" if moving else actual_block
    return _from_docking_result(
        DockingMotionResult(
            action=action,
            stage=stage,
            vx=float(final_vx),
            vy=float(vy),
            wz=float(wz),
            yaw_owner=str(yaw_owner or ""),
            forward_owner=str(forward_owner or ""),
            lateral_owner=str(lateral_owner or ""),
            stop_class=stop_class,
            safety_class=str(safety_class or ""),
            blocked_by=str(blocked_by or ""),
            reason=str(reason or action.value.lower()),
            summary=safe_summary,
            service_may_override=bool(service_may_override),
            not_overridden_by=list(not_overridden_by or []),
        )
    )


def arbitrate_table_docking_motion(
    ctx: Any,
    obs: Any,
    intent: MotionIntent,
    current_summary: Optional[Dict[str, Any]] = None,
) -> ArbitrationResult:
    """Return the single final table-docking motion command.

    Priority order:
    emergency, hard safety, final locked, final yaw, final hold, near/edge
    approach, bbox/search recovery, and zero escape.
    """
    summary = dict(current_summary or {})
    docking_obs = build_docking_observation(ctx, obs, summary)
    docking_stage = update_docking_stage(ctx, docking_obs)
    summary.update(
        {
            "docking_stage": docking_stage.value,
            "docking_observation": docking_obs.to_dict(),
        }
    )
    state = _state_value(ctx)
    fov_level = _fov_level(summary)
    fov_reason = str(summary.get("bbox_fov_guard_reason") or summary.get("fov_guard_reason") or "")
    stale_policy = _stale_policy(summary)
    phase = str(summary.get("control_phase") or "").strip().upper()
    intent_type = str(intent.intent_type or summary.get("control_source") or "").strip().lower()
    desired_vx = float(intent.desired_vx or 0.0)
    desired_vy = float(intent.desired_vy or _float(summary, "desired_vy", _float(summary, "vy_mps", 0.0)))
    desired_wz = float(intent.desired_wz or 0.0)
    emergency_active = bool(_as_bool(summary, "explicit_stop_active") or any(_as_bool(summary, key) for key in _EMERGENCY_KEYS))
    explicit_stop = bool(summary.get("explicit_stop_active", False))
    hard_safety = bool(any(_as_bool(summary, key) for key in _HARD_SAFETY_KEYS))
    lateral_enabled = bool(summary.get("lateral_enabled", False))
    lateral_err_m = docking_obs.lateral_err_m
    lateral_err_norm = docking_obs.lateral_err_norm
    lateral_source = str(docking_obs.lateral_source or "")
    vy_block_reason = "lateral_disabled" if not lateral_enabled else "vy_direction_uncalibrated"
    edge_readiness_score = max(0.0, min(1.0, _float(summary, "edge_readiness_score", float(getattr(ctx, "edge_readiness_score", 0.0) or 0.0))))
    edge_readiness_enter = _float(summary, "edge_readiness_enter_score", 0.65)
    edge_readiness_exit = _float(summary, "edge_readiness_exit_score", 0.35)
    edge_readiness_ready = bool(edge_readiness_score >= edge_readiness_enter)
    edge_handoff_complete = bool(summary.get("edge_handoff_complete", False) or getattr(ctx, "edge_handoff_complete", False))
    edge_readiness_level = str(summary.get("edge_readiness_level") or getattr(ctx, "edge_readiness_level", "") or "").strip().lower()
    edge_approach_gate_ready = bool(
        docking_obs.edge_trusted
        or edge_handoff_complete
        or edge_readiness_ready
        or edge_readiness_level in {"ready", "trusted", "handoff_ready", "approach_ready"}
    )
    now_mono = time.monotonic()
    forward_commit_active = bool(
        float(getattr(ctx, "forward_commit_until_mono", 0.0) or 0.0) > now_mono
        and not bool(summary.get("near_table_latched", False))
        and not bool(summary.get("final_depth_latched", False))
        and not emergency_active
        and not explicit_stop
        and not hard_safety
        and fov_level != FovGuardLevel.HARD
    )
    edge_lost_age_s = 999.0
    if docking_obs.edge_found or docking_obs.edge_valid or docking_obs.edge_trusted:
        edge_lost_age_s = 0.0
    elif docking_obs.last_good_edge_yaw_age_ms is not None:
        edge_lost_age_s = max(0.0, float(docking_obs.last_good_edge_yaw_age_ms) / 1000.0)
    elif float(getattr(ctx, "last_good_edge_yaw_mono", 0.0) or 0.0) > 0.0:
        edge_lost_age_s = max(0.0, now_mono - float(getattr(ctx, "last_good_edge_yaw_mono", 0.0) or 0.0))
    elif float(getattr(ctx, "last_edge_good_mono", 0.0) or 0.0) > 0.0:
        edge_lost_age_s = max(0.0, now_mono - float(getattr(ctx, "last_edge_good_mono", 0.0) or 0.0))
    roi_depth_valid = bool(summary.get("table_roi_depth_valid", getattr(obs, "table_roi_depth_valid", False) if obs is not None else False))
    roi_depth_value = summary.get("table_roi_depth_p10", getattr(obs, "table_roi_depth_p10", None) if obs is not None else None)
    if roi_depth_value is None:
        roi_depth_value = summary.get("table_roi_depth_median", getattr(obs, "table_roi_depth_median", None) if obs is not None else None)
    try:
        roi_depth_m = float(roi_depth_value) if roi_depth_value is not None else None
    except (TypeError, ValueError):
        roi_depth_m = None

    def bbox_recenter_vy(*, near: bool = False) -> tuple[float, str]:
        if not lateral_enabled:
            return 0.0, "lateral_disabled"
        if not docking_obs.bbox_control_valid or docking_obs.bbox_center_error is None:
            return 0.0, "bbox_invalid"
        if fov_level == FovGuardLevel.HARD:
            return 0.0, "bbox_fov_guard_hard"
        if emergency_active or explicit_stop:
            return 0.0, "emergency_or_explicit_stop"
        if hard_safety or stale_policy == StalePolicy.HARD_STOP:
            return 0.0, "hard_safety_or_stale"
        if bool(summary.get("final_depth_latched", False)):
            return 0.0, "final_depth_latched"
        if roi_depth_valid and roi_depth_m is not None and roi_depth_m <= _float(summary, "depth_envelope_stop_p10_m", 0.55):
            return 0.0, "depth_p10_stop"
        bbox_err = float(docking_obs.bbox_center_error)
        deadband = abs(_float(summary, "lateral_deadband_norm", 0.045))
        if abs(bbox_err) < deadband:
            return 0.0, "bbox_vy_deadband"
        kp = abs(_float(summary, "lateral_kp", 0.045))
        near_cap = abs(_float(summary, "near_slow_max_vy_mps", 0.0025))
        slow_depth = bool(roi_depth_valid and roi_depth_m is not None and roi_depth_m <= _float(summary, "depth_envelope_slow_p10_m", 0.65))
        vy_max = near_cap if (near or slow_depth) else abs(_float(summary, "lateral_vy_max_mps", 0.010))
        if vy_max <= 1e-9:
            return 0.0, "bbox_vy_zero_cap"
        raw = -kp * bbox_err
        return max(-vy_max, min(vy_max, raw)), ""

    def with_common(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        bbox_track_started = float(getattr(ctx, "bbox_track_entered_mono", 0.0) or 0.0)
        out = dict(summary)
        out.update(
            {
                "fov_guard_level": fov_level.value,
                "fov_guard_reason": fov_reason,
                "bbox_fov_guard_level": fov_level.value,
                "bbox_fov_guard_reason": fov_reason,
                "stale_policy": stale_policy.value,
                "active_table_docking": bool(_active_table_docking(ctx)),
                "final_locked": bool(getattr(ctx, "final_locked", False)),
                "bbox_track_elapsed_ms": max(0.0, (time.monotonic() - bbox_track_started) * 1000.0) if bbox_track_started > 0.0 else 0.0,
                "bbox_track_exit_reason": str(getattr(ctx, "bbox_track_last_exit_reason", "") or summary.get("bbox_track_exit_reason") or ""),
                "edge_readiness_score": float(edge_readiness_score),
                "edge_readiness_level": str(summary.get("edge_readiness_level") or getattr(ctx, "edge_readiness_level", "") or ""),
                "edge_readiness_enter_score": float(edge_readiness_enter),
                "edge_readiness_exit_score": float(edge_readiness_exit),
                "edge_handoff_block_reason": str(summary.get("edge_handoff_block_reason") or ""),
                "edge_handoff_source": str(summary.get("edge_handoff_source") or ""),
                "lateral_enabled": bool(lateral_enabled),
                "lateral_err_norm": lateral_err_norm,
                "lateral_err_m": lateral_err_m,
                "lateral_source": lateral_source,
                "lateral_owner": "none",
                "vy_enabled": False,
                "vy_block_reason": vy_block_reason,
                "vy_cmd_raw": float(desired_vy),
                "vy_cmd_limited": 0.0,
                "edge_lost_age_s": float(edge_lost_age_s),
            }
        )
        if extra:
            out.update(extra)
        return out

    def edge_final_wz() -> tuple[float, str]:
        edge_wz = 0.0
        yaw_source = ""
        # 1. Use fresh yaw only if edge is usable
        if docking_obs.edge_usable and docking_obs.yaw_err_rad is not None:
            for key, source in (
                ("edge_yaw_cmd_for_final_align", "edge"),
                ("edge_yaw_cmd", "edge"),
                ("wz_from_plane", "edge"),
            ):
                candidate = _float(summary, key, 0.0)
                if abs(candidate) > 1e-9:
                    edge_wz = candidate
                    yaw_source = source
                    break
        # 2. Fall back to last_good_edge only if it is fresh
        last_good_age_ms = summary.get("last_good_edge_yaw_age_ms")
        hold_timeout_ms = float(summary.get("final_yaw_last_good_hold_s") or getattr(getattr(ctx, "cfg", None), "final_yaw_last_good_hold_s", 1.2) or 1.2) * 1000.0
        last_good_fresh = bool(last_good_age_ms is not None and float(last_good_age_ms) <= hold_timeout_ms)
        if abs(edge_wz) <= 1e-9 and last_good_fresh and abs(float(getattr(ctx, "last_good_edge_yaw_cmd", 0.0) or 0.0)) > 1e-9:
            edge_wz = float(getattr(ctx, "last_good_edge_yaw_cmd", 0.0) or 0.0)
            yaw_source = "last_good_edge"
        return edge_wz, str(summary.get("near_stage_yaw_source") or yaw_source or "hold")

    def bbox_recovery_wz() -> float:
        for key in ("bbox_yaw_cmd", "desired_wz", "wz_radps"):
            candidate = _float(summary, key, 0.0)
            if abs(candidate) > 1e-9:
                return candidate
        if abs(desired_wz) > 1e-9:
            return desired_wz
        return _search_wz(ctx, summary)

    def raw_cmd_vx() -> float:
        cmd = summary.get("cmd")
        if isinstance(cmd, dict):
            return _float(cmd, "vx", _float(cmd, "vx_mps", 0.0))
        return 0.0

    if emergency_active or explicit_stop:
        reason = "explicit_stop" if explicit_stop else "emergency_or_obstacle"
        return _docking_result(
            action=DockingAction.EMERGENCY_STOP,
            stage=DockingStage.EMERGENCY_STOP,
            summary=with_common(),
            stop_class=StopClass.EMERGENCY,
            safety_class="emergency",
            blocked_by=reason,
            reason=reason,
            service_may_override=True,
        )

    if hard_safety:
        return _docking_result(
            action=DockingAction.SAFETY_STOP,
            stage=DockingStage.SAFETY_STOP,
            summary=with_common(),
            stop_class=StopClass.SAFETY,
            safety_class="hard_safety",
            blocked_by="hard_safety",
            reason="hard_safety",
            service_may_override=True,
        )

    if bool(summary.get("final_depth_latched", False)):
        final_locked = bool(summary.get("final_locked", False))
        yaw_align_active = bool(summary.get("final_yaw_align_active", False))
        edge_wz, yaw_source = edge_final_wz()
        yaw_abs: Optional[float] = None
        for key in ("edge_yaw", "yaw_err_rad", "yaw_err"):
            if summary.get(key) is not None:
                yaw_abs = abs(_float(summary, key, 0.0))
                break
        yaw_deadband = abs(_float(summary, "final_yaw_deadband_rad", _float(summary, "table_yaw_tol_rad", 0.08)))
        yaw_large = bool(yaw_align_active or (yaw_abs is not None and yaw_abs > yaw_deadband))
        if final_locked:
            return _docking_result(
                action=DockingAction.FINAL_LOCKED_STOP,
                stage=DockingStage.FINAL_LOCKED,
                summary=with_common(
                    {
                    "motion_intent_type": "final_edge_locked",
                    "yaw_owner": yaw_source,
                    "near_stage_yaw_source": yaw_source,
                    "forward_block_reason": "final_depth_latched",
                    }
                ),
                yaw_owner=yaw_source,
                stop_class=StopClass.NONE,
                blocked_by="final_locked",
                reason=str(summary.get("final_lock_reason") or "final_locked"),
            )
        if yaw_large and abs(edge_wz) > 1e-9:
            return _docking_result(
                action=DockingAction.FINAL_YAW_ALIGN,
                stage=DockingStage.FINAL_YAW_ALIGN,
                summary=with_common(
                    {
                    "final_depth_latched": True,
                    "final_yaw_align_active": True,
                    "final_yaw_align_yaw_source": yaw_source,
                    "final_yaw_align_yaw_cmd": float(edge_wz),
                    "final_hold_edge_lost": False,
                    "final_cmd_source": "arbiter_final_yaw_align",
                    "motion_intent_type": "final_align",
                    "yaw_owner": yaw_source,
                    "near_stage_yaw_source": yaw_source,
                    "forward_block_reason": "final_depth_latched",
                    "rotate_block_reason": "",
                    "allow_rotate": True,
                    "rotate_allowed": True,
                    }
                ),
                wz=edge_wz,
                yaw_owner=yaw_source,
                stop_class=StopClass.NONE,
                blocked_by="final_depth_latched",
                reason="final_yaw_align",
            )
        stale_reason = "edge_yaw_stale" if (yaw_large and abs(edge_wz) <= 1e-9) else str(summary.get("final_lock_reason") or "final_hold_edge_lost")
        return _docking_result(
            action=DockingAction.FINAL_LOCKED_STOP,
            stage=DockingStage.FINAL_DISTANCE_HOLD,
            summary=with_common(
                {
                "final_depth_latched": True,
                "final_yaw_align_active": False,
                "final_hold_edge_lost": True,
                "motion_intent_type": "final_hold_edge_lost",
                "yaw_owner": yaw_source,
                "near_stage_yaw_source": yaw_source,
                "forward_block_reason": "final_depth_latched",
                "rotate_block_reason": stale_reason,
                }
            ),
            yaw_owner=yaw_source,
            stop_class=StopClass.NONE,
            blocked_by="final_depth_latched",
            reason=stale_reason,
        )

    final_stop = bool(
        summary.get("depth_roi_stop_ready", False)
        or phase == "DEPTH_FINAL_STOP"
        or state in {"FINAL_SLOW_STOP", "AT_TABLE_EDGE"}
        or str(summary.get("stop_source") or "").strip().lower() in {"roi_depth", "final_lock"}
    )
    if final_stop:
        if state == "AT_TABLE_EDGE":
            return _docking_result(
                action=DockingAction.FINAL_LOCKED_STOP,
                stage=DockingStage.FINAL_LOCKED,
                summary=with_common({"final_locked": True, "final_depth_latched": True, "final_yaw_align_active": False}),
                blocked_by="final_locked",
                reason=str(summary.get("final_lock_reason") or "at_table_edge"),
            )
        edge_wz, yaw_source = edge_final_wz()
        if abs(edge_wz) > 1e-9:
            return _docking_result(
                action=DockingAction.FINAL_YAW_ALIGN,
                stage=DockingStage.FINAL_YAW_ALIGN,
                summary=with_common(
                    {
                        "final_depth_latched": True,
                        "final_yaw_align_active": True,
                        "final_yaw_align_yaw_cmd": float(edge_wz),
                        "final_cmd_source": "arbiter_final_yaw_align",
                        "forward_block_reason": "final_depth_latched",
                        "rotate_block_reason": "",
                        "allow_rotate": True,
                        "rotate_allowed": True,
                    }
                ),
                wz=edge_wz,
                yaw_owner=yaw_source,
                blocked_by="final_depth_stop",
                reason="final_yaw_align",
            )
        return _docking_result(
            action=DockingAction.FINAL_LOCKED_STOP,
            stage=DockingStage.FINAL_DISTANCE_HOLD,
            summary=with_common({"forward_block_reason": "final_depth_latched"}),
            blocked_by="final_depth_stop",
            reason="final_distance_hold",
        )

    last_good_obs_healthy = bool(summary.get("last_good_obs_healthy", False))
    last_good_obs_age_ms = float(summary.get("last_good_obs_age_ms", 999999.0))
    last_good_expired = bool(not last_good_obs_healthy or last_good_obs_age_ms > 2500.0)
    stale_level = str(summary.get("stale_level") or "fresh").strip().lower()
    is_dead_no_last_good = bool(stale_level == "dead" and last_good_expired)

    if is_dead_no_last_good:
        if bool(summary.get("final_depth_latched", False)):
            edge_wz, yaw_source = edge_final_wz()
            return _docking_result(
                action=DockingAction.FINAL_LOCKED_STOP,
                stage=DockingStage.FINAL_DISTANCE_HOLD,
                summary=with_common({
                    "forward_block_reason": "stale_dead_no_last_good",
                    "near_latch_block_reason": "stale_dead_no_last_good",
                }),
                vx=0.0,
                vy=0.0,
                wz=edge_wz if abs(edge_wz) > 1e-9 else 0.0,
                yaw_owner=yaw_source if abs(edge_wz) > 1e-9 else "hold",
                blocked_by="stale_dead_no_last_good",
                reason="stale_dead_no_last_good_final_hold",
            )
        else:
            wz = bbox_recovery_wz()
            return _docking_result(
                action=DockingAction.CONTROL_RECOVERY_ROTATE if abs(wz) > 1e-9 else DockingAction.SEARCH_ROTATE,
                stage=DockingStage.RECOVERY_ROTATE,
                summary=with_common({
                    "forward_block_reason": "stale_dead_no_last_good",
                    "near_latch_block_reason": "stale_dead_no_last_good",
                }),
                vx=0.0,
                vy=0.0,
                wz=wz,
                yaw_owner="last_good_edge" if abs(wz) > 1e-9 else "search",
                stop_class=StopClass.STALE_RECOVERY if abs(wz) <= 1e-9 else StopClass.NONE,
                blocked_by="stale_dead_no_last_good",
                reason="stale_dead_no_last_good_recovery",
            )

    if stale_policy == StalePolicy.DROPOUT_HOLD and edge_lost_age_s <= _float(summary, "edge_long_dropout_s", 1.2):
        vx = max(abs(desired_vx), min(_float(summary, "forward_commit_vx", 0.020), 0.012))
        short_dropout = bool(edge_lost_age_s < _float(summary, "edge_short_dropout_s", 0.8))
        vy, vy_block = bbox_recenter_vy()
        wz = 0.0 if short_dropout and docking_obs.bbox_control_valid else (desired_wz if abs(desired_wz) > 1e-9 else _float(summary, "last_edge_yaw_cmd", 0.0))
        return _docking_result(
            action=DockingAction.PERCEPTION_DROPOUT_HOLD,
            stage=DockingStage.PERCEPTION_DROPOUT_HOLD,
            summary=with_common({
                "perception_dropout_hold_active": True,
                "forward_owner": "approach_commit",
                "lateral_owner": "bbox" if abs(vy) > 1e-9 else "none",
                "vy_block_reason": vy_block,
                "vy_cmd_raw": float(vy),
                "vy_cmd_limited": float(vy),
                "vy_enabled": bool(abs(vy) > 1e-9),
                "advance_condition": "last_good_obs_unexpired",
                "fallback_condition": "dropout_hold_expired",
            }),
            vx=vx,
            vy=vy,
            wz=wz,
            yaw_owner="hold" if short_dropout else "edge_hold",
            forward_owner="approach_commit",
            lateral_owner="bbox" if abs(vy) > 1e-9 else "none",
            reason="perception_dropout_hold",
        )

    if stale_policy == StalePolicy.HARD_STOP and state not in {"SEARCH_TABLE"} and not bool(summary.get("search_table_stale_gate_bypass", False)):
        wz = _float(summary, "last_edge_yaw_cmd", _float(summary, "last_good_edge_yaw_cmd", 0.0))
        if abs(wz) <= 1e-9:
            wz = bbox_recovery_wz()
        return _docking_result(
            action=DockingAction.CONTROL_RECOVERY_ROTATE if abs(wz) > 1e-9 else DockingAction.SEARCH_ROTATE,
            stage=DockingStage.RECOVERY_ROTATE,
            summary=with_common({"forward_block_reason": str(summary.get("stale_level") or "hard_stale")}),
            wz=wz,
            yaw_owner="last_good_edge" if abs(wz) > 1e-9 else "search",
            stop_class=StopClass.STALE_RECOVERY if abs(wz) <= 1e-9 else StopClass.NONE,
            blocked_by=str(summary.get("stale_hold_policy") or summary.get("stale_level") or "hard_stale"),
            reason="stale_recovery",
        )

    edge_block = str(summary.get("edge_guided_commit_block_reason") or "").strip().lower()
    if edge_block in {"explicit_stop", "base_safety"}:
        stop_class = "emergency" if edge_block == "explicit_stop" else "safety"
        return _docking_result(
            action=DockingAction.EMERGENCY_STOP if stop_class == "emergency" else DockingAction.SAFETY_STOP,
            stage=DockingStage.EMERGENCY_STOP if stop_class == "emergency" else DockingStage.SAFETY_STOP,
            summary=with_common(),
            stop_class=StopClass.EMERGENCY if stop_class == "emergency" else StopClass.SAFETY,
            safety_class=stop_class,
            blocked_by=edge_block,
            reason=edge_block,
            service_may_override=True,
        )
    if edge_block in {"hard_stale", "perception_dropout_hold_expired"}:
        wz = _float(summary, "last_edge_yaw_cmd", _float(summary, "last_good_edge_yaw_cmd", 0.0))
        if abs(wz) <= 1e-9:
            wz = bbox_recovery_wz()
        return _docking_result(
            action=DockingAction.CONTROL_RECOVERY_ROTATE,
            stage=DockingStage.RECOVERY_ROTATE,
            summary=with_common({"forward_block_reason": edge_block}),
            wz=wz,
            yaw_owner="last_good_edge" if abs(wz) > 1e-9 else "search",
            stop_class=StopClass.NONE,
            blocked_by=edge_block,
            reason=edge_block,
        )
    if edge_block == "depth_final_stop":
        return _docking_result(
            action=DockingAction.FINAL_LOCKED_STOP,
            stage=DockingStage.FINAL_DISTANCE_HOLD,
            summary=with_common({"forward_block_reason": "final_depth_latched"}),
            blocked_by=edge_block,
            reason="final_distance_hold",
        )
    if edge_block in {"bbox_fov_guard_hard", "edge_yaw_too_large"}:
        wz = bbox_recovery_wz() if edge_block == "bbox_fov_guard_hard" else _float(summary, "edge_yaw_cmd", _float(summary, "wz_from_plane", desired_wz))
        return _docking_result(
            action=DockingAction.CONTROL_RECOVERY_ROTATE,
            stage=DockingStage.RECOVERY_ROTATE,
            summary=with_common({"forward_block_reason": edge_block, "rotate_block_reason": ""}),
            wz=wz,
            yaw_owner="bbox" if edge_block == "bbox_fov_guard_hard" else "edge",
            blocked_by=edge_block,
            reason=edge_block,
        )

    if bool(summary.get("near_table_latched", False)):
        edge_wz = _float(summary, "edge_yaw_cmd", _float(summary, "wz_from_plane", 0.0))
        if abs(edge_wz) <= 1e-9:
            edge_wz = _float(summary, "last_good_edge_yaw_cmd", 0.0)
        yaw_source = str(summary.get("near_stage_yaw_source") or ("last_good_edge" if abs(edge_wz) > 1e-9 else "hold"))
        near_vx = max(0.0, min(abs(desired_vx), abs(_float(summary, "near_slow_max_vx_mps", 0.008))))
        near_vy, near_vy_block = bbox_recenter_vy(near=True)
        return _docking_result(
            action=DockingAction.NEAR_EDGE_FORWARD,
            stage=DockingStage.NEAR_EDGE_APPROACH,
            summary=with_common(
                {
                "motion_intent_type": "near_edge_hold",
                "yaw_owner": yaw_source,
                "forward_owner": "near_depth" if near_vx > 1e-9 else "none",
                "lateral_owner": "bbox" if abs(near_vy) > 1e-9 else "none",
                "vy_block_reason": near_vy_block,
                "vy_cmd_raw": float(near_vy),
                "vy_cmd_limited": float(near_vy),
                "vy_enabled": bool(abs(near_vy) > 1e-9),
                "near_stage_yaw_source": yaw_source,
                "forward_block_reason": "" if near_vx > 1e-9 else "near_table_latched",
                "bbox_lost_ignored_due_to_near_latch": bool(intent_type in {"local_rotate_search", "search"} or phase == "SEARCH_SCAN"),
                }
            ),
            vx=near_vx,
            vy=near_vy,
            wz=edge_wz,
            yaw_owner=yaw_source,
            forward_owner="near_depth" if near_vx > 1e-9 else "none",
            lateral_owner="bbox" if abs(near_vy) > 1e-9 else "none",
            stop_class=StopClass.NONE,
            blocked_by="" if near_vx > 1e-9 else "near_table_latched",
            reason="near_edge_forward" if near_vx > 1e-9 else "near_hold",
        )

    if fov_level == FovGuardLevel.HARD:
        wz = bbox_recovery_wz()
        return _docking_result(
            action=DockingAction.BBOX_REACQUIRE_ROTATE if abs(wz) > 1e-9 else DockingAction.CONTROL_RECOVERY_ROTATE,
            stage=DockingStage.RECOVERY_ROTATE,
            summary=with_common({"forward_block_reason": "bbox_fov_guard_hard", "rotate_block_reason": "", "bbox_track_exit_reason": "bbox_fov_guard_hard"}),
            wz=wz,
            yaw_owner="bbox",
            blocked_by="bbox_fov_guard_hard",
            reason=fov_reason or "bbox_fov_guard_hard",
        )

    if phase == "EDGE_GUIDED_APPROACH" and intent.forward_allowed_by_behavior and edge_approach_gate_ready:
        edge_yaw_abs = abs(_float(summary, "edge_yaw", _float(summary, "yaw_err_rad", 0.0)))
        if edge_yaw_abs > _float(summary, "edge_forward_rotate_only_yaw_rad", 0.18):
            wz = _float(summary, "edge_yaw_cmd", _float(summary, "wz_from_plane", desired_wz))
            return _docking_result(
                action=DockingAction.CONTROL_RECOVERY_ROTATE,
                stage=DockingStage.RECOVERY_ROTATE,
                summary=with_common({"effective_forward_block_reason": "edge_yaw_large", "forward_block_reason": "edge_yaw_large", "rotate_block_reason": ""}),
                wz=wz,
                yaw_owner="edge",
                blocked_by="edge_yaw_large",
                reason="edge_yaw_large",
            )
        vx = desired_vx
        if bool(summary.get("approach_commit_active", False)) or _edge_usable(obs, summary):
            vx = max(abs(vx), _float(summary, "forward_commit_vx", 0.020))
        if fov_level == FovGuardLevel.SOFT:
            max_soft_vx = max(0.0, _float(summary, "forward_commit_vx", 0.020))
            vx = min(max(abs(vx), max_soft_vx), max_soft_vx)
        if abs(vx) > 1e-9:
            edge_vy, edge_vy_block = bbox_recenter_vy()
            return _docking_result(
                action=DockingAction.EDGE_APPROACH_FORWARD,
                stage=DockingStage.EDGE_APPROACH,
                summary=with_common(
                    {
                    "fov_guard_level": fov_level.value,
                    "fov_guard_reason": fov_reason,
                    "bbox_fov_soft_allowed_forward": bool(fov_level == FovGuardLevel.SOFT),
                    "stale_policy": stale_policy.value,
                    "lateral_owner": "bbox" if abs(edge_vy) > 1e-9 else "none",
                    "vy_block_reason": edge_vy_block,
                    "vy_cmd_raw": float(edge_vy),
                    "vy_cmd_limited": float(edge_vy),
                    "vy_enabled": bool(abs(edge_vy) > 1e-9),
                    }
                ),
                vx=vx,
                vy=edge_vy,
                wz=desired_wz,
                yaw_owner=str(intent.yaw_owner or "edge"),
                forward_owner="edge_approach",
                lateral_owner="bbox" if abs(edge_vy) > 1e-9 else "none",
                reason="edge_guided_approach_soft_fov" if fov_level == FovGuardLevel.SOFT else "edge_guided_approach",
            )
    elif phase == "EDGE_GUIDED_APPROACH" and intent.forward_allowed_by_behavior:
        summary["edge_handoff_block_reason"] = "edge_approach_gate_not_ready"
        summary["edge_handoff_source"] = str(summary.get("edge_handoff_source") or "readiness_score")

    if phase in {"BBOX_ACQUIRE", "EDGE_HANDOFF_CONFIRM"} and edge_readiness_ready:
        handoff_wz = _float(summary, "edge_yaw_cmd", _float(summary, "wz_from_plane", 0.0))
        yaw_owner = "edge_candidate" if abs(handoff_wz) > 1e-9 else "bbox_hold"
        if abs(handoff_wz) <= 1e-9:
            handoff_wz = bbox_recovery_wz()
        handoff_vx = 0.0
        handoff_depth_ok = bool(
            not bool(summary.get("near_table_latched", False))
            and not bool(summary.get("final_depth_latched", False))
            and not bool(summary.get("depth_roi_stop_ready", False))
        )
        handoff_bbox_ok = bool(
            docking_obs.bbox_control_valid
            and docking_obs.bbox_center_error is not None
            and abs(float(docking_obs.bbox_center_error)) <= abs(_float(summary, "bbox_track_forward_center_band", 0.20))
            and fov_level != FovGuardLevel.HARD
        )
        if handoff_depth_ok and handoff_bbox_ok:
            handoff_vx = min(abs(_float(summary, "edge_handoff_forward_vx_mps", 0.010)), abs(_float(summary, "bbox_track_forward_max_vx_mps", 0.015)))
        handoff_vy, handoff_vy_block = bbox_recenter_vy()
        return _docking_result(
            action=DockingAction.EDGE_READINESS_HANDOFF,
            stage=DockingStage.EDGE_HANDOFF,
            summary=with_common(
                {
                    "forward_block_reason": "edge_readiness_handoff",
                    "rotate_block_reason": "",
                    "advance_condition": "edge_readiness_enter_score",
                    "fallback_condition": "edge_readiness_exit_score",
                    "edge_handoff_source": "readiness_score",
                    "edge_handoff_block_reason": "",
                    "allow_forward": bool(handoff_vx > 1e-9),
                    "allow_rotate": bool(abs(handoff_wz) > 1e-9),
                    "forward_block_reason": "" if handoff_vx > 1e-9 else "edge_readiness_handoff",
                    "lateral_owner": "bbox" if abs(handoff_vy) > 1e-9 else "none",
                    "vy_block_reason": handoff_vy_block,
                    "vy_cmd_raw": float(handoff_vy),
                    "vy_cmd_limited": float(handoff_vy),
                    "vy_enabled": bool(abs(handoff_vy) > 1e-9),
                }
            ),
            vx=handoff_vx,
            vy=handoff_vy,
            wz=handoff_wz,
            yaw_owner=yaw_owner,
            forward_owner="edge_handoff" if handoff_vx > 1e-9 else "none",
            lateral_owner="bbox" if abs(handoff_vy) > 1e-9 else "none",
            reason="edge_readiness_handoff",
        )

    bbox_err = docking_obs.bbox_center_error
    bbox_forward_vx = raw_cmd_vx()
    bbox_track_enabled = bool(summary.get("bbox_track_forward_enabled", True))
    bbox_track_center_band = abs(_float(summary, "bbox_track_forward_center_band", _float(summary, "yolo_forward_center_good_limit", 0.30)))
    bbox_track_vx = abs(_float(summary, "bbox_track_forward_vx_mps", 0.012))
    bbox_track_max_vx = abs(_float(summary, "bbox_track_forward_max_vx_mps", 0.015))
    if bbox_track_max_vx > 0.0:
        bbox_track_vx = min(bbox_track_vx, bbox_track_max_vx)
    bbox_track_max_wz = abs(_float(summary, "bbox_track_forward_max_wz_radps", 0.06))
    bbox_track_min_hold_ms = max(0.0, _float(summary, "bbox_track_forward_min_hold_ms", 800.0))
    bbox_track_block = ""
    now_track = time.monotonic()
    bbox_track_active_since = float(getattr(ctx, "bbox_track_entered_mono", 0.0) or 0.0)
    bbox_track_elapsed_if_active_ms = max(0.0, (now_track - bbox_track_active_since) * 1000.0) if bbox_track_active_since > 0.0 else 0.0
    bbox_track_hold_active = bool(bbox_track_active_since > 0.0 and bbox_track_elapsed_if_active_ms < bbox_track_min_hold_ms)
    bbox_track_hold_band = max(bbox_track_center_band, abs(_float(summary, "yolo_forward_center_hard_limit", 0.25)))
    if forward_commit_active:
        bbox_track_hold_active = True
        bbox_track_hold_band = max(bbox_track_hold_band, 0.30)
    bbox_track_phase_allowed = bool(
        phase in {"BBOX_ACQUIRE", "EDGE_HANDOFF_CONFIRM"}
        or (phase == "EDGE_GUIDED_APPROACH" and not edge_approach_gate_ready)
    )
    near_depth_floor = _float(summary, "bbox_track_forward_min_depth_m", _float(summary, "near_depth_threshold_m", 0.40))
    roi_depth_too_near = bool(roi_depth_valid and roi_depth_m is not None and roi_depth_m <= near_depth_floor)
    if not bbox_track_enabled:
        bbox_track_block = "bbox_track_disabled"
    elif not bbox_track_phase_allowed:
        bbox_track_block = "not_bbox_track_phase"
    elif not docking_obs.bbox_control_valid:
        bbox_track_block = "bbox_invalid"
    elif bbox_err is None:
        bbox_track_block = "bbox_center_missing"
    elif abs(float(bbox_err)) > bbox_track_center_band and not (bbox_track_hold_active and abs(float(bbox_err)) <= bbox_track_hold_band):
        bbox_track_block = "bbox_center_error_large"
    elif fov_level == FovGuardLevel.HARD:
        bbox_track_block = "bbox_fov_guard_hard"
    elif bool(summary.get("near_table_latched", False)):
        bbox_track_block = "near_table_latched"
    elif bool(summary.get("final_depth_latched", False)):
        bbox_track_block = "final_depth_latched"
    elif bool(summary.get("depth_roi_stop_ready", False)):
        bbox_track_block = "depth_final_stop"
    elif roi_depth_too_near:
        bbox_track_block = "roi_depth_too_near"
    elif edge_readiness_ready:
        bbox_track_block = "edge_readiness_ready"
    elif emergency_active or explicit_stop:
        bbox_track_block = "emergency_or_explicit_stop"
    elif hard_safety:
        bbox_track_block = "hard_safety"
    elif bbox_track_vx <= 1e-9:
        bbox_track_block = "bbox_track_zero_vx"
    if (
        not bbox_track_block
        and bool(summary.get("yolo_forward_allowed", abs(bbox_forward_vx) > 1e-9))
    ):
        if float(getattr(ctx, "bbox_track_entered_mono", 0.0) or 0.0) <= 0.0:
            try:
                ctx.bbox_track_entered_mono = now_track
            except Exception:
                pass
        elapsed_ms = max(0.0, (now_track - float(getattr(ctx, "bbox_track_entered_mono", now_track) or now_track)) * 1000.0)
        if (not roi_depth_valid or roi_depth_m is None or roi_depth_m > 1.2) and docking_obs.bbox_control_valid:
            bbox_track_vx = min(abs(_float(summary, "far_bbox_track_vx_mps", 0.016)), bbox_track_max_vx if bbox_track_max_vx > 0.0 else 0.018)
        elif roi_depth_m is not None and roi_depth_m > 0.8:
            bbox_track_vx = min(max(bbox_track_vx, 0.012), min(bbox_track_max_vx if bbox_track_max_vx > 0.0 else 0.015, 0.015))
        desired_bbox_wz = desired_wz if abs(desired_wz) > 1e-9 else _float(summary, "bbox_yaw_cmd", 0.0)
        if bbox_track_max_wz > 0.0:
            desired_bbox_wz = max(-bbox_track_max_wz, min(bbox_track_max_wz, desired_bbox_wz))
        bbox_vy, bbox_vy_block = bbox_recenter_vy()
        return _docking_result(
            action=DockingAction.BBOX_TRACK_FORWARD,
            stage=DockingStage.BBOX_ACQUIRE,
            summary=with_common(
                {
                    "control_source": "yolo_track_forward",
                    "allow_forward": True,
                    "allow_rotate": bool(abs(desired_wz) > 1e-9),
                    "forward_block_reason": "",
                    "rotate_block_reason": "yolo_track_forward" if abs(desired_wz) <= 1e-9 else "",
                    "fallback_action": "yolo_assist",
                    "bbox_track_forward_enabled": bool(bbox_track_enabled),
                    "bbox_track_forward_vx_mps": float(bbox_track_vx),
                    "bbox_track_forward_max_vx_mps": float(bbox_track_max_vx),
                    "bbox_track_forward_center_band": float(bbox_track_center_band),
                    "bbox_track_forward_min_hold_ms": float(bbox_track_min_hold_ms),
                    "bbox_track_forward_max_wz_radps": float(bbox_track_max_wz),
                    "bbox_track_elapsed_ms": float(elapsed_ms),
                    "bbox_track_exit_reason": "",
                    "forward_owner": "bbox_track",
                    "lateral_owner": "bbox" if abs(bbox_vy) > 1e-9 else "none",
                    "vy_block_reason": bbox_vy_block,
                    "vy_cmd_raw": float(bbox_vy),
                    "vy_cmd_limited": float(bbox_vy),
                    "vy_enabled": bool(abs(bbox_vy) > 1e-9),
                    "advance_condition": "bbox_centered_depth_far",
                    "fallback_condition": "bbox_track_exit",
                }
            ),
            vx=bbox_track_vx,
            vy=bbox_vy,
            wz=desired_bbox_wz,
            yaw_owner="bbox",
            forward_owner="bbox_track",
            lateral_owner="bbox" if abs(bbox_vy) > 1e-9 else "none",
            reason="bbox_track_forward_compatible",
        )
    if bbox_track_block:
        try:
            ctx.bbox_track_last_exit_reason = bbox_track_block
            ctx.bbox_track_entered_mono = 0.0
        except Exception:
            pass
        summary["bbox_track_exit_reason"] = bbox_track_block
        summary["bbox_track_elapsed_ms"] = 0.0

    bbox_reacquire_needed = bool(
        bbox_err is None
        or abs(float(bbox_err)) > bbox_track_center_band
        or fov_level == FovGuardLevel.HARD
    )
    if (
        phase in {"BBOX_ACQUIRE", "EDGE_HANDOFF_CONFIRM"}
        or (phase == "EDGE_GUIDED_APPROACH" and not edge_approach_gate_ready and docking_obs.bbox_control_valid)
    ) and bbox_reacquire_needed and (intent.rotate_allowed_by_behavior or abs(bbox_recovery_wz()) > 1e-9):
        wz = bbox_recovery_wz()
        return _docking_result(
            action=DockingAction.BBOX_REACQUIRE_ROTATE,
            stage=DockingStage.BBOX_ACQUIRE if phase in {"BBOX_ACQUIRE", "EDGE_GUIDED_APPROACH"} else DockingStage.EDGE_HANDOFF,
            summary=with_common({"rotate_block_reason": "", "allow_rotate": True, "forward_block_reason": "bbox_acquire"}),
            wz=wz,
            yaw_owner="bbox",
            reason="bbox_acquire_rotate",
        )

    if state == "SEARCH_TABLE" or intent_type in {"local_rotate_search", "search"} or phase == "SEARCH_SCAN":
        wz = desired_wz if abs(desired_wz) > 1e-9 else _search_wz(ctx, summary)
        return _docking_result(
            action=DockingAction.SEARCH_ROTATE,
            stage=DockingStage.SEARCH,
            summary=with_common({"search_table_stale_gate_bypass": True}),
            wz=wz,
            yaw_owner="search",
            reason="search_rotate",
        )

    if _active_table_docking(ctx) and abs(desired_vx) < 1e-9 and abs(desired_wz) < 1e-9:
        zero_age_ms = _float(summary, "zero_cmd_age_ms", 0.0)
        if zero_age_ms >= 800.0:
            if bool(summary.get("approach_commit_active", False)):
                return _docking_result(
                    action=DockingAction.NEAR_EDGE_FORWARD,
                    stage=DockingStage.RECOVERY_ROTATE,
                    summary=with_common({"zero_escape_reason": "forward_coast"}),
                    vx=_float(summary, "forward_commit_vx", 0.020),
                    wz=_float(summary, "last_edge_yaw_cmd", 0.0),
                    yaw_owner="last_good_edge",
                    reason="zero_watchdog_forward_coast",
                )
            wz = bbox_recovery_wz()
            action = DockingAction.BBOX_REACQUIRE_ROTATE if summary.get("bbox_center_valid") else DockingAction.SEARCH_ROTATE
            return _docking_result(
                action=action,
                stage=DockingStage.RECOVERY_ROTATE,
                summary=with_common({"zero_escape_reason": "bbox_reacquire_rotate" if action == DockingAction.BBOX_REACQUIRE_ROTATE else "search_rotate"}),
                wz=wz,
                yaw_owner="bbox" if action == DockingAction.BBOX_REACQUIRE_ROTATE else "search",
                reason="zero_watchdog_search_rotate",
            )

    return _docking_result(
        action=DockingAction.EDGE_APPROACH_FORWARD if abs(desired_vx) > 1e-9 else DockingAction.CONTROL_RECOVERY_ROTATE,
        stage=docking_stage,
        summary=with_common({"zero_escape_reason": ""}),
        vx=desired_vx,
        vy=desired_vy,
        wz=desired_wz,
        yaw_owner=str(intent.yaw_owner or summary.get("yaw_owner") or ""),
        stop_class=StopClass.NONE if (abs(desired_vx) > 1e-9 or abs(desired_wz) > 1e-9) else StopClass.CONTROL_RECOVERY,
        blocked_by=str(summary.get("forward_block_reason") or ""),
        reason=str(intent.reason or summary.get("reason") or "candidate_intent"),
    )
