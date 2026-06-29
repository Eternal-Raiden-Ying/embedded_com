#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed table-docking control model.

The types in this module are the source of truth for table-docking motion.
Legacy summary fields are derived from :class:`DockingMotionResult` only for
log compatibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DockingStage(str, Enum):
    SEARCH = "SEARCH"
    BBOX_ACQUIRE = "BBOX_ACQUIRE"
    EDGE_HANDOFF = "EDGE_HANDOFF"
    EDGE_APPROACH = "EDGE_APPROACH"
    NEAR_EDGE_APPROACH = "NEAR_EDGE_APPROACH"
    PERCEPTION_DROPOUT_HOLD = "PERCEPTION_DROPOUT_HOLD"
    FINAL_DISTANCE_HOLD = "FINAL_DISTANCE_HOLD"
    FINAL_YAW_ALIGN = "FINAL_YAW_ALIGN"
    FINAL_LOCKED = "FINAL_LOCKED"
    RECOVERY_ROTATE = "RECOVERY_ROTATE"
    SAFETY_STOP = "SAFETY_STOP"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class DockingAction(str, Enum):
    SEARCH_ROTATE = "SEARCH_ROTATE"
    BBOX_REACQUIRE_ROTATE = "BBOX_REACQUIRE_ROTATE"
    BBOX_TRACK_FORWARD = "BBOX_TRACK_FORWARD"
    EDGE_READINESS_HANDOFF = "EDGE_READINESS_HANDOFF"
    EDGE_APPROACH_FORWARD = "EDGE_APPROACH_FORWARD"
    NEAR_EDGE_FORWARD = "NEAR_EDGE_FORWARD"
    NEAR_EDGE_LATERAL_ALIGN = "NEAR_EDGE_LATERAL_ALIGN"
    PERCEPTION_DROPOUT_HOLD = "PERCEPTION_DROPOUT_HOLD"
    FINAL_YAW_ALIGN = "FINAL_YAW_ALIGN"
    FINAL_LOCKED_STOP = "FINAL_LOCKED_STOP"
    CLOSE_RANGE_PROBE = "CLOSE_RANGE_PROBE"
    FINAL_SLOW_PROBE = "FINAL_SLOW_PROBE"
    DEPTH_SAFETY_HOLD = "DEPTH_SAFETY_HOLD"
    CONTROL_RECOVERY_ROTATE = "CONTROL_RECOVERY_ROTATE"
    SAFETY_STOP = "SAFETY_STOP"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class FovGuardLevel(str, Enum):
    NONE = "none"
    SOFT = "soft"
    HARD = "hard"


class StopClass(str, Enum):
    NONE = "none"
    CONTROL_RECOVERY = "control_recovery"
    STALE_RECOVERY = "stale_recovery"
    SAFETY = "safety"
    EMERGENCY = "emergency"


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


@dataclass(frozen=True)
class DockingObservation:
    bbox_valid: bool = False
    bbox_control_valid: bool = False
    bbox_visible: bool = False
    bbox_fresh: bool = False
    bbox_center_error: Optional[float] = None
    bbox_touch_left: bool = False
    bbox_touch_right: bool = False
    bbox_touch_bottom: bool = False
    bbox_area_ratio: Optional[float] = None
    bbox_yaw_cmd: float = 0.0
    bbox_fov_guard_level: FovGuardLevel = FovGuardLevel.NONE
    bbox_fov_guard_reason: str = ""

    edge_found: bool = False
    edge_valid: bool = False
    edge_trusted: bool = False
    edge_usable: bool = False
    yaw_err_rad: Optional[float] = None
    edge_yaw_cmd: float = 0.0
    last_good_edge_yaw_cmd: float = 0.0
    last_good_edge_yaw_age_ms: Optional[float] = None

    depth_valid: bool = False
    table_roi_depth_p10: Optional[float] = None
    table_roi_depth_median: Optional[float] = None
    table_roi_depth_mean: Optional[float] = None
    dist_err_m: Optional[float] = None
    lateral_err_norm: Optional[float] = None
    lateral_err_m: Optional[float] = None
    lateral_source: str = ""
    final_depth_latched: bool = False
    near_table_latched: bool = False
    final_yaw_align_active: bool = False
    final_locked: bool = False

    stale_level: str = "fresh"
    stale_policy: str = "fresh"
    last_good_obs_age_ms: Optional[float] = None
    perception_dropout_hold_active: bool = False
    approach_commit_active: bool = False
    zero_cmd_age_ms: float = 0.0

    explicit_stop: bool = False
    obstacle_active: bool = False
    emergency_active: bool = False
    hard_safety_active: bool = False
    hardware_safety_failure: bool = False

    raw_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["bbox_fov_guard_level"] = self.bbox_fov_guard_level.value
        return d


def _bool(summary: Dict[str, Any], key: str, default: bool = False) -> bool:
    return bool(summary.get(key, default))


def _float(summary: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = summary.get(key, default)
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fov_level(raw: Any) -> FovGuardLevel:
    value = str(raw or "none").strip().lower()
    if value == FovGuardLevel.HARD.value:
        return FovGuardLevel.HARD
    if value == FovGuardLevel.SOFT.value:
        return FovGuardLevel.SOFT
    return FovGuardLevel.NONE


def _state_value(ctx: Any) -> str:
    state = getattr(ctx, "state", "")
    return str(getattr(state, "value", state) or "").strip().upper()


def _obs_bool(obs: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(obs, name, default)) if obs is not None else bool(default)


def _obs_value(obs: Any, name: str, default: Any = None) -> Any:
    return getattr(obs, name, default) if obs is not None else default


def build_docking_observation(ctx: Any, obs: Any, summary: Optional[Dict[str, Any]] = None, now: float = 0.0) -> DockingObservation:
    """Normalize table docking signals without deciding final motion."""
    del now
    summary = dict(summary or {})
    bbox_control_valid = bool(
        summary.get("bbox_center_valid")
        or summary.get("table_bbox_control_valid")
        or summary.get("yolo_table_control_valid")
        or _obs_bool(obs, "table_bbox_control_valid")
        or _obs_bool(obs, "yolo_table_control_valid")
    )
    bbox_visible = bool(summary.get("yolo_table_visible") or _obs_bool(obs, "yolo_table_visible") or _obs_bool(obs, "table_found"))
    bbox_fresh = bool(summary.get("yolo_table_fresh") or _obs_bool(obs, "yolo_table_fresh") or _obs_bool(obs, "table_bbox_current_found"))
    bbox_valid = bool(bbox_control_valid and (bbox_visible or bbox_fresh))
    center_error = summary.get("bbox_center_error_control", summary.get("bbox_center_error", summary.get("center_error")))
    if center_error is None:
        cx = summary.get("bbox_cx_norm_control", _obs_value(obs, "yolo_bbox_center_x_norm"))
        if cx is not None:
            center_error = _optional_float(cx)
            if center_error is not None:
                center_error -= 0.5
    bbox_area = summary.get("bbox_area_ratio", summary.get("table_bbox_area_ratio", _obs_value(obs, "table_bbox_area_ratio")))
    edge_found = bool(summary.get("edge_found") or _obs_bool(obs, "edge_found"))
    edge_valid = bool(summary.get("edge_valid") or _obs_bool(obs, "edge_valid") or edge_found)
    edge_trusted = bool(summary.get("edge_trusted") or _obs_bool(obs, "edge_trusted"))
    edge_usable = bool(summary.get("usable_for_approach") or summary.get("edge_usable") or _obs_bool(obs, "usable_for_approach") or edge_trusted or edge_found)
    depth_valid = bool(summary.get("table_roi_depth_valid") or _obs_bool(obs, "table_roi_depth_valid"))
    stale_level = str(summary.get("stale_level") or "fresh").strip().lower() or "fresh"
    stale_policy = str(summary.get("stale_policy") or summary.get("stale_hold_policy") or stale_level).strip().lower()
    explicit_stop = bool(summary.get("explicit_stop_active", False))
    emergency_active = bool(explicit_stop or any(_bool(summary, key) for key in _EMERGENCY_KEYS))
    hard_safety_active = bool(any(_bool(summary, key) for key in _HARD_SAFETY_KEYS))
    return DockingObservation(
        bbox_valid=bbox_valid,
        bbox_control_valid=bbox_control_valid,
        bbox_visible=bbox_visible,
        bbox_fresh=bbox_fresh,
        bbox_center_error=_optional_float(center_error),
        bbox_touch_left=bool(summary.get("bbox_touch_left") or summary.get("table_bbox_touch_left") or _obs_bool(obs, "table_bbox_touch_left")),
        bbox_touch_right=bool(summary.get("bbox_touch_right") or summary.get("table_bbox_touch_right") or _obs_bool(obs, "table_bbox_touch_right")),
        bbox_touch_bottom=bool(summary.get("bbox_touch_bottom") or summary.get("table_bbox_touch_bottom") or _obs_bool(obs, "table_bbox_touch_bottom")),
        bbox_area_ratio=_optional_float(bbox_area),
        bbox_yaw_cmd=_float(summary, "bbox_yaw_cmd", _float(summary, "desired_wz", 0.0)),
        bbox_fov_guard_level=_fov_level(summary.get("bbox_fov_guard_level", summary.get("fov_guard_level"))),
        bbox_fov_guard_reason=str(summary.get("bbox_fov_guard_reason") or summary.get("fov_guard_reason") or ""),
        edge_found=edge_found,
        edge_valid=edge_valid,
        edge_trusted=edge_trusted,
        edge_usable=edge_usable,
        yaw_err_rad=_optional_float(summary.get("edge_yaw", summary.get("yaw_err_rad", _obs_value(obs, "yaw_err_rad")))),
        edge_yaw_cmd=_float(summary, "edge_yaw_cmd", _float(summary, "wz_from_plane", 0.0)),
        last_good_edge_yaw_cmd=_float(summary, "last_good_edge_yaw_cmd", float(getattr(ctx, "last_good_edge_yaw_cmd", 0.0) or 0.0)),
        last_good_edge_yaw_age_ms=_optional_float(summary.get("last_good_edge_yaw_age_ms")),
        depth_valid=depth_valid,
        table_roi_depth_p10=_optional_float(summary.get("table_roi_depth_p10", _obs_value(obs, "table_roi_depth_p10"))),
        table_roi_depth_median=_optional_float(summary.get("table_roi_depth_median", _obs_value(obs, "table_roi_depth_median"))),
        table_roi_depth_mean=_optional_float(summary.get("table_roi_depth_mean", _obs_value(obs, "table_roi_depth_mean"))),
        dist_err_m=_optional_float(summary.get("dist_err_m", _obs_value(obs, "dist_err_m"))),
        lateral_err_norm=_optional_float(summary.get("lateral_err_norm")),
        lateral_err_m=_optional_float(summary.get("lateral_err_m", _obs_value(obs, "lateral_err_m"))),
        lateral_source=str(summary.get("lateral_source") or ("edge_lateral_err_m" if _obs_value(obs, "lateral_err_m") is not None else "")),
        final_depth_latched=bool(summary.get("final_depth_latched") or getattr(ctx, "final_depth_latched", False)),
        near_table_latched=bool(summary.get("near_table_latched") or getattr(ctx, "near_table_latched", False)),
        final_yaw_align_active=bool(summary.get("final_yaw_align_active") or getattr(ctx, "final_yaw_align_active", False)),
        final_locked=bool(summary.get("final_locked") or getattr(ctx, "final_locked", False)),
        stale_level=stale_level,
        stale_policy=stale_policy,
        last_good_obs_age_ms=_optional_float(summary.get("last_good_table_obs_age_ms", summary.get("last_good_obs_age_ms"))),
        perception_dropout_hold_active=bool(summary.get("perception_dropout_hold_active") or getattr(ctx, "perception_dropout_hold_active", False)),
        approach_commit_active=bool(summary.get("approach_commit_active") or getattr(ctx, "approach_commit_active", False)),
        zero_cmd_age_ms=_float(summary, "zero_cmd_age_ms", 0.0),
        explicit_stop=explicit_stop,
        obstacle_active=bool(summary.get("obstacle_active") or summary.get("obstacle_stop_active")),
        emergency_active=emergency_active,
        hard_safety_active=hard_safety_active,
        hardware_safety_failure=bool(summary.get("hardware_safety_failure", False)),
        raw_summary=summary,
    )


def update_docking_stage(ctx: Any, obs: DockingObservation, now: float = 0.0) -> DockingStage:
    """Return the semantic docking stage; it does not emit velocity."""
    del now
    if obs.explicit_stop or obs.emergency_active:
        return DockingStage.EMERGENCY_STOP
    if obs.hard_safety_active or obs.hardware_safety_failure:
        return DockingStage.SAFETY_STOP
    if obs.final_depth_latched:
        if obs.final_locked:
            return DockingStage.FINAL_LOCKED
        yaw_deadband = _float(obs.raw_summary, "final_yaw_deadband_rad", _float(obs.raw_summary, "table_yaw_tol_rad", 0.08))
        yaw_large = bool(obs.final_yaw_align_active or (obs.yaw_err_rad is not None and abs(float(obs.yaw_err_rad)) > abs(yaw_deadband)))
        yaw_cmd = obs.edge_yaw_cmd or obs.last_good_edge_yaw_cmd
        if yaw_large and abs(float(yaw_cmd)) > 1e-9:
            return DockingStage.FINAL_YAW_ALIGN
        return DockingStage.FINAL_DISTANCE_HOLD
    if obs.near_table_latched:
        return DockingStage.NEAR_EDGE_APPROACH
    if obs.perception_dropout_hold_active:
        return DockingStage.RECOVERY_ROTATE
    phase = str(obs.raw_summary.get("control_phase") or "").strip().upper()
    state = _state_value(ctx)
    if phase == "EDGE_GUIDED_APPROACH":
        return DockingStage.EDGE_APPROACH
    if phase == "EDGE_HANDOFF_CONFIRM":
        return DockingStage.EDGE_HANDOFF
    if phase == "BBOX_ACQUIRE":
        return DockingStage.BBOX_ACQUIRE
    if obs.edge_trusted or obs.edge_usable:
        return DockingStage.EDGE_APPROACH
    if obs.bbox_valid:
        return DockingStage.BBOX_ACQUIRE
    if state in {"SEARCH_TABLE", "YOLO_ACQUIRE_ALIGN", "YOLO_APPROACH", "EDGE_ADJUST"}:
        return DockingStage.SEARCH
    return DockingStage.SEARCH


@dataclass(frozen=True)
class MotionIntent:
    intent_type: str
    desired_vx: float = 0.0
    desired_vy: float = 0.0
    desired_wz: float = 0.0
    yaw_owner: str = ""
    forward_allowed_by_behavior: bool = False
    rotate_allowed_by_behavior: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DockingMotionResult:
    action: DockingAction
    stage: DockingStage
    vx: float
    vy: float
    wz: float
    yaw_owner: str
    stop_class: StopClass
    safety_class: str
    blocked_by: str
    reason: str
    summary: Dict[str, Any] = field(default_factory=dict)
    forward_owner: str = ""
    lateral_owner: str = ""
    advance_condition: str = ""
    fallback_condition: str = ""
    allow_uart_send: bool = True
    service_may_override: bool = False
    overridden_by: List[DockingAction] = field(default_factory=list)
    not_overridden_by: List[str] = field(default_factory=list)

    @property
    def final_vx(self) -> float:
        return float(self.vx)

    @property
    def final_vy(self) -> float:
        return float(self.vy)

    @property
    def final_wz(self) -> float:
        return float(self.wz)

    @property
    def motion_class(self) -> str:
        if self.action == DockingAction.EMERGENCY_STOP:
            return "emergency_stop"
        if self.action == DockingAction.SAFETY_STOP:
            return "safety_stop"
        if self.stop_class in {StopClass.CONTROL_RECOVERY, StopClass.STALE_RECOVERY}:
            return "recovery"
        if abs(self.vx) > 1e-9 or abs(self.vy) > 1e-9 or abs(self.wz) > 1e-9:
            return "normal"
        return "control_stop"

    def _owner_defaults(self) -> tuple[str, str, str]:
        yaw = str(self.yaw_owner or "")
        mapping = {
            DockingAction.SEARCH_ROTATE: ("search", "none", "none"),
            DockingAction.BBOX_REACQUIRE_ROTATE: ("bbox", "none", "none"),
            DockingAction.BBOX_TRACK_FORWARD: ("bbox", "bbox_track", "none"),
            DockingAction.EDGE_READINESS_HANDOFF: ("edge_candidate", "none", "none"),
            DockingAction.EDGE_APPROACH_FORWARD: ("edge", "edge_approach", "none"),
            DockingAction.NEAR_EDGE_FORWARD: (yaw or "last_good_edge", "near_edge", "none"),
            DockingAction.NEAR_EDGE_LATERAL_ALIGN: (yaw or "edge", "none", "none"),
            DockingAction.PERCEPTION_DROPOUT_HOLD: (yaw or "edge_hold", "approach_commit", "none"),
            DockingAction.FINAL_YAW_ALIGN: (yaw or "last_good_edge", "none", "none"),
            DockingAction.FINAL_LOCKED_STOP: ("none", "none", "none"),
            DockingAction.CLOSE_RANGE_PROBE: ("none", "close_range_probe", "none"),
            DockingAction.FINAL_SLOW_PROBE: ("none", "final_edge_servo", "none"),
            DockingAction.CONTROL_RECOVERY_ROTATE: (yaw or "bbox", "none", "none"),
            DockingAction.SAFETY_STOP: ("none", "none", "none"),
            DockingAction.EMERGENCY_STOP: ("none", "none", "none"),
        }
        return mapping.get(self.action, (yaw, "none", "none"))

    def legacy_summary(self) -> Dict[str, Any]:
        out = dict(self.summary or {})
        default_yaw, default_forward, default_lateral = self._owner_defaults()
        yaw_owner = "none" if self.action == DockingAction.FINAL_LOCKED_STOP else str(self.yaw_owner or default_yaw or "none")
        forward_owner = str(self.forward_owner or out.get("forward_owner") or default_forward or "none")
        lateral_owner = str(self.lateral_owner or out.get("lateral_owner") or default_lateral or "none")
        out.update(
            {
                "docking_action": self.action.value,
                "docking_stage": self.stage.value,
                "docking_reason": str(self.reason or ""),
                "motion_intent_type": self.action.value.lower(),
                "yaw_owner": yaw_owner,
                "forward_owner": forward_owner,
                "lateral_owner": lateral_owner,
                "lateral_err_norm": out.get("lateral_err_norm"),
                "lateral_err_m": out.get("lateral_err_m"),
                "lateral_source": str(out.get("lateral_source") or ""),
                "vy_enabled": bool(out.get("vy_enabled", False)),
                "vy_block_reason": str(out.get("vy_block_reason") or "lateral_disabled"),
                "vy_cmd_raw": float(out.get("vy_cmd_raw", self.vy) or 0.0),
                "vy_cmd_limited": 0.0,
                "advance_condition": str(self.advance_condition or out.get("advance_condition") or ""),
                "fallback_condition": str(self.fallback_condition or out.get("fallback_condition") or ""),
                "arbitration_reason": str(self.reason or ""),
                "motion_class": self.motion_class,
                "stop_class": self.stop_class.value,
                "safety_class": str(self.safety_class or ""),
                "blocked_by": str(self.blocked_by or ""),
                "final_vx": float(self.vx),
                "final_vy": float(self.vy),
                "final_wz": float(self.wz),
                "vx_mps": float(self.vx),
                "vy_mps": float(self.vy),
                "wz_radps": float(self.wz),
                "allow_uart_send": bool(self.allow_uart_send),
                "service_may_override": bool(self.service_may_override),
            }
        )
        return out

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["action"] = self.action.value
        d["stage"] = self.stage.value
        d["stop_class"] = self.stop_class.value
        d["overridden_by"] = [item.value for item in self.overridden_by]
        return d
