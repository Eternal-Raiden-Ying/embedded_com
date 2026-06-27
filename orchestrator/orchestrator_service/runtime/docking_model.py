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
    FINAL_DISTANCE_HOLD = "FINAL_DISTANCE_HOLD"
    FINAL_YAW_ALIGN = "FINAL_YAW_ALIGN"
    FINAL_LOCKED = "FINAL_LOCKED"
    RECOVERY_ROTATE = "RECOVERY_ROTATE"
    SAFETY_STOP = "SAFETY_STOP"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class DockingAction(str, Enum):
    SEARCH_ROTATE = "SEARCH_ROTATE"
    BBOX_REACQUIRE_ROTATE = "BBOX_REACQUIRE_ROTATE"
    EDGE_APPROACH_FORWARD = "EDGE_APPROACH_FORWARD"
    NEAR_EDGE_FORWARD = "NEAR_EDGE_FORWARD"
    PERCEPTION_DROPOUT_HOLD = "PERCEPTION_DROPOUT_HOLD"
    FINAL_YAW_ALIGN = "FINAL_YAW_ALIGN"
    FINAL_LOCKED_STOP = "FINAL_LOCKED_STOP"
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

    def legacy_summary(self) -> Dict[str, Any]:
        out = dict(self.summary or {})
        out.update(
            {
                "docking_action": self.action.value,
                "docking_stage": self.stage.value,
                "motion_intent_type": self.action.value.lower(),
                "yaw_owner": str(self.yaw_owner or ""),
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
