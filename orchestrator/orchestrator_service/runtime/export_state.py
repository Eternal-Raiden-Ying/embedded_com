#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config.schema import CarMotionConfig, ControlThresholds
from ..control.types import DockingControlConfig
from ..ipc.protocol import (
    ArmCommand,
    ArmResponse,
    CarState,
    HomeTagObs,
    TableEdgeObs,
    TargetObs,
    TaskCmd,
    make_grasp_req,
    make_tts_event,
    make_vision_idle,
    make_vision_req,
)
from ..bridge.arm_protocol import parse_arm_response
from ..utils.grasp_utils import grasp_to_pose_params
from ..utils.target_utils import target_to_class_id
from .common import monotonic_ts
from .context import RuntimeContext, State
from .controller import MotionController, MotionDecision
from .control_authority import decide_table_control_authority
from .core_types import (
    KNOWN_VISION_STATUS,
    MOVING_STATES,
    TABLE_APPROACH_STATES,
    TABLE_VISION_STATES,
    TARGET_SEARCH_STATES,
    TARGET_VISION_STATES,
    ObstacleSignal,
    VisionStageBinding,
    _GRASP_ARM_TIMEOUT_S,
    _GRASP_REPOSITION_TIMEOUT_S,
    _GRASP_RESPOND_TIMEOUT_S,
    _GRASP_RESULT_TIMEOUT_S,
    _GRASP_RETRY_LIMIT,
)


class ExportStateMixin:
    def export_state_block(self) -> Dict:
        table_obs = self.ctx.last_table_obs
        from .perception_semantics import build_table_perception_semantics
        sem = build_table_perception_semantics(table_obs, self.cfg)

        decision = getattr(self, "last_decision", None)
        summary = decision.control_summary if decision is not None else None
        if summary is None:
            summary = {}

        return {
            "ts": time.time(),
            "state": self.ctx.state.value,
            "control_source": summary.get("control_source") or "stop",
            "control_intent": summary.get("control_intent") or "stop",
            "table_bbox_current_found": bool(sem.table_bbox_current_found),
            "table_bbox_control_valid": bool(sem.table_bbox_control_valid),
            "edge_geometry_valid": bool(sem.edge_geometry_valid),
            "edge_trusted": bool(sem.edge_trusted),
            "allow_forward": bool(summary.get("allow_forward", False)),
            "allow_rotate": bool(summary.get("allow_rotate", False)),
            "allow_lateral": bool(summary.get("allow_lateral", False)),
            "forward_block_reason": str(summary.get("forward_block_reason") or ""),
            "rotate_block_reason": str(summary.get("rotate_block_reason") or ""),
            "lateral_block_reason": str(summary.get("lateral_block_reason") or ""),
            "stop_reason": str(summary.get("stop_reason") or ""),
            "desired_vision_stage": self.ctx.desired_vision_stage,
            "desired_vision_mode": self.ctx.desired_vision_mode,
            "confirmed_vision_stage": self.ctx.confirmed_vision_stage,
            "confirmed_vision_mode": self.ctx.confirmed_vision_mode,
            "vision_stage": self.ctx.confirmed_vision_stage,
            "vision_mode": self.ctx.confirmed_vision_mode,
            # Backwards compatibility fields for unit tests
            "edge_valid": bool(sem.edge_geometry_valid),
            "confidence": float(table_obs.confidence) if table_obs is not None else None,
            "yaw_err_rad": float(table_obs.yaw_err_rad) if (table_obs and table_obs.yaw_err_rad is not None) else None,
            "lock_ready": bool(getattr(table_obs, "edge_ready", False)) if table_obs is not None else False,
        }

    def _float_or_none(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _target_center(self, obs: Optional[TargetObs]) -> Optional[Dict[str, Optional[float]]]:
        if obs is None:
            return None
        full = self._target_center_full_norm(obs)
        offset = self._float_or_none(obs.cx_norm)
        if full is not None and full.get("cx") is not None:
            return full
        cy = self._float_or_none(obs.cy_norm)
        if full is not None and cy is None:
            cy = self._float_or_none(full.get("cy"))
        if offset is None and cy is None:
            return None
        cx = None
        if offset is not None:
            cx = max(0.0, min(1.0, 0.5 - (float(offset) / 2.0)))
        return {
            "cx": cx,
            "cy": cy,
        }

    def _target_center_full_norm(self, obs: Optional[TargetObs]) -> Optional[Dict[str, Optional[float]]]:
        if obs is None:
            return None
        source = obs.matched_center_full_norm if isinstance(obs.matched_center_full_norm, dict) else None
        if source is None and isinstance(obs.matched_center, dict):
            source = obs.matched_center
        cx = None
        cy = None
        if isinstance(source, dict):
            cx = self._float_or_none(source.get("cx", source.get("x_norm", source.get("cx_norm"))))
            cy = self._float_or_none(source.get("cy", source.get("y_norm", source.get("cy_norm"))))
            if cx is not None and not (0.0 <= float(cx) <= 1.0):
                cx = None
            if cy is not None and not (0.0 <= float(cy) <= 1.0):
                cy = None
        if cx is None:
            cx = self._float_or_none(getattr(obs, "x_norm", None))
        if cy is None:
            cy = self._float_or_none(getattr(obs, "y_norm", None))
        if cx is None and cy is None:
            return None
        if cx is not None:
            cx = max(0.0, min(1.0, float(cx)))
        if cy is not None:
            cy = max(0.0, min(1.0, float(cy)))
        return {"cx": cx, "cy": cy}

    def _target_center_offset_norm(self, obs: Optional[TargetObs]) -> Optional[Dict[str, Optional[float]]]:
        if obs is None:
            return None
        source = obs.matched_center_offset_norm if isinstance(obs.matched_center_offset_norm, dict) else None
        dx = None
        dy = None
        if isinstance(source, dict):
            dx = self._float_or_none(source.get("dx"))
            dy = self._float_or_none(source.get("dy"))
        if dx is None:
            dx = self._float_or_none(obs.cx_norm)
        if dy is None and self._target_center_full_norm(obs) is not None:
            full = self._target_center_full_norm(obs) or {}
            cy = self._float_or_none(full.get("cy"))
            if cy is not None:
                dy = max(-1.0, min(1.0, 1.0 - (2.0 * float(cy))))
        if dx is None and dy is None:
            return None
        if dx is not None:
            dx = max(-1.0, min(1.0, float(dx)))
        if dy is not None:
            dy = max(-1.0, min(1.0, float(dy)))
        return {
            "dx": dx,
            "dy": dy,
        }

