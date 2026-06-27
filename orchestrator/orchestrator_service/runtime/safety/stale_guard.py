#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ...config.schema import CarMotionConfig, ControlThresholds
from ...control.types import DockingControlConfig
from ...ipc.protocol import (
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
from ...bridge.arm_protocol import parse_arm_response
from ...utils.grasp_utils import grasp_to_pose_params
from ...utils.target_utils import target_to_class_id
from ..common import monotonic_ts
from ..context import RuntimeContext, State
from ..controller import MotionController, MotionDecision
from ..control_authority import decide_table_control_authority
from ..core_types import (
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


class StaleGuardMixin:
    def _table_obs_age_ms(self, obs: Optional[TableEdgeObs]) -> Optional[float]:
        if obs is None:
            return None
        age_ms: Optional[float] = None
        obs_ts = getattr(obs, "frame_capture_ts", None)
        if obs_ts is None:
            obs_ts = obs.obs_ts if obs.obs_ts is not None else obs.ts
        try:
            age_ms = max(0.0, (time.time() - float(obs_ts)) * 1000.0)
        except Exception:
            age_ms = None
        for candidate in (getattr(obs, "obs_total_age_ms", None), obs.age_ms):
            if candidate is None:
                continue
            try:
                age_ms = max(float(age_ms or 0.0), float(candidate))
            except Exception:
                pass
        return age_ms

    def _table_control_loop_age_ms(self, obs: Optional[TableEdgeObs]) -> Optional[float]:
        if obs is None or getattr(obs, "obs_recv_ts", None) is None:
            return None
        try:
            return max(0.0, (time.time() - float(obs.obs_recv_ts)) * 1000.0)
        except Exception:
            return None

    def _table_obs_stale_level(self, obs: Optional[TableEdgeObs]) -> str:
        if obs is None:
            return "dead"
        age_ms = self._table_obs_age_ms(obs)
        if age_ms is None or bool(getattr(obs, "is_stale", False)) or obs.depth_valid is False:
            return "hard_stale"
        soft = float(getattr(self.cfg, "table_obs_stale_soft_ms", 300) or 300)
        stop = float(getattr(self.cfg, "table_obs_stale_stop_ms", 500) or 500)
        hard = float(getattr(self.cfg, "table_obs_stale_hard_ms", 800) or 800)
        if float(age_ms) <= soft:
            return "fresh"
        if float(age_ms) <= stop:
            return "soft_stale"
        if float(age_ms) <= hard:
            return "hard_stale"
        return "dead"

    def _table_obs_stale_reason(self, obs: Optional[TableEdgeObs]) -> str:
        level = self._table_obs_stale_level(obs)
        if level == "fresh":
            return ""
        age_ms = self._table_obs_age_ms(obs)
        age_text = "unknown" if age_ms is None else f"{float(age_ms):.0f}"
        return f"{level}:obs_total_age_ms={age_text}"

    def _vision_stale_reason(self, obs: Optional[TableEdgeObs]) -> str:
        if obs is None:
            return "no_recent_obs"
        if bool(getattr(obs, "is_stale", False)) or obs.depth_valid is False:
            return "obs_invalid"
        if bool(getattr(obs, "fast_temporal_jump", False)):
            return "temporal_jump"
        reject_reason = str(getattr(obs, "control_reject_reason", "") or getattr(obs, "reject_reason", "") or getattr(obs, "reason", "") or "").strip().lower()
        if reject_reason in {"temporal_jump", "yaw_jump", "dist_jump"}:
            return reject_reason
        if "yaw_jump" in reject_reason:
            return "yaw_jump"
        if "dist_jump" in reject_reason or "far_jump" in reject_reason:
            return "dist_jump"
        age_ms = self._table_obs_age_ms(obs)
        if age_ms is None:
            return "age_over_limit"
        stale_level = self._table_obs_stale_level(obs)
        if stale_level == "soft_stale":
            return "soft_stale"
        if stale_level in {"hard_stale", "dead"}:
            return "hard_stale"
        if reject_reason and reject_reason.lower() not in {"none", "ok", "valid"}:
            return "reject_reason"
        control_level = self._control_level(obs)
        if control_level in {"none", ""} or not bool(getattr(obs, "usable_for_approach", False) or getattr(obs, "usable_for_alignment", False) or getattr(obs, "usable_for_stop", False)):
            return "control_level_not_usable"
        return "unknown"

    def _edge_obs_is_stale(self, obs: Optional[TableEdgeObs]) -> bool:
        if obs is None:
            return True
        if bool(getattr(obs, "is_stale", False)):
            return True
        if obs.depth_valid is False:
            return True
        age_ms = self._table_obs_age_ms(obs)
        if age_ms is None:
            return True
        return self._table_obs_stale_level(obs) != "fresh"

