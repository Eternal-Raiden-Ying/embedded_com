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


def get_fresh_table_obs(ctx: Any, cfg: Any) -> Any:
    obs = ctx.last_table_obs
    if obs is None or time.time() - obs.ts > cfg.table_obs_max_age_s:
        return None
    if ctx.task_start_wall_ts > 0 and obs.ts < ctx.task_start_wall_ts:
        return None
    if obs.session_id and ctx.active_session_id and obs.session_id != ctx.active_session_id:
        return None
    return obs


def apply_base_motion_safety(decision: Any, *, ctx: Any, cfg: Any, log_fn: Optional[Any] = None) -> Any:
    summary = decision.control_summary
    if summary is None:
        summary = {}
        decision.control_summary = summary

    state = ctx.state
    base_motion_allowed_states = {
        State.SEARCH_TABLE,
        State.YOLO_ACQUIRE_ALIGN,
        State.YOLO_APPROACH,
        State.EDGE_ADJUST,
        State.FINAL_SLOW_STOP,
        State.NO_PROGRESS_RECOVERY,
        State.EDGE_SLIDE_SEARCH,
        State.LEAVE_EDGE,
        State.RELOCATE_TO_EDGE,
        State.REACQUIRE_TABLE,
        State.NEXT_TABLE,
        State.RETURN_HOME,
        State.AVOID_OBSTACLE,
    }

    state_allows_motion = False
    if state in base_motion_allowed_states:
        state_allows_motion = True
    elif state == State.GRASP:
        if str(ctx.grasp_substate).upper() == "REPOSITIONING":
            state_allows_motion = True

    if not state_allows_motion:
        state_name = getattr(state, "value", str(state))
        block_reason = f"state_{state_name}_disallowed"
        decision.cmd.vx_mps = 0.0
        decision.cmd.vy_mps = 0.0
        decision.cmd.wz_radps = 0.0
        summary["allow_forward"] = False
        summary["allow_rotate"] = False
        summary["allow_lateral"] = False
        summary["forward_block_reason"] = block_reason
        summary["rotate_block_reason"] = block_reason
        summary["lateral_block_reason"] = block_reason
        summary["stop_reason"] = block_reason
        return decision

    allow_forward = summary.get("allow_forward", True)
    allow_rotate = summary.get("allow_rotate", True)
    allow_lateral = summary.get("allow_lateral", True)

    if not allow_forward and decision.cmd.vx_mps > 1e-9:
        decision.cmd.vx_mps = 0.0
        if not summary.get("forward_block_reason"):
            summary["forward_block_reason"] = "allow_forward_false_intercept"

    if not allow_rotate and abs(decision.cmd.wz_radps) > 1e-9:
        decision.cmd.wz_radps = 0.0
        if not summary.get("rotate_block_reason"):
            summary["rotate_block_reason"] = "allow_rotate_false_intercept"

    if not allow_lateral and abs(decision.cmd.vy_mps) > 1e-9:
        decision.cmd.vy_mps = 0.0
        if not summary.get("lateral_block_reason"):
            summary["lateral_block_reason"] = "allow_lateral_false_intercept"

    obs = get_fresh_table_obs(ctx, cfg)
    if obs is not None and obs.depth_p10 is not None:
        depth_p10 = obs.depth_p10
        near_stop_depth_m = getattr(cfg, "near_stop_depth_m", 0.25)
        if depth_p10 < near_stop_depth_m:
            if decision.cmd.vx_mps != 0.0 or decision.cmd.vy_mps != 0.0 or decision.cmd.wz_radps != 0.0:
                decision.cmd.vx_mps = 0.0
                decision.cmd.vy_mps = 0.0
                decision.cmd.wz_radps = 0.0
                summary["forward_block_reason"] = "depth_p10_too_close"
                summary["rotate_block_reason"] = "depth_p10_too_close"
                summary["lateral_block_reason"] = "depth_p10_too_close"
                summary["stop_reason"] = "depth_p10_too_close"
                if log_fn:
                    try:
                        log_fn("warn", f"Depth safety stop triggered: depth_p10={depth_p10:.3f}m < {near_stop_depth_m:.3f}m")
                    except Exception:
                        pass
            allow_forward = False
            allow_rotate = False
            allow_lateral = False

        elif depth_p10 < getattr(cfg, "near_slow_depth_m", 0.40):
            max_vx = getattr(cfg, "near_slow_max_vx_mps", 0.020)
            max_wz = getattr(cfg, "near_slow_max_wz_radps", 0.04)
            max_vy = getattr(cfg, "near_slow_max_vy_mps", 0.0)

            if decision.cmd.vx_mps > max_vx:
                decision.cmd.vx_mps = max_vx
                summary["forward_block_reason"] = "depth_p10_slowdown"
            if abs(decision.cmd.wz_radps) > max_wz:
                sign = 1.0 if decision.cmd.wz_radps >= 0 else -1.0
                decision.cmd.wz_radps = sign * max_wz
                summary["rotate_block_reason"] = "depth_p10_slowdown"
            if abs(decision.cmd.vy_mps) > max_vy:
                sign = 1.0 if decision.cmd.vy_mps >= 0 else -1.0
                decision.cmd.vy_mps = sign * max_vy
                summary["lateral_block_reason"] = "depth_p10_slowdown"

            if log_fn:
                try:
                    log_fn("info", f"Depth safety slowdown active: depth_p10={depth_p10:.3f}m. Limited to vx={decision.cmd.vx_mps:.3f}, vy={decision.cmd.vy_mps:.3f}, wz={decision.cmd.wz_radps:.3f}")
                except Exception:
                    pass

    summary["allow_forward"] = bool(allow_forward)
    summary["allow_rotate"] = bool(allow_rotate)
    summary["allow_lateral"] = bool(allow_lateral)

    summary["forward_block_reason"] = summary.get("forward_block_reason") or ""
    summary["rotate_block_reason"] = summary.get("rotate_block_reason") or ""
    summary["lateral_block_reason"] = summary.get("lateral_block_reason") or ""
    summary["stop_reason"] = summary.get("stop_reason") or ""

    if decision.cmd.vx_mps <= 1e-9 and abs(decision.cmd.wz_radps) <= 1e-9 and abs(decision.cmd.vy_mps) <= 1e-9:
        if not summary.get("stop_reason"):
            summary["stop_reason"] = summary.get("forward_block_reason") or summary.get("rotate_block_reason") or summary.get("lateral_block_reason") or "stopped"

    return decision


class BaseMotionSafetyMixin:
    def _extract_obstacle_signal(self) -> ObstacleSignal:
        table_obs = self._fresh_table_obs()
        if table_obs is not None and bool(table_obs.obstacle_flag):
            return ObstacleSignal(
                active=True,
                best_turn_dir=str(table_obs.best_turn_dir or ""),
                distance_m=table_obs.obstacle_distance_m,
                source="table_edge_obs",
            )
        target_obs = self._fresh_target_obs()
        if target_obs is not None and bool(target_obs.obstacle_flag):
            return ObstacleSignal(
                active=True,
                best_turn_dir=str(target_obs.best_turn_dir or ""),
                distance_m=target_obs.obstacle_distance_m,
                source="target_obs",
            )
        home_obs = self._fresh_home_obs()
        if home_obs is not None and bool(home_obs.obstacle_flag):
            return ObstacleSignal(
                active=True,
                best_turn_dir=str(home_obs.best_turn_dir or ""),
                distance_m=home_obs.obstacle_distance_m,
                source="home_tag_obs",
            )
        return ObstacleSignal(active=False)

    def _check_safety_interlock(self) -> Optional[MotionDecision]:
        obstacle = self._extract_obstacle_signal()
        if self.ctx.state == State.AVOID_OBSTACLE:
            return None
        if self.ctx.state in MOVING_STATES and obstacle.active:
            self.ctx.last_safety_reason = f"检测到障碍({obstacle.source})"
            self.ctx.resume_state = self.ctx.state
            self.ctx.avoid_retry_count += 1
            self._transition(State.AVOID_OBSTACLE, self.ctx.last_safety_reason)
            self._queue_tts("前方有障碍，开始避障")
            return self.controller.stop_cmd("AVOID_OBSTACLE", brake=True)
        return None

    def _apply_soft_interception_and_safety(self, decision: MotionDecision) -> MotionDecision:
        def log_fn(level: str, msg: str, *args: Any) -> None:
            self._log(level, msg, *args)

        return apply_base_motion_safety(decision, ctx=self.ctx, cfg=self.cfg, log_fn=log_fn)

