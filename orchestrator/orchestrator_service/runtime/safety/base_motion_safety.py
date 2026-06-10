#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from typing import Any, Optional

from ..context import State


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

    # State whitelist check
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
        # GRASP is allowed only when grasp_substate is REPOSITIONING
        if str(ctx.grasp_substate).upper() == "REPOSITIONING":
            state_allows_motion = True

    if not state_allows_motion:
        # Block all base motion
        decision.cmd.vx_mps = 0.0
        decision.cmd.vy_mps = 0.0
        decision.cmd.wz_radps = 0.0
        summary["allow_forward"] = False
        summary["allow_rotate"] = False
        summary["allow_lateral"] = False
        summary["forward_block_reason"] = f"state_{state}_disallowed"
        summary["rotate_block_reason"] = f"state_{state}_disallowed"
        summary["lateral_block_reason"] = f"state_{state}_disallowed"
        summary["stop_reason"] = f"state_{state}_disallowed"
        return decision

    # Retrieve initial values of the allow flags
    allow_forward = summary.get("allow_forward", True)
    allow_rotate = summary.get("allow_rotate", True)
    allow_lateral = summary.get("allow_lateral", True)

    # Apply Soft Interception based on allow flags
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

    # Depth safety checks
    obs = get_fresh_table_obs(ctx, cfg)
    if obs is not None and obs.depth_p10 is not None:
        depth_p10 = obs.depth_p10
        if depth_p10 < cfg.near_stop_depth_m:
            # Block all non-zero velocities
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
                        log_fn("warn", f"Depth safety stop triggered: depth_p10={depth_p10:.3f}m < {cfg.near_stop_depth_m:.3f}m")
                    except Exception:
                        pass
            allow_forward = False
            allow_rotate = False
            allow_lateral = False

        elif depth_p10 < cfg.near_slow_depth_m:
            max_vx = cfg.near_slow_max_vx_mps
            max_wz = cfg.near_slow_max_wz_radps
            max_vy = getattr(cfg, "near_slow_max_vy_mps", cfg.near_slow_max_vx_mps)

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

    # Ensure flags are set in summary, preserving the False state if modified
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
