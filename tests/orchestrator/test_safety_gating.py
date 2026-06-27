#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORCH_ROOT = ROOT / "orchestrator"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

import pytest
from orchestrator_service.ipc.protocol import CmdVel, TableEdgeObs
from orchestrator_service.runtime.context import RuntimeContext, State
from orchestrator_service.config.schema import ControlThresholds
from orchestrator_service.control.motion_controller import MotionDecision
from orchestrator_service.runtime.safety.base_motion_safety import apply_base_motion_safety


def test_forward_too_close():
    ctx = RuntimeContext()
    ctx.state = State.EDGE_ADJUST
    cfg = ControlThresholds()
    cfg.near_stop_depth_m = 0.25

    # Mock obs that is too close (depth_p10 = 0.15 < 0.25)
    obs = TableEdgeObs.from_dict({
        "ts": time.time(),
        "table_found": True,
        "edge_found": True,
        "depth_p10": 0.15,
        "yaw_err_rad": 0.0,
    })
    ctx.last_table_obs = obs

    # Decision with positive forward velocity
    cmd = CmdVel(ts=time.time(), mode="SEARCH", vx_mps=0.1, vy_mps=0.0, wz_radps=0.05)
    decision = MotionDecision(cmd=cmd, control_summary={})

    decision = apply_base_motion_safety(decision, ctx=ctx, cfg=cfg)

    # Both vx and wz must be blocked
    assert decision.cmd.vx_mps == 0.0
    assert decision.cmd.wz_radps == 0.0
    assert decision.control_summary["allow_forward"] is False
    assert decision.control_summary["allow_rotate"] is False
    assert decision.control_summary["allow_lateral"] is False
    assert decision.control_summary["forward_block_reason"] == "depth_p10_too_close"
    assert decision.control_summary["rotate_block_reason"] == "depth_p10_too_close"
    assert decision.control_summary["lateral_block_reason"] == "depth_p10_too_close"


def test_rotate_too_close():
    ctx = RuntimeContext()
    ctx.state = State.EDGE_ADJUST
    cfg = ControlThresholds()
    cfg.near_stop_depth_m = 0.25

    obs = TableEdgeObs.from_dict({
        "ts": time.time(),
        "table_found": True,
        "edge_found": True,
        "depth_p10": 0.15,
        "yaw_err_rad": 0.0,
    })
    ctx.last_table_obs = obs

    # Decision with rotation but no forward
    cmd = CmdVel(ts=time.time(), mode="SEARCH", vx_mps=0.0, vy_mps=0.0, wz_radps=0.08)
    decision = MotionDecision(cmd=cmd, control_summary={})

    decision = apply_base_motion_safety(decision, ctx=ctx, cfg=cfg)

    assert decision.cmd.vx_mps == 0.0
    assert decision.cmd.wz_radps == 0.0
    assert decision.control_summary["allow_forward"] is False
    assert decision.control_summary["allow_rotate"] is False
    assert decision.control_summary["allow_lateral"] is False
    assert decision.control_summary["rotate_block_reason"] == "depth_p10_too_close"


def test_lateral_too_close():
    ctx = RuntimeContext()
    ctx.state = State.EDGE_ADJUST
    cfg = ControlThresholds()
    cfg.near_stop_depth_m = 0.25

    obs = TableEdgeObs.from_dict({
        "ts": time.time(),
        "table_found": True,
        "edge_found": True,
        "depth_p10": 0.15,
        "yaw_err_rad": 0.0,
    })
    ctx.last_table_obs = obs

    # Decision with lateral speed but too close -> any vy is blocked
    cmd = CmdVel(ts=time.time(), mode="SEARCH", vx_mps=0.0, vy_mps=0.1, wz_radps=0.0)
    decision = MotionDecision(cmd=cmd, control_summary={})

    decision = apply_base_motion_safety(decision, ctx=ctx, cfg=cfg)

    assert decision.cmd.vy_mps == 0.0
    assert decision.control_summary["allow_lateral"] is False
    assert decision.control_summary["lateral_block_reason"] == "depth_p10_too_close"


def test_slowdown_zone():
    ctx = RuntimeContext()
    ctx.state = State.EDGE_ADJUST
    cfg = ControlThresholds()
    cfg.near_slow_depth_m = 0.40
    cfg.near_slow_max_vx_mps = 0.010
    cfg.near_slow_max_wz_radps = 0.04
    cfg.near_slow_max_vy_mps = 0.010

    # depth is in slow range (0.30 is between 0.25 and 0.40)
    obs = TableEdgeObs.from_dict({
        "ts": time.time(),
        "table_found": True,
        "edge_found": True,
        "depth_p10": 0.30,
        "yaw_err_rad": 0.0,
    })
    ctx.last_table_obs = obs

    # Velocities exceeding slow limits
    cmd = CmdVel(ts=time.time(), mode="SEARCH", vx_mps=0.05, vy_mps=-0.03, wz_radps=0.1)
    decision = MotionDecision(cmd=cmd, control_summary={})

    decision = apply_base_motion_safety(decision, ctx=ctx, cfg=cfg)

    # Clamped to configured maxima
    assert decision.cmd.vx_mps == 0.010
    assert decision.cmd.vy_mps == -0.010
    assert decision.cmd.wz_radps == 0.04
    assert decision.control_summary["forward_block_reason"] == "depth_p10_slowdown"
    assert decision.control_summary["lateral_block_reason"] == "depth_p10_slowdown"
    assert decision.control_summary["rotate_block_reason"] == "depth_p10_slowdown"


def test_summary_flags_not_overwritten():
    ctx = RuntimeContext()
    ctx.state = State.EDGE_ADJUST
    cfg = ControlThresholds()
    cfg.near_stop_depth_m = 0.25

    obs = TableEdgeObs.from_dict({
        "ts": time.time(),
        "table_found": True,
        "edge_found": True,
        "depth_p10": 0.15,
    })
    ctx.last_table_obs = obs

    cmd = CmdVel(ts=time.time(), mode="SEARCH", vx_mps=0.1, vy_mps=0.1, wz_radps=0.1)
    # The original summary had allow flags as True
    decision = MotionDecision(cmd=cmd, control_summary={
        "allow_forward": True,
        "allow_rotate": True,
        "allow_lateral": True
    })

    decision = apply_base_motion_safety(decision, ctx=ctx, cfg=cfg)

    # Verify they were set to False and NOT overwritten back to True
    assert decision.control_summary["allow_forward"] is False
    assert decision.control_summary["allow_rotate"] is False
    assert decision.control_summary["allow_lateral"] is False


def test_grasp_non_reposition_must_stop():
    ctx = RuntimeContext()
    ctx.state = State.GRASP
    ctx.grasp_substate = "AWAITING_RESULT"
    cfg = ControlThresholds()

    cmd = CmdVel(ts=time.time(), mode="GRASP", vx_mps=0.05, vy_mps=0.05, wz_radps=0.05)
    decision = MotionDecision(cmd=cmd, control_summary={})

    decision = apply_base_motion_safety(decision, ctx=ctx, cfg=cfg)

    # Must be completely stopped
    assert decision.cmd.vx_mps == 0.0
    assert decision.cmd.vy_mps == 0.0
    assert decision.cmd.wz_radps == 0.0
    assert decision.control_summary["allow_forward"] is False
    assert decision.control_summary["allow_rotate"] is False
    assert decision.control_summary["allow_lateral"] is False
    assert decision.control_summary["forward_block_reason"] == "state_GRASP_disallowed"


def test_grasp_repositioning_allowed():
    ctx = RuntimeContext()
    ctx.state = State.GRASP
    ctx.grasp_substate = "REPOSITIONING"
    cfg = ControlThresholds()

    cmd = CmdVel(ts=time.time(), mode="GRASP", vx_mps=0.05, vy_mps=0.05, wz_radps=0.0)
    decision = MotionDecision(cmd=cmd, control_summary={})

    decision = apply_base_motion_safety(decision, ctx=ctx, cfg=cfg)

    # Velocities are allowed to pass through
    assert decision.cmd.vx_mps == 0.05
    assert decision.cmd.vy_mps == 0.05
    assert decision.control_summary["allow_forward"] is True
    assert decision.control_summary["allow_lateral"] is True


def test_avoid_obstacle_remains_allowed():
    ctx = RuntimeContext()
    ctx.state = State.AVOID_OBSTACLE
    cfg = ControlThresholds()

    cmd = CmdVel(ts=time.time(), mode="AVOID_OBSTACLE", vx_mps=0.1, vy_mps=0.0, wz_radps=0.3)
    decision = MotionDecision(cmd=cmd, control_summary={})

    decision = apply_base_motion_safety(decision, ctx=ctx, cfg=cfg)

    assert decision.cmd.vx_mps == 0.1
    assert decision.cmd.wz_radps == 0.3
    assert decision.control_summary["allow_forward"] is True
    assert decision.control_summary["allow_rotate"] is True
