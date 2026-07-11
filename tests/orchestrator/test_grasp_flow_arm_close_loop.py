#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
ORCH_ROOT = ROOT / "orchestrator"
for path in (ROOT, ORCH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from orchestrator_service.control.motion_controller import MotionDecision
from orchestrator_service.ipc.protocol import CmdVel
from orchestrator_service.runtime.context import RuntimeContext
from orchestrator_service.runtime.states.grasp_flow import GraspFlowMixin


CANONICAL_GRASP = {
    "x_cm": 23.9389,
    "y_cm": 3.3307,
    "z_cm": 8.7589,
    "pitch_deg": 8.2860,
    "roll_deg": 67.3864,
    "gripper_width_cm": 10.0,
    "approach_depth_cm": 4.0,
}


class _StopController:
    def stop_cmd(self, mode):
        return MotionDecision(cmd=CmdVel(ts=0.0, mode=str(mode), brake=True))


class _GraspFlowHarness(GraspFlowMixin):
    def __init__(self):
        self.ctx = RuntimeContext()
        self.ctx.grasp_status = "RESULT_READY"
        self.ctx.grasp_result = dict(CANONICAL_GRASP)
        self.ctx.grasp_substate = "AWAITING_RESPOND"
        self.ctx.grasp_timeout_mono = 999999.0
        self.controller = _StopController()
        self.car_cfg = SimpleNamespace(pre_arm_stop_settle_ms=0, grasp_pose_time_ms=800)
        self.logs = []
        self.errors = []

    def _state_elapsed(self):
        return 1.0

    def _log(self, level, message):
        self.logs.append((level, message))

    def _enter_error_recovery(self, reason, *args, **kwargs):
        self.errors.append(reason)

    def _queue_vision_req(self, *args, **kwargs):
        raise AssertionError("ready grasp result should not request another remote respond")

    def _target_lateral_center_x(self, obs):
        if obs is None or not getattr(obs, "found", False):
            return None
        return getattr(obs, "cx", 0.5)

    def _target_lateral_error_x(self, obs):
        cx = self._target_lateral_center_x(obs)
        if cx is None:
            return None
        return cx - 0.5


def test_result_ready_in_awaiting_respond_moves_to_pre_arm_settle():
    flow = _GraspFlowHarness()

    flow._tick_grasp_awaiting_respond(100.0)

    assert flow.ctx.grasp_substate == "PRE_ARM_STOP_SETTLE"
    assert any("grasp_result_ready_consumed" in message for _, message in flow.logs)
    assert flow.errors == []


def test_pre_arm_settle_encodes_arm_command_after_ready_result():
    flow = _GraspFlowHarness()
    flow.ctx.grasp_substate = "PRE_ARM_STOP_SETTLE"
    flow.ctx.pre_arm_stop_settle_start_mono = 100.0

    decision = flow._tick_grasp_pre_arm_stop_settle(100.1)

    assert flow.ctx.grasp_substate == "AWAITING_ARM"
    assert decision.arm_cmd is not None
    assert round(decision.arm_cmd.x_cm) == 24
    assert round(decision.arm_cmd.y_cm) == 3
    assert round(decision.arm_cmd.z_cm) == 9
    assert round(decision.arm_cmd.pitch_deg) == 8
    assert round(decision.arm_cmd.roll_deg) == 67
    assert round(decision.arm_cmd.claw_deg) == 90
    assert decision.arm_cmd.time_ms == 800


def test_pre_arm_settle_missing_pose_field_enters_schema_error():
    flow = _GraspFlowHarness()
    flow.ctx.grasp_substate = "PRE_ARM_STOP_SETTLE"
    flow.ctx.pre_arm_stop_settle_start_mono = 100.0
    flow.ctx.grasp_result = dict(CANONICAL_GRASP)
    flow.ctx.grasp_result.pop("x_cm")

    decision = flow._tick_grasp_pre_arm_stop_settle(100.1)

    assert decision.arm_cmd is None
    assert flow.errors == ["grasp_pose_schema_invalid"]


def test_pre_arm_settle_overrides_gripper_width_from_lookup_table():
    flow = _GraspFlowHarness()
    flow.ctx.grasp_substate = "PRE_ARM_STOP_SETTLE"
    flow.ctx.pre_arm_stop_settle_start_mono = 100.0
    flow.ctx.active_target = "苹果"

    class ConfigMock:
        target_gripper_widths = {
            "apple": 50.0,
            "苹果": 50.0,
        }
    flow.cfg = ConfigMock()

    flow.ctx.grasp_result = dict(CANONICAL_GRASP)

    decision = flow._tick_grasp_pre_arm_stop_settle(100.1)

    assert flow.ctx.grasp_substate == "AWAITING_ARM"
    assert decision.arm_cmd is not None
    assert round(decision.arm_cmd.claw_deg) == 50


def test_fallback_grasp_triggered_if_config_enabled_and_centered():
    flow = _GraspFlowHarness()
    flow.ctx.canonical_target = "bottle"
    flow.ctx.active_target = "bottle"
    
    # Mock last_target_obs to be centered (cx = 0.5, error_x = 0.0)
    class ObsMock:
        found = True
        cx = 0.5
    flow.ctx.last_target_obs = ObsMock()

    # Mock POSE_BOTTLE configuration to enable fallback grasp
    class PoseBottleConfigMock:
        fall_back_grasp = True
        x_center_tolerance = 0.06

    class ConfigMock:
        POSE_BOTTLE = PoseBottleConfigMock()
        target_gripper_widths = {}
    
    # Patch get_config
    import common.config_loader as cl
    orig_get_config = cl.get_config
    cl.get_config = lambda: ConfigMock()

    try:
        decision = flow._tick_grasp_awaiting_respond(100.0)
        assert flow.ctx.use_fallback_grasp is True
        assert flow.ctx.grasp_substate == "PRE_ARM_STOP_SETTLE"
    finally:
        cl.get_config = orig_get_config


def test_fallback_grasp_not_triggered_if_not_centered():
    flow = _GraspFlowHarness()
    flow.ctx.canonical_target = "bottle"
    flow.ctx.active_target = "bottle"
    
    # Mock last_target_obs to be off-center (cx = 0.8, error_x = 0.3)
    class ObsMock:
        found = True
        cx = 0.8
    flow.ctx.last_target_obs = ObsMock()

    # Mock POSE_BOTTLE configuration to enable fallback grasp
    class PoseBottleConfigMock:
        fall_back_grasp = True
        x_center_tolerance = 0.06

    class ConfigMock:
        POSE_BOTTLE = PoseBottleConfigMock()
        target_gripper_widths = {}
    
    # Patch get_config
    import common.config_loader as cl
    orig_get_config = cl.get_config
    cl.get_config = lambda: ConfigMock()

    try:
        # Since it's not centered, it will fall back to normal path,
        # which will try to call _queue_vision_req and raise AssertionError
        # because the harness grasp_status is RESULT_READY (meaning it consumes it).
        # But wait! If grasp_status is RESULT_READY, _tick_grasp_awaiting_respond
        # will consume the ready grasp result, transitioning to PRE_ARM_STOP_SETTLE.
        # So we set grasp_status to "WAITING_RESPONSE" so it attempts to send the request.
        flow.ctx.grasp_status = "WAITING_RESPONSE"
        try:
            flow._tick_grasp_awaiting_respond(100.0)
            assert False, "Should have attempted to queue vision request and failed"
        except AssertionError as exc:
            assert "ready grasp result should not request another remote respond" in str(exc)
        
        # Verify fallback grasp was NOT triggered
        assert getattr(flow.ctx, "use_fallback_grasp", False) is False
        assert flow.ctx.grasp_substate == "AWAITING_RESPOND"
    finally:
        cl.get_config = orig_get_config


def test_pre_arm_settle_with_fallback_grasp_builds_arm_cmd():
    flow = _GraspFlowHarness()
    flow.ctx.use_fallback_grasp = True
    flow.ctx.grasp_substate = "PRE_ARM_STOP_SETTLE"
    flow.ctx.pre_arm_stop_settle_start_mono = 100.0

    class PoseBottleConfigMock:
        fall_back_grasp = True
        x_center_tolerance = 0.06

    class ConfigMock:
        POSE_BOTTLE = PoseBottleConfigMock()
        target_gripper_widths = {}

    import common.config_loader as cl
    orig_get_config = cl.get_config
    cl.get_config = lambda: ConfigMock()

    try:
        decision = flow._tick_grasp_pre_arm_stop_settle(100.1)
        assert flow.ctx.grasp_substate == "AWAITING_ARM"
        assert decision.arm_cmd is not None
        assert decision.arm_cmd.command == "POSE_BOTTLE"
        assert decision.arm_cmd.x_cm == 0.0
        assert decision.arm_cmd.y_cm == 0.0
        assert decision.arm_cmd.z_cm == 0.0
        assert decision.arm_cmd.pitch_deg == 0.0
        assert decision.arm_cmd.roll_deg == 0.0
        assert decision.arm_cmd.claw_deg == 0.0
        assert decision.arm_cmd.time_ms == 0
        assert decision.control_summary["source"] == "fallback_grasp"
        assert decision.control_summary["input_grasp"] == {}
    finally:
        cl.get_config = orig_get_config
