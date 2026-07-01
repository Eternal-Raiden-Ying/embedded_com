#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORCH_ROOT = ROOT / "orchestrator"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from orchestrator_service.bridge.arm_protocol import (
    encode_pose,
    parse_arm_response,
    parse_arm_response_detail,
    pose_matches,
)
from orchestrator_service.utils.grasp_utils import grasp_to_pose_params


CANONICAL_GRASP = {
    "x_cm": 24.41,
    "y_cm": 3.16,
    "z_cm": 9.35,
    "pitch_deg": -2.57,
    "roll_deg": 81.18,
    "gripper_width_cm": 8.68,
}


def test_cloud_grasp_encodes_integer_pose_line():
    cmd = grasp_to_pose_params(CANONICAL_GRASP, time_ms=800)
    line = encode_pose(cmd.x_cm, cmd.y_cm, cmd.z_cm, cmd.pitch_deg, cmd.roll_deg, cmd.claw_deg, cmd.time_ms)

    assert line == "POSE 24 3 9 -3 81 87 800"


def test_latest_canonical_grasp_encodes_clamped_pose_line():
    grasp = {
        "x_cm": 24.488,
        "y_cm": 4.927,
        "z_cm": 9.345,
        "pitch_deg": -2.562,
        "roll_deg": 96.347,
        "gripper_width_cm": 9.227,
    }
    cmd = grasp_to_pose_params(grasp, time_ms=800)
    line = encode_pose(cmd.x_cm, cmd.y_cm, cmd.z_cm, cmd.pitch_deg, cmd.roll_deg, cmd.claw_deg, cmd.time_ms)

    assert line == "POSE 24 5 9 -3 96 90 800"


def test_arm_pose_example_encodes_expected_line():
    grasp = {
        "x_cm": 23.9389,
        "y_cm": 3.3307,
        "z_cm": 8.7589,
        "pitch_deg": 8.2860,
        "roll_deg": 67.3864,
        "gripper_width_cm": 10.0,
        "approach_depth_cm": 4.0,
    }
    cmd = grasp_to_pose_params(grasp, time_ms=800)
    line = encode_pose(cmd.x_cm, cmd.y_cm, cmd.z_cm, cmd.pitch_deg, cmd.roll_deg, cmd.claw_deg, cmd.time_ms)

    assert line == "POSE 24 3 9 8 67 90 800"


def test_missing_pose_field_does_not_default_to_zero():
    bad = dict(CANONICAL_GRASP)
    bad.pop("x_cm")

    try:
        grasp_to_pose_params(bad, time_ms=800)
    except ValueError as exc:
        assert "x_cm" in str(exc)
    else:
        raise AssertionError("missing x_cm should fail")


def test_arm_response_parser_ok_pose_and_err_ik():
    ok = parse_arm_response("OK POSE x=24 y=3 z=9 pitch=-3 roll=81 claw=87 t=800")
    err = parse_arm_response("ERR IK x=24 y=3 z=9 pitch=-3")
    cmd_err = parse_arm_response("ERR CMD")

    assert ok is not None
    assert ok.ok is True
    assert ok.parsed_status == "OK_POSE"
    assert err is not None
    assert err.ok is False
    assert err.parsed_status == "ERR_IK"
    assert cmd_err is not None
    assert cmd_err.ok is False
    assert cmd_err.parsed_status == "ERR_CMD"


def test_arm_response_detail_parses_pose_noise_and_mismatch():
    sent_pose = {"x": 24, "y": 1, "z": 9, "pitch": 8, "roll": 83, "claw": 90, "time_ms": 800}
    ok = parse_arm_response_detail("OK POSE x=24.00 y=1.00 z=9.00 pitch=8.00 roll=83.00 claw=90.00 t=800")
    stale_ok = parse_arm_response_detail("OK POSE x=10.00 y=0.00 z=10.00 pitch=-45.00 roll=0.00 claw=90.00 t=800")
    err_ik = parse_arm_response_detail("ERR IK x=24 y=1 z=9 pitch=8")
    err_cmd = parse_arm_response_detail("ERR CMD")
    noise = parse_arm_response_detail("UART1 direct alive")

    assert ok["status"] == "OK_POSE"
    assert ok["pose"] == sent_pose
    assert pose_matches(sent_pose, ok["pose"]) is True
    assert stale_ok["status"] == "OK_POSE"
    assert pose_matches(sent_pose, stale_ok["pose"]) is False
    assert err_ik["status"] == "ERR_IK"
    assert err_cmd["status"] == "ERR_CMD"
    assert noise["status"] == "NOISE"
