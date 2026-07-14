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

from orchestrator_service.control.velocity_smoother import MotionSmoothingConfig, VelocitySmoother
from orchestrator_service.ipc.protocol import CmdVel


def _cmd(mode="YOLO_APPROACH", vx=0.0, vy=0.0, wz=0.0, brake=False):
    return CmdVel(ts=1.0, mode=mode, vx_mps=vx, vy_mps=vy, wz_radps=wz, hold_ms=150, brake=brake)


def test_vx_ramps_instead_of_jumping():
    smoother = VelocitySmoother(MotionSmoothingConfig())
    out, meta = smoother.apply(_cmd(vx=0.2), state="YOLO_APPROACH", now_monotonic=10.0)
    assert 0.0 < out.vx_mps < 0.2
    assert abs(out.vx_mps - 0.007) < 1e-9
    assert meta["smoothing_applied"] is True


def test_wz_reversal_smooths_through_zero():
    smoother = VelocitySmoother(MotionSmoothingConfig())
    smoother.last_wz = 0.2
    smoother.last_ts_monotonic = 10.0
    out, _ = smoother.apply(_cmd(wz=-0.18), state="YOLO_ACQUIRE_ALIGN", now_monotonic=10.1)
    assert 0.0 < out.wz_radps < 0.2


def test_hard_stop_immediate_zero_and_reset():
    smoother = VelocitySmoother(MotionSmoothingConfig())
    smoother.last_vx = 0.15
    smoother.last_wz = 0.12
    out, meta = smoother.apply(_cmd(mode="STOP", vx=0.2, wz=0.2, brake=True), state="YOLO_APPROACH", now_monotonic=10.0)
    assert out.brake is True
    assert out.vx_mps == 0.0
    assert out.vy_mps == 0.0
    assert out.wz_radps == 0.0
    assert smoother.last_vx == 0.0
    assert smoother.last_wz == 0.0
    assert meta["smoothing_bypassed"] is True


def test_edge_slide_does_not_retain_vx_or_wz():
    smoother = VelocitySmoother(MotionSmoothingConfig())
    smoother.last_vx = 0.2
    smoother.last_vy = 0.0
    smoother.last_wz = 0.18
    smoother.last_ts_monotonic = 10.0
    out, _ = smoother.apply(_cmd(mode="EDGE_SLIDE_SEARCH", vx=0.2, vy=0.03, wz=0.18), state="EDGE_SLIDE_SEARCH", now_monotonic=10.1)
    assert out.vx_mps == 0.0
    assert out.wz_radps == 0.0
    assert 0.0 < out.vy_mps < 0.03


def test_search_table_wz_is_normal_slew_and_reversal_does_not_jump():
    normal = VelocitySmoother(MotionSmoothingConfig())
    normal.last_ts_monotonic = 10.0
    normal.last_wz = 0.20
    out, meta = normal.apply(_cmd(wz=-0.06), state="SEARCH_TABLE", now_monotonic=10.1)
    assert 0.0 < out.wz_radps < 0.20
    assert meta["smoothing_urgent"] is False
    assert meta["smoothing_profile"] != "urgent_wz"


def test_bbox_reacquire_alignment_uses_normal_wz_slew():
    smoother = VelocitySmoother(MotionSmoothingConfig())
    smoother.last_ts_monotonic = 10.0
    smoother.last_wz = 0.20
    out, meta = smoother.apply(
        _cmd(wz=-0.06), state="YOLO_ACQUIRE_ALIGN", now_monotonic=10.1,
        summary={"docking_action": "BBOX_REACQUIRE_ROTATE", "fov_guard_level": "none"},
    )
    assert 0.0 < out.wz_radps < 0.20
    assert meta["smoothing_urgent"] is False
    assert meta["smoothing_profile"] != "urgent_wz"


def test_hard_fov_recovery_keeps_urgent_wz_profile():
    smoother = VelocitySmoother(MotionSmoothingConfig())
    smoother.last_ts_monotonic = 10.0
    out, meta = smoother.apply(
        _cmd(wz=0.3), state="SEARCH_TABLE", now_monotonic=10.1,
        summary={"docking_action": "BBOX_REACQUIRE_ROTATE", "fov_guard_level": "hard"},
    )
    assert out.wz_radps > 0.09
    assert meta["smoothing_urgent"] is True
