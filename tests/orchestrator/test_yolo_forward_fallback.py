#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[2]
ORCH_ROOT = ROOT / "orchestrator"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from orchestrator_service.config.schema import CarMotionConfig, ControlThresholds
from orchestrator_service.control.motion_controller import MotionController
from orchestrator_service.ipc.protocol import TableEdgeObs, compute_bbox_control_geometry
from orchestrator_service.runtime.context import State
from orchestrator_service.runtime.common import monotonic_ts
from orchestrator_service.runtime.state_machine import OrchestratorCore
from orchestrator_service.runtime.motion_arbiter import MotionIntent, arbitrate_table_docking_motion
from orchestrator_service.runtime.context import RuntimeContext


def test_bbox_control_geometry_prefers_bbox_over_legacy_table_cx_placeholder():
    obs = TableEdgeObs.from_dict(
        {
            "ts": 1.0,
            "table_bbox_xyxy": [0.25, 0.35, 0.75, 0.90],
            "table_cx_norm": 0.0,
        }
    )

    geom = compute_bbox_control_geometry(obs)

    assert geom["bbox_center_valid"] is True
    assert geom["bbox_center_source"] == "table_bbox_xyxy_normalized"
    assert abs(geom["bbox_cx_norm_control"] - 0.5) < 1e-6
    assert abs(geom["bbox_center_error_control"]) < 1e-6


def test_bbox_control_geometry_uses_table_cx_norm_as_last_fallback():
    obs = TableEdgeObs.from_dict({"ts": 1.0, "table_cx_norm": 0.7})

    geom = compute_bbox_control_geometry(obs)

    assert geom["bbox_center_valid"] is True
    assert geom["bbox_center_source"] == "table_cx_norm_fallback"
    assert abs(geom["bbox_cx_norm_control"] - 0.7) < 1e-6


def test_bbox_control_geometry_normalizes_pixel_bbox_with_rgb_width():
    obs = TableEdgeObs.from_dict(
        {
            "ts": 1.0,
            "table_bbox_xyxy": [320, 20, 640, 200],
            "rgb_shape": [480, 640],
        }
    )

    geom = compute_bbox_control_geometry(obs)

    assert geom["bbox_center_valid"] is True
    assert geom["bbox_center_source"] == "table_bbox_xyxy_rgb_shape"
    assert abs(geom["bbox_cx_norm_control"] - 0.75) < 1e-6


def test_fov_approach_uses_yolo_forward_when_bbox_exists_but_edge_not_trusted():
    controller = MotionController(ControlThresholds(), CarMotionConfig())
    obs = TableEdgeObs.from_dict(
        {
            "ts": 1.0,
            "table_found": True,
            "edge_found": True,
            "table_bbox_xyxy": [0.25, 0.35, 0.75, 0.90],
            "table_bbox_found": True,
            "table_bbox_control_valid": True,
            "table_cx_norm": 0.0,
            "edge_trusted": False,
            "valid_for_control": False,
            "usable_for_approach": False,
        }
    )

    decision = controller.fov_table_approach_cmd(obs, mode="CONTROLLED_APPROACH")

    assert decision.cmd.vx_mps > 0.0
    assert decision.cmd.wz_radps == 0.0
    assert decision.control_summary["control_source"] == "yolo_track_forward"
    assert decision.control_summary["reason"] == "table_bbox_found_edge_not_trusted_yolo_forward"


def test_search_table_transitions_to_yolo_approach_and_commands_forward_motion():
    cfg = ControlThresholds()
    car_cfg = CarMotionConfig()
    core = OrchestratorCore(cfg, car_cfg)
    now = time.time()
    core.ctx.state = State.SEARCH_TABLE
    core.ctx.task_start_wall_ts = now - 1.0
    # Three distinct fresh observations are required before SEARCH_TABLE hands
    # yaw ownership from the coarse scan to bbox tracking.
    for seq in range(1, 4):
        core.ctx.last_table_obs = _search_bbox_obs(now, seq, left=True)
        decision = core.tick()
        assert core.ctx.state == State.SEARCH_TABLE
        assert decision.control_summary["yaw_owner"] == "search"
    core.ctx.last_table_obs = _search_bbox_obs(now, 4, left=True)
    decision = core.tick()

    assert core.ctx.state == State.YOLO_ACQUIRE_ALIGN
    assert decision.cmd.vx_mps >= 0.0
    assert decision.cmd.brake is False
    assert decision.control_summary["control_source"] == "yolo_track_forward"
    assert decision.control_summary["allow_forward"] is True
    assert decision.control_summary["forward_block_reason"] == ""


def test_yolo_acquire_align_outputs_rotate_command_not_stop_summary():
    cfg = ControlThresholds()
    car_cfg = CarMotionConfig()
    core = OrchestratorCore(cfg, car_cfg)
    now = time.time()
    core.ctx.state = State.SEARCH_TABLE
    core.ctx.task_start_wall_ts = now - 2.0
    for seq in range(1, 4):
        core.ctx.last_table_obs = _search_bbox_obs(now, seq, left=True)
        decision = core.tick()
        assert core.ctx.state == State.SEARCH_TABLE
    core.ctx.last_table_obs = _search_bbox_obs(now, 4, left=True)
    decision = core.tick()

    assert core.ctx.state == State.YOLO_ACQUIRE_ALIGN
    assert abs(decision.cmd.wz_radps) > 0.0
    assert decision.cmd.brake is False
    assert decision.control_summary["speed_profile"] == "search"
    assert decision.control_summary["allow_rotate"] is True


def test_stale_guard_accepts_millisecond_timing_fields_from_vista():
    controller = MotionController(ControlThresholds(), CarMotionConfig())
    now = time.time()
    obs = TableEdgeObs.from_dict(
        {
            "ts": now,
            "table_found": True,
            "edge_found": True,
            "camera_frame_ts_ms": int((now - 0.05) * 1000.0),
            "orchestrator_recv_ts_ms": int((now - 0.01) * 1000.0),
        }
    )

    timing = controller._stale_guard(obs, control_ts=now)

    assert timing["stale_level"] == "fresh"
    assert timing["obs_total_age_ms"] < 100.0
    assert timing["control_loop_age_ms"] < 50.0


def test_lost_search_timeout_is_suppressed_when_fresh_yolo_bbox_is_visible():
    cfg = ControlThresholds()
    cfg.no_table_bbox_timeout_s = 0.01
    car_cfg = CarMotionConfig()
    core = OrchestratorCore(cfg, car_cfg)
    now = time.time()
    core.ctx.state = State.SEARCH_TABLE
    core.ctx.prev_state = State.EDGE_ADJUST
    core.ctx.last_enter_reason = "EDGE_ADJUST lost table bbox, enter SEARCH"
    core.ctx.state_enter_mono = monotonic_ts() - 1.0
    core.ctx.task_start_wall_ts = now - 5.0
    core.ctx.last_table_obs = _search_bbox_obs(now, 1, left=True)

    decision = core.tick()

    assert core.ctx.state == State.SEARCH_TABLE
    assert decision.cmd.vx_mps == 0.0
    assert decision.control_summary["yaw_owner"] == "search"
    assert decision.control_summary["table_lost_search_timeout"] is False
    assert decision.control_summary["selected_timeout_reason"] == "bbox_visible_but_edge_invalid"
    assert decision.control_summary["docking_reason"] == "coarse_scan_no_valid_edge"


def test_lost_search_timeout_enters_error_when_no_fresh_yolo_bbox_exists():
    cfg = ControlThresholds()
    cfg.no_table_bbox_timeout_s = 0.01
    car_cfg = CarMotionConfig()
    core = OrchestratorCore(cfg, car_cfg)
    now = time.time()
    core.ctx.state = State.SEARCH_TABLE
    core.ctx.prev_state = State.EDGE_ADJUST
    core.ctx.last_enter_reason = "EDGE_ADJUST lost table bbox, enter SEARCH"
    core.ctx.state_enter_mono = monotonic_ts() - 1.0
    core.ctx.task_start_wall_ts = now - 5.0
    core.ctx.last_table_obs = TableEdgeObs.from_dict(
        {
            "ts": now,
            "obs_ts": now,
            "table_found": False,
            "edge_found": False,
            "edge_valid": False,
            "table_bbox_found": False,
            "table_bbox_current_found": False,
            "yolo_table_visible": False,
            "yolo_table_fresh": False,
        }
    )

    decision = core.tick()

    assert core.ctx.state == State.ERROR_RECOVERY
    assert decision.cmd.brake is True
    assert decision.control_summary["table_lost_search_timeout"] is True
    assert decision.control_summary["no_table_bbox_timeout"] is True
    assert decision.control_summary["selected_timeout_reason"] == "no_table_bbox_timeout"
    assert decision.control_summary["fallback_action"] == "error_recovery"


def test_error_recovery_returns_idle_without_stopping_vision_by_default():
    cfg = ControlThresholds()
    cfg.error_recovery_hold_s = 0.0
    cfg.keep_vision_alive_after_task = True
    cfg.task_done_shutdown_vision = False
    core = OrchestratorCore(cfg, CarMotionConfig())
    core.ctx.state = State.ERROR_RECOVERY
    core.ctx.active_session_id = "session-1"
    core.ctx.active_epoch = 7

    decision = core.tick()

    assert core.ctx.state == State.IDLE
    assert decision.cmd.brake is True
    assert core.ctx.pending_vision_msgs == []


def test_done_returns_idle_without_stopping_vision_by_default():
    cfg = ControlThresholds()
    cfg.done_hold_s = 0.0
    cfg.keep_vision_alive_after_task = True
    cfg.task_done_shutdown_vision = False
    core = OrchestratorCore(cfg, CarMotionConfig())
    core.ctx.state = State.DONE
    core.ctx.active_session_id = "session-1"
    core.ctx.active_epoch = 7

    decision = core.tick()

    assert core.ctx.state == State.IDLE
    assert decision.cmd.mode == "DONE"
    assert core.ctx.pending_vision_msgs == []


def test_task_done_shutdown_vision_preserves_explicit_stop_behavior():
    cfg = ControlThresholds()
    cfg.done_hold_s = 0.0
    cfg.keep_vision_alive_after_task = True
    cfg.task_done_shutdown_vision = True
    core = OrchestratorCore(cfg, CarMotionConfig())
    core.ctx.state = State.DONE
    core.ctx.active_session_id = "session-1"
    core.ctx.active_epoch = 7

    core.tick()

    assert core.ctx.state == State.IDLE
    assert len(core.ctx.pending_vision_msgs) == 1
    assert core.ctx.pending_vision_msgs[0]["op"] == "STOP"


def _search_bbox_obs(now: float, seq: int, *, left: bool = True, fresh: bool = True,
                     reused: bool = False, roi_latched: bool = False) -> TableEdgeObs:
    # Pixel geometry is intentional: the protocol rejects normalized xyxy boxes.
    bbox = [20, 30, 220, 90] if left else [280, 30, 480, 90]
    return TableEdgeObs.from_dict(
        {
            "ts": now + seq * 0.001,
            "obs_ts": now + seq * 0.001,
            "seq": seq,
            "table_found": True,
            "edge_found": False,
            "edge_valid": False,
            "edge_trusted": False,
            "table_bbox_found": True,
            "table_bbox_current_found": True,
            "table_bbox_xyxy": bbox,
            "rgb_shape": [100, 500],
            "yolo_table_visible": True,
            "yolo_table_fresh": fresh,
            "yolo_table_age_ms": 0.0,
            "yolo_table_control_valid": True,
            "same_obs_reuse_count": 1 if reused else 0,
            "table_roi_latched": roi_latched,
            "depth_valid": True,
        }
    )


def _search_core() -> Tuple[OrchestratorCore, float]:
    core = OrchestratorCore(ControlThresholds(), CarMotionConfig())
    now = time.time()
    core.ctx.state = State.SEARCH_TABLE
    core.ctx.task_start_wall_ts = now - 2.0
    return core, now


def test_search_table_invalid_edge_and_flickering_bbox_keep_latched_coarse_direction():
    core, now = _search_core()
    wz_values = []
    owners = []
    for seq in range(1, 51):
        core.ctx.last_table_obs = _search_bbox_obs(now, seq, left=bool(seq % 2))
        decision = core.tick()
        wz_values.append(decision.cmd.wz_radps)
        owners.append(decision.control_summary["yaw_owner"])
        assert decision.control_summary["docking_reason"] == "coarse_scan_no_valid_edge"

    assert core.ctx.state == State.SEARCH_TABLE
    assert all(wz > 0.0 for wz in wz_values)
    assert sum(1 for before, after in zip(wz_values, wz_values[1:]) if before * after < 0.0) == 0
    assert owners == ["search"] * 50
    assert core.ctx.search_bbox_confirm_streak < 3


def test_reused_or_latched_bbox_cannot_confirm_or_flip_search_owner():
    core, now = _search_core()
    for seq in range(1, 5):
        core.ctx.last_table_obs = _search_bbox_obs(now, seq, left=False, reused=True)
        decision = core.tick()
        assert decision.cmd.wz_radps > 0.0
        assert decision.control_summary["yaw_owner"] == "search"
    assert core.ctx.search_bbox_confirm_streak == 0

    for seq in range(5, 9):
        core.ctx.last_table_obs = _search_bbox_obs(now, seq, left=False, roi_latched=True)
        decision = core.tick()
        assert decision.cmd.wz_radps > 0.0
        assert decision.control_summary["yaw_owner"] == "search"
    assert core.ctx.search_bbox_confirm_streak == 0
    assert core.ctx.state == State.SEARCH_TABLE


def test_three_fresh_same_direction_bbox_frames_allow_one_controlled_handoff():
    core, now = _search_core()
    owners = []
    phases = []
    for seq in range(1, 5):
        core.ctx.last_table_obs = _search_bbox_obs(now, seq, left=True)
        decision = core.tick()
        owners.append(decision.control_summary["yaw_owner"])
        phases.append(core.ctx.state)

    assert owners[:3] == ["search", "search", "search"]
    assert phases[:3] == [State.SEARCH_TABLE] * 3
    assert phases[3] == State.YOLO_ACQUIRE_ALIGN
    assert owners[3] == "bbox"

    # A single lost frame is held by the existing bbox-loss hold rather than
    # immediately re-entering coarse search and reselecting its direction.
    core.ctx.last_table_obs = TableEdgeObs.from_dict(
        {"ts": now + 0.01, "obs_ts": now + 0.01, "seq": 5,
         "table_found": False, "edge_found": False, "edge_valid": False,
         "yolo_table_visible": False, "yolo_table_fresh": False}
    )
    core.tick()
    assert core.ctx.state == State.YOLO_ACQUIRE_ALIGN


def test_live_invalid_edge_is_not_reclassified_as_dead_no_last_good_recovery():
    ctx = RuntimeContext(state=State.SEARCH_TABLE)
    result = arbitrate_table_docking_motion(
        ctx,
        _search_bbox_obs(time.time(), 1),
        MotionIntent("local_rotate_search", desired_wz=0.20, rotate_allowed_by_behavior=True),
        {
            "state": "SEARCH_TABLE",
            "stale_level": "dead",
            "edge_stale_dead": False,
            "perception_dead": False,
            "last_good_obs_healthy": False,
            "last_good_obs_age_ms": 999999.0,
            "edge_found": False,
            "edge_valid": False,
            "edge_trusted": False,
            "search_wz_radps": 0.20,
        },
    )
    assert result.reason == "coarse_scan_no_valid_edge"
    assert result.summary["docking_reason"] == "coarse_scan_no_valid_edge"


def test_real_dead_without_last_good_keeps_recovery_behavior():
    ctx = RuntimeContext(state=State.SEARCH_TABLE)
    result = arbitrate_table_docking_motion(
        ctx,
        None,
        MotionIntent("local_rotate_search", desired_wz=0.20, rotate_allowed_by_behavior=True),
        {
            "state": "SEARCH_TABLE",
            "stale_level": "dead",
            "edge_stale_dead": True,
            "perception_dead": True,
            "last_good_obs_healthy": False,
            "last_good_obs_age_ms": 999999.0,
            "search_wz_radps": 0.20,
        },
    )
    assert result.reason == "stale_dead_no_last_good_recovery"
