import time

from orchestrator_service.config.schema import CarMotionConfig, ControlThresholds
from orchestrator_service.control.motion_controller import MotionController
from orchestrator_service.ipc.protocol import TableEdgeObs
from orchestrator_service.runtime.context import State
from orchestrator_service.runtime.core import OrchestratorCore


def _cfg():
    cfg = ControlThresholds()
    cfg.table_obs_max_age_s = 10.0
    cfg.table_target_dist_m = 0.5
    cfg.final_lock_dist_tol_m = 0.03
    cfg.yolo_forward_vx_mps = 0.015
    car = CarMotionConfig()
    car.table_approach_allow_wz = True
    car.table_controlled_wz_max_radps = 0.12
    car.table_controlled_wz_min_radps = 0.0
    car.table_approach_safe_vx_mps = 0.020
    car.table_approach_max_vx_mps = 0.035
    car.table_approach_yaw_realign_rad = 0.16
    car.table_edge_hard_rotate_only_yaw_rad = 0.45
    car.table_edge_hard_yaw_rotate_only_frames = 3
    car.table_edge_hard_yaw_rotate_only_ms = 350
    car.table_perception_warmup_s = 1.0
    car.table_wz_plane_max_radps = 0.12
    car.yolo_forward_center_good_limit = 0.15
    car.yolo_forward_center_hard_limit = 0.25
    return cfg, car


def _controller():
    cfg, car = _cfg()
    return MotionController(cfg, car)


def _obs(**updates):
    data = {
        "ts": time.time(),
        "table_found": True,
        "table_bbox_found": True,
        "table_bbox_current_found": True,
        "table_bbox_control_valid": True,
        "yolo_table_control_valid": True,
        "yolo_table_visible": True,
        "yolo_table_fresh": True,
        "yolo_reliable": True,
        "table_cx_norm": 0.0,
        "yolo_bbox_center_x_norm": 0.5,
        "edge_found": False,
        "edge_valid": False,
        "edge_trusted": False,
        "confidence": 0.8,
        "edge_conf": 0.8,
        "yolo_table_edge_stable_count": 6,
        "usable_for_approach": False,
        "usable_for_alignment": False,
        "usable_for_stop": False,
        "yaw_err_rad": 0.0,
        "dist_err_m": 0.58,
        "target_dist_m": 0.5,
        "pose_found": True,
        "depth_valid": True,
        "reject_reason": "",
        "point_count": 80,
        "table_point_count": 80,
    }
    data.update(updates)
    return TableEdgeObs(**data)


def test_yolo_offset_bbox_rotates_without_forward():
    ctrl = _controller()
    obs = _obs(table_cx_norm=0.60, yolo_bbox_center_x_norm=0.80)

    decision = ctrl.yolo_table_search_cmd(obs, mode="YOLO_ACQUIRE_ALIGN", control_source="yolo_track_forward")

    assert decision.cmd.vx_mps == 0.0
    assert abs(decision.cmd.wz_radps) > 0.0
    assert decision.control_summary["control_source"] == "yolo_align"
    assert decision.control_summary["forward_block_reason"] == "yolo_center_error_too_large_rotate_only"
    assert decision.cmd.wz_radps > 0.0
    assert decision.control_summary["yolo_forward_center_hard_limit"] == 0.25


def test_yolo_moderate_offset_advances_while_correcting():
    ctrl = _controller()
    obs = _obs(table_cx_norm=0.34, yolo_bbox_center_x_norm=0.67)

    decision = ctrl.yolo_table_search_cmd(obs, mode="YOLO_APPROACH", control_source="yolo_track_forward")

    assert decision.cmd.vx_mps > 0.0
    assert abs(decision.cmd.wz_radps) > 0.0
    assert decision.control_summary["control_source"] == "yolo_track_forward"
    assert decision.control_summary["table_yolo_align_center_x_tol"] == 0.08
    assert decision.control_summary["yolo_forward_center_hard_limit"] == 0.25


def test_edge_trusted_enters_controlled_approach_not_yaw_only():
    ctrl = _controller()
    obs = _obs(
        edge_found=True,
        edge_valid=True,
        edge_trusted=True,
        usable_for_approach=True,
        usable_for_alignment=True,
        yaw_err_rad=0.10,
        dist_err_m=0.58,
    )

    decision = ctrl.fov_table_approach_cmd(obs, phase="PLANE_APPROACH", mode="EDGE_ADJUST")

    assert decision.cmd.vx_mps > 0.0
    assert abs(decision.cmd.wz_radps) > 0.0
    assert decision.control_summary["control_source"] == "edge_guided_forward"
    assert decision.control_summary["forward_block_reason"] != "edge_adjust_yaw_only"


def test_start_with_offset_bbox_uses_alignment_during_warmup():
    cfg, car = _cfg()
    core = OrchestratorCore(cfg, car)
    core.ctx.state = State.SEARCH_TABLE
    core.ctx.task_start_wall_ts = time.time()
    core.ctx.state_enter_mono = time.monotonic()
    core.ctx.last_table_obs = _obs(
        edge_found=True,
        edge_valid=True,
        edge_trusted=True,
        usable_for_approach=True,
        usable_for_alignment=True,
        yaw_err_rad=-0.14,
        dist_err_m=0.57,
        table_cx_norm=0.325,
        yolo_bbox_center_x_norm=0.6625,
        table_bbox_touch_right=True,
        yolo_bbox_touch_right=True,
    )

    decision = core.tick()

    assert core.ctx.state == State.YOLO_APPROACH
    assert decision.cmd.vx_mps > 0.0
    assert decision.cmd.wz_radps > 0.0
    assert decision.control_summary["control_source"] == "yolo_track_forward"


def test_bbox_touch_side_with_edge_valid_still_aligns_before_forward():
    ctrl = _controller()
    obs = _obs(
        edge_found=True,
        edge_valid=True,
        edge_trusted=False,
        usable_for_approach=True,
        table_cx_norm=0.32,
        yolo_bbox_center_x_norm=0.66,
        table_bbox_touch_right=True,
        yolo_bbox_touch_right=True,
    )

    decision = ctrl.yolo_table_search_cmd(obs, mode="YOLO_APPROACH", control_source="yolo_track_forward")

    assert decision.cmd.vx_mps > 0.0
    assert decision.cmd.wz_radps > 0.0
    assert decision.control_summary["control_source"] == "yolo_track_forward"


def test_single_large_yaw_does_not_immediately_block_forward():
    ctrl = _controller()
    obs = _obs(
        edge_found=True,
        edge_valid=True,
        edge_trusted=True,
        usable_for_approach=True,
        usable_for_alignment=True,
        yaw_err_rad=0.50,
        dist_err_m=0.58,
    )

    decision = ctrl.fov_table_approach_cmd(obs, phase="PLANE_APPROACH", mode="EDGE_ADJUST")

    assert decision.cmd.vx_mps > 0.0
    assert decision.control_summary["forward_block_reason"] == ""


def test_consecutive_hard_yaw_uses_rotate_only():
    ctrl = _controller()
    obs = _obs(
        edge_found=True,
        edge_valid=True,
        edge_trusted=True,
        usable_for_approach=True,
        usable_for_alignment=True,
        yaw_err_rad=0.50,
        dist_err_m=0.58,
    )
    setattr(obs, "hard_yaw_rotate_only_active", True)

    decision = ctrl.fov_table_approach_cmd(obs, phase="PLANE_APPROACH", mode="EDGE_ADJUST")

    assert decision.cmd.vx_mps == 0.0
    assert decision.control_summary["forward_block_reason"] == "yaw_too_large_rotate_only"


def test_offset_bbox_does_not_skip_alignment_for_edge_handoff():
    cfg, car = _cfg()
    core = OrchestratorCore(cfg, car)
    core.ctx.state = State.YOLO_ACQUIRE_ALIGN
    core.ctx.state_enter_mono = time.monotonic()
    core.ctx.last_table_obs = _obs(
        edge_found=True,
        edge_valid=True,
        edge_trusted=True,
        usable_for_approach=True,
        usable_for_alignment=True,
        yaw_err_rad=0.10,
        dist_err_m=0.58,
        table_cx_norm=0.28,
        yolo_bbox_center_x_norm=0.64,
    )

    decision = core.tick()

    assert core.ctx.state == State.YOLO_APPROACH
    assert decision.cmd.vx_mps > 0.0
    assert decision.cmd.wz_radps > 0.0
    assert decision.control_summary["docking_action"] == "BBOX_TRACK_FORWARD"


def test_target_distance_compatibility():
    ctrl = _controller()
    ctrl.cfg.table_target_dist_m = 0.30
    obs = _obs(
        obs_target_dist_m=0.50,
        dist_err_m=-0.15,
        target_dist_m=0.50,
        edge_found=True,
        edge_valid=True,
        edge_trusted=True,
        usable_for_approach=True,
        usable_for_alignment=True,
    )
    cmd = ctrl._cmd("EDGE_ADJUST")
    summary = ctrl._summary("EDGE_ADJUST", cmd, obs)
    assert abs(summary["measured_dist_m"] - 0.35) < 1e-4
    assert abs(summary["final_dist_err_m"] - 0.05) < 1e-4


def _core_in_state(state, obs):
    cfg, car = _cfg()
    core = OrchestratorCore(cfg, car)
    core.ctx.state = state
    core.ctx.task_start_wall_ts = time.time() - 2.0
    core.ctx.state_enter_mono = time.monotonic()
    core.ctx.last_table_obs = obs
    return core


def test_right_bbox_keeps_positive_wz_through_authority_and_arbiter():
    core = _core_in_state(
        State.SEARCH_TABLE,
        _obs(obs_seq=1001, table_cx_norm=0.60, yolo_bbox_center_x_norm=0.80),
    )

    decision = core.tick()

    assert core.ctx.state == State.YOLO_ACQUIRE_ALIGN
    assert decision.cmd.vx_mps == 0.0
    assert decision.cmd.wz_radps > 0.0
    assert 0.30 * decision.cmd.wz_radps >= 0.0
    assert decision.control_summary["control_source"] != "local_rotate_search"
    assert "control_invariant_violation" not in decision.control_summary


def test_left_bbox_keeps_negative_wz_through_authority_and_arbiter():
    core = _core_in_state(
        State.SEARCH_TABLE,
        _obs(obs_seq=1002, table_cx_norm=-0.60, yolo_bbox_center_x_norm=0.20),
    )

    decision = core.tick()

    assert core.ctx.state == State.YOLO_ACQUIRE_ALIGN
    assert decision.cmd.vx_mps == 0.0
    assert decision.cmd.wz_radps < 0.0
    assert -0.30 * decision.cmd.wz_radps >= 0.0
    assert decision.control_summary["control_source"] != "local_rotate_search"
    assert "control_invariant_violation" not in decision.control_summary


def test_centered_bbox_enters_approach_with_forward_motion():
    core = _core_in_state(
        State.SEARCH_TABLE,
        _obs(obs_seq=1003, table_cx_norm=0.08, yolo_bbox_center_x_norm=0.54),
    )

    decision = core.tick()

    assert core.ctx.state == State.YOLO_APPROACH
    assert decision.cmd.vx_mps > 0.0
    assert abs(decision.cmd.wz_radps) < 1e-9
    assert decision.control_summary["control_source"] == "yolo_track_forward"


def test_moderate_bbox_error_stays_in_approach_with_vx_and_wz():
    core = _core_in_state(
        State.YOLO_APPROACH,
        _obs(obs_seq=1004, table_cx_norm=0.30, yolo_bbox_center_x_norm=0.65),
    )

    decision = core.tick()

    assert core.ctx.state == State.YOLO_APPROACH
    assert decision.cmd.vx_mps > 0.0
    assert decision.cmd.wz_radps > 0.0
    assert decision.control_summary["control_source"] == "yolo_track_forward"
    assert decision.control_summary["yolo_forward_center_hard_limit"] == 0.25


def test_single_empty_rgb_inference_uses_bbox_lost_hold():
    lost = _obs(
        obs_seq=1101,
        table_found=False,
        table_bbox_found=False,
        table_bbox_current_found=False,
        table_bbox_control_valid=False,
        yolo_table_control_valid=False,
        yolo_table_visible=False,
        yolo_table_fresh=False,
        yolo_reliable=False,
        table_cx_norm=None,
        yolo_bbox_center_x_norm=None,
    )
    core = _core_in_state(State.YOLO_APPROACH, lost)

    decision = core.tick()

    assert core.ctx.state == State.YOLO_APPROACH
    assert decision.cmd.vx_mps == 0.0
    assert abs(decision.cmd.wz_radps) < abs(core.car_cfg.search_table_wz_radps)
    assert decision.control_summary["control_source"] == "bbox_lost_hold"


def test_bbox_loss_reaches_configured_frame_limit_before_search():
    first = _obs(
        obs_seq=1201,
        table_found=False,
        table_bbox_found=False,
        table_bbox_current_found=False,
        table_bbox_control_valid=False,
        yolo_table_control_valid=False,
        yolo_table_visible=False,
        yolo_table_fresh=False,
        yolo_reliable=False,
        table_cx_norm=None,
        yolo_bbox_center_x_norm=None,
    )
    core = _core_in_state(State.YOLO_APPROACH, first)
    core.cfg.yolo_table_lost_to_search_frames = 2
    core.cfg.table_loss_hold_s = 10.0

    first_decision = core.tick()
    core.ctx.last_table_obs = _obs(**{**first.__dict__, "obs_seq": 1202, "ts": time.time()})
    second_decision = core.tick()

    assert first_decision.control_summary["control_source"] == "bbox_lost_hold"
    assert core.ctx.state == State.SEARCH_TABLE
    assert second_decision.control_summary["control_source"] == "local_rotate_search"


def test_expired_search_latch_does_not_override_current_direction_or_bbox():
    core = _core_in_state(
        State.SEARCH_TABLE,
        _obs(obs_seq=1301, table_cx_norm=0.60, yolo_bbox_center_x_norm=0.80),
    )
    core.ctx.search_wz_sign_latched = -1
    core.ctx.search_wz_latch_until_mono = time.monotonic() - 1.0

    decision = core.tick()

    assert core.ctx.search_wz_sign_latched == 0
    assert core.ctx.search_wz_latch_until_mono == 0.0
    assert decision.cmd.wz_radps > 0.0
    assert decision.control_summary["control_source"] != "local_rotate_search"


def test_single_table_search_timeout_brakes_in_error_recovery():
    cfg, car = _cfg()
    cfg.multi_table_enabled = False
    cfg.search_table_timeout_s = 0.01
    core = OrchestratorCore(cfg, car)
    core.ctx.state = State.SEARCH_TABLE
    core.ctx.state_enter_mono = time.monotonic() - 1.0

    decision = core.tick()

    assert core.ctx.state == State.ERROR_RECOVERY
    assert decision.cmd.brake is True
    assert decision.cmd.wz_radps == 0.0
    assert decision.control_summary["control_source"] != "next_table"


def test_single_table_reacquire_timeout_brakes_in_error_recovery():
    cfg, car = _cfg()
    cfg.multi_table_enabled = False
    cfg.reacquire_timeout_s = 0.01
    core = OrchestratorCore(cfg, car)
    core.ctx.state = State.REACQUIRE_TABLE
    core.ctx.state_enter_mono = time.monotonic() - 1.0

    decision = core.tick()

    assert core.ctx.state == State.ERROR_RECOVERY
    assert decision.cmd.brake is True
    assert decision.cmd.wz_radps == 0.0
    assert decision.control_summary["control_source"] != "next_table"


def test_single_table_target_timeout_brakes_in_error_recovery():
    cfg, car = _cfg()
    cfg.multi_table_enabled = False
    core = OrchestratorCore(cfg, car)
    core.ctx.state = State.EDGE_SLIDE_SEARCH
    core.ctx.state_enter_mono = time.monotonic() - 20.0
    core._can_relocate_edge = lambda: False

    decision = core._handle_edge_slide_target_timeout(None, {}, "target_missing")

    assert core.ctx.state == State.ERROR_RECOVERY
    assert decision.cmd.brake is True
    assert decision.cmd.wz_radps == 0.0
    assert decision.control_summary["control_source"] != "next_table"


def test_single_table_no_progress_exhaustion_never_uses_next_table():
    cfg, car = _cfg()
    cfg.multi_table_enabled = False
    cfg.dock_retry_limit = 0
    core = OrchestratorCore(cfg, car)
    core.ctx.state = State.YOLO_APPROACH
    core.ctx.state_enter_mono = time.monotonic()

    decision = core._enter_no_progress_recovery_or_next("test_no_progress")

    assert core.ctx.state == State.SEARCH_TABLE
    assert decision.cmd.mode == "SEARCH_TABLE"
    assert decision.control_summary["control_source"] == "local_rotate_search"
    assert decision.control_summary["multi_table_enabled"] is False


def test_single_table_manual_next_table_state_is_hard_stopped():
    cfg, car = _cfg()
    cfg.multi_table_enabled = False
    core = OrchestratorCore(cfg, car)
    core.ctx.state = State.NEXT_TABLE
    core.ctx.state_enter_mono = time.monotonic()

    decision = core.tick()

    assert core.ctx.state == State.ERROR_RECOVERY
    assert decision.cmd.brake is True
    assert decision.cmd.wz_radps == 0.0
    assert decision.control_summary["control_source"] == "search_failed_stop"


def test_single_table_transition_guard_rejects_next_table():
    cfg, car = _cfg()
    cfg.multi_table_enabled = False
    core = OrchestratorCore(cfg, car)
    core.ctx.state = State.SEARCH_TABLE

    core._transition(State.NEXT_TABLE, "test_forbidden_transition")

    assert core.ctx.state == State.ERROR_RECOVERY
    assert "single_table_next_table_blocked" in core.ctx.last_enter_reason
