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


def test_yolo_visible_allows_forward_with_wz_correction():
    ctrl = _controller()
    obs = _obs(table_cx_norm=0.28, yolo_bbox_center_x_norm=0.64)

    decision = ctrl.yolo_table_search_cmd(obs, mode="YOLO_ACQUIRE_ALIGN", control_source="yolo_track_forward")

    assert decision.cmd.vx_mps > 0.0
    assert abs(decision.cmd.wz_radps) > 0.0
    assert decision.control_summary["control_source"] == "yolo_track_forward"
    assert decision.control_summary["forward_block_reason"] == ""


def test_yolo_moderate_offset_does_not_block_forward():
    ctrl = _controller()
    obs = _obs(table_cx_norm=0.34, yolo_bbox_center_x_norm=0.67)

    decision = ctrl.yolo_table_search_cmd(obs, mode="YOLO_APPROACH", control_source="yolo_track_forward")

    assert decision.cmd.vx_mps > 0.0
    assert abs(decision.cmd.wz_radps) > 0.0
    assert decision.control_summary["forward_block_reason"] == ""


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


def test_start_facing_table_warmup_goes_to_approach_without_search_rotate():
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

    assert core.ctx.state == State.EDGE_ADJUST
    assert decision.cmd.vx_mps > 0.0
    assert decision.control_summary["control_source"] == "edge_guided_forward"
    assert decision.control_summary["forward_block_reason"] == ""
    assert decision.cmd.wz_radps != 0.1


def test_bbox_touch_side_with_edge_valid_slows_forward_not_block():
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
    assert decision.cmd.vx_mps <= 0.015
    assert decision.control_summary["forward_block_reason"] == ""


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


def test_no_same_tick_yolo_approach_to_edge_adjust_blocks_forward():
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

    assert core.ctx.state == State.EDGE_ADJUST
    assert decision.cmd.vx_mps > 0.0
    assert decision.control_summary["forward_block_reason"] == ""
    assert decision.control_summary["control_source"] in {"yolo_track_forward", "edge_guided_forward"}


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

