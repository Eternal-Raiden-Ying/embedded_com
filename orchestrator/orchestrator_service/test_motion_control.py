import copy
import time

from orchestrator_service.config.schema import CarMotionConfig, ControlThresholds
from orchestrator_service.control.motion_controller import MotionController, enforce_bbox_uart_yaw_sign
from orchestrator_service.ipc.protocol import TableEdgeObs, TargetObs
from orchestrator_service.runtime.context import State
from orchestrator_service.runtime.core import OrchestratorCore
from orchestrator_service.bridge.simple_car_protocol import encode_vel


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


def test_table_edge_protocol_preserves_completed_inference_metadata():
    obs = TableEdgeObs.from_dict(
        {
            "ts": time.time(),
            "table_found": True,
            "edge_found": False,
            "table_bbox_current_found": True,
            "table_bbox_control_valid": True,
            "table_bbox_xyxy": [100, 80, 500, 440],
            "inference_executed": True,
            "inference_completed": True,
            "inference_seq": 77,
            "inference_age_ms": 180.0,
            "control_bbox_age_ms": 180.0,
            "has_new_inference": False,
            "explicit_negative_detection": False,
            "bbox_hold_reason": "no_new_inference",
        }
    )
    assert obs.inference_completed is True
    assert obs.inference_seq == 77
    assert obs.has_new_inference is False
    assert obs.control_bbox_age_ms == 180.0
    assert obs.bbox_hold_reason == "no_new_inference"


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


def test_bbox_sign_is_preserved_in_final_uart_encoding():
    encoded = []
    for error in (0.30, -0.30):
        core = _core_in_state(
            State.SEARCH_TABLE,
            _obs(table_cx_norm=error * 2.0, yolo_bbox_center_x_norm=0.5 + error),
        )
        decision = core.tick()
        service_cmd = copy.copy(decision.cmd)
        service_summary = dict(decision.control_summary)
        corrected = enforce_bbox_uart_yaw_sign(service_cmd, service_summary, core.controller)
        assert corrected is False
        line = encode_vel(service_cmd.vx_mps, service_cmd.vy_mps, service_cmd.wz_radps)
        uart_wz = float(line.strip().split()[3])
        encoded.append(uart_wz)
        assert error * decision.control_summary["candidate_cmd"]["wz_radps"] >= 0.0
        assert error * decision.control_summary["arbiter_final_cmd"]["wz_radps"] >= 0.0
        assert error * uart_wz > 0.0
    assert encoded[0] > 0.0
    assert encoded[1] < 0.0


def test_service_corrects_reversed_bbox_yaw_before_uart():
    core = _core_in_state(
        State.SEARCH_TABLE,
        _obs(table_cx_norm=0.60, yolo_bbox_center_x_norm=0.80),
    )
    decision = core.tick()
    service_cmd = copy.copy(decision.cmd)
    service_cmd.wz_radps = -abs(service_cmd.wz_radps)
    summary = dict(decision.control_summary)

    assert enforce_bbox_uart_yaw_sign(service_cmd, summary, core.controller) is True
    assert service_cmd.wz_radps > 0.0
    assert float(encode_vel(service_cmd.vx_mps, service_cmd.vy_mps, service_cmd.wz_radps).split()[3]) > 0.0
    assert "BBOX_FINAL_YAW_SIGN_WRONG" in summary["control_invariant_violations"]


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


def test_low_rate_yolo_high_rate_control_has_30_continuous_commands():
    first = _obs(
        obs_seq=1050,
        inference_completed=True,
        inference_seq=1,
        has_new_inference=True,
        explicit_negative_detection=False,
        control_bbox_age_ms=0.0,
        table_cx_norm=0.30,
        yolo_bbox_center_x_norm=0.65,
    )
    core = _core_in_state(State.YOLO_APPROACH, first)
    vx_values = []
    wz_values = []
    sources = []
    actions = []

    for tick in range(30):
        if tick:
            completed_seq = 1 + tick // 4
            has_new = tick % 4 == 0
            bbox_age_ms = 0.0 if has_new else (tick % 4) * 100.0
            # VISTA republishes the latest completed inference while reporting
            # that this scheduler tick did not produce a new inference.
            core.ctx.last_table_obs = _obs(
                obs_seq=1050 + tick,
                inference_completed=True,
                inference_seq=completed_seq,
                has_new_inference=has_new,
                explicit_negative_detection=False,
                control_bbox_age_ms=bbox_age_ms,
                table_cx_norm=0.30,
                yolo_bbox_center_x_norm=0.65,
            )
        decision = core.tick()
        vx_values.append(round(decision.cmd.vx_mps, 6))
        wz_values.append(round(decision.cmd.wz_radps, 6))
        sources.append(decision.control_summary.get("control_source"))
        actions.append(decision.control_summary.get("docking_action"))
        assert core.ctx.state == State.YOLO_APPROACH
        assert decision.control_summary.get("perception_dropout_hold_active") is not True

    assert all(vx > 0.0 for vx in vx_values)
    assert all(wz > 0.0 for wz in wz_values)
    assert all(source == "yolo_track_forward" for source in sources)
    assert all(action == "BBOX_TRACK_FORWARD" for action in actions)
    assert not any(vx_values[index - 1] > 0.0 and vx_values[index] == 0.0 for index in range(1, 30))


def test_bbox_forward_hard_limit_has_exit_hysteresis():
    core = _core_in_state(State.YOLO_APPROACH, _obs(obs_seq=2000))
    actions = []
    states = []
    for index, error in enumerate((0.24, 0.26, 0.24, 0.26, 0.23, 0.27)):
        core.ctx.last_table_obs = _obs(
            obs_seq=2000 + index,
            table_cx_norm=error * 2.0,
            yolo_bbox_center_x_norm=0.5 + error,
        )
        decision = core.tick()
        actions.append(decision.control_summary.get("docking_action"))
        states.append(core.ctx.state)
        assert decision.cmd.vx_mps > 0.0
    assert actions == ["BBOX_TRACK_FORWARD"] * len(actions)
    assert states == [State.YOLO_APPROACH] * len(states)


def _edge_ready_core():
    cfg, car = _cfg()
    cfg.edge_trusted_stable_frames = 3
    car.table_approach_allow_vy = True
    car.table_approach_allow_wz = True
    car.table_approach_safe_vx_mps = 0.060
    car.table_approach_max_vx_mps = 0.080
    car.table_controlled_vx_min_mps = 0.060
    car.table_controlled_vx_max_mps = 0.080
    car.table_controlled_vy_max_mps = 0.050
    car.table_vy_max_mps = 0.050
    core = OrchestratorCore(cfg, car)
    core.ctx.state = State.YOLO_APPROACH
    core.ctx.state_enter_mono = time.monotonic()
    return core


def _edge_obs(seq, trusted=True, yaw=0.12, error=0.15):
    return _obs(
        obs_seq=seq,
        view_reliable=True,
        edge_found=trusted,
        edge_valid=trusted,
        edge_trusted=trusted,
        usable_for_approach=trusted,
        usable_for_alignment=trusted,
        yaw_err_rad=yaw,
        table_cx_norm=error * 2.0,
        yolo_bbox_center_x_norm=0.5 + error,
        dist_err_m=0.58,
    )


def test_edge_handoff_requires_distinct_streak_and_holds_after_single_false():
    core = _edge_ready_core()
    owners = []
    actions = []
    for index, trusted in enumerate((True, False, True, True, False, True, True, True)):
        core.ctx.last_table_obs = _edge_obs(3000 + index, trusted=trusted)
        decision = core.tick()
        owners.append(decision.control_summary.get("yaw_owner"))
        actions.append(decision.control_summary.get("docking_action"))
    assert "edge" not in owners[:-1]
    assert actions[-1] == "EDGE_READINESS_HANDOFF"
    assert core.ctx.edge_trusted_streak == 3

    core.ctx.edge_handoff_entered_mono = time.monotonic() - 1.0
    core.ctx.last_table_obs = _edge_obs(3010, trusted=True)
    completed = core.tick()
    assert completed.control_summary["docking_action"] == "EDGE_READINESS_HANDOFF"
    core.ctx.last_table_obs = _edge_obs(3011, trusted=True)
    acquired = core.tick()
    assert acquired.control_summary["docking_action"] == "EDGE_APPROACH_FORWARD"
    assert acquired.control_summary["yaw_owner"] == "edge"

    core.ctx.last_table_obs = _edge_obs(3012, trusted=False)
    held = core.tick()
    assert held.control_summary["docking_action"] == "EDGE_APPROACH_FORWARD"
    assert held.control_summary["yaw_owner"] in {"edge", "edge_ambiguous_hold"}
    assert held.cmd.vx_mps > 0.0


def test_complete_docking_action_and_owner_chain_is_monotonic():
    cfg, car = _cfg()
    cfg.edge_trusted_stable_frames = 3
    car.table_approach_allow_vy = True
    car.table_approach_safe_vx_mps = 0.060
    car.table_approach_max_vx_mps = 0.080
    car.table_controlled_vx_min_mps = 0.060
    car.table_controlled_vx_max_mps = 0.080
    car.table_controlled_vy_max_mps = 0.050
    car.table_vy_max_mps = 0.050
    core = OrchestratorCore(cfg, car)
    core.ctx.state = State.SEARCH_TABLE
    core.ctx.state_enter_mono = time.monotonic()

    search = core.tick()
    core.ctx.last_table_obs = _obs(obs_seq=3500, table_cx_norm=0.60, yolo_bbox_center_x_norm=0.80)
    reacquire = core.tick()
    core.ctx.last_table_obs = _obs(obs_seq=3501, table_cx_norm=0.30, yolo_bbox_center_x_norm=0.65)
    track = core.tick()

    for seq in range(3502, 3506):
        core.ctx.last_table_obs = _edge_obs(seq, trusted=True)
        handoff = core.tick()
    core.ctx.edge_handoff_entered_mono = time.monotonic() - 1.0
    core.ctx.last_table_obs = _edge_obs(3506, trusted=True)
    handoff = core.tick()
    core.ctx.last_table_obs = _edge_obs(3507, trusted=True)
    edge = core.tick()

    final = OrchestratorCore(cfg, car)
    final.ctx.state = State.FINAL_SLOW_STOP
    final.ctx.state_enter_mono = time.monotonic()
    final.ctx.last_table_obs = _obs(
        obs_seq=3508,
        view_reliable=True,
        edge_found=True,
        edge_valid=True,
        edge_trusted=True,
        usable_for_approach=True,
        usable_for_stop=True,
        yaw_err_rad=0.01,
        dist_err_m=0.02,
        table_roi_depth_valid=True,
        table_roi_depth_p10=0.52,
        table_roi_depth_median=0.55,
    )
    final_decision = final.tick()

    assert [
        search.control_summary["docking_action"],
        reacquire.control_summary["docking_action"],
        track.control_summary["docking_action"],
        handoff.control_summary["docking_action"],
        edge.control_summary["docking_action"],
    ] == [
        "SEARCH_ROTATE",
        "BBOX_REACQUIRE_ROTATE",
        "BBOX_TRACK_FORWARD",
        "EDGE_READINESS_HANDOFF",
        "EDGE_APPROACH_FORWARD",
    ]
    assert (search.control_summary["yaw_owner"], search.control_summary["forward_owner"]) == ("search", "none")
    assert (reacquire.control_summary["yaw_owner"], reacquire.control_summary["forward_owner"]) == ("bbox", "none")
    assert (track.control_summary["yaw_owner"], track.control_summary["forward_owner"]) == ("bbox", "bbox_track")
    assert handoff.control_summary["yaw_owner"] == "bbox"
    assert (edge.control_summary["yaw_owner"], edge.control_summary["forward_owner"], edge.control_summary["lateral_owner"]) == ("edge", "edge_approach", "edge_view")
    assert edge.cmd.vx_mps > 0.0
    assert 0.0 < abs(edge.cmd.vy_mps) <= car.table_controlled_vy_max_mps
    assert edge.cmd.wz_radps != 0.0
    assert final_decision.control_summary["docking_action"] in {
        "CLOSE_RANGE_PROBE", "FINAL_SLOW_PROBE", "FINAL_YAW_ALIGN", "FINAL_LOCKED_STOP"
    }


def test_edge_approach_outputs_combined_axes_for_20_ticks():
    core = _edge_ready_core()
    core.ctx.edge_handoff_complete = True
    core.ctx.control_phase = "EDGE_GUIDED_APPROACH"
    core.ctx.edge_readiness_score = 1.0
    commands = []
    for tick in range(20):
        yaw = 0.14 - tick * 0.002
        error = 0.16 - tick * 0.002
        core.ctx.last_table_obs = _edge_obs(4000 + tick, trusted=True, yaw=yaw, error=error)
        decision = core.tick()
        commands.append((decision.cmd.vx_mps, decision.cmd.vy_mps, decision.cmd.wz_radps))
        assert decision.control_summary["docking_action"] == "EDGE_APPROACH_FORWARD"
        assert decision.control_summary["yaw_owner"] == "edge"
        assert decision.control_summary["forward_owner"] == "edge_approach"
        assert decision.control_summary["lateral_owner"] == "edge_view"
    assert all(vx > 0.0 and abs(vy) > 0.0 and abs(wz) > 0.0 for vx, vy, wz in commands)


def test_fifty_tick_normal_approach_has_no_zero_command_pulse():
    bbox_core = _core_in_state(State.YOLO_APPROACH, _obs(obs_seq=5000))
    commands = []
    for tick in range(30):
        bbox_core.ctx.last_table_obs = _obs(
            obs_seq=5000 + tick,
            table_cx_norm=0.30,
            yolo_bbox_center_x_norm=0.65,
            has_new_inference=(tick % 4 == 0),
            inference_completed=True,
            inference_seq=1 + tick // 4,
            control_bbox_age_ms=float((tick % 4) * 100),
        )
        decision = bbox_core.tick()
        commands.append((decision.cmd.vx_mps, decision.cmd.vy_mps, decision.cmd.wz_radps))

    edge_core = _edge_ready_core()
    edge_core.ctx.edge_handoff_complete = True
    edge_core.ctx.control_phase = "EDGE_GUIDED_APPROACH"
    edge_core.ctx.edge_readiness_score = 1.0
    for tick in range(20):
        edge_core.ctx.last_table_obs = _edge_obs(6000 + tick, trusted=True, yaw=0.12, error=0.15)
        decision = edge_core.tick()
        commands.append((decision.cmd.vx_mps, decision.cmd.vy_mps, decision.cmd.wz_radps))

    assert len(commands) == 50
    assert all(any(abs(axis) > 1e-9 for axis in command) for command in commands)
    assert all(command[0] > 0.0 for command in commands)


def test_effective_speed_hierarchy_survives_uart_encoding():
    bbox = _core_in_state(
        State.YOLO_APPROACH,
        _obs(table_cx_norm=0.30, yolo_bbox_center_x_norm=0.65),
    ).tick()

    edge_core = _edge_ready_core()
    edge_core.ctx.edge_handoff_complete = True
    edge_core.ctx.control_phase = "EDGE_GUIDED_APPROACH"
    edge_core.ctx.edge_readiness_score = 1.0
    edge_core.ctx.last_table_obs = _edge_obs(6100, trusted=True, yaw=0.12, error=0.15)
    edge = edge_core.tick()

    controller = edge_core.controller
    final_vx, _vy, _wz, _meta = controller._apply_table_speed_profile(
        "FINAL_SLOW_STOP", "PLANE_STOP", 0.05, 0.0, 0.0
    )
    bbox_uart_vx = float(encode_vel(bbox.cmd.vx_mps, bbox.cmd.vy_mps, bbox.cmd.wz_radps).split()[1])
    edge_uart_vx = float(encode_vel(edge.cmd.vx_mps, edge.cmd.vy_mps, edge.cmd.wz_radps).split()[1])
    final_uart_vx = float(encode_vel(final_vx, 0.0, 0.0).split()[1])

    assert bbox_uart_vx > edge_uart_vx > final_uart_vx > 0.0
    assert (bbox_uart_vx, edge_uart_vx, final_uart_vx) == (0.10, 0.06, 0.008)


def _target_obs(seq: int, cx: float) -> TargetObs:
    return TargetObs(
        ts=time.time(),
        found=True,
        target="bottle",
        matched_cls="bottle",
        matched_conf=0.95,
        matched_bbox=[max(0.0, cx - 0.05), 0.3, min(1.0, cx + 0.05), 0.6],
        matched_center_full_norm={"cx": cx, "cy": 0.45},
        matched_area=0.03,
        bbox_valid=True,
        cx_norm=cx,
        obs_seq=seq,
    )


def _target_core(elapsed_s: float = 0.0):
    cfg, car = _cfg()
    cfg.multi_table_enabled = False
    core = OrchestratorCore(cfg, car)
    core.ctx.state = State.EDGE_SLIDE_SEARCH
    core.ctx.state_enter_mono = time.monotonic() - elapsed_s
    core.ctx.active_target = "bottle"
    core.ctx.canonical_target = "bottle"
    core._can_relocate_edge = lambda: False
    return core


def test_visible_improving_target_extends_old_wall_clock_timeout():
    core = _target_core(elapsed_s=11.0)
    decisions = []
    for seq, cx in enumerate((0.75, 0.68, 0.61), start=1):
        core.ctx.last_target_obs = _target_obs(seq, cx)
        decisions.append(core._tick_edge_slide_search())
    assert core.ctx.state == State.EDGE_SLIDE_SEARCH
    assert all(decision.control_summary.get("target_timeout_type", "") == "" for decision in decisions)
    assert decisions[-1].cmd.vy_mps != 0.0


def test_visible_target_without_progress_gets_precise_timeout():
    core = _target_core(elapsed_s=11.0)
    now = time.monotonic()
    core.ctx.target_lateral_last_good_obs_mono = now
    core.ctx.target_lateral_last_progress_mono = now - 11.0
    core.ctx.target_lateral_min_abs_err_x = 0.20
    core.ctx.last_target_obs = _target_obs(10, 0.60)
    decision = core._tick_edge_slide_search()
    assert core.ctx.state == State.ERROR_RECOVERY
    assert core.ctx.last_fail_reason == "target_lateral_no_progress_timeout"
    assert decision.control_summary["slice_timeout_reason"] == "target_lateral_no_progress_timeout"


def test_target_never_found_and_lost_have_distinct_timeout_reasons():
    never = _target_core(elapsed_s=11.0)
    never_decision = never._tick_edge_slide_search()
    assert never.ctx.last_fail_reason == "target_never_found_timeout"
    assert never_decision.control_summary["slice_timeout_reason"] == "target_never_found_timeout"

    lost = _target_core(elapsed_s=2.0)
    lost.ctx.target_lateral_last_good_obs_mono = time.monotonic() - 2.0
    lost.ctx.target_lateral_min_abs_err_x = 0.20
    lost_decision = lost._tick_edge_slide_search()
    assert lost.ctx.last_fail_reason == "target_lost_timeout"
    assert lost_decision.control_summary["slice_timeout_reason"] == "target_lost_timeout"


def test_left_bbox_correction_persists_between_inferences():
    core = _core_in_state(
        State.YOLO_APPROACH,
        _obs(
            inference_completed=True,
            inference_seq=10,
            has_new_inference=True,
            table_cx_norm=-0.30,
            yolo_bbox_center_x_norm=0.35,
        ),
    )
    commands = []
    for tick in range(6):
        if tick:
            core.ctx.last_table_obs = _obs(
                inference_completed=True,
                inference_seq=10,
                has_new_inference=False,
                control_bbox_age_ms=tick * 100.0,
                table_cx_norm=-0.30,
                yolo_bbox_center_x_norm=0.35,
            )
        decision = core.tick()
        commands.append((decision.cmd.vx_mps, decision.cmd.wz_radps))
    assert all(vx > 0.0 for vx, _ in commands)
    assert all(wz < 0.0 for _, wz in commands)


def test_center_deadband_remains_forward_without_yaw_jitter():
    core = _core_in_state(
        State.YOLO_APPROACH,
        _obs(
            inference_completed=True,
            inference_seq=20,
            has_new_inference=False,
            table_cx_norm=0.08,
            yolo_bbox_center_x_norm=0.54,
        ),
    )
    decisions = [core.tick() for _ in range(5)]
    assert all(decision.cmd.vx_mps > 0.0 for decision in decisions)
    assert all(abs(decision.cmd.wz_radps) <= 1e-9 for decision in decisions)


def test_single_explicit_negative_holds_normal_bbox_motion_without_zero_pulse():
    core = _core_in_state(
        State.YOLO_APPROACH,
        _obs(
            inference_completed=True,
            inference_seq=30,
            has_new_inference=True,
            table_cx_norm=0.30,
            yolo_bbox_center_x_norm=0.65,
        ),
    )
    positive = core.tick()
    core.ctx.last_table_obs = _obs(
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
        inference_completed=True,
        inference_seq=31,
        has_new_inference=True,
        explicit_negative_detection=True,
    )
    held = core.tick()

    assert positive.cmd.vx_mps > 0.0
    assert held.cmd.vx_mps > 0.0
    assert held.cmd.wz_radps > 0.0
    assert core.ctx.state == State.YOLO_APPROACH
    assert core.ctx.table_lost_frames == 1
    assert held.control_summary["completed_bbox_hold_active"] is True
    assert held.control_summary["control_source"] == "yolo_track_forward"
    assert held.control_summary["docking_action"] == "BBOX_TRACK_FORWARD"

    # Re-reading the same completed negative without a new inference identity
    # must neither increment the counter nor create a stop pulse.
    core.ctx.last_table_obs.has_new_inference = False
    core.ctx.last_table_obs.explicit_negative_detection = False
    repeated = core.tick()
    assert repeated.cmd.vx_mps > 0.0
    assert core.ctx.table_lost_frames == 1


def test_only_unique_completed_negatives_reach_loss_threshold():
    core = _core_in_state(
        State.YOLO_APPROACH,
        _obs(inference_completed=True, inference_seq=40, has_new_inference=True),
    )
    core.cfg.yolo_table_lost_to_search_frames = 3
    core.cfg.table_loss_hold_s = 10.0
    core.tick()

    for seq in (41, 42):
        core.ctx.last_table_obs = _obs(
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
            inference_completed=True,
            inference_seq=seq,
            has_new_inference=True,
            explicit_negative_detection=True,
        )
        decision = core.tick()
        assert decision.cmd.vx_mps > 0.0
        assert core.ctx.state == State.YOLO_APPROACH
        # Empty scheduler ticks do not count again.
        core.ctx.last_table_obs.has_new_inference = False
        core.ctx.last_table_obs.explicit_negative_detection = False
        core.tick()

    assert core.ctx.table_lost_frames == 2

    core.ctx.last_table_obs = _obs(
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
        inference_completed=True,
        inference_seq=43,
        has_new_inference=True,
        explicit_negative_detection=True,
    )
    threshold = core.tick()
    assert core.ctx.state == State.SEARCH_TABLE
    assert threshold.control_summary["control_source"] == "local_rotate_search"


def test_runtime_invariant_valid_approach_requires_forward_motion():
    obs = _obs(table_cx_norm=0.30, yolo_bbox_center_x_norm=0.65)
    core = _core_in_state(State.YOLO_APPROACH, obs)
    decision = core.controller.yolo_table_search_cmd(obs, mode="YOLO_APPROACH", control_source="yolo_track_forward")
    decision.cmd.vx_mps = 0.0
    checked = core._check_yolo_continuity_invariants(decision, obs)
    assert "YOLO_APPROACH_VALID_BBOX_BUT_NO_FORWARD" in checked.control_summary["control_invariant_violations"]


def test_runtime_invariant_rejects_no_inference_dropout_classification():
    obs = _obs(has_new_inference=False)
    core = _core_in_state(State.YOLO_APPROACH, obs)
    core.ctx.last_valid_yolo_obs = obs
    decision = core.controller.yolo_table_search_cmd(obs, mode="YOLO_APPROACH", control_source="yolo_track_forward")
    decision.control_summary.update({"has_new_inference": False, "control_source": "bbox_lost_hold"})
    checked = core._check_yolo_continuity_invariants(decision, obs)
    assert "NO_INFERENCE_GAP_MISCLASSIFIED_AS_DROPOUT" in checked.control_summary["control_invariant_violations"]


def test_runtime_invariant_detects_missing_recenter_correction():
    obs = _obs(table_cx_norm=0.30, yolo_bbox_center_x_norm=0.65)
    core = _core_in_state(State.YOLO_APPROACH, obs)
    decision = core.controller.yolo_table_search_cmd(obs, mode="YOLO_APPROACH", control_source="yolo_track_forward")
    decision.cmd.wz_radps = 0.0
    checked = core._check_yolo_continuity_invariants(decision, obs)
    assert "BBOX_RECENTER_CORRECTION_MISSING" in checked.control_summary["control_invariant_violations"]


def test_runtime_invariant_detects_wrong_recenter_direction():
    obs = _obs(table_cx_norm=0.30, yolo_bbox_center_x_norm=0.65)
    core = _core_in_state(State.YOLO_APPROACH, obs)
    decision = core.controller.yolo_table_search_cmd(obs, mode="YOLO_APPROACH", control_source="yolo_track_forward")
    decision.cmd.wz_radps = -abs(decision.cmd.wz_radps)
    checked = core._check_yolo_continuity_invariants(decision, obs)
    assert "BBOX_RECENTER_DIRECTION_WRONG" in checked.control_summary["control_invariant_violations"]


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
