from orchestrator.orchestrator_service.runtime.context import RuntimeContext, State
from orchestrator.orchestrator_service.runtime.motion_arbiter import MotionIntent, arbitrate_table_docking_motion


def _base_summary(**extra):
    summary = {
        "lateral_enabled": True,
        "distance_scaled_lateral_enabled": True,
        "lateral_kp": 0.30,
        "lateral_deadband_norm": 0.020,
        "lateral_distance_ref_m": 0.50,
        "lateral_distance_scale_min": 0.80,
        "lateral_distance_scale_max": 2.0,
        "far_lateral_vy_max_mps": 0.18,
        "mid_lateral_vy_max_mps": 0.14,
        "near_lateral_vy_max_mps": 0.060,
        "min_forward_vx_mps": 0.04,
        "bbox_track_forward_vx_mps": 0.10,
        "bbox_track_forward_max_vx_mps": 0.20,
        "far_bbox_track_vx_mps": 0.20,
        "bbox_track_forward_max_wz_radps": 0.20,
        "table_target_dist_m": 0.30,
        "final_servo_enter_p10_m": 0.45,
        "edge_final_enter_margin_m": 0.06,
        "edge_final_stop_margin_m": 0.02,
        "close_range_enter_p10_m": 0.55,
        "final_probe_vx_mps": 0.008,
        "final_missing_probe_vx_mps": 0.004,
        "close_range_probe_vx_mps": 0.008,
        "close_range_missing_probe_vx_mps": 0.004,
        "roi_final_stop_p10_m": 0.42,
        "roi_final_slow_p10_m": 0.52,
        "roi_final_probe_vx_mps": 0.008,
        "roi_final_missing_probe_vx_mps": 0.004,
        "roi_final_missing_hold_s": 0.8,
        "depth_envelope_stop_p10_m": 0.35,
        "depth_envelope_slow_p10_m": 0.50,
        "depth_envelope_mid_p10_m": 0.70,
        "depth_envelope_slow_vx_mps": 0.006,
        "depth_envelope_mid_vx_mps": 0.015,
        "lateral_priority_mid_error_norm": 0.99,
        "lateral_priority_large_error_norm": 0.99,
        "lateral_priority_mid_vx_cap_mps": 0.08,
        "lateral_priority_vx_cap_mps": 0.04,
        "edge_yaw_align_allow_lateral": True,
        "edge_yaw_align_lateral_vy_max_mps": 0.08,
        "edge_yaw_control_enter_rad": 0.30,
        "edge_yaw_control_exit_rad": 0.12,
        "edge_yaw_reject_rad": 1.40,
        "edge_yaw_kp": 0.22,
        "edge_yaw_min_wz_radps": 0.08,
        "edge_yaw_max_wz_radps": 0.18,
        "table_plane_yaw_sign": -1.0,
    }
    summary.update(extra)
    return summary


def test_distance_scaled_lateral_direction_and_distance_gain():
    intent = MotionIntent("yolo_track_forward", desired_vx=0.10, forward_allowed_by_behavior=True)
    common = {
        "control_phase": "BBOX_ACQUIRE",
        "bbox_center_valid": True,
        "bbox_center_error": 0.10,
        "yolo_forward_allowed": True,
        "table_roi_depth_valid": True,
    }
    mid = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        intent,
        _base_summary(**common, table_roi_depth_p10=0.80),
    )
    far = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        intent,
        _base_summary(**common, table_roi_depth_p10=1.60),
    )
    assert mid.final_vy < 0.0
    assert abs(far.final_vy) > abs(mid.final_vy)

    strong_mid = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        intent,
        _base_summary(**{**common, "bbox_center_error": 0.20}, table_roi_depth_p10=0.75),
    )
    assert strong_mid.final_vy < 0.0
    assert abs(strong_mid.final_vy) >= 0.06
    assert abs(strong_mid.summary["lateral_distance_scale"] - 1.5) < 1e-9

    stronger_mid = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        intent,
        _base_summary(**{**common, "bbox_center_error": 0.30}, table_roi_depth_p10=0.75),
    )
    assert stronger_mid.final_vy < 0.0
    assert abs(stronger_mid.final_vy) >= 0.10
    assert abs(stronger_mid.final_vy) <= stronger_mid.summary["far_lateral_vy_max_mps"]

    left = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        intent,
        _base_summary(**{**common, "bbox_center_error": -0.25}, table_roi_depth_p10=1.60),
    )
    assert left.final_vy > 0.0


def test_yaw_flip_ambiguous_prioritizes_lateral_over_recovery_rotate():
    ctx = RuntimeContext(state=State.YOLO_APPROACH)
    intent = MotionIntent(
        "edge_guided_forward",
        desired_vx=0.08,
        desired_wz=0.08,
        yaw_owner="edge",
        forward_allowed_by_behavior=True,
    )
    summary = _base_summary(
        control_phase="EDGE_GUIDED_APPROACH",
        edge_found=True,
        edge_valid=True,
        usable_for_approach=True,
        edge_readiness_score=1.0,
        edge_readiness_enter_score=0.65,
        bbox_center_valid=True,
        bbox_center_error=0.12,
        table_roi_depth_valid=True,
        table_roi_depth_p10=1.20,
        yaw_flip_hold_window_s=0.8,
        yaw_flip_count_limit=2,
        yaw_ambiguous_wz_cap=0.0,
        yaw_ambiguous_vy_boost=1.5,
    )
    for yaw in (0.06, -0.06, 0.06):
        result = arbitrate_table_docking_motion(ctx, None, intent, dict(summary, yaw_err_rad=yaw))
    assert result.reason == "edge_yaw_ambiguous_lateral_priority"
    assert result.summary["docking_action"] == "EDGE_APPROACH_FORWARD"
    assert result.final_wz == 0.0
    assert result.final_vy != 0.0


def test_final_reverse_requires_confirmation_and_locked_stops():
    ctx = RuntimeContext(state=State.FINAL_SLOW_STOP)
    ctx.final_depth_latched = True
    intent = MotionIntent("final_hold")
    summary = _base_summary(
        control_phase="DEPTH_FINAL_STOP",
        final_depth_latched=True,
        dist_err_m=-0.10,
        final_dist_deadband_m=0.03,
        final_dist_kp=0.08,
        final_forward_vx_max_mps=0.006,
        final_reverse_vx_max_mps=0.004,
        final_reverse_confirm_frames=3,
    )
    first = arbitrate_table_docking_motion(ctx, None, intent, summary)
    second = arbitrate_table_docking_motion(ctx, None, intent, summary)
    third = arbitrate_table_docking_motion(ctx, None, intent, summary)
    assert first.final_vx == 0.0
    assert second.final_vx == 0.0
    assert third.final_vx < 0.0

    locked = arbitrate_table_docking_motion(ctx, None, intent, dict(summary, final_locked=True))
    assert locked.final_vx == 0.0
    assert locked.final_vy == 0.0
    assert locked.final_wz == 0.0


def test_final_distance_error_uses_observation_target_then_control_target():
    ctx = RuntimeContext(state=State.FINAL_SLOW_STOP)
    ctx.final_depth_latched = True
    result = arbitrate_table_docking_motion(
        ctx,
        None,
        MotionIntent("final_hold"),
        _base_summary(
            control_phase="DEPTH_FINAL_STOP",
            final_depth_latched=True,
            target_dist_m=0.50,
            dist_err_m=-0.10,
            final_dist_deadband_m=0.03,
            final_forward_vx_max_mps=0.006,
        ),
    )
    assert result.summary["obs_target_dist_m"] == 0.50
    assert result.summary["table_target_dist_m"] == 0.30
    assert abs(result.summary["measured_dist_m"] - 0.40) < 1e-9
    assert abs(result.summary["final_dist_err_m"] - 0.10) < 1e-9
    assert result.final_vx > 0.0


def test_near_p10_enters_close_range_but_edge_remains_priority_when_usable():
    result = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("edge_guided_forward", desired_vx=0.08, desired_wz=0.03, yaw_owner="edge", forward_allowed_by_behavior=True),
        _base_summary(
            control_phase="EDGE_GUIDED_APPROACH",
            near_table_latched=True,
            edge_found=True,
            edge_valid=True,
            usable_for_approach=True,
            table_roi_depth_valid=True,
            table_roi_depth_p10=0.39,
            dist_err_m=0.12,
        ),
    )
    assert result.summary["docking_action"] == "FINAL_SLOW_PROBE"
    assert result.summary["close_range_latched"] is True
    assert result.summary["final_roi_mode_latched"] is False
    assert result.summary["measured_dist_source"] == "edge"
    assert result.final_vy == 0.0
    assert result.final_wz == 0.0
    assert 0.0 <= result.final_vx <= 0.006


def test_depth_envelope_caps_near_speed_but_lateral_priority_keeps_far_vx_fast():
    intent = MotionIntent("yolo_track_forward", desired_vx=0.20, desired_wz=0.0, yaw_owner="bbox", forward_allowed_by_behavior=True)
    mid_depth = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        intent,
        _base_summary(
            control_phase="BBOX_ACQUIRE",
            bbox_center_valid=True,
            bbox_center_error=0.02,
            yolo_forward_allowed=True,
            table_roi_depth_valid=True,
            table_roi_depth_p10=0.55,
        ),
    )
    assert mid_depth.final_vx <= 0.015

    slow_depth = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        intent,
        _base_summary(
            control_phase="BBOX_ACQUIRE",
            bbox_center_valid=True,
            bbox_center_error=0.02,
            yolo_forward_allowed=True,
            table_roi_depth_valid=True,
            table_roi_depth_p10=0.42,
        ),
    )
    assert slow_depth.final_vx <= 0.006

    mid_error = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        intent,
        _base_summary(
            control_phase="BBOX_ACQUIRE",
            bbox_center_valid=True,
            bbox_center_error=0.12,
            yolo_forward_allowed=True,
            table_roi_depth_valid=True,
            table_roi_depth_p10=1.30,
        ),
    )
    assert mid_error.final_vx >= 0.19
    assert mid_error.final_vy != 0.0

    large_error = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        intent,
        _base_summary(
            control_phase="BBOX_ACQUIRE",
            bbox_center_valid=True,
            bbox_center_error=0.20,
            yolo_forward_allowed=True,
            table_roi_depth_valid=True,
            table_roi_depth_p10=1.30,
            bbox_track_forward_center_band=0.45,
        ),
    )
    assert large_error.final_vx >= 0.19
    assert abs(large_error.final_vy) >= 0.10


def test_final_roi_latch_stops_on_p10_and_disables_yaw():
    ctx = RuntimeContext(state=State.YOLO_APPROACH)
    intent = MotionIntent("edge_guided_forward", desired_vx=0.08, desired_wz=0.15, yaw_owner="edge", forward_allowed_by_behavior=True)
    enter = arbitrate_table_docking_motion(
        ctx,
        None,
        intent,
        _base_summary(
            control_phase="EDGE_GUIDED_APPROACH",
            near_table_latched=True,
            table_roi_depth_valid=True,
            table_roi_depth_p10=0.44,
            yaw_err_rad=0.30,
        ),
    )
    assert enter.summary["final_roi_mode_latched"] is True
    assert enter.final_vx <= 0.008
    assert enter.final_vy == 0.0
    assert enter.final_wz == 0.0

    stopped = arbitrate_table_docking_motion(
        ctx,
        None,
        intent,
        _base_summary(
            control_phase="EDGE_GUIDED_APPROACH",
            final_roi_mode_latched=True,
            edge_found=False,
            edge_valid=False,
            usable_for_approach=False,
            table_roi_depth_valid=True,
            table_roi_depth_p10=0.39,
            yaw_err_rad=0.30,
        ),
    )
    assert stopped.summary["final_roi_reason"] == "roi_p10_stop_confirming"
    stopped = arbitrate_table_docking_motion(
        ctx,
        None,
        intent,
        _base_summary(
            control_phase="EDGE_GUIDED_APPROACH",
            final_roi_mode_latched=True,
            edge_found=False,
            edge_valid=False,
            usable_for_approach=False,
            table_roi_depth_valid=True,
            table_roi_depth_p10=0.39,
            yaw_err_rad=0.30,
        ),
    )
    assert stopped.summary["docking_action"] == "FINAL_LOCKED_STOP"
    assert stopped.summary["final_locked"] is True
    assert stopped.summary["final_roi_reason"] == "roi_p10_stop"
    assert stopped.final_vx == 0.0
    assert stopped.final_vy == 0.0
    assert stopped.final_wz == 0.0


def test_final_roi_probe_and_roi_missing_hold_do_not_return_to_near_or_rotate():
    ctx = RuntimeContext(state=State.YOLO_APPROACH)
    ctx.final_roi_mode_latched = True
    ctx.final_roi_mode_since_mono = 100.0
    ctx.final_roi_last_valid_mono = 100.0
    intent = MotionIntent("edge_guided_forward", desired_vx=0.08, desired_wz=0.15, yaw_owner="edge", forward_allowed_by_behavior=True)

    probe = arbitrate_table_docking_motion(
        ctx,
        None,
        intent,
        _base_summary(
            control_phase="EDGE_GUIDED_APPROACH",
            final_roi_mode_latched=True,
            near_slow_max_vx_mps=0.03,
            edge_found=True,
            edge_valid=True,
            usable_for_approach=True,
            table_roi_depth_valid=True,
            table_roi_depth_p10=0.46,
            yaw_err_rad=0.40,
        ),
    )
    assert probe.summary["final_roi_reason"] == "roi_p10_slow_probe"
    assert 0.0 < probe.final_vx <= 0.008
    assert probe.final_vy == 0.0
    assert probe.final_wz == 0.0
    assert probe.summary["docking_action"] == "FINAL_SLOW_PROBE"
    assert probe.summary["docking_action"] != "FINAL_LOCKED_STOP"

    ctx.final_roi_last_valid_mono = 0.0
    ctx.final_roi_mode_since_mono = 0.0
    missing_short = arbitrate_table_docking_motion(
        ctx,
        None,
        intent,
        _base_summary(
            control_phase="EDGE_GUIDED_APPROACH",
            final_roi_mode_latched=True,
            table_roi_depth_valid=False,
            yaw_err_rad=0.40,
        ),
    )
    assert missing_short.summary["final_roi_reason"] == "roi_missing_slow_probe"
    assert 0.0 < missing_short.final_vx <= 0.004
    assert missing_short.final_vy == 0.0
    assert missing_short.final_wz == 0.0
    assert missing_short.summary["docking_action"] not in {"NEAR_EDGE_FORWARD", "SEARCH_ROTATE", "CONTROL_RECOVERY_ROTATE"}

    ctx.final_roi_last_valid_mono = -10.0
    ctx.final_roi_mode_since_mono = -10.0
    missing_long = arbitrate_table_docking_motion(
        ctx,
        None,
        intent,
        _base_summary(
            control_phase="EDGE_GUIDED_APPROACH",
            final_roi_mode_latched=True,
            table_roi_depth_valid=False,
            yaw_err_rad=0.40,
        ),
    )
    assert missing_long.summary["final_roi_reason"] == "roi_missing_hold"
    assert missing_long.final_vx == 0.0
    assert missing_long.final_vy == 0.0
    assert missing_long.final_wz == 0.0
    assert missing_long.summary["docking_action"] == "CLOSE_RANGE_PROBE"


def test_edge_final_latch_and_stop_use_measured_minus_control_target():
    ctx = RuntimeContext(state=State.YOLO_APPROACH)
    intent = MotionIntent("edge_guided_forward", desired_vx=0.20, desired_wz=0.15, yaw_owner="edge", forward_allowed_by_behavior=True)
    enter = arbitrate_table_docking_motion(
        ctx,
        None,
        intent,
        _base_summary(
            control_phase="EDGE_GUIDED_APPROACH",
            edge_found=True,
            edge_valid=True,
            usable_for_approach=True,
            target_dist_m=0.50,
            dist_err_m=-0.15,
            yaw_err_rad=0.30,
        ),
    )
    assert enter.summary["final_edge_mode_latched"] is True
    assert enter.summary["close_range_latched"] is True
    assert abs(enter.summary["measured_dist_m"] - 0.35) < 1e-9
    assert abs(enter.summary["final_dist_err_m"] - 0.05) < 1e-9
    assert enter.summary["docking_action"] == "FINAL_SLOW_PROBE"
    assert 0.0 <= enter.final_vx <= 0.006
    assert enter.final_vy == 0.0
    assert enter.final_wz == 0.0

    stop = arbitrate_table_docking_motion(
        ctx,
        None,
        intent,
        _base_summary(
            control_phase="EDGE_GUIDED_APPROACH",
            edge_found=True,
            edge_valid=True,
            usable_for_approach=True,
            target_dist_m=0.50,
            dist_err_m=-0.181,
            yaw_err_rad=0.30,
        ),
    )
    assert stop.reason == "edge_final_dist_stop_confirming"
    stop = arbitrate_table_docking_motion(
        ctx,
        None,
        intent,
        _base_summary(
            control_phase="EDGE_GUIDED_APPROACH",
            edge_found=True,
            edge_valid=True,
            usable_for_approach=True,
            target_dist_m=0.50,
            dist_err_m=-0.181,
            yaw_err_rad=0.30,
        ),
    )
    assert stop.summary["final_edge_mode_latched"] is True
    assert stop.summary["final_locked"] is True
    assert stop.reason == "edge_final_dist_stop"
    assert stop.final_vx == 0.0
    assert stop.final_vy == 0.0
    assert stop.final_wz == 0.0


def test_close_range_latch_blocks_bbox_forward_and_recovery_rotate():
    intent = MotionIntent("yolo_track_forward", desired_vx=0.20, desired_wz=0.15, yaw_owner="bbox", forward_allowed_by_behavior=True, rotate_allowed_by_behavior=True)
    close = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        intent,
        _base_summary(
            control_phase="BBOX_ACQUIRE",
            bbox_center_valid=True,
            bbox_center_error=0.30,
            yolo_forward_allowed=True,
            table_roi_depth_valid=True,
            table_roi_depth_p10=0.54,
        ),
    )
    assert close.summary["close_range_latched"] is True
    assert close.summary["docking_action"] == "CLOSE_RANGE_PROBE"
    assert close.summary["docking_action"] not in {"BBOX_TRACK_FORWARD", "BBOX_REACQUIRE_ROTATE", "SEARCH_ROTATE", "CONTROL_RECOVERY_ROTATE"}
    assert 0.0 < close.final_vx <= 0.008
    assert close.final_vy == 0.0
    assert close.final_wz == 0.0

    latched = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        intent,
        _base_summary(
            control_phase="BBOX_ACQUIRE",
            close_range_latched=True,
            bbox_center_valid=True,
            bbox_center_error=0.40,
            yolo_forward_allowed=True,
            table_roi_depth_valid=False,
        ),
    )
    assert latched.summary["close_range_latched"] is True
    assert latched.summary["docking_action"] == "CLOSE_RANGE_PROBE"
    assert latched.summary["docking_action"] not in {"BBOX_REACQUIRE_ROTATE", "SEARCH_ROTATE", "CONTROL_RECOVERY_ROTATE"}
    assert 0.0 < latched.final_vx <= 0.004
    assert latched.final_vy == 0.0
    assert latched.final_wz == 0.0


def test_final_locked_hold_outranks_roi_missing_probe():
    result = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("edge_guided_forward", desired_vx=0.20, desired_wz=0.15, yaw_owner="edge", forward_allowed_by_behavior=True),
        _base_summary(
            control_phase="EDGE_GUIDED_APPROACH",
            final_locked=True,
            final_roi_mode_latched=True,
            close_range_latched=True,
            table_roi_depth_valid=False,
        ),
    )
    assert result.summary["docking_action"] == "FINAL_LOCKED_STOP"
    assert result.reason == "final_locked_hold"
    assert result.final_vx == 0.0
    assert result.final_vy == 0.0
    assert result.final_wz == 0.0


def test_non_final_edge_yaw_large_never_outputs_all_zero():
    result = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("edge_guided_forward", desired_vx=0.0, desired_wz=0.0, yaw_owner="edge", forward_allowed_by_behavior=True),
        _base_summary(
            control_phase="EDGE_GUIDED_APPROACH",
            edge_found=True,
            edge_valid=True,
            usable_for_approach=True,
            edge_readiness_score=1.0,
            edge_readiness_enter_score=0.65,
            yaw_err_rad=0.31,
            edge_yaw=0.31,
            edge_forward_rotate_only_yaw_rad=0.18,
            search_wz_radps=0.20,
            table_roi_depth_valid=True,
            table_roi_depth_p10=1.20,
        ),
    )
    assert result.summary["docking_action"] == "EDGE_APPROACH_FORWARD"
    assert result.summary["edge_yaw_control_active"] is True
    assert result.summary["yaw_owner"] == "edge"
    assert result.final_vx == 0.0
    assert result.final_vy == 0.0
    assert abs(result.final_wz) >= result.summary["edge_yaw_min_wz_radps"]
    assert result.final_wz < 0.0


def test_large_edge_yaw_takes_yaw_owner_before_close_range():
    result = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("yolo_track_forward", desired_vx=0.20, desired_wz=0.01, yaw_owner="bbox", forward_allowed_by_behavior=True),
        _base_summary(
            control_phase="BBOX_ACQUIRE",
            edge_found=True,
            edge_valid=True,
            usable_for_approach=True,
            usable_for_alignment=True,
            bbox_center_valid=True,
            bbox_center_error=0.08,
            yolo_forward_allowed=True,
            yaw_err_rad=0.60,
            edge_yaw=0.60,
            yaw_conflict=True,
            table_roi_depth_valid=True,
            table_roi_depth_p10=1.20,
        ),
    )

    assert result.summary["edge_yaw_control_active"] is True
    assert result.summary["yaw_owner"] == "edge"
    assert result.summary["yaw_source"] == "edge"
    assert abs(result.final_wz) >= result.summary["edge_yaw_min_wz_radps"]
    assert result.final_wz < 0.0
    assert result.final_vx == 0.0
    assert result.summary["lateral_owner"] == "bbox"
    assert result.final_vy < 0.0
    assert abs(result.final_vy) <= result.summary["edge_yaw_align_lateral_vy_max_mps"]


def test_edge_yaw_control_keeps_bbox_lateral_capped():
    result = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("edge_guided_forward", desired_vx=0.08, desired_wz=0.0, yaw_owner="bbox", forward_allowed_by_behavior=True),
        _base_summary(
            control_phase="EDGE_GUIDED_APPROACH",
            edge_found=True,
            edge_valid=True,
            usable_for_approach=True,
            bbox_center_valid=True,
            bbox_center_error=0.30,
            yolo_forward_allowed=True,
            yaw_err_rad=0.60,
            edge_yaw=0.60,
            table_roi_depth_valid=True,
            table_roi_depth_p10=1.60,
        ),
    )

    assert result.summary["edge_yaw_control_active"] is True
    assert result.summary["yaw_owner"] == "edge"
    assert result.summary["lateral_owner"] == "bbox"
    assert result.final_wz < 0.0
    assert result.final_vy < 0.0
    assert abs(result.final_vy) == result.summary["edge_yaw_align_lateral_vy_max_mps"]


def test_one_rad_edge_yaw_is_still_correction_not_reject():
    result = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("edge_guided_forward", desired_vx=0.08, desired_wz=0.0, yaw_owner="bbox", forward_allowed_by_behavior=True),
        _base_summary(
            control_phase="EDGE_GUIDED_APPROACH",
            edge_found=True,
            edge_valid=True,
            usable_for_approach=True,
            yaw_err_rad=1.00,
            edge_yaw=1.00,
            table_roi_depth_valid=True,
            table_roi_depth_p10=1.20,
        ),
    )

    assert result.summary["edge_yaw_control_active"] is True
    assert result.summary["edge_yaw_rejected_near_vertical"] is False
    assert result.summary["yaw_owner"] == "edge"
    assert abs(result.final_wz) >= result.summary["edge_yaw_min_wz_radps"]
    assert result.final_wz < 0.0


def test_large_negative_edge_yaw_uses_positive_wz_with_configured_sign():
    result = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("edge_guided_forward", desired_vx=0.08, desired_wz=0.0, yaw_owner="bbox", forward_allowed_by_behavior=True),
        _base_summary(
            control_phase="EDGE_GUIDED_APPROACH",
            edge_found=True,
            edge_valid=True,
            usable_for_approach=True,
            yaw_err_rad=-0.60,
            edge_yaw=-0.60,
            table_roi_depth_valid=True,
            table_roi_depth_p10=1.20,
        ),
    )

    assert result.summary["edge_yaw_control_active"] is True
    assert result.summary["edge_yaw_sign"] == -1.0
    assert result.final_wz > 0.0


def test_near_vertical_edge_yaw_is_marked_rejected_for_fallback():
    result = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("yolo_track_forward", desired_vx=0.10, desired_wz=0.02, yaw_owner="bbox", forward_allowed_by_behavior=True),
        _base_summary(
            control_phase="BBOX_ACQUIRE",
            edge_found=True,
            edge_valid=True,
            usable_for_approach=True,
            bbox_center_valid=True,
            bbox_center_error=0.02,
            yolo_forward_allowed=True,
            yaw_err_rad=1.45,
            edge_yaw=1.45,
            table_roi_depth_valid=True,
            table_roi_depth_p10=1.20,
        ),
    )

    assert result.summary["edge_yaw_rejected_near_vertical"] is True
    assert result.summary["edge_yaw_reject_reason"] == "edge_yaw_rejected_near_vertical"
    assert result.summary["edge_yaw_control_active"] is False


def test_close_range_still_disables_edge_yaw_correction():
    result = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("edge_guided_forward", desired_vx=0.08, desired_wz=0.15, yaw_owner="edge", forward_allowed_by_behavior=True),
        _base_summary(
            control_phase="EDGE_GUIDED_APPROACH",
            close_range_latched=True,
            edge_found=True,
            edge_valid=True,
            usable_for_approach=True,
            yaw_err_rad=0.60,
            edge_yaw=0.60,
            table_roi_depth_valid=False,
        ),
    )

    assert result.summary["close_range_latched"] is True
    assert result.summary["edge_yaw_control_active"] is False
    assert result.final_wz == 0.0
    assert result.summary["yaw_owner"] == "none"


def test_edge_missing_allows_bbox_yaw_fallback():
    result = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("yolo_track_forward", desired_vx=0.10, desired_wz=0.04, yaw_owner="bbox", forward_allowed_by_behavior=True),
        _base_summary(
            control_phase="BBOX_ACQUIRE",
            edge_found=False,
            edge_valid=False,
            bbox_center_valid=True,
            bbox_center_error=0.04,
            yolo_forward_allowed=True,
            table_roi_depth_valid=True,
            table_roi_depth_p10=1.20,
        ),
    )

    assert result.summary["edge_yaw_control_active"] is False
    assert result.summary["yaw_owner"] == "bbox"
    assert abs(result.final_wz) > 0.0
