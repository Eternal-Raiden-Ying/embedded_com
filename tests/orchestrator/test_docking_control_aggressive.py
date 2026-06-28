from orchestrator.orchestrator_service.runtime.context import RuntimeContext, State
from orchestrator.orchestrator_service.runtime.motion_arbiter import MotionIntent, arbitrate_table_docking_motion


def _base_summary(**extra):
    summary = {
        "lateral_enabled": True,
        "distance_scaled_lateral_enabled": True,
        "lateral_kp": 0.10,
        "lateral_deadband_norm": 0.025,
        "lateral_distance_ref_m": 0.80,
        "lateral_distance_scale_min": 1.0,
        "lateral_distance_scale_max": 4.0,
        "far_lateral_vy_max_mps": 0.15,
        "mid_lateral_vy_max_mps": 0.08,
        "near_lateral_vy_max_mps": 0.030,
        "min_forward_vx_mps": 0.04,
        "bbox_track_forward_vx_mps": 0.10,
        "bbox_track_forward_max_vx_mps": 0.20,
        "far_bbox_track_vx_mps": 0.20,
        "bbox_track_forward_max_wz_radps": 0.20,
        "table_target_dist_m": 0.30,
        "final_servo_enter_p10_m": 0.45,
        "roi_final_stop_p10_m": 0.40,
        "roi_final_slow_p10_m": 0.50,
        "roi_final_probe_vx_mps": 0.004,
        "roi_final_missing_probe_vx_mps": 0.002,
        "roi_final_missing_hold_s": 0.8,
        "depth_envelope_stop_p10_m": 0.35,
        "depth_envelope_slow_p10_m": 0.50,
        "depth_envelope_mid_p10_m": 0.70,
        "depth_envelope_slow_vx_mps": 0.006,
        "depth_envelope_mid_vx_mps": 0.015,
        "lateral_priority_mid_error_norm": 0.10,
        "lateral_priority_large_error_norm": 0.18,
        "lateral_priority_mid_vx_cap_mps": 0.08,
        "lateral_priority_vx_cap_mps": 0.04,
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

    left = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        intent,
        _base_summary(**{**common, "bbox_center_error": -0.10}, table_roi_depth_p10=1.60),
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
        final_dist_deadband_m=0.04,
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
            final_dist_deadband_m=0.04,
            final_forward_vx_max_mps=0.006,
        ),
    )
    assert result.summary["obs_target_dist_m"] == 0.50
    assert result.summary["table_target_dist_m"] == 0.30
    assert abs(result.summary["measured_dist_m"] - 0.40) < 1e-9
    assert abs(result.summary["final_dist_err_m"] - 0.10) < 1e-9
    assert result.final_vx > 0.0


def test_near_p10_enters_final_roi_stop_instead_of_near_forward():
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
    assert result.summary["docking_action"] == "FINAL_LOCKED_STOP"
    assert result.summary["final_roi_mode_latched"] is True
    assert result.summary["final_roi_reason"] == "roi_p10_stop"
    assert result.final_vy == 0.0
    assert result.final_wz == 0.0
    assert result.final_vx == 0.0


def test_depth_envelope_and_lateral_priority_caps_forward_speed_but_keeps_vy():
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
    assert 0.0 < mid_error.final_vx <= 0.08
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
    assert 0.0 < large_error.final_vx <= 0.04
    assert large_error.final_vy != 0.0


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
    assert enter.final_vx <= 0.004
    assert enter.final_vy == 0.0
    assert enter.final_wz == 0.0

    stopped = arbitrate_table_docking_motion(
        ctx,
        None,
        intent,
        _base_summary(
            control_phase="EDGE_GUIDED_APPROACH",
            final_roi_mode_latched=True,
            edge_found=True,
            edge_valid=True,
            usable_for_approach=True,
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
    assert 0.0 < probe.final_vx <= 0.004
    assert probe.final_vy == 0.0
    assert probe.final_wz == 0.0
    assert probe.summary["docking_action"] == "FINAL_LOCKED_STOP"

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
    assert 0.0 < missing_short.final_vx <= 0.002
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
    assert missing_long.summary["docking_action"] == "FINAL_LOCKED_STOP"
