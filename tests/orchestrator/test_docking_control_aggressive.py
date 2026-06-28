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
        "near_lateral_vy_max_mps": 0.015,
        "min_forward_vx_mps": 0.04,
        "bbox_track_forward_vx_mps": 0.10,
        "bbox_track_forward_max_vx_mps": 0.20,
        "far_bbox_track_vx_mps": 0.20,
        "bbox_track_forward_max_wz_radps": 0.20,
        "table_target_dist_m": 0.30,
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
        final_forward_vx_max_mps=0.03,
        final_reverse_vx_max_mps=0.02,
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
