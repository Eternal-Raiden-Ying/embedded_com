#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from ..control.types import DockingControlConfig


_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LOG_DIR = _DEFAULT_PROJECT_ROOT / "logs"
_DEFAULT_RUNS_DIR = _DEFAULT_PROJECT_ROOT / "runs"
_DEFAULT_PID_DIR = _DEFAULT_PROJECT_ROOT / "pids"


@dataclass
class SocketEndpoint:
    transport: str = "uds"
    ipc_socket_path: str = ""
    tcp_host: str = "127.0.0.1"
    tcp_port: int = 0
    send_mode: str = "persistent"
    async_enabled: bool = False
    async_queue_size: int = 64
    async_drop_oldest: bool = True

    @property
    def uds_path(self) -> str:
        return self.ipc_socket_path

    @uds_path.setter
    def uds_path(self, value: str) -> None:
        self.ipc_socket_path = value

    @property
    def host(self) -> str:
        return self.tcp_host

    @host.setter
    def host(self, value: str) -> None:
        self.tcp_host = value

    @property
    def port(self) -> int:
        return self.tcp_port

    @port.setter
    def port(self, value: int) -> None:
        self.tcp_port = int(value)


@dataclass
class SerialConfig:
    port: str = "/dev/ttyHS1"
    baudrate: int = 115200
    timeout_s: float = 0.10
    dry_run: bool = False
    readback_enabled: bool = True
    dry_run_echo_stdout: bool = False
    dry_run_echo_on_change_only: bool = True
    dry_run_echo_summary_period_s: float = 5.0
    dry_run_quiet_idle_stop: bool = True
    uart_lowfreq_period_s: float = 5.0
    stm32_status_enabled: bool = False
    stm32_status_period_s: float = 1.0


@dataclass
class ArmSerialConfig:
    enabled: bool = True
    dry_run: bool = False
    port: str = "/dev/ttyUSB0"
    baudrate: int = 9600
    timeout_s: float = 0.10
    open_settle_s: float = 3.0
    bytesize: int = 8
    parity: str = "N"
    stopbits: int = 1
    rtscts: bool = False
    dsrdtr: bool = False
    set_dtr: bool = False
    set_rts: bool = False
    readback_enabled: bool = True
    response_timeout_s: float = 10.0


@dataclass
class MotionSmoothingConfig:
    enabled: bool = True
    bypass_on_safety_stop: bool = True
    vx_accel_mps2: float = 0.35
    vx_decel_mps2: float = 0.70
    vy_accel_mps2: float = 0.20
    vy_decel_mps2: float = 0.35
    wz_accel_radps2: float = 0.90
    wz_decel_radps2: float = 1.40
    urgent_wz_accel_radps2: float = 2.20
    urgent_wz_decel_radps2: float = 2.80
    dt_min_s: float = 0.02
    dt_max_s: float = 0.20
    reset_gap_s: float = 0.50


@dataclass
class RuntimeConfig:
    project_root: str = field(default_factory=lambda: str(_DEFAULT_PROJECT_ROOT))
    log_dir: str = field(default_factory=lambda: str(_DEFAULT_LOG_DIR))
    log_file: str = field(default_factory=lambda: str(_DEFAULT_LOG_DIR / "orchestrator.log"))
    runs_dir: str = field(default_factory=lambda: str(_DEFAULT_RUNS_DIR))
    pid_dir: str = field(default_factory=lambda: str(_DEFAULT_PID_DIR))
    pid_file: str = field(default_factory=lambda: str(_DEFAULT_PID_DIR / "orchestrator.pid"))
    stack_run_id: str = ""
    tick_hz: float = 10.0
    log_profile: str = "normal"
    resource_sample_interval_s: float = 1.0
    log_mode: str = "concise"
    log_enabled: bool = True
    debug: bool = False
    state_block_period_s: float = 1.0
    heartbeat_period_s: float = 1.0
    loaded_config_files: List[str] = field(default_factory=list)


@dataclass
class ControlThresholds:
    cmd_confidence_th: float = 0.60
    target_obs_max_age_s: float = 1.00
    table_obs_max_age_s: float = 1.00
    home_obs_max_age_s: float = 1.00

    search_table_timeout_s: float = 20.0
    approach_timeout_s: float = 14.0
    target_search_timeout_s: float = 10.0
    return_search_timeout_s: float = 15.0
    req_resend_period_s: float = 1.0
    stop_after_table_docking: bool = False
    locate_tts_ack_timeout_s: float = 8.0
    locate_ring_timeout_s: float = 30.0

    table_found_frames_to_approach: int = 2
    table_lost_frames_to_reacquire: int = 4
    table_loss_hold_s: float = 1.20
    approach_min_dwell_s: float = 0.80
    table_approach_warmup_s: float = 2.0
    table_approach_warmup_min_fresh_obs: int = 1

    coarse_align_frames_to_advance: int = 2
    coarse_align_done_rad: float = 0.25
    align_to_approach_yaw_rad: float = 0.08
    approach_to_align_yaw_rad: float = 0.16
    align_to_approach_stable_obs: int = 2
    approach_to_align_stable_obs: int = 2
    coarse_align_min_dwell_s: float = 0.80
    controlled_approach_min_dwell_s: float = 0.80
    final_lock_frames_to_arrive: int = 3
    final_lock_yaw_tol_rad: float = 0.25
    final_lock_dist_tol_m: float = 0.03  # Strict distance tolerance for declaring final lock stop condition
    final_lock_lateral_tol_m: float = 0.03
    table_edge_only_test: bool = False
    table_target_dist_m: float = 0.30  # Nominal target docking distance (stopped position relative to table edge)
    table_dist_tol_m: float = 0.05     # Allowable distance error tolerance during docking/alignment
    table_yaw_tol_rad: float = 0.13962634015954636  # Target yaw alignment error threshold (8 degrees in radians)
    table_stop_margin_m: float = 0.05  # Safety stop margin added to target distance in stop conditions checking
    table_settle_s: float = 0.50
    table_stable_frames: int = 5
    table_yolo_align_center_x_target: float = 0.50
    table_yolo_align_center_x_tol: float = 0.08
    # SEARCH_TABLE remains a coarse, latched scan until a real bbox direction is
    # confirmed by several distinct fresh observations.
    search_bbox_confirm_frames: int = 3
    search_direction_min_dwell_s: float = 0.80
    yolo_table_control_enable: bool = True
    yolo_table_conf_min: float = 0.25
    yolo_table_edge_stable_frames: int = 5
    edge_trusted_stable_frames: int = 5
    edge_trusted_min_conf: float = 0.60
    edge_trusted_max_residual: float = 0.05
    edge_trusted_min_support_count: int = 0
    edge_trusted_min_inlier_count: int = 0
    edge_trusted_min_x_span_m: float = 0.0
    edge_trusted_max_background_penalty: float = 0.0
    yolo_table_near_dist_m: float = 0.45
    yolo_table_lost_to_search_frames: int = 8
    no_table_bbox_timeout_s: float = 10.0
    edge_geometry_timeout_s: float = 10.0
    table_memory_timeout_sec: float = 3.0
    table_center_loss_hold_sec: float = 1.0
    rotate_search_timeout_s: float = 10.0
    rotate_require_edge_stable_frames: int = 5
    rotate_yaw_threshold_rad: float = 0.20
    yolo_edge_conflict_block_rotate: bool = True
    final_lock_required_ready_obs: int = 3
    final_lock_window_ms: int = 1000
    final_lock_max_consecutive_lost: int = 2
    final_lock_soft_stale_hold: bool = True
    table_max_micro_adjust: int = 3
    enable_final_lock: bool = True
    enable_micro_adjust: bool = False
    final_lock_enter_dist_th_m: float = 0.08
    final_lock_enter_yaw_th_rad: float = 0.10
    final_yaw_deadband_rad: float = 0.12
    final_lock_yaw_rad: float = 0.12
    final_yaw_realign_rad: float = 0.18
    final_yaw_stable_frames: int = 6
    final_yaw_align_min_duration_ms: int = 1000
    final_yaw_last_good_hold_s: float = 1.2
    edge_settle_s: float = 0.80
    at_table_edge_settle_s: float = 0.10
    target_search_fast_start_enable: bool = True
    target_fast_start_confirm_enable: bool = False
    target_prewarm_max_age_ms: int = 180
    target_prewarm_stable_obs: int = 2
    final_to_lateral_max_vx_mps: float = 0.02
    dock_retry_limit: int = 2
    dock_retry_backoff_s: float = 0.60

    # Depth safety thresholds
    near_slow_depth_m: float = 0.40
    near_stop_depth_m: float = 0.25
    near_slow_max_vx_mps: float = 0.030
    near_slow_max_vy_mps: float = 0.040
    near_slow_max_wz_radps: float = 0.04
    final_servo_enter_p10_m: float = 0.45
    final_enter_depth_threshold_m: float = 0.58
    final_enter_stable_count_required: int = 2
    final_handoff_on_yolo_lost_enable: bool = True
    final_handoff_recent_obs_max_age_s: float = 1.0
    final_handoff_min_recent_depth_m: float = 0.65
    final_fixed_roi_stop_threshold_m: float = 0.45
    final_fixed_roi_stop_stable_count_required: int = 3
    final_slow_probe_vx_mps: float = 0.05
    near_start_final_enable: bool = True
    near_start_final_depth_m: float = 0.58
    near_start_align_enable: bool = True
    near_start_align_timeout_s: float = 2.0
    remote_init_min_interval_s: float = 30.0
    remote_init_auto_enabled: bool = False
    edge_final_enter_margin_m: float = 0.06
    edge_final_stop_margin_m: float = 0.02
    close_range_enter_p10_m: float = 0.55
    final_probe_vx_mps: float = 0.020
    final_missing_probe_vx_mps: float = 0.004
    final_missing_probe_grace_s: float = 2.0
    final_missing_roi_continue_forward_enable: bool = True
    final_missing_roi_probe_vx_mps: float = 0.020
    final_entry_bridge_vx_mps: float = 0.030
    final_slow_stop_timeout_s: float = 12.0
    close_range_probe_vx_mps: float = 0.008
    close_range_missing_probe_vx_mps: float = 0.004
    roi_final_stop_p10_m: float = 0.42
    roi_final_slow_p10_m: float = 0.52
    roi_final_probe_vx_mps: float = 0.020
    roi_final_missing_probe_vx_mps: float = 0.004
    roi_final_missing_hold_s: float = 0.8
    depth_envelope_stop_p10_m: float = 0.30
    depth_envelope_slow_p10_m: float = 0.50
    depth_emergency_stop_p10_m: float = 0.20
    depth_envelope_mid_p10_m: float = 0.70
    depth_envelope_slow_vx_mps: float = 0.006
    depth_envelope_mid_vx_mps: float = 0.015
    yolo_approach_far_vx_mps: float = 0.50
    yolo_approach_mid_vx_mps: float = 0.20
    yolo_approach_near_vx_mps: float = 0.10
    yolo_approach_min_vx_mps: float = 0.05
    yolo_approach_depth_stat_for_envelope: str = "median"
    yolo_approach_use_p10_for_safety_only: bool = True
    builtin_bottle_grasp_enable: bool = True
    builtin_bottle_skip_remote: bool = True
    builtin_bottle_pose_line: str = "POSE_BOTTLE"
    builtin_bottle_pose_start_ack: str = "OK POSE_BOTTLE START"
    builtin_bottle_pose_done_ack: str = "OK POSE_BOTTLE DONE"
    builtin_bottle_pose_timeout_s: float = 15.0
    builtin_bottle_grab_enable: bool = True
    builtin_bottle_grab_line: str = "GRABBED"
    builtin_bottle_grab_done_ack: str = "OK GRABBED DONE"
    builtin_bottle_grab_timeout_s: float = 10.0
    builtin_apple_grasp_enable: bool = True
    builtin_apple_skip_remote: bool = True
    builtin_apple_pose_line: str = "POSE_APPLE"
    builtin_apple_pose_start_ack: str = "OK POSE_APPLE START"
    builtin_apple_pose_done_ack: str = "OK POSE_APPLE DONE"
    builtin_apple_pose_timeout_s: float = 15.0
    builtin_apple_grab_enable: bool = True
    builtin_apple_grab_line: str = "GRABBED"
    builtin_apple_grab_done_ack: str = "OK GRABBED DONE"
    builtin_apple_grab_timeout_s: float = 10.0
    post_grasp_fixed_flow_enable: bool = True
    post_grasp_turn_direction_sign: int = -1
    post_grasp_forward_vx_mps: float = 0.08
    post_grasp_forward_duration_s: float = 2.0
    post_grasp_stop_hold_s: float = 0.5
    post_grasp_rise_enable: bool = True
    post_grasp_rise_line: str = "POSE_RISE"
    post_grasp_rise_start_ack: str = "OK POSE_RISE START"
    post_grasp_rise_done_ack: str = "OK POSE_RISE DONE"
    post_grasp_rise_timeout_s: float = 10.0
    bbox_track_forward_enabled: bool = True
    min_forward_vx_mps: float = 0.040
    bbox_track_forward_vx_mps: float = 0.100
    bbox_track_forward_max_vx_mps: float = 0.200
    bbox_track_forward_center_band: float = 0.45
    far_bbox_track_vx_mps: float = 0.200
    bbox_track_forward_min_hold_ms: int = 800
    bbox_track_forward_max_wz_radps: float = 0.200
    edge_readiness_enabled: bool = True
    edge_readiness_enter_score: float = 0.65
    edge_readiness_exit_score: float = 0.35
    edge_readiness_rise: float = 0.15
    edge_readiness_decay: float = 0.10
    edge_readiness_min_inliers: int = 30
    edge_readiness_yaw_max_rad: float = 0.35
    edge_handoff_min_hold_ms: int = 800
    edge_handoff_forward_vx_mps: float = 0.080
    forward_commit_min_s: float = 1.8
    far_forward_commit_min_s: float = 2.0
    lateral_enabled: bool = True
    lateral_vy_max_mps: float = 0.180
    lateral_deadband_norm: float = 0.020
    lateral_kp: float = 0.300
    lateral_target_center_x_norm: float = 0.5
    lateral_owner_default: str = "none"
    distance_scaled_lateral_enabled: bool = True
    lateral_distance_ref_m: float = 0.50
    lateral_distance_scale_min: float = 0.80
    lateral_distance_scale_max: float = 2.00
    far_lateral_vy_max_mps: float = 0.180
    mid_lateral_vy_max_mps: float = 0.140
    near_lateral_vy_max_mps: float = 0.060
    lateral_priority_mid_error_norm: float = 0.99
    lateral_priority_large_error_norm: float = 0.99
    lateral_priority_mid_vx_cap_mps: float = 0.080
    lateral_priority_vx_cap_mps: float = 0.040
    edge_yaw_align_allow_lateral: bool = True
    edge_yaw_align_lateral_vy_max_mps: float = 0.080
    yaw_flip_hold_window_s: float = 0.80
    yaw_flip_count_limit: int = 2
    yaw_ambiguous_wz_cap: float = 0.0
    yaw_ambiguous_vy_boost: float = 1.5
    edge_yaw_control_enter_rad: float = 0.30
    edge_yaw_control_exit_rad: float = 0.12
    edge_yaw_reject_rad: float = 1.40
    edge_yaw_kp: float = 0.22
    edge_yaw_min_wz_radps: float = 0.08
    edge_yaw_max_wz_radps: float = 0.18
    final_dist_deadband_m: float = 0.030
    final_dist_kp: float = 0.080
    final_forward_vx_max_mps: float = 0.006
    final_reverse_vx_max_mps: float = 0.004
    final_reverse_confirm_frames: int = 3

    # Final lock stabilization thresholds
    final_lock_min_hold_ms: int = 800
    final_lock_lost_timeout_ms: int = 1000
    progress_window_ms: float = 15000.0
    min_progress_m: float = 0.010
    multi_table_enabled: bool = False

    target_fast_start_confirm_enable: bool = False
    edge_slide_min_duration_s: float = 1.50
    edge_slide_min_lateral_distance_m: float = 0.08
    edge_slide_min_frames_before_confirm: int = 10
    search_target_init_hold_s: float = 0.25
    target_found_frames_to_confirm: int = 3
    target_confirm_conf_th: float = 0.30
    target_confirm_dwell_s: float = 0.0
    target_confirm_min_s: float = 0.80
    target_confirm_timeout_s: float = 3.00
    target_confirm_lost_frames: int = 2
    target_confirm_lost_hold_s: float = 1.20
    target_confirm_min_bbox_area: float = 0.0
    target_confirm_window_s: float = 1.50
    target_confirm_found_ratio_th: float = 0.50
    target_lateral_align_enable: bool = True
    target_lateral_align_center_x_target: float = 0.40
    target_lateral_align_center_x_tol: float = 0.06
    # Legacy compatibility only; runtime derives deadband from 0.5 * target_lateral_align_center_x_tol.
    target_lateral_align_center_x_deadband: float = 0.03
    target_lateral_align_kp_vy: float = 0.05
    target_lateral_align_vy_min_mps: float = 0.008
    target_lateral_align_vy_max_mps: float = 0.025
    target_lateral_align_stable_frames: int = 3
    target_lateral_align_lost_hold_s: float = 0.80
    target_lateral_hold_enable: bool = True
    target_lateral_hold_s: float = 0.8
    target_lateral_lost_stop_s: float = 1.2
    target_lateral_min_vy_mps: float = 0.016
    target_lateral_align_timeout_s: float = 12.0
    post_grasp_place_enable: bool = True
    post_grasp_turn_enable: bool = True
    post_grasp_turn_wz_radps: float = 0.45
    post_grasp_turn_duration_s: float = 3.5
    post_grasp_turn_direction: str = "left"
    basket_search_timeout_s: float = 15.0
    basket_search_wz_radps: float = 0.25
    basket_align_center_x_target: float = 0.50
    basket_align_center_x_tol: float = 0.08
    basket_approach_vx_mps: float = 0.08
    basket_approach_vx_slow_mps: float = 0.035
    basket_stop_bbox_area_norm: float = 0.18
    basket_stop_bbox_height_norm: float = 0.38
    basket_stop_stable_count_required: int = 3
    basket_approach_timeout_s: float = 20.0
    place_pose_x_cm: float = 12.0
    place_pose_y_cm: float = 0.0
    place_pose_z_cm: float = 8.0
    place_pose_pitch_deg: float = 0.0
    place_pose_roll_deg: float = 0.0
    place_gripper_width: float = 80.0
    place_duration_ms: int = 800
    target_lock_conf_th: float = 0.40
    target_lock_found_ratio_th: float = 0.60
    target_lock_settle_s: float = 0.50
    target_lock_stable_s: float = 1.20
    target_lock_center_jitter_th: float = 0.08
    target_lock_lost_hold_s: float = 1.50
    target_locked_freeze_after_s: float = 1.00
    freeze_settle_s: float = 0.60
    edge_slide_pause_s: float = 0.20
    edge_slide_segment_s: float = 1.20
    edge_slide_dist_tolerance_m: float = 0.05
    edge_slide_fallback_state: str = "FINAL_LOCK"
    edge_slide_pause_hold_s: float = 0.80
    edge_slide_dist_out_of_range_hold_s: float = 0.80
    edge_slide_max_relock_attempts: int = 3
    edge_slide_relock_failure_is_fatal: bool = True
    edge_slide_recover_timeout_s: float = 2.50
    edge_slide_direct_fallback_to_controlled_approach: bool = False
    table_edge_obs_max_age_ms: int = 500
    table_obs_stale_soft_ms: int = 300
    table_obs_stale_stop_ms: int = 500
    table_obs_stale_hard_ms: int = 800
    table_step_mode_enable: bool = False
    table_step_burst_ms: int = 150
    table_step_hold_until_new_obs: bool = True
    edge_follow_log_period_ms: int = 500
    edge_follow_min_edge_conf: float = 0.60
    edge_follow_min_edge_conf_table_edge_perception: float = 0.60
    edge_follow_min_edge_conf_track_local: float = 0.20
    edge_follow_weak_edge_conf_track_local: float = 0.15
    edge_follow_strong_edge_conf_track_local: float = 0.35
    edge_follow_low_conf_hold_s: float = 2.00
    edge_follow_low_conf_exit_s: float = 3.00
    edge_follow_recover_conf_th: float = 0.25
    edge_identity_yaw_mismatch_rad: float = 0.15
    edge_identity_dist_mismatch_m: float = 0.04
    edge_follow_stale_fallback_state: str = "FINAL_LOCK"
    edge_follow_stale_hold_s: float = 1.20
    edge_follow_track_local_edge_update_hz: float = 5.0
    edge_handoff_min_s: float = 0.50
    edge_handoff_max_s: float = 1.00
    edge_handoff_samples: int = 3

    edge_relocate_enabled: bool = True
    max_edge_transitions_per_task: int = 3
    leave_edge_backoff_s: float = 0.80
    relocate_turn_s: float = 1.10
    reacquire_timeout_s: float = 8.0
    next_table_dwell_s: float = 1.50

    tag_lost_frames_to_search: int = 4
    return_lost_hold_s: float = 1.00
    return_min_dwell_s: float = 0.60
    return_done_distance_m: float = 0.35
    tag_arrived_frames_to_stop: int = 2

    avoid_clear_frames_to_resume: int = 2
    avoid_timeout_s: float = 4.0
    avoid_retry_limit: int = 3

    done_hold_s: float = 1.20
    error_recovery_hold_s: float = 1.20

    car_timeout_to_stop: bool = True
    car_fault_to_fail: bool = True
    car_estop_to_stop: bool = True
    post_stop_ignore_s: float = 0.80
    vision_req_fail_to_stop: bool = True
    vision_req_fail_threshold: int = 2
    keep_vision_alive_after_task: bool = True
    task_done_shutdown_vision: bool = False
    enable_pick_pipeline: bool = False
    assume_grasp_success_for_test: bool = False
    target_gripper_widths: Dict[str, float] = field(default_factory=lambda: {
        "苹果": 50.0,
        "猕猴桃": 65.0,
        "瓶子": 70.0,
        "apple": 80.0,
        "bottle": 80.0,
    })


@dataclass
class CarMotionConfig:
    grasp_reposition_speed_cm_s: float = 10.0
    pre_arm_stop_settle_ms: int = 150
    grasp_pose_time_ms: int = 800
    search_table_wz_radps: float = 0.10
    fallback_align_turn_wz_min_radps: float = 0.10
    fallback_align_turn_wz_max_radps: float = 0.45
    fallback_forward_vx_mps_min: float = 0.06
    fallback_forward_vx_mps_max: float = 0.28
    fallback_dead_zone_x: float = 0.10
    fallback_spin_only_x_th: float = 0.82
    fallback_forward_align_exp: float = 2.0

    table_fov_soft_th: float = 0.25
    table_fov_hard_th: float = 0.40
    table_view_memory_ttl_s: float = 0.80
    table_coarse_align_vx_max_mps: float = 0.000
    table_coarse_align_vy_min_mps: float = 0.000
    table_coarse_align_vy_max_mps: float = 0.000
    table_coarse_align_wz_min_radps: float = 0.080
    table_coarse_align_wz_max_radps: float = 0.150
    table_controlled_vx_min_mps: float = 0.020
    table_controlled_vx_max_mps: float = 0.035
    table_controlled_vy_min_mps: float = 0.000
    table_controlled_vy_max_mps: float = 0.000
    table_controlled_wz_min_radps: float = 0.000
    table_controlled_wz_max_radps: float = 0.120
    table_approach_safe_vx_mps: float = 0.020
    table_approach_max_vx_mps: float = 0.035
    table_approach_yaw_deadband_rad: float = 0.08
    table_approach_yaw_realign_rad: float = 0.16
    table_edge_hard_rotate_only_yaw_rad: float = 1.40
    table_edge_hard_yaw_rotate_only_frames: int = 3
    table_edge_hard_yaw_rotate_only_ms: int = 350
    table_perception_warmup_s: float = 1.0
    table_approach_allow_wz: bool = True
    table_approach_allow_vy: bool = False
    table_pose_missing_safe_vx_mps: float = 0.040
    table_pose_missing_max_hold_s: float = 3.0
    table_final_lock_vx_min_mps: float = 0.000
    table_final_lock_vx_max_mps: float = 0.008
    table_final_lock_vy_min_mps: float = 0.006
    table_final_lock_vy_max_mps: float = 0.012
    table_final_lock_wz_min_radps: float = 0.010
    table_final_lock_wz_max_radps: float = 0.025
    table_vx_deadband_mps: float = 0.004
    table_vy_deadband_mps: float = 0.001
    table_wz_deadband_radps: float = 0.006
    table_stage_a_wz_radps: float = 0.04
    table_stage_b_vx_max_mps: float = 0.03
    table_stage_c_vx_max_mps: float = 0.03
    table_stage_c_vx_min_mps: float = 0.0
    table_min_forward_dist_err_m: float = 0.07
    table_vx_mps_min: float = 0.040
    table_vx_mps_max: float = 0.100
    table_vx_kp_mps_per_m: float = 0.30
    table_yaw_slow_th_rad: float = 0.12
    table_yaw_stop_th_rad: float = 0.45
    table_near_dist_err_th_m: float = 0.10
    table_vy_max_mps: float = 0.067
    table_wz_view_max_radps: float = 0.05
    table_wz_plane_max_radps: float = 0.06
    table_dist_kp_mps_per_m: float = 0.12
    yolo_table_yaw_gain: float = 0.20
    yolo_table_max_wz_radps: float = 0.06
    yolo_table_forward_vx_mps: float = 0.015
    table_view_wz_kp: float = 0.18
    table_view_vy_kp: float = 0.04
    table_view_recover_vy_mps: float = 0.008
    table_view_recover_wz_radps: float = 0.04
    table_plane_yaw_kp_radps_per_rad: float = 0.60
    table_view_wz_sign: float = -1.0
    table_view_vy_sign: float = -1.0
    table_plane_yaw_sign: float = -1.0
    table_vx_slew_per_s: float = 0.12
    table_vy_slew_per_s: float = 0.06
    table_wz_slew_per_s: float = 0.18

    return_turn_wz_min_radps: float = 0.20
    return_turn_wz_max_radps: float = 0.75
    return_vx_mps_min: float = 0.10
    return_vx_mps_max: float = 0.45

    edge_slide_vy_mps: float = 0.14
    edge_slide_dist_kp_mps_per_m: float = 1.20
    edge_slide_yaw_kp_radps_per_rad: float = 1.20
    edge_slide_max_vx_mps: float = 0.10
    edge_slide_max_wz_radps: float = 0.12
    edge_slide_weak_vy_mps: float = 0.05
    leave_edge_vx_mps: float = -0.12
    relocate_turn_wz_radps: float = 0.28
    avoid_turn_wz_radps: float = 0.38
    avoid_reverse_vx_mps: float = 0.12

    cmd_hold_ms: int = 150
    send_period_ms: int = 100
    uart_keepalive_hz: float = 10.0
    min_uart_keepalive_hz: float = 7.0
    motion_hold_ms: int = 400
    hard_stale_stop_ms: int = 800
    soft_stale_hold_enable: bool = True
    max_vx_mps: float = 1.0
    max_vy_mps: float = 1.0
    max_wz_radps: float = 1.0
    stm32_wheel_speed_limit: int = 100
    stm32_vx_scale: float = 100.0
    stm32_vy_scale: float = 100.0
    stm32_wz_scale: float = 100.0
    jog_forward_speed: float = 0.02
    jog_turn_speed: float = 0.05
    jog_duration_ms: int = 100
    stop_on_state_enter: bool = False
    mode_line_on_change: bool = True
    mode_line_every_cmd: bool = False
    serial_float_digits: int = 3


@dataclass
class TtsFeedbackConfig:
    """Policy for structured Phone-TTS task feedback."""
    enabled: bool = True
    verbosity: str = "detailed"
    min_interval_s: float = 1.5
    default_ttl_s: float = 8.0
    dedupe_window_s: float = 30.0
    detailed_progress_enabled: bool = True


@dataclass
class OrchestratorConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    serial: SerialConfig = field(default_factory=SerialConfig)
    arm_serial: ArmSerialConfig = field(default_factory=ArmSerialConfig)
    motion_smoothing: MotionSmoothingConfig = field(default_factory=MotionSmoothingConfig)
    control: ControlThresholds = field(default_factory=ControlThresholds)
    car: CarMotionConfig = field(default_factory=CarMotionConfig)
    docking: DockingControlConfig = field(default_factory=DockingControlConfig)
    task_cmd_in: SocketEndpoint = field(default_factory=lambda: SocketEndpoint(
        transport="tcp",
        ipc_socket_path="/tmp/robot_stack/task_cmd.sock",
        tcp_port=19101,
    ))
    task_ack_out: SocketEndpoint = field(default_factory=lambda: SocketEndpoint(
        transport="tcp",
        ipc_socket_path="/tmp/robot_stack/task_ack.sock",
        tcp_port=19102,
        send_mode="oneshot",
    ))
    vision_obs_in: SocketEndpoint = field(default_factory=lambda: SocketEndpoint(
        transport="tcp",
        ipc_socket_path="/tmp/robot_stack/vision_obs.sock",
        tcp_port=19103,
    ))
    vision_req_out: SocketEndpoint = field(default_factory=lambda: SocketEndpoint(
        transport="tcp",
        ipc_socket_path="/tmp/robot_stack/vision_req.sock",
        tcp_port=19104,
        send_mode="oneshot",
        async_enabled=True,
    ))
    tts_feedback: TtsFeedbackConfig = field(default_factory=TtsFeedbackConfig)
    tts_event_out: SocketEndpoint = field(default_factory=lambda: SocketEndpoint(
        transport="disabled",
        ipc_socket_path="/tmp/robot_stack/tts_event.sock",
        async_enabled=True,
    ))
    # Playback ACKs are forwarded transparently by Mobile Gateway when enabled.
    tts_playback_in: SocketEndpoint = field(default_factory=lambda: SocketEndpoint(
        transport="disabled",
        ipc_socket_path="/tmp/robot_stack/orchestrator_tts_playback.sock",
        async_enabled=True,
    ))
    frozen_targets: Dict[str, List[str]] = field(default_factory=dict)  # legacy input ignored; catalog is authoritative
