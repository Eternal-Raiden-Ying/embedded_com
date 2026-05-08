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
    transport: str = "tcp"
    host: str = "127.0.0.1"
    port: int = 0
    uds_path: str = ""
    send_mode: str = "persistent"
    async_enabled: bool = False
    async_queue_size: int = 64
    async_drop_oldest: bool = True


@dataclass
class SerialConfig:
    port: str = "/dev/ttyHS1"
    baudrate: int = 115200
    timeout_s: float = 0.10
    dry_run: bool = False
    readback_enabled: bool = True
    dry_run_echo_stdout: bool = True
    dry_run_echo_on_change_only: bool = True
    dry_run_echo_summary_period_s: float = 5.0
    dry_run_quiet_idle_stop: bool = True
    uart_lowfreq_period_s: float = 5.0


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
    log_mode: str = "concise"
    log_enabled: bool = True
    debug: bool = False
    state_block_period_s: float = 1.0
    heartbeat_period_s: float = 1.0
    stage_params_file: str = ""
    car_cmd_params_file: str = ""
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

    table_found_frames_to_approach: int = 2
    table_lost_frames_to_reacquire: int = 4
    table_loss_hold_s: float = 1.20
    approach_min_dwell_s: float = 0.80

    coarse_align_frames_to_advance: int = 2
    coarse_align_done_rad: float = 0.10
    final_lock_frames_to_arrive: int = 3
    final_lock_yaw_tol_rad: float = 0.04
    final_lock_dist_tol_m: float = 0.03
    final_lock_lateral_tol_m: float = 0.03
    edge_settle_s: float = 0.80
    dock_retry_limit: int = 2
    dock_retry_backoff_s: float = 0.60

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
    enable_pick_pipeline: bool = False


@dataclass
class CarMotionConfig:
    search_table_wz_norm: float = 0.22
    fallback_align_turn_norm_min: float = 0.10
    fallback_align_turn_norm_max: float = 0.45
    fallback_forward_vx_norm_min: float = 0.06
    fallback_forward_vx_norm_max: float = 0.28
    fallback_dead_zone_x: float = 0.10
    fallback_spin_only_x_th: float = 0.82
    fallback_forward_align_exp: float = 2.0

    return_turn_norm_min: float = 0.20
    return_turn_norm_max: float = 0.75
    return_vx_norm_min: float = 0.10
    return_vx_norm_max: float = 0.45

    edge_slide_vy_norm: float = 0.14
    edge_slide_dist_kp_norm_per_m: float = 1.20
    edge_slide_yaw_kp_norm_per_rad: float = 1.20
    edge_slide_max_vx_norm: float = 0.10
    edge_slide_max_wz_norm: float = 0.12
    edge_slide_weak_vy_norm: float = 0.05
    leave_edge_vx_norm: float = -0.12
    relocate_turn_wz_norm: float = 0.28
    avoid_turn_norm: float = 0.38
    avoid_reverse_vx_norm: float = 0.12

    cmd_hold_ms: int = 150
    send_period_ms: int = 100
    max_vx_norm: float = 1.0
    max_vy_norm: float = 1.0
    max_wz_norm: float = 1.0
    stop_on_state_enter: bool = False
    mode_line_on_change: bool = True
    mode_line_every_cmd: bool = False
    serial_float_digits: int = 3


@dataclass
class OrchestratorConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    serial: SerialConfig = field(default_factory=SerialConfig)
    control: ControlThresholds = field(default_factory=ControlThresholds)
    car: CarMotionConfig = field(default_factory=CarMotionConfig)
    docking: DockingControlConfig = field(default_factory=DockingControlConfig)
    task_cmd_in: SocketEndpoint = field(default_factory=lambda: SocketEndpoint(
        transport="tcp",
        host="127.0.0.1",
        port=9001,
        uds_path="/tmp/robot_stack/task_cmd.sock",
    ))
    task_ack_out: SocketEndpoint = field(default_factory=lambda: SocketEndpoint(
        transport="tcp",
        host="127.0.0.1",
        port=9012,
        uds_path="/tmp/robot_stack/task_ack.sock",
        send_mode="oneshot",
    ))
    vision_obs_in: SocketEndpoint = field(default_factory=lambda: SocketEndpoint(
        transport="tcp",
        host="127.0.0.1",
        port=9002,
        uds_path="/tmp/robot_stack/vision_obs.sock",
    ))
    vision_req_out: SocketEndpoint = field(default_factory=lambda: SocketEndpoint(
        transport="tcp",
        host="127.0.0.1",
        port=9003,
        uds_path="/tmp/robot_stack/vision_req.sock",
        send_mode="oneshot",
        async_enabled=True,
    ))
    tts_event_out: SocketEndpoint = field(default_factory=lambda: SocketEndpoint(
        transport="disabled",
        host="127.0.0.1",
        port=9011,
        uds_path="/tmp/robot_stack/tts_event.sock",
        async_enabled=True,
    ))
    frozen_targets: Dict[str, List[str]] = field(default_factory=lambda: {
        "cup": ["水杯", "杯子", "杯", "马克杯", "玻璃杯"],
        "bottle": ["瓶子", "水瓶", "饮料瓶"],
        "phone": ["手机", "电话"],
        "remote": ["遥控器", "遥控"],
        "medicine_box": ["药盒", "药箱", "药"],
        "keys": ["钥匙", "钥匙串"],
        "apple": ["苹果"],
        "banana": ["香蕉"],
        "book": ["书", "书本"],
        "wallet": ["钱包"],
    })
