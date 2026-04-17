#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Dict, List

from ..control.types import DockingControlConfig


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
    project_root: str = "/home/aidlux/2026/orchestrator"
    log_dir: str = "/home/aidlux/2026/orchestrator/logs"
    log_file: str = "/home/aidlux/2026/orchestrator/logs/orchestrator.log"
    runs_dir: str = "/home/aidlux/2026/orchestrator/runs"
    pid_dir: str = "/home/aidlux/2026/orchestrator/pids"
    pid_file: str = "/home/aidlux/2026/orchestrator/pids/orchestrator.pid"
    stack_run_id: str = ""
    tick_hz: float = 10.0
    log_mode: str = "concise"
    log_enabled: bool = True
    debug: bool = False
    state_block_period_s: float = 1.0
    heartbeat_period_s: float = 1.0


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
    target_found_frames_to_confirm: int = 2
    target_confirm_lost_frames: int = 2
    target_lock_settle_s: float = 0.50
    freeze_settle_s: float = 0.60
    edge_slide_pause_s: float = 0.20
    edge_slide_segment_s: float = 1.20

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
    leave_edge_vx_norm: float = -0.12
    relocate_turn_wz_norm: float = 0.28
    avoid_turn_norm: float = 0.38
    avoid_reverse_vx_norm: float = 0.12

    cmd_hold_ms: int = 150
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
