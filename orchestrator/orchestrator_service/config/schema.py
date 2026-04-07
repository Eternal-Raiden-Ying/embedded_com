#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Dict, List


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
    home_obs_max_age_s: float = 1.00
    search_timeout_s: float = 20.0
    return_search_timeout_s: float = 15.0
    req_resend_period_s: float = 1.0
    found_frames_to_approach: int = 2
    initial_found_frames_to_search: int = 1
    reacquire_found_frames_to_search: int = 2
    arrived_frames_to_stop: int = 2
    lost_frames_to_search: int = 6
    tag_found_frames_to_track: int = 2
    tag_arrived_frames_to_stop: int = 2
    tag_lost_frames_to_search: int = 4
    search_lost_hold_s: float = 1.20
    return_lost_hold_s: float = 1.00
    search_min_dwell_s: float = 0.80
    return_min_dwell_s: float = 0.60
    dead_zone_x: float = 0.10
    dead_zone_yaw: float = 0.10
    align_turn_threshold: float = 0.18
    stop_size_norm: float = 0.45
    return_stop_distance_m: float = 0.35
    car_timeout_to_stop: bool = True
    car_fault_to_fail: bool = True
    car_estop_to_stop: bool = True
    post_stop_ignore_s: float = 0.80
    vision_req_fail_to_stop: bool = True
    vision_req_fail_threshold: int = 2


@dataclass
class CarMotionConfig:
    search_turn_norm_min: float = 0.12
    search_turn_norm_max: float = 0.55
    search_vx_norm_min: float = 0.06
    search_vx_norm_max: float = 0.30
    search_spin_only_x_th: float = 0.82
    search_forward_align_exp: float = 2.0
    return_turn_norm_min: float = 0.20
    return_turn_norm_max: float = 0.75
    return_vx_norm_min: float = 0.10
    return_vx_norm_max: float = 0.45
    mode_line_on_change: bool = True
    mode_line_every_cmd: bool = False
    serial_float_digits: int = 3


@dataclass
class OrchestratorConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    serial: SerialConfig = field(default_factory=SerialConfig)
    control: ControlThresholds = field(default_factory=ControlThresholds)
    car: CarMotionConfig = field(default_factory=CarMotionConfig)
    task_cmd_in: SocketEndpoint = field(default_factory=lambda: SocketEndpoint(
        transport="tcp", host="127.0.0.1", port=9001, uds_path="/tmp/robot_stack/task_cmd.sock",
    ))
    task_ack_out: SocketEndpoint = field(default_factory=lambda: SocketEndpoint(
        transport="tcp", host="127.0.0.1", port=9012, uds_path="/tmp/robot_stack/task_ack.sock", send_mode="oneshot",
    ))
    vision_obs_in: SocketEndpoint = field(default_factory=lambda: SocketEndpoint(
        transport="tcp", host="127.0.0.1", port=9002, uds_path="/tmp/robot_stack/vision_obs.sock",
    ))
    vision_req_out: SocketEndpoint = field(default_factory=lambda: SocketEndpoint(
        transport="tcp", host="127.0.0.1", port=9003, uds_path="/tmp/robot_stack/vision_req.sock", send_mode="oneshot", async_enabled=True,
    ))
    tts_event_out: SocketEndpoint = field(default_factory=lambda: SocketEndpoint(
        transport="disabled", host="127.0.0.1", port=9011, uds_path="/tmp/robot_stack/tts_event.sock", async_enabled=True,
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
