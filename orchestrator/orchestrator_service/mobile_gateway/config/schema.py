#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


_ORCH_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class GatewayEndpoint:
    transport: str = "tcp"
    host: str = "127.0.0.1"
    port: int = 0
    uds_path: str = ""
    send_mode: str = "oneshot"
    async_enabled: bool = False
    async_queue_size: int = 64
    async_drop_oldest: bool = True


@dataclass
class GatewayRuntimeConfig:
    project_root: str = field(default_factory=lambda: str(_ORCH_ROOT))
    repo_root: str = field(default_factory=lambda: str(_REPO_ROOT))
    log_dir: str = field(default_factory=lambda: str(_ORCH_ROOT / "logs"))
    log_file: str = field(default_factory=lambda: str(_ORCH_ROOT / "logs" / "mobile_gateway.log"))
    runs_dir: str = field(default_factory=lambda: str(_ORCH_ROOT / "runs"))
    pid_dir: str = field(default_factory=lambda: str(_ORCH_ROOT / "pids"))
    pid_file: str = field(default_factory=lambda: str(_ORCH_ROOT / "pids" / "mobile_gateway.pid"))
    stack_run_id: str = ""
    tick_hz: float = 10.0
    heartbeat_period_s: float = 1.0
    log_mode: str = "concise"
    log_enabled: bool = True
    status_stdout: bool = True
    stdin_enabled: bool = False


@dataclass
class GatewayBackendConfig:
    mode: str = "mock"  # mock / orchestrator_tcp / tcp_no_ack
    default_robot_id: str = "sc171_v2"
    default_confidence: float = 0.99
    mock_step_interval_s: float = 0.20
    enforce_single_flight: bool = True
    observer_enabled: bool = True
    observer_poll_interval_s: float = 0.25
    orchestrator_runs_dir: str = field(default_factory=lambda: str(_ORCH_ROOT / "runs"))
    state_blocks_path: str = ""
    stop_cooldown_s: float = 1.0


@dataclass
class MqttTopicConfig:
    cmd: str = "robot/v1/{robot_id}/mobile/cmd"
    ack: str = "robot/v1/{robot_id}/mobile/ack"
    status: str = "robot/v1/{robot_id}/mobile/status"
    heartbeat: str = "robot/v1/{robot_id}/heartbeat"


@dataclass
class MqttAdapterConfig:
    enabled: bool = False
    transport: str = "websocket"
    use_tls: bool = True
    broker_host: str = ""
    broker_port: int = 443
    websocket_path: str = "/mqtt"
    username: str = ""
    password: str = ""
    client_id: str = "sc171-v2-mobile-gateway"
    robot_id: str = "sc171_v2"
    qos: int = 1
    retain_status: bool = False
    keepalive_s: int = 60
    connect_timeout_s: float = 5.0
    topics: MqttTopicConfig = field(default_factory=MqttTopicConfig)


@dataclass
class MobileGatewayConfig:
    config_file: str = ""
    runtime: GatewayRuntimeConfig = field(default_factory=GatewayRuntimeConfig)
    backend: GatewayBackendConfig = field(default_factory=GatewayBackendConfig)
    mqtt: MqttAdapterConfig = field(default_factory=MqttAdapterConfig)
    command_in: GatewayEndpoint = field(default_factory=lambda: GatewayEndpoint(
        transport="tcp",
        host="127.0.0.1",
        port=9101,
        uds_path="/tmp/robot_stack/mobile_gateway_cmd.sock",
    ))
    status_out: GatewayEndpoint = field(default_factory=lambda: GatewayEndpoint(
        transport="disabled",
        host="127.0.0.1",
        port=9102,
        uds_path="/tmp/robot_stack/mobile_gateway_status.sock",
        send_mode="oneshot",
    ))
    orchestrator_task_cmd_out: GatewayEndpoint = field(default_factory=lambda: GatewayEndpoint(
        transport="tcp",
        host="127.0.0.1",
        port=9001,
        uds_path="/tmp/robot_stack/task_cmd.sock",
        send_mode="oneshot",
    ))
    orchestrator_task_ack_in: GatewayEndpoint = field(default_factory=lambda: GatewayEndpoint(
        transport="disabled",
        host="127.0.0.1",
        port=9103,
        uds_path="/tmp/robot_stack/mobile_gateway_ack.sock",
    ))
