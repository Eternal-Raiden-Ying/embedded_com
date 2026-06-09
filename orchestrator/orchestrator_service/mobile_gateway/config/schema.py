#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..protocol import MQTT_TOPIC_ACK, MQTT_TOPIC_CMD, MQTT_TOPIC_HEARTBEAT, MQTT_TOPIC_STATUS, ROBOT_ID

_ORCH_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class GatewayEndpoint:
    transport: str = "uds"
    ipc_socket_path: str = ""
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
    mode: str = "production"
    log_level: str = "INFO"
    tick_hz: float = 10.0
    heartbeat_period_s: float = 1.0
    heartbeat_log_interval_s: float = 30.0
    suppress_heartbeat_success_log: bool = True
    enable_raw_mqtt_debug: bool = False
    enable_legacy_command_compat: bool = True
    cmd_dedup_cache_size: int = 64
    log_mode: str = "concise"
    log_enabled: bool = True
    status_stdout: bool = True
    stdin_enabled: bool = False


@dataclass
class GatewayBackendConfig:
    mode: str = "mock"  # mock / orchestrator_tcp / tcp_no_ack
    default_robot_id: str = ROBOT_ID
    default_confidence: float = 0.99
    mock_step_interval_s: float = 0.20
    enforce_single_flight: bool = True
    observer_enabled: bool = True
    observer_poll_interval_s: float = 0.25
    orchestrator_runs_dir: str = field(default_factory=lambda: str(_ORCH_ROOT / "runs"))
    state_blocks_path: str = ""
    state_block_log_mode: str = "summary"
    state_block_log_period_s: float = 1.0
    stop_cooldown_s: float = 1.0


@dataclass
class MqttTopicConfig:
    cmd: str = MQTT_TOPIC_CMD
    ack: str = MQTT_TOPIC_ACK
    status: str = MQTT_TOPIC_STATUS
    heartbeat: str = MQTT_TOPIC_HEARTBEAT


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
    client_id: str = "sc171_car_01"
    robot_id: str = ROBOT_ID
    cmd_qos: int = 1
    ack_qos: int = 1
    status_qos: int = 0
    heartbeat_qos: int = 0
    retain_status: bool = False
    retain_heartbeat: bool = False
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
        transport="uds",
        ipc_socket_path="/tmp/robot_stack/mobile_gateway_cmd.sock",
    ))
    status_out: GatewayEndpoint = field(default_factory=lambda: GatewayEndpoint(
        transport="disabled",
        ipc_socket_path="/tmp/robot_stack/mobile_gateway_status.sock",
        send_mode="oneshot",
    ))
    orchestrator_task_cmd_out: GatewayEndpoint = field(default_factory=lambda: GatewayEndpoint(
        transport="uds",
        ipc_socket_path="/tmp/robot_stack/task_cmd.sock",
        send_mode="oneshot",
    ))
    orchestrator_task_ack_in: GatewayEndpoint = field(default_factory=lambda: GatewayEndpoint(
        transport="disabled",
        ipc_socket_path="/tmp/robot_stack/mobile_gateway_ack.sock",
    ))
