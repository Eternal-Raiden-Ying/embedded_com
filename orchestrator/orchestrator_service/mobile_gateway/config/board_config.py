#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from typing import Any, Dict, Iterable

from .schema import MobileGatewayConfig


def _env_first(names: Iterable[str]) -> str:
    for name in names:
        raw = os.getenv(name)
        if raw is not None:
            return str(raw).strip()
    return ""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_bool_any(names: Iterable[str], default: bool) -> bool:
    raw = _env_first(names)
    if raw == "":
        return bool(default)
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_float_any(names: Iterable[str], default: float) -> float:
    raw = _env_first(names)
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_int_any(names: Iterable[str], default: int) -> int:
    raw = _env_first(names)
    if raw == "":
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return str(raw).strip() if raw is not None else str(default)


def _env_str_any(names: Iterable[str], default: str) -> str:
    raw = _env_first(names)
    return raw if raw != "" else str(default)


def _load_config_dict(path: str) -> Dict[str, Any]:
    file_path = str(path or "").strip()
    if not file_path:
        return {}
    if file_path.lower().endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as fp:
            return dict(json.load(fp) or {})
    if file_path.lower().endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. "
                "Install with `pip install pyyaml` or use JSON config."
            ) from exc
        with open(file_path, "r", encoding="utf-8") as fp:
            return dict(yaml.safe_load(fp) or {})
    raise RuntimeError(f"unsupported config file format: {file_path}")


def _apply_config_file(cfg: MobileGatewayConfig, data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        return
    cfg.backend.default_robot_id = str(data.get("robot_id") or cfg.backend.default_robot_id)
    cfg.mqtt.robot_id = cfg.backend.default_robot_id
    cfg.backend.mode = str(data.get("backend") or cfg.backend.mode)

    mqtt = dict(data.get("mqtt") or {})
    topics = dict(mqtt.get("topics") or {})
    cfg.mqtt.enabled = bool(mqtt.get("enabled", cfg.mqtt.enabled))
    cfg.mqtt.transport = str(mqtt.get("transport") or cfg.mqtt.transport)
    cfg.mqtt.use_tls = bool(mqtt.get("use_tls", cfg.mqtt.use_tls))
    cfg.mqtt.broker_host = str(mqtt.get("broker_host") or cfg.mqtt.broker_host)
    cfg.mqtt.broker_port = int(mqtt.get("broker_port", cfg.mqtt.broker_port) or cfg.mqtt.broker_port)
    cfg.mqtt.websocket_path = str(mqtt.get("websocket_path") or cfg.mqtt.websocket_path)
    cfg.mqtt.username = str(mqtt.get("username") or cfg.mqtt.username)
    cfg.mqtt.password = str(mqtt.get("password") or cfg.mqtt.password)
    cfg.mqtt.client_id = str(mqtt.get("client_id") or cfg.mqtt.client_id)
    cfg.mqtt.topics.cmd = str(topics.get("cmd") or cfg.mqtt.topics.cmd)
    cfg.mqtt.topics.ack = str(topics.get("ack") or cfg.mqtt.topics.ack)
    cfg.mqtt.topics.status = str(topics.get("status") or cfg.mqtt.topics.status)
    cfg.mqtt.topics.heartbeat = str(topics.get("heartbeat") or cfg.mqtt.topics.heartbeat)

    orch = dict(data.get("orchestrator") or {})
    cfg.orchestrator_task_cmd_out.host = str(orch.get("task_cmd_host") or cfg.orchestrator_task_cmd_out.host)
    cfg.orchestrator_task_cmd_out.port = int(orch.get("task_cmd_port", cfg.orchestrator_task_cmd_out.port) or cfg.orchestrator_task_cmd_out.port)
    cfg.orchestrator_task_ack_in.host = str(orch.get("task_ack_host") or cfg.orchestrator_task_ack_in.host)
    cfg.orchestrator_task_ack_in.port = int(orch.get("task_ack_port", cfg.orchestrator_task_ack_in.port) or cfg.orchestrator_task_ack_in.port)
    cfg.backend.state_blocks_path = str(orch.get("state_blocks_path") or cfg.backend.state_blocks_path)

    gateway = dict(data.get("mobile_gateway") or {})
    cfg.command_in.host = str(gateway.get("cmd_in_host") or cfg.command_in.host)
    cfg.command_in.port = int(gateway.get("cmd_in_port", cfg.command_in.port) or cfg.command_in.port)
    cfg.status_out.host = str(gateway.get("status_out_host") or cfg.status_out.host)
    cfg.status_out.port = int(gateway.get("status_out_port", cfg.status_out.port) or cfg.status_out.port)


CONFIG = MobileGatewayConfig()

CONFIG.config_file = _env_str("MOBILE_GATEWAY_CONFIG_FILE", CONFIG.config_file)
if CONFIG.config_file:
    _apply_config_file(CONFIG, _load_config_dict(CONFIG.config_file))

CONFIG.runtime.project_root = _env_str("MOBILE_GATEWAY_PROJECT_ROOT", CONFIG.runtime.project_root)
CONFIG.runtime.repo_root = _env_str("MOBILE_GATEWAY_REPO_ROOT", CONFIG.runtime.repo_root)
CONFIG.runtime.log_dir = _env_str("MOBILE_GATEWAY_LOG_DIR", CONFIG.runtime.log_dir)
CONFIG.runtime.log_file = _env_str("MOBILE_GATEWAY_LOG_FILE", CONFIG.runtime.log_file)
CONFIG.runtime.runs_dir = _env_str("MOBILE_GATEWAY_RUNS_DIR", CONFIG.runtime.runs_dir)
CONFIG.runtime.pid_dir = _env_str("MOBILE_GATEWAY_PID_DIR", CONFIG.runtime.pid_dir)
CONFIG.runtime.pid_file = _env_str("MOBILE_GATEWAY_PID_FILE", CONFIG.runtime.pid_file)
CONFIG.runtime.stack_run_id = _env_str("STACK_RUN_ID", CONFIG.runtime.stack_run_id)
CONFIG.runtime.tick_hz = _env_float("MOBILE_GATEWAY_TICK_HZ", CONFIG.runtime.tick_hz)
CONFIG.runtime.heartbeat_period_s = _env_float("MOBILE_GATEWAY_HEARTBEAT_PERIOD_S", CONFIG.runtime.heartbeat_period_s)
CONFIG.runtime.log_mode = _env_str("MOBILE_GATEWAY_LOG_MODE", CONFIG.runtime.log_mode)
CONFIG.runtime.log_enabled = _env_bool("MOBILE_GATEWAY_LOG_ENABLED", CONFIG.runtime.log_enabled)
CONFIG.runtime.status_stdout = _env_bool("MOBILE_GATEWAY_STATUS_STDOUT", CONFIG.runtime.status_stdout)
CONFIG.runtime.stdin_enabled = _env_bool("MOBILE_GATEWAY_STDIN_ENABLED", CONFIG.runtime.stdin_enabled)

CONFIG.backend.mode = _env_str("MOBILE_GATEWAY_BACKEND", CONFIG.backend.mode)
CONFIG.backend.default_robot_id = _env_str("MOBILE_GATEWAY_ROBOT_ID", CONFIG.backend.default_robot_id)
CONFIG.backend.default_confidence = _env_float("MOBILE_GATEWAY_DEFAULT_CONFIDENCE", CONFIG.backend.default_confidence)
CONFIG.backend.mock_step_interval_s = _env_float("MOBILE_GATEWAY_MOCK_STEP_INTERVAL_S", CONFIG.backend.mock_step_interval_s)
CONFIG.backend.enforce_single_flight = _env_bool("MOBILE_GATEWAY_SINGLE_FLIGHT", CONFIG.backend.enforce_single_flight)
CONFIG.backend.observer_enabled = _env_bool("MOBILE_GATEWAY_OBSERVER_ENABLED", CONFIG.backend.observer_enabled)
CONFIG.backend.observer_poll_interval_s = _env_float("MOBILE_GATEWAY_OBSERVER_POLL_INTERVAL_S", CONFIG.backend.observer_poll_interval_s)
CONFIG.backend.orchestrator_runs_dir = _env_str("MOBILE_GATEWAY_ORCH_RUNS_DIR", CONFIG.backend.orchestrator_runs_dir)
CONFIG.backend.state_blocks_path = _env_str_any(
    ["MOBILE_GATEWAY_STATE_BLOCKS_PATH", "MOBILE_GATEWAY_ORCH_STATE_BLOCKS_PATH"],
    CONFIG.backend.state_blocks_path,
)
CONFIG.backend.stop_cooldown_s = _env_float("MOBILE_GATEWAY_STOP_COOLDOWN_S", CONFIG.backend.stop_cooldown_s)

CONFIG.command_in.transport = _env_str("MOBILE_GATEWAY_CMD_IN_TRANSPORT", CONFIG.command_in.transport)
CONFIG.command_in.host = _env_str("MOBILE_GATEWAY_CMD_IN_HOST", CONFIG.command_in.host)
CONFIG.command_in.port = _env_int("MOBILE_GATEWAY_CMD_IN_PORT", CONFIG.command_in.port)
CONFIG.command_in.uds_path = _env_str("MOBILE_GATEWAY_CMD_IN_UDS", CONFIG.command_in.uds_path)

CONFIG.status_out.transport = _env_str("MOBILE_GATEWAY_STATUS_OUT_TRANSPORT", CONFIG.status_out.transport)
CONFIG.status_out.host = _env_str("MOBILE_GATEWAY_STATUS_OUT_HOST", CONFIG.status_out.host)
CONFIG.status_out.port = _env_int("MOBILE_GATEWAY_STATUS_OUT_PORT", CONFIG.status_out.port)
CONFIG.status_out.uds_path = _env_str("MOBILE_GATEWAY_STATUS_OUT_UDS", CONFIG.status_out.uds_path)
CONFIG.status_out.send_mode = _env_str("MOBILE_GATEWAY_STATUS_OUT_SEND_MODE", CONFIG.status_out.send_mode)

CONFIG.orchestrator_task_cmd_out.transport = _env_str_any(
    ["MOBILE_GATEWAY_ORCH_TASK_CMD_TRANSPORT", "MOBILE_GATEWAY_TASK_CMD_OUT_TRANSPORT"],
    CONFIG.orchestrator_task_cmd_out.transport,
)
CONFIG.orchestrator_task_cmd_out.host = _env_str_any(
    ["MOBILE_GATEWAY_ORCH_TASK_CMD_HOST", "MOBILE_GATEWAY_TASK_CMD_OUT_HOST"],
    CONFIG.orchestrator_task_cmd_out.host,
)
CONFIG.orchestrator_task_cmd_out.port = _env_int_any(
    ["MOBILE_GATEWAY_ORCH_TASK_CMD_PORT", "MOBILE_GATEWAY_TASK_CMD_OUT_PORT"],
    CONFIG.orchestrator_task_cmd_out.port,
)
CONFIG.orchestrator_task_cmd_out.uds_path = _env_str_any(
    ["MOBILE_GATEWAY_ORCH_TASK_CMD_UDS", "MOBILE_GATEWAY_TASK_CMD_OUT_UDS"],
    CONFIG.orchestrator_task_cmd_out.uds_path,
)
CONFIG.orchestrator_task_cmd_out.send_mode = _env_str_any(
    ["MOBILE_GATEWAY_ORCH_TASK_CMD_SEND_MODE", "MOBILE_GATEWAY_TASK_CMD_OUT_SEND_MODE"],
    CONFIG.orchestrator_task_cmd_out.send_mode,
)

CONFIG.orchestrator_task_ack_in.transport = _env_str_any(
    ["MOBILE_GATEWAY_ORCH_TASK_ACK_TRANSPORT", "MOBILE_GATEWAY_TASK_ACK_IN_TRANSPORT"],
    CONFIG.orchestrator_task_ack_in.transport,
)
CONFIG.orchestrator_task_ack_in.host = _env_str_any(
    ["MOBILE_GATEWAY_ORCH_TASK_ACK_HOST", "MOBILE_GATEWAY_TASK_ACK_IN_HOST"],
    CONFIG.orchestrator_task_ack_in.host,
)
CONFIG.orchestrator_task_ack_in.port = _env_int_any(
    ["MOBILE_GATEWAY_ORCH_TASK_ACK_PORT", "MOBILE_GATEWAY_TASK_ACK_IN_PORT"],
    CONFIG.orchestrator_task_ack_in.port,
)
CONFIG.orchestrator_task_ack_in.uds_path = _env_str_any(
    ["MOBILE_GATEWAY_ORCH_TASK_ACK_UDS", "MOBILE_GATEWAY_TASK_ACK_IN_UDS"],
    CONFIG.orchestrator_task_ack_in.uds_path,
)

CONFIG.mqtt.enabled = _env_bool("MOBILE_GATEWAY_MQTT_ENABLED", CONFIG.mqtt.enabled)
CONFIG.mqtt.transport = _env_str("MOBILE_GATEWAY_MQTT_TRANSPORT", CONFIG.mqtt.transport)
CONFIG.mqtt.use_tls = _env_bool("MOBILE_GATEWAY_MQTT_USE_TLS", CONFIG.mqtt.use_tls)
CONFIG.mqtt.broker_host = _env_str("MOBILE_GATEWAY_MQTT_BROKER_HOST", CONFIG.mqtt.broker_host)
CONFIG.mqtt.broker_port = _env_int("MOBILE_GATEWAY_MQTT_BROKER_PORT", CONFIG.mqtt.broker_port)
CONFIG.mqtt.websocket_path = _env_str("MOBILE_GATEWAY_MQTT_WEBSOCKET_PATH", CONFIG.mqtt.websocket_path)
CONFIG.mqtt.username = _env_str("MOBILE_GATEWAY_MQTT_USERNAME", CONFIG.mqtt.username)
CONFIG.mqtt.password = _env_str("MOBILE_GATEWAY_MQTT_PASSWORD", CONFIG.mqtt.password)
CONFIG.mqtt.client_id = _env_str("MOBILE_GATEWAY_MQTT_CLIENT_ID", CONFIG.mqtt.client_id)
CONFIG.mqtt.robot_id = _env_str("MOBILE_GATEWAY_MQTT_ROBOT_ID", CONFIG.backend.default_robot_id)
CONFIG.mqtt.qos = _env_int("MOBILE_GATEWAY_MQTT_QOS", CONFIG.mqtt.qos)
CONFIG.mqtt.retain_status = _env_bool("MOBILE_GATEWAY_MQTT_RETAIN_STATUS", CONFIG.mqtt.retain_status)
CONFIG.mqtt.keepalive_s = _env_int("MOBILE_GATEWAY_MQTT_KEEPALIVE_S", CONFIG.mqtt.keepalive_s)
CONFIG.mqtt.connect_timeout_s = _env_float("MOBILE_GATEWAY_MQTT_CONNECT_TIMEOUT_S", CONFIG.mqtt.connect_timeout_s)
CONFIG.mqtt.topics.cmd = _env_str("MOBILE_GATEWAY_MQTT_CMD_TOPIC", CONFIG.mqtt.topics.cmd)
CONFIG.mqtt.topics.ack = _env_str("MOBILE_GATEWAY_MQTT_ACK_TOPIC", CONFIG.mqtt.topics.ack)
CONFIG.mqtt.topics.status = _env_str("MOBILE_GATEWAY_MQTT_STATUS_TOPIC", CONFIG.mqtt.topics.status)
CONFIG.mqtt.topics.heartbeat = _env_str("MOBILE_GATEWAY_MQTT_HEARTBEAT_TOPIC", CONFIG.mqtt.topics.heartbeat)
