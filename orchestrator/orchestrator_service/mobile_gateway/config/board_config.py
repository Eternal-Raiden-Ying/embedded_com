#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from typing import Any, Dict, Iterable

from ..protocol import MQTT_TOPIC_ACK, MQTT_TOPIC_CMD, MQTT_TOPIC_HEARTBEAT, MQTT_TOPIC_STATUS, ROBOT_ID
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
            with open(file_path, "r", encoding="utf-8") as fp:
                return _parse_simple_yaml(fp.read())
        with open(file_path, "r", encoding="utf-8") as fp:
            return dict(yaml.safe_load(fp) or {})
    raise RuntimeError(f"unsupported config file format: {file_path}")


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Parse a tiny YAML subset used by mobile_gateway configs.

    Supported:
    - nested mappings with spaces-only indentation
    - string / int / float / bool / empty string values
    - comments and blank lines
    """

    root: Dict[str, Any] = {}
    stack = [(-1, root)]

    def _scalar(raw: str) -> Any:
        value = str(raw).strip()
        if value == "":
            return ""
        if value in {'""', "''"}:
            return ""
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        try:
            if "." in value:
                return float(value)
            return int(value)
        except Exception:
            return value

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if value == "":
            new_dict: Dict[str, Any] = {}
            current[key] = new_dict
            stack.append((indent, new_dict))
        else:
            current[key] = _scalar(value)

    return root


def _apply_config_file(cfg: MobileGatewayConfig, data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        return
    cfg.backend.default_robot_id = str(data.get("robot_id") or cfg.backend.default_robot_id)
    cfg.mqtt.robot_id = ROBOT_ID
    cfg.backend.mode = str(data.get("backend") or cfg.backend.mode)

    runtime = dict(data.get("runtime") or {})
    cfg.runtime.mode = str(runtime.get("mode") or cfg.runtime.mode)
    cfg.runtime.log_level = str(runtime.get("log_level") or cfg.runtime.log_level)
    cfg.runtime.tick_hz = float(runtime.get("tick_hz", cfg.runtime.tick_hz) or cfg.runtime.tick_hz)
    cfg.runtime.heartbeat_period_s = float(
        runtime.get("heartbeat_period_s", cfg.runtime.heartbeat_period_s) or cfg.runtime.heartbeat_period_s
    )
    cfg.runtime.heartbeat_log_interval_s = float(
        runtime.get("heartbeat_log_interval_s", cfg.runtime.heartbeat_log_interval_s) or cfg.runtime.heartbeat_log_interval_s
    )
    cfg.runtime.suppress_heartbeat_success_log = bool(
        runtime.get("suppress_heartbeat_success_log", cfg.runtime.suppress_heartbeat_success_log)
    )
    cfg.runtime.enable_raw_mqtt_debug = bool(runtime.get("enable_raw_mqtt_debug", cfg.runtime.enable_raw_mqtt_debug))
    cfg.runtime.enable_legacy_command_compat = bool(
        runtime.get("enable_legacy_command_compat", cfg.runtime.enable_legacy_command_compat)
    )
    cfg.runtime.cmd_dedup_cache_size = int(
        runtime.get("cmd_dedup_cache_size", cfg.runtime.cmd_dedup_cache_size) or cfg.runtime.cmd_dedup_cache_size
    )
    cfg.runtime.log_mode = str(runtime.get("log_mode") or cfg.runtime.log_mode)
    cfg.runtime.log_enabled = bool(runtime.get("log_enabled", cfg.runtime.log_enabled))
    cfg.runtime.status_stdout = bool(runtime.get("status_stdout", cfg.runtime.status_stdout))
    cfg.runtime.stdin_enabled = bool(runtime.get("stdin_enabled", cfg.runtime.stdin_enabled))

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
    cfg.mqtt.cmd_qos = int(mqtt.get("cmd_qos", cfg.mqtt.cmd_qos) or cfg.mqtt.cmd_qos)
    cfg.mqtt.ack_qos = int(mqtt.get("ack_qos", cfg.mqtt.ack_qos) or cfg.mqtt.ack_qos)
    cfg.mqtt.status_qos = int(mqtt.get("status_qos", cfg.mqtt.status_qos) or cfg.mqtt.status_qos)
    cfg.mqtt.heartbeat_qos = int(mqtt.get("heartbeat_qos", cfg.mqtt.heartbeat_qos) or cfg.mqtt.heartbeat_qos)
    cfg.mqtt.retain_status = bool(mqtt.get("retain_status", cfg.mqtt.retain_status))
    cfg.mqtt.retain_heartbeat = bool(mqtt.get("retain_heartbeat", cfg.mqtt.retain_heartbeat))
    cfg.mqtt.topics.cmd = str(topics.get("cmd") or cfg.mqtt.topics.cmd)
    cfg.mqtt.topics.ack = str(topics.get("ack") or cfg.mqtt.topics.ack)
    cfg.mqtt.topics.status = str(topics.get("status") or cfg.mqtt.topics.status)
    cfg.mqtt.topics.heartbeat = str(topics.get("heartbeat") or cfg.mqtt.topics.heartbeat)

    orch = dict(data.get("orchestrator") or {})
    cfg.orchestrator_task_cmd_out.host = str(orch.get("task_cmd_host") or cfg.orchestrator_task_cmd_out.host)
    cfg.orchestrator_task_cmd_out.port = int(orch.get("task_cmd_port", cfg.orchestrator_task_cmd_out.port) or cfg.orchestrator_task_cmd_out.port)
    task_ack_transport = str(orch.get("task_ack_transport") or "").strip()
    if task_ack_transport:
        cfg.orchestrator_task_ack_in.transport = task_ack_transport
    elif "task_ack_host" in orch or "task_ack_port" in orch:
        cfg.orchestrator_task_ack_in.transport = "tcp"
    cfg.orchestrator_task_ack_in.host = str(orch.get("task_ack_host") or cfg.orchestrator_task_ack_in.host)
    cfg.orchestrator_task_ack_in.port = int(orch.get("task_ack_port", cfg.orchestrator_task_ack_in.port) or cfg.orchestrator_task_ack_in.port)
    cfg.backend.state_blocks_path = str(orch.get("state_blocks_path") or cfg.backend.state_blocks_path)

    gateway = dict(data.get("mobile_gateway") or {})
    cfg.command_in.host = str(gateway.get("cmd_in_host") or cfg.command_in.host)
    cfg.command_in.port = int(gateway.get("cmd_in_port", cfg.command_in.port) or cfg.command_in.port)
    cfg.status_out.host = str(gateway.get("status_out_host") or cfg.status_out.host)
    cfg.status_out.port = int(gateway.get("status_out_port", cfg.status_out.port) or cfg.status_out.port)


def build_config(config_file: str = "") -> MobileGatewayConfig:
    cfg = MobileGatewayConfig()

    cfg.config_file = str(config_file or _env_str("MOBILE_GATEWAY_CONFIG_FILE", cfg.config_file)).strip()
    if cfg.config_file:
        _apply_config_file(cfg, _load_config_dict(cfg.config_file))

    cfg.runtime.project_root = _env_str("MOBILE_GATEWAY_PROJECT_ROOT", cfg.runtime.project_root)
    cfg.runtime.repo_root = _env_str("MOBILE_GATEWAY_REPO_ROOT", cfg.runtime.repo_root)
    cfg.runtime.log_dir = _env_str("MOBILE_GATEWAY_LOG_DIR", cfg.runtime.log_dir)
    cfg.runtime.log_file = _env_str("MOBILE_GATEWAY_LOG_FILE", cfg.runtime.log_file)
    cfg.runtime.runs_dir = _env_str("MOBILE_GATEWAY_RUNS_DIR", cfg.runtime.runs_dir)
    cfg.runtime.pid_dir = _env_str("MOBILE_GATEWAY_PID_DIR", cfg.runtime.pid_dir)
    cfg.runtime.pid_file = _env_str("MOBILE_GATEWAY_PID_FILE", cfg.runtime.pid_file)
    cfg.runtime.stack_run_id = _env_str("STACK_RUN_ID", cfg.runtime.stack_run_id)
    cfg.runtime.mode = _env_str("MOBILE_GATEWAY_RUNTIME_MODE", cfg.runtime.mode)
    cfg.runtime.log_level = _env_str("MOBILE_GATEWAY_LOG_LEVEL", cfg.runtime.log_level)
    cfg.runtime.tick_hz = _env_float("MOBILE_GATEWAY_TICK_HZ", cfg.runtime.tick_hz)
    cfg.runtime.heartbeat_period_s = _env_float("MOBILE_GATEWAY_HEARTBEAT_PERIOD_S", cfg.runtime.heartbeat_period_s)
    cfg.runtime.heartbeat_log_interval_s = _env_float(
        "MOBILE_GATEWAY_HEARTBEAT_LOG_INTERVAL_S",
        cfg.runtime.heartbeat_log_interval_s,
    )
    cfg.runtime.suppress_heartbeat_success_log = _env_bool(
        "MOBILE_GATEWAY_SUPPRESS_HEARTBEAT_SUCCESS_LOG",
        cfg.runtime.suppress_heartbeat_success_log,
    )
    cfg.runtime.enable_raw_mqtt_debug = _env_bool(
        "MOBILE_GATEWAY_ENABLE_RAW_MQTT_DEBUG",
        cfg.runtime.enable_raw_mqtt_debug,
    )
    cfg.runtime.enable_legacy_command_compat = _env_bool(
        "MOBILE_GATEWAY_ENABLE_LEGACY_COMMAND_COMPAT",
        cfg.runtime.enable_legacy_command_compat,
    )
    cfg.runtime.cmd_dedup_cache_size = _env_int(
        "MOBILE_GATEWAY_CMD_DEDUP_CACHE_SIZE",
        cfg.runtime.cmd_dedup_cache_size,
    )
    cfg.runtime.log_mode = _env_str("MOBILE_GATEWAY_LOG_MODE", cfg.runtime.log_mode)
    cfg.runtime.log_enabled = _env_bool("MOBILE_GATEWAY_LOG_ENABLED", cfg.runtime.log_enabled)
    cfg.runtime.status_stdout = _env_bool("MOBILE_GATEWAY_STATUS_STDOUT", cfg.runtime.status_stdout)
    cfg.runtime.stdin_enabled = _env_bool("MOBILE_GATEWAY_STDIN_ENABLED", cfg.runtime.stdin_enabled)

    runtime_mode = str(cfg.runtime.mode or "production").strip().lower()
    if runtime_mode == "debug":
        if "MOBILE_GATEWAY_LOG_MODE" not in os.environ:
            cfg.runtime.log_mode = "full"
        if "MOBILE_GATEWAY_LOG_LEVEL" not in os.environ:
            cfg.runtime.log_level = "DEBUG"
        if "MOBILE_GATEWAY_SUPPRESS_HEARTBEAT_SUCCESS_LOG" not in os.environ:
            cfg.runtime.suppress_heartbeat_success_log = False
        if "MOBILE_GATEWAY_ENABLE_RAW_MQTT_DEBUG" not in os.environ:
            cfg.runtime.enable_raw_mqtt_debug = True
    else:
        if "MOBILE_GATEWAY_LOG_MODE" not in os.environ:
            cfg.runtime.log_mode = "concise"
        if "MOBILE_GATEWAY_LOG_LEVEL" not in os.environ:
            cfg.runtime.log_level = "INFO"

    cfg.backend.mode = _env_str("MOBILE_GATEWAY_BACKEND", cfg.backend.mode)
    cfg.backend.default_robot_id = _env_str("MOBILE_GATEWAY_ROBOT_ID", cfg.backend.default_robot_id)
    cfg.backend.default_confidence = _env_float("MOBILE_GATEWAY_DEFAULT_CONFIDENCE", cfg.backend.default_confidence)
    cfg.backend.mock_step_interval_s = _env_float("MOBILE_GATEWAY_MOCK_STEP_INTERVAL_S", cfg.backend.mock_step_interval_s)
    cfg.backend.enforce_single_flight = _env_bool("MOBILE_GATEWAY_SINGLE_FLIGHT", cfg.backend.enforce_single_flight)
    cfg.backend.observer_enabled = _env_bool("MOBILE_GATEWAY_OBSERVER_ENABLED", cfg.backend.observer_enabled)
    cfg.backend.observer_poll_interval_s = _env_float("MOBILE_GATEWAY_OBSERVER_POLL_INTERVAL_S", cfg.backend.observer_poll_interval_s)
    cfg.backend.orchestrator_runs_dir = _env_str("MOBILE_GATEWAY_ORCH_RUNS_DIR", cfg.backend.orchestrator_runs_dir)
    cfg.backend.state_blocks_path = _env_str_any(
        ["MOBILE_GATEWAY_STATE_BLOCKS_PATH", "MOBILE_GATEWAY_ORCH_STATE_BLOCKS_PATH"],
        cfg.backend.state_blocks_path,
    )
    cfg.backend.stop_cooldown_s = _env_float("MOBILE_GATEWAY_STOP_COOLDOWN_S", cfg.backend.stop_cooldown_s)

    cfg.command_in.transport = _env_str("MOBILE_GATEWAY_CMD_IN_TRANSPORT", cfg.command_in.transport)
    cfg.command_in.host = _env_str("MOBILE_GATEWAY_CMD_IN_HOST", cfg.command_in.host)
    cfg.command_in.port = _env_int("MOBILE_GATEWAY_CMD_IN_PORT", cfg.command_in.port)
    cfg.command_in.uds_path = _env_str("MOBILE_GATEWAY_CMD_IN_UDS", cfg.command_in.uds_path)

    cfg.status_out.transport = _env_str("MOBILE_GATEWAY_STATUS_OUT_TRANSPORT", cfg.status_out.transport)
    cfg.status_out.host = _env_str("MOBILE_GATEWAY_STATUS_OUT_HOST", cfg.status_out.host)
    cfg.status_out.port = _env_int("MOBILE_GATEWAY_STATUS_OUT_PORT", cfg.status_out.port)
    cfg.status_out.uds_path = _env_str("MOBILE_GATEWAY_STATUS_OUT_UDS", cfg.status_out.uds_path)
    cfg.status_out.send_mode = _env_str("MOBILE_GATEWAY_STATUS_OUT_SEND_MODE", cfg.status_out.send_mode)

    cfg.orchestrator_task_cmd_out.transport = _env_str_any(
        ["MOBILE_GATEWAY_ORCH_TASK_CMD_TRANSPORT", "MOBILE_GATEWAY_TASK_CMD_OUT_TRANSPORT"],
        cfg.orchestrator_task_cmd_out.transport,
    )
    cfg.orchestrator_task_cmd_out.host = _env_str_any(
        ["MOBILE_GATEWAY_ORCH_TASK_CMD_HOST", "MOBILE_GATEWAY_TASK_CMD_OUT_HOST"],
        cfg.orchestrator_task_cmd_out.host,
    )
    cfg.orchestrator_task_cmd_out.port = _env_int_any(
        ["MOBILE_GATEWAY_ORCH_TASK_CMD_PORT", "MOBILE_GATEWAY_TASK_CMD_OUT_PORT"],
        cfg.orchestrator_task_cmd_out.port,
    )
    cfg.orchestrator_task_cmd_out.uds_path = _env_str_any(
        ["MOBILE_GATEWAY_ORCH_TASK_CMD_UDS", "MOBILE_GATEWAY_TASK_CMD_OUT_UDS"],
        cfg.orchestrator_task_cmd_out.uds_path,
    )
    cfg.orchestrator_task_cmd_out.send_mode = _env_str_any(
        ["MOBILE_GATEWAY_ORCH_TASK_CMD_SEND_MODE", "MOBILE_GATEWAY_TASK_CMD_OUT_SEND_MODE"],
        cfg.orchestrator_task_cmd_out.send_mode,
    )

    cfg.orchestrator_task_ack_in.transport = _env_str_any(
        ["MOBILE_GATEWAY_ORCH_TASK_ACK_TRANSPORT", "MOBILE_GATEWAY_TASK_ACK_IN_TRANSPORT"],
        cfg.orchestrator_task_ack_in.transport,
    )
    cfg.orchestrator_task_ack_in.host = _env_str_any(
        ["MOBILE_GATEWAY_ORCH_TASK_ACK_HOST", "MOBILE_GATEWAY_TASK_ACK_IN_HOST"],
        cfg.orchestrator_task_ack_in.host,
    )
    cfg.orchestrator_task_ack_in.port = _env_int_any(
        ["MOBILE_GATEWAY_ORCH_TASK_ACK_PORT", "MOBILE_GATEWAY_TASK_ACK_IN_PORT"],
        cfg.orchestrator_task_ack_in.port,
    )
    cfg.orchestrator_task_ack_in.uds_path = _env_str_any(
        ["MOBILE_GATEWAY_ORCH_TASK_ACK_UDS", "MOBILE_GATEWAY_TASK_ACK_IN_UDS"],
        cfg.orchestrator_task_ack_in.uds_path,
    )

    if str(cfg.backend.mode).strip().lower() == "orchestrator_tcp" and str(cfg.orchestrator_task_ack_in.transport).strip().lower() == "disabled":
        cfg.orchestrator_task_ack_in.transport = "tcp"

    cfg.mqtt.enabled = _env_bool("MOBILE_GATEWAY_MQTT_ENABLED", cfg.mqtt.enabled)
    cfg.mqtt.transport = _env_str("MOBILE_GATEWAY_MQTT_TRANSPORT", cfg.mqtt.transport)
    cfg.mqtt.use_tls = _env_bool("MOBILE_GATEWAY_MQTT_USE_TLS", cfg.mqtt.use_tls)
    cfg.mqtt.broker_host = _env_str("MOBILE_GATEWAY_MQTT_BROKER_HOST", cfg.mqtt.broker_host)
    cfg.mqtt.broker_port = _env_int("MOBILE_GATEWAY_MQTT_BROKER_PORT", cfg.mqtt.broker_port)
    cfg.mqtt.websocket_path = _env_str("MOBILE_GATEWAY_MQTT_WEBSOCKET_PATH", cfg.mqtt.websocket_path)
    cfg.mqtt.username = _env_str("MOBILE_GATEWAY_MQTT_USERNAME", cfg.mqtt.username)
    cfg.mqtt.password = _env_str("MOBILE_GATEWAY_MQTT_PASSWORD", cfg.mqtt.password)
    cfg.mqtt.client_id = _env_str("MOBILE_GATEWAY_MQTT_CLIENT_ID", cfg.mqtt.client_id)
    cfg.mqtt.robot_id = ROBOT_ID
    cfg.mqtt.cmd_qos = _env_int("MOBILE_GATEWAY_MQTT_CMD_QOS", cfg.mqtt.cmd_qos)
    cfg.mqtt.ack_qos = _env_int("MOBILE_GATEWAY_MQTT_ACK_QOS", cfg.mqtt.ack_qos)
    cfg.mqtt.status_qos = _env_int("MOBILE_GATEWAY_MQTT_STATUS_QOS", cfg.mqtt.status_qos)
    cfg.mqtt.heartbeat_qos = _env_int("MOBILE_GATEWAY_MQTT_HEARTBEAT_QOS", cfg.mqtt.heartbeat_qos)
    cfg.mqtt.retain_status = _env_bool("MOBILE_GATEWAY_MQTT_RETAIN_STATUS", cfg.mqtt.retain_status)
    cfg.mqtt.retain_heartbeat = _env_bool("MOBILE_GATEWAY_MQTT_RETAIN_HEARTBEAT", cfg.mqtt.retain_heartbeat)
    cfg.mqtt.keepalive_s = _env_int("MOBILE_GATEWAY_MQTT_KEEPALIVE_S", cfg.mqtt.keepalive_s)
    cfg.mqtt.connect_timeout_s = _env_float("MOBILE_GATEWAY_MQTT_CONNECT_TIMEOUT_S", cfg.mqtt.connect_timeout_s)
    cfg.backend.default_robot_id = ROBOT_ID
    cfg.mqtt.robot_id = ROBOT_ID
    cfg.mqtt.topics.cmd = MQTT_TOPIC_CMD
    cfg.mqtt.topics.ack = MQTT_TOPIC_ACK
    cfg.mqtt.topics.status = MQTT_TOPIC_STATUS
    cfg.mqtt.topics.heartbeat = MQTT_TOPIC_HEARTBEAT

    return cfg


CONFIG = build_config()
