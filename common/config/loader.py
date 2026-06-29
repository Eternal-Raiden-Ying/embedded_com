#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified configuration loader for the robot stack."""

import dataclasses
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from .schema import SystemGlobalConfig
from .validators import validate_config

# Cache singleton config instance
_GLOBAL_CONFIG = None


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Fallback lightweight YAML parser that does not require external PyYAML package."""
    root = {}
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
        if lowered in {"null", "none", "~"}:
            return None
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                return []
            return [_scalar(part.strip()) for part in inner.split(",")]
        try:
            if "." in value:
                return float(value)
            return int(value)
        except Exception:
            return value

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, value = line.strip().split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if value.strip() == "":
            child = {}
            current[key.strip()] = child
            stack.append((indent, child))
        else:
            current[key.strip()] = _scalar(value)
    return root


def load_yaml_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a YAML file, falling back to simple parser if PyYAML is missing."""
    file_path = Path(path)
    if not file_path.is_file():
        return {}
    try:
        import yaml  # type: ignore
        with open(file_path, "r", encoding="utf-8") as fp:
            return dict(yaml.safe_load(fp) or {})
    except Exception:
        return _parse_simple_yaml(file_path.read_text(encoding="utf-8"))


def apply_env_overrides(config: SystemGlobalConfig) -> None:
    """Apply environment variable overrides as initial fallback adjustments."""
    def _set_from_env(obj, attr, env_name, val_type):
        raw = os.getenv(env_name)
        if raw is not None:
            if val_type is bool:
                setattr(obj, attr, raw.strip().lower() in {"1", "true", "yes", "y", "on"})
            elif val_type is int:
                try:
                    setattr(obj, attr, int(raw))
                except ValueError:
                    pass
            elif val_type is float:
                try:
                    setattr(obj, attr, float(raw))
                except ValueError:
                    pass
            else:
                setattr(obj, attr, str(raw).strip())

    def _set_endpoint_from_env(endpoint, prefix: str) -> None:
        _set_from_env(endpoint, "transport", f"{prefix}_TRANSPORT", str)
        _set_from_env(endpoint, "ipc_socket_path", f"{prefix}_SOCKET_PATH", str)
        _set_from_env(endpoint, "send_mode", f"{prefix}_SEND_MODE", str)
        _set_from_env(endpoint, "async_enabled", f"{prefix}_ASYNC_ENABLED", bool)

    # Vision Runtime
    _set_from_env(config.vision.runtime, "project_root", "VISION_PROJECT_ROOT", str)
    _set_from_env(config.vision.runtime, "runs_dir", "VISION_RUNS_DIR", str)
    _set_from_env(config.vision.runtime, "stack_run_id", "STACK_RUN_ID", str)

    # Vision Model
    _set_from_env(config.vision.model, "active_model", "VISTA_TABLE_MODEL", str)

    # Vision debug / preview compatibility with start_robot_stack.sh
    _set_from_env(config.vision.debug, "preview", "VISTA_DEBUG_PREVIEW", bool)
    _set_from_env(config.vision.debug, "preview", "VISTA_PREVIEW_RGB", bool)
    _set_from_env(config.vision.preview, "show_rgb", "VISTA_PREVIEW_RGB", bool)
    _set_from_env(config.vision.debug, "table_bbox_enabled", "VISTA_TABLE_BBOX_ENABLE", bool)
    _set_from_env(config.vision.debug, "mock_table_bbox", "VISTA_MOCK_TABLE_BBOX", str)

    # Vision IPC endpoints
    _set_from_env(config.vision.req_in, "transport", "VISION_REQ_IN_TRANSPORT", str)
    _set_from_env(config.vision.req_in, "ipc_socket_path", "VISION_REQ_IN_SOCKET_PATH", str)
    _set_from_env(config.vision.obs_out, "transport", "VISION_OBS_OUT_TRANSPORT", str)
    _set_from_env(config.vision.obs_out, "ipc_socket_path", "VISION_OBS_OUT_SOCKET_PATH", str)

    # Orchestrator Runtime
    _set_from_env(config.orchestrator.runtime, "project_root", "ORCH_PROJECT_ROOT", str)
    _set_from_env(config.orchestrator.runtime, "log_dir", "ORCH_LOG_DIR", str)
    _set_from_env(config.orchestrator.runtime, "log_file", "ORCH_LOG_FILE", str)
    _set_from_env(config.orchestrator.runtime, "runs_dir", "ORCH_RUNS_DIR", str)
    _set_from_env(config.orchestrator.runtime, "pid_dir", "ORCH_PID_DIR", str)
    _set_from_env(config.orchestrator.runtime, "pid_file", "ORCH_PID_FILE", str)
    _set_from_env(config.orchestrator.runtime, "stack_run_id", "STACK_RUN_ID", str)
    _set_from_env(config.orchestrator.runtime, "tick_hz", "ORCH_TICK_HZ", float)
    _set_from_env(config.orchestrator.runtime, "log_mode", "ORCH_LOG_MODE", str)
    _set_from_env(config.orchestrator.runtime, "log_enabled", "ORCH_LOG_ENABLED", bool)
    _set_from_env(config.orchestrator.runtime, "debug", "ORCH_DEBUG", bool)
    _set_from_env(config.orchestrator.runtime, "state_block_period_s", "ORCH_STATE_BLOCK_PERIOD_S", float)
    _set_from_env(config.orchestrator.runtime, "heartbeat_period_s", "ORCH_HEARTBEAT_PERIOD_S", float)

    # Orchestrator Car
    _set_from_env(config.orchestrator.car, "table_controlled_wz_max_radps", "ORCH_CAR_TABLE_CONTROLLED_WZ_MAX_RADPS", float)
    _set_from_env(config.orchestrator.car, "table_wz_view_max_radps", "ORCH_CAR_TABLE_WZ_VIEW_MAX_RADPS", float)

    # Orchestrator Serial
    _set_from_env(config.orchestrator.serial, "port", "ORCH_SERIAL_PORT", str)
    _set_from_env(config.orchestrator.serial, "baudrate", "ORCH_SERIAL_BAUDRATE", int)
    _set_from_env(config.orchestrator.serial, "timeout_s", "ORCH_SERIAL_TIMEOUT_S", float)
    _set_from_env(config.orchestrator.serial, "dry_run", "ORCH_SERIAL_DRY_RUN", bool)
    _set_from_env(config.orchestrator.serial, "readback_enabled", "ORCH_READBACK_ENABLED", bool)
    _set_from_env(config.orchestrator.serial, "dry_run_echo_stdout", "ORCH_DRY_RUN_ECHO_STDOUT", bool)
    _set_from_env(config.orchestrator.serial, "dry_run_echo_on_change_only", "ORCH_DRY_RUN_ECHO_ON_CHANGE_ONLY", bool)
    _set_from_env(config.orchestrator.serial, "dry_run_echo_summary_period_s", "ORCH_DRY_RUN_ECHO_SUMMARY_PERIOD_S", float)
    _set_from_env(config.orchestrator.serial, "dry_run_quiet_idle_stop", "ORCH_DRY_RUN_QUIET_IDLE_STOP", bool)
    _set_from_env(config.orchestrator.serial, "uart_lowfreq_period_s", "ORCH_UART_LOWFREQ_PERIOD_S", float)
    _set_from_env(config.orchestrator.serial, "stm32_status_enabled", "ORCH_STM32_STATUS_ENABLED", bool)
    _set_from_env(config.orchestrator.serial, "stm32_status_period_s", "ORCH_STM32_STATUS_PERIOD_S", float)

    # Orchestrator IPC endpoints
    _set_from_env(config.orchestrator.task_cmd_in, "transport", "ORCH_TASK_CMD_IN_TRANSPORT", str)
    _set_from_env(config.orchestrator.task_cmd_in, "ipc_socket_path", "ORCH_TASK_CMD_IN_SOCKET_PATH", str)
    _set_from_env(config.orchestrator.task_cmd_in, "tcp_host", "ORCH_TASK_CMD_IN_HOST", str)
    _set_from_env(config.orchestrator.task_cmd_in, "tcp_port", "ORCH_TASK_CMD_IN_PORT", int)
    _set_from_env(config.orchestrator.task_ack_out, "transport", "ORCH_TASK_ACK_OUT_TRANSPORT", str)
    _set_from_env(config.orchestrator.task_ack_out, "ipc_socket_path", "ORCH_TASK_ACK_OUT_SOCKET_PATH", str)
    _set_from_env(config.orchestrator.task_ack_out, "tcp_host", "ORCH_TASK_ACK_OUT_HOST", str)
    _set_from_env(config.orchestrator.task_ack_out, "tcp_port", "ORCH_TASK_ACK_OUT_PORT", int)
    _set_from_env(config.orchestrator.vision_obs_in, "transport", "ORCH_VISION_OBS_IN_TRANSPORT", str)
    _set_from_env(config.orchestrator.vision_obs_in, "ipc_socket_path", "ORCH_VISION_OBS_IN_SOCKET_PATH", str)
    _set_from_env(config.orchestrator.vision_obs_in, "tcp_host", "ORCH_VISION_OBS_IN_HOST", str)
    _set_from_env(config.orchestrator.vision_obs_in, "tcp_port", "ORCH_VISION_OBS_IN_PORT", int)
    _set_from_env(config.orchestrator.vision_req_out, "transport", "ORCH_VISION_REQ_OUT_TRANSPORT", str)
    _set_from_env(config.orchestrator.vision_req_out, "ipc_socket_path", "ORCH_VISION_REQ_OUT_SOCKET_PATH", str)
    _set_from_env(config.orchestrator.vision_req_out, "tcp_host", "ORCH_VISION_REQ_OUT_HOST", str)
    _set_from_env(config.orchestrator.vision_req_out, "tcp_port", "ORCH_VISION_REQ_OUT_PORT", int)

    # Gateway Runtime
    _set_from_env(config.gateway.runtime, "project_root", "MOBILE_GATEWAY_PROJECT_ROOT", str)
    _set_from_env(config.gateway.runtime, "repo_root", "MOBILE_GATEWAY_REPO_ROOT", str)
    _set_from_env(config.gateway.runtime, "log_dir", "MOBILE_GATEWAY_LOG_DIR", str)
    _set_from_env(config.gateway.runtime, "log_file", "MOBILE_GATEWAY_LOG_FILE", str)
    _set_from_env(config.gateway.runtime, "runs_dir", "MOBILE_GATEWAY_RUNS_DIR", str)
    _set_from_env(config.gateway.runtime, "pid_dir", "MOBILE_GATEWAY_PID_DIR", str)
    _set_from_env(config.gateway.runtime, "pid_file", "MOBILE_GATEWAY_PID_FILE", str)
    _set_from_env(config.gateway.runtime, "stack_run_id", "STACK_RUN_ID", str)
    _set_from_env(config.gateway.runtime, "mode", "MOBILE_GATEWAY_RUNTIME_MODE", str)
    _set_from_env(config.gateway.runtime, "log_level", "MOBILE_GATEWAY_LOG_LEVEL", str)
    _set_from_env(config.gateway.runtime, "tick_hz", "MOBILE_GATEWAY_TICK_HZ", float)
    _set_from_env(config.gateway.runtime, "heartbeat_period_s", "MOBILE_GATEWAY_HEARTBEAT_PERIOD_S", float)
    _set_from_env(config.gateway.runtime, "heartbeat_log_interval_s", "MOBILE_GATEWAY_HEARTBEAT_LOG_INTERVAL_S", float)
    _set_from_env(config.gateway.runtime, "suppress_heartbeat_success_log", "MOBILE_GATEWAY_SUPPRESS_HEARTBEAT_SUCCESS_LOG", bool)
    _set_from_env(config.gateway.runtime, "enable_raw_mqtt_debug", "MOBILE_GATEWAY_ENABLE_RAW_MQTT_DEBUG", bool)
    _set_from_env(config.gateway.runtime, "enable_legacy_command_compat", "MOBILE_GATEWAY_ENABLE_LEGACY_COMMAND_COMPAT", bool)
    _set_from_env(config.gateway.runtime, "cmd_dedup_cache_size", "MOBILE_GATEWAY_CMD_DEDUP_CACHE_SIZE", int)
    _set_from_env(config.gateway.runtime, "log_mode", "MOBILE_GATEWAY_LOG_MODE", str)
    _set_from_env(config.gateway.runtime, "log_enabled", "MOBILE_GATEWAY_LOG_ENABLED", bool)
    _set_from_env(config.gateway.runtime, "status_stdout", "MOBILE_GATEWAY_STATUS_STDOUT", bool)
    _set_from_env(config.gateway.runtime, "stdin_enabled", "MOBILE_GATEWAY_STDIN_ENABLED", bool)

    # Gateway Backend
    _set_from_env(config.gateway.backend, "mode", "MOBILE_GATEWAY_BACKEND", str)
    _set_from_env(config.gateway.backend, "default_robot_id", "MOBILE_GATEWAY_ROBOT_ID", str)
    _set_from_env(config.gateway.backend, "default_confidence", "MOBILE_GATEWAY_DEFAULT_CONFIDENCE", float)
    _set_from_env(config.gateway.backend, "mock_step_interval_s", "MOBILE_GATEWAY_MOCK_STEP_INTERVAL_S", float)
    _set_from_env(config.gateway.backend, "enforce_single_flight", "MOBILE_GATEWAY_SINGLE_FLIGHT", bool)
    _set_from_env(config.gateway.backend, "observer_enabled", "MOBILE_GATEWAY_OBSERVER_ENABLED", bool)
    _set_from_env(config.gateway.backend, "observer_poll_interval_s", "MOBILE_GATEWAY_OBSERVER_POLL_INTERVAL_S", float)
    _set_from_env(config.gateway.backend, "orchestrator_runs_dir", "MOBILE_GATEWAY_ORCH_RUNS_DIR", str)
    _set_from_env(config.gateway.backend, "state_blocks_path", "MOBILE_GATEWAY_ORCH_STATE_BLOCKS_PATH", str)
    _set_from_env(config.gateway.backend, "state_block_log_mode", "MOBILE_GATEWAY_ORCH_STATE_BLOCK_LOG_MODE", str)
    _set_from_env(config.gateway.backend, "state_block_log_period_s", "MOBILE_GATEWAY_ORCH_STATE_BLOCK_LOG_PERIOD_S", float)
    _set_from_env(config.gateway.backend, "stop_cooldown_s", "MOBILE_GATEWAY_STOP_COOLDOWN_S", float)

    # Gateway MQTT
    _set_from_env(config.gateway.mqtt, "enabled", "MOBILE_GATEWAY_MQTT_ENABLED", bool)
    _set_from_env(config.gateway.mqtt, "transport", "MOBILE_GATEWAY_MQTT_TRANSPORT", str)
    _set_from_env(config.gateway.mqtt, "use_tls", "MOBILE_GATEWAY_MQTT_USE_TLS", bool)
    _set_from_env(config.gateway.mqtt, "broker_host", "MOBILE_GATEWAY_MQTT_BROKER_HOST", str)
    _set_from_env(config.gateway.mqtt, "broker_port", "MOBILE_GATEWAY_MQTT_BROKER_PORT", int)
    _set_from_env(config.gateway.mqtt, "websocket_path", "MOBILE_GATEWAY_MQTT_WEBSOCKET_PATH", str)
    _set_from_env(config.gateway.mqtt, "username", "MOBILE_GATEWAY_MQTT_USERNAME", str)
    _set_from_env(config.gateway.mqtt, "password", "MOBILE_GATEWAY_MQTT_PASSWORD", str)
    _set_from_env(config.gateway.mqtt, "client_id", "MOBILE_GATEWAY_MQTT_CLIENT_ID", str)
    _set_from_env(config.gateway.mqtt, "robot_id", "MOBILE_GATEWAY_MQTT_ROBOT_ID", str)
    _set_from_env(config.gateway.mqtt.topics, "cmd", "MOBILE_GATEWAY_MQTT_TOPIC_CMD", str)
    _set_from_env(config.gateway.mqtt.topics, "ack", "MOBILE_GATEWAY_MQTT_TOPIC_ACK", str)
    _set_from_env(config.gateway.mqtt.topics, "status", "MOBILE_GATEWAY_MQTT_TOPIC_STATUS", str)
    _set_from_env(config.gateway.mqtt.topics, "heartbeat", "MOBILE_GATEWAY_MQTT_TOPIC_HEARTBEAT", str)

    # Gateway endpoints
    _set_from_env(config.gateway.command_in, "transport", "MOBILE_GATEWAY_COMMAND_IN_TRANSPORT", str)
    _set_from_env(config.gateway.command_in, "ipc_socket_path", "MOBILE_GATEWAY_COMMAND_IN_SOCKET_PATH", str)
    _set_from_env(config.gateway.command_in, "tcp_host", "MOBILE_GATEWAY_COMMAND_IN_HOST", str)
    _set_from_env(config.gateway.command_in, "tcp_port", "MOBILE_GATEWAY_COMMAND_IN_PORT", int)
    _set_from_env(config.gateway.status_out, "transport", "MOBILE_GATEWAY_STATUS_OUT_TRANSPORT", str)
    _set_from_env(config.gateway.status_out, "ipc_socket_path", "MOBILE_GATEWAY_STATUS_OUT_SOCKET_PATH", str)
    _set_from_env(config.gateway.status_out, "tcp_host", "MOBILE_GATEWAY_STATUS_OUT_HOST", str)
    _set_from_env(config.gateway.status_out, "tcp_port", "MOBILE_GATEWAY_STATUS_OUT_PORT", int)
    _set_from_env(config.gateway.orchestrator_task_cmd_out, "transport", "MOBILE_GATEWAY_ORCH_TASK_CMD_TRANSPORT", str)
    _set_from_env(config.gateway.orchestrator_task_cmd_out, "ipc_socket_path", "MOBILE_GATEWAY_ORCH_TASK_CMD_SOCKET_PATH", str)
    _set_from_env(config.gateway.orchestrator_task_cmd_out, "tcp_host", "MOBILE_GATEWAY_ORCH_TASK_CMD_HOST", str)
    _set_from_env(config.gateway.orchestrator_task_cmd_out, "tcp_port", "MOBILE_GATEWAY_ORCH_TASK_CMD_PORT", int)
    _set_from_env(config.gateway.orchestrator_task_ack_in, "transport", "MOBILE_GATEWAY_ORCH_TASK_ACK_TRANSPORT", str)
    _set_from_env(config.gateway.orchestrator_task_ack_in, "ipc_socket_path", "MOBILE_GATEWAY_ORCH_TASK_ACK_SOCKET_PATH", str)
    _set_from_env(config.gateway.orchestrator_task_ack_in, "tcp_host", "MOBILE_GATEWAY_ORCH_TASK_ACK_HOST", str)
    _set_from_env(config.gateway.orchestrator_task_ack_in, "tcp_port", "MOBILE_GATEWAY_ORCH_TASK_ACK_PORT", int)

    # Online Edge Runtime
    _set_from_env(config.online_edge.runtime, "project_root", "EDGE_PROJECT_ROOT", str)
    _set_from_env(config.online_edge.runtime, "log_dir", "EDGE_LOG_DIR", str)
    _set_from_env(config.online_edge.runtime, "log_file", "EDGE_LOG_FILE", str)
    _set_from_env(config.online_edge.runtime, "runs_dir", "EDGE_RUNS_DIR", str)
    _set_from_env(config.online_edge.runtime, "pid_dir", "EDGE_PID_DIR", str)
    _set_from_env(config.online_edge.runtime, "pid_file", "EDGE_PID_FILE", str)
    _set_from_env(config.online_edge.runtime, "stack_run_id", "STACK_RUN_ID", str)
    _set_from_env(config.online_edge.runtime, "loop_hz", "EDGE_LOOP_HZ", float)
    _set_from_env(config.online_edge.runtime, "preview", "EDGE_PREVIEW", bool)
    _set_from_env(config.online_edge.runtime, "save_snapshot_period_s", "EDGE_SNAPSHOT_PERIOD_S", float)
    _set_from_env(config.online_edge.runtime, "snapshot_dir", "EDGE_SNAPSHOT_DIR", str)
    _set_from_env(config.online_edge.runtime, "log_mode", "EDGE_LOG_MODE", str)
    _set_from_env(config.online_edge.runtime, "log_enabled", "EDGE_LOG_ENABLED", bool)

    # Online Edge Output
    _set_from_env(config.online_edge.output, "transport", "EDGE_OUT_TRANSPORT", str)
    _set_from_env(config.online_edge.output, "ipc_socket_path", "EDGE_OUT_SOCKET_PATH", str)
    _set_from_env(config.online_edge.output, "send_interval_s", "EDGE_OUT_PERIOD_S", float)

    # Online Edge Camera
    _set_from_env(config.online_edge.camera, "bag_path", "EDGE_BAG_PATH", str)
    _set_from_env(config.online_edge.camera, "align_to_color", "EDGE_ALIGN_TO_COLOR", bool)
    _set_from_env(config.online_edge.camera, "depth_enabled", "EDGE_DEPTH_ENABLED", bool)
    _set_from_env(config.online_edge.camera, "depth_width", "EDGE_DEPTH_WIDTH", int)
    _set_from_env(config.online_edge.camera, "depth_height", "EDGE_DEPTH_HEIGHT", int)
    _set_from_env(config.online_edge.camera, "depth_fps", "EDGE_DEPTH_FPS", int)
    _set_from_env(config.online_edge.camera, "color_enabled", "EDGE_COLOR_ENABLED", bool)
    _set_from_env(config.online_edge.camera, "color_width", "EDGE_COLOR_WIDTH", int)
    _set_from_env(config.online_edge.camera, "color_height", "EDGE_COLOR_HEIGHT", int)
    _set_from_env(config.online_edge.camera, "color_fps", "EDGE_COLOR_FPS", int)

    # Online Edge Detector
    _set_from_env(config.online_edge.detector, "calib_json", "EDGE_CALIB_JSON", str)
    _set_from_env(config.online_edge.detector, "target_dist_m_override", "EDGE_TARGET_DIST_M", float)
    _set_from_env(config.online_edge.detector, "roi_y0", "EDGE_ROI_Y0", int)
    _set_from_env(config.online_edge.detector, "roi_y1", "EDGE_ROI_Y1", int)
    _set_from_env(config.online_edge.detector, "roi_x0", "EDGE_ROI_X0", int)
    _set_from_env(config.online_edge.detector, "roi_x1", "EDGE_ROI_X1", int)
    _set_from_env(config.online_edge.detector, "z_min", "EDGE_Z_MIN", float)
    _set_from_env(config.online_edge.detector, "z_max", "EDGE_Z_MAX", float)
    _set_from_env(config.online_edge.detector, "table_y_min", "EDGE_TABLE_Y_MIN", float)
    _set_from_env(config.online_edge.detector, "table_y_max", "EDGE_TABLE_Y_MAX", float)
    _set_from_env(config.online_edge.detector, "min_all_points", "EDGE_MIN_ALL_POINTS", int)
    _set_from_env(config.online_edge.detector, "min_table_points", "EDGE_MIN_TABLE_POINTS", int)
    _set_from_env(config.online_edge.detector, "ransac_iters", "EDGE_RANSAC_ITERS", int)
    _set_from_env(config.online_edge.detector, "residual_threshold_m", "EDGE_RANSAC_THRESHOLD_M", float)
    _set_from_env(config.online_edge.detector, "random_seed", "EDGE_RANDOM_SEED", int)
    _set_from_env(config.online_edge.detector, "depth_median_ksize", "EDGE_MEDIAN_KSIZE", int)


def merge_dict_into_dataclass(obj: Any, data: Dict[str, Any]) -> None:
    """Recursively merge dictionary values into a pre-initialized dataclass hierarchy."""
    for key, value in data.items():
        if not hasattr(obj, key):
            continue

        current_val = getattr(obj, key)

        if dataclasses.is_dataclass(current_val):
            if isinstance(value, dict):
                merge_dict_into_dataclass(current_val, value)
        elif isinstance(current_val, dict):
            if isinstance(value, dict):
                for k, v in value.items():
                    if k in current_val:
                        if dataclasses.is_dataclass(current_val[k]):
                            if isinstance(v, dict):
                                merge_dict_into_dataclass(current_val[k], v)
                            elif v is not None:
                                current_val[k] = v
                        elif isinstance(current_val[k], dict) and isinstance(v, dict):
                            current_val[k].update(v)
                        else:
                            if v is not None and v != "":
                                current_val[k] = v
                    else:
                        current_val[k] = v
        else:
            # Skip empty strings and None values to preserve schema defaults
            if value is None or value == "":
                continue

            target_type = type(current_val) if current_val is not None else type(value)

            coerced_value = value
            if current_val is not None:
                if target_type is bool and not isinstance(value, bool):
                    coerced_value = str(value).lower() in ("true", "1", "yes")
                elif target_type is int and not isinstance(value, int):
                    try:
                        coerced_value = int(value)
                    except Exception:
                        pass
                elif target_type is float and not isinstance(value, float):
                    try:
                        coerced_value = float(value)
                    except Exception:
                        pass
                elif target_type is tuple and isinstance(value, list):
                    coerced_value = tuple(value)
                elif target_type is list and isinstance(value, tuple):
                    coerced_value = list(value)

            setattr(obj, key, coerced_value)


def sync_orchestrator_config(config: SystemGlobalConfig) -> None:
    """Synchronize duplicative absolute-speed config values for backward compatibility."""
    car = config.orchestrator.car
    car.table_vx_mps_min = abs(float(car.table_controlled_vx_min_mps))
    car.table_vx_mps_max = abs(float(car.table_controlled_vx_max_mps))
    car.table_vy_max_mps = abs(float(car.table_controlled_vy_max_mps))
    if getattr(car, "table_wz_view_max_radps", 0.0) == 0.0:
        car.table_wz_view_max_radps = abs(float(car.table_controlled_wz_max_radps))
    car.table_wz_plane_max_radps = abs(float(car.table_coarse_align_wz_max_radps))


def _resolve_config_path(config_path_str: str, system_config_dir: Optional[Path]) -> Path:
    p = Path(config_path_str)
    if p.is_absolute():
        return p
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / p
    if candidate.is_file():
        return candidate
    if system_config_dir:
        candidate = system_config_dir / p
        if candidate.is_file():
            return candidate
    candidate = Path.cwd() / p
    if candidate.is_file():
        return candidate
    return repo_root / p


_CONFIG_SECTION_KEYS = {"vision", "orchestrator", "gateway", "online_edge"}


def _record_loaded_file(config: SystemGlobalConfig, path: Path) -> None:
    value = str(path)
    for loaded in (
        config.vision.runtime.loaded_config_files,
        config.orchestrator.runtime.loaded_config_files,
    ):
        if value not in loaded:
            loaded.append(value)


def _record_primary_config_file(config: SystemGlobalConfig, path: Path) -> None:
    value = str(path)
    for loaded in (
        config.vision.runtime.loaded_config_files,
        config.orchestrator.runtime.loaded_config_files,
    ):
        while value in loaded:
            loaded.remove(value)
        loaded.insert(0, value)


def _merge_known_sections(config: SystemGlobalConfig, data: Dict[str, Any]) -> None:
    known = {key: value for key, value in data.items() if key in _CONFIG_SECTION_KEYS}
    if known:
        merge_dict_into_dataclass(config, known)


def _profile_name_from_data(config: SystemGlobalConfig, data: Dict[str, Any]) -> str:
    raw = os.getenv("SYSTEM_CONFIG_PROFILE") or os.getenv("CONFIG_PROFILE") or data.get("profile") or config.profile
    if isinstance(raw, dict):
        raw = raw.get("name") or raw.get("id") or ""
    return str(raw or "").strip()


def _load_profile_data(profile_name: str, system_config_dir: Optional[Path]) -> Optional[Path]:
    if not profile_name:
        return None
    profile_path = Path(profile_name)
    if not profile_path.suffix:
        profile_path = Path("profiles") / f"{profile_name}.yaml"
    return _resolve_config_path(str(profile_path), system_config_dir)


def _merge_layered_system_config(
    config: SystemGlobalConfig,
    yaml_data: Dict[str, Any],
    system_config_dir: Optional[Path],
) -> None:
    defaults = yaml_data.get("defaults")
    if isinstance(defaults, dict):
        _merge_known_sections(config, defaults)

    _merge_known_sections(config, yaml_data)

    profile_name = _profile_name_from_data(config, yaml_data)
    if profile_name:
        config.profile = profile_name
        config.orchestrator.runtime.config_profile = profile_name
        profile_path = _load_profile_data(profile_name, system_config_dir)
        if profile_path and profile_path.is_file():
            profile_data = load_yaml_file(profile_path)
            profile_defaults = profile_data.get("defaults")
            if isinstance(profile_defaults, dict):
                _merge_known_sections(config, profile_defaults)
            _merge_known_sections(config, profile_data)
            overrides = profile_data.get("runtime_overrides")
            if isinstance(overrides, dict):
                _merge_known_sections(config, overrides)
            _record_loaded_file(config, profile_path)

    runtime_overrides = yaml_data.get("runtime_overrides")
    if isinstance(runtime_overrides, dict):
        _merge_known_sections(config, runtime_overrides)

    if config.profile:
        config.orchestrator.runtime.config_profile = config.profile


def load_global_config(config_path: str = None) -> SystemGlobalConfig:
    """Load the global configuration structure.
    
    Priority Rules:
    1. Environment variable overrides take highest priority.
    2. Values in YAML file override schema defaults.
    3. Fallback to schema defaults (defined in schema.py).
    """
    # 1. Instantiate default configurations from schema
    config = SystemGlobalConfig()

    # 2. Locate system config YAML file path
    default_yaml_path = Path(__file__).resolve().parents[2] / "configs" / "system_config.yaml"
    env_config_path = os.getenv("SYSTEM_CONFIG_FILE")
    
    target_path = None
    if env_config_path:
        target_path = Path(env_config_path)
    elif config_path:
        target_path = Path(config_path)
    else:
        target_path = default_yaml_path

    # 3. Load YAML and merge it into the config
    if target_path and target_path.is_file():
        yaml_data = load_yaml_file(target_path)
        _merge_layered_system_config(config, yaml_data, target_path.parent)
        _record_primary_config_file(config, target_path)

    # 4. Apply environment variable overrides (highest priority)
    apply_env_overrides(config)

    # 5. Validate configuration
    validate_config(config)

    sync_orchestrator_config(config)

    return config


def get_config(reload: bool = False) -> SystemGlobalConfig:
    """Return the global configuration singleton."""
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None or reload:
        _GLOBAL_CONFIG = load_global_config()
    return _GLOBAL_CONFIG
