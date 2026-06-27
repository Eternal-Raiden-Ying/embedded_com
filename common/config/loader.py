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
    _set_from_env(config.vision.runtime, "vision_params_file", "VISION_PARAMS_FILE", str)
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
    _set_from_env(config.orchestrator.runtime, "stage_params_file", "ORCH_STAGE_PARAMS_FILE", str)
    _set_from_env(config.orchestrator.runtime, "car_cmd_params_file", "ORCH_CAR_CMD_PARAMS_FILE", str)

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


def _load_and_merge_stage_params(config: SystemGlobalConfig, file_path: Path) -> None:
    yaml_data = load_yaml_file(file_path)
    if not yaml_data:
        return
    
    def _sf(v) -> float:
        try:
            return float(v)
        except Exception:
            return 0.0

    def _si(v) -> int:
        try:
            return int(v)
        except Exception:
            return 0

    def _sb(v) -> bool:
        if isinstance(v, bool):
            return v
        return str(v).lower() in ("true", "1", "yes")

    fl = yaml_data.get("final_lock", {})
    if fl:
        ctrl = config.orchestrator.control
        dock = config.orchestrator.docking
        if "yaw_abs_th" in fl:
            ctrl.final_lock_yaw_tol_rad = _sf(fl["yaw_abs_th"])
        if "dist_abs_th_m" in fl:
            ctrl.final_lock_dist_tol_m = _sf(fl["dist_abs_th_m"])
        if "edge_conf_th" in fl:
            dock.min_confidence = _sf(fl["edge_conf_th"])
        if "stable_ms" in fl:
            ctrl.final_lock_min_hold_ms = _si(fl["stable_ms"])
        if "timeout_ms" in fl:
            ctrl.approach_timeout_s = _sf(fl["timeout_ms"]) / 1000.0
        if "target_dist_m" in fl:
            ctrl.table_target_dist_m = _sf(fl["target_dist_m"])
        if "stop_margin_m" in fl:
            ctrl.table_stop_margin_m = _sf(fl["stop_margin_m"])
        if "settle_ms" in fl:
            ctrl.table_settle_s = _sf(fl["settle_ms"]) / 1000.0
        if "stable_frames" in fl:
            ctrl.table_stable_frames = _si(fl["stable_frames"])
        if "required_ready_obs" in fl:
            ctrl.final_lock_required_ready_obs = _si(fl["required_ready_obs"])
        if "window_ms" in fl:
            ctrl.final_lock_window_ms = _si(fl["window_ms"])
        if "max_consecutive_lost" in fl:
            ctrl.final_lock_max_consecutive_lost = _si(fl["max_consecutive_lost"])
        if "soft_stale_hold" in fl:
            ctrl.final_lock_soft_stale_hold = _sb(fl["soft_stale_hold"])
        if "max_micro_adjust" in fl:
            ctrl.table_max_micro_adjust = _si(fl["max_micro_adjust"])

    td = yaml_data.get("table_docking", {})
    if td:
        ctrl = config.orchestrator.control
        if "enable_final_lock" in td:
            ctrl.enable_final_lock = _sb(td["enable_final_lock"])
        if "enable_micro_adjust" in td:
            ctrl.enable_micro_adjust = _sb(td["enable_micro_adjust"])
        if "final_lock_enter_dist_th_m" in td:
            ctrl.final_lock_enter_dist_th_m = _sf(td["final_lock_enter_dist_th_m"])
        if "final_lock_enter_yaw_th_rad" in td:
            ctrl.final_lock_enter_yaw_th_rad = _sf(td["final_lock_enter_yaw_th_rad"])
        if "align_to_approach_yaw_rad" in td:
            ctrl.align_to_approach_yaw_rad = _sf(td["align_to_approach_yaw_rad"])
        if "approach_to_align_yaw_rad" in td:
            ctrl.approach_to_align_yaw_rad = _sf(td["approach_to_align_yaw_rad"])
        if "align_to_approach_stable_obs" in td:
            ctrl.align_to_approach_stable_obs = _si(td["align_to_approach_stable_obs"])
        if "approach_to_align_stable_obs" in td:
            ctrl.approach_to_align_stable_obs = _si(td["approach_to_align_stable_obs"])
        if "coarse_align_min_dwell_s" in td:
            ctrl.coarse_align_min_dwell_s = _sf(td["coarse_align_min_dwell_s"])
        if "controlled_approach_min_dwell_s" in td:
            ctrl.controlled_approach_min_dwell_s = _sf(td["controlled_approach_min_dwell_s"])
        if "near_slow_max_vx_mps" in td:
            ctrl.near_slow_max_vx_mps = _sf(td["near_slow_max_vx_mps"])
        if "near_slow_max_vy_mps" in td:
            ctrl.near_slow_max_vy_mps = _sf(td["near_slow_max_vy_mps"])
        if "bbox_track_forward_enabled" in td:
            ctrl.bbox_track_forward_enabled = _sb(td["bbox_track_forward_enabled"])
        if "bbox_track_forward_vx_mps" in td:
            ctrl.bbox_track_forward_vx_mps = _sf(td["bbox_track_forward_vx_mps"])
        if "bbox_track_forward_max_vx_mps" in td:
            ctrl.bbox_track_forward_max_vx_mps = _sf(td["bbox_track_forward_max_vx_mps"])
        if "bbox_track_forward_center_band" in td:
            ctrl.bbox_track_forward_center_band = _sf(td["bbox_track_forward_center_band"])
        if "bbox_track_forward_min_hold_ms" in td:
            ctrl.bbox_track_forward_min_hold_ms = _si(td["bbox_track_forward_min_hold_ms"])
        if "bbox_track_forward_max_wz_radps" in td:
            ctrl.bbox_track_forward_max_wz_radps = _sf(td["bbox_track_forward_max_wz_radps"])
        if "edge_readiness_enabled" in td:
            ctrl.edge_readiness_enabled = _sb(td["edge_readiness_enabled"])
        if "edge_readiness_enter_score" in td:
            ctrl.edge_readiness_enter_score = _sf(td["edge_readiness_enter_score"])
        if "edge_readiness_exit_score" in td:
            ctrl.edge_readiness_exit_score = _sf(td["edge_readiness_exit_score"])
        if "edge_readiness_rise" in td:
            ctrl.edge_readiness_rise = _sf(td["edge_readiness_rise"])
        if "edge_readiness_decay" in td:
            ctrl.edge_readiness_decay = _sf(td["edge_readiness_decay"])
        if "edge_readiness_min_inliers" in td:
            ctrl.edge_readiness_min_inliers = _si(td["edge_readiness_min_inliers"])
        if "edge_readiness_yaw_max_rad" in td:
            ctrl.edge_readiness_yaw_max_rad = _sf(td["edge_readiness_yaw_max_rad"])
        if "edge_handoff_min_hold_ms" in td:
            ctrl.edge_handoff_min_hold_ms = _si(td["edge_handoff_min_hold_ms"])
        if "lateral_enabled" in td:
            ctrl.lateral_enabled = _sb(td["lateral_enabled"])
        if "lateral_vy_max_mps" in td:
            ctrl.lateral_vy_max_mps = _sf(td["lateral_vy_max_mps"])
        if "lateral_deadband_norm" in td:
            ctrl.lateral_deadband_norm = _sf(td["lateral_deadband_norm"])
        if "lateral_kp" in td:
            ctrl.lateral_kp = _sf(td["lateral_kp"])
        if "lateral_target_center_x_norm" in td:
            ctrl.lateral_target_center_x_norm = _sf(td["lateral_target_center_x_norm"])
        if "lateral_owner_default" in td:
            ctrl.lateral_owner_default = str(td["lateral_owner_default"])
        if "final_yaw_deadband_rad" in td:
            ctrl.final_yaw_deadband_rad = _sf(td["final_yaw_deadband_rad"])
        if "final_lock_yaw_rad" in td:
            ctrl.final_lock_yaw_rad = _sf(td["final_lock_yaw_rad"])
        if "final_yaw_realign_rad" in td:
            ctrl.final_yaw_realign_rad = _sf(td["final_yaw_realign_rad"])
        if "final_yaw_stable_frames" in td:
            ctrl.final_yaw_stable_frames = _si(td["final_yaw_stable_frames"])
        if "final_yaw_align_min_duration_ms" in td:
            ctrl.final_yaw_align_min_duration_ms = _si(td["final_yaw_align_min_duration_ms"])
        if "final_yaw_last_good_hold_s" in td:
            ctrl.final_yaw_last_good_hold_s = _sf(td["final_yaw_last_good_hold_s"])

    yt = yaml_data.get("yolo_table", {})
    if yt:
        ctrl = config.orchestrator.control
        car = config.orchestrator.car
        if "yolo_table_control_enable" in yt:
            ctrl.yolo_table_control_enable = _sb(yt["yolo_table_control_enable"])
        if "yolo_table_conf_min" in yt:
            ctrl.yolo_table_conf_min = _sf(yt["yolo_table_conf_min"])
        if "yolo_table_edge_stable_frames" in yt:
            ctrl.yolo_table_edge_stable_frames = _si(yt["yolo_table_edge_stable_frames"])
        if "edge_trusted_stable_frames" in yt:
            ctrl.edge_trusted_stable_frames = _si(yt["edge_trusted_stable_frames"])
        if "edge_trusted_min_conf" in yt:
            ctrl.edge_trusted_min_conf = _sf(yt["edge_trusted_min_conf"])
        if "edge_trusted_max_residual" in yt:
            ctrl.edge_trusted_max_residual = _sf(yt["edge_trusted_max_residual"])
        if "edge_trusted_min_support_count" in yt:
            ctrl.edge_trusted_min_support_count = _si(yt["edge_trusted_min_support_count"])
        if "edge_trusted_min_inlier_count" in yt:
            ctrl.edge_trusted_min_inlier_count = _si(yt["edge_trusted_min_inlier_count"])
        if "edge_trusted_min_x_span_m" in yt:
            ctrl.edge_trusted_min_x_span_m = _sf(yt["edge_trusted_min_x_span_m"])
        if "edge_trusted_max_background_penalty" in yt:
            ctrl.edge_trusted_max_background_penalty = _sf(yt["edge_trusted_max_background_penalty"])
        if "yolo_table_near_dist_m" in yt:
            ctrl.yolo_table_near_dist_m = _sf(yt["yolo_table_near_dist_m"])
        if "yolo_forward_vx_mps" in yt:
            car.yolo_table_forward_vx_mps = _sf(yt["yolo_forward_vx_mps"])
        if "yolo_yaw_gain" in yt:
            car.yolo_table_yaw_gain = _sf(yt["yolo_yaw_gain"])
        if "yolo_max_wz_radps" in yt:
            car.yolo_table_max_wz_radps = _sf(yt["yolo_max_wz_radps"])
        if "yolo_table_lost_to_search_frames" in yt:
            ctrl.yolo_table_lost_to_search_frames = _si(yt["yolo_table_lost_to_search_frames"])
        if "rotate_search_wz_radps" in yt:
            car.search_table_wz_radps = _sf(yt["rotate_search_wz_radps"])
        if "rotate_search_timeout_s" in yt:
            ctrl.rotate_search_timeout_s = _sf(yt["rotate_search_timeout_s"])
        if "rotate_require_edge_stable_frames" in yt:
            ctrl.rotate_require_edge_stable_frames = _si(yt["rotate_require_edge_stable_frames"])
        if "rotate_yaw_threshold_rad" in yt:
            ctrl.rotate_yaw_threshold_rad = _sf(yt["rotate_yaw_threshold_rad"])
        if "yolo_edge_conflict_block_rotate" in yt:
            ctrl.yolo_edge_conflict_block_rotate = _sb(yt["yolo_edge_conflict_block_rotate"])

    ca = yaml_data.get("controlled_approach", {})
    if ca:
        car = config.orchestrator.car
        dock = config.orchestrator.docking
        if "target_dist_m" in ca:
            dock.precise_dist_tol_m = _sf(ca["target_dist_m"])
        if "max_vx_mps" in ca:
            car.table_controlled_vx_max_mps = _sf(ca["max_vx_mps"])
        if "max_wz_radps" in ca:
            car.table_controlled_wz_max_radps = _sf(ca["max_wz_radps"])
        if "dist_kp" in ca:
            dock.dist_pid.kp = _sf(ca["dist_kp"])
        if "yaw_kp" in ca:
            dock.yaw_pid.kp = _sf(ca["yaw_kp"])
        if "fov_soft_th" in ca:
            car.table_fov_soft_th = _sf(ca["fov_soft_th"])
        if "fov_hard_th" in ca:
            car.table_fov_hard_th = _sf(ca["fov_hard_th"])
        if "stage_a_wz_radps" in ca:
            car.table_stage_a_wz_radps = _sf(ca["stage_a_wz_radps"])
        if "stage_b_vx_max_mps" in ca:
            car.table_stage_b_vx_max_mps = _sf(ca["stage_b_vx_max_mps"])
        if "stage_c_vx_max_mps" in ca:
            car.table_stage_c_vx_max_mps = _sf(ca["stage_c_vx_max_mps"])
        if "view_vy_max_mps" in ca:
            car.table_controlled_vy_max_mps = _sf(ca["view_vy_max_mps"])
        if "view_wz_max_radps" in ca:
            car.table_wz_view_max_radps = _sf(ca["view_wz_max_radps"])
        if "plane_wz_max_radps" in ca:
            car.table_coarse_align_wz_max_radps = _sf(ca["plane_wz_max_radps"])

    tdm = yaml_data.get("table_docking_motion", {})
    if tdm:
        car = config.orchestrator.car
        if "approach_safe_vx_mps" in tdm:
            car.table_approach_safe_vx_mps = _sf(tdm["approach_safe_vx_mps"])
        if "approach_max_vx_mps" in tdm:
            car.table_approach_max_vx_mps = _sf(tdm["approach_max_vx_mps"])
        if "approach_yaw_deadband_rad" in tdm:
            car.table_approach_yaw_deadband_rad = _sf(tdm["approach_yaw_deadband_rad"])
        if "approach_yaw_realign_rad" in tdm:
            car.table_approach_yaw_realign_rad = _sf(tdm["approach_yaw_realign_rad"])
        if "edge_hard_rotate_only_yaw_rad" in tdm:
            car.table_edge_hard_rotate_only_yaw_rad = _sf(tdm["edge_hard_rotate_only_yaw_rad"])
        if "edge_hard_yaw_rotate_only_frames" in tdm:
            car.table_edge_hard_yaw_rotate_only_frames = int(tdm["edge_hard_yaw_rotate_only_frames"])
        if "edge_hard_yaw_rotate_only_ms" in tdm:
            car.table_edge_hard_yaw_rotate_only_ms = int(tdm["edge_hard_yaw_rotate_only_ms"])
        if "perception_warmup_s" in tdm:
            car.table_perception_warmup_s = _sf(tdm["perception_warmup_s"])
        if "approach_allow_wz" in tdm:
            car.table_approach_allow_wz = _sb(tdm["approach_allow_wz"])
        if "approach_allow_vy" in tdm:
            car.table_approach_allow_vy = _sb(tdm["approach_allow_vy"])
        if "pose_missing_safe_vx_mps" in tdm:
            car.table_pose_missing_safe_vx_mps = _sf(tdm["pose_missing_safe_vx_mps"])
        if "pose_missing_max_hold_s" in tdm:
            car.table_pose_missing_max_hold_s = _sf(tdm["pose_missing_max_hold_s"])
        
        ca_sub = tdm.get("coarse_align", {})
        if ca_sub:
            if "vx_max_mps" in ca_sub:
                car.table_coarse_align_vx_max_mps = _sf(ca_sub["vx_max_mps"])
            if "vy_min_mps" in ca_sub:
                car.table_coarse_align_vy_min_mps = _sf(ca_sub["vy_min_mps"])
            if "vy_max_mps" in ca_sub:
                car.table_coarse_align_vy_max_mps = _sf(ca_sub["vy_max_mps"])
            if "wz_min_radps" in ca_sub:
                car.table_coarse_align_wz_min_radps = _sf(ca_sub["wz_min_radps"])
            if "wz_max_radps" in ca_sub:
                car.table_coarse_align_wz_max_radps = _sf(ca_sub["wz_max_radps"])
                
        cap_sub = tdm.get("controlled_approach", {})
        if cap_sub:
            if "vx_mps" in cap_sub:
                car.table_controlled_vx_min_mps = _sf(cap_sub["vx_mps"])
            if "vx_min_mps" in cap_sub:
                car.table_controlled_vx_min_mps = _sf(cap_sub["vx_min_mps"])
            if "vx_max_mps" in cap_sub:
                car.table_controlled_vx_max_mps = _sf(cap_sub["vx_max_mps"])
            if "vy_min_mps" in cap_sub:
                car.table_controlled_vy_min_mps = _sf(cap_sub["vy_min_mps"])
            if "vy_max_mps" in cap_sub:
                car.table_controlled_vy_max_mps = _sf(cap_sub["vy_max_mps"])
            if "wz_min_radps" in cap_sub:
                car.table_controlled_wz_min_radps = _sf(cap_sub["wz_min_radps"])
            if "wz_max_radps" in cap_sub:
                car.table_controlled_wz_max_radps = _sf(cap_sub["wz_max_radps"])
            if "allow_vy" in cap_sub:
                car.table_approach_allow_vy = _sb(cap_sub["allow_vy"])
            if "allow_wz" in cap_sub:
                car.table_approach_allow_wz = _sb(cap_sub["allow_wz"])

        fl_sub = tdm.get("final_lock", {})
        if fl_sub:
            if "vx_min_mps" in fl_sub:
                car.table_final_lock_vx_min_mps = _sf(fl_sub["vx_min_mps"])
            if "vx_max_mps" in fl_sub:
                car.table_final_lock_vx_max_mps = _sf(fl_sub["vx_max_mps"])
            if "vy_min_mps" in fl_sub:
                car.table_final_lock_vy_min_mps = _sf(fl_sub["vy_min_mps"])
            if "vy_max_mps" in fl_sub:
                car.table_final_lock_vy_max_mps = _sf(fl_sub["vy_max_mps"])
            if "wz_min_radps" in fl_sub:
                car.table_final_lock_wz_min_radps = _sf(fl_sub["wz_min_radps"])
            if "wz_max_radps" in fl_sub:
                car.table_final_lock_wz_max_radps = _sf(fl_sub["wz_max_radps"])
                
        db_sub = tdm.get("deadband", {})
        if db_sub:
            if "vx_mps" in db_sub:
                car.table_vx_deadband_mps = _sf(db_sub["vx_mps"])
            if "vy_mps" in db_sub:
                car.table_vy_deadband_mps = _sf(db_sub["vy_mps"])
            if "wz_radps" in db_sub:
                car.table_wz_deadband_radps = _sf(db_sub["wz_radps"])

    ess = yaml_data.get("edge_slide_search", {})
    if ess:
        car = config.orchestrator.car
        ctrl = config.orchestrator.control
        if "slide_vy_mps" in ess:
            car.edge_slide_vy_mps = _sf(ess["slide_vy_mps"])
        if "weak_slide_vy_mps" in ess:
            car.edge_slide_weak_vy_mps = _sf(ess["weak_slide_vy_mps"])
        
        max_vx = ess.get("max_forward_vx_mps", ess.get("max_vx_correction_mps"))
        if max_vx is not None:
            car.edge_slide_max_vx_mps = _sf(max_vx)
            
        max_wz = ess.get("max_wz_radps", ess.get("max_wz_correction_radps"))
        if max_wz is not None:
            car.edge_slide_max_wz_radps = _sf(max_wz)
            
        if "dist_kp" in ess:
            ctrl.edge_slide_dist_kp_mps_per_m = _sf(ess["dist_kp"])
        if "yaw_kp" in ess:
            ctrl.edge_slide_yaw_kp_radps_per_rad = _sf(ess["yaw_kp"])
        if "strong_edge_conf_track_local" in ess:
            ctrl.edge_follow_strong_edge_conf_track_local = _sf(ess["strong_edge_conf_track_local"])
        if "weak_edge_conf_track_local" in ess:
            ctrl.edge_follow_weak_edge_conf_track_local = _sf(ess["weak_edge_conf_track_local"])
        if "keep_dist_tolerance_m" in ess:
            ctrl.edge_slide_dist_tolerance_m = _sf(ess["keep_dist_tolerance_m"])
        if "dist_out_of_range_hold_ms" in ess:
            ctrl.edge_slide_dist_out_of_range_hold_s = _sf(ess["dist_out_of_range_hold_ms"]) / 1000.0
        if "max_relock_attempts" in ess:
            ctrl.edge_slide_max_relock_attempts = _si(ess["max_relock_attempts"])
        if "relock_failure_is_fatal" in ess:
            ctrl.edge_slide_relock_failure_is_fatal = _sb(ess["relock_failure_is_fatal"])
        if "pause_hold_ms" in ess:
            ctrl.edge_slide_pause_hold_s = _sf(ess["pause_hold_ms"]) / 1000.0
        if "recover_timeout_ms" in ess:
            ctrl.edge_slide_recover_timeout_s = _sf(ess["recover_timeout_ms"]) / 1000.0
        if "fallback_state" in ess:
            ctrl.edge_slide_fallback_state = str(ess["fallback_state"])
        if "direct_fallback_to_controlled_approach" in ess:
            ctrl.edge_slide_direct_fallback_to_controlled_approach = _sb(ess["direct_fallback_to_controlled_approach"])

    ef = yaml_data.get("edge_follow", {})
    if ef:
        ctrl = config.orchestrator.control
        if "table_edge_obs_max_age_ms" in ef:
            ctrl.table_edge_obs_max_age_ms = _si(ef["table_edge_obs_max_age_ms"])
        if "table_obs_stale_soft_ms" in ef:
            ctrl.table_obs_stale_soft_ms = _si(ef["table_obs_stale_soft_ms"])
        if "table_obs_stale_stop_ms" in ef:
            ctrl.table_obs_stale_stop_ms = _si(ef["table_obs_stale_stop_ms"])
        if "table_obs_stale_hard_ms" in ef:
            ctrl.table_obs_stale_hard_ms = _si(ef["table_obs_stale_hard_ms"])
        if "table_step_mode_enable" in ef:
            ctrl.table_step_mode_enable = _sb(ef["table_step_mode_enable"])
        if "table_step_burst_ms" in ef:
            ctrl.table_step_burst_ms = _si(ef["table_step_burst_ms"])
        if "table_step_hold_until_new_obs" in ef:
            ctrl.table_step_hold_until_new_obs = _sb(ef["table_step_hold_until_new_obs"])
        if "min_edge_conf" in ef:
            ctrl.edge_follow_min_edge_conf = _sf(ef["min_edge_conf"])
        if "min_edge_conf_table_edge_perception" in ef:
            ctrl.edge_follow_min_edge_conf_table_edge_perception = _sf(ef["min_edge_conf_table_edge_perception"])
        if "min_edge_conf_track_local" in ef:
            ctrl.edge_follow_min_edge_conf_track_local = _sf(ef["min_edge_conf_track_local"])
        if "weak_edge_conf_track_local" in ef:
            ctrl.edge_follow_weak_edge_conf_track_local = _sf(ef["weak_edge_conf_track_local"])
        if "strong_edge_conf_track_local" in ef:
            ctrl.edge_follow_strong_edge_conf_track_local = _sf(ef["strong_edge_conf_track_local"])
        if "low_conf_hold_ms" in ef:
            ctrl.edge_follow_low_conf_hold_s = _sf(ef["low_conf_hold_ms"]) / 1000.0
        if "low_conf_exit_ms" in ef:
            ctrl.edge_follow_low_conf_exit_s = _sf(ef["low_conf_exit_ms"]) / 1000.0
        if "recover_conf_th" in ef:
            ctrl.edge_follow_recover_conf_th = _sf(ef["recover_conf_th"])
        if "identity_yaw_mismatch_rad" in ef:
            ctrl.edge_identity_yaw_mismatch_rad = _sf(ef["identity_yaw_mismatch_rad"])
        if "identity_dist_mismatch_m" in ef:
            ctrl.edge_identity_dist_mismatch_m = _sf(ef["identity_dist_mismatch_m"])
        if "log_period_ms" in ef:
            ctrl.edge_follow_log_period_ms = _si(ef["log_period_ms"])
        if "track_local_edge_update_hz" in ef:
            ctrl.edge_follow_track_local_edge_update_hz = _sf(ef["track_local_edge_update_hz"])
        if "handoff_min_ms" in ef:
            ctrl.edge_handoff_min_s = _sf(ef["handoff_min_ms"]) / 1000.0
        if "handoff_max_ms" in ef:
            ctrl.edge_handoff_max_s = _sf(ef["handoff_max_ms"]) / 1000.0
        if "handoff_samples" in ef:
            ctrl.edge_handoff_samples = _si(ef["handoff_samples"])
        if "stale_hold_ms" in ef:
            ctrl.edge_follow_stale_hold_s = _sf(ef["stale_hold_ms"]) / 1000.0
        if "stale_fallback_state" in ef:
            ctrl.edge_follow_stale_fallback_state = str(ef["stale_fallback_state"])

    tc = yaml_data.get("target_confirm", {})
    if tc:
        ctrl = config.orchestrator.control
        if "confirm_conf_th" in tc:
            ctrl.target_confirm_conf_th = _sf(tc["confirm_conf_th"])
        if "confirm_enter_frames" in tc:
            ctrl.target_found_frames_to_confirm = _si(tc["confirm_enter_frames"])
        if "confirm_min_ms" in tc:
            ctrl.target_confirm_min_s = _sf(tc["confirm_min_ms"]) / 1000.0
        if "confirm_timeout_ms" in tc:
            ctrl.target_confirm_timeout_s = _sf(tc["confirm_timeout_ms"]) / 1000.0
        if "confirm_lost_hold_ms" in tc:
            ctrl.target_confirm_lost_hold_s = _sf(tc["confirm_lost_hold_ms"]) / 1000.0
        if "window_ms" in tc:
            ctrl.target_confirm_window_s = _sf(tc["window_ms"]) / 1000.0
        if "confirm_found_ratio_th" in tc:
            ctrl.target_confirm_found_ratio_th = _sf(tc["confirm_found_ratio_th"])
        if "min_bbox_area" in tc:
            ctrl.target_confirm_min_bbox_area = _sf(tc["min_bbox_area"])

    tl = yaml_data.get("target_locked", {})
    if tl:
        ctrl = config.orchestrator.control
        if "lock_conf_th" in tl:
            ctrl.target_lock_conf_th = _sf(tl["lock_conf_th"])
        if "lock_found_ratio_th" in tl:
            ctrl.target_lock_found_ratio_th = _sf(tl["lock_found_ratio_th"])
        if "lock_stable_ms" in tl:
            ctrl.target_lock_stable_s = _sf(tl["lock_stable_ms"]) / 1000.0
        if "center_jitter_th" in tl:
            ctrl.target_lock_center_jitter_th = _sf(tl["center_jitter_th"])
        if "locked_lost_hold_ms" in tl:
            ctrl.target_lock_lost_hold_s = _sf(tl["locked_lost_hold_ms"]) / 1000.0
        if "freeze_after_locked_ms" in tl:
            ctrl.target_locked_freeze_after_s = _sf(tl["freeze_after_locked_ms"]) / 1000.0


def _load_and_merge_car_cmd_params(config: SystemGlobalConfig, file_path: Path) -> None:
    yaml_data = load_yaml_file(file_path)
    if not yaml_data:
        return
    cc = yaml_data.get("car_cmd", {})
    if cc:
        car = config.orchestrator.car
        if "send_period_ms" in cc:
            car.send_period_ms = int(cc["send_period_ms"])
        if "uart_keepalive_hz" in cc:
            car.uart_keepalive_hz = float(cc["uart_keepalive_hz"])
        if "min_uart_keepalive_hz" in cc:
            car.min_uart_keepalive_hz = float(cc["min_uart_keepalive_hz"])
        if "hold_ms" in cc:
            car.cmd_hold_ms = int(cc["hold_ms"])
        if "motion_hold_ms" in cc:
            car.motion_hold_ms = int(cc["motion_hold_ms"])
        if "hard_stale_stop_ms" in cc:
            car.hard_stale_stop_ms = int(cc["hard_stale_stop_ms"])
        if "soft_stale_hold_enable" in cc:
            car.soft_stale_hold_enable = bool(cc["soft_stale_hold_enable"])
        if "max_vx_mps" in cc:
            car.max_vx_mps = float(cc["max_vx_mps"])
        if "max_vy_mps" in cc:
            car.max_vy_mps = float(cc["max_vy_mps"])
        if "max_wz_radps" in cc:
            car.max_wz_radps = float(cc["max_wz_radps"])
        if "stop_on_state_enter" in cc:
            car.stop_on_state_enter = bool(cc["stop_on_state_enter"])


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

    # 5. Resolve and load stage_params_file (Fail fast ONLY when explicitly specified)
    stage_file_str = config.orchestrator.runtime.stage_params_file
    if stage_file_str:
        system_dir = target_path.parent if target_path else None
        resolved_stage_path = _resolve_config_path(stage_file_str, system_dir)
        if resolved_stage_path.is_file():
            _load_and_merge_stage_params(config, resolved_stage_path)
            if str(resolved_stage_path) not in config.orchestrator.runtime.loaded_config_files:
                config.orchestrator.runtime.loaded_config_files.append(str(resolved_stage_path))
        else:
            raise FileNotFoundError(f"Configured stage_params_file not found: {stage_file_str} (resolved to: {resolved_stage_path})")

    # 6. Resolve and load car_cmd_params_file (Fail fast ONLY when explicitly specified)
    car_file_str = config.orchestrator.runtime.car_cmd_params_file
    if car_file_str:
        system_dir = target_path.parent if target_path else None
        resolved_car_path = _resolve_config_path(car_file_str, system_dir)
        if resolved_car_path.is_file():
            _load_and_merge_car_cmd_params(config, resolved_car_path)
            if str(resolved_car_path) not in config.orchestrator.runtime.loaded_config_files:
                config.orchestrator.runtime.loaded_config_files.append(str(resolved_car_path))
        else:
            raise FileNotFoundError(f"Configured car_cmd_params_file not found: {car_file_str} (resolved to: {resolved_car_path})")

    # 7. Re-apply env overrides at the very end to ensure env variables take highest priority
    apply_env_overrides(config)

    # 8. Validate configuration
    validate_config(config)

    # Synchronize orchestrator speed configs
    sync_orchestrator_config(config)

    return config


def get_config(reload: bool = False) -> SystemGlobalConfig:
    """Return the global configuration singleton."""
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None or reload:
        _GLOBAL_CONFIG = load_global_config()
    return _GLOBAL_CONFIG
