#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from .schema import OrchestratorConfig


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    raw = str(raw).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
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


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return str(raw).strip() if raw is not None else str(default)


CONFIG = OrchestratorConfig()

CONFIG.runtime.project_root = _env_str("ORCH_PROJECT_ROOT", "/home/aidlux/2026/orchestrator")
CONFIG.runtime.runs_dir = _env_str("ORCH_RUNS_DIR", "/home/aidlux/2026/orchestrator/runs")
CONFIG.runtime.tick_hz = _env_float("ORCH_TICK_HZ", 10.0)
CONFIG.runtime.log_mode = _env_str("ORCH_LOG_MODE", "concise")
CONFIG.runtime.debug = _env_bool("ORCH_DEBUG", False)
CONFIG.runtime.state_block_period_s = _env_float("ORCH_STATE_BLOCK_PERIOD_S", 1.0)
CONFIG.runtime.heartbeat_period_s = _env_float("ORCH_HEARTBEAT_PERIOD_S", 1.0)

CONFIG.serial.port = _env_str("ORCH_SERIAL_PORT", "/dev/ttyHS1")
CONFIG.serial.baudrate = _env_int("ORCH_SERIAL_BAUDRATE", 115200)
CONFIG.serial.timeout_s = _env_float("ORCH_SERIAL_TIMEOUT_S", 0.10)
CONFIG.serial.dry_run = _env_bool("ORCH_SERIAL_DRY_RUN", True)
CONFIG.serial.readback_enabled = _env_bool("ORCH_READBACK_ENABLED", True)
CONFIG.serial.dry_run_echo_stdout = _env_bool("ORCH_DRY_RUN_ECHO_STDOUT", True)
CONFIG.serial.dry_run_echo_on_change_only = _env_bool("ORCH_DRY_RUN_ECHO_ON_CHANGE_ONLY", True)
CONFIG.serial.dry_run_echo_summary_period_s = _env_float("ORCH_DRY_RUN_ECHO_SUMMARY_PERIOD_S", 5.0)
CONFIG.serial.dry_run_quiet_idle_stop = _env_bool("ORCH_DRY_RUN_QUIET_IDLE_STOP", True)
CONFIG.serial.uart_lowfreq_period_s = _env_float("ORCH_UART_LOWFREQ_PERIOD_S", 5.0)

CONFIG.control.cmd_confidence_th = 0.60
CONFIG.control.target_obs_max_age_s = _env_float("ORCH_TARGET_OBS_MAX_AGE_S", 1.00)
CONFIG.control.home_obs_max_age_s = _env_float("ORCH_HOME_OBS_MAX_AGE_S", 1.00)
CONFIG.control.search_timeout_s = 20.0
CONFIG.control.return_search_timeout_s = 15.0
CONFIG.control.req_resend_period_s = 1.0
CONFIG.control.found_frames_to_approach = _env_int("ORCH_FOUND_FRAMES_TO_APPROACH", 2)
CONFIG.control.initial_found_frames_to_search = _env_int("ORCH_INITIAL_FOUND_FRAMES_TO_SEARCH", 1)
CONFIG.control.reacquire_found_frames_to_search = _env_int("ORCH_REACQUIRE_FOUND_FRAMES_TO_SEARCH", 2)
CONFIG.control.arrived_frames_to_stop = 2
CONFIG.control.lost_frames_to_search = _env_int("ORCH_LOST_FRAMES_TO_SEARCH", 6)
CONFIG.control.tag_found_frames_to_track = 2
CONFIG.control.tag_arrived_frames_to_stop = 2
CONFIG.control.tag_lost_frames_to_search = _env_int("ORCH_TAG_LOST_FRAMES_TO_SEARCH", 4)
CONFIG.control.search_lost_hold_s = _env_float("ORCH_SEARCH_LOST_HOLD_S", 1.20)
CONFIG.control.return_lost_hold_s = _env_float("ORCH_RETURN_LOST_HOLD_S", 1.00)
CONFIG.control.search_min_dwell_s = _env_float("ORCH_SEARCH_MIN_DWELL_S", 0.80)
CONFIG.control.return_min_dwell_s = _env_float("ORCH_RETURN_MIN_DWELL_S", 0.60)
CONFIG.control.dead_zone_x = 0.10
CONFIG.control.dead_zone_yaw = 0.10
CONFIG.control.align_turn_threshold = 0.18
CONFIG.control.stop_size_norm = 0.45
CONFIG.control.return_stop_distance_m = 0.35
CONFIG.control.car_timeout_to_stop = True
CONFIG.control.car_fault_to_fail = True
CONFIG.control.car_estop_to_stop = True
CONFIG.control.post_stop_ignore_s = 0.80
CONFIG.control.vision_req_fail_to_stop = True
CONFIG.control.vision_req_fail_threshold = 2

CONFIG.car.search_turn_norm_min = _env_float("ORCH_SEARCH_TURN_NORM_MIN", 0.12)
CONFIG.car.search_turn_norm_max = _env_float("ORCH_SEARCH_TURN_NORM_MAX", 0.55)
CONFIG.car.search_vx_norm_min = _env_float("ORCH_SEARCH_VX_NORM_MIN", 0.06)
CONFIG.car.search_vx_norm_max = _env_float("ORCH_SEARCH_VX_NORM_MAX", 0.30)
CONFIG.car.return_turn_norm_min = _env_float("ORCH_RETURN_TURN_NORM_MIN", 0.20)
CONFIG.car.return_turn_norm_max = _env_float("ORCH_RETURN_TURN_NORM_MAX", 0.75)
CONFIG.car.return_vx_norm_min = _env_float("ORCH_RETURN_VX_NORM_MIN", 0.10)
CONFIG.car.return_vx_norm_max = _env_float("ORCH_RETURN_VX_NORM_MAX", 0.45)
CONFIG.car.search_spin_only_x_th = _env_float("ORCH_SEARCH_SPIN_ONLY_X_TH", 0.82)
CONFIG.car.search_forward_align_exp = _env_float("ORCH_SEARCH_FORWARD_ALIGN_EXP", 2.0)
CONFIG.car.mode_line_on_change = True
CONFIG.car.mode_line_every_cmd = False
CONFIG.car.serial_float_digits = 3

CONFIG.task_cmd_in.transport = _env_str("ORCH_TASK_CMD_IN_TRANSPORT", CONFIG.task_cmd_in.transport)
CONFIG.task_cmd_in.host = _env_str("ORCH_TASK_CMD_IN_HOST", CONFIG.task_cmd_in.host)
CONFIG.task_cmd_in.port = _env_int("ORCH_TASK_CMD_IN_PORT", CONFIG.task_cmd_in.port)
CONFIG.task_cmd_in.uds_path = _env_str("ORCH_TASK_CMD_IN_UDS", CONFIG.task_cmd_in.uds_path)

CONFIG.task_ack_out.transport = _env_str("ORCH_TASK_ACK_OUT_TRANSPORT", CONFIG.task_ack_out.transport)
CONFIG.task_ack_out.host = _env_str("ORCH_TASK_ACK_OUT_HOST", CONFIG.task_ack_out.host)
CONFIG.task_ack_out.port = _env_int("ORCH_TASK_ACK_OUT_PORT", CONFIG.task_ack_out.port)
CONFIG.task_ack_out.uds_path = _env_str("ORCH_TASK_ACK_OUT_UDS", CONFIG.task_ack_out.uds_path)
CONFIG.task_ack_out.send_mode = _env_str("ORCH_TASK_ACK_SEND_MODE", CONFIG.task_ack_out.send_mode)

CONFIG.vision_obs_in.transport = _env_str("ORCH_VISION_OBS_IN_TRANSPORT", CONFIG.vision_obs_in.transport)
CONFIG.vision_obs_in.host = _env_str("ORCH_VISION_OBS_IN_HOST", CONFIG.vision_obs_in.host)
CONFIG.vision_obs_in.port = _env_int("ORCH_VISION_OBS_IN_PORT", CONFIG.vision_obs_in.port)
CONFIG.vision_obs_in.uds_path = _env_str("ORCH_VISION_OBS_IN_UDS", CONFIG.vision_obs_in.uds_path)

CONFIG.vision_req_out.transport = _env_str("ORCH_VISION_REQ_OUT_TRANSPORT", CONFIG.vision_req_out.transport)
CONFIG.vision_req_out.host = _env_str("ORCH_VISION_REQ_OUT_HOST", CONFIG.vision_req_out.host)
CONFIG.vision_req_out.port = _env_int("ORCH_VISION_REQ_OUT_PORT", CONFIG.vision_req_out.port)
CONFIG.vision_req_out.uds_path = _env_str("ORCH_VISION_REQ_OUT_UDS", CONFIG.vision_req_out.uds_path)
CONFIG.vision_req_out.async_enabled = _env_bool("ORCH_VISION_REQ_ASYNC", True)
CONFIG.vision_req_out.async_queue_size = _env_int("ORCH_VISION_REQ_ASYNC_QUEUE_SIZE", 64)
CONFIG.vision_req_out.async_drop_oldest = _env_bool("ORCH_VISION_REQ_ASYNC_DROP_OLDEST", True)
CONFIG.vision_req_out.send_mode = _env_str("ORCH_VISION_REQ_SEND_MODE", "oneshot")

CONFIG.tts_event_out.transport = _env_str("ORCH_TTS_EVENT_OUT_TRANSPORT", CONFIG.tts_event_out.transport)
CONFIG.tts_event_out.host = _env_str("ORCH_TTS_EVENT_OUT_HOST", CONFIG.tts_event_out.host)
CONFIG.tts_event_out.port = _env_int("ORCH_TTS_EVENT_OUT_PORT", CONFIG.tts_event_out.port)
CONFIG.tts_event_out.uds_path = _env_str("ORCH_TTS_EVENT_OUT_UDS", CONFIG.tts_event_out.uds_path)
CONFIG.tts_event_out.async_enabled = _env_bool("ORCH_TTS_EVENT_ASYNC", True)
CONFIG.tts_event_out.async_queue_size = _env_int("ORCH_TTS_EVENT_ASYNC_QUEUE_SIZE", 32)
CONFIG.tts_event_out.async_drop_oldest = _env_bool("ORCH_TTS_EVENT_ASYNC_DROP_OLDEST", True)
