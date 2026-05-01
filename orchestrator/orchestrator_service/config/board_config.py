#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from pathlib import Path
from typing import Any, Dict

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


def _ms_to_s(value: Any, default_s: float) -> float:
    try:
        return max(0.0, float(value) / 1000.0)
    except Exception:
        return float(default_s)


def _ms_to_frames(value: Any, tick_hz: float, default_frames: int) -> int:
    try:
        return max(1, int(round((float(value) / 1000.0) * max(1.0, float(tick_hz)))))
    except Exception:
        return int(default_frames)


def _load_config_dict(path: str) -> Dict[str, Any]:
    file_path = str(path or "").strip()
    if not file_path or not Path(file_path).is_file():
        return {}
    if file_path.lower().endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as fp:
            return dict(json.load(fp) or {})
    if file_path.lower().endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
        except ImportError:
            return _parse_simple_yaml(Path(file_path).read_text(encoding="utf-8"))
        with open(file_path, "r", encoding="utf-8") as fp:
            return dict(yaml.safe_load(fp) or {})
    return {}


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
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
        if not line.strip() or ":" not in line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, value = line.strip().split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if value.strip() == "":
            child: Dict[str, Any] = {}
            current[key.strip()] = child
            stack.append((indent, child))
        else:
            current[key.strip()] = _scalar(value)
    return root


def _loaded(path: str) -> None:
    path = str(path or "").strip()
    if path and Path(path).is_file() and path not in CONFIG.runtime.loaded_config_files:
        CONFIG.runtime.loaded_config_files.append(path)


def _apply_stage_params(data: Dict[str, Any]) -> None:
    if not data:
        return
    final_lock = dict(data.get("final_lock") or {})
    controlled = dict(data.get("controlled_approach") or {})
    edge_slide = dict(data.get("edge_slide_search") or {})
    edge_follow = dict(data.get("edge_follow") or {})
    target_confirm = dict(data.get("target_confirm") or {})
    target_locked = dict(data.get("target_locked") or {})

    CONFIG.control.final_lock_yaw_tol_rad = float(final_lock.get("yaw_abs_th", CONFIG.control.final_lock_yaw_tol_rad))
    CONFIG.control.final_lock_dist_tol_m = float(final_lock.get("dist_abs_th_m", CONFIG.control.final_lock_dist_tol_m))
    CONFIG.docking.min_confidence = float(final_lock.get("edge_conf_th", CONFIG.docking.min_confidence))
    CONFIG.control.final_lock_frames_to_arrive = _ms_to_frames(
        final_lock.get("stable_ms"),
        CONFIG.runtime.tick_hz,
        CONFIG.control.final_lock_frames_to_arrive,
    )
    CONFIG.control.approach_timeout_s = _ms_to_s(final_lock.get("timeout_ms"), CONFIG.control.approach_timeout_s)

    CONFIG.docking.precise_dist_tol_m = float(controlled.get("target_dist_m", CONFIG.docking.precise_dist_tol_m))
    CONFIG.docking.approach_max_vx_norm = float(controlled.get("max_vx_norm", CONFIG.docking.approach_max_vx_norm))
    CONFIG.docking.approach_max_wz_norm = float(controlled.get("max_wz_norm", CONFIG.docking.approach_max_wz_norm))
    CONFIG.docking.dist_pid.kp = float(controlled.get("dist_kp", CONFIG.docking.dist_pid.kp))
    CONFIG.docking.yaw_pid.kp = float(controlled.get("yaw_kp", CONFIG.docking.yaw_pid.kp))

    CONFIG.car.edge_slide_vy_norm = float(edge_slide.get("slide_vy_norm", CONFIG.car.edge_slide_vy_norm))
    CONFIG.car.edge_slide_weak_vy_norm = float(edge_slide.get("weak_slide_vy_norm", CONFIG.car.edge_slide_weak_vy_norm))
    CONFIG.car.edge_slide_max_vx_norm = float(edge_slide.get("max_vx_correction_norm", CONFIG.car.edge_slide_max_vx_norm))
    CONFIG.car.edge_slide_max_wz_norm = float(edge_slide.get("max_wz_correction_norm", CONFIG.car.edge_slide_max_wz_norm))
    CONFIG.car.edge_slide_dist_kp_norm_per_m = float(edge_slide.get("dist_kp", CONFIG.car.edge_slide_dist_kp_norm_per_m))
    CONFIG.car.edge_slide_yaw_kp_norm_per_rad = float(edge_slide.get("yaw_kp", CONFIG.car.edge_slide_yaw_kp_norm_per_rad))
    CONFIG.control.edge_slide_dist_tolerance_m = float(edge_slide.get("keep_dist_tolerance_m", CONFIG.control.edge_slide_dist_tolerance_m))
    CONFIG.control.edge_follow_weak_edge_conf_track_local = float(edge_slide.get("weak_edge_conf_track_local", CONFIG.control.edge_follow_weak_edge_conf_track_local))
    CONFIG.control.edge_follow_strong_edge_conf_track_local = float(edge_slide.get("strong_edge_conf_track_local", CONFIG.control.edge_follow_strong_edge_conf_track_local))
    CONFIG.control.edge_slide_pause_hold_s = _ms_to_s(edge_slide.get("pause_hold_ms"), CONFIG.control.edge_slide_pause_hold_s)
    CONFIG.control.edge_slide_recover_timeout_s = _ms_to_s(edge_slide.get("recover_timeout_ms"), CONFIG.control.edge_slide_recover_timeout_s)
    CONFIG.control.table_loss_hold_s = _ms_to_s(edge_slide.get("edge_lost_hold_ms"), CONFIG.control.table_loss_hold_s)
    CONFIG.control.approach_min_dwell_s = _ms_to_s(edge_slide.get("min_dwell_ms"), CONFIG.control.approach_min_dwell_s)
    CONFIG.control.edge_slide_fallback_state = str(edge_slide.get("fallback_state", CONFIG.control.edge_slide_fallback_state)).strip().upper()
    CONFIG.control.edge_slide_direct_fallback_to_controlled_approach = bool(edge_slide.get("direct_fallback_to_controlled_approach", CONFIG.control.edge_slide_direct_fallback_to_controlled_approach))

    CONFIG.control.table_edge_obs_max_age_ms = int(edge_follow.get("table_edge_obs_max_age_ms", CONFIG.control.table_edge_obs_max_age_ms))
    CONFIG.control.edge_follow_log_period_ms = int(edge_follow.get("log_period_ms", CONFIG.control.edge_follow_log_period_ms))
    CONFIG.control.edge_follow_min_edge_conf = float(edge_follow.get("min_edge_conf", CONFIG.control.edge_follow_min_edge_conf))
    CONFIG.control.edge_follow_min_edge_conf_table_edge_perception = float(edge_follow.get("min_edge_conf_table_edge_perception", CONFIG.control.edge_follow_min_edge_conf_table_edge_perception))
    CONFIG.control.edge_follow_min_edge_conf_track_local = float(edge_follow.get("min_edge_conf_track_local", CONFIG.control.edge_follow_min_edge_conf_track_local))
    CONFIG.control.edge_follow_weak_edge_conf_track_local = float(edge_follow.get("weak_edge_conf_track_local", CONFIG.control.edge_follow_weak_edge_conf_track_local))
    CONFIG.control.edge_follow_strong_edge_conf_track_local = float(edge_follow.get("strong_edge_conf_track_local", CONFIG.control.edge_follow_strong_edge_conf_track_local))
    CONFIG.control.edge_follow_low_conf_hold_s = _ms_to_s(edge_follow.get("low_conf_hold_ms"), CONFIG.control.edge_follow_low_conf_hold_s)
    CONFIG.control.edge_follow_low_conf_exit_s = _ms_to_s(edge_follow.get("low_conf_exit_ms"), CONFIG.control.edge_follow_low_conf_exit_s)
    CONFIG.control.edge_follow_recover_conf_th = float(edge_follow.get("recover_conf_th", CONFIG.control.edge_follow_recover_conf_th))
    CONFIG.control.edge_identity_yaw_mismatch_rad = float(edge_follow.get("identity_yaw_mismatch_rad", CONFIG.control.edge_identity_yaw_mismatch_rad))
    CONFIG.control.edge_identity_dist_mismatch_m = float(edge_follow.get("identity_dist_mismatch_m", CONFIG.control.edge_identity_dist_mismatch_m))
    CONFIG.control.edge_follow_stale_fallback_state = str(edge_follow.get("stale_fallback_state", CONFIG.control.edge_follow_stale_fallback_state)).strip().upper()
    CONFIG.control.edge_follow_stale_hold_s = _ms_to_s(edge_follow.get("stale_hold_ms"), CONFIG.control.edge_follow_stale_hold_s)
    CONFIG.control.edge_follow_track_local_edge_update_hz = float(
        edge_follow.get("track_local_edge_update_hz", CONFIG.control.edge_follow_track_local_edge_update_hz)
    )
    CONFIG.control.edge_handoff_min_s = _ms_to_s(edge_follow.get("handoff_min_ms"), CONFIG.control.edge_handoff_min_s)
    CONFIG.control.edge_handoff_max_s = _ms_to_s(edge_follow.get("handoff_max_ms"), CONFIG.control.edge_handoff_max_s)
    CONFIG.control.edge_handoff_samples = int(edge_follow.get("handoff_samples", CONFIG.control.edge_handoff_samples))

    CONFIG.control.target_confirm_conf_th = float(target_confirm.get("confirm_conf_th", CONFIG.control.target_confirm_conf_th))
    CONFIG.control.target_found_frames_to_confirm = int(target_confirm.get("confirm_enter_frames", CONFIG.control.target_found_frames_to_confirm))
    CONFIG.control.target_confirm_dwell_s = _ms_to_s(target_confirm.get("confirm_dwell_ms"), CONFIG.control.target_confirm_dwell_s)
    CONFIG.control.target_confirm_min_s = _ms_to_s(
        target_confirm.get("confirm_min_ms", target_confirm.get("confirm_dwell_ms")),
        CONFIG.control.target_confirm_min_s,
    )
    CONFIG.control.target_confirm_timeout_s = _ms_to_s(
        target_confirm.get("confirm_timeout_ms"),
        CONFIG.control.target_confirm_timeout_s,
    )
    CONFIG.control.target_confirm_lost_hold_s = _ms_to_s(
        target_confirm.get("confirm_lost_hold_ms", target_confirm.get("lost_hold_ms")),
        CONFIG.control.target_confirm_lost_hold_s,
    )
    CONFIG.control.target_confirm_min_bbox_area = float(
        target_confirm.get("min_bbox_area", CONFIG.control.target_confirm_min_bbox_area)
    )
    CONFIG.control.target_confirm_window_s = _ms_to_s(
        target_confirm.get("window_ms"),
        CONFIG.control.target_confirm_window_s,
    )
    CONFIG.control.target_confirm_found_ratio_th = float(
        target_confirm.get("confirm_found_ratio_th", CONFIG.control.target_confirm_found_ratio_th)
    )

    CONFIG.control.target_lock_conf_th = float(target_locked.get("lock_conf_th", CONFIG.control.target_lock_conf_th))
    CONFIG.control.target_lock_found_ratio_th = float(target_locked.get("lock_found_ratio_th", CONFIG.control.target_lock_found_ratio_th))
    CONFIG.control.target_lock_stable_s = _ms_to_s(target_locked.get("lock_stable_ms"), CONFIG.control.target_lock_stable_s)
    CONFIG.control.target_lock_settle_s = CONFIG.control.target_lock_stable_s
    CONFIG.control.target_lock_center_jitter_th = float(target_locked.get("center_jitter_th", CONFIG.control.target_lock_center_jitter_th))
    CONFIG.control.target_lock_lost_hold_s = _ms_to_s(
        target_locked.get("locked_lost_hold_ms", target_locked.get("lost_hold_ms")),
        CONFIG.control.target_lock_lost_hold_s,
    )
    CONFIG.control.target_locked_freeze_after_s = _ms_to_s(
        target_locked.get("freeze_after_locked_ms"),
        CONFIG.control.target_locked_freeze_after_s,
    )


def _apply_car_cmd_params(data: Dict[str, Any]) -> None:
    car_cmd = dict(data.get("car_cmd") or data or {})
    CONFIG.car.send_period_ms = int(car_cmd.get("send_period_ms", CONFIG.car.send_period_ms))
    if CONFIG.car.send_period_ms > 0:
        CONFIG.runtime.tick_hz = 1000.0 / float(CONFIG.car.send_period_ms)
    CONFIG.car.cmd_hold_ms = int(car_cmd.get("hold_ms", CONFIG.car.cmd_hold_ms))
    CONFIG.car.max_vx_norm = float(car_cmd.get("max_vx_norm", CONFIG.car.max_vx_norm))
    CONFIG.car.max_vy_norm = float(car_cmd.get("max_vy_norm", CONFIG.car.max_vy_norm))
    CONFIG.car.max_wz_norm = float(car_cmd.get("max_wz_norm", CONFIG.car.max_wz_norm))
    CONFIG.car.stop_on_state_enter = bool(car_cmd.get("stop_on_state_enter", CONFIG.car.stop_on_state_enter))


CONFIG = OrchestratorConfig()

_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LOG_DIR = _DEFAULT_PROJECT_ROOT / "logs"
_DEFAULT_RUNS_DIR = _DEFAULT_PROJECT_ROOT / "runs"
_DEFAULT_PID_DIR = _DEFAULT_PROJECT_ROOT / "pids"

CONFIG.runtime.project_root = _env_str("ORCH_PROJECT_ROOT", str(_DEFAULT_PROJECT_ROOT))
CONFIG.runtime.log_dir = _env_str("ORCH_LOG_DIR", str(_DEFAULT_LOG_DIR))
CONFIG.runtime.log_file = _env_str("ORCH_LOG_FILE", f"{CONFIG.runtime.log_dir}/orchestrator.log")
CONFIG.runtime.runs_dir = _env_str("ORCH_RUNS_DIR", str(_DEFAULT_RUNS_DIR))
CONFIG.runtime.pid_dir = _env_str("ORCH_PID_DIR", str(_DEFAULT_PID_DIR))
CONFIG.runtime.pid_file = _env_str("ORCH_PID_FILE", f"{CONFIG.runtime.pid_dir}/orchestrator.pid")
CONFIG.runtime.stack_run_id = _env_str("STACK_RUN_ID", "")
CONFIG.runtime.tick_hz = _env_float("ORCH_TICK_HZ", CONFIG.runtime.tick_hz)
CONFIG.runtime.log_mode = _env_str("ORCH_LOG_MODE", CONFIG.runtime.log_mode)
CONFIG.runtime.log_enabled = _env_bool("ORCH_LOG_ENABLED", CONFIG.runtime.log_enabled)
CONFIG.runtime.debug = _env_bool("ORCH_DEBUG", CONFIG.runtime.debug)
CONFIG.runtime.state_block_period_s = _env_float("ORCH_STATE_BLOCK_PERIOD_S", CONFIG.runtime.state_block_period_s)
CONFIG.runtime.heartbeat_period_s = _env_float("ORCH_HEARTBEAT_PERIOD_S", CONFIG.runtime.heartbeat_period_s)
CONFIG.runtime.stage_params_file = _env_str(
    "ORCH_STAGE_PARAMS_FILE",
    str(Path(CONFIG.runtime.project_root) / "configs" / "stage_params.yaml"),
)
CONFIG.runtime.car_cmd_params_file = _env_str(
    "ORCH_CAR_CMD_PARAMS_FILE",
    str(Path(CONFIG.runtime.project_root) / "configs" / "car_cmd_params.yaml"),
)
_apply_car_cmd_params(_load_config_dict(CONFIG.runtime.car_cmd_params_file))
_loaded(CONFIG.runtime.car_cmd_params_file)
CONFIG.runtime.tick_hz = _env_float("ORCH_TICK_HZ", CONFIG.runtime.tick_hz)
_apply_stage_params(_load_config_dict(CONFIG.runtime.stage_params_file))
_loaded(CONFIG.runtime.stage_params_file)

CONFIG.serial.port = _env_str("ORCH_SERIAL_PORT", CONFIG.serial.port)
CONFIG.serial.baudrate = _env_int("ORCH_SERIAL_BAUDRATE", CONFIG.serial.baudrate)
CONFIG.serial.timeout_s = _env_float("ORCH_SERIAL_TIMEOUT_S", CONFIG.serial.timeout_s)
CONFIG.serial.dry_run = _env_bool("ORCH_SERIAL_DRY_RUN", True)
CONFIG.serial.readback_enabled = _env_bool("ORCH_READBACK_ENABLED", CONFIG.serial.readback_enabled)
CONFIG.serial.dry_run_echo_stdout = _env_bool("ORCH_DRY_RUN_ECHO_STDOUT", CONFIG.serial.dry_run_echo_stdout)
CONFIG.serial.dry_run_echo_on_change_only = _env_bool("ORCH_DRY_RUN_ECHO_ON_CHANGE_ONLY", CONFIG.serial.dry_run_echo_on_change_only)
CONFIG.serial.dry_run_echo_summary_period_s = _env_float("ORCH_DRY_RUN_ECHO_SUMMARY_PERIOD_S", CONFIG.serial.dry_run_echo_summary_period_s)
CONFIG.serial.dry_run_quiet_idle_stop = _env_bool("ORCH_DRY_RUN_QUIET_IDLE_STOP", CONFIG.serial.dry_run_quiet_idle_stop)
CONFIG.serial.uart_lowfreq_period_s = _env_float("ORCH_UART_LOWFREQ_PERIOD_S", CONFIG.serial.uart_lowfreq_period_s)

CONFIG.control.cmd_confidence_th = _env_float("ORCH_CMD_CONFIDENCE_TH", CONFIG.control.cmd_confidence_th)
CONFIG.control.target_obs_max_age_s = _env_float("ORCH_TARGET_OBS_MAX_AGE_S", CONFIG.control.target_obs_max_age_s)
CONFIG.control.table_obs_max_age_s = _env_float("ORCH_TABLE_OBS_MAX_AGE_S", CONFIG.control.table_obs_max_age_s)
CONFIG.control.home_obs_max_age_s = _env_float("ORCH_HOME_OBS_MAX_AGE_S", CONFIG.control.home_obs_max_age_s)
CONFIG.control.search_table_timeout_s = _env_float("ORCH_SEARCH_TABLE_TIMEOUT_S", CONFIG.control.search_table_timeout_s)
CONFIG.control.approach_timeout_s = _env_float("ORCH_APPROACH_TIMEOUT_S", CONFIG.control.approach_timeout_s)
CONFIG.control.target_search_timeout_s = _env_float("ORCH_TARGET_SEARCH_TIMEOUT_S", CONFIG.control.target_search_timeout_s)
CONFIG.control.return_search_timeout_s = _env_float("ORCH_RETURN_TIMEOUT_S", CONFIG.control.return_search_timeout_s)
CONFIG.control.req_resend_period_s = _env_float("ORCH_REQ_RESEND_PERIOD_S", CONFIG.control.req_resend_period_s)

CONFIG.control.table_found_frames_to_approach = _env_int("ORCH_TABLE_FOUND_FRAMES", CONFIG.control.table_found_frames_to_approach)
CONFIG.control.table_lost_frames_to_reacquire = _env_int("ORCH_TABLE_LOST_FRAMES", CONFIG.control.table_lost_frames_to_reacquire)
CONFIG.control.table_loss_hold_s = _env_float("ORCH_TABLE_LOSS_HOLD_S", CONFIG.control.table_loss_hold_s)
CONFIG.control.approach_min_dwell_s = _env_float("ORCH_APPROACH_MIN_DWELL_S", CONFIG.control.approach_min_dwell_s)
CONFIG.control.coarse_align_frames_to_advance = _env_int("ORCH_COARSE_ALIGN_FRAMES", CONFIG.control.coarse_align_frames_to_advance)
CONFIG.control.coarse_align_done_rad = _env_float("ORCH_COARSE_ALIGN_DONE_RAD", CONFIG.control.coarse_align_done_rad)
CONFIG.control.final_lock_frames_to_arrive = _env_int("ORCH_FINAL_LOCK_FRAMES", CONFIG.control.final_lock_frames_to_arrive)
CONFIG.control.final_lock_yaw_tol_rad = _env_float("ORCH_FINAL_LOCK_YAW_TOL", CONFIG.control.final_lock_yaw_tol_rad)
CONFIG.control.final_lock_dist_tol_m = _env_float("ORCH_FINAL_LOCK_DIST_TOL", CONFIG.control.final_lock_dist_tol_m)
CONFIG.control.final_lock_lateral_tol_m = _env_float("ORCH_FINAL_LOCK_LAT_TOL", CONFIG.control.final_lock_lateral_tol_m)
CONFIG.control.edge_settle_s = _env_float("ORCH_EDGE_SETTLE_S", CONFIG.control.edge_settle_s)
CONFIG.control.dock_retry_limit = _env_int("ORCH_DOCK_RETRY_LIMIT", CONFIG.control.dock_retry_limit)
CONFIG.control.dock_retry_backoff_s = _env_float("ORCH_DOCK_RETRY_BACKOFF_S", CONFIG.control.dock_retry_backoff_s)

CONFIG.control.search_target_init_hold_s = _env_float("ORCH_SEARCH_TARGET_INIT_HOLD_S", CONFIG.control.search_target_init_hold_s)
CONFIG.control.target_found_frames_to_confirm = _env_int("ORCH_TARGET_CONFIRM_FRAMES", CONFIG.control.target_found_frames_to_confirm)
CONFIG.control.target_confirm_conf_th = _env_float("ORCH_TARGET_CONFIRM_CONF_TH", CONFIG.control.target_confirm_conf_th)
CONFIG.control.target_confirm_dwell_s = _env_float("ORCH_TARGET_CONFIRM_DWELL_S", CONFIG.control.target_confirm_dwell_s)
CONFIG.control.target_confirm_min_s = _env_float("ORCH_TARGET_CONFIRM_MIN_S", CONFIG.control.target_confirm_min_s)
CONFIG.control.target_confirm_timeout_s = _env_float("ORCH_TARGET_CONFIRM_TIMEOUT_S", CONFIG.control.target_confirm_timeout_s)
CONFIG.control.target_confirm_lost_frames = _env_int("ORCH_TARGET_CONFIRM_LOST_FRAMES", CONFIG.control.target_confirm_lost_frames)
CONFIG.control.target_confirm_lost_hold_s = _env_float("ORCH_TARGET_CONFIRM_LOST_HOLD_S", CONFIG.control.target_confirm_lost_hold_s)
CONFIG.control.target_confirm_min_bbox_area = _env_float("ORCH_TARGET_CONFIRM_MIN_BBOX_AREA", CONFIG.control.target_confirm_min_bbox_area)
CONFIG.control.target_confirm_window_s = _env_float("ORCH_TARGET_CONFIRM_WINDOW_S", CONFIG.control.target_confirm_window_s)
CONFIG.control.target_confirm_found_ratio_th = _env_float("ORCH_TARGET_CONFIRM_FOUND_RATIO_TH", CONFIG.control.target_confirm_found_ratio_th)
CONFIG.control.target_lock_conf_th = _env_float("ORCH_TARGET_LOCK_CONF_TH", CONFIG.control.target_lock_conf_th)
CONFIG.control.target_lock_found_ratio_th = _env_float("ORCH_TARGET_LOCK_FOUND_RATIO_TH", CONFIG.control.target_lock_found_ratio_th)
CONFIG.control.target_lock_settle_s = _env_float("ORCH_TARGET_LOCK_SETTLE_S", CONFIG.control.target_lock_settle_s)
CONFIG.control.target_lock_stable_s = _env_float("ORCH_TARGET_LOCK_STABLE_S", CONFIG.control.target_lock_stable_s)
if "ORCH_TARGET_LOCK_STABLE_S" not in os.environ and "ORCH_TARGET_LOCK_SETTLE_S" in os.environ:
    CONFIG.control.target_lock_stable_s = CONFIG.control.target_lock_settle_s
CONFIG.control.target_lock_center_jitter_th = _env_float("ORCH_TARGET_LOCK_CENTER_JITTER_TH", CONFIG.control.target_lock_center_jitter_th)
CONFIG.control.target_lock_lost_hold_s = _env_float("ORCH_TARGET_LOCK_LOST_HOLD_S", CONFIG.control.target_lock_lost_hold_s)
CONFIG.control.target_locked_freeze_after_s = _env_float("ORCH_TARGET_LOCKED_FREEZE_AFTER_S", CONFIG.control.target_locked_freeze_after_s)
CONFIG.control.freeze_settle_s = _env_float("ORCH_FREEZE_SETTLE_S", CONFIG.control.freeze_settle_s)
CONFIG.control.edge_slide_pause_s = _env_float("ORCH_EDGE_SLIDE_PAUSE_S", CONFIG.control.edge_slide_pause_s)
CONFIG.control.table_edge_obs_max_age_ms = _env_int("ORCH_EDGE_FOLLOW_TABLE_EDGE_OBS_MAX_AGE_MS", CONFIG.control.table_edge_obs_max_age_ms)
CONFIG.control.edge_follow_log_period_ms = _env_int("ORCH_EDGE_FOLLOW_LOG_PERIOD_MS", CONFIG.control.edge_follow_log_period_ms)
CONFIG.control.edge_follow_min_edge_conf = _env_float("ORCH_EDGE_FOLLOW_MIN_EDGE_CONF", CONFIG.control.edge_follow_min_edge_conf)
CONFIG.control.edge_follow_min_edge_conf_table_edge_perception = _env_float("ORCH_EDGE_FOLLOW_MIN_EDGE_CONF_TABLE_EDGE_PERCEPTION", CONFIG.control.edge_follow_min_edge_conf_table_edge_perception)
CONFIG.control.edge_follow_min_edge_conf_track_local = _env_float("ORCH_EDGE_FOLLOW_MIN_EDGE_CONF_TRACK_LOCAL", CONFIG.control.edge_follow_min_edge_conf_track_local)
CONFIG.control.edge_follow_weak_edge_conf_track_local = _env_float("ORCH_EDGE_FOLLOW_WEAK_EDGE_CONF_TRACK_LOCAL", CONFIG.control.edge_follow_weak_edge_conf_track_local)
CONFIG.control.edge_follow_strong_edge_conf_track_local = _env_float("ORCH_EDGE_FOLLOW_STRONG_EDGE_CONF_TRACK_LOCAL", CONFIG.control.edge_follow_strong_edge_conf_track_local)
CONFIG.control.edge_follow_low_conf_hold_s = _env_float("ORCH_EDGE_FOLLOW_LOW_CONF_HOLD_S", CONFIG.control.edge_follow_low_conf_hold_s)
CONFIG.control.edge_follow_low_conf_exit_s = _env_float("ORCH_EDGE_FOLLOW_LOW_CONF_EXIT_S", CONFIG.control.edge_follow_low_conf_exit_s)
CONFIG.control.edge_follow_recover_conf_th = _env_float("ORCH_EDGE_FOLLOW_RECOVER_CONF_TH", CONFIG.control.edge_follow_recover_conf_th)
CONFIG.control.edge_identity_yaw_mismatch_rad = _env_float("ORCH_EDGE_IDENTITY_YAW_MISMATCH_RAD", CONFIG.control.edge_identity_yaw_mismatch_rad)
CONFIG.control.edge_identity_dist_mismatch_m = _env_float("ORCH_EDGE_IDENTITY_DIST_MISMATCH_M", CONFIG.control.edge_identity_dist_mismatch_m)
CONFIG.control.edge_follow_stale_fallback_state = _env_str("ORCH_EDGE_FOLLOW_STALE_FALLBACK_STATE", CONFIG.control.edge_follow_stale_fallback_state).strip().upper()
CONFIG.control.edge_follow_stale_hold_s = _env_float("ORCH_EDGE_FOLLOW_STALE_HOLD_S", CONFIG.control.edge_follow_stale_hold_s)
CONFIG.control.edge_follow_track_local_edge_update_hz = _env_float("ORCH_EDGE_FOLLOW_TRACK_LOCAL_EDGE_UPDATE_HZ", CONFIG.control.edge_follow_track_local_edge_update_hz)
CONFIG.control.edge_handoff_min_s = _env_float("ORCH_EDGE_HANDOFF_MIN_S", CONFIG.control.edge_handoff_min_s)
CONFIG.control.edge_handoff_max_s = _env_float("ORCH_EDGE_HANDOFF_MAX_S", CONFIG.control.edge_handoff_max_s)
CONFIG.control.edge_handoff_samples = _env_int("ORCH_EDGE_HANDOFF_SAMPLES", CONFIG.control.edge_handoff_samples)
CONFIG.control.edge_slide_segment_s = _env_float("ORCH_EDGE_SLIDE_SEGMENT_S", CONFIG.control.edge_slide_segment_s)
CONFIG.control.edge_slide_dist_tolerance_m = _env_float("ORCH_EDGE_SLIDE_DIST_TOL_M", CONFIG.control.edge_slide_dist_tolerance_m)
CONFIG.control.edge_slide_fallback_state = _env_str("ORCH_EDGE_SLIDE_FALLBACK_STATE", CONFIG.control.edge_slide_fallback_state).upper()
CONFIG.control.edge_slide_pause_hold_s = _env_float("ORCH_EDGE_SLIDE_PAUSE_HOLD_S", CONFIG.control.edge_slide_pause_hold_s)
CONFIG.control.edge_slide_recover_timeout_s = _env_float("ORCH_EDGE_SLIDE_RECOVER_TIMEOUT_S", CONFIG.control.edge_slide_recover_timeout_s)
CONFIG.control.edge_slide_direct_fallback_to_controlled_approach = _env_bool("ORCH_EDGE_SLIDE_DIRECT_FALLBACK_TO_CONTROLLED_APPROACH", CONFIG.control.edge_slide_direct_fallback_to_controlled_approach)

CONFIG.control.edge_relocate_enabled = _env_bool("ORCH_EDGE_RELOCATE_ENABLED", CONFIG.control.edge_relocate_enabled)
CONFIG.control.max_edge_transitions_per_task = _env_int("ORCH_MAX_EDGE_TRANSITIONS", CONFIG.control.max_edge_transitions_per_task)
CONFIG.control.leave_edge_backoff_s = _env_float("ORCH_LEAVE_EDGE_BACKOFF_S", CONFIG.control.leave_edge_backoff_s)
CONFIG.control.relocate_turn_s = _env_float("ORCH_RELOCATE_TURN_S", CONFIG.control.relocate_turn_s)
CONFIG.control.reacquire_timeout_s = _env_float("ORCH_REACQUIRE_TIMEOUT_S", CONFIG.control.reacquire_timeout_s)
CONFIG.control.next_table_dwell_s = _env_float("ORCH_NEXT_TABLE_DWELL_S", CONFIG.control.next_table_dwell_s)

CONFIG.control.tag_lost_frames_to_search = _env_int("ORCH_TAG_LOST_FRAMES", CONFIG.control.tag_lost_frames_to_search)
CONFIG.control.return_lost_hold_s = _env_float("ORCH_RETURN_LOST_HOLD_S", CONFIG.control.return_lost_hold_s)
CONFIG.control.return_min_dwell_s = _env_float("ORCH_RETURN_MIN_DWELL_S", CONFIG.control.return_min_dwell_s)
CONFIG.control.return_done_distance_m = _env_float("ORCH_RETURN_DONE_DISTANCE_M", CONFIG.control.return_done_distance_m)
CONFIG.control.tag_arrived_frames_to_stop = _env_int("ORCH_RETURN_DONE_FRAMES", CONFIG.control.tag_arrived_frames_to_stop)

CONFIG.control.avoid_clear_frames_to_resume = _env_int("ORCH_AVOID_CLEAR_FRAMES", CONFIG.control.avoid_clear_frames_to_resume)
CONFIG.control.avoid_timeout_s = _env_float("ORCH_AVOID_TIMEOUT_S", CONFIG.control.avoid_timeout_s)
CONFIG.control.avoid_retry_limit = _env_int("ORCH_AVOID_RETRY_LIMIT", CONFIG.control.avoid_retry_limit)
CONFIG.control.done_hold_s = _env_float("ORCH_DONE_HOLD_S", CONFIG.control.done_hold_s)
CONFIG.control.error_recovery_hold_s = _env_float("ORCH_ERROR_RECOVERY_HOLD_S", CONFIG.control.error_recovery_hold_s)
CONFIG.control.car_timeout_to_stop = _env_bool("ORCH_CAR_TIMEOUT_TO_STOP", CONFIG.control.car_timeout_to_stop)
CONFIG.control.car_fault_to_fail = _env_bool("ORCH_CAR_FAULT_TO_FAIL", CONFIG.control.car_fault_to_fail)
CONFIG.control.car_estop_to_stop = _env_bool("ORCH_CAR_ESTOP_TO_STOP", CONFIG.control.car_estop_to_stop)
CONFIG.control.post_stop_ignore_s = _env_float("ORCH_POST_STOP_IGNORE_S", CONFIG.control.post_stop_ignore_s)
CONFIG.control.vision_req_fail_to_stop = _env_bool("ORCH_VISION_REQ_FAIL_TO_STOP", CONFIG.control.vision_req_fail_to_stop)
CONFIG.control.vision_req_fail_threshold = _env_int("ORCH_VISION_REQ_FAIL_THRESHOLD", CONFIG.control.vision_req_fail_threshold)
CONFIG.control.enable_pick_pipeline = _env_bool("ORCH_ENABLE_PICK_PIPELINE", CONFIG.control.enable_pick_pipeline)

CONFIG.car.search_table_wz_norm = _env_float("ORCH_SEARCH_TABLE_WZ", CONFIG.car.search_table_wz_norm)
CONFIG.car.fallback_align_turn_norm_min = _env_float("ORCH_FALLBACK_TURN_MIN", CONFIG.car.fallback_align_turn_norm_min)
CONFIG.car.fallback_align_turn_norm_max = _env_float("ORCH_FALLBACK_TURN_MAX", CONFIG.car.fallback_align_turn_norm_max)
CONFIG.car.fallback_forward_vx_norm_min = _env_float("ORCH_FALLBACK_VX_MIN", CONFIG.car.fallback_forward_vx_norm_min)
CONFIG.car.fallback_forward_vx_norm_max = _env_float("ORCH_FALLBACK_VX_MAX", CONFIG.car.fallback_forward_vx_norm_max)
CONFIG.car.fallback_dead_zone_x = _env_float("ORCH_FALLBACK_DEAD_ZONE_X", CONFIG.car.fallback_dead_zone_x)
CONFIG.car.fallback_spin_only_x_th = _env_float("ORCH_FALLBACK_SPIN_ONLY_X_TH", CONFIG.car.fallback_spin_only_x_th)
CONFIG.car.fallback_forward_align_exp = _env_float("ORCH_FALLBACK_FORWARD_ALIGN_EXP", CONFIG.car.fallback_forward_align_exp)
CONFIG.car.return_turn_norm_min = _env_float("ORCH_RETURN_TURN_NORM_MIN", CONFIG.car.return_turn_norm_min)
CONFIG.car.return_turn_norm_max = _env_float("ORCH_RETURN_TURN_NORM_MAX", CONFIG.car.return_turn_norm_max)
CONFIG.car.return_vx_norm_min = _env_float("ORCH_RETURN_VX_NORM_MIN", CONFIG.car.return_vx_norm_min)
CONFIG.car.return_vx_norm_max = _env_float("ORCH_RETURN_VX_NORM_MAX", CONFIG.car.return_vx_norm_max)
CONFIG.car.edge_slide_vy_norm = _env_float("ORCH_EDGE_SLIDE_VY", CONFIG.car.edge_slide_vy_norm)
CONFIG.car.edge_slide_weak_vy_norm = _env_float("ORCH_EDGE_SLIDE_WEAK_VY", CONFIG.car.edge_slide_weak_vy_norm)
CONFIG.car.edge_slide_dist_kp_norm_per_m = _env_float("ORCH_EDGE_SLIDE_DIST_KP", CONFIG.car.edge_slide_dist_kp_norm_per_m)
CONFIG.car.edge_slide_yaw_kp_norm_per_rad = _env_float("ORCH_EDGE_SLIDE_YAW_KP", CONFIG.car.edge_slide_yaw_kp_norm_per_rad)
CONFIG.car.edge_slide_max_vx_norm = _env_float("ORCH_EDGE_SLIDE_MAX_VX", CONFIG.car.edge_slide_max_vx_norm)
CONFIG.car.edge_slide_max_wz_norm = _env_float("ORCH_EDGE_SLIDE_MAX_WZ", CONFIG.car.edge_slide_max_wz_norm)
CONFIG.car.leave_edge_vx_norm = _env_float("ORCH_LEAVE_EDGE_VX", CONFIG.car.leave_edge_vx_norm)
CONFIG.car.relocate_turn_wz_norm = _env_float("ORCH_RELOCATE_WZ", CONFIG.car.relocate_turn_wz_norm)
CONFIG.car.avoid_turn_norm = _env_float("ORCH_AVOID_TURN_WZ", CONFIG.car.avoid_turn_norm)
CONFIG.car.avoid_reverse_vx_norm = _env_float("ORCH_AVOID_REVERSE_VX", CONFIG.car.avoid_reverse_vx_norm)
CONFIG.car.cmd_hold_ms = _env_int("ORCH_CMD_HOLD_MS", CONFIG.car.cmd_hold_ms)
CONFIG.car.send_period_ms = _env_int("ORCH_CAR_SEND_PERIOD_MS", CONFIG.car.send_period_ms)
if CONFIG.car.send_period_ms > 0 and "ORCH_TICK_HZ" not in os.environ:
    CONFIG.runtime.tick_hz = 1000.0 / float(CONFIG.car.send_period_ms)
CONFIG.car.max_vx_norm = _env_float("ORCH_CAR_MAX_VX", CONFIG.car.max_vx_norm)
CONFIG.car.max_vy_norm = _env_float("ORCH_CAR_MAX_VY", CONFIG.car.max_vy_norm)
CONFIG.car.max_wz_norm = _env_float("ORCH_CAR_MAX_WZ", CONFIG.car.max_wz_norm)
CONFIG.car.stop_on_state_enter = _env_bool("ORCH_STOP_ON_STATE_ENTER", CONFIG.car.stop_on_state_enter)
CONFIG.car.mode_line_on_change = _env_bool("ORCH_MODE_LINE_ON_CHANGE", CONFIG.car.mode_line_on_change)
CONFIG.car.mode_line_every_cmd = _env_bool("ORCH_MODE_LINE_EVERY_CMD", CONFIG.car.mode_line_every_cmd)
CONFIG.car.serial_float_digits = _env_int("ORCH_SERIAL_FLOAT_DIGITS", CONFIG.car.serial_float_digits)

CONFIG.docking.min_confidence = _env_float("ORCH_DOCKING_MIN_CONFIDENCE", CONFIG.docking.min_confidence)
CONFIG.docking.obs_timeout_s = _env_float("ORCH_DOCKING_OBS_TIMEOUT_S", CONFIG.docking.obs_timeout_s)
CONFIG.docking.dt_min_s = _env_float("ORCH_DOCKING_DT_MIN_S", CONFIG.docking.dt_min_s)
CONFIG.docking.reset_on_mode_change = _env_bool("ORCH_DOCKING_RESET_ON_MODE_CHANGE", CONFIG.docking.reset_on_mode_change)
CONFIG.docking.coarse_align_enter_rad = _env_float("ORCH_DOCKING_COARSE_ENTER_RAD", CONFIG.docking.coarse_align_enter_rad)
CONFIG.docking.coarse_align_exit_rad = _env_float("ORCH_DOCKING_COARSE_EXIT_RAD", CONFIG.docking.coarse_align_exit_rad)
CONFIG.docking.spin_only_yaw_rad = _env_float("ORCH_DOCKING_SPIN_ONLY_YAW_RAD", CONFIG.docking.spin_only_yaw_rad)
CONFIG.docking.precise_yaw_tol_rad = _env_float("ORCH_DOCKING_PRECISE_YAW_RAD", CONFIG.docking.precise_yaw_tol_rad)
CONFIG.docking.precise_dist_tol_m = _env_float("ORCH_DOCKING_PRECISE_DIST_M", CONFIG.docking.precise_dist_tol_m)
CONFIG.docking.precise_lateral_tol_m = _env_float("ORCH_DOCKING_PRECISE_LAT_M", CONFIG.docking.precise_lateral_tol_m)
CONFIG.docking.precise_stable_s = _env_float("ORCH_DOCKING_PRECISE_STABLE_S", CONFIG.docking.precise_stable_s)
CONFIG.docking.coarse_max_wz_norm = _env_float("ORCH_DOCKING_COARSE_MAX_WZ", CONFIG.docking.coarse_max_wz_norm)
CONFIG.docking.approach_max_vx_norm = _env_float("ORCH_DOCKING_APPROACH_MAX_VX", CONFIG.docking.approach_max_vx_norm)
CONFIG.docking.approach_max_vy_norm = _env_float("ORCH_DOCKING_APPROACH_MAX_VY", CONFIG.docking.approach_max_vy_norm)
CONFIG.docking.approach_max_wz_norm = _env_float("ORCH_DOCKING_APPROACH_MAX_WZ", CONFIG.docking.approach_max_wz_norm)
CONFIG.docking.final_max_vx_norm = _env_float("ORCH_DOCKING_FINAL_MAX_VX", CONFIG.docking.final_max_vx_norm)
CONFIG.docking.final_max_vy_norm = _env_float("ORCH_DOCKING_FINAL_MAX_VY", CONFIG.docking.final_max_vy_norm)
CONFIG.docking.final_max_wz_norm = _env_float("ORCH_DOCKING_FINAL_MAX_WZ", CONFIG.docking.final_max_wz_norm)
CONFIG.docking.vx_slew_per_s = _env_float("ORCH_DOCKING_VX_SLEW", CONFIG.docking.vx_slew_per_s)
CONFIG.docking.vy_slew_per_s = _env_float("ORCH_DOCKING_VY_SLEW", CONFIG.docking.vy_slew_per_s)
CONFIG.docking.wz_slew_per_s = _env_float("ORCH_DOCKING_WZ_SLEW", CONFIG.docking.wz_slew_per_s)
CONFIG.docking.enable_lateral_control = _env_bool("ORCH_DOCKING_ENABLE_LATERAL", CONFIG.docking.enable_lateral_control)

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
CONFIG.vision_req_out.async_enabled = _env_bool("ORCH_VISION_REQ_ASYNC", CONFIG.vision_req_out.async_enabled)
CONFIG.vision_req_out.async_queue_size = _env_int("ORCH_VISION_REQ_ASYNC_QUEUE_SIZE", CONFIG.vision_req_out.async_queue_size)
CONFIG.vision_req_out.async_drop_oldest = _env_bool("ORCH_VISION_REQ_ASYNC_DROP_OLDEST", CONFIG.vision_req_out.async_drop_oldest)
CONFIG.vision_req_out.send_mode = _env_str("ORCH_VISION_REQ_SEND_MODE", "oneshot")

CONFIG.tts_event_out.transport = _env_str("ORCH_TTS_EVENT_OUT_TRANSPORT", CONFIG.tts_event_out.transport)
CONFIG.tts_event_out.host = _env_str("ORCH_TTS_EVENT_OUT_HOST", CONFIG.tts_event_out.host)
CONFIG.tts_event_out.port = _env_int("ORCH_TTS_EVENT_OUT_PORT", CONFIG.tts_event_out.port)
CONFIG.tts_event_out.uds_path = _env_str("ORCH_TTS_EVENT_OUT_UDS", CONFIG.tts_event_out.uds_path)
CONFIG.tts_event_out.async_enabled = _env_bool("ORCH_TTS_EVENT_ASYNC", CONFIG.tts_event_out.async_enabled)
CONFIG.tts_event_out.async_queue_size = _env_int("ORCH_TTS_EVENT_ASYNC_QUEUE_SIZE", CONFIG.tts_event_out.async_queue_size)
CONFIG.tts_event_out.async_drop_oldest = _env_bool("ORCH_TTS_EVENT_ASYNC_DROP_OLDEST", CONFIG.tts_event_out.async_drop_oldest)
