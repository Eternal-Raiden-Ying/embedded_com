#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

try:
    from .schema import OnlineEdgeConfig
except ImportError:
    from schema import OnlineEdgeConfig


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


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


CONFIG = OnlineEdgeConfig()

HERE = Path(__file__).resolve().parent
CONFIG.runtime.project_root = "."
CONFIG.runtime.log_dir = _env_str("EDGE_LOG_DIR", "logs")
CONFIG.runtime.log_file = _env_str("EDGE_LOG_FILE", str(Path(CONFIG.runtime.log_dir) / "online_edge.log"))
CONFIG.runtime.runs_dir = _env_str("EDGE_RUNS_DIR", "runs")
CONFIG.runtime.pid_dir = _env_str("EDGE_PID_DIR", "pids")
CONFIG.runtime.pid_file = _env_str("EDGE_PID_FILE", str(Path(CONFIG.runtime.pid_dir) / "online_edge.pid"))
CONFIG.runtime.stack_run_id = _env_str("STACK_RUN_ID", CONFIG.runtime.stack_run_id)
CONFIG.runtime.loop_hz = _env_float("EDGE_LOOP_HZ", CONFIG.runtime.loop_hz)
CONFIG.runtime.preview = _env_bool("EDGE_PREVIEW", CONFIG.runtime.preview)
CONFIG.runtime.save_snapshot_period_s = _env_float("EDGE_SNAPSHOT_PERIOD_S", CONFIG.runtime.save_snapshot_period_s)
CONFIG.runtime.snapshot_dir = _env_str("EDGE_SNAPSHOT_DIR", "snapshots")
CONFIG.runtime.log_mode = _env_str("EDGE_LOG_MODE", CONFIG.runtime.log_mode)
CONFIG.runtime.log_enabled = _env_bool("EDGE_LOG_ENABLED", CONFIG.runtime.log_enabled)

CONFIG.output.transport = _env_str("EDGE_OUT_TRANSPORT", CONFIG.output.transport)
CONFIG.output.host = _env_str("EDGE_OUT_HOST", CONFIG.output.host)
CONFIG.output.port = _env_int("EDGE_OUT_PORT", CONFIG.output.port)
CONFIG.output.uds_path = _env_str("EDGE_OUT_UDS", CONFIG.output.uds_path)
CONFIG.output.send_interval_s = _env_float("EDGE_OUT_PERIOD_S", CONFIG.output.send_interval_s)

CONFIG.camera.bag_path = _env_str("EDGE_BAG_PATH", CONFIG.camera.bag_path)
CONFIG.camera.align_to_color = _env_bool("EDGE_ALIGN_TO_COLOR", CONFIG.camera.align_to_color)
CONFIG.camera.depth_enabled = _env_bool("EDGE_DEPTH_ENABLED", CONFIG.camera.depth_enabled)
CONFIG.camera.depth_width = _env_int("EDGE_DEPTH_WIDTH", CONFIG.camera.depth_width)
CONFIG.camera.depth_height = _env_int("EDGE_DEPTH_HEIGHT", CONFIG.camera.depth_height)
CONFIG.camera.depth_fps = _env_int("EDGE_DEPTH_FPS", CONFIG.camera.depth_fps)
CONFIG.camera.color_enabled = _env_bool("EDGE_COLOR_ENABLED", CONFIG.camera.color_enabled)
CONFIG.camera.color_width = _env_int("EDGE_COLOR_WIDTH", CONFIG.camera.color_width)
CONFIG.camera.color_height = _env_int("EDGE_COLOR_HEIGHT", CONFIG.camera.color_height)
CONFIG.camera.color_fps = _env_int("EDGE_COLOR_FPS", CONFIG.camera.color_fps)

CONFIG.detector.calib_json = _env_str("EDGE_CALIB_JSON", "../calib.json")
CONFIG.detector.target_dist_m_override = _env_float("EDGE_TARGET_DIST_M", CONFIG.detector.target_dist_m_override)
CONFIG.detector.roi_y0 = _env_int("EDGE_ROI_Y0", CONFIG.detector.roi_y0)
CONFIG.detector.roi_y1 = _env_int("EDGE_ROI_Y1", CONFIG.detector.roi_y1)
CONFIG.detector.roi_x0 = _env_int("EDGE_ROI_X0", CONFIG.detector.roi_x0)
CONFIG.detector.roi_x1 = _env_int("EDGE_ROI_X1", CONFIG.detector.roi_x1)
CONFIG.detector.z_min = _env_float("EDGE_Z_MIN", CONFIG.detector.z_min)
CONFIG.detector.z_max = _env_float("EDGE_Z_MAX", CONFIG.detector.z_max)
CONFIG.detector.table_y_min = _env_float("EDGE_TABLE_Y_MIN", CONFIG.detector.table_y_min)
CONFIG.detector.table_y_max = _env_float("EDGE_TABLE_Y_MAX", CONFIG.detector.table_y_max)
CONFIG.detector.min_all_points = _env_int("EDGE_MIN_ALL_POINTS", CONFIG.detector.min_all_points)
CONFIG.detector.min_table_points = _env_int("EDGE_MIN_TABLE_POINTS", CONFIG.detector.min_table_points)
CONFIG.detector.ransac_iters = _env_int("EDGE_RANSAC_ITERS", CONFIG.detector.ransac_iters)
CONFIG.detector.residual_threshold_m = _env_float("EDGE_RANSAC_THRESHOLD_M", CONFIG.detector.residual_threshold_m)
CONFIG.detector.random_seed = _env_int("EDGE_RANDOM_SEED", CONFIG.detector.random_seed)
CONFIG.detector.depth_median_ksize = _env_int("EDGE_MEDIAN_KSIZE", CONFIG.detector.depth_median_ksize)
