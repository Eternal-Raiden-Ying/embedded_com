#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import platform
from typing import Dict
from pathlib import Path

from .schema import VisionServiceConfig, SingleModelConfig
from .data import coco80, grasping_coco20

CONFIG = VisionServiceConfig()

_HERE = Path(__file__).resolve()
_DEFAULT_PROJECT_ROOT = str(_HERE.parents[2])
_DEFAULT_MODEL_ROOT = str(Path(_DEFAULT_PROJECT_ROOT) / "vision_module" / "model")
_DEFAULT_DETECT_MODEL = (
    Path(_DEFAULT_MODEL_ROOT)
    / "qnn216"
    / "model_farm_yolov7_qcs6490_qnn2.16_int8_aidlite"
    / "models"
    / "cutoff_yolov7_w8a8.qnn216.ctx.bin"
)
_DEFAULT_SEG_MODEL_QNN216 = (
    Path(_DEFAULT_MODEL_ROOT)
    / "yolo26s-seg-grasp"
    / "yolo26s-seg-grasp_split_w8a8.qnn216.ctx.bin"
)

CONFIG.runtime.project_root = os.getenv("VISION_PROJECT_ROOT", _DEFAULT_PROJECT_ROOT)
CONFIG.runtime.log_dir = os.getenv("VISION_LOG_DIR", f"{CONFIG.runtime.project_root}/logs")
CONFIG.runtime.log_file = os.getenv("VISION_LOG_FILE", f"{CONFIG.runtime.log_dir}/vision.log")
CONFIG.runtime.runs_dir = os.getenv("VISION_RUNS_DIR", f"{CONFIG.runtime.project_root}/runs")
CONFIG.runtime.pid_dir = os.getenv("VISION_PID_DIR", f"{CONFIG.runtime.project_root}/pids")
CONFIG.runtime.pid_file = os.getenv("VISION_PID_FILE", f"{CONFIG.runtime.pid_dir}/vision.pid")
CONFIG.runtime.stack_run_id = os.getenv("STACK_RUN_ID", "")
CONFIG.runtime.loop_hz = 8.0
CONFIG.runtime.send_hz = 5.0
CONFIG.runtime.track_local_send_hz = float(os.getenv("VISION_TRACK_LOCAL_SEND_HZ", "8.0") or 8.0)
CONFIG.runtime.stale_req_s = 3.0
CONFIG.runtime.hot_standby_s = 30.0
CONFIG.runtime.keep_preview_after_stop = True
CONFIG.runtime.keep_model_hot_in_standby = True
CONFIG.runtime.enable_infer_during_hot_standby = False
CONFIG.runtime.log_mode = os.getenv("VISION_LOG_MODE", "concise")
CONFIG.runtime.log_enabled = os.getenv("VISION_LOG_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
CONFIG.runtime.debug = os.getenv("VISION_DEBUG", "0").strip().lower() in {"1", "true", "yes"}
_placeholder_default = "1" if platform.system().lower().startswith("win") else "0"
# This flag is retained for tests and explicit mock scaffolding only.
# Camera/predictor runtime backend selection now belongs to package-level
# selectors via VISTA_BACKEND, not to this config field.
CONFIG.runtime.capability_placeholder = os.getenv("VISION_CAPABILITY_PLACEHOLDER", _placeholder_default).strip().lower() in {"1", "true", "yes"}
CONFIG.runtime.heartbeat_enabled = os.getenv("VISION_HEARTBEAT_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
CONFIG.runtime.heartbeat_interval_s = float(os.getenv("VISION_HEARTBEAT_INTERVAL_S", "2.0") or 2.0)
CONFIG.runtime.console_mode = os.getenv("VISION_CONSOLE_MODE", "operator").strip().lower() or "operator"
CONFIG.runtime.operator_summary_interval_s = float(os.getenv("VISION_OPERATOR_SUMMARY_INTERVAL_S", "1.0") or 1.0)
CONFIG.runtime.ipc_console = os.getenv("VISION_IPC_CONSOLE", "0").strip().lower() in {"1", "true", "yes"}
CONFIG.runtime.heartbeat_console = os.getenv("VISION_HEARTBEAT_CONSOLE", "0").strip().lower() in {"1", "true", "yes"}

# camera config
rgb = CONFIG.camera.streams["rgb"]
rgb.source = "6"
rgb.in_w = 1280
rgb.in_h = 720
rgb.out_w = 640
rgb.out_h = 640
rgb.format = "BGR"
rgb.crop_x = 280
rgb.crop_y = 0
rgb.crop_w = 720
rgb.crop_h = 720
rgb.in_format = "YUY2"
CONFIG.camera.max_fps = 30

depth = CONFIG.camera.streams["depth"]
depth.source = "2"
depth.width = 424
depth.height = 240
depth.fps = 15

grey = CONFIG.camera.streams["grey"]
grey.source = "4"
grey.in_w = 640
grey.in_h = 480
grey.out_w = 640
grey.out_h = 480
grey.in_format = "GRAY8"
grey.format = "BGR"
grey.fps = 30
grey.crop_x = 0
grey.crop_y = 0
grey.crop_w = 0
grey.crop_h = 0

# model config
CONFIG.model.active_model = os.getenv("VISION_ACTIVE_MODEL", "yolov7_detect")
CONFIG.model.profiles["yolov7_detect"] = SingleModelConfig(
    target_model=os.getenv(
        "VISION_DETECT_MODEL_PATH",
        str(_DEFAULT_DETECT_MODEL),
    ),
    width=640,
    height=640,
    conf_thres=0.25,
    iou_thres=0.45,
    class_num=80,
    classes=coco80,
    predictor_type="detect",
    model_backend=os.getenv("VISION_DETECT_MODEL_BACKEND", "qnn"),
    anchors=(
        (12, 16, 19, 36, 40, 28),
        (36, 75, 76, 55, 72, 146),
        (142, 110, 192, 243, 459, 401),
    ),
    strides=(8, 16, 32),
)
CONFIG.model.profiles["yolov8s_seg"] = SingleModelConfig(
    target_model=str(Path(_DEFAULT_MODEL_ROOT) / "yolov8s-seg" / "cutoff_yolov8s-seg_qcs6490_w8a8.qnn236.ctx.bin"),
    width=640,
    height=640,
    conf_thres=0.45,
    iou_thres=0.45,
    class_num=80,
    classes=coco80,
    predictor_type="segment",
    model_backend="qnn",
)
CONFIG.model.profiles["yolo26s_seg"] = SingleModelConfig(
    target_model=os.getenv(
        "VISION_SEG_MODEL_PATH",
        str(_DEFAULT_SEG_MODEL_QNN216),
    ),
    width=640,
    height=640,
    conf_thres=0.25,
    iou_thres=0.15,
    class_num=20,
    classes=grasping_coco20,
    predictor_type="segment",
    model_backend="qnn",
)
CONFIG.model.profiles["yolo26s_seg_qnn216"] = SingleModelConfig(
    target_model=os.getenv(
        "VISION_SEG_MODEL_QNN216_PATH",
        str(_DEFAULT_SEG_MODEL_QNN216),
    ),
    width=640,
    height=640,
    conf_thres=0.25,
    iou_thres=0.15,
    class_num=20,
    classes=grasping_coco20,
    predictor_type="segment",
    model_backend="qnn",
)

# debug config
_preview_default = "0" if platform.system().lower().startswith("win") else "1"
CONFIG.debug.preview = os.getenv("VISION_PREVIEW", _preview_default).strip().lower() in {"1", "true", "yes"}
CONFIG.debug.draw_boxes = os.getenv("VISION_DRAW_BOXES", "1").strip().lower() in {"1", "true", "yes"}
CONFIG.debug.draw_masks = os.getenv("VISION_DRAW_MASKS", "0").strip().lower() in {"1", "true", "yes"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(float(raw))
    except Exception:
        return int(default)


# table-edge / table-detection debug config.
# 现场持久调参优先改这里；环境变量只用于临时覆盖。
CONFIG.debug.edge_debug_enabled = _env_bool(
    "VISTA_EDGE_DBG_ENABLED",
    _env_bool("VISTA_EDGE_DBG", CONFIG.debug.edge_debug_enabled),
)
CONFIG.debug.edge_debug_period_s = _env_float("VISTA_EDGE_DBG_PERIOD_S", CONFIG.debug.edge_debug_period_s)
CONFIG.debug.table_det_enabled = _env_bool("ORCH_TABLE_DET_ENABLED", CONFIG.debug.table_det_enabled)
CONFIG.debug.table_det_min_conf = _env_float("ORCH_TABLE_DET_MIN_CONF", CONFIG.debug.table_det_min_conf)
CONFIG.debug.table_det_center_tol = _env_float("ORCH_TABLE_DET_CENTER_TOL", CONFIG.debug.table_det_center_tol)

CONFIG.table_edge.roi_preset = os.getenv("VISTA_TABLE_EDGE_ROI_PRESET", CONFIG.table_edge.roi_preset).strip().lower()
CONFIG.table_edge.static_roi_enabled = _env_bool(
    "VISTA_TABLE_EDGE_STATIC_ROI",
    _env_bool("VISTA_FORCE_STATIC_EDGE_ROI", CONFIG.table_edge.static_roi_enabled),
)
CONFIG.table_edge.update_hz = _env_float("VISTA_TABLE_EDGE_HZ", CONFIG.table_edge.update_hz)
CONFIG.table_edge.track_local_update_hz = _env_float(
    "VISTA_TRACK_LOCAL_EDGE_UPDATE_HZ",
    _env_float("VISTA_EDGE_FOLLOW_TRACK_LOCAL_EDGE_UPDATE_HZ", CONFIG.table_edge.track_local_update_hz),
)
CONFIG.table_edge.track_local_light_edge = _env_bool(
    "VISTA_TRACK_LOCAL_LIGHT_EDGE",
    CONFIG.table_edge.track_local_light_edge,
)
CONFIG.table_edge.track_local_edge_stride = max(
    1,
    _env_int("VISTA_TRACK_LOCAL_EDGE_STRIDE", CONFIG.table_edge.track_local_edge_stride),
)


def _preview_mode_layouts(defaults: Dict[str, str]) -> Dict[str, str]:
    layouts = {str(k).upper(): str(v).strip() for k, v in dict(defaults or {}).items()}
    raw = os.getenv("VISION_PREVIEW_MODE_LAYOUTS", "").strip()
    if raw:
        for item in raw.replace(";", ",").split(","):
            if ":" not in item:
                continue
            mode, layout = item.split(":", 1)
            mode = mode.strip().upper()
            layout = layout.strip()
            if mode and layout:
                layouts[mode] = layout
    for mode in list(layouts):
        env_key = f"VISION_PREVIEW_LAYOUT_{mode}"
        value = os.getenv(env_key)
        if value is not None and value.strip():
            layouts[mode] = value.strip()
    return layouts


CONFIG.preview.mode_layouts = _preview_mode_layouts(CONFIG.preview.mode_layouts)
CONFIG.preview.debug_four_panel_in_track_local = _env_bool(
    "VISION_PREVIEW_DEBUG_FOUR_PANEL_IN_TRACK_LOCAL",
    CONFIG.preview.debug_four_panel_in_track_local,
)
CONFIG.preview.show_edge_overlay_in_track_local = _env_bool(
    "VISION_PREVIEW_SHOW_EDGE_OVERLAY_IN_TRACK_LOCAL",
    CONFIG.preview.show_edge_overlay_in_track_local,
)
CONFIG.preview.show_age_ms = _env_bool("VISION_PREVIEW_SHOW_AGE_MS", CONFIG.preview.show_age_ms)
CONFIG.preview.clear_overlay_on_mode_switch = _env_bool(
    "VISION_PREVIEW_CLEAR_OVERLAY_ON_MODE_SWITCH",
    CONFIG.preview.clear_overlay_on_mode_switch,
)

# orchestrator -> vision
CONFIG.req_in.transport = os.getenv("VISION_REQ_TRANSPORT", "tcp").strip() or "tcp"
CONFIG.req_in.host = os.getenv("VISION_REQ_HOST", "127.0.0.1").strip() or "127.0.0.1"
CONFIG.req_in.port = int(os.getenv("VISION_REQ_PORT", "9003") or 9003)
CONFIG.req_in.uds_path = os.getenv("VISION_REQ_UDS_PATH", "/tmp/robot_stack/vision_req.sock").strip() or "/tmp/robot_stack/vision_req.sock"

# vision -> orchestrator
CONFIG.obs_out.transport = os.getenv("VISION_OBS_TRANSPORT", "tcp").strip() or "tcp"
CONFIG.obs_out.host = os.getenv("VISION_OBS_HOST", "127.0.0.1").strip() or "127.0.0.1"
CONFIG.obs_out.port = int(os.getenv("VISION_OBS_PORT", "9002") or 9002)
CONFIG.obs_out.uds_path = os.getenv("VISION_OBS_UDS_PATH", "/tmp/robot_stack/vision_obs.sock").strip() or "/tmp/robot_stack/vision_obs.sock"
