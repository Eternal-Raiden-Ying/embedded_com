#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import platform
from typing import Any, Dict
from pathlib import Path

from .schema import VisionServiceConfig, SingleModelConfig
from .data import coco80, grasping_coco20

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
        if lowered in {"true", "false"}:
            return lowered == "true"
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
            child: Dict[str, Any] = {}
            current[key.strip()] = child
            stack.append((indent, child))
        else:
            current[key.strip()] = _scalar(value)
    return root


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


def _loaded(path: str) -> None:
    path = str(path or "").strip()
    if path and Path(path).is_file() and path not in CONFIG.runtime.loaded_config_files:
        CONFIG.runtime.loaded_config_files.append(path)


def _assign_attrs(obj: Any, values: Dict[str, Any], keys) -> None:
    for key in tuple(keys or ()):
        if key in values and values.get(key) is not None:
            setattr(obj, key, values.get(key))


def _apply_vision_params(data: Dict[str, Any]) -> None:
    if not data:
        return
    runtime = dict(data.get("runtime") or {})
    camera = dict(data.get("camera") or {})
    model = dict(data.get("model") or {})
    debug = dict(data.get("debug") or {})
    table_edge = dict(data.get("table_edge") or {})
    preview = dict(data.get("preview") or {})
    ipc = dict(data.get("ipc") or {})
    mode_profiles = dict(data.get("mode_profiles") or {})

    _assign_attrs(
        CONFIG.runtime,
        runtime,
        (
            "loop_hz",
            "send_hz",
            "track_local_send_hz",
            "stale_req_s",
            "hot_standby_s",
            "keep_preview_after_stop",
            "keep_model_hot_in_standby",
            "enable_infer_during_hot_standby",
            "log_mode",
            "log_enabled",
            "debug",
            "heartbeat_enabled",
            "heartbeat_interval_s",
            "heartbeat_console",
            "console_mode",
            "operator_summary_interval_s",
            "ipc_console",
        ),
    )
    if "max_fps" in camera and camera.get("max_fps") is not None:
        CONFIG.camera.max_fps = int(camera.get("max_fps"))
    camera_keys = (
        "source",
        "in_w",
        "in_h",
        "out_w",
        "out_h",
        "width",
        "height",
        "in_format",
        "format",
        "fps",
        "crop_x",
        "crop_y",
        "crop_w",
        "crop_h",
        "enable",
        "auto_exposure",
        "exposure",
        "brightness",
    )
    for name in ("rgb", "depth", "grey"):
        section = dict(camera.get(name) or {})
        stream = CONFIG.camera.streams.get(name)
        if stream is not None:
            _assign_attrs(stream, section, camera_keys)

    if model.get("active_model") is not None:
        CONFIG.model.active_model = str(model.get("active_model")).strip() or CONFIG.model.active_model
    profiles = dict(model.get("profiles") or {})
    for name, values in profiles.items():
        profile = CONFIG.model.profiles.get(str(name))
        if profile is None or not isinstance(values, dict):
            continue
        _assign_attrs(
            profile,
            values,
            ("width", "height", "conf_thres", "iou_thres", "class_num", "predictor_type", "model_backend", "anchors", "strides"),
        )
        target_model = values.get("target_model")
        if target_model:
            profile.target_model = str(target_model)

    _assign_attrs(
        CONFIG.debug,
        debug,
        (
            "preview",
            "draw_boxes",
            "draw_masks",
            "edge_debug_enabled",
            "edge_debug_period_s",
            "table_det_enabled",
            "table_det_min_conf",
            "table_det_center_tol",
        ),
    )
    _assign_attrs(
        CONFIG.table_edge,
        table_edge,
        (
            "roi_preset",
            "static_roi_enabled",
            "detector_mode",
            "fast_plane_stride",
            "camera_pitch_deg",
            "camera_height_m",
            "camera_roll_deg",
            "camera_yaw_deg",
            "table_height_m",
            "front_face_z_min_m",
            "front_face_z_max_m",
            "min_vertical_z_span_m",
            "min_vertical_support_points",
            "x_bin_width_m",
            "y_cluster_bin_m",
            "min_front_face_columns",
            "min_front_face_x_span_m",
            "max_yaw_abs_rad",
            "target_hz",
            "update_hz",
            "preview_hz",
            "track_local_update_hz",
            "track_local_light_edge",
            "track_local_edge_stride",
            "require_yolo_table_confirm",
            "enable_yolo_in_plane_only",
            "save_debug_frames",
            "profile_log_interval_s",
        ),
    )
    if float(getattr(CONFIG.table_edge, "target_hz", 0.0) or 0.0) > 0:
        CONFIG.table_edge.update_hz = float(CONFIG.table_edge.target_hz)
    if getattr(CONFIG.table_edge, "roi_preset", ""):
        CONFIG.table_edge.roi_preset = str(CONFIG.table_edge.roi_preset).strip().lower()
    CONFIG.table_edge.detector_mode = str(getattr(CONFIG.table_edge, "detector_mode", "full") or "full").strip().lower()
    CONFIG.table_edge.fast_plane_stride = max(1, int(float(getattr(CONFIG.table_edge, "fast_plane_stride", 4) or 4)))

    layouts = preview.get("mode_layouts")
    if isinstance(layouts, dict):
        CONFIG.preview.mode_layouts.update({str(k).upper(): str(v).strip() for k, v in layouts.items() if v is not None})
    _assign_attrs(
        CONFIG.preview,
        preview,
        (
            "debug_four_panel_in_track_local",
            "show_edge_overlay_in_track_local",
            "show_age_ms",
            "clear_overlay_on_mode_switch",
        ),
    )

    req_in = dict(ipc.get("req_in") or {})
    obs_out = dict(ipc.get("obs_out") or {})
    _assign_attrs(CONFIG.req_in, req_in, ("transport", "host", "port", "uds_path"))
    _assign_attrs(CONFIG.obs_out, obs_out, ("transport", "host", "port", "uds_path"))
    CONFIG.mode_profiles = mode_profiles


CONFIG = VisionServiceConfig()

CONFIG.runtime.project_root = os.getenv("VISION_PROJECT_ROOT", _DEFAULT_PROJECT_ROOT)
CONFIG.runtime.log_dir = os.getenv("VISION_LOG_DIR", f"{CONFIG.runtime.project_root}/logs")
CONFIG.runtime.log_file = os.getenv("VISION_LOG_FILE", f"{CONFIG.runtime.log_dir}/vision.log")
CONFIG.runtime.runs_dir = os.getenv("VISION_RUNS_DIR", f"{CONFIG.runtime.project_root}/runs")
CONFIG.runtime.pid_dir = os.getenv("VISION_PID_DIR", f"{CONFIG.runtime.project_root}/pids")
CONFIG.runtime.pid_file = os.getenv("VISION_PID_FILE", f"{CONFIG.runtime.pid_dir}/vision.pid")
CONFIG.runtime.vision_params_file = os.getenv(
    "VISION_PARAMS_FILE",
    str(Path(CONFIG.runtime.project_root) / "configs" / "vision_params.yaml"),
)
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
CONFIG.debug.preview = str(_preview_default).strip().lower() in {"1", "true", "yes"}
CONFIG.debug.draw_boxes = True
CONFIG.debug.draw_masks = False


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


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return str(raw).strip() if raw is not None else str(default)


_apply_vision_params(_load_config_dict(CONFIG.runtime.vision_params_file))
_loaded(CONFIG.runtime.vision_params_file)

# Environment variables stay available for one-off tests and override YAML.
CONFIG.runtime.track_local_send_hz = _env_float("VISION_TRACK_LOCAL_SEND_HZ", CONFIG.runtime.track_local_send_hz)
CONFIG.runtime.log_mode = _env_str("VISION_LOG_MODE", CONFIG.runtime.log_mode)
CONFIG.runtime.log_enabled = _env_bool("VISION_LOG_ENABLED", CONFIG.runtime.log_enabled)
CONFIG.runtime.debug = _env_bool("VISION_DEBUG", CONFIG.runtime.debug)
CONFIG.runtime.heartbeat_enabled = _env_bool("VISION_HEARTBEAT_ENABLED", CONFIG.runtime.heartbeat_enabled)
CONFIG.runtime.heartbeat_interval_s = _env_float("VISION_HEARTBEAT_INTERVAL_S", CONFIG.runtime.heartbeat_interval_s)
CONFIG.runtime.console_mode = _env_str("VISION_CONSOLE_MODE", CONFIG.runtime.console_mode).lower() or CONFIG.runtime.console_mode
CONFIG.runtime.operator_summary_interval_s = _env_float(
    "VISION_OPERATOR_SUMMARY_INTERVAL_S",
    CONFIG.runtime.operator_summary_interval_s,
)
CONFIG.runtime.ipc_console = _env_bool("VISION_IPC_CONSOLE", CONFIG.runtime.ipc_console)
CONFIG.runtime.heartbeat_console = _env_bool("VISION_HEARTBEAT_CONSOLE", CONFIG.runtime.heartbeat_console)

CONFIG.model.active_model = _env_str("VISION_ACTIVE_MODEL", CONFIG.model.active_model)
CONFIG.model.profiles["yolov7_detect"].target_model = _env_str(
    "VISION_DETECT_MODEL_PATH",
    CONFIG.model.profiles["yolov7_detect"].target_model,
)
CONFIG.model.profiles["yolov7_detect"].model_backend = _env_str(
    "VISION_DETECT_MODEL_BACKEND",
    CONFIG.model.profiles["yolov7_detect"].model_backend,
)
CONFIG.model.profiles["yolo26s_seg"].target_model = _env_str(
    "VISION_SEG_MODEL_PATH",
    CONFIG.model.profiles["yolo26s_seg"].target_model,
)
CONFIG.model.profiles["yolo26s_seg_qnn216"].target_model = _env_str(
    "VISION_SEG_MODEL_QNN216_PATH",
    CONFIG.model.profiles["yolo26s_seg_qnn216"].target_model,
)

CONFIG.debug.preview = _env_bool("VISION_PREVIEW", CONFIG.debug.preview)
CONFIG.debug.draw_boxes = _env_bool("VISION_DRAW_BOXES", CONFIG.debug.draw_boxes)
CONFIG.debug.draw_masks = _env_bool("VISION_DRAW_MASKS", CONFIG.debug.draw_masks)

# table-edge / table-detection debug config.
CONFIG.debug.edge_debug_enabled = _env_bool(
    "VISTA_EDGE_DBG_ENABLED",
    _env_bool("VISTA_EDGE_DBG", CONFIG.debug.edge_debug_enabled),
)
CONFIG.debug.edge_debug_period_s = _env_float("VISTA_EDGE_DBG_PERIOD_S", CONFIG.debug.edge_debug_period_s)
CONFIG.debug.table_det_enabled = _env_bool("ORCH_TABLE_DET_ENABLED", CONFIG.debug.table_det_enabled)
CONFIG.debug.table_det_min_conf = _env_float("ORCH_TABLE_DET_MIN_CONF", CONFIG.debug.table_det_min_conf)
CONFIG.debug.table_det_center_tol = _env_float("ORCH_TABLE_DET_CENTER_TOL", CONFIG.debug.table_det_center_tol)

CONFIG.table_edge.roi_preset = os.getenv("VISTA_TABLE_EDGE_ROI_PRESET", CONFIG.table_edge.roi_preset).strip().lower()
CONFIG.table_edge.detector_mode = os.getenv("VISTA_TABLE_EDGE_DETECTOR_MODE", CONFIG.table_edge.detector_mode).strip().lower()
CONFIG.table_edge.fast_plane_stride = max(
    1,
    _env_int("VISTA_TABLE_EDGE_FAST_PLANE_STRIDE", CONFIG.table_edge.fast_plane_stride),
)
CONFIG.table_edge.static_roi_enabled = _env_bool(
    "VISTA_TABLE_EDGE_STATIC_ROI",
    _env_bool("VISTA_FORCE_STATIC_EDGE_ROI", CONFIG.table_edge.static_roi_enabled),
)
_target_hz_env = os.getenv("VISTA_TABLE_EDGE_TARGET_HZ")
if _target_hz_env is not None:
    CONFIG.table_edge.target_hz = _env_float("VISTA_TABLE_EDGE_TARGET_HZ", CONFIG.table_edge.target_hz)
    CONFIG.table_edge.update_hz = float(CONFIG.table_edge.target_hz)
CONFIG.table_edge.update_hz = _env_float("VISTA_TABLE_EDGE_HZ", CONFIG.table_edge.update_hz)
CONFIG.table_edge.preview_hz = _env_float("VISTA_TABLE_EDGE_PREVIEW_HZ", CONFIG.table_edge.preview_hz)
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
CONFIG.table_edge.require_yolo_table_confirm = _env_bool(
    "VISTA_TABLE_EDGE_REQUIRE_YOLO",
    CONFIG.table_edge.require_yolo_table_confirm,
)
CONFIG.table_edge.enable_yolo_in_plane_only = _env_bool(
    "VISTA_TABLE_EDGE_ENABLE_YOLO_IN_PLANE_ONLY",
    CONFIG.table_edge.enable_yolo_in_plane_only,
)
CONFIG.table_edge.yolo_table_min_conf = _env_float("VISTA_TABLE_EDGE_YOLO_MIN_CONF", CONFIG.table_edge.yolo_table_min_conf)
CONFIG.table_edge.save_debug_frames = _env_bool("VISTA_TABLE_EDGE_SAVE_DEBUG_FRAMES", CONFIG.table_edge.save_debug_frames)
CONFIG.table_edge.profile_log_interval_s = _env_float(
    "VISTA_TABLE_EDGE_PROFILE_LOG_INTERVAL_S",
    CONFIG.table_edge.profile_log_interval_s,
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
