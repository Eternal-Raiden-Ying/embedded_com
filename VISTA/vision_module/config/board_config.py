#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import platform
from typing import Any, Dict
from pathlib import Path

from .schema import VisionServiceConfig, SingleModelConfig
from .data import coco80, finetune_yolo26s_bgr15, grasping_coco20

_HERE = Path(__file__).resolve()
_DEFAULT_PROJECT_ROOT = str(_HERE.parents[2])
_DEFAULT_STACK_ROOT = str(_HERE.parents[3])
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
    if "enable_yolo26" in model:
        CONFIG.model.enable_yolo26 = bool(model.get("enable_yolo26"))
    if "enable_yolo_table_search" in model:
        CONFIG.model.enable_yolo_table_search = bool(model.get("enable_yolo_table_search"))
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
            "table_bbox_enabled",
            "mock_table_bbox",
        ),
    )
    _assign_attrs(
        CONFIG.table_edge,
        table_edge,
        (
            "roi_preset",
            "profile_log_interval_s",
            "save_debug_frames",
            "target_hz",
            "preview_hz",
            "fast_debug_pixels",
            "fast_debug_pixels_online",
            "fast_debug_pixels_offline",
            "fast_debug_pixel_cap",
            "fast_candidate_point_cap",
            "fast_front_edge_col_step",
            "fast_front_edge_row_step",
        ),
    )
    if getattr(CONFIG.table_edge, "roi_preset", ""):
        CONFIG.table_edge.roi_preset = str(CONFIG.table_edge.roi_preset).strip().lower()

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
            "scale",
            "canvas_w",
            "canvas_h",
            "show_rgb",
            "show_depth",
            "show_edge",
            "destroy_all_on_close",
        ),
    )

    req_in = dict(ipc.get("req_in") or {})
    obs_out = dict(ipc.get("obs_out") or {})
    _assign_attrs(CONFIG.req_in, req_in, ("transport", "host", "port", "uds_path"))
    _assign_attrs(CONFIG.obs_out, obs_out, ("transport", "host", "port", "uds_path"))
    CONFIG.mode_profiles = mode_profiles


CONFIG = VisionServiceConfig()

CONFIG.runtime.project_root = os.getenv("VISION_PROJECT_ROOT", _DEFAULT_PROJECT_ROOT)
CONFIG.runtime.log_dir = f"{CONFIG.runtime.project_root}/logs"
CONFIG.runtime.log_file = f"{CONFIG.runtime.log_dir}/vision.log"
CONFIG.runtime.runs_dir = os.getenv("VISION_RUNS_DIR", str(Path(_DEFAULT_STACK_ROOT) / "logs" / "runs"))
CONFIG.runtime.pid_dir = f"{CONFIG.runtime.project_root}/pids"
CONFIG.runtime.pid_file = f"{CONFIG.runtime.pid_dir}/vision.pid"
CONFIG.runtime.vision_params_file = os.getenv(
    "VISION_PARAMS_FILE",
    str(Path(CONFIG.runtime.project_root) / "configs" / "vision_params.yaml"),
)
CONFIG.runtime.stack_run_id = os.getenv("STACK_RUN_ID", "")
CONFIG.runtime.loop_hz = 8.0
CONFIG.runtime.send_hz = 5.0
CONFIG.runtime.track_local_send_hz = 8.0
CONFIG.runtime.stale_req_s = 3.0
CONFIG.runtime.hot_standby_s = 30.0
CONFIG.runtime.keep_preview_after_stop = True
CONFIG.runtime.keep_model_hot_in_standby = True
CONFIG.runtime.enable_infer_during_hot_standby = False
CONFIG.runtime.log_mode = "concise"
CONFIG.runtime.log_enabled = True
CONFIG.runtime.debug = False
CONFIG.runtime.capability_placeholder = platform.system().lower().startswith("win")
CONFIG.runtime.heartbeat_enabled = True
CONFIG.runtime.heartbeat_interval_s = 2.0
CONFIG.runtime.console_mode = "operator"
CONFIG.runtime.operator_summary_interval_s = 1.0
CONFIG.runtime.ipc_console = False
CONFIG.runtime.heartbeat_console = False

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
CONFIG.model.active_model = "yolo26s_detect"
CONFIG.model.enable_yolo26 = True
CONFIG.model.enable_yolo_table_search = False
CONFIG.model.profiles["yolov7_detect"] = SingleModelConfig(
    target_model=str(_DEFAULT_DETECT_MODEL),
    width=640,
    height=640,
    conf_thres=0.25,
    iou_thres=0.45,
    class_num=80,
    classes=coco80,
    predictor_type="detect",
    model_backend="qnn",
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
    target_model=str(_DEFAULT_SEG_MODEL_QNN216),
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
    target_model=str(_DEFAULT_SEG_MODEL_QNN216),
    width=640,
    height=640,
    conf_thres=0.25,
    iou_thres=0.15,
    class_num=20,
    classes=grasping_coco20,
    predictor_type="segment",
    model_backend="qnn",
)
CONFIG.model.profiles["yolo26s_detect"] = SingleModelConfig(
    target_model=str(
        Path(_DEFAULT_MODEL_ROOT)
        / "yolo26s"
        / "models"
        / "finetune"
        / "yolo26s-cutoff-bgr_qcs6490_w8a8.qnn236.ctx.bin"
    ),
    width=640,
    height=640,
    conf_thres=0.25,
    iou_thres=0.45,
    class_num=15,
    classes=finetune_yolo26s_bgr15,
    predictor_type="detect26",
    model_backend="qnn",
)

# debug config
_preview_default = "0" if platform.system().lower().startswith("win") else "1"
CONFIG.debug.preview = str(_preview_default).strip().lower() in {"1", "true", "yes"}
CONFIG.debug.draw_boxes = True
CONFIG.debug.draw_masks = False


_apply_vision_params(_load_config_dict(CONFIG.runtime.vision_params_file))
_loaded(CONFIG.runtime.vision_params_file)
