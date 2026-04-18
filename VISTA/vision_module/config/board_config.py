#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import platform
from pathlib import Path

from .schema import VisionServiceConfig, SingleModelConfig
from .data import coco80, grasping_coco20

CONFIG = VisionServiceConfig()

_HERE = Path(__file__).resolve()
_DEFAULT_PROJECT_ROOT = str(_HERE.parents[2]) if os.name == "nt" else "/home/aidlux/2026/VISTA"
_DEFAULT_MODEL_ROOT = str(Path(_DEFAULT_PROJECT_ROOT) / "vision_module" / "model")

CONFIG.runtime.project_root = os.getenv("VISION_PROJECT_ROOT", _DEFAULT_PROJECT_ROOT)
CONFIG.runtime.log_dir = os.getenv("VISION_LOG_DIR", f"{CONFIG.runtime.project_root}/logs")
CONFIG.runtime.log_file = os.getenv("VISION_LOG_FILE", f"{CONFIG.runtime.log_dir}/vision.log")
CONFIG.runtime.runs_dir = os.getenv("VISION_RUNS_DIR", f"{CONFIG.runtime.project_root}/runs")
CONFIG.runtime.pid_dir = os.getenv("VISION_PID_DIR", f"{CONFIG.runtime.project_root}/pids")
CONFIG.runtime.pid_file = os.getenv("VISION_PID_FILE", f"{CONFIG.runtime.pid_dir}/vision.pid")
CONFIG.runtime.stack_run_id = os.getenv("STACK_RUN_ID", "")
CONFIG.runtime.loop_hz = 8.0
CONFIG.runtime.send_hz = 5.0
CONFIG.runtime.stale_req_s = 3.0
CONFIG.runtime.hot_standby_s = 30.0
CONFIG.runtime.keep_preview_after_stop = True
CONFIG.runtime.keep_model_hot_in_standby = True
CONFIG.runtime.enable_infer_during_hot_standby = False
CONFIG.runtime.log_mode = os.getenv("VISION_LOG_MODE", "concise")
CONFIG.runtime.log_enabled = os.getenv("VISION_LOG_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
CONFIG.runtime.debug = os.getenv("VISION_DEBUG", "0").strip().lower() in {"1", "true", "yes"}
_placeholder_default = "1" if platform.system().lower().startswith("win") else "0"
CONFIG.runtime.capability_placeholder = os.getenv("VISION_CAPABILITY_PLACEHOLDER", _placeholder_default).strip().lower() in {"1", "true", "yes"}
CONFIG.runtime.heartbeat_enabled = os.getenv("VISION_HEARTBEAT_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
CONFIG.runtime.heartbeat_interval_s = float(os.getenv("VISION_HEARTBEAT_INTERVAL_S", "2.0") or 2.0)

# camera config
rgb = CONFIG.camera.streams["rgb"]
rgb.source = "6"
rgb.in_w = 1280
rgb.in_h = 720
rgb.out_w = 640
rgb.out_h = 640
rgb.format = "RGB"
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
        str(Path(_DEFAULT_MODEL_ROOT) / "cutoff_yolov7_w8a8.qnn216.ctx.bin"),
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
    target_model=str(Path(_DEFAULT_MODEL_ROOT) / "yolo26s-seg-grasp" / "yolo26s-seg-grasp_split_qcs6490_w8a8.qnn236.ctx.bin.amf"),
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
