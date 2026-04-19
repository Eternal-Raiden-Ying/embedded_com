#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from .schema import VisionServiceConfig, SingleModelConfig
from .data import coco80, grasping_coco20

CONFIG = VisionServiceConfig()

CONFIG.runtime.project_root = os.getenv("VISION_PROJECT_ROOT", "/home/aidlux/2026/VISTA")
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
CONFIG.model.active_model = "yolo26s_seg"
CONFIG.model.profiles["yolov8s_seg"] = SingleModelConfig(
    target_model="/home/aidlux/2026/VISTA/model/cutoff_yolov8s-seg_qcs6490_w8a8.qnn236.ctx.bin",
    width=640, height=640, conf_thres=0.45, iou_thres=0.45, class_num=80, classes=coco80,
)
CONFIG.model.profiles["yolo26s_seg"] = SingleModelConfig(
    target_model="/home/aidlux/2026/VISTA/vision_module/model/yolo26s-seg-grasp/yolo26s-seg-grasp_split_w8a8.qnn216.ctx.bin.amf",
    width=640, height=640, conf_thres=0.25, iou_thres=0.15, class_num=20, classes=grasping_coco20,
)

# debug config
CONFIG.debug.preview = True
CONFIG.debug.draw_boxes = True
CONFIG.debug.draw_masks = False

# orchestrator -> vision
CONFIG.req_in.transport = "tcp"
CONFIG.req_in.host = "127.0.0.1"
CONFIG.req_in.port = 9003
CONFIG.req_in.uds_path = "/tmp/robot_stack/vision_req.sock"

# vision -> orchestrator
CONFIG.obs_out.transport = "tcp"
CONFIG.obs_out.host = "127.0.0.1"
CONFIG.obs_out.port = 9002
CONFIG.obs_out.uds_path = "/tmp/robot_stack/vision_obs.sock"
