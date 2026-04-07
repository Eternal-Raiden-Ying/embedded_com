#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Dict, Optional, Union


@dataclass
class RuntimeConfig:
    project_root: str = "/home/aidlux/2026/VISTA"
    log_dir: str = "/home/aidlux/2026/VISTA/logs"
    log_file: str = "/home/aidlux/2026/VISTA/logs/vision.log"
    runs_dir: str = "/home/aidlux/2026/VISTA/runs"
    pid_dir: str = "/home/aidlux/2026/VISTA/pids"
    pid_file: str = "/home/aidlux/2026/VISTA/pids/vision.pid"
    stack_run_id: str = ""
    loop_hz: float = 8.0
    send_hz: float = 5.0
    stale_req_s: float = 3.0
    hot_standby_s: float = 30.0
    keep_preview_after_stop: bool = True
    keep_model_hot_in_standby: bool = True
    enable_infer_during_hot_standby: bool = False
    log_mode: str = "concise"
    log_enabled: bool = True
    debug: bool = False


@dataclass
class IPCConfig:
    transport: str = "tcp"  # tcp / uds / disabled
    host: str = "127.0.0.1"
    port: int = 0
    uds_path: str = ""


@dataclass
class DepthCameraConfig:
    source: str = "2"
    height: int = 1280
    width:  int = 720
    fps:    int = 30
    enable: bool = False

@dataclass
class IRCameraConfig:
    source: str = "4"
    in_w: int = 1280
    in_h: int = 720
    out_w: int = 640
    out_h: int = 640
    in_format: str = "GREY"
    format: str = "GRAY8" 
    fps: int = 30
    crop_x: int = 280
    crop_y: int = 0
    crop_w: int = 720
    crop_h: int = 720
    enable: bool = False


@dataclass
class ColorCameraConfig:
    source: str = "6"
    in_w: int = 1280
    in_h: int = 720
    out_w: int = 640
    out_h: int = 640
    in_format: str = "YUY2"
    format: str = "RGB"  # RGB
    fps: int = 30
    crop_x: int = 280
    crop_y: int = 0
    crop_w: int = 720
    crop_h: int = 720
    enable: bool = True
    auto_exposure: bool = True,    
    exposure: int = None,          
    brightness: int = None

@dataclass
class CameraConfig:
    streams: Dict[str, Union[DepthCameraConfig, IRCameraConfig, ColorCameraConfig]] = field(default_factory=lambda: {
        "rgb": ColorCameraConfig(source="6", enable=True),
        "depth": DepthCameraConfig(source="2", enable=False),
        "grey": IRCameraConfig(source="4", enable=False),
    })
    max_fps: int = 30


@dataclass
class SingleModelConfig:
    target_model: str = ""
    width: int = 640
    height: int = 640
    conf_thres: float = 0.45
    iou_thres: float = 0.45
    class_num: int = 80
    classes: Optional[tuple] = None


@dataclass
class ModelConfig:
    active_model: str = "yolov8s_seg"
    profiles: Dict[str, SingleModelConfig] = field(default_factory=dict)


@dataclass
class DebugConfig:
    preview: bool = False
    draw_boxes: bool = True
    draw_masks: bool = False


@dataclass
class VisionServiceConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    req_in: IPCConfig = field(default_factory=IPCConfig)
    obs_out: IPCConfig = field(default_factory=IPCConfig)
