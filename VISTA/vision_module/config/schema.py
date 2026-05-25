#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union


_HERE = Path(__file__).resolve()
_DEFAULT_PROJECT_ROOT = _HERE.parents[2]
_DEFAULT_LOG_DIR = _DEFAULT_PROJECT_ROOT / "logs"
_DEFAULT_RUNS_DIR = _DEFAULT_PROJECT_ROOT / "runs"
_DEFAULT_PID_DIR = _DEFAULT_PROJECT_ROOT / "pids"


@dataclass
class RuntimeConfig:
    project_root: str = field(default_factory=lambda: str(_DEFAULT_PROJECT_ROOT))
    log_dir: str = field(default_factory=lambda: str(_DEFAULT_LOG_DIR))
    log_file: str = field(default_factory=lambda: str(_DEFAULT_LOG_DIR / "vision.log"))
    runs_dir: str = field(default_factory=lambda: str(_DEFAULT_RUNS_DIR))
    pid_dir: str = field(default_factory=lambda: str(_DEFAULT_PID_DIR))
    pid_file: str = field(default_factory=lambda: str(_DEFAULT_PID_DIR / "vision.pid"))
    vision_params_file: str = ""
    loaded_config_files: list = field(default_factory=list)
    stack_run_id: str = ""
    loop_hz: float = 8.0
    send_hz: float = 5.0
    track_local_send_hz: float = 8.0
    stale_req_s: float = 3.0
    hot_standby_s: float = 30.0
    keep_preview_after_stop: bool = True
    keep_model_hot_in_standby: bool = True
    enable_infer_during_hot_standby: bool = False
    capability_placeholder: bool = False
    heartbeat_enabled: bool = False
    heartbeat_interval_s: float = 5.0
    heartbeat_console: bool = False
    console_mode: str = "operator"
    operator_summary_interval_s: float = 1.0
    ipc_console: bool = False
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
    height: int = 240
    width:  int = 424
    fps:    int = 15
    enable: bool = False

@dataclass
class IRCameraConfig:
    source: str = "4"
    in_w: int = 640
    in_h: int = 480
    out_w: int = 640
    out_h: int = 480
    in_format: str = "GRAY8"
    format: str = "BGR"
    fps: int = 30
    crop_x: int = 0
    crop_y: int = 0
    crop_w: int = 0
    crop_h: int = 0
    enable: bool = False


@dataclass
class ColorCameraConfig:
    source: str = "6"
    in_w: int = 1280
    in_h: int = 720
    out_w: int = 640
    out_h: int = 640
    in_format: str = "YUY2"
    format: str = "BGR"
    fps: int = 30
    crop_x: int = 280
    crop_y: int = 0
    crop_w: int = 720
    crop_h: int = 720
    enable: bool = True
    auto_exposure: bool = True
    exposure: int = None
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
    predictor_type: str = "detect"
    model_backend: str = "qnn"
    anchors: Optional[tuple] = None
    strides: Optional[tuple] = None


@dataclass
class ModelConfig:
    active_model: str = "yolov7_detect"
    profiles: Dict[str, SingleModelConfig] = field(default_factory=dict)


@dataclass
class DebugConfig:
    preview: bool = False
    draw_boxes: bool = True
    draw_masks: bool = False
    edge_debug_enabled: bool = False
    edge_debug_period_s: float = 1.0
    table_det_enabled: bool = False
    table_det_min_conf: float = 0.25
    table_det_center_tol: float = 0.12


@dataclass
class TableEdgeConfig:
    roi_preset: str = ""
    static_roi_enabled: bool = False
    detector_mode: str = "full"
    fast_plane_stride: int = 4
    camera_pitch_deg: float = 30.0
    camera_height_m: float = 0.60
    camera_roll_deg: float = 0.0
    camera_yaw_deg: float = 0.0
    table_height_m: float = 0.40
    front_face_z_min_m: float = 0.03
    front_face_z_max_m: float = 0.43
    min_vertical_z_span_m: float = 0.12
    min_vertical_support_points: int = 3
    x_bin_width_m: float = 0.04
    y_cluster_bin_m: float = 0.04
    min_front_face_columns: int = 3
    min_front_face_x_span_m: float = 0.07
    front_cluster_gap_m: float = 0.10
    max_yaw_abs_rad: float = 0.75
    target_hz: float = 10.0
    update_hz: float = 10.0
    preview_hz: float = 2.0
    track_local_update_hz: float = 5.0
    track_local_light_edge: bool = False
    track_local_edge_stride: int = 4
    require_yolo_table_confirm: bool = False
    enable_yolo_in_plane_only: bool = False
    yolo_table_min_conf: float = 0.25
    save_debug_frames: bool = False
    profile_log_interval_s: float = 2.0


@dataclass
class PreviewConfig:
    mode_layouts: Dict[str, str] = field(default_factory=lambda: {
        "IDLE": "rgb_minimal",
        "DEPTH_PERCEPTION": "rgb_depth_edge",
        "TABLE_EDGE_PERCEPTION": "rgb_depth_edge",
        "TRACK_LOCAL": "rgb_yolo_edge_overlay",
        "MICRO_ADJUST": "rgb_minimal",
        "GRASP_REMOTE": "rgb_depth_edge",
        "IDLE_HOT": "rgb_hot_preview",
    })
    debug_four_panel_in_track_local: bool = False
    show_edge_overlay_in_track_local: bool = True
    show_age_ms: bool = True
    clear_overlay_on_mode_switch: bool = True


@dataclass
class VisionServiceConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    table_edge: TableEdgeConfig = field(default_factory=TableEdgeConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    mode_profiles: Dict[str, Dict] = field(default_factory=dict)
    req_in: IPCConfig = field(default_factory=IPCConfig)
    obs_out: IPCConfig = field(default_factory=IPCConfig)
