#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field


@dataclass
class RuntimeConfig:
    project_root: str = "."
    log_dir: str = "logs"
    log_file: str = "logs/online_edge.log"
    runs_dir: str = "runs"
    pid_dir: str = "pids"
    pid_file: str = "pids/online_edge.pid"
    stack_run_id: str = ""
    loop_hz: float = 10.0
    preview: bool = True
    save_snapshot_period_s: float = 0.0
    snapshot_dir: str = "snapshots"
    log_mode: str = "concise"
    log_enabled: bool = True


@dataclass
class OutputConfig:
    transport: str = "disabled"
    host: str = "127.0.0.1"
    port: int = 9102
    uds_path: str = "table_edge_obs.sock"
    send_interval_s: float = 0.20


@dataclass
class RealSenseConfig:
    bag_path: str = ""
    align_to_color: bool = True
    depth_enabled: bool = True
    depth_width: int = 424
    depth_height: int = 240
    depth_fps: int = 15
    color_enabled: bool = True
    color_width: int = 1280
    color_height: int = 720
    color_fps: int = 15


@dataclass
class DetectorConfig:
    calib_json: str = "../calib.json"
    target_dist_m_override: float = -1.0
    roi_y0: int = 100
    roi_y1: int = 380
    roi_x0: int = 100
    roi_x1: int = 540
    z_min: float = 0.2
    z_max: float = 2.0
    table_y_min: float = -0.2
    table_y_max: float = 0.2
    min_all_points: int = 1000
    min_table_points: int = 500
    ransac_iters: int = 120
    residual_threshold_m: float = 0.05
    random_seed: int = 42
    depth_median_ksize: int = 5


@dataclass
class OnlineEdgeConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    camera: RealSenseConfig = field(default_factory=RealSenseConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
