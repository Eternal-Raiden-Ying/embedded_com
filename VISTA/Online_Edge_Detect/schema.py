#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from pathlib import Path


_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_LOG_DIR = _DEFAULT_PROJECT_ROOT / "logs"
_DEFAULT_RUNS_DIR = _DEFAULT_PROJECT_ROOT / "runs"
_DEFAULT_PID_DIR = _DEFAULT_PROJECT_ROOT / "pids"
_DEFAULT_SNAPSHOT_DIR = _DEFAULT_PROJECT_ROOT / "snapshots"
_DEFAULT_CALIB_JSON = _DEFAULT_PROJECT_ROOT.parent / "Offline_Edge_Test" / "calib.json"


@dataclass
class RuntimeConfig:
    project_root: str = field(default_factory=lambda: str(_DEFAULT_PROJECT_ROOT))
    log_dir: str = field(default_factory=lambda: str(_DEFAULT_LOG_DIR))
    log_file: str = field(default_factory=lambda: str(_DEFAULT_LOG_DIR / "online_edge.log"))
    runs_dir: str = field(default_factory=lambda: str(_DEFAULT_RUNS_DIR))
    pid_dir: str = field(default_factory=lambda: str(_DEFAULT_PID_DIR))
    pid_file: str = field(default_factory=lambda: str(_DEFAULT_PID_DIR / "online_edge.pid"))
    stack_run_id: str = ""
    loop_hz: float = 10.0
    preview: bool = True
    save_snapshot_period_s: float = 0.0
    snapshot_dir: str = field(default_factory=lambda: str(_DEFAULT_SNAPSHOT_DIR))
    log_mode: str = "concise"
    log_enabled: bool = True


@dataclass
class OutputConfig:
    transport: str = "disabled"
    host: str = "127.0.0.1"
    port: int = 9102
    uds_path: str = "/tmp/robot_stack/table_edge_obs.sock"
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
    calib_json: str = field(default_factory=lambda: str(_DEFAULT_CALIB_JSON))
    target_dist_m_override: float = 0.50
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
    plane_only_mode: bool = True
    enable_crease_line: bool = False
    trend_window_px: int = 12
    trend_col_step_px: int = 6
    trend_min_valid_ratio: float = 0.70
    trend_min_slope_delta: float = 0.0025
    trend_min_candidate_count: int = 35
    trend_topk_per_col: int = 3
    upper_line_y_norm_min: float = 0.12
    upper_line_y_norm_max: float = 0.62
    lower_line_y_norm_min: float = 0.42
    lower_line_y_norm_max: float = 0.92
    line_min_x_span_m: float = 0.18
    line_max_residual_m: float = 0.035
    line_select_min_confidence: float = 0.35
    line_select_min_x_span_m: float = 0.16
    line_select_max_residual_m: float = 0.040
    line_select_max_plane_yaw_diff_rad: float = 0.22
    line_plane_boundary_soft_dist_px: float = 14.0
    line_plane_boundary_max_dist_px: float = 32.0
    line_plane_boundary_weight: float = 0.20
    line_object_like_max_score: float = 0.68
    line_object_like_penalty_weight: float = 0.25
    plane_min_inliers: int = 220
    plane_min_x_span_m: float = 0.20
    plane_max_residual_m: float = 0.035
    front_plane_min_score: float = 0.45
    front_plane_min_area_ratio: float = 0.03
    front_plane_min_x_span_m: float = 0.20
    front_plane_max_residual_m: float = 0.035
    plane_max_abs_normal_y: float = 0.70
    plane_min_abs_normal_z: float = 0.25
    front_face_min_area_ratio: float = 0.03
    fusion_yaw_consistency_rad: float = 0.18
    table_geometry_approach_score: float = 0.35
    table_geometry_alignment_score: float = 0.55
    table_geometry_stop_score: float = 0.70
    front_plane_score_weight: float = 0.35
    line_score_weight: float = 0.25
    plane_line_consistency_weight: float = 0.15
    roi_boundary_score_weight: float = 0.10
    temporal_score_weight: float = 0.15
    roi_boundary_margin_px: int = 8
    roi_boundary_max_touch_ratio: float = 0.25
    fusion_line_min_boundary_consistency: float = 0.45
    fusion_plane_prefer_boundary_consistency: float = 0.65
    control_min_confidence: float = 0.45
    control_approach_min_score: float = 0.35
    control_alignment_min_score: float = 0.55
    control_stop_min_score: float = 0.70
    control_min_stable_frames: int = 3
    control_max_yaw_jump_rad: float = 0.18
    control_max_dist_jump_m: float = 0.12
    control_max_yaw_rad: float = 0.70
    control_approach_min_stable_frames: int = 1
    control_alignment_min_stable_frames: int = 3
    control_stop_min_stable_frames: int = 5
    control_stop_dist_abs_max_m: float = 0.08


@dataclass
class OnlineEdgeConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    camera: RealSenseConfig = field(default_factory=RealSenseConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
