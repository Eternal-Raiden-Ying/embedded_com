#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified configuration schemas for the entire robot stack (Vision, Orchestrator, Gateway)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

# Base path definitions relative to workspace root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ORCH_ROOT = _REPO_ROOT / "orchestrator"
_VISION_ROOT = _REPO_ROOT / "VISTA"
_DEFAULT_MODEL_ROOT = _VISION_ROOT / "vision_module" / "model"
_DEFAULT_DETECT_MODEL = (
    _DEFAULT_MODEL_ROOT
    / "qnn216"
    / "model_farm_yolov7_qcs6490_qnn2.16_int8_aidlite"
    / "models"
    / "cutoff_yolov7_w8a8.qnn216.ctx.bin"
)
_DEFAULT_SEG_MODEL_QNN216 = (
    _DEFAULT_MODEL_ROOT
    / "yolo26s-seg-grasp"
    / "yolo26s-seg-grasp_split_w8a8.qnn216.ctx.bin"
)

# Default coco categories
_COCO80 = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
)

_FINETUNE_YOLO26S_BGR15 = (
    "tag_home", "tag_station", "car", "table", "person", "chair", "couch", "bed",
    "bottle", "cup", "bowl", "apple", "banana", "orange", "basket"
)

_GRASPING_COCO20 = (
    "person", "backpack", "umbrella", "handbag", "tie", "suitcase", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog"
)


# ==============================================================================
# Shared / Common Configs
# ==============================================================================

@dataclass
class SocketEndpoint:
    """Socket communication endpoint configuration."""
    transport: str = "uds"  # tcp / uds / disabled
    ipc_socket_path: str = ""
    tcp_host: str = "127.0.0.1"
    tcp_port: int = 0
    send_mode: str = "persistent"  # persistent / oneshot
    async_enabled: bool = False
    async_queue_size: int = 64
    async_drop_oldest: bool = True

    @property
    def uds_path(self) -> str:
        return self.ipc_socket_path

    @uds_path.setter
    def uds_path(self, value: str) -> None:
        self.ipc_socket_path = value

    @property
    def host(self) -> str:
        return self.tcp_host

    @host.setter
    def host(self, value: str) -> None:
        self.tcp_host = value

    @property
    def port(self) -> int:
        return self.tcp_port

    @port.setter
    def port(self, value: int) -> None:
        self.tcp_port = int(value)


# ==============================================================================
# Vision Module Configs
# ==============================================================================

@dataclass
class VisionRuntimeConfig:
    """Runtime environment configuration for the vision service."""
    project_root: str = field(default_factory=lambda: str(_VISION_ROOT))
    log_dir: str = field(default_factory=lambda: str(_VISION_ROOT / "logs"))
    log_file: str = field(default_factory=lambda: str(_VISION_ROOT / "logs" / "vision.log"))
    runs_dir: str = field(default_factory=lambda: str(_REPO_ROOT / "logs" / "runs"))
    pid_dir: str = field(default_factory=lambda: str(_VISION_ROOT / "pids"))
    pid_file: str = field(default_factory=lambda: str(_VISION_ROOT / "pids" / "vision.pid"))
    vision_params_file: str = ""
    loaded_config_files: List[str] = field(default_factory=list)
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
    """IPC transport configurations for incoming and outgoing data."""
    transport: str = "uds"
    ipc_socket_path: str = ""


@dataclass
class DepthCameraConfig:
    source: str = "2"
    height: int = 240
    width: int = 424
    fps: int = 15
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
    exposure: Optional[int] = None
    brightness: Optional[int] = None


@dataclass
class CameraConfig:
    streams: Dict[str, Union[DepthCameraConfig, IRCameraConfig, ColorCameraConfig]] = field(
        default_factory=lambda: {
            "rgb": ColorCameraConfig(source="6", enable=True),
            "depth": DepthCameraConfig(source="2", enable=False),
            "grey": IRCameraConfig(source="4", enable=False),
        }
    )
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
    active_model: str = "yolo26s_detect"
    profiles: Dict[str, SingleModelConfig] = field(default_factory=lambda: {
        "yolov7_detect": SingleModelConfig(
            target_model=str(_DEFAULT_DETECT_MODEL),
            width=640,
            height=640,
            conf_thres=0.25,
            iou_thres=0.45,
            class_num=80,
            classes=_COCO80,
            predictor_type="detect",
            model_backend="qnn",
            anchors=(
                (12, 16, 19, 36, 40, 28),
                (36, 75, 76, 55, 72, 146),
                (142, 110, 192, 243, 459, 401),
            ),
            strides=(8, 16, 32),
        ),
        "yolov8s_seg": SingleModelConfig(
            target_model=str(Path(_DEFAULT_MODEL_ROOT) / "yolov8s-seg" / "cutoff_yolov8s-seg_qcs6490_w8a8.qnn236.ctx.bin"),
            width=640,
            height=640,
            conf_thres=0.45,
            iou_thres=0.45,
            class_num=80,
            classes=_COCO80,
            predictor_type="segment",
            model_backend="qnn",
        ),
        "yolo26s_seg": SingleModelConfig(
            target_model=str(_DEFAULT_SEG_MODEL_QNN216),
            width=640,
            height=640,
            conf_thres=0.25,
            iou_thres=0.15,
            class_num=20,
            classes=_GRASPING_COCO20,
            predictor_type="segment",
            model_backend="qnn",
        ),
        "yolo26s_seg_qnn216": SingleModelConfig(
            target_model=str(_DEFAULT_SEG_MODEL_QNN216),
            width=640,
            height=640,
            conf_thres=0.25,
            iou_thres=0.15,
            class_num=20,
            classes=_GRASPING_COCO20,
            predictor_type="segment",
            model_backend="qnn",
        ),
        "yolo26s_detect": SingleModelConfig(
            target_model=str(
                _DEFAULT_MODEL_ROOT
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
            classes=_FINETUNE_YOLO26S_BGR15,
            predictor_type="detect26",
            model_backend="qnn",
        ),
    })
    enable_yolo26: bool = True
    enable_yolo_table_search: bool = False


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
    table_bbox_enabled: bool = True
    mock_table_bbox: str = ""


@dataclass
class TableEdgeConfig:
    """Table-edge detection parameters (includes all parameters previously hardcoded in business logic)."""
    roi_preset: str = ""
    yolo_table_roi_enable: bool = True
    yolo_table_class_id: int = 0
    yolo_table_conf_min: float = 0.25
    yolo_table_roi_use_rgb_depth_mapping: bool = True
    yolo_table_roi_mode: str = "centered_bbox_scale"
    yolo_table_roi_scale_x: float = 0.50
    yolo_table_roi_scale_y: float = 0.50
    rgb_depth_mapping_mode: str = "centered_scale"
    rgb_fov_in_depth_scale_x: float = 0.75
    rgb_fov_in_depth_scale_y: float = 0.75
    rgb_depth_center_offset_x: float = 0.0
    rgb_depth_center_offset_y: float = 0.0
    yolo_table_bbox_hold_enable: bool = True
    yolo_table_bbox_hold_frames: int = 8
    yolo_table_roi_hold_enable: bool = True
    yolo_table_roi_boundary_extend_enable: bool = True
    yolo_table_roi_boundary_margin_norm: float = 0.03
    yolo_table_edge_stable_frames: int = 5
    edge_trusted_min_conf: float = 0.60
    edge_trusted_max_residual: float = 0.0
    edge_trusted_min_support_count: int = 0
    edge_trusted_min_inlier_count: int = 0
    edge_trusted_min_x_span_m: float = 0.0
    edge_trusted_max_background_penalty: float = 0.0
    yolo_table_near_dist_m: float = 0.45
    yolo_table_near_bottom_norm: float = 0.60
    profile_log_interval_s: float = 2.0
    save_debug_frames: bool = False
    target_hz: float = 10.0
    preview_hz: float = 2.0
    fast_debug_pixels: bool = True
    fast_debug_pixels_online: bool = False
    fast_debug_pixels_offline: bool = True
    fast_debug_pixel_cap: int = 300
    fast_candidate_point_cap: int = 1800
    fast_front_edge_col_step: int = 2
    fast_front_edge_row_step: int = 2
    depth_stride: int = 2

    # Previously hardcoded configurations inside table_edge_manager.py business layer
    detector_mode: str = "lightweight"
    update_hz: float = 5.0
    light_stride: int = 4
    fast_plane_stride: int = 4
    require_yolo_confirm: bool = True
    static_roi_enabled: bool = False
    camera_pitch_deg: float = 15.0
    camera_height_m: float = 0.70
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
    enable_yolo_in_plane_only: bool = False
    yolo_table_min_conf: float = 0.25


@dataclass
class PreviewConfig:
    mode_layouts: Dict[str, str] = field(
        default_factory=lambda: {
            "IDLE": "rgb_minimal",
            "FIND_EDGE": "rgb_depth_edge",
            "FIND_OBJECT": "rgb_yolo_edge_overlay",
            "MICRO_ADJUST": "rgb_minimal",
            "GRASP_REMOTE": "rgb_depth_edge",
            "IDLE_HOT": "rgb_hot_preview",
        }
    )
    debug_four_panel_in_track_local: bool = False
    show_edge_overlay_in_track_local: bool = True
    show_age_ms: bool = True
    clear_overlay_on_mode_switch: bool = True
    scale: float = 1.0
    canvas_w: int = 1280
    canvas_h: int = 720
    show_rgb: bool = True
    show_depth: bool = True
    show_edge: bool = True
    destroy_all_on_close: bool = True


@dataclass
class VisionServiceConfig:
    """Configuration structure representing the Vision Service."""
    runtime: VisionRuntimeConfig = field(default_factory=VisionRuntimeConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    table_edge: TableEdgeConfig = field(default_factory=TableEdgeConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    mode_profiles: Dict[str, Dict] = field(default_factory=dict)
    req_in: IPCConfig = field(default_factory=IPCConfig)
    obs_out: IPCConfig = field(default_factory=IPCConfig)


# ==============================================================================
# Orchestrator Module Configs
# ==============================================================================

@dataclass
class OrchestratorRuntimeConfig:
    """Runtime environment configuration for the orchestrator."""
    project_root: str = field(default_factory=lambda: str(_ORCH_ROOT))
    log_dir: str = field(default_factory=lambda: str(_ORCH_ROOT / "logs"))
    log_file: str = field(default_factory=lambda: str(_ORCH_ROOT / "logs" / "orchestrator.log"))
    runs_dir: str = field(default_factory=lambda: str(_ORCH_ROOT / "runs"))
    pid_dir: str = field(default_factory=lambda: str(_ORCH_ROOT / "pids"))
    pid_file: str = field(default_factory=lambda: str(_ORCH_ROOT / "pids" / "orchestrator.pid"))
    stack_run_id: str = ""
    tick_hz: float = 10.0
    log_mode: str = "concise"
    log_enabled: bool = True
    debug: bool = False
    state_block_period_s: float = 1.0
    heartbeat_period_s: float = 1.0
    stage_params_file: str = ""
    car_cmd_params_file: str = ""
    config_profile: str = ""
    loaded_config_files: List[str] = field(default_factory=list)


@dataclass
class SerialConfig:
    """Serial communication port configuration for microcontrollers."""
    port: str = "/dev/ttyHS1"
    baudrate: int = 115200
    timeout_s: float = 0.10
    dry_run: bool = False
    readback_enabled: bool = True
    dry_run_echo_stdout: bool = False
    dry_run_echo_on_change_only: bool = True
    dry_run_echo_summary_period_s: float = 5.0
    dry_run_quiet_idle_stop: bool = True
    uart_lowfreq_period_s: float = 5.0
    stm32_status_enabled: bool = False
    stm32_status_period_s: float = 1.0


@dataclass
class ControlThresholds:
    """Control and timing thresholds for the vehicle/perception state machine."""
    cmd_confidence_th: float = 0.60
    target_obs_max_age_s: float = 1.00
    table_obs_max_age_s: float = 1.00
    home_obs_max_age_s: float = 1.00

    search_table_timeout_s: float = 20.0
    approach_timeout_s: float = 14.0
    target_search_timeout_s: float = 10.0
    return_search_timeout_s: float = 15.0
    req_resend_period_s: float = 1.0

    table_found_frames_to_approach: int = 2
    table_lost_frames_to_reacquire: int = 4
    table_loss_hold_s: float = 1.20
    approach_min_dwell_s: float = 0.80
    table_approach_warmup_s: float = 2.0
    table_approach_warmup_min_fresh_obs: int = 1

    coarse_align_frames_to_advance: int = 2
    coarse_align_done_rad: float = 0.25
    align_to_approach_yaw_rad: float = 0.08
    approach_to_align_yaw_rad: float = 0.16
    align_to_approach_stable_obs: int = 2
    approach_to_align_stable_obs: int = 2
    coarse_align_min_dwell_s: float = 0.80
    controlled_approach_min_dwell_s: float = 0.80
    final_lock_frames_to_arrive: int = 3
    final_lock_yaw_tol_rad: float = 0.25
    final_lock_dist_tol_m: float = 0.03  # Strict distance tolerance for declaring final lock stop condition
    final_lock_lateral_tol_m: float = 0.03
    table_edge_only_test: bool = False
    table_target_dist_m: float = 0.30  # Nominal target docking distance (stopped position relative to table edge)
    table_dist_tol_m: float = 0.05     # Allowable distance error tolerance during docking/alignment
    table_yaw_tol_rad: float = 0.13962634015954636  # Target yaw alignment error threshold (8 degrees in radians)
    table_stop_margin_m: float = 0.05  # Safety stop margin added to target distance in stop conditions checking
    table_settle_s: float = 0.50
    table_stable_frames: int = 5
    yolo_table_control_enable: bool = True
    yolo_table_conf_min: float = 0.25
    yolo_table_edge_stable_frames: int = 5
    edge_trusted_stable_frames: int = 5
    edge_trusted_min_conf: float = 0.60
    edge_trusted_max_residual: float = 0.05
    edge_trusted_min_support_count: int = 0
    edge_trusted_min_inlier_count: int = 0
    edge_trusted_min_x_span_m: float = 0.0
    edge_trusted_max_background_penalty: float = 0.0
    yolo_table_near_dist_m: float = 0.45
    near_slow_depth_m: float = 0.40
    near_stop_depth_m: float = 0.25
    yolo_table_lost_to_search_frames: int = 8
    table_memory_timeout_sec: float = 3.0
    table_center_loss_hold_sec: float = 1.0
    rotate_search_timeout_s: float = 10.0
    rotate_require_edge_stable_frames: int = 5
    rotate_yaw_threshold_rad: float = 0.20
    yolo_edge_conflict_block_rotate: bool = True
    final_lock_required_ready_obs: int = 3
    final_lock_window_ms: int = 1000
    final_lock_max_consecutive_lost: int = 2
    final_lock_soft_stale_hold: bool = True
    table_max_micro_adjust: int = 3
    enable_final_lock: bool = True
    enable_micro_adjust: bool = False
    final_lock_enter_dist_th_m: float = 0.08
    final_lock_enter_yaw_th_rad: float = 0.10
    edge_settle_s: float = 0.80
    dock_retry_limit: int = 2
    dock_retry_backoff_s: float = 0.60

    search_target_init_hold_s: float = 0.25
    target_found_frames_to_confirm: int = 3
    target_confirm_conf_th: float = 0.30
    target_confirm_dwell_s: float = 0.0
    target_confirm_min_s: float = 0.80
    target_confirm_timeout_s: float = 3.00
    target_confirm_lost_frames: int = 2
    target_confirm_lost_hold_s: float = 1.20
    target_confirm_min_bbox_area: float = 0.0
    target_confirm_window_s: float = 1.50
    target_confirm_found_ratio_th: float = 0.50
    target_lock_conf_th: float = 0.40
    target_lock_found_ratio_th: float = 0.60
    target_lock_settle_s: float = 0.50
    target_lock_stable_s: float = 1.20
    target_lock_center_jitter_th: float = 0.08
    target_lock_lost_hold_s: float = 1.50
    target_locked_freeze_after_s: float = 1.00
    freeze_settle_s: float = 0.60
    edge_slide_pause_s: float = 0.20
    edge_slide_segment_s: float = 1.20
    edge_slide_dist_tolerance_m: float = 0.05
    edge_slide_fallback_state: str = "FINAL_LOCK"
    edge_slide_pause_hold_s: float = 0.80
    edge_slide_dist_out_of_range_hold_s: float = 0.80
    edge_slide_max_relock_attempts: int = 3
    edge_slide_relock_failure_is_fatal: bool = True
    edge_slide_recover_timeout_s: float = 2.50
    edge_slide_direct_fallback_to_controlled_approach: bool = False
    table_edge_obs_max_age_ms: int = 500
    table_obs_stale_soft_ms: int = 300
    table_obs_stale_stop_ms: int = 500
    table_obs_stale_hard_ms: int = 800
    table_step_mode_enable: bool = False
    table_step_burst_ms: int = 150
    table_step_hold_until_new_obs: bool = True
    edge_follow_log_period_ms: int = 500
    edge_follow_min_edge_conf: float = 0.60
    edge_follow_min_edge_conf_table_edge_perception: float = 0.60
    edge_follow_min_edge_conf_track_local: float = 0.20
    edge_follow_weak_edge_conf_track_local: float = 0.15
    edge_follow_strong_edge_conf_track_local: float = 0.35
    edge_follow_low_conf_hold_s: float = 2.00
    edge_follow_low_conf_exit_s: float = 3.00
    edge_follow_recover_conf_th: float = 0.25
    edge_identity_yaw_mismatch_rad: float = 0.15
    edge_identity_dist_mismatch_m: float = 0.04
    edge_follow_stale_fallback_state: str = "FINAL_LOCK"
    edge_follow_stale_hold_s: float = 1.20
    edge_follow_track_local_edge_update_hz: float = 5.0
    edge_handoff_min_s: float = 0.50
    edge_handoff_max_s: float = 1.00
    edge_handoff_samples: int = 3

    edge_relocate_enabled: bool = True
    max_edge_transitions_per_task: int = 3
    leave_edge_backoff_s: float = 0.80
    relocate_turn_s: float = 1.10
    reacquire_timeout_s: float = 8.0
    next_table_dwell_s: float = 1.50

    tag_lost_frames_to_search: int = 4
    return_lost_hold_s: float = 1.00
    return_min_dwell_s: float = 0.60
    return_done_distance_m: float = 0.35
    tag_arrived_frames_to_stop: int = 2

    avoid_clear_frames_to_resume: int = 2
    avoid_timeout_s: float = 4.0
    avoid_retry_limit: int = 3

    done_hold_s: float = 1.20
    error_recovery_hold_s: float = 1.20

    car_timeout_to_stop: bool = True
    car_fault_to_fail: bool = True
    car_estop_to_stop: bool = True
    post_stop_ignore_s: float = 0.80
    vision_req_fail_to_stop: bool = True
    vision_req_fail_threshold: int = 2
    enable_pick_pipeline: bool = False
    assume_grasp_success_for_test: bool = False


@dataclass
class CarMotionConfig:
    """PID, limits, and behavior variables for actual motion control."""
    search_table_wz_radps: float = 0.10
    fallback_align_turn_wz_min_radps: float = 0.10
    fallback_align_turn_wz_max_radps: float = 0.45
    fallback_forward_vx_mps_min: float = 0.06
    fallback_forward_vx_mps_max: float = 0.28
    fallback_dead_zone_x: float = 0.10
    fallback_spin_only_x_th: float = 0.82
    fallback_forward_align_exp: float = 2.0

    table_fov_soft_th: float = 0.25
    table_fov_hard_th: float = 0.40
    table_view_memory_ttl_s: float = 0.80
    table_coarse_align_vx_max_mps: float = 0.000
    table_coarse_align_vy_min_mps: float = 0.000
    table_coarse_align_vy_max_mps: float = 0.000
    table_coarse_align_wz_min_radps: float = 0.080
    table_coarse_align_wz_max_radps: float = 0.150
    table_controlled_vx_min_mps: float = 0.020
    table_controlled_vx_max_mps: float = 0.035
    table_controlled_vy_min_mps: float = 0.000
    table_controlled_vy_max_mps: float = 0.000
    table_controlled_wz_min_radps: float = 0.000
    table_controlled_wz_max_radps: float = 0.000
    table_approach_safe_vx_mps: float = 0.020
    table_approach_max_vx_mps: float = 0.035
    table_approach_yaw_deadband_rad: float = 0.08
    table_approach_yaw_realign_rad: float = 0.16
    table_approach_allow_wz: bool = False
    table_approach_allow_vy: bool = False
    table_pose_missing_safe_vx_mps: float = 0.020
    table_pose_missing_max_hold_s: float = 3.0
    table_final_lock_vx_min_mps: float = 0.000
    table_final_lock_vx_max_mps: float = 0.008
    table_final_lock_vy_min_mps: float = 0.006
    table_final_lock_vy_max_mps: float = 0.012
    table_final_lock_wz_min_radps: float = 0.010
    table_final_lock_wz_max_radps: float = 0.025
    table_vx_deadband_mps: float = 0.004
    table_vy_deadband_mps: float = 0.001
    table_wz_deadband_radps: float = 0.006
    table_stage_a_wz_radps: float = 0.04
    table_stage_b_vx_max_mps: float = 0.03
    table_stage_c_vx_max_mps: float = 0.03
    table_stage_c_vx_min_mps: float = 0.0
    table_min_forward_dist_err_m: float = 0.07
    table_vx_mps_min: float = 0.040
    table_vx_mps_max: float = 0.100
    table_vx_kp_mps_per_m: float = 0.30
    table_yaw_slow_th_rad: float = 0.12
    table_yaw_stop_th_rad: float = 0.45
    table_near_dist_err_th_m: float = 0.10
    table_vy_max_mps: float = 0.067
    table_wz_view_max_radps: float = 0.05
    table_wz_plane_max_radps: float = 0.06
    table_dist_kp_mps_per_m: float = 0.12
    yolo_table_yaw_gain: float = 0.20
    yolo_table_max_wz_radps: float = 0.06
    yolo_table_forward_vx_mps: float = 0.015
    table_view_wz_kp: float = 0.18
    table_view_vy_kp: float = 0.04
    table_view_recover_vy_mps: float = 0.008
    table_view_recover_wz_radps: float = 0.04
    table_plane_yaw_kp_radps_per_rad: float = 0.60
    table_view_wz_sign: float = -1.0
    table_view_vy_sign: float = -1.0
    table_plane_yaw_sign: float = -1.0
    table_vx_slew_per_s: float = 0.12
    table_vy_slew_per_s: float = 0.06
    table_wz_slew_per_s: float = 0.18

    return_turn_wz_min_radps: float = 0.20
    return_turn_wz_max_radps: float = 0.75
    return_vx_mps_min: float = 0.10
    return_vx_mps_max: float = 0.45

    edge_slide_vy_mps: float = 0.14
    edge_slide_dist_kp_mps_per_m: float = 1.20
    edge_slide_yaw_kp_radps_per_rad: float = 1.20
    edge_slide_max_vx_mps: float = 0.10
    edge_slide_max_wz_radps: float = 0.12
    edge_slide_weak_vy_mps: float = 0.05
    leave_edge_vx_mps: float = -0.12
    relocate_turn_wz_radps: float = 0.28
    avoid_turn_wz_radps: float = 0.38
    avoid_reverse_vx_mps: float = 0.12

    cmd_hold_ms: int = 150
    send_period_ms: int = 100
    uart_keepalive_hz: float = 10.0
    min_uart_keepalive_hz: float = 7.0
    motion_hold_ms: int = 400
    hard_stale_stop_ms: int = 800
    soft_stale_hold_enable: bool = True
    max_vx_mps: float = 1.0
    max_vy_mps: float = 1.0
    max_wz_radps: float = 1.0
    stm32_wheel_speed_limit: int = 100
    stm32_vx_scale: float = 100.0
    stm32_vy_scale: float = 100.0
    stm32_wz_scale: float = 100.0
    jog_forward_speed: float = 0.02
    jog_turn_speed: float = 0.05
    jog_duration_ms: int = 100
    stop_on_state_enter: bool = False
    emergency_stop_command: str = "STOP"
    soft_stop_command: str = "SSTOP"
    stop_policy: str = "STOP=emergency,SSTOP=soft"
    mode_line_on_change: bool = True
    mode_line_every_cmd: bool = False
    serial_float_digits: int = 3


@dataclass
class PIDAxisConfig:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    integral_limit: float = 0.5
    output_limit: float = 1.0
    derivative_alpha: float = 0.30
    deadband: float = 0.0
    min_abs_output: float = 0.0


@dataclass
class DockingControlConfig:
    """PID controller parameters and targets for precise edge tracking."""
    min_confidence: float = 0.55
    obs_timeout_s: float = 0.35
    dt_min_s: float = 0.02
    reset_on_mode_change: bool = True

    coarse_align_enter_rad: float = 0.18
    coarse_align_exit_rad: float = 0.08
    spin_only_yaw_rad: float = 0.18

    precise_yaw_tol_rad: float = 0.025
    precise_dist_tol_m: float = 0.015  # Distance error stopping tolerance/threshold for precise approach (mapped from controlled_approach.target_dist_m in stage_params.yaml)
    precise_lateral_tol_m: float = 0.015
    precise_stable_s: float = 0.50

    coarse_max_wz_radps: float = 0.45
    approach_max_vx_mps: float = 0.28
    approach_max_vy_mps: float = 0.18
    approach_max_wz_radps: float = 0.32
    final_max_vx_mps: float = 0.12
    final_max_vy_mps: float = 0.12
    final_max_wz_radps: float = 0.18

    vx_slew_per_s: float = 0.80
    vy_slew_per_s: float = 0.80
    wz_slew_per_s: float = 1.20

    enable_lateral_control: bool = True

    yaw_pid: PIDAxisConfig = field(default_factory=lambda: PIDAxisConfig(
        kp=1.8, ki=0.02, kd=0.10, integral_limit=0.40, output_limit=0.80,
        derivative_alpha=0.35, deadband=0.010, min_abs_output=0.06,
    ))
    dist_pid: PIDAxisConfig = field(default_factory=lambda: PIDAxisConfig(
        kp=1.4, ki=0.03, kd=0.08, integral_limit=0.35, output_limit=0.40,
        derivative_alpha=0.30, deadband=0.004, min_abs_output=0.04,
    ))
    lateral_pid: PIDAxisConfig = field(default_factory=lambda: PIDAxisConfig(
        kp=1.2, ki=0.02, kd=0.08, integral_limit=0.30, output_limit=0.30,
        derivative_alpha=0.30, deadband=0.004, min_abs_output=0.04,
    ))


@dataclass
class OrchestratorConfig:
    """Configuration structure representing the Orchestrator service."""
    runtime: OrchestratorRuntimeConfig = field(default_factory=OrchestratorRuntimeConfig)
    serial: SerialConfig = field(default_factory=SerialConfig)
    control: ControlThresholds = field(default_factory=ControlThresholds)
    car: CarMotionConfig = field(default_factory=CarMotionConfig)
    docking: DockingControlConfig = field(default_factory=DockingControlConfig)
    task_cmd_in: SocketEndpoint = field(default_factory=lambda: SocketEndpoint(
        transport="tcp", ipc_socket_path="/tmp/robot_stack/task_cmd.sock", tcp_port=19101,
    ))
    task_ack_out: SocketEndpoint = field(default_factory=lambda: SocketEndpoint(
        transport="tcp", ipc_socket_path="/tmp/robot_stack/task_ack.sock", tcp_port=19102, send_mode="oneshot",
    ))
    vision_obs_in: SocketEndpoint = field(default_factory=lambda: SocketEndpoint(
        transport="tcp", ipc_socket_path="/tmp/robot_stack/vision_obs.sock", tcp_port=19103,
    ))
    vision_req_out: SocketEndpoint = field(default_factory=lambda: SocketEndpoint(
        transport="tcp", ipc_socket_path="/tmp/robot_stack/vision_req.sock", tcp_port=19104, send_mode="oneshot", async_enabled=True,
    ))
    tts_event_out: SocketEndpoint = field(default_factory=lambda: SocketEndpoint(
        transport="disabled", ipc_socket_path="/tmp/robot_stack/tts_event.sock", async_enabled=True,
    ))
    frozen_targets: Dict[str, List[str]] = field(default_factory=lambda: {
        "bottle": ["瓶子", "水瓶", "饮料瓶"],
        "key": ["钥匙", "钥匙串"],
        "keys": ["钥匙", "钥匙串"],
        "apple": ["苹果"],
        "banana": ["香蕉"],
        "basket": ["篮子"],
        "grape": ["葡萄"],
        "kiwi fruit": ["猕猴桃", "奇异果"],
        "kiwi": ["猕猴桃", "奇异果"],
        "lemon": ["柠檬"],
        "mango": ["芒果"],
        "mouse": ["鼠标"],
        "orange": ["橙子"],
        "peach": ["桃子"],
        "star fruit": ["杨桃"],
        "starfruit": ["杨桃"],
        "strawberry": ["草莓"],
    })


# ==============================================================================
# Mobile Gateway Module Configs
# ==============================================================================

@dataclass
class GatewayEndpoint:
    transport: str = "uds"
    ipc_socket_path: str = ""
    tcp_host: str = "127.0.0.1"
    tcp_port: int = 0
    send_mode: str = "oneshot"
    async_enabled: bool = False
    async_queue_size: int = 64
    async_drop_oldest: bool = True

    @property
    def uds_path(self) -> str:
        return self.ipc_socket_path

    @uds_path.setter
    def uds_path(self, value: str) -> None:
        self.ipc_socket_path = value

    @property
    def host(self) -> str:
        return self.tcp_host

    @host.setter
    def host(self, value: str) -> None:
        self.tcp_host = value

    @property
    def port(self) -> int:
        return self.tcp_port

    @port.setter
    def port(self, value: int) -> None:
        self.tcp_port = int(value)


@dataclass
class GatewayRuntimeConfig:
    project_root: str = field(default_factory=lambda: str(_ORCH_ROOT))
    repo_root: str = field(default_factory=lambda: str(_REPO_ROOT))
    log_dir: str = field(default_factory=lambda: str(_ORCH_ROOT / "logs"))
    log_file: str = field(default_factory=lambda: str(_ORCH_ROOT / "logs" / "mobile_gateway.log"))
    runs_dir: str = field(default_factory=lambda: str(_ORCH_ROOT / "runs"))
    pid_dir: str = field(default_factory=lambda: str(_ORCH_ROOT / "pids"))
    pid_file: str = field(default_factory=lambda: str(_ORCH_ROOT / "pids" / "mobile_gateway.pid"))
    stack_run_id: str = ""
    mode: str = "production"
    log_level: str = "INFO"
    tick_hz: float = 10.0
    heartbeat_period_s: float = 1.0
    heartbeat_log_interval_s: float = 30.0
    suppress_heartbeat_success_log: bool = True
    enable_raw_mqtt_debug: bool = False
    enable_legacy_command_compat: bool = True
    cmd_dedup_cache_size: int = 64
    log_mode: str = "concise"
    log_enabled: bool = True
    status_stdout: bool = True
    stdin_enabled: bool = False


@dataclass
class GatewayBackendConfig:
    mode: str = "tcp_no_ack"  # mock / orchestrator_tcp / tcp_no_ack
    default_robot_id: str = "sc171_car_01"
    default_confidence: float = 0.99
    mock_step_interval_s: float = 0.20
    enforce_single_flight: bool = True
    observer_enabled: bool = True
    observer_poll_interval_s: float = 0.25
    orchestrator_runs_dir: str = field(default_factory=lambda: str(_ORCH_ROOT / "runs"))
    state_blocks_path: str = ""
    state_block_log_mode: str = "summary"
    state_block_log_period_s: float = 1.0
    stop_cooldown_s: float = 1.0


@dataclass
class MqttTopicConfig:
    cmd: str = "robot/sc171_car_01/cmd"
    ack: str = "robot/sc171_car_01/ack"
    status: str = "robot/sc171_car_01/status"
    heartbeat: str = "robot/sc171_car_01/heartbeat"


@dataclass
class MqttAdapterConfig:
    enabled: bool = False
    transport: str = "websocket"
    use_tls: bool = True
    broker_host: str = ""
    broker_port: int = 443
    websocket_path: str = "/mqtt"
    username: str = ""
    password: str = ""
    client_id: str = "sc171_car_01"
    robot_id: str = "sc171_car_01"
    cmd_qos: int = 1
    ack_qos: int = 1
    status_qos: int = 0
    heartbeat_qos: int = 0
    retain_status: bool = False
    retain_heartbeat: bool = False
    keepalive_s: int = 60
    connect_timeout_s: float = 5.0
    topics: MqttTopicConfig = field(default_factory=MqttTopicConfig)


@dataclass
class MobileGatewayConfig:
    """Configuration structure representing the Mobile Gateway."""
    config_file: str = ""
    runtime: GatewayRuntimeConfig = field(default_factory=GatewayRuntimeConfig)
    backend: GatewayBackendConfig = field(default_factory=GatewayBackendConfig)
    mqtt: MqttAdapterConfig = field(default_factory=MqttAdapterConfig)
    command_in: GatewayEndpoint = field(default_factory=lambda: GatewayEndpoint(
        transport="http", ipc_socket_path="/tmp/robot_stack/mobile_gateway_cmd.sock", tcp_host="0.0.0.0", tcp_port=9001,
    ))
    status_out: GatewayEndpoint = field(default_factory=lambda: GatewayEndpoint(
        transport="disabled", ipc_socket_path="/tmp/robot_stack/mobile_gateway_status.sock", send_mode="oneshot",
    ))
    orchestrator_task_cmd_out: GatewayEndpoint = field(default_factory=lambda: GatewayEndpoint(
        transport="uds", ipc_socket_path="/tmp/robot_stack/task_cmd.sock", send_mode="oneshot",
    ))
    orchestrator_task_ack_in: GatewayEndpoint = field(default_factory=lambda: GatewayEndpoint(
        transport="disabled", ipc_socket_path="/tmp/robot_stack/mobile_gateway_ack.sock",
    ))


# ==============================================================================
# Online Edge Detector Configs
# ==============================================================================

@dataclass
class OnlineEdgeRuntimeConfig:
    project_root: str = field(default_factory=lambda: str(_VISION_ROOT / "vision_module" / "backend" / "edge_detect"))
    log_dir: str = field(default_factory=lambda: str(_VISION_ROOT / "vision_module" / "backend" / "edge_detect" / "logs"))
    log_file: str = field(default_factory=lambda: str(_VISION_ROOT / "vision_module" / "backend" / "edge_detect" / "logs" / "online_edge.log"))
    runs_dir: str = field(default_factory=lambda: str(_VISION_ROOT / "vision_module" / "backend" / "edge_detect" / "runs"))
    pid_dir: str = field(default_factory=lambda: str(_VISION_ROOT / "vision_module" / "backend" / "edge_detect" / "pids"))
    pid_file: str = field(default_factory=lambda: str(_VISION_ROOT / "vision_module" / "backend" / "edge_detect" / "pids" / "online_edge.pid"))
    stack_run_id: str = ""
    loop_hz: float = 10.0
    preview: bool = True
    save_snapshot_period_s: float = 0.0
    snapshot_dir: str = field(default_factory=lambda: str(_VISION_ROOT / "vision_module" / "backend" / "edge_detect" / "snapshots"))
    log_mode: str = "concise"
    log_enabled: bool = True


@dataclass
class OutputConfig:
    transport: str = "disabled"
    ipc_socket_path: str = "/tmp/robot_stack/table_edge_obs.sock"
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
    calib_json: str = field(default_factory=lambda: str(_VISION_ROOT / "vision_module" / "backend" / "edge_detect" / "calib.json"))
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
    """Configuration structure representing the Online Edge Detector service."""
    runtime: OnlineEdgeRuntimeConfig = field(default_factory=OnlineEdgeRuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    camera: RealSenseConfig = field(default_factory=RealSenseConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)


# ==============================================================================
# Nested Top-Level System Config
# ==============================================================================

@dataclass
class SystemGlobalConfig:
    """The root configuration class representing the entire system."""
    profile: str = ""
    vision: VisionServiceConfig = field(default_factory=VisionServiceConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    gateway: MobileGatewayConfig = field(default_factory=MobileGatewayConfig)
    online_edge: OnlineEdgeConfig = field(default_factory=OnlineEdgeConfig)
