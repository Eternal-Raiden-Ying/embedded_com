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

from common.target_catalog import model_class_names

# The model class order is sourced from the same catalog as Voice/VISTA/Orchestrator.
_FINETUNE_YOLO26S_BGR15 = model_class_names()

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


@dataclass
class VoiceKwsConfig:
    """Canonical Voice Gateway keyword-spotter configuration."""
    backend: str = "sherpa_onnx"
    model_dir: str = "Voice/kws/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
    keywords_file: str = "Voice/kws/sherpa_custom/extreme.txt"
    provider: str = "cpu"
    num_threads: int = 1
    max_active_paths: int = 4
    num_trailing_blanks: int = 1
    trigger_cooldown_ms: int = 800
    max_consecutive_errors: int = 3
    wake_keyword: str = "你好小车"
    stop_keyword: str = "停止小车"


@dataclass
class VoiceGatewayConfig:
    """Voice settings owned by the repository-wide configuration schema."""
    kws: VoiceKwsConfig = field(default_factory=VoiceKwsConfig)


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
    log_profile: str = "normal"
    resource_sample_interval_s: float = 1.0
    loop_hz: float = 12.0
    send_hz: float = 10.0
    track_local_send_hz: float = 10.0
    stale_req_s: float = 3.0
    hot_standby_s: float = 30.0
    keep_preview_after_stop: bool = True
    keep_vision_alive_after_task: bool = True
    keep_preview_alive_after_task: bool = True
    release_model_on_idle: bool = False
    keep_model_hot_in_standby: bool = True
    enable_infer_during_hot_standby: bool = False
    remote_payload_archive_enable: bool = True
    remote_payload_archive_max_keep: int = 20
    remote_rgb_correction_enable: bool = False
    remote_rgb_correction_mode: str = "none"
    remote_rgb_white_balance_enable: bool = True
    remote_rgb_exposure_target_mean: float = 90.0
    remote_rgb_max_gain: float = 4.0
    remote_rgb_gamma: float = 1.4
    remote_rgb_saturation_scale: float = 1.25
    remote_rgb_save_raw: bool = False
    remote_rgb_jpeg_quality: int = 95
    remote_rgb_capture_warmup_frames: int = 10
    remote_rgb_capture_wait_timeout_s: float = 2.0
    remote_rgb_min_luma_mean: float = 40.0
    remote_rgb_require_fresh_after_mode_enter: bool = True
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
    active_model: str = "yolo26s_detect_imgsz640"
    profiles: Dict[str, SingleModelConfig] = field(default_factory=lambda: {
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
        "yolo26s_detect_imgsz640": SingleModelConfig(
            target_model=str(
                _DEFAULT_MODEL_ROOT
                / "yolo26s"
                / "models"
                / "yolo26s-cutoff-bgr-imgsz640_2_qcs6490_w8a8.qnn236.ctx.bin"
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
        "yolo26s_detect_imgsz1280": SingleModelConfig(
            target_model=str(
                _DEFAULT_MODEL_ROOT
                / "yolo26s"
                / "models"
                / "yolo26s-cutoff-bgr-imgsz1280_qcs6490_w8a8.qnn236.ctx.bin"
            ),
            width=1280,
            height=1280,
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
    yolo_table_class_id: int = 1
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
    perception_sync_max_delta_ms: float = 100.0
    matched_roi_hold_ttl_ms: float = 200.0
    yolo_table_roi_hold_enable: bool = True
    final_roi_latch_enable: bool = True
    final_roi_latch_max_age_s: float = 2.0
    table_roi_depth_latched_min_valid_ratio: float = 0.03
    table_roi_depth_latched_min_sample_count: int = 32
    table_roi_depth_current_min_valid_ratio: float = 0.08
    table_roi_depth_current_min_sample_count: int = 64
    final_fixed_roi_enable: bool = True
    final_fixed_roi_x0_norm: float = 0.32
    final_fixed_roi_x1_norm: float = 0.68
    final_fixed_roi_y0_norm: float = 0.72
    final_fixed_roi_y1_norm: float = 0.88
    final_fixed_roi_min_valid_ratio: float = 0.03
    final_fixed_roi_min_sample_count: int = 32
    final_depth_debug_enable: bool = False
    yolo_table_roi_boundary_extend_enable: bool = True
    yolo_table_roi_boundary_margin_norm: float = 0.03
    boundary_extend_mode: str = "fov_aligned_bounded"
    extended_roi_scale_x: float = 1.25
    extended_roi_scale_y: float = 1.25
    extended_roi_lower_band_center_ratio: float = 0.75
    extended_roi_bottom_margin_px: int = 8
    extended_roi_max_width_px: int = 200
    extended_roi_max_height_px: int = 120
    extended_roi_max_area_px: int = 24000
    fallback_roi_lower_band_center_ratio: float = 0.75
    fallback_roi_width_px: int = 160
    fallback_roi_height_px: int = 90
    depth_margin_extension_enable: bool = True
    bbox_center_edge_band_x_ratio: float = 0.12
    bbox_center_edge_band_y_ratio: float = 0.12
    depth_margin_max_extend_left_px: int = 40
    depth_margin_max_extend_right_px: int = 40
    depth_margin_max_extend_bottom_px: int = 28
    depth_margin_max_extend_top_px: int = 0
    adaptive_sampling_enable: bool = True
    adaptive_target_sample_count: int = 300
    adaptive_min_stride: int = 4
    adaptive_max_stride: int = 16
    yolo_table_edge_stable_frames: int = 5
    edge_trusted_min_conf: float = 0.60
    edge_trusted_max_residual: float = 0.0
    edge_trusted_min_support_count: int = 0
    edge_trusted_min_inlier_count: int = 0
    edge_trusted_min_x_span_m: float = 0.0
    edge_trusted_max_background_penalty: float = 0.0
    yolo_table_near_dist_m: float = 0.45
    yolo_table_near_bottom_norm: float = 0.60
    table_target_dist_m: float = 0.30
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
    plane_fit_fast_path_enable: bool = True
    plane_fit_fast_accept_inlier_ratio: float = 0.75
    plane_fit_fast_accept_residual_scale: float = 1.0
    plane_fit_ransac_max_iterations: int = 20
    depth_stride: int = 2

    # Previously hardcoded configurations inside table_edge_manager.py business layer
    detector_mode: str = "fast_plane_only"
    update_hz: float = 5.0
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
    edge_sync_threshold_s: float = 0.25


@dataclass
class PreviewConfig:
    preview_mode: str = "light"
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
    light_depth_min_m: float = 0.20
    light_depth_max_m: float = 2.00
    light_output_width: int = 848
    light_output_height: int = 480


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
    log_profile: str = "normal"
    resource_sample_interval_s: float = 1.0
    log_mode: str = "concise"
    log_enabled: bool = True
    debug: bool = False
    state_block_period_s: float = 1.0
    heartbeat_period_s: float = 1.0
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
class ArmSerialConfig:
    enabled: bool = True
    dry_run: bool = False
    port: str = "/dev/ttyUSB0"
    baudrate: int = 9600
    timeout_s: float = 0.10
    open_settle_s: float = 3.0
    boot_drain_max_s: float = 5.0
    boot_quiet_s: float = 0.5
    bytesize: int = 8
    parity: str = "N"
    stopbits: int = 1
    rtscts: bool = False
    dsrdtr: bool = False
    set_dtr: bool = False
    set_rts: bool = False
    readback_enabled: bool = True
    response_timeout_s: float = 10.0


@dataclass
class MotionSmoothingConfig:
    enabled: bool = True
    bypass_on_safety_stop: bool = True
    vx_accel_mps2: float = 0.35
    vx_decel_mps2: float = 0.70
    vy_accel_mps2: float = 0.20
    vy_decel_mps2: float = 0.35
    wz_accel_radps2: float = 0.90
    wz_decel_radps2: float = 1.40
    urgent_wz_accel_radps2: float = 2.20
    urgent_wz_decel_radps2: float = 2.80
    dt_min_s: float = 0.02
    dt_max_s: float = 0.20
    reset_gap_s: float = 0.50


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
    target_search_absolute_timeout_s: float = 60.0
    target_lateral_no_progress_timeout_s: float = 10.0
    target_lateral_uart_no_progress_timeout_s: float = 3.0
    target_lateral_progress_min_delta: float = 0.01
    return_search_timeout_s: float = 15.0
    req_resend_period_s: float = 1.0
    stop_after_table_docking: bool = False

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
    table_yolo_align_center_x_target: float = 0.50
    table_yolo_align_center_x_tol: float = 0.08
    yolo_table_control_enable: bool = True
    yolo_table_conf_min: float = 0.25
    yolo_table_edge_stable_frames: int = 5
    edge_trusted_stable_frames: int = 3
    edge_trusted_min_conf: float = 0.60
    edge_trusted_max_residual: float = 0.05
    edge_trusted_min_support_count: int = 0
    edge_trusted_min_inlier_count: int = 0
    edge_trusted_min_x_span_m: float = 0.0
    edge_trusted_max_background_penalty: float = 0.0
    yolo_table_near_dist_m: float = 0.45
    near_slow_depth_m: float = 0.40
    near_stop_depth_m: float = 0.25
    near_slow_max_vx_mps: float = 0.030
    near_slow_max_vy_mps: float = 0.040
    near_slow_max_wz_radps: float = 0.04
    final_servo_enter_p10_m: float = 0.45
    final_enter_depth_threshold_m: float = 0.58
    final_enter_stable_count_required: int = 2
    final_handoff_on_yolo_lost_enable: bool = True
    final_handoff_recent_obs_max_age_s: float = 1.0
    final_handoff_min_recent_depth_m: float = 0.65
    final_fixed_roi_stop_threshold_m: float = 0.45
    final_fixed_roi_stop_stable_count_required: int = 3
    final_slow_probe_vx_mps: float = 0.05
    near_start_final_enable: bool = True
    near_start_final_depth_m: float = 0.58
    near_start_align_enable: bool = True
    near_start_align_timeout_s: float = 2.0
    remote_init_min_interval_s: float = 30.0
    remote_init_auto_enabled: bool = False
    edge_final_enter_margin_m: float = 0.06
    edge_final_stop_margin_m: float = 0.02
    close_range_enter_p10_m: float = 0.55
    final_probe_vx_mps: float = 0.020
    final_missing_probe_vx_mps: float = 0.010
    final_missing_probe_grace_s: float = 2.0
    final_missing_roi_continue_forward_enable: bool = True
    final_missing_roi_probe_vx_mps: float = 0.020
    final_slow_stop_timeout_s: float = 12.0
    final_missing_reuse_s: float = 0.50
    final_missing_probe_margin_m: float = 0.04
    close_range_probe_vx_mps: float = 0.015
    close_range_missing_probe_vx_mps: float = 0.008
    roi_final_stop_p10_m: float = 0.42
    roi_final_slow_p10_m: float = 0.52
    roi_final_probe_vx_mps: float = 0.020
    roi_final_missing_probe_vx_mps: float = 0.008
    roi_final_missing_hold_s: float = 0.8
    depth_envelope_stop_p10_m: float = 0.30
    depth_envelope_slow_p10_m: float = 0.50
    depth_emergency_stop_p10_m: float = 0.20
    depth_envelope_mid_p10_m: float = 0.70
    depth_envelope_slow_vx_mps: float = 0.012
    depth_envelope_mid_vx_mps: float = 0.015
    yolo_approach_far_vx_mps: float = 0.30
    yolo_approach_mid_vx_mps: float = 0.20
    yolo_approach_near_vx_mps: float = 0.10
    bbox_track_forward_enabled: bool = True
    min_forward_vx_mps: float = 0.040
    bbox_track_forward_center_band: float = 0.45
    bbox_track_forward_min_hold_ms: int = 800
    bbox_track_forward_max_wz_radps: float = 0.200
    edge_readiness_enabled: bool = True
    edge_readiness_enter_score: float = 0.65
    edge_readiness_exit_score: float = 0.35
    edge_readiness_rise: float = 0.15
    edge_readiness_decay: float = 0.10
    edge_readiness_min_inliers: int = 30
    edge_readiness_yaw_max_rad: float = 0.35
    edge_handoff_min_hold_ms: int = 0
    forward_commit_min_s: float = 1.8
    far_forward_commit_min_s: float = 2.0
    lateral_enabled: bool = True
    lateral_vy_max_mps: float = 0.750
    lateral_deadband_norm: float = 0.020
    lateral_kp: float = 0.300
    lateral_target_center_x_norm: float = 0.5
    lateral_owner_default: str = "none"
    distance_scaled_lateral_enabled: bool = True
    lateral_distance_ref_m: float = 0.50
    lateral_distance_scale_min: float = 0.80
    lateral_distance_scale_max: float = 2.00
    far_lateral_vy_max_mps: float = 0.750
    mid_lateral_vy_max_mps: float = 0.750
    near_lateral_vy_max_mps: float = 0.750
    lateral_priority_mid_error_norm: float = 0.99
    lateral_priority_large_error_norm: float = 0.99
    lateral_priority_mid_vx_cap_mps: float = 0.080
    lateral_priority_vx_cap_mps: float = 0.040
    edge_yaw_align_allow_lateral: bool = True
    edge_yaw_align_lateral_vy_max_mps: float = 0.750
    yaw_flip_hold_window_s: float = 0.80
    yaw_flip_count_limit: int = 2
    yaw_ambiguous_wz_cap: float = 0.0
    yaw_ambiguous_vy_boost: float = 1.5
    edge_yaw_control_enter_rad: float = 0.30
    edge_yaw_control_exit_rad: float = 0.12
    edge_yaw_reject_rad: float = 1.40
    edge_yaw_kp: float = 0.22
    edge_yaw_min_wz_radps: float = 0.08
    edge_yaw_max_wz_radps: float = 0.18
    final_dist_deadband_m: float = 0.030
    final_dist_kp: float = 0.080
    final_forward_vx_max_mps: float = 0.015
    final_reverse_vx_max_mps: float = 0.004
    final_reverse_confirm_frames: int = 3
    final_yaw_deadband_rad: float = 0.12
    final_lock_yaw_rad: float = 0.12
    final_yaw_realign_rad: float = 0.18
    final_yaw_stable_frames: int = 6
    final_yaw_align_min_duration_ms: int = 1000
    final_yaw_last_good_hold_s: float = 1.2
    final_lock_min_hold_ms: int = 800
    final_lock_lost_timeout_ms: int = 1000
    progress_window_ms: float = 15000.0
    min_progress_m: float = 0.010
    multi_table_enabled: bool = False
    yolo_table_lost_to_search_frames: int = 8
    no_table_bbox_timeout_s: float = 10.0
    edge_geometry_timeout_s: float = 10.0
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
    at_table_edge_settle_s: float = 0.10
    target_search_fast_start_enable: bool = True
    target_fast_start_confirm_enable: bool = False
    target_prewarm_max_age_ms: int = 180
    target_prewarm_stable_obs: int = 2
    final_to_lateral_max_vx_mps: float = 0.02
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
    target_lateral_align_enable: bool = True
    target_lateral_align_center_x_target: float = 0.40
    target_lateral_align_center_x_tol: float = 0.06
    # Legacy compatibility only; runtime derives deadband from 0.5 * target_lateral_align_center_x_tol.
    target_lateral_align_center_x_deadband: float = 0.03
    target_lateral_align_kp_vy: float = 0.10
    target_lateral_align_vy_min_mps: float = 0.016
    target_lateral_align_vy_max_mps: float = 0.750
    target_lateral_align_stable_frames: int = 3
    target_lateral_align_lost_hold_s: float = 0.80
    target_lateral_hold_enable: bool = True
    target_lateral_hold_s: float = 0.8
    target_lateral_lost_stop_s: float = 1.2
    target_lateral_min_vy_mps: float = 0.016
    target_lateral_align_timeout_s: float = 12.0
    post_grasp_place_enable: bool = True
    post_grasp_turn_enable: bool = True
    post_grasp_turn_wz_radps: float = 0.45
    post_grasp_turn_duration_s: float = 3.5
    post_grasp_turn_direction: str = "left"
    basket_search_timeout_s: float = 15.0
    basket_search_wz_radps: float = 0.25
    basket_align_center_x_target: float = 0.50
    basket_align_center_x_tol: float = 0.08
    basket_approach_vx_mps: float = 0.08
    basket_approach_vx_slow_mps: float = 0.035
    basket_stop_bbox_area_norm: float = 0.18
    basket_stop_bbox_height_norm: float = 0.38
    basket_stop_stable_count_required: int = 3
    basket_approach_timeout_s: float = 20.0
    place_pose_x_cm: float = 12.0
    place_pose_y_cm: float = 0.0
    place_pose_z_cm: float = 8.0
    place_pose_pitch_deg: float = 0.0
    place_pose_roll_deg: float = 0.0
    place_gripper_width: float = 80.0
    place_duration_ms: int = 800
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
    keep_vision_alive_after_task: bool = True
    task_done_shutdown_vision: bool = False
    enable_pick_pipeline: bool = False
    assume_grasp_success_for_test: bool = False
    target_gripper_widths: Dict[str, float] = field(default_factory=lambda: {
        "苹果": 50.0,
        "猕猴桃": 65.0,
        "瓶子": 70.0,
        "apple": 80.0,
        "bottle": 80.0,
    })


@dataclass
class CarMotionConfig:
    """PID, limits, and behavior variables for actual motion control."""
    grasp_reposition_speed_cm_s: float = 10.0
    grasp_pose_time_ms: int = 800
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
    table_controlled_vx_min_mps: float = 0.060
    table_controlled_vx_max_mps: float = 0.300
    table_controlled_vy_min_mps: float = 0.000
    table_controlled_vy_max_mps: float = 0.750
    table_controlled_wz_min_radps: float = 0.000
    table_controlled_wz_max_radps: float = 0.120
    table_approach_safe_vx_mps: float = 0.060
    table_approach_yaw_deadband_rad: float = 0.08
    table_approach_yaw_realign_rad: float = 0.16
    table_edge_hard_rotate_only_yaw_rad: float = 1.40
    table_edge_hard_yaw_rotate_only_frames: int = 3
    table_edge_hard_yaw_rotate_only_ms: int = 350
    table_perception_warmup_s: float = 1.0
    table_approach_allow_wz: bool = True
    table_approach_allow_vy: bool = True
    table_pose_missing_safe_vx_mps: float = 0.040
    table_pose_missing_max_hold_s: float = 3.0
    table_final_lock_vx_min_mps: float = 0.000
    table_final_lock_vx_max_mps: float = 0.015
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
    table_vx_kp_mps_per_m: float = 0.30
    table_yaw_slow_th_rad: float = 0.12
    table_yaw_stop_th_rad: float = 0.45
    table_near_dist_err_th_m: float = 0.10
    table_vy_max_mps: float = 0.750
    table_wz_view_max_radps: float = 0.05
    table_wz_plane_max_radps: float = 0.06
    table_dist_kp_mps_per_m: float = 0.12
    yolo_table_yaw_gain: float = 0.20
    yolo_table_max_wz_radps: float = 0.12
    yolo_table_forward_vx_mps: float = 0.015
    yolo_forward_center_hard_limit: float = 0.30
    yolo_forward_center_exit_limit: float = 0.33
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
    max_vx_mps: float = 0.5
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
    precise_dist_tol_m: float = 0.015  # Distance error stopping tolerance/threshold for precise approach
    precise_lateral_tol_m: float = 0.015
    precise_stable_s: float = 0.50

    coarse_max_wz_radps: float = 0.45
    approach_max_vx_mps: float = 0.30
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
class TtsFeedbackConfig:
    """Policy for structured Phone-TTS task feedback."""
    enabled: bool = True
    verbosity: str = "detailed"
    min_interval_s: float = 1.5
    default_ttl_s: float = 8.0
    dedupe_window_s: float = 30.0
    detailed_progress_enabled: bool = True


@dataclass
class OrchestratorConfig:
    """Configuration structure representing the Orchestrator service."""
    runtime: OrchestratorRuntimeConfig = field(default_factory=OrchestratorRuntimeConfig)
    serial: SerialConfig = field(default_factory=SerialConfig)
    arm_serial: ArmSerialConfig = field(default_factory=ArmSerialConfig)
    motion_smoothing: MotionSmoothingConfig = field(default_factory=MotionSmoothingConfig)
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
    tts_feedback: TtsFeedbackConfig = field(default_factory=TtsFeedbackConfig)
    tts_event_out: SocketEndpoint = field(default_factory=lambda: SocketEndpoint(
        transport="disabled", ipc_socket_path="/tmp/robot_stack/tts_event.sock", async_enabled=True,
    ))
    # Playback ACKs are forwarded transparently by Mobile Gateway when enabled.
    tts_playback_in: SocketEndpoint = field(default_factory=lambda: SocketEndpoint(
        transport="disabled", ipc_socket_path="/tmp/robot_stack/orchestrator_tts_playback.sock", async_enabled=True,
    ))
    frozen_targets: Dict[str, List[str]] = field(default_factory=dict)  # legacy input ignored; catalog is authoritative


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
    # Northbound mobile/voice ownership.  Legacy *_only aliases are accepted
    # by validation and normalized before being published to the mini-program.
    task_input_mode: str = "mobile"
    feedback_output_mode: str = "optional"
    mobile_task_commands_allowed: bool = True
    mobile_manual_control_allowed: bool = True
    mobile_core_control_allowed: bool = True
    mobile_emergency_stop_allowed: bool = True


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
    cmd: str = "robot/v1/SC171/mobile/cmd"
    ack: str = "robot/v1/SC171/mobile/ack"
    status: str = "robot/v1/SC171/mobile/status"
    heartbeat: str = "robot/v1/SC171/heartbeat"
    tts: str = "robot/v1/SC171/mobile/tts"
    tts_ack: str = "robot/v1/SC171/mobile/tts_ack"


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
    tts_qos: int = 1
    retain_status: bool = False
    retain_heartbeat: bool = False
    keepalive_s: int = 60
    connect_timeout_s: float = 5.0
    # Lets a playback-only gateway receive mini-program TTS acknowledgements
    # without exposing the normal task-command ingress.
    accept_commands: bool = True
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
    tts_event_in: GatewayEndpoint = field(default_factory=lambda: GatewayEndpoint(transport="disabled", ipc_socket_path="/tmp/robot_stack/mobile_tts_event.sock"))
    tts_playback_out: GatewayEndpoint = field(default_factory=lambda: GatewayEndpoint(transport="disabled", ipc_socket_path="/tmp/robot_stack/tts_playback.sock", send_mode="oneshot"))
    # Optional transparent tee to Orchestrator for task-level playback sequencing.
    orchestrator_tts_playback_out: GatewayEndpoint = field(default_factory=lambda: GatewayEndpoint(transport="disabled", ipc_socket_path="/tmp/robot_stack/orchestrator_tts_playback.sock", send_mode="oneshot"))


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
    z_max: float = 3.0
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
    voice_gateway: VoiceGatewayConfig = field(default_factory=VoiceGatewayConfig)
    online_edge: OnlineEdgeConfig = field(default_factory=OnlineEdgeConfig)
