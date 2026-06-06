#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class RemoteProfile:
    """Capability requirements for remote inference or grasp cooperation."""

    enabled: bool = False
    base_url: Optional[str] = "192.168.6.43"    # required to be updated
    command: str = "predict"
    require_depth: bool = False
    kind: str = "loop"       # "loop" | "task"
    action: str = ""         # task only: "init" | "predict" | "release"
    max_retries: int = 1     # task only
    timeout_s: float = 10.0
    rgb_encoding: str = "jpeg"
    depth_encoding: str = "png"
    rgb_quality: int = 90
    depth_compression: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreviewProfile:
    """Preview sink settings associated with a mode profile."""

    enabled: bool = False
    sink_name: str = "null"
    overlay_enabled: bool = True
    window_name: str = "VISTA App Dashboard"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TableEdgeProfile:
    """Capability requirements for table-edge depth perception."""

    enabled: bool = False
    detector_mode: str = "lightweight"  # "lightweight" | "full" | "fast_plane_only"
    update_hz: float = 5.0
    light_stride: int = 4
    fast_plane_stride: int = 4
    require_yolo_confirm: bool = True
    static_roi_enabled: bool = False
    # fast_plane_only geometry
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
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModeProfile:
    """Resource-only definition for one runtime mode."""

    name: str
    enabled_cameras: Tuple[str, ...] = ()
    camera_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    predictor_enabled: bool = False
    predictor_model: Optional[str] = None
    remote: RemoteProfile = field(default_factory=RemoteProfile)
    preview: PreviewProfile = field(default_factory=PreviewProfile)
    table_edge: TableEdgeProfile = field(default_factory=TableEdgeProfile)
    loop_hz: Optional[float] = None
    send_hz: Optional[float] = None
    release_cooldown_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def camera_enabled(self, name: str) -> bool:
        """Check whether a camera should be active in this mode."""
        return str(name) in set(self.enabled_cameras)
