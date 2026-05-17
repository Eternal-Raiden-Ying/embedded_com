#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


def now_ts() -> float:
    return time.time()


def _compact(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            compacted = _compact(item)
            if compacted is None:
                continue
            out[key] = compacted
        return out
    if isinstance(value, list):
        return [_compact(item) for item in value if item is not None]
    return value


@dataclass
class TableEdgeObsMsg:
    ts: float
    table_found: bool
    edge_found: bool
    confidence: float
    yaw_err_rad: Optional[float] = None
    dist_err_m: Optional[float] = None
    edge_k: Optional[float] = None
    edge_b: Optional[float] = None
    edge_valid: bool = False
    raw_found: bool = False
    pose_found: bool = False
    valid_for_control: bool = False
    pose_source: str = "none"
    plane_found: bool = False
    line_found: bool = False
    plane_confidence: float = 0.0
    line_confidence: float = 0.0
    plane_residual_mean: float = 0.0
    line_residual_mean: float = 0.0
    plane_x_span_m: float = 0.0
    line_x_span_m: float = 0.0
    candidate_count: int = 0
    inlier_count: int = 0
    stable_count: int = 0
    front_face_area_ratio: float = 0.0
    reject_reason: str = ""
    selected_line_type: str = "none"
    upper_line_found: bool = False
    upper_line_confidence: float = 0.0
    upper_line_candidate_count: int = 0
    upper_line_inlier_count: int = 0
    upper_line_residual_mean: float = 0.0
    upper_line_x_span_m: float = 0.0
    upper_line_y_norm_mean: float = 0.0
    upper_line_k: Optional[float] = None
    upper_line_b: Optional[float] = None
    upper_line_yaw_err_rad: Optional[float] = None
    upper_line_dist_err_m: Optional[float] = None
    lower_line_found: bool = False
    lower_line_confidence: float = 0.0
    lower_line_candidate_count: int = 0
    lower_line_inlier_count: int = 0
    lower_line_residual_mean: float = 0.0
    lower_line_x_span_m: float = 0.0
    lower_line_y_norm_mean: float = 0.0
    lower_line_k: Optional[float] = None
    lower_line_b: Optional[float] = None
    lower_line_yaw_err_rad: Optional[float] = None
    lower_line_dist_err_m: Optional[float] = None
    table_geometry_score: float = 0.0
    front_plane_score: float = 0.0
    line_score: float = 0.0
    plane_line_consistency_score: float = 0.0
    roi_boundary_score: float = 0.0
    temporal_score: float = 0.0
    geometry_reject_reason: str = ""
    usable_for_approach: bool = False
    usable_for_alignment: bool = False
    usable_for_stop: bool = False
    control_level: str = "none"
    control_reject_reason: str = ""
    selected_line_plane_boundary_dist: float = 0.0
    selected_line_plane_consistency: float = 0.0
    line_reject_reason: str = ""
    line_drift_rejected: bool = False
    object_like_line_score: float = 0.0
    final_pose_source: str = "none"
    depth_valid: bool = True
    point_count: int = 0
    table_point_count: int = 0
    frame_id: int = 0
    source: str = "online_edge_detect"
    type: str = "table_edge_obs"

    def to_dict(self) -> Dict[str, Any]:
        return _compact(asdict(self))
