#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strategy Pattern implementation for Robot Table-Docking Control State Gating."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


class TableDockingStrategy:
    """Encapsulates control state gating decisions for robot table docking.

    This class decouples alignment state decisions (e.g., stop_ready, align,
    approach_slow, rotate_only, stop, alignment, approach) from data gathering
    layers. Online docking uses the fast_plane_only detector path.
    """

    def __init__(
        self,
        # Distance stage boundaries (meters)
        near_dist_threshold: float = 0.25,
        middle_dist_threshold: float = 0.60,
        stop_dist_threshold: float = 0.12,
        
        # Heading error boundaries (radians)
        align_yaw_near: float = 0.30,
        align_yaw_middle: float = 0.40,
        align_yaw_far: float = 0.25,
        align_yaw_general_max: float = 0.45,
        rotate_only_yaw_min: float = 0.55,
        
        # Mode-specific confidence limits (usable_min, align_min)
        usable_min_near: float = 0.40,
        align_min_near: float = 0.52,
        usable_min_middle: float = 0.44,
        align_min_middle: float = 0.58,
        usable_min_far: float = 0.38,
        align_min_far: float = 0.66,
        
        # Structural / Geometry ratio thresholds
        min_width_ratio: float = 0.90,
        min_width_absolute: float = 0.06,
        min_span_ratio: float = 0.90,
        local_band_min_factor: int = 2,
        local_band_min_absolute: int = 20,
        
        # Noise / Stability filters
        temporal_jump_confidence_gate: float = 0.62,
        background_blocked_penalty_limit: float = 0.60,
        inconsistent_edge_inlier_limit: int = 4,
        inconsistent_edge_consistency_limit: float = 0.20,
        inconsistent_edge_penalty_limit: float = 0.30,
        
        # stop_ready safety downgrade bounds
        stop_ready_max_background_penalty: float = 0.0,
        stop_ready_min_local_band_span: float = 0.30,
        stop_ready_min_local_band_support_factor: int = 2,
        stop_ready_min_local_band_support_absolute: int = 40,
        stop_ready_min_edge_consistency: float = 0.25,
        stop_ready_min_edge_inlier_count: int = 4,
        
        # Fallback/Downgrade limits
        fallback_max_inliers_edge: int = 3,
        fallback_max_span_edge: float = 0.15,
        fallback_max_inliers_partial: int = 7,
        fallback_max_span_partial: float = 0.30,
    ):
        # Boundaries & Thresholds
        self.near_dist_threshold = near_dist_threshold
        self.middle_dist_threshold = middle_dist_threshold
        self.stop_dist_threshold = stop_dist_threshold
        
        self.align_yaw_near = align_yaw_near
        self.align_yaw_middle = align_yaw_middle
        self.align_yaw_far = align_yaw_far
        self.align_yaw_general_max = align_yaw_general_max
        self.rotate_only_yaw_min = rotate_only_yaw_min
        
        self.usable_min_near = usable_min_near
        self.align_min_near = align_min_near
        self.usable_min_middle = usable_min_middle
        self.align_min_middle = align_min_middle
        self.usable_min_far = usable_min_far
        self.align_min_far = align_min_far
        
        self.min_width_ratio = min_width_ratio
        self.min_width_absolute = min_width_absolute
        self.min_span_ratio = min_span_ratio
        self.local_band_min_factor = local_band_min_factor
        self.local_band_min_absolute = local_band_min_absolute
        
        self.temporal_jump_confidence_gate = temporal_jump_confidence_gate
        self.background_blocked_penalty_limit = background_blocked_penalty_limit
        self.inconsistent_edge_inlier_limit = inconsistent_edge_inlier_limit
        self.inconsistent_edge_consistency_limit = inconsistent_edge_consistency_limit
        self.inconsistent_edge_penalty_limit = inconsistent_edge_penalty_limit
        
        self.stop_ready_max_background_penalty = stop_ready_max_background_penalty
        self.stop_ready_min_local_band_span = stop_ready_min_local_band_span
        self.stop_ready_min_local_band_support_factor = stop_ready_min_local_band_support_factor
        self.stop_ready_min_local_band_support_absolute = stop_ready_min_local_band_support_absolute
        self.stop_ready_min_edge_consistency = stop_ready_min_edge_consistency
        self.stop_ready_min_edge_inlier_count = stop_ready_min_edge_inlier_count
        
        self.fallback_max_inliers_edge = fallback_max_inliers_edge
        self.fallback_max_span_edge = fallback_max_span_edge
        self.fallback_max_inliers_partial = fallback_max_inliers_partial
        self.fallback_max_span_partial = fallback_max_span_partial

    def evaluate_control_level(
        self,
        mode: str,
        dist_err_m: float,
        yaw_err_rad: float,
        confidence: float,
        x_span: float,
        **kwargs: Any,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Evaluate and determine control level and rejection reasons.

        Args:
            mode: 'fast' (fast_plane_only).
            dist_err_m: Lateral/Forward distance error in meters.
            yaw_err_rad: Heading error in radians.
            confidence: Evaluated confidence score.
            x_span: Computed line segment span in meters.
            **kwargs: Extra parameters depending on mode (e.g. rep_count, geometry_score).

        Returns:
            A tuple of:
                - control_level: 'stop_ready', 'align', 'approach_slow', 'rotate_only', 'none'.
                - reject_reason: Description of the rejection (empty if accepted).
                - extras: Mode-specific extra output parameters.
        """
        if mode == "fast":
            return self._evaluate_fast(dist_err_m, yaw_err_rad, confidence, x_span, **kwargs)
        raise ValueError(f"Unsupported docking strategy mode: {mode}")

    def _evaluate_fast(
        self,
        dist_err_m: float,
        yaw_err_rad: float,
        confidence: float,
        x_span: float,
        representative_inlier_count: int,
        support_inlier_count: int,
        rep_count: int,
        residual_mean: float,
        residual_threshold: float,
        max_yaw: float,
        min_front_face_columns: int,
        min_vertical_support: int,
        min_front_face_x_span: float,
        raw_width_norm: Optional[float] = None,
        local_band_support_count: int = 0,
        local_band_x_span: float = 0.0,
        local_band_edge_support: int = 0,
        edge_cue_inlier_count: int = 0,
        edge_consistency_score: float = 0.0,
        background_penalty: float = 0.0,
        background_blocked: bool = False,
        near_stage_far_jump: bool = False,
        selected_cluster_index: int = 0,
        selected_cluster_support: int = 0,
        temporal_jump: bool = False,
        line_source: str = "vertical",
        **kwargs: Any,
    ) -> Tuple[str, str, Dict[str, Any]]:
        yaw_abs = abs(yaw_err_rad)
        
        # Dynamic configurations
        width_min = max(self.min_width_absolute, float(min_front_face_x_span) * self.min_width_ratio)
        hard_span_min = max(float(min_front_face_x_span), float(min_front_face_x_span) * self.min_span_ratio)
        local_band_min = max(self.local_band_min_absolute, min_front_face_columns * min_vertical_support * self.local_band_min_factor)

        # Distance stage categorization
        if dist_err_m > self.middle_dist_threshold:
            distance_stage = "far"
        elif dist_err_m > self.near_dist_threshold:
            distance_stage = "middle"
        else:
            distance_stage = "near"

        reject_reason = ""
        control_level = "none"

        # Sequential gates checks
        if near_stage_far_jump:
            reject_reason = "near_stage_far_jump"
        elif background_blocked and background_penalty >= self.background_blocked_penalty_limit:
            reject_reason = "far_background_selected_blocked"
        elif selected_cluster_index > 0 and distance_stage == "near":
            reject_reason = "background_only"
        elif representative_inlier_count < min_front_face_columns:
            reject_reason = "vertical_support_low"
        elif support_inlier_count < min_front_face_columns * min_vertical_support:
            reject_reason = "vertical_support_low"
        elif rep_count < min_front_face_columns:
            reject_reason = "front_face_columns_low"
        elif residual_mean > residual_threshold:
            reject_reason = "residual_too_large"
        elif yaw_abs > max_yaw:
            reject_reason = "yaw_out_of_range"
        elif float(raw_width_norm or 0.0) < width_min:
            reject_reason = "width_too_small"
        elif x_span < hard_span_min:
            reject_reason = "front_face_x_span_low"
        elif local_band_support_count < local_band_min and local_band_edge_support < 3:
            reject_reason = "front_line_weak"
        elif edge_cue_inlier_count >= self.inconsistent_edge_inlier_limit and edge_consistency_score < self.inconsistent_edge_consistency_limit and background_penalty > self.inconsistent_edge_penalty_limit:
            reject_reason = "edge_inconsistent"
        elif temporal_jump and confidence < self.temporal_jump_confidence_gate:
            reject_reason = "temporal_jump"
        else:
            # Stage based thresholds
            if distance_stage == "near":
                usable_min = self.usable_min_near
                align_min = self.align_min_near
            elif distance_stage == "middle":
                usable_min = self.usable_min_middle
                align_min = self.align_min_middle
            else:
                usable_min = self.usable_min_far
                align_min = self.align_min_far

            # Control level decision logic
            if confidence < usable_min:
                reject_reason = "confidence_too_low"
            elif yaw_abs >= self.rotate_only_yaw_min:
                control_level = "rotate_only"
            elif distance_stage == "near":
                if abs(dist_err_m) <= self.stop_dist_threshold and yaw_abs < self.align_yaw_near and confidence >= align_min:
                    control_level = "stop_ready"
                elif yaw_abs < self.align_yaw_general_max and confidence >= align_min:
                    control_level = "align"
                else:
                    control_level = "approach_slow"
            elif distance_stage == "middle":
                middle_span_ok = x_span >= max(float(min_front_face_x_span) * 1.35, 0.28) and float(raw_width_norm or 0.0) >= 0.28
                if yaw_abs < self.align_yaw_middle and confidence >= align_min and middle_span_ok:
                    control_level = "align"
                elif yaw_abs >= self.align_yaw_general_max:
                    control_level = "rotate_only"
                else:
                    control_level = "approach_slow"
            else:
                if yaw_abs < self.align_yaw_far and confidence >= align_min:
                    control_level = "align"
                elif yaw_abs >= self.align_yaw_general_max:
                    control_level = "rotate_only"
                else:
                    control_level = "approach_slow"

        # Post-gating refinements
        if not reject_reason:
            if yaw_abs > max_yaw:
                reject_reason = "yaw_out_of_range"
                control_level = "none"
            elif background_blocked:
                reject_reason = "far_background_selected_blocked"
                control_level = "none"
            elif representative_inlier_count <= self.fallback_max_inliers_edge or x_span < self.fallback_max_span_edge:
                if control_level in {"stop_ready", "align"}:
                    control_level = "approach_slow"
            elif (self.fallback_max_inliers_edge < representative_inlier_count <= self.fallback_max_inliers_partial) or (self.fallback_max_span_edge <= x_span < self.fallback_max_span_partial):
                if control_level == "stop_ready":
                    control_level = "align"
            
            # stop_ready alignment downgrades
            if control_level == "stop_ready" and (
                line_source == "vertical"
                and edge_consistency_score < self.stop_ready_min_edge_consistency
                and edge_cue_inlier_count >= self.stop_ready_min_edge_inlier_count
            ):
                control_level = "align"
            
            if control_level == "stop_ready" and (
                local_band_support_count < max(self.stop_ready_min_local_band_support_absolute, local_band_min * self.stop_ready_min_local_band_support_factor)
                or local_band_x_span < self.stop_ready_min_local_band_span
                or background_penalty > self.stop_ready_max_background_penalty
            ):
                control_level = "align"

        return control_level, reject_reason, {"distance_stage": distance_stage}
