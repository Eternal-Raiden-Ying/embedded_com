#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical visual-semantics helpers for table docking.

This module deliberately separates *perception facts* from *control authority*.
The vision process should publish stable, human-readable fields such as
`table_bbox_current_found`, `edge_detected`, `edge_geometry_valid`,
`edge_stable`, `edge_trusted`, and `edge_quality`.  The orchestrator/control
layer can then consume those fields without guessing legacy names.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence


@dataclass
class TableEdgeObservation:
    """Canonical visual-semantics observation for table docking.

    This dataclass encapsulates perception facts and metrics computed by the
    vision process to allow control layers or state machines to access
    strictly-typed fields without guessing raw dictionary keys.

    Attributes:
        edge_detected: True if the algorithm detected a candidate geometric edge.
        edge_geometry_valid: True if a single-frame geometric result is valid in current ROI.
        edge_stable: True if the detection is stable over consecutive frames.
        edge_trusted: True if stable and passes all quality/confidence gates.
        edge_quality: Dictionary grouping raw geometric quality features.
        edge_trust_reason: Reason why the edge is trusted.
        edge_reject_for_control_reason: Semi-colon separated reasons why edge control is blocked.
        edge_stable_required_frames: Minimum frames of persistence required for stability.
        edge_trusted_min_conf: Minimum confidence threshold for trusting an edge.
        edge_trusted_min_support_count: Minimum support points for trusting an edge.
        edge_trusted_min_inlier_count: Minimum inlier points for trusting an edge.
        edge_trusted_min_x_span_m: Minimum x span in meters for trusting an edge.
        edge_valid: Legacy compatibility alias for edge_geometry_valid.
        valid_for_control: Legacy compatibility alias for edge_trusted.
        edge_control_allowed: Legacy compatibility alias for edge_trusted.
        edge_control_block_reason: Legacy compatibility alias for edge_reject_for_control_reason.

        table_bbox_current_found: True if table bounding box is detected in current frame.
        table_bbox_control_valid: True if control may use a bbox (includes hold/history).
        table_bbox_hold_active: True if the bounding box is currently held from history.
        table_bbox_hold_age_frames: Number of frames the bounding box has been held.
        table_bbox_xyxy: Normalized bbox coordinates [x1, y1, x2, y2].
        table_bbox_source: Source identifier of the bounding box (e.g., 'yolo_table_bbox').
        table_bbox_invalid_reason: Description of why the bounding box is invalid.
        table_bbox_conf_raw: Raw detection confidence from YOLO.
        table_bbox_conf_used_for_gate: Flag indicating if confidence is used for gating.
        table_bbox_area_ratio: Ratio of bounding box area relative to image shape.
        table_bbox_center: Center [cx, cy] of bounding box.
        table_bbox_center_norm: Normalized center [cx, cy] of bounding box.
        table_bbox_found: Legacy alias for table_bbox_current_found.
        table_bbox_detected: Legacy alias for table_bbox_current_found.
        yolo_table_control_valid: Legacy alias for table_bbox_control_valid.
        table_confirmed_by_yolo: Legacy alias for table_bbox_current_found.
        yolo_valid_reason: Legacy validity reason description.
        yolo_invalid_reason: Legacy invalid reason description.
        docking_enabled_by_yolo: Legacy alias for table_bbox_control_valid.

        dist_err_m: Lateral distance error in meters to the table edge.
        yaw_err_rad: Yaw alignment error in radians to the table edge.
        roi_source: Source ROI method (e.g., local_perception_table_bbox).
        table_quadrant: Quadrant of the table relative to the vehicle (e.g., LT, RT).
        fast_raw_dist_err_m: Raw lateral distance error in meters from fast fit.
        plane_dist_err_m: Lateral distance error from plane detection in meters.
        line_dist_err_m: Lateral distance error from line detection in meters.
        upper_line_dist_err_m: Lateral distance error from upper line detection in meters.
        lower_line_dist_err_m: Lateral distance error from lower line detection in meters.
        target_dist_m: Target docking distance in meters.
        dist_err: Alias for dist_err_m.

        extra_fields: Dynamic dictionary holding any other payload-specific fields
            not explicitly declared as properties, for full backward compatibility.
    """
    # 桌子边界检测与控制状态
    edge_detected: bool = False
    edge_geometry_valid: bool = False
    edge_stable: bool = False
    edge_trusted: bool = False
    edge_quality: dict[str, Any] = field(default_factory=dict)
    edge_trust_reason: str = ""
    edge_reject_for_control_reason: str = ""
    edge_stable_required_frames: int = 5
    edge_trusted_min_conf: float = 0.60
    edge_trusted_min_support_count: int = 0
    edge_trusted_min_inlier_count: int = 0
    edge_trusted_min_x_span_m: float = 0.0
    edge_valid: bool = False
    valid_for_control: bool = False
    edge_control_allowed: bool = False
    edge_control_block_reason: str = ""

    # 桌子 Bounding Box 相关的语义
    table_bbox_current_found: bool = False
    table_bbox_control_valid: bool = False
    table_bbox_hold_active: bool = False
    table_bbox_hold_age_frames: int = 0
    table_bbox_xyxy: Optional[list[float]] = None
    table_bbox_source: str = ""
    table_bbox_invalid_reason: str = ""
    table_bbox_conf_raw: Optional[float] = None
    table_bbox_conf_used_for_gate: bool = False
    table_bbox_area_ratio: Optional[float] = None
    table_bbox_center: Optional[list[float]] = None
    table_bbox_center_norm: Optional[list[float]] = None
    table_bbox_found: bool = False
    table_bbox_detected: bool = False
    yolo_table_control_valid: bool = False
    table_confirmed_by_yolo: bool = False
    yolo_valid_reason: str = ""
    yolo_invalid_reason: str = ""
    docking_enabled_by_yolo: bool = False

    # 测量误差/控制所需的目标观测值
    dist_err_m: Optional[float] = None
    yaw_err_rad: Optional[float] = None
    roi_source: Optional[str] = None
    table_quadrant: Optional[str] = None
    fast_raw_dist_err_m: Optional[float] = None
    plane_dist_err_m: Optional[float] = None
    line_dist_err_m: Optional[float] = None
    upper_line_dist_err_m: Optional[float] = None
    lower_line_dist_err_m: Optional[float] = None
    target_dist_m: Optional[float] = None
    dist_err: Optional[float] = None

    # Depth safety stop metrics
    depth_p10: Optional[float] = None
    close_depth_ratio: Optional[float] = None

    # 未明确声明的动态字段
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the observation to a dictionary for backward compatibility.

        Merges all strictly typed attributes and any extra fields stored in extra_fields.
        """
        out = {}
        for f in dataclasses.fields(self):
            if f.name == "extra_fields":
                continue
            out[f.name] = getattr(self, f.name)
        out.update(self.extra_fields)
        return out

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.extra_fields:
            return self.extra_fields[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.extra_fields[key] = value

    def __contains__(self, key: str) -> bool:
        if hasattr(self, key):
            return not key.startswith("_") and not callable(getattr(self, key))
        return key in self.extra_fields

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
            return default
        return self[key]

    def keys(self):
        fields = {f.name for f in dataclasses.fields(self) if f.name != "extra_fields"}
        return fields.union(self.extra_fields.keys())

    def values(self):
        return [self[k] for k in self.keys()]

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def update(self, other: Any) -> None:
        if isinstance(other, dict):
            for k, v in other.items():
                self[k] = v
        elif hasattr(other, "keys"):
            for k in other.keys():
                self[k] = other[k]
        else:
            for k, v in other:
                self[k] = v


def _to_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _shape_hw(shape: Any) -> Optional[tuple[int, int]]:
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        return None
    try:
        h = int(shape[0])
        w = int(shape[1])
    except Exception:
        return None
    if h <= 0 or w <= 0:
        return None
    return h, w


def _bbox_xyxy(value: Any) -> Optional[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value[:4]]
    except Exception:
        return None
    return [x1, y1, x2, y2]


def bbox_metrics(bbox: Any, shape: Any) -> Dict[str, Any]:
    xyxy = _bbox_xyxy(bbox)
    hw = _shape_hw(shape)
    if xyxy is None or hw is None:
        return {
            "area_ratio": None,
            "center": None,
            "center_norm": None,
            "width": None,
            "height": None,
        }
    h, w = hw
    x1, y1, x2, y2 = xyxy
    x1, x2 = sorted((max(0.0, min(float(w), x1)), max(0.0, min(float(w), x2))))
    y1, y2 = sorted((max(0.0, min(float(h), y1)), max(0.0, min(float(h), y2))))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return {
        "area_ratio": float((bw * bh) / max(1.0, float(w * h))),
        "center": [float(cx), float(cy)],
        "center_norm": [float(cx / max(1.0, float(w))), float(cy / max(1.0, float(h)))],
        "width": float(bw),
        "height": float(bh),
    }


def _bbox_conf(bbox: Any) -> Optional[float]:
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 5:
        return _to_float(bbox[4])
    return None


def local_table_bbox_semantics(payload: Dict[str, Any], *, rgb_shape: Any = None) -> Dict[str, Any]:
    """Return canonical table-bbox fields for local_perception/table-edge payloads.

    Field contract:
    - table_bbox_current_found: the current frame truly contains a table bbox.
    - table_bbox_control_valid: control may use a bbox; this can be true via hold/history.
    - table_bbox_hold_active: control_valid is maintained by history rather than current detection.

    Confidence is reported but never used as the table-bbox existence gate.
    """
    rgb_shape = rgb_shape if rgb_shape is not None else payload.get("rgb_shape")
    raw_bbox = (
        payload.get("table_bbox_xyxy")
        or payload.get("table_bbox")
        or payload.get("detected_table_bbox")
        or payload.get("yolo_table_bbox")
        or payload.get("mock_table_bbox")
    )
    bbox = _bbox_xyxy(raw_bbox)
    explicit_current_found = payload.get("table_bbox_current_found", None)
    if explicit_current_found is None:
        current_found = bbox is not None
    else:
        current_found = bool(explicit_current_found)

    explicit_control_valid = payload.get("table_bbox_control_valid", None)
    hold_active = bool(payload.get("table_bbox_hold_active", False))
    hold_age = _to_int(payload.get("table_bbox_hold_age_frames")) or 0
    if explicit_control_valid is None:
        control_valid = bool(current_found or hold_active or (bbox is not None and not current_found))
    else:
        control_valid = bool(explicit_control_valid)
    if current_found:
        hold_active = False
        hold_age = 0
    elif control_valid:
        hold_active = True

    metrics = bbox_metrics(bbox, rgb_shape)
    conf_raw = payload.get("table_bbox_conf_raw")
    if conf_raw is None:
        conf_raw = _bbox_conf(payload.get("detected_table_bbox") or payload.get("table_bbox") or payload.get("table_bbox_xyxy") or bbox)
    source = str(payload.get("table_bbox_source") or payload.get("table_roi_source") or "")
    if current_found:
        source = source if source and source not in {"fallback", "none", "yolo_unavailable"} else "yolo_table_bbox"
    elif control_valid:
        source = source or "table_bbox_hold"
    else:
        source = source or "none"

    invalid_reason = "" if control_valid else str(payload.get("table_bbox_invalid_reason") or payload.get("yolo_invalid_reason") or "table_bbox_unavailable")
    return {
        "table_bbox_current_found": bool(current_found),
        "table_bbox_control_valid": bool(control_valid),
        "table_bbox_hold_active": bool(hold_active),
        "table_bbox_hold_age_frames": int(hold_age),
        "table_bbox_xyxy": bbox,
        "table_bbox_source": source,
        "table_bbox_invalid_reason": invalid_reason,
        "table_bbox_conf_raw": conf_raw,
        "table_bbox_conf_used_for_gate": False,
        "table_bbox_area_ratio": metrics.get("area_ratio"),
        "table_bbox_center": metrics.get("center"),
        "table_bbox_center_norm": metrics.get("center_norm"),
        # Compatibility aliases for legacy code and logs. New code should use
        # the canonical fields above.
        "table_bbox_found": bool(current_found),
        "table_bbox_detected": bool(current_found),
        "yolo_table_control_valid": bool(control_valid),
        "table_confirmed_by_yolo": bool(current_found),
        "yolo_valid_reason": "table_bbox_current_found" if current_found else ("table_bbox_hold" if control_valid else ""),
        "yolo_invalid_reason": invalid_reason,
        "docking_enabled_by_yolo": bool(control_valid),
        # This means only that table bbox permits edge to be considered; final
        # edge_control_allowed is overwritten by standardize_table_edge_payload
        # and follows edge_trusted.
        "edge_control_allowed": False,
        "edge_control_block_reason": "" if control_valid else "table_bbox_unavailable",
    }


def edge_quality_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Collect fast/full edge quality features into a single dictionary.

    This dictionary is the only place where geometric quality fields are grouped.
    Control code should avoid reading scattered fast_* names directly.
    """
    residual_mean = payload.get("fast_residual_mean")
    if residual_mean is None:
        residual_mean = payload.get("line_residual")
    if residual_mean is None:
        residual_mean = payload.get("residual")
    if residual_mean is None:
        residual_mean = payload.get("plane_residual_mean")
    if residual_mean is None:
        residual_mean = payload.get("fast_edge_residual")

    inlier_count = payload.get("fast_rep_inlier_count")
    if inlier_count is None:
        inlier_count = payload.get("fast_edge_inlier_count")
    if inlier_count is None:
        inlier_count = payload.get("edge_inlier_count")
    if inlier_count is None:
        inlier_count = payload.get("inlier_count")

    support_count = payload.get("fast_support_point_count")
    if support_count is None:
        support_count = payload.get("support_point_count")
    if support_count is None:
        support_count = payload.get("support_count")

    x_span = payload.get("fast_fit_inlier_x_span_m")
    if x_span is None:
        x_span = payload.get("fast_edge_x_span_m")
    if x_span is None:
        x_span = payload.get("plane_x_span_m")

    return {
        "edge_conf": _to_float(payload.get("edge_conf", payload.get("confidence"))),
        "residual_mean": _to_float(residual_mean),
        "residual_p90": _to_float(payload.get("fast_residual_p90")),
        "residual_max": _to_float(payload.get("fast_residual_max", payload.get("plane_residual_max"))),
        "candidate_count": _to_int(payload.get("fast_candidate_point_count", payload.get("candidate_count"))) or 0,
        "support_count": _to_int(support_count) or 0,
        "inlier_count": _to_int(inlier_count) or 0,
        "x_span_m": _to_float(x_span),
        "line_score": _to_float(payload.get("fast_line_score")),
        "frontness_score": _to_float(payload.get("fast_frontness_score")),
        "edge_consistency_score": _to_float(payload.get("fast_edge_consistency_score")),
        "selected_cluster_score": _to_float(payload.get("fast_selected_cluster_score")),
        "background_penalty": _to_float(payload.get("fast_background_penalty")),
        "reject_reason": str(payload.get("reject_reason") or payload.get("reason") or ""),
    }


def standardize_table_edge_payload(
    payload: Dict[str, Any],
    *,
    edge_stable_required_frames: int = 5,
    edge_trusted_min_conf: float = 0.60,
    edge_trusted_max_residual: Optional[float] = None,
    edge_trusted_min_support_count: int = 0,
    edge_trusted_min_inlier_count: int = 0,
    edge_trusted_min_x_span_m: float = 0.0,
    edge_trusted_max_background_penalty: Optional[float] = None,
) -> TableEdgeObservation:
    """Add canonical table/edge semantic fields to a table_edge_obs payload.

    Important semantic split:
    - edge_detected: the algorithm found a candidate geometric edge.
    - edge_geometry_valid: a single-frame geometric result exists in the current ROI.
    - edge_stable: the result persisted for enough frames.
    - edge_trusted: edge_stable plus quality gates, allowed for posture control.
    """
    out = dict(payload or {})
    table_sem = local_table_bbox_semantics(out, rgb_shape=out.get("rgb_shape"))
    out.update(table_sem)

    unavailable = bool(out.get("edge_obs_unavailable", False))
    edge_detected = bool(out.get("edge_detected", out.get("edge_found", False)))
    if "edge_geometry_valid" in payload:
        edge_geometry_valid = bool(payload.get("edge_geometry_valid")) and not unavailable
    else:
        edge_geometry_valid = bool(edge_detected and not unavailable)

    quality = edge_quality_from_payload(out)
    edge_conf = quality.get("edge_conf")
    residual = quality.get("residual_mean")
    support_count = int(quality.get("support_count") or 0)
    inlier_count = int(quality.get("inlier_count") or 0)
    x_span_m = quality.get("x_span_m")
    background_penalty = quality.get("background_penalty")
    stable_count = _to_int(out.get("edge_stable_count")) or 0
    stable_required = max(1, int(edge_stable_required_frames or 1))
    edge_stable = bool(edge_geometry_valid and stable_count >= stable_required)

    reject_reasons = []
    if not bool(out.get("table_bbox_control_valid", False)):
        reject_reasons.append("table_bbox_unavailable")
    if not edge_detected:
        reject_reasons.append("edge_not_detected")
    if not edge_geometry_valid:
        reject_reasons.append("edge_geometry_invalid")
    if not edge_stable:
        reject_reasons.append(f"edge_not_stable:{stable_count}/{stable_required}")
    if edge_conf is not None and edge_conf < float(edge_trusted_min_conf):
        reject_reasons.append(f"edge_conf_low:{edge_conf:.3f}<{float(edge_trusted_min_conf):.3f}")
    if edge_trusted_max_residual is not None and residual is not None and residual > float(edge_trusted_max_residual):
        reject_reasons.append(f"edge_residual_high:{residual:.4f}>{float(edge_trusted_max_residual):.4f}")
    if int(edge_trusted_min_support_count or 0) > 0 and support_count < int(edge_trusted_min_support_count):
        reject_reasons.append(f"support_low:{support_count}<{int(edge_trusted_min_support_count)}")
    if int(edge_trusted_min_inlier_count or 0) > 0 and inlier_count < int(edge_trusted_min_inlier_count):
        reject_reasons.append(f"inlier_low:{inlier_count}<{int(edge_trusted_min_inlier_count)}")
    if float(edge_trusted_min_x_span_m or 0.0) > 0.0 and x_span_m is not None and float(x_span_m) < float(edge_trusted_min_x_span_m):
        reject_reasons.append(f"x_span_low:{float(x_span_m):.3f}<{float(edge_trusted_min_x_span_m):.3f}")
    if edge_trusted_max_background_penalty is not None and background_penalty is not None and float(background_penalty) > float(edge_trusted_max_background_penalty):
        reject_reasons.append(f"background_penalty_high:{float(background_penalty):.3f}>{float(edge_trusted_max_background_penalty):.3f}")

    edge_trusted = bool(not reject_reasons)
    edge_trust_reason = "table_bbox_and_stable_quality_edge" if edge_trusted else ""
    block_reason = "" if edge_trusted else ";".join(reject_reasons)

    out.update(
        {
            "edge_detected": bool(edge_detected),
            "edge_geometry_valid": bool(edge_geometry_valid),
            "edge_stable": bool(edge_stable),
            "edge_trusted": bool(edge_trusted),
            "edge_quality": quality,
            "edge_trust_reason": edge_trust_reason,
            "edge_reject_for_control_reason": block_reason,
            "edge_stable_required_frames": int(stable_required),
            "edge_trusted_min_conf": float(edge_trusted_min_conf),
            "edge_trusted_min_support_count": int(edge_trusted_min_support_count or 0),
            "edge_trusted_min_inlier_count": int(edge_trusted_min_inlier_count or 0),
            "edge_trusted_min_x_span_m": float(edge_trusted_min_x_span_m or 0.0),
            "edge_valid": bool(edge_geometry_valid),
            "valid_for_control": bool(edge_trusted),
            "edge_control_allowed": bool(edge_trusted),
            "docking_enabled_by_yolo": bool(out.get("table_bbox_control_valid", False)),
            "edge_control_block_reason": block_reason,
        }
    )

    field_names = {f.name for f in dataclasses.fields(TableEdgeObservation) if f.name != "extra_fields"}
    init_kwargs = {}
    extra_fields = {}
    for k, v in out.items():
        if k in field_names:
            init_kwargs[k] = v
        else:
            extra_fields[k] = v

    return TableEdgeObservation(extra_fields=extra_fields, **init_kwargs)
