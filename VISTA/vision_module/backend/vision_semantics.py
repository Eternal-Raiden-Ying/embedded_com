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

from typing import Any, Dict, Optional, Sequence


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
    """Return canonical table-bbox fields for local_perception payloads.

    Confidence is reported but not used as a table validity gate.  The current
    design uses table-bbox existence as the high-level direction/ROI signal.
    """
    rgb_shape = rgb_shape if rgb_shape is not None else payload.get("rgb_shape")
    bbox = (
        payload.get("table_bbox_xyxy")
        or payload.get("detected_table_bbox")
        or payload.get("table_bbox")
        or payload.get("yolo_table_bbox")
    )
    bbox = _bbox_xyxy(bbox)
    found = bbox is not None
    metrics = bbox_metrics(bbox, rgb_shape)
    conf_raw = payload.get("table_bbox_conf_raw")
    if conf_raw is None:
        conf_raw = _bbox_conf(payload.get("detected_table_bbox") or payload.get("table_bbox") or bbox)
    source = "yolo_table_bbox" if found else str(payload.get("table_bbox_source") or payload.get("table_roi_source") or "none")
    return {
        "table_bbox_current_found": bool(found),
        "table_bbox_control_valid": bool(found),
        "table_bbox_xyxy": bbox,
        "table_bbox_source": source,
        "table_bbox_conf_raw": conf_raw,
        "table_bbox_conf_used_for_gate": False,
        "table_bbox_area_ratio": metrics.get("area_ratio"),
        "table_bbox_center": metrics.get("center"),
        "table_bbox_center_norm": metrics.get("center_norm"),
        # Compatibility aliases for legacy code and logs.
        "table_bbox_found": bool(found),
        "table_bbox_detected": bool(found),
        "yolo_table_control_valid": bool(found),
        "table_confirmed_by_yolo": bool(found),
        "yolo_valid_reason": "table_bbox_found" if found else "",
        "yolo_invalid_reason": "" if found else "table_bbox_unavailable",
        "docking_enabled_by_yolo": bool(found),
        "edge_control_allowed": bool(found),
        "edge_control_block_reason": "" if found else "table_bbox_unavailable",
    }


def edge_quality_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Collect fast/full edge quality features into a single dictionary."""
    return {
        "edge_conf": _to_float(payload.get("edge_conf", payload.get("confidence"))),
        "residual_mean": _to_float(payload.get("fast_residual_mean", payload.get("plane_residual_mean", payload.get("fast_edge_residual")))),
        "residual_p90": _to_float(payload.get("fast_residual_p90")),
        "residual_max": _to_float(payload.get("fast_residual_max", payload.get("plane_residual_max"))),
        "candidate_count": _to_int(payload.get("fast_candidate_point_count", payload.get("candidate_count"))) or 0,
        "support_count": _to_int(payload.get("fast_support_point_count", payload.get("support_point_count"))) or 0,
        "inlier_count": _to_int(payload.get("fast_rep_inlier_count", payload.get("fast_edge_inlier_count", payload.get("edge_inlier_count", payload.get("inlier_count"))))) or 0,
        "x_span_m": _to_float(payload.get("fast_fit_inlier_x_span_m", payload.get("fast_edge_x_span_m", payload.get("plane_x_span_m")))),
        "line_score": _to_float(payload.get("fast_line_score")),
        "frontness_score": _to_float(payload.get("fast_frontness_score")),
        "edge_consistency_score": _to_float(payload.get("fast_edge_consistency_score")),
        "background_penalty": _to_float(payload.get("fast_background_penalty")),
        "reject_reason": str(payload.get("reject_reason") or payload.get("reason") or ""),
    }


def standardize_table_edge_payload(
    payload: Dict[str, Any],
    *,
    edge_stable_required_frames: int = 5,
    edge_trusted_min_conf: float = 0.60,
    edge_trusted_max_residual: Optional[float] = None,
) -> Dict[str, Any]:
    """Add canonical table/edge semantic fields to a table_edge_obs payload.

    Important semantic split:
    - edge_detected: the algorithm found a geometric edge candidate.
    - edge_geometry_valid: single-frame geometric result exists and depth is available.
    - edge_stable: valid edge persisted for enough frames.
    - edge_trusted: stable edge is allowed to be consumed by control.
    """
    out = dict(payload or {})
    table_sem = local_table_bbox_semantics(out, rgb_shape=out.get("rgb_shape"))
    # Do not overwrite a more specific bbox supplied by table-edge ROI logic, but
    # do fill canonical fields and aliases consistently.
    if out.get("table_bbox_xyxy") is not None:
        table_sem = local_table_bbox_semantics({**out, "detected_table_bbox": out.get("table_bbox_xyxy")}, rgb_shape=out.get("rgb_shape"))
    out.update({k: v for k, v in table_sem.items() if k not in out or out.get(k) in (None, "")})
    # Ensure aliases are consistent even if legacy fields were already present.
    out["table_bbox_current_found"] = bool(out.get("table_bbox_current_found", out.get("table_bbox_found", False)))
    out["table_bbox_control_valid"] = bool(out.get("table_bbox_control_valid", out.get("table_bbox_current_found", False)))
    out["table_bbox_found"] = bool(out["table_bbox_current_found"])
    out["yolo_table_control_valid"] = bool(out["table_bbox_control_valid"])
    out["table_bbox_conf_used_for_gate"] = False

    unavailable = bool(out.get("edge_obs_unavailable", False))
    edge_detected = bool(out.get("edge_detected", out.get("edge_found", False)))
    edge_geometry_valid = bool(edge_detected and not unavailable)
    quality = edge_quality_from_payload(out)
    edge_conf = quality.get("edge_conf")
    residual = quality.get("residual_mean")
    stable_count = _to_int(out.get("edge_stable_count")) or 0
    stable_required = max(1, int(edge_stable_required_frames or 1))
    edge_stable = bool(edge_geometry_valid and stable_count >= stable_required)

    trust_reasons = []
    reject_reasons = []
    if not bool(out.get("table_bbox_control_valid", False)):
        reject_reasons.append("table_bbox_unavailable")
    if not edge_geometry_valid:
        reject_reasons.append("edge_geometry_invalid")
    if not edge_stable:
        reject_reasons.append(f"edge_not_stable:{stable_count}/{stable_required}")
    if edge_conf is not None and edge_conf < float(edge_trusted_min_conf):
        reject_reasons.append(f"edge_conf_low:{edge_conf:.3f}<{float(edge_trusted_min_conf):.3f}")
    if edge_trusted_max_residual is not None and residual is not None and residual > float(edge_trusted_max_residual):
        reject_reasons.append(f"edge_residual_high:{residual:.4f}>{float(edge_trusted_max_residual):.4f}")

    edge_trusted = bool(not reject_reasons)
    if edge_trusted:
        trust_reasons.append("table_bbox_and_stable_quality_edge")

    out.update(
        {
            "edge_detected": bool(edge_detected),
            "edge_geometry_valid": bool(edge_geometry_valid),
            "edge_stable": bool(edge_stable),
            "edge_trusted": bool(edge_trusted),
            "edge_quality": quality,
            "edge_trust_reason": ";".join(trust_reasons),
            "edge_reject_for_control_reason": ";".join(reject_reasons),
            "edge_stable_required_frames": int(stable_required),
            # Compatibility aliases.  `edge_valid` now means geometric validity;
            # `valid_for_control` is stricter and follows edge_trusted.
            "edge_valid": bool(edge_geometry_valid),
            "valid_for_control": bool(edge_trusted),
            "edge_control_allowed": bool(edge_trusted),
            "docking_enabled_by_yolo": bool(out.get("table_bbox_control_valid", False)),
            "edge_control_block_reason": "" if edge_trusted else ";".join(reject_reasons),
        }
    )
    return out
