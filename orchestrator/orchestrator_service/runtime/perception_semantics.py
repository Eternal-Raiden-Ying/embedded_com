#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Control-facing perception semantics for table docking.

This module is deliberately small and declarative.  It does not generate motion.
It normalizes raw TableEdgeObs fields into a clean contract consumed by the
control authority layer.

Naming rules used from control-refactor-v2 onward:

- table_bbox_current_found:
    Current frame has a table bbox from YOLO/table detector.
- table_bbox_control_valid:
    Control is allowed to treat table bbox as available.  This may later include
    hold / memory, but in the current implementation it is equivalent to current
    bbox existence unless upstream explicitly supplies a valid hold flag.
- edge_detected:
    Fast edge / docking perception saw a candidate geometric structure.
- edge_geometry_valid:
    Single-frame geometric result passed the perception-level validity checks.
    This is not control permission.
- edge_stable:
    edge_geometry_valid has persisted for enough frames.
- edge_trusted:
    edge_stable plus basic quality gates; only this may enter posture control.

Deprecated aliases such as edge_valid / yolo_reliable are intentionally not part
of this dataclass.  They may still be exported as compatibility logs by older
callers, but new control code should use the names below.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TablePerceptionSemantics:
    # YOLO/table bbox semantics
    table_bbox_current_found: bool = False
    table_bbox_control_valid: bool = False
    table_bbox_hold_active: bool = False
    table_bbox_hold_age_frames: int = 0
    table_bbox_xyxy: Optional[list] = None
    table_bbox_conf_raw: Optional[float] = None
    table_bbox_conf_used_for_gate: bool = False

    # Edge / docking perception semantics
    edge_detected: bool = False
    edge_geometry_valid: bool = False
    edge_stable: bool = False
    edge_trusted: bool = False
    edge_stable_count: int = 0
    edge_conf_raw: Optional[float] = None
    edge_residual_raw: Optional[float] = None
    edge_support_count: Optional[int] = None
    edge_inlier_count: Optional[int] = None
    edge_x_span_m: Optional[float] = None
    edge_background_penalty: Optional[float] = None
    edge_quality: Dict[str, Any] = field(default_factory=dict)
    edge_trust_reason: str = ""
    edge_reject_for_control_reason: str = ""

    # Timing / freshness semantics
    stale_level: str = ""
    stale_source: str = ""

    @property
    def table_bbox_found(self) -> bool:
        """Backward-compatible alias for legacy controller code.

        New code should use table_bbox_current_found for the current-frame
        detector result, or table_bbox_control_valid for the control-facing
        availability gate.  This alias intentionally maps to current-frame
        existence to avoid silently treating hold/memory as a fresh detection.
        """
        return bool(self.table_bbox_current_found)

    @property
    def yolo_table_control_valid(self) -> bool:
        """Backward-compatible alias for control-facing table bbox validity."""
        return bool(self.table_bbox_control_valid)

    @property
    def edge_valid(self) -> bool:
        """Backward-compatible alias for perception-level edge geometry validity."""
        return bool(self.edge_geometry_valid)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        # Compatibility fields are exported explicitly for older logs/tools.
        out["table_bbox_found"] = bool(self.table_bbox_current_found)
        out["yolo_table_control_valid"] = bool(self.table_bbox_control_valid)
        out["edge_valid"] = bool(self.edge_geometry_valid)
        return out


def _float_or_none(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        out = float(v)
    except Exception:
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _int_or_none(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def _list_bbox(v: Any) -> Optional[list]:
    if isinstance(v, list) and len(v) >= 4:
        return v[:4]
    if isinstance(v, tuple) and len(v) >= 4:
        return list(v[:4])
    return None


def table_bbox_current_found(obs: Any) -> bool:
    """Current-frame table bbox existence. Confidence is deliberately ignored."""
    if obs is None:
        return False
    explicit = getattr(obs, "table_bbox_current_found", None)
    bbox = (
        getattr(obs, "table_bbox_xyxy", None)
        or getattr(obs, "table_bbox", None)
        or getattr(obs, "yolo_table_bbox", None)
        or getattr(obs, "detected_table_bbox", None)
    )
    if explicit is not None:
        return bool(explicit) or _list_bbox(bbox) is not None
    if bool(getattr(obs, "table_bbox_found", False)):
        return True
    return _list_bbox(bbox) is not None


# Backward-compatible import name used by older controller code.
def table_bbox_found(obs: Any) -> bool:
    return table_bbox_current_found(obs)


def build_table_perception_semantics(
    obs: Any,
    cfg: Any = None,
    *,
    stale_level: str = "",
    stale_source: str = "",
) -> TablePerceptionSemantics:
    current_found = table_bbox_current_found(obs)

    bbox = None
    if obs is not None:
        for name in ("table_bbox_xyxy", "table_bbox", "yolo_table_bbox", "detected_table_bbox"):
            bbox = _list_bbox(getattr(obs, name, None))
            if bbox is not None:
                break

    # Control-valid is explicit if upstream provides it; otherwise current bbox
    # is the only gate. Confidence is not used.
    explicit_control_valid = bool(getattr(obs, "table_bbox_control_valid", False)) if obs is not None else False
    legacy_control_valid = bool(getattr(obs, "yolo_table_control_valid", False)) if obs is not None else False
    control_valid = bool(current_found or explicit_control_valid or legacy_control_valid)

    hold_active = bool(getattr(obs, "table_bbox_hold_active", False)) if obs is not None else False
    if not current_found and control_valid:
        hold_active = True
    hold_age = _int_or_none(getattr(obs, "table_bbox_hold_age_frames", None) if obs is not None else None)
    if hold_age is None:
        hold_age = 0

    edge_detected = bool(getattr(obs, "edge_detected", getattr(obs, "edge_found", False))) if obs is not None else False

    raw_geom = None
    if obs is not None:
        for name in ("edge_geometry_valid", "edge_valid", "valid_for_control", "usable_for_approach", "usable_for_alignment"):
            if hasattr(obs, name):
                raw_geom = bool(getattr(obs, name))
                if raw_geom:
                    break
    edge_geometry_valid = bool(control_valid and edge_detected and bool(raw_geom if raw_geom is not None else edge_detected))

    stable_count = 0
    if obs is not None:
        for name in ("edge_stable_count", "yolo_table_edge_stable_count", "stable_count"):
            val = _int_or_none(getattr(obs, name, None))
            if val is not None:
                stable_count = val
                break
    stable_required = int(
        getattr(cfg, "edge_trusted_stable_frames", getattr(cfg, "yolo_table_edge_stable_frames", 5))
        if cfg is not None else 5
    )
    stable_required = max(1, stable_required)
    edge_stable = bool(edge_geometry_valid and stable_count >= stable_required)

    raw_quality = getattr(obs, "edge_quality", None) if obs is not None else None
    edge_quality = dict(raw_quality or {}) if isinstance(raw_quality, dict) else {}

    def quality_or_attr(key: str, *attrs: str) -> Any:
        if key in edge_quality:
            return edge_quality.get(key)
        if obs is None:
            return None
        for attr in attrs:
            if hasattr(obs, attr):
                return getattr(obs, attr)
        return None

    conf = None
    for value in (quality_or_attr("edge_conf", "edge_conf", "confidence"), quality_or_attr("line_score", "fast_line_score"), quality_or_attr("edge_consistency_score", "fast_edge_consistency_score")):
        conf = _float_or_none(value)
        if conf is not None:
            break
    min_conf = _float_or_none(getattr(cfg, "edge_trusted_min_conf", None) if cfg is not None else None)
    if min_conf is None:
        min_conf = _float_or_none(getattr(cfg, "edge_follow_min_edge_conf", None) if cfg is not None else None)

    residual = None
    for value in (quality_or_attr("residual_mean", "line_residual", "residual", "fast_residual_mean"), quality_or_attr("residual_p90", "fast_residual_p90")):
        residual = _float_or_none(value)
        if residual is not None:
            break
    max_residual = _float_or_none(getattr(cfg, "edge_trusted_max_residual", None) if cfg is not None else None)

    support_count = _int_or_none(quality_or_attr("support_count", "fast_support_point_count", "support_point_count", "support_count"))
    inlier_count = _int_or_none(quality_or_attr("inlier_count", "fast_rep_inlier_count", "inlier_count", "fast_inlier_count"))
    x_span_m = _float_or_none(quality_or_attr("x_span_m", "fast_fit_inlier_x_span_m", "fast_edge_x_span_m"))
    background_penalty = _float_or_none(quality_or_attr("background_penalty", "fast_background_penalty"))
    min_support = _int_or_none(getattr(cfg, "edge_trusted_min_support_count", None) if cfg is not None else None) or 0
    min_inlier = _int_or_none(getattr(cfg, "edge_trusted_min_inlier_count", None) if cfg is not None else None) or 0
    min_x_span = _float_or_none(getattr(cfg, "edge_trusted_min_x_span_m", None) if cfg is not None else None) or 0.0
    max_background = _float_or_none(getattr(cfg, "edge_trusted_max_background_penalty", None) if cfg is not None else None)
    if max_background is not None and max_background <= 0.0:
        max_background = None

    if not control_valid:
        edge_trusted = False
        reason = ""
        reject = "table_bbox_unavailable"
    elif not edge_detected:
        edge_trusted = False
        reason = ""
        reject = "edge_not_detected"
    elif not edge_geometry_valid:
        edge_trusted = False
        reason = ""
        reject = "edge_geometry_invalid"
    elif not edge_stable:
        edge_trusted = False
        reason = ""
        reject = f"edge_not_stable:{stable_count}/{stable_required}"
    elif min_conf is not None and conf is not None and conf < float(min_conf):
        edge_trusted = False
        reason = ""
        reject = f"edge_conf_low:{conf:.3f}<{float(min_conf):.3f}"
    elif max_residual is not None and residual is not None and residual > float(max_residual):
        edge_trusted = False
        reason = ""
        reject = f"edge_residual_high:{residual:.3f}>{float(max_residual):.3f}"
    elif int(min_support) > 0 and (support_count is None or int(support_count) < int(min_support)):
        edge_trusted = False
        reason = ""
        reject = f"support_low:{int(support_count or 0)}<{int(min_support)}"
    elif int(min_inlier) > 0 and (inlier_count is None or int(inlier_count) < int(min_inlier)):
        edge_trusted = False
        reason = ""
        reject = f"inlier_low:{int(inlier_count or 0)}<{int(min_inlier)}"
    elif float(min_x_span) > 0.0 and x_span_m is not None and float(x_span_m) < float(min_x_span):
        edge_trusted = False
        reason = ""
        reject = f"x_span_low:{float(x_span_m):.3f}<{float(min_x_span):.3f}"
    elif max_background is not None and background_penalty is not None and float(background_penalty) > float(max_background):
        edge_trusted = False
        reason = ""
        reject = f"background_penalty_high:{float(background_penalty):.3f}>{float(max_background):.3f}"
    else:
        edge_trusted = True
        reason = "table_bbox_and_stable_quality_edge"
        reject = ""

    return TablePerceptionSemantics(
        table_bbox_current_found=bool(current_found),
        table_bbox_control_valid=bool(control_valid),
        table_bbox_hold_active=bool(hold_active),
        table_bbox_hold_age_frames=int(hold_age),
        table_bbox_xyxy=bbox,
        table_bbox_conf_raw=_float_or_none(getattr(obs, "table_bbox_conf_raw", getattr(obs, "yolo_table_conf", None)) if obs is not None else None),
        table_bbox_conf_used_for_gate=False,
        edge_detected=bool(edge_detected),
        edge_geometry_valid=bool(edge_geometry_valid),
        edge_stable=bool(edge_stable),
        edge_trusted=bool(edge_trusted),
        edge_stable_count=int(stable_count),
        edge_conf_raw=conf,
        edge_residual_raw=residual,
        edge_support_count=support_count,
        edge_inlier_count=inlier_count,
        edge_x_span_m=x_span_m,
        edge_background_penalty=background_penalty,
        edge_quality=edge_quality,
        edge_trust_reason=reason,
        edge_reject_for_control_reason=reject,
        stale_level=str(stale_level or ""),
        stale_source=str(stale_source or ""),
    )
