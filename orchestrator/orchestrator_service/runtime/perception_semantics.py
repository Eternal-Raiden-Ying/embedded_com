#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Perception semantic normalization for table docking control.

This module intentionally contains no motion generation.  It converts the raw
TableEdgeObs payload into stable control-facing semantic flags:

- table_bbox_found: YOLO/table detector found a table bbox.  Confidence is not
  used as a gate in the current strategy.
- edge_valid: a local geometric edge exists in the current ROI.
- edge_stable: edge_valid has been stable for enough consecutive observations.
- edge_trusted: edge information is allowed to participate in posture control.

The ROI correctness / RGB-depth mapping problem belongs to the vision/ROI layer.
The control layer only consumes this semantic contract.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TablePerceptionSemantics:
    table_bbox_found: bool = False
    table_bbox_xyxy: Optional[list] = None
    table_bbox_conf_raw: Optional[float] = None
    table_bbox_conf_used_for_gate: bool = False

    edge_detected: bool = False
    edge_valid: bool = False
    edge_stable: bool = False
    edge_trusted: bool = False

    edge_stable_count: int = 0
    edge_trust_reason: str = ""
    edge_reject_for_control_reason: str = ""

    stale_level: str = ""
    stale_source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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


def table_bbox_found(obs: Any) -> bool:
    """YOLO/table bbox existence gate.  Confidence is deliberately ignored."""
    if obs is None:
        return False
    if bool(getattr(obs, "table_bbox_found", False)):
        return True
    bbox = getattr(obs, "table_bbox_xyxy", None) or getattr(obs, "yolo_table_bbox", None)
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return True
    if bool(getattr(obs, "yolo_table_control_valid", False)):
        return True
    # Backward-compatible fallback for older vision payloads.
    return bool(getattr(obs, "table_confirmed_by_yolo", False)) and getattr(obs, "table_cx_norm", None) is not None


def build_table_perception_semantics(obs: Any, cfg: Any = None, *, stale_level: str = "", stale_source: str = "") -> TablePerceptionSemantics:
    bbox_found = table_bbox_found(obs)
    bbox = None
    bbox_val = getattr(obs, "table_bbox_xyxy", None) if obs is not None else None
    if isinstance(bbox_val, list):
        bbox = bbox_val
    elif isinstance(bbox_val, tuple):
        bbox = list(bbox_val)

    edge_detected = bool(getattr(obs, "edge_found", False)) if obs is not None else False
    raw_edge_valid = getattr(obs, "edge_valid", None) if obs is not None else None
    if raw_edge_valid is None:
        edge_valid = edge_detected
    else:
        edge_valid = bool(raw_edge_valid)
    # Geometry must have a table bbox context before it can affect control.
    edge_valid = bool(bbox_found and edge_detected and edge_valid)

    stable_count = int(getattr(obs, "yolo_table_edge_stable_count", 0) or 0) if obs is not None else 0
    stable_required = int(
        getattr(cfg, "edge_trusted_stable_frames", getattr(cfg, "yolo_table_edge_stable_frames", 5))
        if cfg is not None else 5
    )
    stable_required = max(1, stable_required)
    edge_stable = bool(edge_valid and stable_count >= stable_required)

    conf = _float_or_none(getattr(obs, "edge_conf", None) if obs is not None else None)
    if conf is None:
        conf = _float_or_none(getattr(obs, "confidence", None) if obs is not None else None)
    min_conf = _float_or_none(getattr(cfg, "edge_trusted_min_conf", None) if cfg is not None else None)
    if min_conf is None:
        min_conf = _float_or_none(getattr(cfg, "edge_follow_min_edge_conf", None) if cfg is not None else None)
    if min_conf is None:
        min_conf = 0.0

    residual = _float_or_none(getattr(obs, "line_residual", None) if obs is not None else None)
    if residual is None:
        residual = _float_or_none(getattr(obs, "residual", None) if obs is not None else None)
    max_residual = _float_or_none(getattr(cfg, "edge_trusted_max_residual", None) if cfg is not None else None)

    if not bbox_found:
        edge_trusted = False
        reason = ""
        reject = "table_bbox_unavailable"
    elif not edge_valid:
        edge_trusted = False
        reason = ""
        reject = "edge_invalid"
    elif not edge_stable:
        edge_trusted = False
        reason = ""
        reject = f"edge_not_stable:{stable_count}/{stable_required}"
    elif conf is not None and conf < float(min_conf):
        edge_trusted = False
        reason = ""
        reject = f"edge_conf_low:{conf:.3f}<{float(min_conf):.3f}"
    elif max_residual is not None and residual is not None and residual > max_residual:
        edge_trusted = False
        reason = ""
        reject = f"line_residual_high:{residual:.3f}>{max_residual:.3f}"
    else:
        edge_trusted = True
        reason = "table_bbox_and_stable_edge"
        reject = ""

    return TablePerceptionSemantics(
        table_bbox_found=bool(bbox_found),
        table_bbox_xyxy=bbox,
        table_bbox_conf_raw=_float_or_none(getattr(obs, "table_bbox_conf_raw", getattr(obs, "yolo_table_conf", None)) if obs is not None else None),
        table_bbox_conf_used_for_gate=False,
        edge_detected=bool(edge_detected),
        edge_valid=bool(edge_valid),
        edge_stable=bool(edge_stable),
        edge_trusted=bool(edge_trusted),
        edge_stable_count=int(stable_count),
        edge_trust_reason=reason,
        edge_reject_for_control_reason=reject,
        stale_level=str(stale_level or ""),
        stale_source=str(stale_source or ""),
    )
