#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""One-line operator summary formatters.

Keep these helpers free of runtime side effects. They only shape small payload
snapshots into short console lines for field debugging.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def _upper(value: Any, default: str = "IDLE") -> str:
    text = str(value or default).strip().upper()
    return text or default


def _fmt_float(value: Any, default: float = 0.0, digits: int = 2, signed: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    sign = "+" if signed else ""
    return f"{number:{sign}.{digits}f}"


def _short(value: Any, default: str = "n/a", limit: int = 42) -> str:
    text = str(value if value not in (None, "") else default).strip() or default
    return text[:limit]


def format_table_edge_summary(
    status: Optional[Dict[str, Any]] = None,
    table_edge: Optional[Dict[str, Any]] = None,
    **overrides: Any,
) -> str:
    status = dict(status or {})
    edge = dict(table_edge or {})
    edge.update({k: v for k, v in overrides.items() if v is not None})
    found = bool(edge.get("edge_found", edge.get("edge", False)))
    conf = edge.get("confidence", edge.get("conf", 0.0))
    yaw = edge.get("yaw_err_rad", edge.get("yaw", 0.0))
    dist = edge.get("dist_err_m", edge.get("dist", 0.0))
    roi = edge.get("roi_source") or edge.get("depth_edge_roi") or edge.get("edge_roi") or "n/a"
    quadrant = edge.get("table_quadrant") or edge.get("quadrant") or edge.get("q") or "n/a"
    return (
        f"[VISTA] EDGE stage={_upper(status.get('stage') or edge.get('stage'))} "
        f"mode={_upper(status.get('mode') or edge.get('mode'))} "
        f"edge={int(found)} conf={_fmt_float(conf)} "
        f"yaw={_fmt_float(yaw, digits=3, signed=True)} dist={_fmt_float(dist, digits=3, signed=True)} "
        f"roi={_short(roi, limit=48)} q={_short(quadrant, limit=8)}"
    )


def format_target_summary(
    status: Optional[Dict[str, Any]] = None,
    target_obs: Optional[Dict[str, Any]] = None,
    **overrides: Any,
) -> str:
    status = dict(status or {})
    target = dict(target_obs or {})
    target.update({k: v for k, v in overrides.items() if v is not None})
    found = bool(target.get("found", False))
    cls = target.get("class_name") or target.get("cls") or target.get("target") or status.get("target") or "target"
    conf = target.get("confidence", target.get("conf", 0.0))
    full_center = target.get("matched_center_full_norm")
    if not isinstance(full_center, dict):
        full_center = target.get("matched_center") if isinstance(target.get("matched_center"), dict) else {}
    cx = full_center.get("cx", full_center.get("x_norm", target.get("x_norm", target.get("cx", 0.0))))
    cy = full_center.get("cy", full_center.get("y_norm", target.get("y_norm", target.get("cy_norm", target.get("cy", 0.0)))))
    return (
        f"[VISTA] TARGET stage={_upper(status.get('stage') or target.get('stage'))} "
        f"mode={_upper(status.get('mode') or target.get('mode'))} "
        f"found={int(found)} cls={_short(cls, limit=32)} conf={_fmt_float(conf)} "
        f"cx={_fmt_float(cx)} cy={_fmt_float(cy)}"
    )


def format_runtime_summary(
    status: Optional[Dict[str, Any]] = None,
    **overrides: Any,
) -> str:
    data = dict(status or {})
    data.update({k: v for k, v in overrides.items() if v is not None})
    parts = [
        f"[VISTA] RUNTIME stage={_upper(data.get('stage'))}",
        f"mode={_upper(data.get('mode'))}",
    ]
    if data.get("req_id"):
        parts.append(f"req={data.get('req_id')}")
    if data.get("epoch") is not None:
        parts.append(f"epoch={data.get('epoch')}")
    if data.get("reason"):
        parts.append(f"reason={_short(data.get('reason'), limit=48)}")
    return " ".join(parts)
