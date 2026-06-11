#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Dict, Optional


def payload_has_table_edge_obs(payload: Optional[Dict[str, object]]) -> bool:
    return isinstance(payload, dict) and (
        isinstance(payload.get("table_edge_obs"), dict) or isinstance(payload.get("mock_table_edge_obs"), dict)
    )


def default_table_edge_obs() -> Dict[str, object]:
    return {
        "table_found": False,
        "edge_found": False,
        "edge_valid": False,
        "confidence": 0.0,
        "edge_conf": 0.0,
        "yaw_err_rad": None,
        "yaw_err": None,
        "dist_err_m": None,
        "dist_err": None,
        "edge_k": None,
        "edge_b": None,
        "depth_valid": False,
        "edge_obs_unavailable": True,
        "point_count": 0,
        "table_point_count": 0,
        "obs_ts": None,
        "age_ms": None,
        "frame_id": None,
        "seq": None,
        "source_mode": "",
        "is_stale": True,
        "source": "vision_table_edge_manager",
        "type": "table_edge_obs",
    }


def table_edge_obs_from_payload(payload: Optional[Dict[str, object]]) -> Dict[str, object]:
    base = default_table_edge_obs()
    source = None
    if isinstance(payload, dict):
        source = payload.get("table_edge_obs") or payload.get("mock_table_edge_obs")
    if isinstance(source, dict):
        base.update(source)
    return base


def table_edge_obs_from_results(results: Dict[str, object]) -> Optional[Dict[str, object]]:
    table_edge = (results or {}).get("table_edge_obs")
    if not isinstance(table_edge, dict):
        return None
    merged = default_table_edge_obs()
    merged.update(table_edge)
    merged["type"] = "table_edge_obs"
    if "edge_obs_unavailable" not in table_edge:
        merged["edge_obs_unavailable"] = str(merged.get("reason") or "") in {
            "depth_unavailable",
            "depth_frame_missing",
            "depth_frame_not_2d",
            "detector_unavailable",
        }
    if "is_stale" not in table_edge:
        merged["is_stale"] = False
    return merged


def table_edge_stale_ms() -> float:
    try:
        return max(0.0, float(os.getenv("VISTA_TABLE_EDGE_STALE_MS", "500") or 500.0))
    except Exception:
        return 500.0


def annotate_table_edge_obs(
    obs: Dict[str, object],
    *,
    tick_ts: float,
    source: str,
    source_mode: str,
) -> Dict[str, object]:
    out = dict(obs or default_table_edge_obs())
    out["type"] = "table_edge_obs"
    out["source_mode"] = str(source_mode or "").strip().upper()
    out["edge_conf"] = float(out.get("edge_conf", out.get("confidence", 0.0)) or 0.0)
    out["yaw_err"] = out.get("yaw_err", out.get("yaw_err_rad"))
    out["dist_err"] = out.get("dist_err", out.get("dist_err_m"))
    out.setdefault("seq", out.get("frame_seq", out.get("frame_id")))
    out.setdefault("frame_id", out.get("frame_seq", out.get("seq")))
    out["edge_obs_unavailable"] = bool(
        out.get("edge_obs_unavailable", False)
        or out.get("reason") in {"depth_unavailable", "depth_frame_missing", "depth_frame_not_2d", "detector_unavailable"}
    )
    out["edge_valid"] = bool(out.get("edge_found", False) and not out.get("edge_obs_unavailable", False))
    obs_ts = out.get("obs_ts", out.get("ts"))
    age_ms = None
    try:
        if obs_ts is not None:
            obs_ts_f = float(obs_ts)
            out["obs_ts"] = obs_ts_f
            age_ms = max(0.0, (float(tick_ts) - obs_ts_f) * 1000.0)
    except Exception:
        age_ms = None
    if age_ms is None:
        out.setdefault("obs_ts", None)
        out["age_ms"] = None
        stale = True
    else:
        upstream_age = out.get("age_ms")
        try:
            age_ms = max(float(age_ms), float(upstream_age))
        except Exception:
            pass
        out["age_ms"] = float(age_ms)
        stale = bool(age_ms > table_edge_stale_ms())
    if source != "results":
        stale = True
        out.setdefault("reason", "no_new_table_edge_obs_result")
    out["is_stale"] = bool(out.get("is_stale", False) or stale or out.get("edge_obs_unavailable", False))
    return out
