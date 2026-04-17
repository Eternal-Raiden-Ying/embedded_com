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
            if compacted is None or compacted == {}:
                continue
            out[key] = compacted
        return out
    if isinstance(value, list):
        return [_compact(item) for item in value if _compact(item) is not None]
    return value


def _as_ts(payload: Dict[str, Any]) -> float:
    if payload.get("ts") is not None:
        return float(payload.get("ts"))
    if payload.get("ts_ms") is not None:
        return float(payload.get("ts_ms")) / 1000.0
    return now_ts()


def _opt_str(payload: Dict[str, Any], key: str) -> Optional[str]:
    val = payload.get(key)
    if val is None:
        return None
    text = str(val).strip()
    return text if text else None


@dataclass
class VisionReq:
    ts: float
    mode: str
    stage: str
    target: Optional[str] = None
    session_id: Optional[str] = None
    req_id: Optional[str] = None
    epoch: int = 0
    type: str = "vision_req"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VisionReq":
        mode = str(payload.get("mode", "IDLE")).strip().upper() or "IDLE"
        stage = str(payload.get("stage", mode)).strip().upper() or mode
        return cls(
            ts=_as_ts(payload),
            mode=mode,
            stage=stage,
            target=_opt_str(payload, "target"),
            session_id=_opt_str(payload, "session_id"),
            req_id=_opt_str(payload, "req_id"),
            epoch=int(payload.get("epoch", 0) or 0),
            type=str(payload.get("type", "vision_req") or "vision_req"),
        )


@dataclass
class TableEdgeObs:
    ts: float
    table_found: bool
    edge_found: bool
    confidence: float = 0.0
    table_cx_norm: float = 0.0
    table_size_norm: float = 0.0
    depth_valid: bool = False
    session_id: Optional[str] = None
    req_id: Optional[str] = None
    epoch: int = 0
    source: str = "vision_module_v2"
    type: str = "table_edge_obs"

    def to_dict(self) -> Dict[str, Any]:
        return _compact(asdict(self))


@dataclass
class TargetObs:
    ts: float
    found: bool
    target: Optional[str] = None
    confidence: float = 0.0
    cx_norm: float = 0.0
    size_norm: float = 0.0
    bbox: Optional[list] = None
    mask_ready: bool = False
    mask_shape: Optional[list] = None
    mask_area_ratio: Optional[float] = None
    mask_bbox: Optional[list] = None
    session_id: Optional[str] = None
    req_id: Optional[str] = None
    epoch: int = 0
    source: str = "vision_module_v2"
    type: str = "target_obs"

    def to_dict(self) -> Dict[str, Any]:
        return _compact(asdict(self))
