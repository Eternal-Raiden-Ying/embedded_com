#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


def now_ts() -> float:
    return time.time()


def _opt_str(payload: Dict[str, Any], key: str) -> Optional[str]:
    val = payload.get(key)
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


def _opt_int(payload: Dict[str, Any], key: str, default: int = 0) -> int:
    val = payload.get(key, default)
    try:
        return int(val)
    except Exception:
        return int(default)


@dataclass
class VisionReq:
    ts: float
    mode: str
    target: Optional[str] = None
    session_id: Optional[str] = None
    req_id: Optional[str] = None
    epoch: int = 0
    type: str = "vision_req"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VisionReq":
        return cls(
            ts=float(payload.get("ts", now_ts())),
            mode=str(payload.get("mode", "FIND")).strip().upper(),
            target=_opt_str(payload, "target"),
            session_id=_opt_str(payload, "session_id"),
            req_id=_opt_str(payload, "req_id"),
            epoch=_opt_int(payload, "epoch", 0),
            type=str(payload.get("type", "vision_req")),
        )


@dataclass
class HomeTagReq:
    ts: float
    mode: str = "RETURN"
    session_id: Optional[str] = None
    req_id: Optional[str] = None
    epoch: int = 0
    type: str = "home_tag_req"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "HomeTagReq":
        return cls(
            ts=float(payload.get("ts", now_ts())),
            mode=str(payload.get("mode", "RETURN")).strip().upper(),
            session_id=_opt_str(payload, "session_id"),
            req_id=_opt_str(payload, "req_id"),
            epoch=_opt_int(payload, "epoch", 0),
            type=str(payload.get("type", "home_tag_req")),
        )


@dataclass
class TargetObs:
    ts: float
    found: bool
    target: Optional[str] = None
    confidence: float = 0.0
    cx_norm: float = 0.0
    size_norm: float = 0.0
    track_id: Optional[int] = None
    bbox: Optional[list] = None
    center_priority: float = 0.0
    area_norm: float = 0.0
    session_id: Optional[str] = None
    req_id: Optional[str] = None
    epoch: int = 0
    type: str = "target_obs"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HomeTagObs:
    ts: float
    found: bool
    tag_id: Optional[int] = None
    confidence: float = 0.0
    cx_norm: float = 0.0
    area_norm: float = 0.0
    bbox: Optional[list] = None
    session_id: Optional[str] = None
    req_id: Optional[str] = None
    epoch: int = 0
    type: str = "home_tag_obs"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
