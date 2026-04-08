#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from dataclasses import asdict, dataclass
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


def _opt_dict(payload: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    val = payload.get(key)
    return val if isinstance(val, dict) else None


def _upper_text(value: Any, default: str = "") -> str:
    text = str(value or default).strip().upper()
    return text or str(default).strip().upper()


def _compact(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            compacted = _compact(item)
            if compacted is None:
                continue
            if compacted == {}:
                continue
            out[key] = compacted
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            compacted = _compact(item)
            if compacted is None:
                continue
            out.append(compacted)
        return out
    return value


def _canonical_stage(payload: Dict[str, Any]) -> str:
    stage = _upper_text(payload.get("stage", ""))
    if stage:
        return stage

    typ = _upper_text(payload.get("type", "vision_req"))
    mode = _upper_text(payload.get("mode", ""))
    if typ == "HOME_TAG_REQ":
        return "RETURN"
    if mode == "FIND":
        return "SEARCH"
    if mode == "RETURN":
        return "RETURN"
    if mode == "GRASP":
        return "GRASP"
    if mode in {"IDLE", "STOP", "CANCEL"}:
        return "IDLE"
    return "IDLE"


def _canonical_op(payload: Dict[str, Any], stage: str) -> str:
    op = _upper_text(payload.get("op", ""))
    if op:
        return op

    typ = _upper_text(payload.get("type", "vision_req"))
    mode = _upper_text(payload.get("mode", ""))
    if typ == "HOME_TAG_REQ":
        return "START"
    if mode in {"IDLE", "STOP", "CANCEL"} or stage == "IDLE":
        return "STOP"
    return "START"


@dataclass
class VisionReq:
    ts: float
    op: str
    stage: str
    target: Optional[str] = None
    mode_hint: Optional[str] = None
    session_id: Optional[str] = None
    req_id: Optional[str] = None
    epoch: int = 0
    interaction_id: Optional[str] = None
    response: Optional[Dict[str, Any]] = None
    payload: Optional[Dict[str, Any]] = None
    legacy_type: Optional[str] = None
    type: str = "vision_req"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VisionReq":
        stage = _canonical_stage(payload)
        return cls(
            ts=float(payload.get("ts", now_ts())),
            op=_canonical_op(payload, stage),
            stage=stage,
            target=_opt_str(payload, "target"),
            mode_hint=_upper_text(payload.get("mode_hint", "")) or None,
            session_id=_opt_str(payload, "session_id"),
            req_id=_opt_str(payload, "req_id"),
            epoch=_opt_int(payload, "epoch", 0),
            interaction_id=_opt_str(payload, "interaction_id"),
            response=_opt_dict(payload, "response"),
            payload=_opt_dict(payload, "payload"),
            legacy_type=_opt_str(payload, "type") if _upper_text(payload.get("type", "VISION_REQ")) != "VISION_REQ" else None,
            type="vision_req",
        )

    def to_dict(self) -> Dict[str, Any]:
        return _compact(asdict(self))

    def is_stop(self) -> bool:
        return self.op == "STOP" or self.stage == "IDLE"


@dataclass
class TargetObs:
    found: bool
    target: Optional[str] = None
    confidence: float = 0.0
    cx_norm: float = 0.0
    size_norm: float = 0.0
    track_id: Optional[int] = None
    bbox: Optional[list] = None
    center_priority: float = 0.0
    area_norm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return _compact(asdict(self))


@dataclass
class HomeTagObs:
    found: bool
    tag_id: Optional[int] = None
    confidence: float = 0.0
    cx_norm: float = 0.0
    area_norm: float = 0.0
    bbox: Optional[list] = None

    def to_dict(self) -> Dict[str, Any]:
        return _compact(asdict(self))


@dataclass
class VisionObs:
    ts: float
    stage: str
    mode: str
    status: str
    session_id: Optional[str] = None
    req_id: Optional[str] = None
    epoch: int = 0
    interaction: Optional[Dict[str, Any]] = None
    perception: Optional[Dict[str, Any]] = None
    proposal: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    type: str = "vision_obs"

    def to_dict(self) -> Dict[str, Any]:
        return _compact(asdict(self))
