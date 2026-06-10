#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import msgpack


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


_PRESERVE_NONE_KEYS = {
    "yaw_err",
    "dist_err",
    "obs_ts",
    "age_ms",
    "frame_id",
    "seq",
}


def _compact(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            compacted = _compact(item)
            if compacted is None and key not in _PRESERVE_NONE_KEYS:
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


def canonical_vision_mode(mode: Optional[str]) -> Optional[str]:
    if not mode:
        return None
    mode_upper = str(mode).strip().upper()
    if mode_upper == "TRACK_LOCAL":
        return "FIND_OBJECT"
    if mode_upper in ("DEPTH_PERCEPTION", "TABLE_EDGE_PERCEPTION"):
        return "FIND_EDGE"
    return mode_upper


@dataclass
class VisionReq:
    ts: float
    op: str
    stage: str
    target: Optional[str] = None
    mode_hint: Optional[str] = None
    session_id: Optional[str] = None
    req_id: Optional[str] = None
    req_type: Optional[str] = None
    epoch: int = 0
    interaction_id: Optional[str] = None
    response: Optional[Dict[str, Any]] = None
    payload: Optional[Dict[str, Any]] = None
    legacy_type: Optional[str] = None
    type: str = "vision_req"

    def __post_init__(self) -> None:
        self.mode_hint = canonical_vision_mode(self.mode_hint)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VisionReq":
        stage = _canonical_stage(payload)
        mode_hint_raw = _opt_str(payload, "mode_hint")
        canonical_hint = canonical_vision_mode(mode_hint_raw)
        legacy_type = _opt_str(payload, "type") if _upper_text(payload.get("type", "VISION_REQ")) != "VISION_REQ" else None
        if mode_hint_raw and mode_hint_raw != canonical_hint and not legacy_type:
            legacy_type = f"mode_hint:{mode_hint_raw}"
        return cls(
            ts=float(payload.get("ts", now_ts())),
            op=_canonical_op(payload, stage),
            stage=stage,
            target=_opt_str(payload, "target"),
            mode_hint=canonical_hint,
            session_id=_opt_str(payload, "session_id"),
            req_id=_opt_str(payload, "req_id"),
            req_type=_opt_str(payload, "req_type"),
            epoch=_opt_int(payload, "epoch", 0),
            interaction_id=_opt_str(payload, "interaction_id"),
            response=_opt_dict(payload, "response"),
            payload=_opt_dict(payload, "payload"),
            legacy_type=legacy_type,
            type="vision_req",
        )


    def to_dict(self) -> Dict[str, Any]:
        return _compact(asdict(self))

    _DEFAULT_MODE_HINTS = {"SEARCH": "FIND_OBJECT", "GRASP": "SILENT", "RETURN": "SILENT"}

    def get_default_mode_hint(self) -> Optional[str]:
        """Protocol-compat fallback: return the well-known mode for this stage."""
        return self._DEFAULT_MODE_HINTS.get(self.stage)

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
    obs_class: str = "control"

    def to_dict(self) -> Dict[str, Any]:
        return _compact(asdict(self))


def pack_msg(payload: Dict[str, Any]) -> bytes:
    return msgpack.packb(payload, use_bin_type=True)


def unpack_msg(data: bytes) -> Dict[str, Any]:
    return msgpack.unpackb(data, raw=False)
