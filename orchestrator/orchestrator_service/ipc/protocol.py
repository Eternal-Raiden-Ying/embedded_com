#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Set

ALLOWED_INTENTS: Set[str] = {"FIND", "RETURN", "STOP"}


class ProtocolError(ValueError):
    pass


def now_ts() -> float:
    return time.time()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _pick_optional_float(payload: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if payload.get(key) is None:
            continue
        try:
            return float(payload.get(key))
        except Exception:
            continue
    return None


@dataclass
class TaskCmd:
    ts: float
    intent: str
    confidence: float
    target: Optional[str] = None
    cmd_id: str = ""
    session_id: str = ""
    epoch: int = 0
    source: str = "voice"
    type: str = "task_cmd"
    text: Optional[str] = None
    raw_text: Optional[str] = None
    high_priority: bool = False
    state: Optional[str] = None
    wake_score: Optional[float] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], frozen_targets: Set[str]) -> "TaskCmd":
        intent = str(payload.get("intent", "")).upper().strip()
        if intent not in ALLOWED_INTENTS:
            raise ProtocolError(f"非法 intent: {intent!r}")
        confidence = float(payload.get("confidence", 0.0))
        target = payload.get("target")
        if intent == "FIND":
            target = str(target or "").strip()
            if not target:
                raise ProtocolError("FIND 缺少 target")
            if target not in frozen_targets:
                raise ProtocolError(f"target 不在冻结词表中: {target}")
        else:
            target = None
        return cls(
            ts=float(payload.get("ts", now_ts())),
            intent=intent,
            confidence=confidence,
            target=target,
            cmd_id=str(payload.get("cmd_id") or _new_id("cmd")),
            session_id=str(payload.get("session_id") or _new_id("sess")),
            epoch=int(payload.get("epoch", 0) or 0),
            source=str(payload.get("source", "voice") or "voice"),
            type=str(payload.get("type", "task_cmd") or "task_cmd"),
            text=(str(payload.get("text")).strip() if payload.get("text") is not None else None),
            raw_text=(str(payload.get("raw_text")).strip() if payload.get("raw_text") is not None else None),
            high_priority=bool(payload.get("high_priority", False)),
            state=(str(payload.get("state")).strip() if payload.get("state") is not None else None),
            wake_score=(float(payload["wake_score"]) if payload.get("wake_score") is not None else None),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None and v != ""}


@dataclass
class TaskAck:
    ts: float
    cmd_id: str
    accepted: bool
    state: str
    session_id: str = ""
    epoch: int = 0
    reason: str = ""
    source: str = "orchestrator"
    type: str = "task_ack"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, "")}


@dataclass
class TargetObs:
    ts: float
    found: bool
    target: Optional[str] = None
    confidence: Optional[float] = None
    cx_norm: float = 0.0
    size_norm: float = 0.0
    track_id: Optional[int] = None
    bbox: Optional[list] = None
    req_id: Optional[str] = None
    session_id: Optional[str] = None
    epoch: int = 0
    vx_norm: Optional[float] = None
    wz_norm: Optional[float] = None
    type: str = "target_obs"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TargetObs":
        return cls(
            ts=float(payload.get("ts", now_ts())),
            found=bool(payload.get("found", False)),
            target=(str(payload.get("target")).strip() if payload.get("target") is not None else None),
            confidence=(float(payload["confidence"]) if payload.get("confidence") is not None else None),
            cx_norm=float(payload.get("cx_norm", 0.0)),
            size_norm=float(payload.get("size_norm", 0.0)),
            track_id=(int(payload["track_id"]) if payload.get("track_id") is not None else None),
            bbox=payload.get("bbox"),
            req_id=(str(payload.get("req_id")).strip() if payload.get("req_id") is not None else None),
            session_id=(str(payload.get("session_id")).strip() if payload.get("session_id") is not None else None),
            epoch=int(payload.get("epoch", 0) or 0),
            vx_norm=_pick_optional_float(payload, "vx_norm", "vx", "v_norm", "linear_norm"),
            wz_norm=_pick_optional_float(payload, "wz_norm", "wz", "omega_norm", "angular_norm"),
            type=str(payload.get("type", "target_obs") or "target_obs"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class HomeTagObs:
    ts: float
    found: bool
    yaw_err_rad: float = 0.0
    distance_m: Optional[float] = None
    req_id: Optional[str] = None
    session_id: Optional[str] = None
    epoch: int = 0
    vx_norm: Optional[float] = None
    wz_norm: Optional[float] = None
    type: str = "home_tag_obs"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "HomeTagObs":
        return cls(
            ts=float(payload.get("ts", now_ts())),
            found=bool(payload.get("found", False)),
            yaw_err_rad=float(payload.get("yaw_err_rad", 0.0)),
            distance_m=(float(payload["distance_m"]) if payload.get("distance_m") is not None else None),
            req_id=(str(payload.get("req_id")).strip() if payload.get("req_id") is not None else None),
            session_id=(str(payload.get("session_id")).strip() if payload.get("session_id") is not None else None),
            epoch=int(payload.get("epoch", 0) or 0),
            vx_norm=_pick_optional_float(payload, "vx_norm", "vx", "v_norm", "linear_norm"),
            wz_norm=_pick_optional_float(payload, "wz_norm", "wz", "omega_norm", "angular_norm"),
            type=str(payload.get("type", "home_tag_obs") or "home_tag_obs"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class CarState:
    ts: float
    state: str = "UNKNOWN"
    ok: bool = False
    timeout: bool = False
    estop: bool = False
    fault: bool = False
    mode: Optional[str] = None
    message: Optional[str] = None
    raw: Optional[str] = None
    source: str = "uart"
    type: str = "car_state"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CarState":
        state = str(payload.get("state", payload.get("status", "UNKNOWN"))).strip().upper() or "UNKNOWN"
        message = payload.get("message")
        return cls(
            ts=float(payload.get("ts", now_ts())),
            state=state,
            ok=bool(payload.get("ok", state == "OK")),
            timeout=bool(payload.get("timeout", state == "TIMEOUT")),
            estop=bool(payload.get("estop", state in {"ESTOP", "E_STOP"})),
            fault=bool(payload.get("fault", state in {"FAULT", "ERROR"})),
            mode=(str(payload.get("mode")).strip().upper() if payload.get("mode") is not None else None),
            message=(str(message).strip() if message is not None else None),
            raw=(str(payload.get("raw")).rstrip("\n") if payload.get("raw") is not None else None),
            source=(str(payload.get("source", "uart")).strip() or "uart"),
            type=str(payload.get("type", "car_state")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class CmdVel:
    ts: float
    mode: str
    vx_norm: float = 0.0
    wz_norm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": float(self.ts),
            "mode": self.mode,
            "vx_norm": float(self.vx_norm),
            "wz_norm": float(self.wz_norm),
            "vx_mps": float(self.vx_norm),
            "wz_rps": float(self.wz_norm),
        }


def make_task_ack(cmd: TaskCmd, accepted: bool, state: str, reason: str = "") -> Dict[str, Any]:
    return TaskAck(
        ts=now_ts(),
        cmd_id=cmd.cmd_id,
        session_id=cmd.session_id,
        epoch=cmd.epoch,
        accepted=bool(accepted),
        state=str(state),
        reason=str(reason or ""),
    ).to_dict()


def make_vision_req(target: str, session_id: str = "", epoch: int = 0, req_id: str = "") -> Dict[str, Any]:
    return {
        "ts": now_ts(),
        "type": "vision_req",
        "mode": "FIND",
        "target": target,
        "session_id": session_id,
        "epoch": int(epoch),
        "req_id": req_id or _new_id("req"),
    }


def make_home_tag_req(session_id: str = "", epoch: int = 0, req_id: str = "") -> Dict[str, Any]:
    return {
        "ts": now_ts(),
        "type": "home_tag_req",
        "mode": "RETURN",
        "session_id": session_id,
        "epoch": int(epoch),
        "req_id": req_id or _new_id("req"),
    }


def make_vision_idle(session_id: str = "", epoch: int = 0, req_id: str = "") -> Dict[str, Any]:
    return {
        "ts": now_ts(),
        "type": "vision_req",
        "mode": "IDLE",
        "session_id": session_id,
        "epoch": int(epoch),
        "req_id": req_id or _new_id("req"),
    }


def make_tts_event(text: str, source: str = "orchestrator", interrupt: bool = False) -> Dict[str, Any]:
    text = str(text).strip()
    if not text:
        raise ProtocolError("tts_event.text 不能为空")
    return {"ts": now_ts(), "type": "tts_event", "text": text, "source": source, "interrupt": bool(interrupt)}
