#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set

ALLOWED_INTENTS: Set[str] = {"FIND", "RETURN", "STOP"}
ALLOWED_VISTA_OPS: Set[str] = {"START", "UPDATE", "RESPOND", "STOP"}
ALLOWED_VISTA_STAGES: Set[str] = {"SEARCH", "GRASP", "RETURN", "IDLE"}


class ProtocolError(ValueError):
    pass


def now_ts() -> float:
    return time.time()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _upper_text(value: Any, default: str = "") -> str:
    text = str(value or default).strip().upper()
    return text or str(default).strip().upper()


def _payload_ts(payload: Dict[str, Any]) -> float:
    if payload.get("ts") is not None:
        return float(payload.get("ts"))
    if payload.get("ts_ms") is not None:
        return float(payload.get("ts_ms")) / 1000.0
    return now_ts()


def _pick_optional_float(payload: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if payload.get(key) is None:
            continue
        try:
            return float(payload.get(key))
        except Exception:
            continue
    return None


def _pick_optional_bool(payload: Dict[str, Any], *keys: str) -> Optional[bool]:
    for key in keys:
        if key in payload and payload.get(key) is not None:
            return bool(payload.get(key))
    return None


def _pick_optional_str(payload: Dict[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        if payload.get(key) is None:
            continue
        value = str(payload.get(key)).strip()
        if value:
            return value
    return None


def _pick_optional_int(payload: Dict[str, Any], *keys: str) -> Optional[int]:
    for key in keys:
        if payload.get(key) is None:
            continue
        try:
            return int(payload.get(key))
        except Exception:
            continue
    return None


def _pick_optional_dict(payload: Dict[str, Any], *keys: str) -> Optional[Dict[str, Any]]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return None


def _compact(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            compacted = _compact(item)
            if compacted is None or compacted == {}:
                continue
            out[key] = compacted
        return out
    if isinstance(value, list):
        return [_compact(item) for item in value if _compact(item) is not None]
    return value


@dataclass
class VisionReqMsg:
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
    type: str = "vision_req"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VisionReqMsg":
        stage = _upper_text(payload.get("stage"), "IDLE")
        op = _upper_text(payload.get("op"), "START")
        if stage not in ALLOWED_VISTA_STAGES:
            raise ProtocolError(f"非法 VISTA stage: {stage!r}")
        if op not in ALLOWED_VISTA_OPS:
            raise ProtocolError(f"非法 VISTA op: {op!r}")
        return cls(
            ts=_payload_ts(payload),
            op=op,
            stage=stage,
            target=_pick_optional_str(payload, "target"),
            mode_hint=_pick_optional_str(payload, "mode_hint"),
            session_id=_pick_optional_str(payload, "session_id"),
            req_id=_pick_optional_str(payload, "req_id"),
            epoch=int(payload.get("epoch", 0) or 0),
            interaction_id=_pick_optional_str(payload, "interaction_id"),
            response=_pick_optional_dict(payload, "response"),
            payload=_pick_optional_dict(payload, "payload"),
            type=str(payload.get("type", "vision_req") or "vision_req"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _compact(asdict(self))


@dataclass
class VisionObsEnvelope:
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
    source: Optional[str] = None
    type: str = "vision_obs"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VisionObsEnvelope":
        msg_type = str(payload.get("type", "vision_obs") or "vision_obs").strip()
        if msg_type != "vision_obs":
            raise ProtocolError(f"非法 vision_obs.type: {msg_type!r}")
        return cls(
            ts=_payload_ts(payload),
            stage=_upper_text(payload.get("stage"), "IDLE"),
            mode=_upper_text(payload.get("mode"), "IDLE"),
            status=_upper_text(payload.get("status"), "RUNNING"),
            session_id=_pick_optional_str(payload, "session_id"),
            req_id=_pick_optional_str(payload, "req_id"),
            epoch=int(payload.get("epoch", 0) or 0),
            interaction=_pick_optional_dict(payload, "interaction"),
            perception=_pick_optional_dict(payload, "perception"),
            proposal=_pick_optional_dict(payload, "proposal"),
            result=_pick_optional_dict(payload, "result"),
            source=_pick_optional_str(payload, "source"),
            type=msg_type,
        )

    def to_dict(self) -> Dict[str, Any]:
        return _compact(asdict(self))


def iter_vision_perception_payloads(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    msg_type = str(payload.get("type", "") or "").strip().lower()
    if msg_type in {"table_edge_obs", "target_obs", "home_tag_obs"}:
        legacy_payload = dict(payload)
        legacy_payload["_from_vision_obs_envelope"] = False
        legacy_payload["_perception_priority"] = 0
        return [legacy_payload]
    if msg_type != "vision_obs":
        return []
    env = VisionObsEnvelope.from_dict(payload)
    perception = dict(env.perception or {})
    if not perception:
        return []
    base = {
        "ts": env.ts,
        "session_id": env.session_id,
        "req_id": env.req_id,
        "epoch": int(env.epoch),
        "source": env.source or "vision_obs",
        "vision_stage": env.stage,
        "vision_mode": env.mode,
        "vision_status": env.status,
    }
    out: List[Dict[str, Any]] = []
    for key in ("table_edge_obs", "target_obs", "home_tag_obs"):
        item = perception.get(key)
        if not isinstance(item, dict):
            continue
        merged = dict(base)
        merged.update(item)
        merged["type"] = key
        merged["_from_vision_obs_envelope"] = True
        merged["_perception_priority"] = 1
        out.append(merged)
    return out


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
            ts=_payload_ts(payload),
            intent=intent,
            confidence=confidence,
            target=target,
            cmd_id=str(payload.get("cmd_id") or payload.get("task_id") or _new_id("cmd")),
            session_id=str(payload.get("session_id") or payload.get("task_id") or _new_id("sess")),
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
class TableEdgeObs:
    ts: float
    table_found: bool
    edge_found: bool
    confidence: float = 0.0
    obs_ts: Optional[float] = None
    age_ms: Optional[float] = None
    frame_id: Optional[int] = None
    seq: Optional[int] = None
    source_mode: Optional[str] = None
    is_stale: bool = False
    yaw_err_rad: Optional[float] = None
    dist_err_m: Optional[float] = None
    lateral_err_m: Optional[float] = None
    edge_angle_rad: Optional[float] = None
    edge_k: Optional[float] = None
    edge_b: Optional[float] = None
    depth_valid: Optional[bool] = None
    table_cx_norm: Optional[float] = None
    table_size_norm: Optional[float] = None
    edge_ready: Optional[bool] = None
    best_turn_dir: Optional[str] = None
    obstacle_flag: bool = False
    obstacle_distance_m: Optional[float] = None
    point_count: Optional[int] = None
    table_point_count: Optional[int] = None
    valid_edge_points: Optional[int] = None
    edge_inlier_count: Optional[int] = None
    target_dist_m: Optional[float] = None
    depth_edge_roi: Optional[list] = None
    table_edge_roi: Optional[list] = None
    edge_roi: Optional[list] = None
    roi_format: Optional[str] = None
    reason: Optional[str] = None
    req_id: Optional[str] = None
    session_id: Optional[str] = None
    epoch: int = 0
    source: Optional[str] = None
    type: str = "table_edge_obs"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TableEdgeObs":
        table_found = bool(payload.get("table_found", payload.get("found", False)))
        edge_found = bool(payload.get("edge_found", payload.get("table_edge_found", table_found)))
        confidence = _pick_optional_float(payload, "confidence", "score", "edge_confidence", "table_confidence") or 0.0
        obs_ts = _pick_optional_float(payload, "obs_ts", "observation_ts", "frame_ts", "ts")
        return cls(
            ts=float(obs_ts) if obs_ts is not None else _payload_ts(payload),
            table_found=table_found,
            edge_found=edge_found,
            confidence=float(confidence),
            obs_ts=obs_ts,
            age_ms=_pick_optional_float(payload, "age_ms", "edge_obs_age_ms"),
            frame_id=_pick_optional_int(payload, "frame_id"),
            seq=_pick_optional_int(payload, "seq", "frame_seq"),
            source_mode=_pick_optional_str(payload, "source_mode", "vision_mode", "mode"),
            is_stale=bool(payload.get("is_stale", payload.get("edge_obs_is_stale", False))),
            yaw_err_rad=_pick_optional_float(payload, "yaw_err_rad", "edge_yaw_err_rad", "yaw_error_rad"),
            dist_err_m=_pick_optional_float(payload, "dist_err_m", "edge_dist_err_m", "distance_error_m", "table_edge_distance_m", "edge_distance_m"),
            lateral_err_m=_pick_optional_float(payload, "lateral_err_m", "edge_lateral_err_m", "lateral_error_m"),
            edge_angle_rad=_pick_optional_float(payload, "edge_angle_rad"),
            edge_k=_pick_optional_float(payload, "edge_k"),
            edge_b=_pick_optional_float(payload, "edge_b"),
            depth_valid=_pick_optional_bool(payload, "depth_valid"),
            table_cx_norm=_pick_optional_float(payload, "table_cx_norm", "table_cx"),
            table_size_norm=_pick_optional_float(payload, "table_size_norm", "table_area_norm", "table_area", "size_norm"),
            edge_ready=_pick_optional_bool(payload, "edge_ready", "table_edge_ready"),
            best_turn_dir=_pick_optional_str(payload, "best_turn_dir", "avoid_dir"),
            obstacle_flag=bool(payload.get("obstacle_flag", payload.get("obstacle", False))),
            obstacle_distance_m=_pick_optional_float(payload, "obstacle_distance_m", "obstacle_distance", "front_obstacle_m"),
            point_count=_pick_optional_int(payload, "point_count"),
            table_point_count=_pick_optional_int(payload, "table_point_count"),
            valid_edge_points=_pick_optional_int(payload, "valid_edge_points", "edge_point_count"),
            edge_inlier_count=_pick_optional_int(payload, "edge_inlier_count", "inlier_count"),
            target_dist_m=_pick_optional_float(payload, "target_dist_m", "target_distance_m"),
            depth_edge_roi=payload.get("depth_edge_roi"),
            table_edge_roi=payload.get("table_edge_roi"),
            edge_roi=payload.get("edge_roi"),
            roi_format=_pick_optional_str(payload, "roi_format"),
            reason=_pick_optional_str(payload, "reason", "error"),
            req_id=_pick_optional_str(payload, "req_id"),
            session_id=_pick_optional_str(payload, "session_id", "task_id"),
            epoch=int(payload.get("epoch", 0) or 0),
            source=_pick_optional_str(payload, "source"),
            type=str(payload.get("type", "table_edge_obs") or "table_edge_obs"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class TargetObs:
    ts: float
    found: bool
    target: Optional[str] = None
    target_found: Optional[bool] = None
    matched_cls: Optional[str] = None
    matched_conf: Optional[float] = None
    matched_bbox: Optional[list] = None
    matched_center: Optional[Dict[str, Any]] = None
    matched_area: Optional[float] = None
    matched_rank_in_all_boxes: Optional[int] = None
    num_target_candidates: Optional[int] = None
    all_candidate_classes: Optional[list] = None
    confidence: Optional[float] = None
    cx_norm: float = 0.0
    cy_norm: Optional[float] = None
    size_norm: float = 0.0
    track_id: Optional[int] = None
    bbox: Optional[list] = None
    boxes_count: Optional[int] = None
    best_cls: Optional[str] = None
    best_conf: Optional[float] = None
    reason: Optional[str] = None
    depth_m: Optional[float] = None
    mask_ready: bool = False
    mask_shape: Optional[list] = None
    mask_area_ratio: Optional[float] = None
    mask_bbox: Optional[list] = None
    req_id: Optional[str] = None
    session_id: Optional[str] = None
    epoch: int = 0
    vx_norm: Optional[float] = None
    vy_norm: Optional[float] = None
    wz_norm: Optional[float] = None
    obstacle_flag: bool = False
    best_turn_dir: Optional[str] = None
    obstacle_distance_m: Optional[float] = None
    source: Optional[str] = None
    type: str = "target_obs"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TargetObs":
        matched_center = _pick_optional_dict(payload, "matched_center")
        cx_value = payload.get("cx_norm", 0.0)
        cy_value = payload.get("cy_norm", payload.get("cy"))
        if isinstance(matched_center, dict):
            cx_value = matched_center.get("cx_norm", cx_value)
            cy_value = matched_center.get("cy_norm", matched_center.get("y_norm", cy_value))
        return cls(
            ts=_payload_ts(payload),
            found=bool(payload.get("target_found", payload.get("found", False))),
            target=(str(payload.get("target")).strip() if payload.get("target") is not None else None),
            target_found=_pick_optional_bool(payload, "target_found"),
            matched_cls=_pick_optional_str(payload, "matched_cls", "target_cls"),
            matched_conf=_pick_optional_float(payload, "matched_conf", "target_conf"),
            matched_bbox=payload.get("matched_bbox"),
            matched_center=matched_center,
            matched_area=_pick_optional_float(payload, "matched_area"),
            matched_rank_in_all_boxes=_pick_optional_int(payload, "matched_rank_in_all_boxes"),
            num_target_candidates=_pick_optional_int(payload, "num_target_candidates"),
            all_candidate_classes=payload.get("all_candidate_classes"),
            confidence=_pick_optional_float(payload, "matched_conf", "confidence", "score"),
            cx_norm=float(cx_value or 0.0),
            cy_norm=(float(cy_value) if cy_value is not None else None),
            size_norm=float(payload.get("matched_area", payload.get("size_norm", payload.get("area_norm", 0.0))) or 0.0),
            track_id=_pick_optional_int(payload, "track_id"),
            bbox=payload.get("matched_bbox") or payload.get("bbox"),
            boxes_count=_pick_optional_int(payload, "boxes_count", "box_count"),
            best_cls=_pick_optional_str(payload, "best_cls", "best_class"),
            best_conf=_pick_optional_float(payload, "best_conf", "best_confidence"),
            reason=_pick_optional_str(payload, "reason"),
            depth_m=_pick_optional_float(payload, "depth_m"),
            mask_ready=bool(payload.get("mask_ready", payload.get("mask_available", False))),
            mask_shape=payload.get("mask_shape"),
            mask_area_ratio=_pick_optional_float(payload, "mask_area_ratio"),
            mask_bbox=payload.get("mask_bbox"),
            req_id=_pick_optional_str(payload, "req_id"),
            session_id=_pick_optional_str(payload, "session_id", "task_id"),
            epoch=int(payload.get("epoch", 0) or 0),
            vx_norm=_pick_optional_float(payload, "vx_norm", "vx", "v_norm", "linear_norm"),
            vy_norm=_pick_optional_float(payload, "vy_norm", "vy", "lateral_norm"),
            wz_norm=_pick_optional_float(payload, "wz_norm", "wz", "omega_norm", "angular_norm"),
            obstacle_flag=bool(payload.get("obstacle_flag", payload.get("obstacle", False))),
            best_turn_dir=_pick_optional_str(payload, "best_turn_dir", "avoid_dir"),
            obstacle_distance_m=_pick_optional_float(payload, "obstacle_distance_m", "obstacle_distance", "front_obstacle_m"),
            source=_pick_optional_str(payload, "source"),
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
    vy_norm: Optional[float] = None
    wz_norm: Optional[float] = None
    obstacle_flag: bool = False
    best_turn_dir: Optional[str] = None
    obstacle_distance_m: Optional[float] = None
    source: Optional[str] = None
    type: str = "home_tag_obs"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "HomeTagObs":
        return cls(
            ts=_payload_ts(payload),
            found=bool(payload.get("found", False)),
            yaw_err_rad=float(payload.get("yaw_err_rad", 0.0)),
            distance_m=_pick_optional_float(payload, "distance_m"),
            req_id=_pick_optional_str(payload, "req_id"),
            session_id=_pick_optional_str(payload, "session_id", "task_id"),
            epoch=int(payload.get("epoch", 0) or 0),
            vx_norm=_pick_optional_float(payload, "vx_norm", "vx", "v_norm", "linear_norm"),
            vy_norm=_pick_optional_float(payload, "vy_norm", "vy", "lateral_norm"),
            wz_norm=_pick_optional_float(payload, "wz_norm", "wz", "omega_norm", "angular_norm"),
            obstacle_flag=bool(payload.get("obstacle_flag", payload.get("obstacle", False))),
            best_turn_dir=_pick_optional_str(payload, "best_turn_dir", "avoid_dir"),
            obstacle_distance_m=_pick_optional_float(payload, "obstacle_distance_m", "obstacle_distance", "front_obstacle_m"),
            source=_pick_optional_str(payload, "source"),
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
    vx: Optional[float] = None
    vy: Optional[float] = None
    wz: Optional[float] = None
    fault_code: Optional[str] = None
    source: str = "uart"
    type: str = "car_state"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CarState":
        state = str(payload.get("state", payload.get("status", "UNKNOWN"))).strip().upper() or "UNKNOWN"
        message = payload.get("message")
        return cls(
            ts=_payload_ts(payload),
            state=state,
            ok=bool(payload.get("ok", state in {"OK", "BUSY", "DONE"})),
            timeout=bool(payload.get("timeout", state == "TIMEOUT")),
            estop=bool(payload.get("estop", state in {"ESTOP", "E_STOP"})),
            fault=bool(payload.get("fault", state in {"FAULT", "ERROR"})),
            mode=(str(payload.get("mode")).strip().upper() if payload.get("mode") is not None else None),
            message=(str(message).strip() if message is not None else None),
            raw=(str(payload.get("raw")).rstrip("\n") if payload.get("raw") is not None else None),
            vx=_pick_optional_float(payload, "vx"),
            vy=_pick_optional_float(payload, "vy"),
            wz=_pick_optional_float(payload, "wz"),
            fault_code=_pick_optional_str(payload, "fault_code"),
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
    vy_norm: float = 0.0
    wz_norm: float = 0.0
    hold_ms: int = 150
    brake: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": float(self.ts),
            "mode": self.mode,
            "vx_norm": float(self.vx_norm),
            "vy_norm": float(self.vy_norm),
            "wz_norm": float(self.wz_norm),
            "vx": float(self.vx_norm),
            "vy": float(self.vy_norm),
            "wz": float(self.wz_norm),
            "hold_ms": int(self.hold_ms),
            "brake": bool(self.brake),
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


def make_vision_req(
    target: Optional[str] = None,
    session_id: str = "",
    epoch: int = 0,
    req_id: str = "",
    *,
    op: str = "START",
    stage: str = "SEARCH",
    mode_hint: str = "",
    interaction_id: str = "",
    response: Optional[Dict[str, Any]] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return VisionReqMsg(
        ts=now_ts(),
        op=_upper_text(op, "START"),
        stage=_upper_text(stage, "SEARCH"),
        target=(str(target).strip() if target is not None and str(target).strip() else None),
        mode_hint=(str(mode_hint).strip().upper() if mode_hint else None),
        session_id=(str(session_id).strip() if session_id else None),
        req_id=req_id or _new_id("req"),
        epoch=int(epoch),
        interaction_id=(str(interaction_id).strip() if interaction_id else None),
        response=dict(response or {}) if isinstance(response, dict) else None,
        payload=dict(payload or {}) if isinstance(payload, dict) else None,
    ).to_dict()


def make_home_tag_req(
    session_id: str = "",
    epoch: int = 0,
    req_id: str = "",
    *,
    op: str = "START",
    mode_hint: str = "TRACK_LOCAL",
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return make_vision_req(
        target=None,
        session_id=session_id,
        epoch=epoch,
        req_id=req_id,
        op=op,
        stage="RETURN",
        mode_hint=mode_hint,
        payload=payload,
    )


def make_vision_idle(session_id: str = "", epoch: int = 0, req_id: str = "") -> Dict[str, Any]:
    return make_vision_req(
        target=None,
        session_id=session_id,
        epoch=epoch,
        req_id=req_id,
        op="STOP",
        stage="IDLE",
    )


def make_tts_event(text: str, source: str = "orchestrator", interrupt: bool = False) -> Dict[str, Any]:
    text = str(text).strip()
    if not text:
        raise ProtocolError("tts_event.text 不能为空")
    return {"ts": now_ts(), "type": "tts_event", "text": text, "source": source, "interrupt": bool(interrupt)}
