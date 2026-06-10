#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set

import msgpack

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


def _pick_optional_bbox(payload: Dict[str, Any], *keys: str) -> Optional[list]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            value = value.get("bbox") or value.get("xyxy") or value.get("box")
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            continue
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in value[:4]]
        except Exception:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        return [x1, y1, x2, y2]
    return None


def _pick_bbox_area_ratio(payload: Dict[str, Any]) -> Optional[float]:
    direct = _pick_optional_float(payload, "table_bbox_area_ratio", "yolo_bbox_area_norm", "table_bbox_area_norm", "yolo_table_bbox_area_ratio")
    if direct is not None:
        return max(0.0, min(1.0, float(direct)))
    bbox = _pick_optional_dict(payload, "table_bbox", "yolo_bbox", "bbox")
    bbox_list = _pick_optional_bbox(payload, "table_bbox_xyxy", "yolo_table_bbox", "table_bbox", "yolo_bbox", "bbox")
    if bbox_list is not None:
        x1, y1, x2, y2 = [float(v) for v in bbox_list[:4]]
    elif bbox:
        x1 = _pick_optional_float(bbox, "x1", "left")
        y1 = _pick_optional_float(bbox, "y1", "top")
        x2 = _pick_optional_float(bbox, "x2", "right")
        y2 = _pick_optional_float(bbox, "y2", "bottom")
    else:
        return None
    rgb_w = _pick_optional_float(payload, "rgb_w", "rgb_width", "image_w", "image_width", "frame_w", "frame_width")
    rgb_h = _pick_optional_float(payload, "rgb_h", "rgb_height", "image_h", "image_height", "frame_h", "frame_height")
    shape = payload.get("rgb_shape")
    if (rgb_w is None or rgb_h is None) and isinstance(shape, (list, tuple)) and len(shape) >= 2:
        rgb_h = _pick_optional_float({"h": shape[0]}, "h")
        rgb_w = _pick_optional_float({"w": shape[1]}, "w")
    if None in (x1, y1, x2, y2, rgb_w, rgb_h) or float(rgb_w or 0.0) <= 0.0 or float(rgb_h or 0.0) <= 0.0:
        return None
    area = max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))
    return max(0.0, min(1.0, area / (float(rgb_w) * float(rgb_h))))


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
    req_type: Optional[str] = None
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
            req_type=_pick_optional_str(payload, "req_type"),
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
    edge_valid: Optional[bool] = None
    confidence: float = 0.0
    edge_conf: Optional[float] = None
    obs_ts: Optional[float] = None
    age_ms: Optional[float] = None
    frame_id: Optional[int] = None
    seq: Optional[int] = None
    frame_capture_ts: Optional[float] = None
    vision_start_ts: Optional[float] = None
    vision_done_ts: Optional[float] = None
    obs_publish_ts: Optional[float] = None
    obs_recv_ts: Optional[float] = None
    control_ts: Optional[float] = None
    obs_seq: Optional[int] = None
    camera_frame_seq: Optional[int] = None
    camera_frame_ts_ms: Optional[float] = None
    vision_process_start_ts_ms: Optional[float] = None
    vision_process_end_ts_ms: Optional[float] = None
    vision_publish_ts_ms: Optional[float] = None
    obs_out_send_ts_ms: Optional[float] = None
    orchestrator_recv_ts_ms: Optional[float] = None
    state_machine_consume_ts_ms: Optional[float] = None
    cmd_publish_ts_ms: Optional[float] = None
    frame_age_ms: Optional[float] = None
    vision_process_ms: Optional[float] = None
    publish_delay_ms: Optional[float] = None
    obs_total_age_ms: Optional[float] = None
    control_loop_age_ms: Optional[float] = None
    edge_update_interval_ms: Optional[float] = None
    camera_frame_interval_ms: Optional[float] = None
    camera_frame_hz: Optional[float] = None
    vision_process_interval_ms: Optional[float] = None
    vision_publish_interval_ms: Optional[float] = None
    table_edge_worker_interval_ms: Optional[float] = None
    table_edge_no_new_frame_count: Optional[int] = None
    scheduler_publish_ms: Optional[float] = None
    obs_out_send_interval_ms: Optional[float] = None
    obs_out_send_hz: Optional[float] = None
    obs_out_drop_or_skip_count: Optional[int] = None
    obs_out_skip_reason: Optional[str] = None
    send_hz_config: Optional[float] = None
    track_local_send_hz_config: Optional[float] = None
    table_edge_obs_recv_interval_ms: Optional[float] = None
    orchestrator_recv_interval_ms: Optional[float] = None
    table_edge_obs_recv_hz: Optional[float] = None
    state_machine_tick_interval_ms: Optional[float] = None
    state_machine_consume_interval_ms: Optional[float] = None
    same_obs_reuse_count: Optional[int] = None
    obs_seq_gap: Optional[int] = None
    obs_age_at_consume_ms: Optional[float] = None
    vision_publish_to_orch_recv_ms: Optional[float] = None
    orch_recv_to_state_consume_ms: Optional[float] = None
    edge_process_ms: Optional[float] = None
    dropped_frame_count: Optional[int] = None
    processed_frame_count: Optional[int] = None
    latest_frame_lag_ms: Optional[float] = None
    source_mode: Optional[str] = None
    is_stale: bool = False
    yaw_err_rad: Optional[float] = None
    dist_err_m: Optional[float] = None
    lateral_err_m: Optional[float] = None
    depth_p10: Optional[float] = None
    close_depth_ratio: Optional[float] = None
    edge_angle_rad: Optional[float] = None
    edge_k: Optional[float] = None
    edge_b: Optional[float] = None
    depth_valid: Optional[bool] = None
    table_cx_norm: Optional[float] = None
    table_size_norm: Optional[float] = None
    plane_cx_norm: Optional[float] = None
    plane_width_norm: Optional[float] = None
    plane_area_ratio: Optional[float] = None
    plane_touch_left: bool = False
    plane_touch_right: bool = False
    plane_touch_top: bool = False
    plane_touch_bottom: bool = False
    view_err_norm: Optional[float] = None
    view_source: Optional[str] = None
    view_reliable: bool = False
    fov_guard_active: bool = False
    fov_guard_reason: Optional[str] = None
    table_bbox_found: bool = False
    table_bbox_xyxy: Optional[list] = None
    table_bbox_area_ratio: Optional[float] = None
    table_bbox_conf_raw: Optional[float] = None
    table_bbox_conf_used_for_gate: bool = False
    yolo_reliable: bool = False
    yolo_valid_reason: Optional[str] = None
    yolo_invalid_reason: Optional[str] = None
    docking_enabled_by_yolo: bool = False
    edge_control_allowed: bool = False
    edge_control_block_reason: Optional[str] = None
    yolo_bbox_area_norm: Optional[float] = None
    yolo_bbox_touch_left: bool = False
    yolo_bbox_touch_right: bool = False
    yolo_bbox_touch_bottom: bool = False
    yolo_bbox_touch_boundary: bool = False
    table_bbox_touch_left: bool = False
    table_bbox_touch_right: bool = False
    table_bbox_touch_bottom: bool = False
    table_bbox_boundary_allowed: bool = False
    yolo_table_control_valid: bool = False
    yolo_table_roi_valid: bool = False
    yolo_gate_open: bool = False
    yolo_table_conf: Optional[float] = None
    yolo_bbox_center_x_norm: Optional[float] = None
    yolo_roi_center_x_norm: Optional[float] = None
    roi_source: Optional[str] = None
    roi_reason: Optional[str] = None
    roi_phase: Optional[str] = None
    yolo_table_edge_stable_count: Optional[int] = None
    edge_ready: Optional[bool] = None
    best_turn_dir: Optional[str] = None
    obstacle_flag: bool = False
    obstacle_distance_m: Optional[float] = None
    point_count: Optional[int] = None
    table_point_count: Optional[int] = None
    valid_edge_points: Optional[int] = None
    edge_inlier_count: Optional[int] = None
    target_dist_m: Optional[float] = None
    edge_trusted: bool = False
    pose_found: bool = False
    pose_source: Optional[str] = None
    final_pose_source: Optional[str] = None
    table_geometry_score: Optional[float] = None
    front_plane_score: Optional[float] = None
    usable_for_approach: bool = False
    usable_for_alignment: bool = False
    usable_for_stop: bool = False
    control_reject_reason: Optional[str] = None
    reject_reason: Optional[str] = None
    fast_temporal_jump: bool = False
    table_confirmed_by_yolo: bool = False
    yolo_gate_reason: Optional[str] = None
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

    def __post_init__(self):
        if getattr(self, "table_found", False):
            if not getattr(self, "table_bbox_found", False):
                self.table_bbox_found = True
            if not getattr(self, "yolo_reliable", False):
                self.yolo_reliable = True

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TableEdgeObs":
        table_found = bool(payload.get("table_found", payload.get("found", False)))
        edge_valid = _pick_optional_bool(payload, "edge_valid")
        edge_found = bool(payload.get("edge_found", payload.get("table_edge_found", edge_valid if edge_valid is not None else table_found)))
        confidence = _pick_optional_float(payload, "confidence", "edge_conf", "score", "edge_confidence", "table_confidence") or 0.0
        obs_ts = _pick_optional_float(payload, "obs_ts", "observation_ts", "frame_ts", "ts")
        return cls(
            ts=float(obs_ts) if obs_ts is not None else _payload_ts(payload),
            table_found=table_found,
            edge_found=edge_found,
            edge_valid=edge_valid if edge_valid is not None else edge_found,
            confidence=float(confidence),
            edge_conf=_pick_optional_float(payload, "edge_conf", "confidence", "edge_confidence"),
            obs_ts=obs_ts,
            age_ms=_pick_optional_float(payload, "age_ms", "edge_obs_age_ms"),
            frame_id=_pick_optional_int(payload, "frame_id"),
            seq=_pick_optional_int(payload, "seq", "frame_seq"),
            frame_capture_ts=_pick_optional_float(payload, "frame_capture_ts", "capture_ts", "frame_ts"),
            vision_start_ts=_pick_optional_float(payload, "vision_start_ts"),
            vision_done_ts=_pick_optional_float(payload, "vision_done_ts"),
            obs_publish_ts=_pick_optional_float(payload, "obs_publish_ts", "publish_ts"),
            obs_recv_ts=_pick_optional_float(payload, "obs_recv_ts"),
            control_ts=_pick_optional_float(payload, "control_ts"),
            obs_seq=_pick_optional_int(payload, "obs_seq"),
            camera_frame_seq=_pick_optional_int(payload, "camera_frame_seq"),
            camera_frame_ts_ms=_pick_optional_float(payload, "camera_frame_ts_ms"),
            vision_process_start_ts_ms=_pick_optional_float(payload, "vision_process_start_ts_ms"),
            vision_process_end_ts_ms=_pick_optional_float(payload, "vision_process_end_ts_ms"),
            vision_publish_ts_ms=_pick_optional_float(payload, "vision_publish_ts_ms"),
            obs_out_send_ts_ms=_pick_optional_float(payload, "obs_out_send_ts_ms"),
            orchestrator_recv_ts_ms=_pick_optional_float(payload, "orchestrator_recv_ts_ms"),
            state_machine_consume_ts_ms=_pick_optional_float(payload, "state_machine_consume_ts_ms"),
            cmd_publish_ts_ms=_pick_optional_float(payload, "cmd_publish_ts_ms"),
            frame_age_ms=_pick_optional_float(payload, "frame_age_ms"),
            vision_process_ms=_pick_optional_float(payload, "vision_process_ms", "edge_process_ms", "total_edge_process_ms"),
            publish_delay_ms=_pick_optional_float(payload, "publish_delay_ms"),
            obs_total_age_ms=_pick_optional_float(payload, "obs_total_age_ms"),
            control_loop_age_ms=_pick_optional_float(payload, "control_loop_age_ms"),
            edge_update_interval_ms=_pick_optional_float(payload, "edge_update_interval_ms", "edge_obs_period_ms"),
            camera_frame_interval_ms=_pick_optional_float(payload, "camera_frame_interval_ms"),
            camera_frame_hz=_pick_optional_float(payload, "camera_frame_hz", "camera_frames_hz"),
            vision_process_interval_ms=_pick_optional_float(payload, "vision_process_interval_ms", "table_edge_process_interval_ms"),
            vision_publish_interval_ms=_pick_optional_float(payload, "vision_publish_interval_ms", "table_edge_publish_interval_ms"),
            table_edge_worker_interval_ms=_pick_optional_float(payload, "table_edge_worker_interval_ms"),
            table_edge_no_new_frame_count=_pick_optional_int(payload, "table_edge_no_new_frame_count"),
            scheduler_publish_ms=_pick_optional_float(payload, "scheduler_publish_ms"),
            obs_out_send_interval_ms=_pick_optional_float(payload, "obs_out_send_interval_ms"),
            obs_out_send_hz=_pick_optional_float(payload, "obs_out_send_hz"),
            obs_out_drop_or_skip_count=_pick_optional_int(payload, "obs_out_drop_or_skip_count"),
            obs_out_skip_reason=_pick_optional_str(payload, "obs_out_skip_reason"),
            send_hz_config=_pick_optional_float(payload, "send_hz_config"),
            track_local_send_hz_config=_pick_optional_float(payload, "track_local_send_hz_config"),
            table_edge_obs_recv_interval_ms=_pick_optional_float(payload, "table_edge_obs_recv_interval_ms"),
            orchestrator_recv_interval_ms=_pick_optional_float(payload, "orchestrator_recv_interval_ms", "table_edge_obs_recv_interval_ms"),
            table_edge_obs_recv_hz=_pick_optional_float(payload, "table_edge_obs_recv_hz"),
            state_machine_tick_interval_ms=_pick_optional_float(payload, "state_machine_tick_interval_ms"),
            state_machine_consume_interval_ms=_pick_optional_float(payload, "state_machine_consume_interval_ms"),
            same_obs_reuse_count=_pick_optional_int(payload, "same_obs_reuse_count"),
            obs_seq_gap=_pick_optional_int(payload, "obs_seq_gap"),
            obs_age_at_consume_ms=_pick_optional_float(payload, "obs_age_at_consume_ms"),
            vision_publish_to_orch_recv_ms=_pick_optional_float(payload, "vision_publish_to_orch_recv_ms"),
            orch_recv_to_state_consume_ms=_pick_optional_float(payload, "orch_recv_to_state_consume_ms"),
            edge_process_ms=_pick_optional_float(payload, "edge_process_ms"),
            dropped_frame_count=_pick_optional_int(payload, "dropped_frame_count"),
            processed_frame_count=_pick_optional_int(payload, "processed_frame_count"),
            latest_frame_lag_ms=_pick_optional_float(payload, "latest_frame_lag_ms"),
            source_mode=_pick_optional_str(payload, "source_mode", "vision_mode", "mode"),
            is_stale=bool(payload.get("is_stale", payload.get("edge_obs_is_stale", False))),
            yaw_err_rad=_pick_optional_float(payload, "yaw_err_rad", "yaw_err", "edge_yaw_err_rad", "yaw_error_rad"),
            dist_err_m=_pick_optional_float(payload, "dist_err_m", "dist_err", "edge_dist_err_m", "distance_error_m", "table_edge_distance_m", "edge_distance_m"),
            lateral_err_m=_pick_optional_float(payload, "lateral_err_m", "edge_lateral_err_m", "lateral_error_m"),
            depth_p10=_pick_optional_float(payload, "depth_p10"),
            close_depth_ratio=_pick_optional_float(payload, "close_depth_ratio"),
            edge_angle_rad=_pick_optional_float(payload, "edge_angle_rad"),
            edge_k=_pick_optional_float(payload, "edge_k"),
            edge_b=_pick_optional_float(payload, "edge_b"),
            depth_valid=_pick_optional_bool(payload, "depth_valid"),
            table_cx_norm=_pick_optional_float(payload, "table_cx_norm", "table_cx"),
            table_size_norm=_pick_optional_float(payload, "table_size_norm", "table_area_norm", "table_area", "size_norm"),
            plane_cx_norm=_pick_optional_float(payload, "plane_cx_norm", "front_plane_cx_norm"),
            plane_width_norm=_pick_optional_float(payload, "plane_width_norm", "front_plane_width_norm"),
            plane_area_ratio=_pick_optional_float(payload, "plane_area_ratio", "front_plane_area_ratio", "front_face_area_ratio"),
            plane_touch_left=bool(payload.get("plane_touch_left", False)),
            plane_touch_right=bool(payload.get("plane_touch_right", False)),
            plane_touch_top=bool(payload.get("plane_touch_top", False)),
            plane_touch_bottom=bool(payload.get("plane_touch_bottom", False)),
            view_err_norm=_pick_optional_float(payload, "view_err_norm", "view_error_norm"),
            view_source=_pick_optional_str(payload, "view_source"),
            view_reliable=bool(payload.get("view_reliable", False)),
            fov_guard_active=bool(payload.get("fov_guard_active", False)),
            fov_guard_reason=_pick_optional_str(payload, "fov_guard_reason"),
            table_bbox_found=bool(payload.get("table_bbox_found", _pick_optional_bbox(payload, "table_bbox_xyxy", "yolo_table_bbox", "table_bbox", "detected_table_bbox") is not None)),
            table_bbox_xyxy=_pick_optional_bbox(payload, "table_bbox_xyxy", "yolo_table_bbox", "table_bbox", "detected_table_bbox"),
            table_bbox_area_ratio=_pick_bbox_area_ratio(payload),
            table_bbox_conf_raw=_pick_optional_float(payload, "table_bbox_conf_raw", "yolo_table_conf"),
            table_bbox_conf_used_for_gate=bool(payload.get("table_bbox_conf_used_for_gate", False)),
            yolo_reliable=bool(payload.get("yolo_reliable", payload.get("table_bbox_found", _pick_optional_bbox(payload, "table_bbox_xyxy", "yolo_table_bbox", "table_bbox", "detected_table_bbox") is not None))),
            yolo_valid_reason=_pick_optional_str(payload, "yolo_valid_reason"),
            yolo_invalid_reason=_pick_optional_str(payload, "yolo_invalid_reason"),
            docking_enabled_by_yolo=bool(payload.get("docking_enabled_by_yolo", payload.get("table_bbox_found", _pick_optional_bbox(payload, "table_bbox_xyxy", "yolo_table_bbox", "table_bbox", "detected_table_bbox") is not None))),
            edge_control_allowed=bool(payload.get("edge_control_allowed", payload.get("table_bbox_found", _pick_optional_bbox(payload, "table_bbox_xyxy", "yolo_table_bbox", "table_bbox", "detected_table_bbox") is not None))),
            edge_control_block_reason=_pick_optional_str(payload, "edge_control_block_reason"),
            yolo_bbox_area_norm=_pick_bbox_area_ratio(payload),
            yolo_bbox_touch_left=bool(payload.get("yolo_bbox_touch_left", False)),
            yolo_bbox_touch_right=bool(payload.get("yolo_bbox_touch_right", False)),
            yolo_bbox_touch_bottom=bool(payload.get("yolo_bbox_touch_bottom", False)),
            yolo_bbox_touch_boundary=bool(payload.get("yolo_bbox_touch_boundary", False)),
            table_bbox_touch_left=bool(payload.get("table_bbox_touch_left", payload.get("yolo_bbox_touch_left", False))),
            table_bbox_touch_right=bool(payload.get("table_bbox_touch_right", payload.get("yolo_bbox_touch_right", False))),
            table_bbox_touch_bottom=bool(payload.get("table_bbox_touch_bottom", payload.get("yolo_bbox_touch_bottom", False))),
            table_bbox_boundary_allowed=bool(payload.get("table_bbox_boundary_allowed", False)),
            yolo_table_control_valid=bool(payload.get("yolo_table_control_valid", payload.get("table_bbox_found", _pick_optional_bbox(payload, "table_bbox_xyxy", "yolo_table_bbox", "table_bbox", "detected_table_bbox") is not None or payload.get("yolo_reliable", False)))),
            yolo_table_roi_valid=bool(payload.get("yolo_table_roi_valid", False)),
            yolo_gate_open=bool(payload.get("yolo_gate_open", False)),
            yolo_table_conf=_pick_optional_float(payload, "yolo_table_conf"),
            yolo_bbox_center_x_norm=_pick_optional_float(payload, "yolo_bbox_center_x_norm"),
            yolo_roi_center_x_norm=_pick_optional_float(payload, "yolo_roi_center_x_norm"),
            roi_source=_pick_optional_str(payload, "roi_source"),
            roi_reason=_pick_optional_str(payload, "roi_reason"),
            roi_phase=_pick_optional_str(payload, "roi_phase"),
            yolo_table_edge_stable_count=_pick_optional_int(payload, "yolo_table_edge_stable_count", "edge_stable_count"),
            edge_ready=_pick_optional_bool(payload, "edge_ready", "table_edge_ready"),
            best_turn_dir=_pick_optional_str(payload, "best_turn_dir", "avoid_dir"),
            obstacle_flag=bool(payload.get("obstacle_flag", payload.get("obstacle", False))),
            obstacle_distance_m=_pick_optional_float(payload, "obstacle_distance_m", "obstacle_distance", "front_obstacle_m"),
            point_count=_pick_optional_int(payload, "point_count"),
            table_point_count=_pick_optional_int(payload, "table_point_count"),
            valid_edge_points=_pick_optional_int(payload, "valid_edge_points", "edge_point_count"),
            edge_inlier_count=_pick_optional_int(payload, "edge_inlier_count", "inlier_count"),
            target_dist_m=_pick_optional_float(payload, "target_dist_m", "target_distance_m"),
            edge_trusted=bool(payload.get("edge_trusted", payload.get("valid_for_control", False))),
            pose_found=bool(payload.get("pose_found", False)),
            pose_source=_pick_optional_str(payload, "pose_source"),
            final_pose_source=_pick_optional_str(payload, "final_pose_source"),
            table_geometry_score=_pick_optional_float(payload, "table_geometry_score"),
            front_plane_score=_pick_optional_float(payload, "front_plane_score"),
            usable_for_approach=bool(payload.get("usable_for_approach", False)),
            usable_for_alignment=bool(payload.get("usable_for_alignment", False)),
            usable_for_stop=bool(payload.get("usable_for_stop", False)),
            control_reject_reason=_pick_optional_str(payload, "control_reject_reason"),
            reject_reason=_pick_optional_str(payload, "reject_reason", "fast_gate_reject_reason", "fast_raw_reject_reason"),
            fast_temporal_jump=bool(payload.get("fast_temporal_jump", False)),
            table_confirmed_by_yolo=bool(payload.get("table_confirmed_by_yolo", False)),
            yolo_gate_reason=_pick_optional_str(payload, "yolo_gate_reason"),
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
    matched_center_full_norm: Optional[Dict[str, Any]] = None
    matched_center_offset_norm: Optional[Dict[str, Any]] = None
    matched_area: Optional[float] = None
    matched_rank_in_all_boxes: Optional[int] = None
    num_target_candidates: Optional[int] = None
    all_candidate_classes: Optional[list] = None
    confidence: Optional[float] = None
    x_norm: Optional[float] = None
    y_norm: Optional[float] = None
    cx_norm: float = 0.0
    cy_norm: Optional[float] = None
    size_norm: float = 0.0
    track_id: Optional[int] = None
    bbox: Optional[list] = None
    boxes_count: Optional[int] = None
    best_cls: Optional[str] = None
    best_conf: Optional[float] = None
    bbox_valid: Optional[bool] = None
    bbox_invalid_reason: Optional[str] = None
    reason: Optional[str] = None
    depth_m: Optional[float] = None
    mask_ready: bool = False
    mask_shape: Optional[list] = None
    mask_area_ratio: Optional[float] = None
    mask_bbox: Optional[list] = None
    req_id: Optional[str] = None
    session_id: Optional[str] = None
    epoch: int = 0
    vx_mps: Optional[float] = None
    vy_mps: Optional[float] = None
    wz_radps: Optional[float] = None
    obstacle_flag: bool = False
    best_turn_dir: Optional[str] = None
    obstacle_distance_m: Optional[float] = None
    source: Optional[str] = None
    type: str = "target_obs"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TargetObs":
        matched_center = _pick_optional_dict(payload, "matched_center")
        matched_center_full_norm = _pick_optional_dict(payload, "matched_center_full_norm")
        matched_center_offset_norm = _pick_optional_dict(payload, "matched_center_offset_norm")
        cx_value = payload.get("cx_norm", 0.0)
        cy_value = payload.get("cy_norm", payload.get("cy"))
        if cy_value is None and isinstance(matched_center_full_norm, dict):
            cy_value = matched_center_full_norm.get("cy")
        return cls(
            ts=_payload_ts(payload),
            found=bool(payload.get("target_found", payload.get("found", False))),
            target=(str(payload.get("target")).strip() if payload.get("target") is not None else None),
            target_found=_pick_optional_bool(payload, "target_found"),
            matched_cls=_pick_optional_str(payload, "matched_cls", "target_cls"),
            matched_conf=_pick_optional_float(payload, "matched_conf", "target_conf"),
            matched_bbox=payload.get("matched_bbox"),
            matched_center=matched_center,
            matched_center_full_norm=matched_center_full_norm,
            matched_center_offset_norm=matched_center_offset_norm,
            matched_area=_pick_optional_float(payload, "matched_area"),
            matched_rank_in_all_boxes=_pick_optional_int(payload, "matched_rank_in_all_boxes"),
            num_target_candidates=_pick_optional_int(payload, "num_target_candidates"),
            all_candidate_classes=payload.get("all_candidate_classes"),
            confidence=_pick_optional_float(payload, "matched_conf", "confidence", "score"),
            x_norm=_pick_optional_float(payload, "x_norm"),
            y_norm=_pick_optional_float(payload, "y_norm"),
            cx_norm=float(cx_value or 0.0),
            cy_norm=(float(cy_value) if cy_value is not None else None),
            size_norm=float(payload.get("matched_area", payload.get("size_norm", payload.get("area_norm", 0.0))) or 0.0),
            track_id=_pick_optional_int(payload, "track_id"),
            bbox=payload.get("matched_bbox") or payload.get("bbox"),
            boxes_count=_pick_optional_int(payload, "boxes_count", "box_count"),
            best_cls=_pick_optional_str(payload, "best_cls", "best_class"),
            best_conf=_pick_optional_float(payload, "best_conf", "best_confidence"),
            bbox_valid=_pick_optional_bool(payload, "bbox_valid"),
            bbox_invalid_reason=_pick_optional_str(payload, "bbox_invalid_reason"),
            reason=_pick_optional_str(payload, "reason"),
            depth_m=_pick_optional_float(payload, "depth_m"),
            mask_ready=bool(payload.get("mask_ready", payload.get("mask_available", False))),
            mask_shape=payload.get("mask_shape"),
            mask_area_ratio=_pick_optional_float(payload, "mask_area_ratio"),
            mask_bbox=payload.get("mask_bbox"),
            req_id=_pick_optional_str(payload, "req_id"),
            session_id=_pick_optional_str(payload, "session_id", "task_id"),
            epoch=int(payload.get("epoch", 0) or 0),
            vx_mps=_pick_optional_float(payload, "vx_mps", "vx", "v_norm", "linear_norm"),
            vy_mps=_pick_optional_float(payload, "vy_mps", "vy", "lateral_norm"),
            wz_radps=_pick_optional_float(payload, "wz_radps", "wz", "omega_norm", "angular_norm"),
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
    vx_mps: Optional[float] = None
    vy_mps: Optional[float] = None
    wz_radps: Optional[float] = None
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
            vx_mps=_pick_optional_float(payload, "vx_mps", "vx", "v_norm", "linear_norm"),
            vy_mps=_pick_optional_float(payload, "vy_mps", "vy", "lateral_norm"),
            wz_radps=_pick_optional_float(payload, "wz_radps", "wz", "omega_norm", "angular_norm"),
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
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    wz_radps: float = 0.0
    hold_ms: int = 150
    brake: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": float(self.ts),
            "mode": self.mode,
            "vx_mps": float(self.vx_mps),
            "vy_mps": float(self.vy_mps),
            "wz_radps": float(self.wz_radps),
            "vx": float(self.vx_mps),
            "vy": float(self.vy_mps),
            "wz": float(self.wz_radps),
            "hold_ms": int(self.hold_ms),
            "brake": bool(self.brake),
        }


@dataclass
class ArmCommand:
    x_cm: float
    y_cm: float
    z_cm: float
    pitch_deg: float
    roll_deg: float
    claw_deg: float
    time_ms: int = 500

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArmResponse:
    ok: bool
    message: str = ""
    raw_line: str = ""
    ts: float = 0.0

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ArmResponse":
        return cls(
            ok=bool(payload.get("ok", False)),
            message=str(payload.get("message", "")),
            raw_line=str(payload.get("raw_line", "")),
            ts=float(payload.get("ts", now_ts())),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "message": self.message, "raw_line": self.raw_line, "ts": self.ts}


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
    req_type: str = "",
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
        req_type=(str(req_type).strip().lower() if req_type else None),
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
    mode_hint: str = "FIND_OBJECT",
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


def make_grasp_req(
    target: str,
    class_id: int,
    session_id: str = "",
    epoch: int = 0,
    req_id: str = "",
    *,
    op: str = "START",
) -> Dict[str, Any]:
    return make_vision_req(
        target=target,
        session_id=session_id,
        epoch=epoch,
        req_id=req_id,
        op=op,
        stage="GRASP",
        mode_hint="GRASP_REMOTE",
        payload={
            "class_id": int(class_id),
            "remote_grasp": True,
            "need_depth": True,
        },
    )


def make_vision_idle(session_id: str = "", epoch: int = 0, req_id: str = "") -> Dict[str, Any]:
    return make_vision_req(
        target=None,
        session_id=session_id,
        epoch=epoch,
        req_id=req_id,
        op="STOP",
        stage="IDLE",
        req_type="mode_request",
    )


def make_tts_event(text: str, source: str = "orchestrator", interrupt: bool = False) -> Dict[str, Any]:
    text = str(text).strip()
    if not text:
        raise ProtocolError("tts_event.text 不能为空")
    return {"ts": now_ts(), "type": "tts_event", "text": text, "source": source, "interrupt": bool(interrupt)}


def pack_msg(payload: Dict[str, Any]) -> bytes:
    return msgpack.packb(payload, use_bin_type=True)


def unpack_msg(data: bytes) -> Dict[str, Any]:
    return msgpack.unpackb(data, raw=False)
