#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Set

ROBOT_ID = "SC171"
MQTT_TOPIC_CMD = "robot/v1/SC171/mobile/cmd"
MQTT_TOPIC_ACK = "robot/v1/SC171/mobile/ack"
MQTT_TOPIC_STATUS = "robot/v1/SC171/mobile/status"
MQTT_TOPIC_HEARTBEAT = "robot/v1/SC171/heartbeat"

SUPPORTED_COMMANDS: Set[str] = {
    "fetch_object",
    "stop",
    "resume",
    "retry_search",
    "go_home",
    "query_status",
}

SUPPORTED_TARGETS: Set[str] = {"apple", "banana", "bottle", "cup"}

ERROR_CODES: Dict[str, int] = {
    "invalid_json": 1001,
    "invalid_command": 1002,
    "invalid_target": 1003,
    "missing_target": 1004,
    "busy": 1005,
    "resume_unavailable": 1006,
    "backend_unavailable": 1007,
    "task_rejected": 1008,
}


class MobileProtocolError(ValueError):
    def __init__(self, message: str, error_code: int):
        super().__init__(message)
        self.error_code = int(error_code)


def now_ts() -> float:
    return time.time()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@dataclass
class MobileCommand:
    cmd_id: str
    robot_id: str
    session_id: str
    cmd: str
    ts: float
    target: Optional[str] = None
    text: Optional[str] = None
    epoch: int = 0
    source: str = "mobile"

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
        default_robot_id: str,
        supported_targets: Optional[Set[str]] = None,
        allow_legacy_command_compat: bool = True,
    ) -> "MobileCommand":
        supported = set(supported_targets or SUPPORTED_TARGETS)
        legacy_type = str(payload.get("type", "")).strip().upper()
        cmd = str(payload.get("cmd", "")).strip().lower()
        if not cmd and allow_legacy_command_compat and legacy_type == "FIND_AND_PICK":
            cmd = "fetch_object"
        if cmd not in SUPPORTED_COMMANDS:
            raise MobileProtocolError(f"unsupported cmd: {cmd!r}", ERROR_CODES["invalid_command"])
        target = payload.get("target")
        if cmd == "fetch_object":
            target = str(target or "").strip().lower()
            if not target:
                raise MobileProtocolError("fetch_object requires target", ERROR_CODES["missing_target"])
            if target not in supported:
                raise MobileProtocolError(f"unsupported target: {target!r}", ERROR_CODES["invalid_target"])
        else:
            target = str(target).strip().lower() if target is not None and str(target).strip() else None
        session_id = str(payload.get("session_id") or new_id("sess"))
        return cls(
            cmd_id=str(payload.get("cmd_id") or new_id("cmd")),
            robot_id=str(default_robot_id or ROBOT_ID),
            session_id=session_id,
            cmd=cmd,
            ts=float(payload.get("ts", now_ts())),
            target=target,
            text=(str(payload.get("text")).strip() if payload.get("text") is not None else None),
            epoch=int(payload.get("epoch", 0) or 0),
            source=str(payload.get("source", "wechat_miniprogram") or "wechat_miniprogram"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value not in (None, "", [], {})}


@dataclass
class MobileStatus:
    robot_id: str
    session_id: str
    state: str
    ts: float
    type: str = "mobile_status"
    kind: str = "status"
    target: Optional[str] = None
    message: str = ""
    progress: Optional[int] = None
    error_code: Optional[int] = None
    source: str = "mobile_gateway"
    command: Optional[str] = None
    backend_state: Optional[str] = None
    epoch: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value not in (None, "", [], {})}


def make_error_status(
    robot_id: str,
    session_id: str,
    message: str,
    error_code: int,
    *,
    target: Optional[str] = None,
    command: Optional[str] = None,
    epoch: int = 0,
) -> Dict[str, Any]:
    return MobileStatus(
        robot_id=robot_id,
        session_id=session_id,
        state="error",
        target=target,
        message=message,
        progress=0,
        error_code=int(error_code),
        command=command,
        epoch=epoch,
        ts=now_ts(),
    ).to_dict()
