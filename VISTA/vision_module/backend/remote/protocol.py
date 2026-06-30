#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_REMOTE_ROBOT_ID = "sc171_car_01"


_IMAGE_ENCODING_MAP = {
    "jpg": (".jpg", "image/jpeg"),
    "jpeg": (".jpg", "image/jpeg"),
    "png": (".png", "image/png"),
}


def normalize_image_encoding(value: Any, default: str = "png") -> str:
    text = str(value or default).strip().lower()
    if text in _IMAGE_ENCODING_MAP:
        return text
    fallback = str(default or "png").strip().lower()
    if fallback in _IMAGE_ENCODING_MAP:
        return fallback
    return "png"


def image_encoding_info(value: Any, default: str = "png") -> Tuple[str, str]:
    return _IMAGE_ENCODING_MAP[normalize_image_encoding(value, default=default)]


@dataclass
class RemoteMetadata:
    """Metadata shared with the remote grasp server."""

    robot_id: str = DEFAULT_REMOTE_ROBOT_ID
    cmd: str = "predict"
    command: str = "predict"
    request_id: str = ""
    session_id: str = ""
    target: str = ""
    class_id: Optional[int] = None
    frame_seq: int = 0
    frame_seq_source: str = "fallback"
    timestamp_ms: Optional[int] = None
    camera_names: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_metadata_payload(self) -> Dict[str, Any]:
        if self.class_id is None:
            raise ValueError("remote predict class_id is required")
        command = str(self.command or "predict").strip() or "predict"
        cmd = str(self.cmd or command).strip() or command
        timestamp_ms = self.timestamp_ms
        if timestamp_ms is None:
            timestamp_ms = int(round(time.time() * 1000.0))
        payload: Dict[str, Any] = {
            "robot_id": str(self.robot_id or DEFAULT_REMOTE_ROBOT_ID).strip() or DEFAULT_REMOTE_ROBOT_ID,
            "cmd": cmd,
            "command": command,
            "request_id": str(self.request_id or ""),
            "session_id": str(self.session_id or ""),
            "target": str(self.target or ""),
            "class_id": int(self.class_id),
            "frame_seq": int(self.frame_seq or 0),
            "frame_seq_source": str(self.frame_seq_source or "fallback"),
            "timestamp_ms": int(timestamp_ms),
            "camera_names": list(self.camera_names or []),
        }
        for key, value in dict(self.extras or {}).items():
            if key not in payload:
                payload[key] = value
        return payload


@dataclass
class RemotePredictRequest:
    """One remote grasp request assembled from local synchronized inputs."""

    rgb_bytes: Optional[bytes] = None
    depth_bytes: Optional[bytes] = None
    class_id: Optional[int] = None
    metadata: RemoteMetadata = field(default_factory=RemoteMetadata)
    timeout_s: float = 10.0
    rgb_encoding: str = "jpeg"
    depth_encoding: str = "png"


@dataclass
class RemotePredictResponse:
    """Typed remote response envelope used by the remote capability layer."""

    ok: bool = False
    payload: Dict[str, Any] = field(default_factory=dict)
    status_code: Optional[int] = None
    error: str = ""


def build_predict_multipart(request: RemotePredictRequest) -> Tuple[Dict[str, Any], Dict[str, Tuple[str, bytes, str]]]:
    """Build multipart form fields following the current remote grasp protocol."""
    class_id = request.class_id if request.class_id is not None else request.metadata.class_id
    if class_id is None:
        raise ValueError("remote predict class_id is required")
    request.metadata.class_id = int(class_id)
    metadata_payload = request.metadata.to_metadata_payload()
    data: Dict[str, Any] = {
        "robot_id": str(metadata_payload["robot_id"]),
        "cmd": str(metadata_payload["cmd"]),
        "command": str(metadata_payload["command"]),
        "request_id": str(metadata_payload["request_id"]),
        "session_id": str(metadata_payload["session_id"]),
        "target": str(metadata_payload["target"]),
        "class_id": str(metadata_payload["class_id"]),
        "frame_seq": str(metadata_payload["frame_seq"]),
        "frame_seq_source": str(metadata_payload["frame_seq_source"]),
        "timestamp_ms": str(metadata_payload["timestamp_ms"]),
        "metadata": json.dumps(metadata_payload, ensure_ascii=False),
    }

    files: Dict[str, Tuple[str, bytes, str]] = {}
    if request.rgb_bytes is not None:
        rgb_ext, rgb_mime = image_encoding_info(request.rgb_encoding, default="jpeg")
        files["rgb_file"] = (f"rgb{rgb_ext}", request.rgb_bytes, rgb_mime)
    if request.depth_bytes is not None:
        depth_ext, depth_mime = image_encoding_info(request.depth_encoding, default="png")
        files["depth_file"] = (f"depth{depth_ext}", request.depth_bytes, depth_mime)
    return data, files
