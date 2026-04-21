#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


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

    robot_id: str = "arm_001"
    command: str = "predict"
    class_id: Optional[int] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_form_fields(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"robot_id": self.robot_id, "cmd": self.command}
        if self.class_id is not None:
            payload["class_id"] = self.class_id
        payload.update(self.extras)
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
    data: Dict[str, Any] = {"metadata": json.dumps(request.metadata.to_form_fields(), ensure_ascii=False)}
    if request.class_id is not None:
        data["class_id"] = str(request.class_id)

    files: Dict[str, Tuple[str, bytes, str]] = {}
    if request.rgb_bytes is not None:
        rgb_ext, rgb_mime = image_encoding_info(request.rgb_encoding, default="jpeg")
        files["rgb_file"] = (f"rgb{rgb_ext}", request.rgb_bytes, rgb_mime)
    if request.depth_bytes is not None:
        depth_ext, depth_mime = image_encoding_info(request.depth_encoding, default="png")
        files["depth_file"] = (f"depth{depth_ext}", request.depth_bytes, depth_mime)
    return data, files
