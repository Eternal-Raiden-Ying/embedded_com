#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class RemoteMetadata:
    """Metadata shared with the remote grasp server."""

    robot_id: str = "arm_001"
    command: str = "predict"
    class_id: Optional[int] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_form_fields(self) -> Dict[str, Any]:
        """Build request form fields for remote multipart requests."""
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
    seg_bytes: Optional[bytes] = None
    class_id: Optional[int] = None
    metadata: RemoteMetadata = field(default_factory=RemoteMetadata)
    timeout_s: float = 10.0


@dataclass
class RemotePredictResponse:
    """Typed remote response envelope used by the remote capability layer."""

    ok: bool = False
    payload: Dict[str, Any] = field(default_factory=dict)
    status_code: Optional[int] = None
    error: str = ""


def build_predict_multipart(request: RemotePredictRequest) -> Tuple[Dict[str, Any], Dict[str, Tuple[str, bytes, str]]]:
    """Build multipart form fields following the current remote grasp protocol.

    The concrete endpoint behavior should stay aligned with the protocol used by
    ``VISTA/grasp_module/simulate_client_request.py``.
    """
    data: Dict[str, Any] = {"metadata": json.dumps(request.metadata.to_form_fields(), ensure_ascii=False)}
    if request.class_id is not None:
        data["class_id"] = str(request.class_id)

    files: Dict[str, Tuple[str, bytes, str]] = {}
    if request.rgb_bytes is not None:
        files["rgb_file"] = ("rgb.jpg", request.rgb_bytes, "image/jpeg")
    if request.depth_bytes is not None:
        files["depth_file"] = ("depth.png", request.depth_bytes, "image/png")
    if request.seg_bytes is not None:
        files["seg_file"] = ("seg.png", request.seg_bytes, "image/png")
    return data, files
