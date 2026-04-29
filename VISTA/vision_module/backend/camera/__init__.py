#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import os
import platform
from typing import Dict

from .base import ICamera  # noqa: F401
from .mock import MockCamera  # noqa: F401


_LOG = logging.getLogger("vision.camera")

ColorCamera = None  # type: ignore
IRCamera = None  # type: ignore
HardwareCamera = None  # type: ignore
RealSenseDepthCamera = None  # type: ignore


def _requested_backend() -> str:
    explicit = str(os.environ.get("VISTA_BACKEND", "")).strip().lower()
    if explicit in {"mock", "real", "auto"}:
        return explicit
    legacy = str(os.environ.get("ENV", "")).strip().lower()
    if legacy == "mock":
        return "mock"
    if legacy in {"prod", "real"}:
        return "real"
    return "auto"


def _prefer_mock_platform() -> bool:
    return platform.system().lower().startswith("win")


def _set_mock_exports() -> None:
    global ColorCamera, IRCamera, HardwareCamera, RealSenseDepthCamera

    ColorCamera = MockCamera
    IRCamera = MockCamera
    HardwareCamera = MockCamera
    RealSenseDepthCamera = MockCamera


def _set_real_exports() -> None:
    global ColorCamera, IRCamera, HardwareCamera, RealSenseDepthCamera

    from .ColorCamera import ColorCamera as _ColorCamera
    from .IRCamera import IRCamera as _IRCamera
    from .HardwareCamera import HardwareCamera as _HardwareCamera
    from .RealSenseDepthCamera import RealSenseDepthCamera as _RealSenseDepthCamera

    ColorCamera = _ColorCamera
    IRCamera = _IRCamera
    HardwareCamera = _HardwareCamera
    RealSenseDepthCamera = _RealSenseDepthCamera


_BACKEND = _requested_backend()
_RESOLVED_BACKEND = ""
_BACKEND_NOTE = ""

if _BACKEND == "mock":
    _set_mock_exports()
    _RESOLVED_BACKEND = "mock"
elif _BACKEND == "real":
    _set_real_exports()
    _RESOLVED_BACKEND = "real"
else:
    if _prefer_mock_platform():
        _set_mock_exports()
        _RESOLVED_BACKEND = "mock"
        _BACKEND_NOTE = "auto resolved to mock on Windows; set VISTA_BACKEND=real to validate the real camera path"
        _LOG.info(_BACKEND_NOTE)
    else:
        try:
            _set_real_exports()
            _RESOLVED_BACKEND = "real"
        except Exception as exc:
            _set_mock_exports()
            _RESOLVED_BACKEND = "mock"
            _BACKEND_NOTE = f"auto fallback to mock after real camera import failed: {exc}"
            _LOG.warning(_BACKEND_NOTE)


def camera_backend_status() -> Dict[str, str]:
    return {
        "requested_backend": _BACKEND,
        "resolved_backend": _RESOLVED_BACKEND,
        "note": _BACKEND_NOTE,
    }


__all__ = [
    "ICamera",
    "MockCamera",
    "ColorCamera",
    "IRCamera",
    "HardwareCamera",
    "RealSenseDepthCamera",
    "camera_backend_status",
]
