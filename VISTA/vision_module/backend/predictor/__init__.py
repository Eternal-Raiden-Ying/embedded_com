#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import os
import platform
from typing import Dict

from .base import IPredictor  # noqa: F401
from .mock import MockPredictor  # noqa: F401


_LOG = logging.getLogger("vision.predictor")

QNN_YOLO_Dectec_Predictor = None  # type: ignore
QNN_YOLO_Detect_Predictor = None  # type: ignore
QNN_YOLO_Segment_Predictor = None  # type: ignore


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
    global QNN_YOLO_Dectec_Predictor, QNN_YOLO_Detect_Predictor, QNN_YOLO_Segment_Predictor

    QNN_YOLO_Dectec_Predictor = MockPredictor  # noqa: N816
    QNN_YOLO_Detect_Predictor = MockPredictor  # noqa: N816
    QNN_YOLO_Segment_Predictor = MockPredictor  # noqa: N816


def _set_real_exports() -> None:
    global QNN_YOLO_Dectec_Predictor, QNN_YOLO_Detect_Predictor, QNN_YOLO_Segment_Predictor
    from .QNN_YOLO_Dectec_Predictor import QNN_YOLO_Dectec_Predictor, QNN_YOLO_Detect_Predictor  # noqa: F401
    from .QNN_YOLO_Segment_Predictor import QNN_YOLO_Segment_Predictor  # noqa: F401


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
        _BACKEND_NOTE = "auto resolved to mock on Windows; set VISTA_BACKEND=real to validate the real predictor path"
        _LOG.info(_BACKEND_NOTE)
    else:
        try:
            _set_real_exports()
            _RESOLVED_BACKEND = "real"
        except Exception as exc:
            _set_mock_exports()
            _RESOLVED_BACKEND = "mock"
            _BACKEND_NOTE = f"auto fallback to mock after real predictor import failed: {exc}"
            _LOG.warning(_BACKEND_NOTE)

def predictor_backend_status() -> Dict[str, str]:
    return {
        "requested_backend": _BACKEND,
        "resolved_backend": _RESOLVED_BACKEND,
        "note": _BACKEND_NOTE,
    }


__all__ = [
    "IPredictor",
    "MockPredictor",
    "QNN_YOLO_Dectec_Predictor",
    "QNN_YOLO_Detect_Predictor",
    "QNN_YOLO_Segment_Predictor",
    "predictor_backend_status",
]
