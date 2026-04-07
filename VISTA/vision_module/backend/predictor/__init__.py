#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predictor backend selector.

Priority:
1. `VISTA_BACKEND=mock|real|auto`
2. legacy `ENV=mock|prod`
3. auto-detect by platform/import availability
"""

import os
import platform

from .base import IPredictor  # noqa: F401


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


_BACKEND = _requested_backend()

if _BACKEND == "mock":
    from .mock import MockPredictor as QNN_YOLO_Segment_Predictor  # noqa: F401
elif _BACKEND == "real":
    from .QNNPredictor import QNN_YOLO_Segment_Predictor  # noqa: F401
else:
    if _prefer_mock_platform():
        from .mock import MockPredictor as QNN_YOLO_Segment_Predictor  # noqa: F401
    else:
        try:
            from .QNNPredictor import QNN_YOLO_Segment_Predictor  # noqa: F401
        except Exception:
            from .mock import MockPredictor as QNN_YOLO_Segment_Predictor  # noqa: F401

__all__ = ["IPredictor", "QNN_YOLO_Segment_Predictor"]
