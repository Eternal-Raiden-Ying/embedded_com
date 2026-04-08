#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import platform

from .base import IPredictor  # noqa: F401

MockPredictor = None  # type: ignore
QNNPredictor = None  # type: ignore


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
    from .mock import MockPredictor  # noqa: F401
    QNNPredictor = MockPredictor  # noqa: N816
elif _BACKEND == "real":
    from .QNNPredictor import QNNPredictor  # noqa: F401
else:
    if _prefer_mock_platform():
        from .mock import MockPredictor  # noqa: F401
        QNNPredictor = MockPredictor  # noqa: N816
    else:
        try:
            from .QNNPredictor import QNNPredictor  # noqa: F401
        except Exception:
            from .mock import MockPredictor  # noqa: F401
            QNNPredictor = MockPredictor  # noqa: N816

QNN_YOLO_Segment_Predictor = QNNPredictor

__all__ = ["IPredictor", "MockPredictor", "QNNPredictor", "QNN_YOLO_Segment_Predictor"]
