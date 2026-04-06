#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理器包入口 — 硬件抽象层（HAL）工厂。

根据环境变量 ENV 透明地切换实现：
  ENV=mock  → MockPredictor（Windows 本地开发，无 NPU 依赖）
  ENV=prod  → QNN_YOLO_Segment_Predictor（AidLux QNN DSP，默认）

调用方（vision_engine.py）无需感知 ENV，保持原有 import 风格：
    from .predictor import QNN_YOLO_Segment_Predictor
"""
import os

from .base import IPredictor  # noqa: F401

_IS_MOCK = os.environ.get("ENV", "prod").lower() == "mock"

if _IS_MOCK:
    from .mock import MockPredictor as QNN_YOLO_Segment_Predictor  # noqa: F401
else:
    from .QNNPredictor import QNN_YOLO_Segment_Predictor  # noqa: F401

__all__ = ["IPredictor", "QNN_YOLO_Segment_Predictor"]
