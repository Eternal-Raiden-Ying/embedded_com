#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机包入口 — 硬件抽象层（HAL）工厂。

根据环境变量 ENV 透明地切换实现：
  ENV=mock  → MockCamera（Windows 本地开发，无硬件依赖）
  ENV=prod  → HardwareCamera / RealSenseDepthCamera（AidLux，默认）

调用方（vision_engine.py）无需感知 ENV，保持原有 import 风格：
    from .camera import HardwareCamera, RealSenseDepthCamera
"""
import os

from .base import ICamera  # noqa: F401

_IS_MOCK = os.environ.get("ENV", "prod").lower() == "mock"

if _IS_MOCK:
    from .mock import MockCamera as HardwareCamera       # noqa: F401
    from .mock import MockCamera as RealSenseDepthCamera  # noqa: F401
else:
    from .HardwareCamera import HardwareCamera            # noqa: F401
    from .RealSenseDepthCamera import RealSenseDepthCamera  # noqa: F401

__all__ = ["ICamera", "HardwareCamera", "RealSenseDepthCamera"]
