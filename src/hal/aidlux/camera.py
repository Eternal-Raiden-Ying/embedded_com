#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AidLux 真实相机实现。
懒加载 HardwareCamera，避免在非 ARM 平台 import 时崩溃。
"""
import os
import sys
from typing import Optional
import numpy as np
from ..base import ICamera


def _ensure_camera_path():
    """将 HardwareCamera 所在目录加入 sys.path"""
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    _cam_dir = os.path.join(_root, 'VISTA', 'vision_module', 'backend', 'camera')
    if _cam_dir not in sys.path:
        sys.path.insert(0, _cam_dir)


class AidluxCamera(ICamera):
    """AidLux GStreamer 硬件相机，懒加载 fast_cam C++ 扩展。"""

    def __init__(self, **kwargs):
        _ensure_camera_path()
        from HardwareCamera import HardwareCamera
        self._cam = HardwareCamera(**kwargs)

    def read_frame(self) -> Optional[np.ndarray]:
        return self._cam.read_frame()

    def release(self) -> None:
        self._cam.release()
