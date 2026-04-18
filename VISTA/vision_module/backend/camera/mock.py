#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MockCamera：Windows 本地开发用 mock 相机。
不依赖任何硬件，read_frame() 返回全零帧。
"""
import logging
import time

import numpy as np
from .base import ICamera


logger = logging.getLogger("vision.camera.mock")


class MockCamera(ICamera):
    """
    Mock 相机实现，同时兼容 HardwareCamera 和 RealSenseDepthCamera 的构造参数。
      - HardwareCamera 参数：out_w, out_h, device, format, fps, in_w, in_h, ...
      - RealSenseDepthCamera 参数：width, height, fps
    多余参数通过 **kwargs 忽略。
    """

    def __init__(
        self,
        out_w: int = 640,
        out_h: int = 640,
        width: int = None,
        height: int = None,
        fps: int = 8,
        **kwargs,
    ):
        self._w = int(width if width is not None else out_w)
        self._h = int(height if height is not None else out_h)
        fps_value = max(1, int(fps or 8))
        self._frame_interval_s = 1.0 / float(fps_value)
        self._next_frame_ts = 0.0
        logger.info("mock camera initialized | size=%sx%s", self._w, self._h)

    def read_frame(self) -> np.ndarray:
        """返回全零的 H×W×3 uint8 数组，模拟空白帧。"""
        now = time.time()
        if self._next_frame_ts > now:
            time.sleep(self._next_frame_ts - now)
        self._next_frame_ts = time.time() + self._frame_interval_s
        return np.zeros((self._h, self._w, 3), dtype=np.uint8)

    def release(self) -> None:
        logger.info("mock camera released | size=%sx%s", self._w, self._h)
