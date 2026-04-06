#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mock 相机实现：返回全零帧，不依赖任何硬件。"""
import numpy as np
from ..base import ICamera


class MockCamera(ICamera):
    """
    Windows 本地开发用 mock 相机。
    read_frame() 返回全零的 out_h x out_w x 3 uint8 数组。
    """

    def __init__(self, out_w: int = 640, out_h: int = 640, **kwargs):
        self._w = int(out_w)
        self._h = int(out_h)
        print(f"[MockCamera] 初始化 mock 相机 {self._w}x{self._h}（无硬件依赖）")

    def read_frame(self) -> np.ndarray:
        return np.zeros((self._h, self._w, 3), dtype=np.uint8)

    def release(self) -> None:
        pass
