#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""相机硬件抽象接口（ICamera）"""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class ICamera(ABC):
    """
    相机抽象基类。
    AidLux 真实实现（HardwareCamera / RealSenseDepthCamera）和
    Windows mock 实现（MockCamera）均继承此接口。
    """

    @abstractmethod
    def read_frame(self) -> Optional[np.ndarray]:
        """读取一帧图像，返回 HxWxC uint8 numpy 数组；失败返回 None 或 size==0 的数组。"""
        ...

    @abstractmethod
    def release(self) -> None:
        """释放底层硬件资源。"""
        ...

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
