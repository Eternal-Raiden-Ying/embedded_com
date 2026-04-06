#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAL 抽象接口层
定义硬件无关的抽象基类，供 aidlux/ 和 mock/ 两套实现共同遵守。
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class ICamera(ABC):
    """相机抽象接口"""

    @abstractmethod
    def read_frame(self) -> Optional[np.ndarray]:
        """读取一帧图像，返回 HxWxC uint8 numpy 数组，失败返回 None"""
        ...

    @abstractmethod
    def release(self) -> None:
        """释放硬件资源"""
        ...

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


class IPredictor(ABC):
    """NPU 推理器抽象接口"""

    @abstractmethod
    def predict_frame(self, frame: np.ndarray) -> Tuple[list, list]:
        """
        对单帧 RGB 图像进行推理。
        返回 (out_boxes, masks)，格式与 QNN_YOLO_Segment_Predictor 保持一致。
        """
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """推理器是否就绪"""
        ...

    @abstractmethod
    def release(self) -> None:
        """释放 NPU/DSP 资源"""
        ...
