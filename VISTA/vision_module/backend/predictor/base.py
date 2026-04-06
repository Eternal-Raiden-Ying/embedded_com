#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NPU 推理器硬件抽象接口（IPredictor）"""
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class IPredictor(ABC):
    """
    NPU 推理器抽象基类。
    AidLux 真实实现（QNN_YOLO_Segment_Predictor）和
    Windows mock 实现（MockPredictor）均继承此接口。
    """

    @abstractmethod
    def predict_frame(self, frame: np.ndarray) -> Tuple[list, list]:
        """
        对单帧 RGB 图像进行推理。
        :param frame: HxWx3 uint8 numpy 数组
        :return: (out_boxes, masks)，格式与 QNN_YOLO_Segment_Predictor 保持一致
        """
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """推理器是否就绪（模型已加载、资源未释放）。"""
        ...

    @abstractmethod
    def release(self) -> None:
        """释放 NPU/DSP 资源。"""
        ...
