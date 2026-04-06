#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MockPredictor：Windows 本地开发用 mock 推理器。
不依赖 QNN/DSP，predict_frame() 始终返回空检测结果。
"""
from typing import Tuple
import numpy as np
from .base import IPredictor


class MockPredictor(IPredictor):
    """
    Mock 推理器实现，兼容 QNN_YOLO_Segment_Predictor 的构造参数（args 配置对象）。
    predict_frame() 返回 ([], [])，模拟无目标场景。
    """

    def __init__(self, args=None, **kwargs):
        print("[MockPredictor] 初始化 mock 推理器（无 NPU 依赖）")

    def predict_frame(self, frame: np.ndarray) -> Tuple[list, list]:
        return [], []

    def is_ready(self) -> bool:
        return True

    def release(self) -> None:
        pass
