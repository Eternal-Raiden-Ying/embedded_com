#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AidLux 真实 NPU 推理器实现。
懒加载 QNN_YOLO_Segment_Predictor，避免在非 ARM 平台 import 时崩溃。
"""
import os
import sys
from typing import Tuple
import numpy as np
from ..base import IPredictor


def _ensure_predictor_path():
    """将 QNNPredictor 所在目录加入 sys.path"""
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    _pred_dir = os.path.join(_root, 'VISTA', 'vision_module', 'backend', 'predictor')
    if _pred_dir not in sys.path:
        sys.path.insert(0, _pred_dir)


class AidluxPredictor(IPredictor):
    """AidLux QNN DSP 推理器，懒加载 aidlite SDK。"""

    def __init__(self, model_profile):
        _ensure_predictor_path()
        from QNNPredictor import QNN_YOLO_Segment_Predictor
        self._predictor = QNN_YOLO_Segment_Predictor(model_profile)

    def predict_frame(self, frame: np.ndarray) -> Tuple[list, list]:
        return self._predictor.predict_frame(frame)

    def is_ready(self) -> bool:
        return self._predictor.is_ready()

    def release(self) -> None:
        self._predictor.release()
