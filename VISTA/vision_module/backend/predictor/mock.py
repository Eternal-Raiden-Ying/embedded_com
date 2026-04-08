#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import Tuple

import numpy as np

from .base import IPredictor


logger = logging.getLogger("vision.inference")


class MockPredictor(IPredictor):
    def __init__(self, args=None, **kwargs):
        logger.info("mock predictor initialized")

    def predict_frame(self, frame: np.ndarray) -> Tuple[list, list]:
        return [], []

    def is_ready(self) -> bool:
        return True

    def release(self) -> None:
        logger.info("mock predictor released")
