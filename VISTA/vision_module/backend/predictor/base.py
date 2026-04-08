#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class IPredictor(ABC):
    @abstractmethod
    def predict_frame(self, frame: np.ndarray) -> Tuple[list, list]:
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        ...

    @abstractmethod
    def release(self) -> None:
        ...
