#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Control package public exports.

Keep this module lightweight.  `config.schema` imports `control.types`, so importing
heavy controller implementations here creates a circular import:

    config.schema -> control.__init__ -> control.motion_controller -> config.schema

Import `MotionController` from `orchestrator_service.control.motion_controller` or
from the backward-compatible `orchestrator_service.runtime.controller` shim when
needed.
"""

from .docking_controller import DockingController
from .motion_adapter import Stm32MotionAdapter
from .pid import PIDController
from .types import DockingCommand, DockingControlConfig, EdgeControlObservation, PIDAxisConfig

__all__ = [
    "DockingController",
    "Stm32MotionAdapter",
    "PIDController",
    "DockingCommand",
    "DockingControlConfig",
    "EdgeControlObservation",
    "PIDAxisConfig",
]
