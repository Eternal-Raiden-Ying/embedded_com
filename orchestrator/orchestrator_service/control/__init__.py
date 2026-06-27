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

from .pid import PIDController
from .types import DockingCommand, DockingControlConfig, EdgeControlObservation, PIDAxisConfig

__all__ = [
    "PIDController",
    "DockingCommand",
    "DockingControlConfig",
    "EdgeControlObservation",
    "PIDAxisConfig",
]
