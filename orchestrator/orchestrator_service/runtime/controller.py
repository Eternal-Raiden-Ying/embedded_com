#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compatible import shim.

Motion control implementation now lives in orchestrator_service.control.motion_controller
so runtime/state_machine.py no longer owns controller-level implementation details.
"""

from ..control.motion_controller import MotionController, MotionDecision

__all__ = ["MotionController", "MotionDecision"]
