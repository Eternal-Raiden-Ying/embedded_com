#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .context import State


_GRASP_RESPOND_TIMEOUT_S = 5.0
_GRASP_RESULT_TIMEOUT_S = 15.0
_GRASP_ARM_TIMEOUT_S = 10.0
_GRASP_RETRY_LIMIT = 3
_GRASP_REPOSITION_TIMEOUT_S = 5.0


KNOWN_VISION_STATUS = {"RUNNING", "WAITING_RESPONSE", "RESULT_READY", "FAILED", "RELAXING"}


MOVING_STATES = {
    State.SEARCH_TABLE,
    State.YOLO_ACQUIRE_ALIGN,
    State.YOLO_APPROACH,
    State.EDGE_ADJUST,
    State.FINAL_SLOW_STOP,
    State.NO_PROGRESS_RECOVERY,
    State.EDGE_SLIDE_SEARCH,
    State.LEAVE_EDGE,
    State.RELOCATE_TO_EDGE,
    State.REACQUIRE_TABLE,
    State.NEXT_TABLE,
    State.RETURN_HOME,
    State.AVOID_OBSTACLE,
    State.GRASP,
}


TABLE_VISION_STATES = {
    State.SEARCH_TABLE,
    State.YOLO_ACQUIRE_ALIGN,
    State.YOLO_APPROACH,
    State.EDGE_ADJUST,
    State.FINAL_SLOW_STOP,
    State.REACQUIRE_TABLE,
}


TARGET_VISION_STATES = {
    State.SEARCH_TARGET_INIT,
    State.EDGE_SLIDE_SEARCH,
    State.TARGET_CONFIRM,
    State.TARGET_LOCKED,
    State.FREEZE_BASE,
}


TABLE_APPROACH_STATES = {
    State.SEARCH_TABLE,
    State.YOLO_ACQUIRE_ALIGN,
    State.YOLO_APPROACH,
    State.EDGE_ADJUST,
    State.FINAL_SLOW_STOP,
}


TARGET_SEARCH_STATES = {
    State.SEARCH_TARGET_INIT,
    State.EDGE_SLIDE_SEARCH,
    State.TARGET_CONFIRM,
    State.TARGET_LOCKED,
    State.FREEZE_BASE,
}


@dataclass
class ObstacleSignal:
    active: bool
    best_turn_dir: str = ""
    distance_m: Optional[float] = None
    source: str = ""


@dataclass(frozen=True)
class VisionStageBinding:
    stage: str
    mode_hint: str
    target: Optional[str] = None
    payload: Optional[Dict[str, object]] = None
