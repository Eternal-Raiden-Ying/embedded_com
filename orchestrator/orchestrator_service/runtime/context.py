#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from ..ipc.protocol import CarState, HomeTagObs, TargetObs, TaskCmd
from .common import monotonic_ts


class State(str, Enum):
    STOP = "STOP"
    AUTOSEARCH = "AUTOSEARCH"
    AUTOEXPLORE = "AUTOEXPLORE"
    SEARCH = "SEARCH"
    RETURN = "RETURN"


@dataclass
class RuntimeContext:
    state: State = State.STOP
    task_intent: str = ""
    active_target: Optional[str] = None
    active_session_id: str = ""
    active_epoch: int = 0
    active_req_id: str = ""
    last_task_cmd: Optional[TaskCmd] = None
    last_target_obs: Optional[TargetObs] = None
    last_home_obs: Optional[HomeTagObs] = None
    last_car_state: Optional[CarState] = None
    state_enter_mono: float = field(default_factory=monotonic_ts)
    state_enter_wall_ts: float = field(default_factory=time.time)
    task_start_wall_ts: float = 0.0
    last_car_state_mono: float = 0.0
    last_target_found_wall_ts: float = 0.0
    last_home_found_wall_ts: float = 0.0
    found_frames: int = 0
    lost_frames: int = 0
    arrived_frames: int = 0
    tag_found_frames: int = 0
    tag_lost_frames: int = 0
    tag_arrived_frames: int = 0
    pending_vision_msgs: List[Dict] = field(default_factory=list)
    pending_tts_msgs: List[Dict] = field(default_factory=list)
    last_fail_reason: str = ""
    last_enter_reason: str = ""
    vision_req_fail_streak: int = 0

    def clear_motion_counters(self):
        self.found_frames = 0
        self.lost_frames = 0
        self.arrived_frames = 0
        self.tag_found_frames = 0
        self.tag_lost_frames = 0
        self.tag_arrived_frames = 0

    def clear_perception_cache(self):
        self.last_target_obs = None
        self.last_home_obs = None
        self.last_target_found_wall_ts = 0.0
        self.last_home_found_wall_ts = 0.0

    def clear_task_context(self):
        self.task_intent = ""
        self.active_target = None
        self.active_session_id = ""
        self.active_epoch = 0
        self.active_req_id = ""
        self.task_start_wall_ts = 0.0
        self.clear_perception_cache()
        self.clear_motion_counters()
        self.vision_req_fail_streak = 0
