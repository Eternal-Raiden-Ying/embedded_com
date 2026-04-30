#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from ..ipc.protocol import CarState, HomeTagObs, TableEdgeObs, TargetObs, TaskCmd
from .common import monotonic_ts


class State(str, Enum):
    IDLE = "IDLE"
    SEARCH_TABLE = "SEARCH_TABLE"
    COARSE_ALIGN = "COARSE_ALIGN"
    CONTROLLED_APPROACH = "CONTROLLED_APPROACH"
    FINAL_LOCK = "FINAL_LOCK"
    DOCK_RETRY = "DOCK_RETRY"
    AT_TABLE_EDGE = "AT_TABLE_EDGE"
    SEARCH_TARGET_INIT = "SEARCH_TARGET_INIT"
    EDGE_SLIDE_SEARCH = "EDGE_SLIDE_SEARCH"
    TARGET_CONFIRM = "TARGET_CONFIRM"
    TARGET_LOCKED = "TARGET_LOCKED"
    FREEZE_BASE = "FREEZE_BASE"
    LEAVE_EDGE = "LEAVE_EDGE"
    RELOCATE_TO_EDGE = "RELOCATE_TO_EDGE"
    REACQUIRE_EDGE = "REACQUIRE_EDGE"
    NEXT_TABLE = "NEXT_TABLE"
    AVOID_OBSTACLE = "AVOID_OBSTACLE"
    RETURN_HOME = "RETURN_HOME"
    ERROR_RECOVERY = "ERROR_RECOVERY"
    DONE = "DONE"


@dataclass
class RuntimeContext:
    state: State = State.IDLE
    prev_state: Optional[State] = None
    resume_state: Optional[State] = None

    task_intent: str = ""
    active_target: Optional[str] = None
    active_session_id: str = ""
    active_epoch: int = 0
    active_req_id: str = ""
    active_vision_stage: str = ""
    active_vision_mode: str = ""
    current_edge_id: str = "front"
    edge_visit_order: List[str] = field(default_factory=lambda: ["front", "right", "back", "left"])
    edge_visit_index: int = 0
    edge_transition_count: int = 0
    table_cycle_count: int = 0
    relocate_turn_sign: int = 1
    slide_direction_sign: int = 1

    last_task_cmd: Optional[TaskCmd] = None
    last_table_obs: Optional[TableEdgeObs] = None
    last_target_obs: Optional[TargetObs] = None
    last_home_obs: Optional[HomeTagObs] = None
    last_car_state: Optional[CarState] = None

    state_enter_mono: float = field(default_factory=monotonic_ts)
    state_enter_wall_ts: float = field(default_factory=time.time)
    task_start_wall_ts: float = 0.0
    last_car_state_mono: float = 0.0

    pending_vision_msgs: List[Dict] = field(default_factory=list)
    pending_tts_msgs: List[Dict] = field(default_factory=list)

    last_fail_reason: str = ""
    last_enter_reason: str = ""
    last_safety_reason: str = ""
    vision_req_fail_streak: int = 0

    table_found_frames: int = 0
    table_lost_frames: int = 0
    table_lock_frames: int = 0
    approach_aligned_frames: int = 0
    target_found_frames: int = 0
    target_lost_frames: int = 0
    target_lock_frames: int = 0
    tag_lost_frames: int = 0
    tag_arrived_frames: int = 0
    avoid_clear_frames: int = 0
    avoid_retry_count: int = 0
    dock_retry_count: int = 0

    table_loss_since_mono: float = 0.0
    target_loss_since_mono: float = 0.0
    tag_loss_since_mono: float = 0.0
    target_stable_since_mono: float = 0.0
    target_center_history: List[Dict[str, float]] = field(default_factory=list)
    target_last_center_jitter: float = 0.0
    target_last_lost_reason: str = ""
    target_last_transition_reason: str = ""

    def clear_motion_counters(self):
        self.table_found_frames = 0
        self.table_lost_frames = 0
        self.table_lock_frames = 0
        self.approach_aligned_frames = 0
        self.target_found_frames = 0
        self.target_lost_frames = 0
        self.target_lock_frames = 0
        self.tag_lost_frames = 0
        self.tag_arrived_frames = 0
        self.avoid_clear_frames = 0
        self.table_loss_since_mono = 0.0
        self.target_loss_since_mono = 0.0
        self.tag_loss_since_mono = 0.0
        self.target_stable_since_mono = 0.0
        self.target_center_history.clear()
        self.target_last_center_jitter = 0.0
        self.target_last_lost_reason = ""
        self.target_last_transition_reason = ""

    def clear_perception_cache(self):
        self.last_table_obs = None
        self.last_target_obs = None
        self.last_home_obs = None

    def reset_edge_plan(self):
        self.current_edge_id = self.edge_visit_order[0] if self.edge_visit_order else "front"
        self.edge_visit_index = 0
        self.edge_transition_count = 0
        self.relocate_turn_sign = 1
        self.slide_direction_sign = 1

    def advance_edge(self) -> bool:
        if not self.edge_visit_order:
            return False
        if self.edge_visit_index + 1 >= len(self.edge_visit_order):
            return False
        self.edge_visit_index += 1
        self.current_edge_id = self.edge_visit_order[self.edge_visit_index]
        self.edge_transition_count += 1
        self.relocate_turn_sign *= -1
        self.slide_direction_sign = 1
        return True

    def clear_task_context(self):
        self.task_intent = ""
        self.active_target = None
        self.active_session_id = ""
        self.active_epoch = 0
        self.active_req_id = ""
        self.active_vision_stage = ""
        self.active_vision_mode = ""
        self.task_start_wall_ts = 0.0
        self.resume_state = None
        self.last_safety_reason = ""
        self.table_cycle_count = 0
        self.avoid_retry_count = 0
        self.dock_retry_count = 0
        self.vision_req_fail_streak = 0
        self.reset_edge_plan()
        self.clear_perception_cache()
        self.clear_motion_counters()
