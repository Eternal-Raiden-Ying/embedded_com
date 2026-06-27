#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config.schema import CarMotionConfig, ControlThresholds
from ..control.types import DockingControlConfig
from ..ipc.protocol import (
    ArmCommand,
    ArmResponse,
    CarState,
    HomeTagObs,
    TableEdgeObs,
    TargetObs,
    TaskCmd,
    make_grasp_req,
    make_tts_event,
    make_vision_idle,
    make_vision_req,
)
from ..bridge.arm_protocol import parse_arm_response
from ..utils.grasp_utils import grasp_to_pose_params
from ..utils.target_utils import target_to_class_id
from .common import monotonic_ts
from .context import RuntimeContext, State
from .controller import MotionController, MotionDecision
from .control_authority import decide_table_control_authority
from .core_types import (
    KNOWN_VISION_STATUS,
    MOVING_STATES,
    TABLE_APPROACH_STATES,
    TABLE_VISION_STATES,
    TARGET_SEARCH_STATES,
    TARGET_VISION_STATES,
    ObstacleSignal,
    VisionStageBinding,
    _GRASP_ARM_TIMEOUT_S,
    _GRASP_REPOSITION_TIMEOUT_S,
    _GRASP_RESPOND_TIMEOUT_S,
    _GRASP_RESULT_TIMEOUT_S,
    _GRASP_RETRY_LIMIT,
)

from .export_state import ExportStateMixin
from .safety.base_motion_safety import BaseMotionSafetyMixin
from .safety.stale_guard import StaleGuardMixin
from .states.grasp_flow import GraspFlowMixin
from .states.recovery import RecoveryMixin
from .states.table_docking import TableDockingMixin
from .states.target_search import TargetSearchMixin
from .task_runtime import TaskRuntimeMixin
from .transitions import TransitionsMixin
from .vision_sync import VisionSyncMixin


class OrchestratorCore(ExportStateMixin, VisionSyncMixin, TransitionsMixin, TaskRuntimeMixin, TableDockingMixin, TargetSearchMixin, GraspFlowMixin, RecoveryMixin, StaleGuardMixin, BaseMotionSafetyMixin):
    def __init__(
        self,
        cfg: ControlThresholds,
        car_cfg: CarMotionConfig,
        docking_cfg: Optional[DockingControlConfig] = None,
        logger: Optional[Callable] = None,
    ):
        self.cfg = cfg
        self.car_cfg = car_cfg
        self.ctx = RuntimeContext()
        self._logger = logger
        self.controller = MotionController(cfg, car_cfg, docking_cfg)
        self.transition_observer: Optional[Callable[[str, str, str], None]] = None
        self.last_transition_snapshot: Dict[str, Any] = {}
        self._pending_reset_traces: List[Dict[str, Any]] = []
        self._last_req_mono = 0.0
        self._last_mode_request_key = ""
        self._last_target_update_key = ""
        self._last_target_update_mono = 0.0
        self._last_stop_mono = 0.0

    def _log(self, level: str, msg: str, *args):
        if self._logger:
            self._logger(level, "state_machine", msg, {"args": [str(a) for a in args]} if args else None)

    def tick(self) -> MotionDecision:
        safety_override = self._check_safety_interlock()
        if safety_override is not None:
            decision = safety_override
        else:
            dispatch = {
                State.IDLE: self._tick_idle,
                State.SEARCH_TABLE: self._tick_search_table,
                State.YOLO_ACQUIRE_ALIGN: self._tick_yolo_acquire_align,
                State.YOLO_APPROACH: self._tick_yolo_approach,
                State.EDGE_ADJUST: self._tick_edge_adjust,
                State.FINAL_SLOW_STOP: self._tick_final_slow_stop,
                State.NO_PROGRESS_RECOVERY: self._tick_no_progress_recovery,
                State.AT_TABLE_EDGE: self._tick_at_table_edge,
                State.SEARCH_TARGET_INIT: self._tick_search_target_init,
                State.EDGE_SLIDE_SEARCH: self._tick_edge_slide_search,
                State.TARGET_CONFIRM: self._tick_target_confirm,
                State.TARGET_LOCKED: self._tick_target_locked,
                State.FREEZE_BASE: self._tick_freeze_base,
                State.LEAVE_EDGE: self._tick_leave_edge,
                State.RELOCATE_TO_EDGE: self._tick_relocate_to_edge,
                State.REACQUIRE_TABLE: self._tick_reacquire_table,
                State.NEXT_TABLE: self._tick_next_table,
                State.AVOID_OBSTACLE: self._tick_avoid_obstacle,
                State.RETURN_HOME: self._tick_return_home,
                State.ERROR_RECOVERY: self._tick_error_recovery,
                State.DONE: self._tick_done,
                State.GRASP: self._tick_grasp,
            }
            decision = dispatch.get(self.ctx.state, self._tick_idle)()
        decision = self._apply_soft_interception_and_safety(decision)
        self.last_decision = decision
        return decision

    def _table_visible(self, obs: Optional[TableEdgeObs]) -> bool:
        return bool(obs is not None and self._table_yolo_reliable(obs) and (obs.table_found or self._table_plane_stable(obs) or self._table_yolo_reliable(obs)))

