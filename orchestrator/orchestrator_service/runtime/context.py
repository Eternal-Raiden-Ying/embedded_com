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
    YOLO_ACQUIRE_ALIGN = "YOLO_ACQUIRE_ALIGN"
    YOLO_APPROACH = "YOLO_APPROACH"
    EDGE_ADJUST = "EDGE_ADJUST"
    FINAL_SLOW_STOP = "FINAL_SLOW_STOP"
    AT_TABLE_EDGE = "AT_TABLE_EDGE"
    SEARCH_TARGET_INIT = "SEARCH_TARGET_INIT"
    EDGE_SLIDE_SEARCH = "EDGE_SLIDE_SEARCH"
    TARGET_CONFIRM = "TARGET_CONFIRM"
    TARGET_LOCKED = "TARGET_LOCKED"
    FREEZE_BASE = "FREEZE_BASE"
    GRASP = "GRASP"
    LEAVE_EDGE = "LEAVE_EDGE"
    RELOCATE_TO_EDGE = "RELOCATE_TO_EDGE"
    REACQUIRE_TABLE = "REACQUIRE_TABLE"
    NEXT_TABLE = "NEXT_TABLE"
    AVOID_OBSTACLE = "AVOID_OBSTACLE"
    RETURN_HOME = "RETURN_HOME"
    ERROR_RECOVERY = "ERROR_RECOVERY"
    DONE = "DONE"
    NO_PROGRESS_RECOVERY = "NO_PROGRESS_RECOVERY"


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
    desired_vision_stage: str = ""
    desired_vision_mode: str = ""
    confirmed_vision_stage: str = ""
    confirmed_vision_mode: str = ""
    vision_confirm_source: str = ""
    current_edge_id: str = "front"
    edge_visit_order: List[str] = field(default_factory=lambda: ["front", "right", "back", "left"])
    edge_visit_index: int = 0
    edge_transition_count: int = 0
    table_cycle_count: int = 0
    relocate_turn_sign: int = 1
    slide_direction_sign: int = 1
    locked_edge_id: str = ""
    locked_edge_line: Optional[Dict[str, float]] = None
    locked_roi: Optional[list] = None
    locked_yaw_err: Optional[float] = None
    locked_dist_err: Optional[float] = None
    locked_edge_conf: Optional[float] = None
    locked_obs_seq: Optional[int] = None
    slide_ref_ready: bool = False
    slide_ref_yaw_err: Optional[float] = None
    slide_ref_dist_err: Optional[float] = None
    slide_ref_edge_conf: Optional[float] = None
    slide_ref_roi: Optional[list] = None
    slide_ref_seq: Optional[int] = None
    slide_ref_samples: List[Dict[str, object]] = field(default_factory=list)
    slide_ref_last_sample_key: str = ""
    handoff_state: str = ""
    last_edge_quality: Dict[str, object] = field(default_factory=dict)

    last_task_cmd: Optional[TaskCmd] = None
    last_table_obs: Optional[TableEdgeObs] = None
    last_target_obs: Optional[TargetObs] = None
    last_home_obs: Optional[HomeTagObs] = None
    last_car_state: Optional[CarState] = None

    last_table_bbox_xyxy: Optional[List[float]] = None
    last_table_center_x_norm: Optional[float] = None
    last_table_center_y_norm: Optional[float] = None
    last_table_seen_ts: float = 0.0
    last_table_seen_frame: Optional[int] = None
    last_table_side: str = "unknown"
    last_table_touch_left: bool = False
    last_table_touch_right: bool = False
    last_table_touch_bottom: bool = False
    
    current_search_direction_source: str = "default"
    current_search_direction_reason: str = "no_memory"

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
    table_approach_warmup_fresh_obs_count: int = 0
    table_approach_warmup_plane_seen_count: int = 0
    table_approach_warmup_last_obs_key: str = ""
    table_dock_phase: str = ""
    table_dock_phase_since_mono: float = 0.0
    table_micro_adjust_count: int = 0
    table_stop_sent: bool = False
    final_lock_ready_window: List[Dict[str, object]] = field(default_factory=list)
    final_lock_last_obs_key: str = ""
    final_lock_same_obs_reuse_count: int = 0
    final_lock_consecutive_lost_count: int = 0
    final_lock_last_transition_reason: str = ""
    docking_done_printed: bool = False
    approach_aligned_frames: int = 0
    approach_realign_frames: int = 0
    align_hysteresis_last_obs_key: str = ""
    approach_hysteresis_last_obs_key: str = ""
    table_motion_pending_transition_reason: str = ""
    edge_hard_yaw_frames: int = 0
    edge_hard_yaw_since_mono: float = 0.0
    control_phase: str = "SEARCH_SCAN"
    control_phase_since_mono: float = 0.0
    bbox_valid_streak: int = 0
    bbox_centered_streak: int = 0
    edge_trusted_streak: int = 0
    edge_yaw_ema: Optional[float] = None
    edge_handoff_started_mono: float = 0.0
    edge_handoff_complete: bool = False
    edge_handoff_timeout: bool = False
    approach_commit_active: bool = False
    last_forward_cmd_mono: float = 0.0
    forward_commit_until_mono: float = 0.0
    forward_commit_reason: str = ""
    last_edge_yaw_cmd: float = 0.0
    last_edge_good_mono: float = 0.0
    zero_cmd_started_mono: float = 0.0
    bbox_track_entered_mono: float = 0.0
    bbox_track_last_exit_reason: str = ""
    edge_conf_score: float = 0.0
    edge_readiness_score: float = 0.0
    edge_readiness_last_update_mono: float = 0.0
    edge_readiness_level: str = ""
    edge_handoff_entered_mono: float = 0.0
    last_good_table_obs_mono: float = 0.0
    last_good_table_obs_summary: Dict[str, object] = field(default_factory=dict)
    perception_dropout_hold_active: bool = False
    perception_dropout_hold_started_mono: float = 0.0
    perception_dropout_hold_reason: str = ""
    motion_intent_type: str = ""
    yaw_owner: str = ""
    forward_owner: str = ""
    lateral_owner: str = ""
    arbitration_reason: str = ""
    motion_class: str = ""
    stop_class: str = "none"
    blocked_by: str = ""
    fov_guard_level: str = "none"
    fov_guard_reason: str = ""
    zero_escape_reason: str = ""
    near_table_latched: bool = False
    near_table_latched_mono: float = 0.0
    final_depth_latched: bool = False
    final_depth_latched_mono: float = 0.0
    final_yaw_align_active: bool = False
    final_locked: bool = False
    last_good_edge_yaw_cmd: float = 0.0
    last_good_edge_yaw_mono: float = 0.0
    last_good_near_depth_mono: float = 0.0
    near_table_latch_reason: str = ""
    final_depth_latch_reason: str = ""
    final_lock_reason: str = ""
    near_depth_stable_frames: int = 0
    near_dist_stable_frames: int = 0
    final_depth_stable_frames: int = 0
    final_yaw_aligned_frames: int = 0
    final_yaw_align_start_mono: float = 0.0
    final_yaw_initially_small: bool = False
    final_yaw_realign_count: int = 0
    bbox_fov_violation_streak: int = 0
    bbox_lost_since_mono: float = 0.0
    bbox_lost_hold_active: bool = False
    last_bbox_yaw_cmd: float = 0.0
    search_wz_sign_latched: int = 0
    search_wz_latch_until_mono: float = 0.0
    target_found_frames: int = 0
    target_lost_frames: int = 0
    target_lock_frames: int = 0
    tag_lost_frames: int = 0
    tag_arrived_frames: int = 0
    avoid_clear_frames: int = 0
    avoid_retry_count: int = 0
    no_progress_recovery_count: int = 0
    edge_slide_relock_attempts: int = 0

    table_loss_since_mono: float = 0.0
    target_loss_since_mono: float = 0.0
    tag_loss_since_mono: float = 0.0
    target_stable_since_mono: float = 0.0
    min_dist_seen: float = 999.0
    dist_progress_last_refreshed_mono: float = 0.0
    dist_missing_started_mono: float = 0.0
    target_center_history: List[Dict[str, float]] = field(default_factory=list)
    target_obs_window: List[Dict[str, object]] = field(default_factory=list)
    target_last_center_jitter: float = 0.0
    target_last_lost_reason: str = ""
    target_last_transition_reason: str = ""
    task_slide_entries_count: int = 0
    task_target_confirm_count: int = 0
    task_target_locked_count: int = 0
    task_warning_history: List[str] = field(default_factory=list)
    task_done_summary_emitted: bool = False

    grasp_substate: str = ""
    grasp_result: Optional[Dict] = None
    grasp_status: str = ""
    grasp_reason: str = ""
    grasp_reposition_proposal: Optional[Dict] = None
    grasp_reposition_start_mono: float = 0.0
    pre_arm_stop_settle_start_mono: float = 0.0
    grasp_retry_count: int = 0
    arm_response: Optional[object] = None
    grasp_timeout_mono: float = 0.0
    grasp_verify_reported: bool = False

    def clear_motion_counters(self):
        self.table_found_frames = 0
        self.table_lost_frames = 0
        self.table_lock_frames = 0
        self.table_approach_warmup_fresh_obs_count = 0
        self.table_approach_warmup_plane_seen_count = 0
        self.table_approach_warmup_last_obs_key = ""
        self.table_dock_phase = ""
        self.table_dock_phase_since_mono = 0.0
        self.table_micro_adjust_count = 0
        self.table_stop_sent = False
        self.final_lock_ready_window.clear()
        self.final_lock_last_obs_key = ""
        self.final_lock_same_obs_reuse_count = 0
        self.final_lock_consecutive_lost_count = 0
        self.final_lock_last_transition_reason = ""
        self.docking_done_printed = False
        self.approach_aligned_frames = 0
        self.approach_realign_frames = 0
        self.align_hysteresis_last_obs_key = ""
        self.approach_hysteresis_last_obs_key = ""
        self.table_motion_pending_transition_reason = ""
        self.edge_hard_yaw_frames = 0
        self.edge_hard_yaw_since_mono = 0.0
        self.control_phase = "SEARCH_SCAN"
        self.control_phase_since_mono = 0.0
        self.bbox_valid_streak = 0
        self.bbox_centered_streak = 0
        self.edge_trusted_streak = 0
        self.edge_yaw_ema = None
        self.edge_handoff_started_mono = 0.0
        self.edge_handoff_complete = False
        self.edge_handoff_timeout = False
        self.approach_commit_active = False
        self.last_forward_cmd_mono = 0.0
        self.forward_commit_until_mono = 0.0
        self.forward_commit_reason = ""
        self.last_edge_yaw_cmd = 0.0
        self.last_edge_good_mono = 0.0
        self.zero_cmd_started_mono = 0.0
        self.bbox_track_entered_mono = 0.0
        self.bbox_track_last_exit_reason = ""
        self.edge_conf_score = 0.0
        self.edge_readiness_score = 0.0
        self.edge_readiness_last_update_mono = 0.0
        self.edge_readiness_level = ""
        self.edge_handoff_entered_mono = 0.0
        self.last_good_table_obs_mono = 0.0
        self.last_good_table_obs_summary.clear()
        self.perception_dropout_hold_active = False
        self.perception_dropout_hold_started_mono = 0.0
        self.perception_dropout_hold_reason = ""
        self.motion_intent_type = ""
        self.yaw_owner = ""
        self.forward_owner = ""
        self.lateral_owner = ""
        self.arbitration_reason = ""
        self.motion_class = ""
        self.stop_class = "none"
        self.blocked_by = ""
        self.fov_guard_level = "none"
        self.fov_guard_reason = ""
        self.zero_escape_reason = ""
        self.near_table_latched = False
        self.near_table_latched_mono = 0.0
        self.final_depth_latched = False
        self.final_depth_latched_mono = 0.0
        self.final_yaw_align_active = False
        self.final_locked = False
        self.last_good_edge_yaw_cmd = 0.0
        self.last_good_edge_yaw_mono = 0.0
        self.last_good_near_depth_mono = 0.0
        self.near_table_latch_reason = ""
        self.final_depth_latch_reason = ""
        self.final_lock_reason = ""
        self.near_depth_stable_frames = 0
        self.near_dist_stable_frames = 0
        self.final_depth_stable_frames = 0
        self.final_yaw_aligned_frames = 0
        self.final_yaw_align_mono = 0.0
        self.final_yaw_align_start_mono = 0.0
        self.final_yaw_initially_small = False
        self.final_yaw_realign_count = 0
        self.bbox_fov_violation_streak = 0
        self.bbox_lost_since_mono = 0.0
        self.bbox_lost_hold_active = False
        self.search_wz_sign_latched = 0
        self.search_wz_latch_until_mono = 0.0
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
        self.min_dist_seen = 999.0
        self.dist_progress_last_refreshed_mono = 0.0
        self.dist_missing_started_mono = 0.0
        self.target_center_history.clear()
        self.target_obs_window.clear()
        self.target_last_center_jitter = 0.0
        self.target_last_lost_reason = ""
        self.target_last_transition_reason = ""
        self.grasp_retry_count = 0
        self.grasp_substate = ""
        self.grasp_verify_reported = False

    def clear_perception_cache(self):
        self.last_table_obs = None
        self.last_target_obs = None
        self.last_home_obs = None
        self.last_table_bbox_xyxy = None
        self.last_table_center_x_norm = None
        self.last_table_center_y_norm = None
        self.last_table_seen_ts = 0.0
        self.last_table_seen_frame = None
        self.last_table_side = "unknown"
        self.last_table_touch_left = False
        self.last_table_touch_right = False
        self.last_table_touch_bottom = False

    def reset_edge_plan(self):
        self.current_edge_id = self.edge_visit_order[0] if self.edge_visit_order else "front"
        self.edge_visit_index = 0
        self.edge_transition_count = 0
        self.relocate_turn_sign = 1
        self.slide_direction_sign = 1
        self.locked_edge_id = ""
        self.locked_edge_line = None
        self.locked_roi = None
        self.locked_yaw_err = None
        self.locked_dist_err = None
        self.locked_edge_conf = None
        self.locked_obs_seq = None
        self.slide_ref_ready = False
        self.slide_ref_yaw_err = None
        self.slide_ref_dist_err = None
        self.slide_ref_edge_conf = None
        self.slide_ref_roi = None
        self.slide_ref_seq = None
        self.slide_ref_samples.clear()
        self.slide_ref_last_sample_key = ""
        self.handoff_state = ""
        self.last_edge_quality.clear()

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
        self.desired_vision_stage = ""
        self.desired_vision_mode = ""
        self.confirmed_vision_stage = ""
        self.confirmed_vision_mode = ""
        self.vision_confirm_source = ""
        self.task_start_wall_ts = 0.0
        self.resume_state = None
        self.last_safety_reason = ""
        self.last_fail_reason = ""
        self.last_enter_reason = ""
        self.table_cycle_count = 0
        self.avoid_retry_count = 0
        self.no_progress_recovery_count = 0
        self.edge_slide_relock_attempts = 0
        self.vision_req_fail_streak = 0
        self.task_slide_entries_count = 0
        self.task_target_confirm_count = 0
        self.task_target_locked_count = 0
        self.task_warning_history.clear()
        self.task_done_summary_emitted = False
        self.grasp_substate = ""
        self.grasp_result = None
        self.grasp_status = ""
        self.grasp_reason = ""
        self.grasp_reposition_proposal = None
        self.grasp_reposition_start_mono = 0.0
        self.pre_arm_stop_settle_start_mono = 0.0
        self.grasp_retry_count = 0
        self.arm_response = None
        self.grasp_timeout_mono = 0.0
        self.grasp_verify_reported = False
        self.reset_edge_plan()
        self.clear_perception_cache()
        self.clear_motion_counters()

    @property
    def active_vision_stage(self) -> str:
        return self.confirmed_vision_stage

    @active_vision_stage.setter
    def active_vision_stage(self, val: str):
        self.confirmed_vision_stage = val

    @property
    def active_vision_mode(self) -> str:
        return self.confirmed_vision_mode

    @active_vision_mode.setter
    def active_vision_mode(self, val: str):
        self.confirmed_vision_mode = val
