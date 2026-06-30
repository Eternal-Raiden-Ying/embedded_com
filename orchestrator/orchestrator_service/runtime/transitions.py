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


class TransitionsMixin:
    def _transition(self, new_state: State, reason: str):
        old_state = self.ctx.state
        if old_state == State.AT_TABLE_EDGE and getattr(self.ctx, "final_locked", False):
            if new_state in {State.SEARCH_TABLE, State.REACQUIRE_TABLE, State.NO_PROGRESS_RECOVERY, State.LEAVE_EDGE, State.ERROR_RECOVERY}:
                self._log("warn", f"Bypassing transition from AT_TABLE_EDGE to {new_state.value} due to final_locked=True")
                return
        if old_state == new_state:
            self.ctx.last_enter_reason = reason
            return
        preserve_target_debounce = (
            old_state in {State.EDGE_SLIDE_SEARCH, State.TARGET_CONFIRM, State.TARGET_LOCKED}
            and new_state in {State.TARGET_CONFIRM, State.TARGET_LOCKED, State.FREEZE_BASE}
        )
        target_debounce = self._target_debounce_snapshot() if preserve_target_debounce else {}
        self.ctx.target_last_transition_reason = str(reason or "")
        snapshot = self._build_transition_snapshot(old_state, new_state, reason)
        self._log("info", f"状态切换 {old_state.value} -> {new_state.value} ({reason})")
        self.ctx.prev_state = old_state
        self.ctx.state = new_state
        self.ctx.state_enter_mono = monotonic_ts()
        self.ctx.state_enter_wall_ts = time.time()
        self.ctx.last_enter_reason = reason
        self.last_transition_snapshot = snapshot
        if new_state == State.EDGE_SLIDE_SEARCH:
            self.ctx.task_slide_entries_count += 1
        elif new_state == State.TARGET_CONFIRM:
            self.ctx.task_target_confirm_count += 1
        elif new_state == State.TARGET_LOCKED:
            self.ctx.task_target_locked_count += 1
            self._log("info", "target_locked_enter")
        if new_state == State.DONE and self.ctx.last_fail_reason:
            warning = str(self.ctx.last_fail_reason).strip()
            if warning and warning not in self.ctx.task_warning_history:
                self.ctx.task_warning_history.append(warning)
        self.ctx.clear_motion_counters()
        if new_state in {State.YOLO_APPROACH, State.EDGE_ADJUST, State.FINAL_SLOW_STOP}:
            self.ctx.min_dist_seen = 999.0
            self.ctx.dist_progress_last_refreshed_mono = monotonic_ts()
            self.ctx.dist_missing_started_mono = 0.0
        if preserve_target_debounce:
            self._restore_target_debounce_snapshot(target_debounce)
        if self.transition_observer is not None:
            try:
                self.transition_observer(old_state.value, new_state.value, reason)
            except Exception:
                pass
        self._on_enter_state(new_state)

    def _frames_to_ms(self, frames: int) -> int:
        try:
            tick_hz = max(1.0, float(getattr(self.cfg, "tick_hz", 10.0)))
        except Exception:
            tick_hz = 10.0
        return int(round((max(0, int(frames)) / tick_hz) * 1000.0))

    def _transition_stable_ms(self, old_state: State) -> int:
        if old_state == State.SEARCH_TABLE:
            return self._frames_to_ms(self.ctx.table_found_frames)
        if old_state == State.EDGE_ADJUST:
            return self._frames_to_ms(self.ctx.approach_aligned_frames)
        if old_state == State.FINAL_SLOW_STOP:
            return self._frames_to_ms(self.ctx.table_lock_frames)
        if old_state in {State.TARGET_CONFIRM, State.TARGET_LOCKED}:
            return self._target_stable_ms()
        return int(round(self._state_elapsed() * 1000.0))

    def _transition_lost_ms(self, old_state: State) -> int:
        if old_state in TABLE_APPROACH_STATES and self.ctx.table_loss_since_mono:
            return int(round(self._loss_elapsed(self.ctx.table_loss_since_mono) * 1000.0))
        if old_state in TARGET_SEARCH_STATES and self.ctx.target_loss_since_mono:
            return int(round(self._loss_elapsed(self.ctx.target_loss_since_mono) * 1000.0))
        return 0

    def _transition_condition_snapshot(self, old_state: State, new_state: State) -> Dict[str, Any]:
        return {
            "old_state": old_state.value,
            "new_state": new_state.value,
            "state_elapsed_ms": int(round(self._state_elapsed() * 1000.0)),
            "table_found_frames": int(self.ctx.table_found_frames),
            "table_found_frames_to_approach": int(self.cfg.table_found_frames_to_approach),
            "coarse_align_frames": int(self.ctx.approach_aligned_frames),
            "coarse_align_frames_to_advance": int(self.cfg.coarse_align_frames_to_advance),
            "table_lock_frames": int(self.ctx.table_lock_frames),
            "final_lock_frames_to_arrive": int(self.cfg.final_lock_frames_to_arrive),
            "required_lock_count": int(self._required_lock_count()),
            "final_lock_required_ready_obs": int(self._final_lock_required_ready_obs()),
            "final_lock_window_ms": int(self._final_lock_window_ms()),
            "final_lock_max_consecutive_lost": int(self._final_lock_max_consecutive_lost()),
            "final_lock_soft_stale_hold": bool(getattr(self.cfg, "final_lock_soft_stale_hold", True)),
            "target_found_frames": int(self.ctx.target_found_frames),
            "target_found_frames_to_confirm": int(self.cfg.target_found_frames_to_confirm),
            "target_lost_frames": int(self.ctx.target_lost_frames),
            "target_confirm_lost_frames": int(self.cfg.target_confirm_lost_frames),
            "confirm_conf_th": float(self.cfg.target_confirm_conf_th),
            "confirm_min_ms": int(round(float(self.cfg.target_confirm_min_s) * 1000.0)),
            "confirm_timeout_ms": int(round(float(self.cfg.target_confirm_timeout_s) * 1000.0)),
            "confirm_lost_hold_ms": int(round(float(self.cfg.target_confirm_lost_hold_s) * 1000.0)),
            "min_bbox_area": float(self.cfg.target_confirm_min_bbox_area),
            "lock_conf_th": float(self.cfg.target_lock_conf_th),
            "lock_stable_ms": int(round(float(self.cfg.target_lock_stable_s) * 1000.0)),
            "center_jitter_th": float(self.cfg.target_lock_center_jitter_th),
            "locked_lost_hold_ms": int(round(float(self.cfg.target_lock_lost_hold_s) * 1000.0)),
            "freeze_after_locked_ms": int(round(float(self.cfg.target_locked_freeze_after_s) * 1000.0)),
            "edge_settle_ms": int(round(float(self.cfg.edge_settle_s) * 1000.0)),
            "table_stable_frames": int(getattr(self.cfg, "table_stable_frames", self.cfg.final_lock_frames_to_arrive)),
            "table_settle_ms": int(round(float(getattr(self.cfg, "table_settle_s", 0.3)) * 1000.0)),
            "table_stop_margin_m": float(getattr(self.cfg, "table_stop_margin_m", 0.05)),
            "table_max_micro_adjust": int(getattr(self.cfg, "table_max_micro_adjust", 4)),
            "final_lock_enabled": self._table_final_lock_enabled(),
            "micro_adjust_enabled": self._table_micro_adjust_enabled(),
            "final_lock_enter_dist_th_m": float(getattr(self.cfg, "final_lock_enter_dist_th_m", 0.08)),
            "final_lock_enter_yaw_th_rad": float(getattr(self.cfg, "final_lock_enter_yaw_th_rad", 0.10)),
            "search_target_init_hold_ms": int(round(float(self.cfg.search_target_init_hold_s) * 1000.0)),
            "target_lock_settle_ms": int(round(float(self.cfg.target_lock_settle_s) * 1000.0)),
            "freeze_settle_ms": int(round(float(self.cfg.freeze_settle_s) * 1000.0)),
            "approach_timeout_ms": int(round(float(self.cfg.approach_timeout_s) * 1000.0)),
            "target_search_timeout_ms": int(round(float(self.cfg.target_search_timeout_s) * 1000.0)),
            "target_window_ms": int(round(float(getattr(self.cfg, "target_confirm_window_s", 1.5) or 1.5) * 1000.0)),
            "confirm_found_ratio_th": float(getattr(self.cfg, "target_confirm_found_ratio_th", 0.5) or 0.5),
            "lock_found_ratio_th": float(getattr(self.cfg, "target_lock_found_ratio_th", 0.6) or 0.6),
        }

    def _build_transition_snapshot(self, old_state: State, new_state: State, reason: str) -> Dict[str, Any]:
        table_obs = self.ctx.last_table_obs
        from .perception_semantics import build_table_perception_semantics
        sem = build_table_perception_semantics(table_obs, self.cfg)

        decision = getattr(self, "last_decision", None)
        summary = decision.control_summary if decision is not None else None
        if summary is None:
            summary = {}

        return {
            "ts": time.time(),
            "state": new_state.value,
            "control_source": summary.get("control_source") or "stop",
            "control_intent": summary.get("control_intent") or "stop",
            "table_bbox_current_found": bool(sem.table_bbox_current_found),
            "table_bbox_control_valid": bool(sem.table_bbox_control_valid),
            "edge_geometry_valid": bool(sem.edge_geometry_valid),
            "edge_trusted": bool(sem.edge_trusted),
            "allow_forward": bool(summary.get("allow_forward", False)),
            "allow_rotate": bool(summary.get("allow_rotate", False)),
            "forward_block_reason": str(summary.get("forward_block_reason") or ""),
            "rotate_block_reason": str(summary.get("rotate_block_reason") or ""),
            "stop_reason": str(summary.get("stop_reason") or ""),
            "event": "state_transition",
            "previous_state": old_state.value,
            "next_state": new_state.value,
            "reason": str(reason or ""),
            "transition_reason": str(reason or ""),
            "target_found": bool(getattr(self.ctx.last_target_obs, "found", False)),
            "target_cls": str(getattr(self.ctx.last_target_obs, "matched_cls", None) or getattr(self.ctx.last_target_obs, "target", "") or "") if self.ctx.last_target_obs is not None else "",
            "target_conf": self._target_conf_value(self.ctx.last_target_obs) if self.ctx.last_target_obs is not None else None,
            "target_center_x_norm": self._target_lateral_center_x(self.ctx.last_target_obs) if hasattr(self, "_target_lateral_center_x") else None,
            "target_err_x": self._target_lateral_error_x(self.ctx.last_target_obs) if hasattr(self, "_target_lateral_error_x") else None,
            "target_lateral_align_active": False,
            "target_lateral_align_reason": str(getattr(self.ctx, "target_lateral_align_reason", "") or ""),
            "target_lateral_vy_cmd": float(getattr(self.ctx, "target_lateral_vy_cmd", 0.0) or 0.0),
            "target_lateral_stable_count": int(getattr(self.ctx, "target_lateral_stable_count", 0) or 0),
            "target_locked": bool(getattr(self.ctx, "target_locked", False)),
            "grasp_request_sent": bool(new_state == State.GRASP),
            "grasp_dry_run": bool(new_state == State.DONE and str(reason or "").strip() == "grasp_request_dry_run"),
            "hard_stop_barrier_active": bool(
                new_state == State.AT_TABLE_EDGE
                and float(getattr(self.ctx, "hard_stop_barrier_until_mono", 0.0) or 0.0) > monotonic_ts()
            ),
            "hard_stop_barrier_reason": str(getattr(self.ctx, "hard_stop_barrier_reason", "") or ""),
            "hard_stop_barrier_left_ms": max(
                0,
                int(round((float(getattr(self.ctx, "hard_stop_barrier_until_mono", 0.0) or 0.0) - monotonic_ts()) * 1000.0)),
            ),
        }

    def _on_enter_state(self, state: State):
        if state == State.IDLE:
            self.reset_task_runtime("enter_idle", keep_session=False)
            self.ctx.desired_vision_stage = ""
            self.ctx.desired_vision_mode = ""
            self.ctx.confirmed_vision_stage = ""
            self.ctx.confirmed_vision_mode = ""
            self.ctx.resume_state = None
            return
        if state == State.AT_TABLE_EDGE:
            duration_s = min(1.0, max(0.5, float(getattr(self.cfg, "edge_settle_s", 0.8) or 0.8)))
            self.ctx.hard_stop_barrier_until_mono = monotonic_ts() + duration_s
            self.ctx.hard_stop_barrier_reason = "at_table_edge_entry_sstop_barrier"
        if state == State.SEARCH_TARGET_INIT:
            self._log("info", "target_search_enter")
            self._log("info", "docking_final_latch_frozen_for_target_search")
            self.ctx.final_locked = False
            self.ctx.final_depth_latched = False
            self.ctx.final_roi_mode_latched = False
            self.ctx.final_edge_mode_latched = False
            self.ctx.close_range_latched = False
            self.ctx.final_yaw_align_active = False
            self.ctx.final_lock_reason = ""
            self.ctx.final_lock_last_transition_reason = ""
            self._reset_slide_ref_handoff()
        if state == State.SEARCH_TABLE:
            self.reset_edge_tracking("enter_search_table")
            self.reset_target_tracking("enter_search_table")
        elif state == State.FREEZE_BASE:
            self._log("info", "freeze_base_enter")
        elif state in {State.NO_PROGRESS_RECOVERY, State.LEAVE_EDGE, State.NEXT_TABLE}:
            reason = f"enter_{state.value.lower()}"
            self.reset_edge_tracking(reason)
            self.reset_target_tracking(reason)
            self.reset_slide_reference(reason)
        elif state == State.GRASP:
            self._log("info", "grasp_enter")
            self.ctx.grasp_substate = "AWAITING_RESPOND"
            self.ctx.grasp_result = None
            self.ctx.grasp_status = ""
            self.ctx.grasp_reason = ""
            self.ctx.grasp_reposition_proposal = None
            self.ctx.grasp_reposition_start_mono = 0.0
            self.ctx.grasp_retry_count = 0
            self.ctx.arm_response = None
            self.ctx.grasp_timeout_mono = monotonic_ts() + _GRASP_RESPOND_TIMEOUT_S
            self.ctx.grasp_verify_reported = False
            self._log("info", "grasp_remote_request_sent")
        elif state == State.DONE:
            if self.ctx.last_fail_reason:
                warning = str(self.ctx.last_fail_reason).strip()
                if warning and warning not in self.ctx.task_warning_history:
                    self.ctx.task_warning_history.append(warning)
            self.ctx.last_fail_reason = ""
            self.ctx.task_done_summary_emitted = False
        req = self._active_req_payload()
        if req is not None:
            self._queue_vision_req(req, force=True)
