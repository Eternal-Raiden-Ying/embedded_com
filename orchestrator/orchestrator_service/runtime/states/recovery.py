#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ...config.schema import CarMotionConfig, ControlThresholds
from ...control.types import DockingControlConfig
from ...ipc.protocol import (
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
from ...bridge.arm_protocol import parse_arm_response
from ...utils.grasp_utils import grasp_to_pose_params
from ...utils.target_utils import target_to_class_id
from ..common import monotonic_ts
from ..context import RuntimeContext, State
from ..controller import MotionController, MotionDecision
from ..control_authority import decide_table_control_authority
from ..core_types import (
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


class RecoveryMixin:
    def _should_send_vision_idle_after_task(self) -> bool:
        if bool(getattr(self.cfg, "task_done_shutdown_vision", False)):
            return True
        return not bool(getattr(self.cfg, "keep_vision_alive_after_task", True))

    def _finish_task_to_idle(self, reason: str) -> None:
        if self._should_send_vision_idle_after_task():
            self._queue_vision_req(make_vision_idle(session_id=self.ctx.active_session_id, epoch=self.ctx.active_epoch), force=True)
        else:
            self._log(
                "info",
                "task reached idle; keep vision hot "
                f"reason={reason} session={self.ctx.active_session_id} epoch={self.ctx.active_epoch}",
            )
        self._transition(State.IDLE, reason)

    def _tick_no_progress_recovery(self) -> MotionDecision:
        if self._state_elapsed() < float(self.cfg.dock_retry_backoff_s):
            return self.controller.leave_edge_cmd()
        self._transition(State.SEARCH_TABLE, "停靠重试完成，重新搜索桌边")
        return self.controller.search_table_cmd(*self._get_memory_search_params())

    def _tick_leave_edge(self) -> MotionDecision:
        if self._state_elapsed() < float(self.cfg.leave_edge_backoff_s):
            return self.controller.leave_edge_cmd()
        self._transition(State.RELOCATE_TO_EDGE, f"准备重定位到边 {self.ctx.current_edge_id}")
        return self.controller.relocate_cmd(turn_sign=self.ctx.relocate_turn_sign)

    def _tick_relocate_to_edge(self) -> MotionDecision:
        if self._state_elapsed() < float(self.cfg.relocate_turn_s):
            return self.controller.relocate_cmd(turn_sign=self.ctx.relocate_turn_sign)
        self._transition(State.REACQUIRE_TABLE, f"开始重捕获边 {self.ctx.current_edge_id}")
        return self.controller.search_table_cmd(*self._get_memory_search_params())

    def _tick_reacquire_table(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        obs = self._fresh_table_obs()
        if self._table_visible(obs):
            self.ctx.table_found_frames += 1
            if self.ctx.table_found_frames >= int(self.cfg.table_found_frames_to_approach):
                self._transition(State.EDGE_ADJUST, f"已重捕获边 {self.ctx.current_edge_id}")
                return self._tick_edge_adjust()
        else:
            self.ctx.table_found_frames = 0
        if self._state_elapsed() >= float(self.cfg.reacquire_timeout_s):
            self.ctx.last_fail_reason = f"重捕获边 {self.ctx.current_edge_id} 超时"
            self._transition(State.NEXT_TABLE, self.ctx.last_fail_reason)
            return self.controller.next_table_cmd(turn_sign=self.ctx.relocate_turn_sign)
        return self.controller.search_table_cmd(*self._get_memory_search_params())

    def _tick_next_table(self) -> MotionDecision:
        if self._state_elapsed() >= float(self.cfg.next_table_dwell_s):
            self.ctx.table_cycle_count += 1
            self.ctx.reset_edge_plan()
            self._transition(State.SEARCH_TABLE, "切换到下一张桌后重新搜索")
            return self.controller.search_table_cmd(*self._get_memory_search_params())
        return self.controller.next_table_cmd(turn_sign=self.ctx.relocate_turn_sign)

    def _tick_avoid_obstacle(self) -> MotionDecision:
        obstacle = self._extract_obstacle_signal()
        if self._state_elapsed() >= float(self.cfg.avoid_timeout_s):
            self.ctx.last_fail_reason = "避障超时"
            self._enter_error_recovery(self.ctx.last_fail_reason, tts_text="避障失败，已停止", interrupt_tts=True)
            return self.controller.stop_cmd("ERROR_RECOVERY", brake=True)
        if self.ctx.avoid_retry_count > int(self.cfg.avoid_retry_limit):
            self.ctx.last_fail_reason = "连续避障失败次数过多"
            self._enter_error_recovery(self.ctx.last_fail_reason, tts_text="连续避障失败，已停止", interrupt_tts=True)
            return self.controller.stop_cmd("ERROR_RECOVERY", brake=True)
        if obstacle.active:
            self.ctx.avoid_clear_frames = 0
            return self.controller.avoid_cmd(obstacle.best_turn_dir)
        self.ctx.avoid_clear_frames += 1
        if self.ctx.avoid_clear_frames >= int(self.cfg.avoid_clear_frames_to_resume):
            resume_state = self.ctx.resume_state or State.SEARCH_TABLE
            self._transition(resume_state, "障碍清除，恢复主任务")
        return self.controller.stop_cmd("AVOID_OBSTACLE")

    def _tick_return_home(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        obs = self._fresh_home_obs()
        if obs is None or not obs.found:
            self.ctx.tag_lost_frames += 1
            self._start_loss_timer("tag_loss_since_mono")
            lost_frames_ok = self.ctx.tag_lost_frames >= int(self.cfg.tag_lost_frames_to_search)
            lost_hold_ok = self._loss_elapsed(self.ctx.tag_loss_since_mono) >= float(self.cfg.return_lost_hold_s)
            min_dwell_ok = self._state_elapsed() >= float(self.cfg.return_min_dwell_s)
            if lost_frames_ok and lost_hold_ok and min_dwell_ok:
                return self.controller.search_table_cmd(*self._get_memory_search_params())
            return self.controller.return_hold_cmd()
        self.ctx.tag_lost_frames = 0
        self.ctx.tag_loss_since_mono = 0.0
        if obs.distance_m is not None and float(obs.distance_m) <= float(self.cfg.return_done_distance_m):
            self.ctx.tag_arrived_frames += 1
            if self.ctx.tag_arrived_frames >= int(self.cfg.tag_arrived_frames_to_stop):
                self._transition(State.DONE, "已返回起点")
                self._queue_tts("已返回起点")
                return self.controller.stop_cmd("DONE")
        else:
            self.ctx.tag_arrived_frames = 0
        return self.controller.return_cmd(obs)

    def _tick_error_recovery(self) -> MotionDecision:
        if self._state_elapsed() >= float(self.cfg.error_recovery_hold_s):
            self._finish_task_to_idle("错误恢复完成，回到空闲")
        return self.controller.stop_cmd("ERROR_RECOVERY", brake=True)

    def _tick_done(self) -> MotionDecision:
        if self._state_elapsed() >= float(self.cfg.done_hold_s):
            self._finish_task_to_idle("任务完成，回到空闲")
        return self.controller.stop_cmd("DONE")

    def _handle_table_loss(self, reason: str, fallback_state: State, hold_mode: str) -> MotionDecision:
        self.ctx.table_lost_frames += 1
        self._start_loss_timer("table_loss_since_mono")
        required_lost_frames = int(self.cfg.table_lost_frames_to_reacquire)
        if fallback_state == State.SEARCH_TABLE and bool(getattr(self.cfg, "yolo_table_control_enable", True)):
            required_lost_frames = max(
                required_lost_frames,
                int(getattr(self.cfg, "yolo_table_lost_to_search_frames", required_lost_frames) or required_lost_frames),
            )
        lost_frames_ok = self.ctx.table_lost_frames >= required_lost_frames
        lost_hold_ok = self._loss_elapsed(self.ctx.table_loss_since_mono) >= float(self.cfg.table_loss_hold_s)
        min_dwell_ok = self._state_elapsed() >= float(self.cfg.approach_min_dwell_s)
        if lost_frames_ok and lost_hold_ok and min_dwell_ok:
            if fallback_state == State.SEARCH_TABLE and (bool(getattr(self.ctx, "near_table_latched", False)) or bool(getattr(self.ctx, "final_depth_latched", False))):
                # Keep holding final slow stop or near edge approach, do not fall back to search.
                # compatibility only; table docking semantic state is DockingStage
                return self.controller.stop_cmd(hold_mode)
            self._transition(fallback_state, reason)
            decision = self.controller.search_table_cmd(*self._get_memory_search_params())
            if fallback_state == State.SEARCH_TABLE:
                decision.control_summary.update(
                    {
                        "control_source": "local_rotate_search",
                        "table_lost_search_active": True,
                        "table_lost_search_elapsed_s": 0.0,
                        "table_lost_search_timeout": False,
                        "search_table_stale_gate_bypass": True,
                    }
                )
            return decision
        decision = self.controller.stop_cmd(hold_mode)
        decision.control_summary["yolo_table_lost_to_search_frames"] = int(required_lost_frames)
        decision.control_summary["search_blocked_by_yolo_valid"] = False
        decision.control_summary["yolo_lost_frames_before_search"] = int(self.ctx.table_lost_frames)
        decision.control_summary["table_lost_search_active"] = False
        decision.control_summary["table_lost_search_timeout"] = False
        return decision

    def _enter_no_progress_recovery_or_next(self, reason: str) -> MotionDecision:
        self.ctx.last_fail_reason = reason
        if self.ctx.no_progress_recovery_count < int(self.cfg.dock_retry_limit):
            self.ctx.no_progress_recovery_count += 1
            self._transition(State.NO_PROGRESS_RECOVERY, f"{reason}，准备重试第 {self.ctx.no_progress_recovery_count} 次")
            return self.controller.leave_edge_cmd()
        if bool(getattr(self.cfg, "multi_table_enabled", False)):
            self._transition(State.NEXT_TABLE, reason)
            return self.controller.next_table_cmd(turn_sign=self.ctx.relocate_turn_sign)
        self._transition(State.SEARCH_TABLE, f"{reason}，单桌模式重新搜索桌边")
        return self.controller.search_table_cmd(*self._get_memory_search_params())
