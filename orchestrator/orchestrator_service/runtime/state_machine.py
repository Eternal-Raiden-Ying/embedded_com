#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from dataclasses import dataclass
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


_GRASP_RESPOND_TIMEOUT_S = 5.0
_GRASP_RESULT_TIMEOUT_S = 15.0
_GRASP_ARM_TIMEOUT_S = 10.0
_GRASP_RETRY_LIMIT = 3
_GRASP_REPOSITION_TIMEOUT_S = 5.0
_GRASP_REPOSITION_SPEED = 0.15
_GRASP_REPOSITION_SPEED_CM_S = 10.0


MOVING_STATES = {
    State.SEARCH_TABLE,
    State.COARSE_ALIGN,
    State.CONTROLLED_APPROACH,
    State.FINAL_LOCK,
    State.DOCK_RETRY,
    State.EDGE_SLIDE_SEARCH,
    State.LEAVE_EDGE,
    State.RELOCATE_TO_EDGE,
    State.REACQUIRE_EDGE,
    State.NEXT_TABLE,
    State.RETURN_HOME,
    State.AVOID_OBSTACLE,
    State.GRASP,
}


TABLE_VISION_STATES = {
    State.TABLE_APPROACH_WARMUP,
    State.SEARCH_TABLE,
    State.COARSE_ALIGN,
    State.CONTROLLED_APPROACH,
    State.FINAL_LOCK,
    State.REACQUIRE_EDGE,
}


TARGET_VISION_STATES = {
    State.SEARCH_TARGET_INIT,
    State.EDGE_SLIDE_SEARCH,
    State.TARGET_CONFIRM,
    State.TARGET_LOCKED,
    State.FREEZE_BASE,
}


TABLE_APPROACH_STATES = {
    State.TABLE_APPROACH_WARMUP,
    State.SEARCH_TABLE,
    State.COARSE_ALIGN,
    State.CONTROLLED_APPROACH,
    State.FINAL_LOCK,
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


class OrchestratorCore:
    def __init__(
        self,
        cfg: ControlThresholds,
        car_cfg: CarMotionConfig,
        docking_cfg: Optional[DockingControlConfig] = None,
        logger: Optional[Callable] = None,
    ):
        self.cfg = cfg
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

    def _reset_vision_request_dedupe(self) -> None:
        self._last_req_mono = 0.0
        self._last_mode_request_key = ""
        self._last_target_update_key = ""
        self._last_target_update_mono = 0.0

    def _log(self, level: str, msg: str, *args):
        if self._logger:
            self._logger(level, "state_machine", msg, {"args": [str(a) for a in args]} if args else None)

    def handle_task_cmd(self, cmd: TaskCmd) -> Tuple[bool, str]:
        self.ctx.last_task_cmd = cmd
        if cmd.intent == "STOP":
            self._last_stop_mono = monotonic_ts()
            self.ctx.active_session_id = cmd.session_id
            self.ctx.active_epoch = cmd.epoch
            self._interrupt_to_idle("收到 STOP 命令", tts_text="已停止", interrupt_tts=True, send_vision_idle=True)
            return True, "STOP accepted"
        if self._last_stop_mono > 0 and (monotonic_ts() - self._last_stop_mono) < float(self.cfg.post_stop_ignore_s):
            self._log("info", f"忽略 STOP 短窗内的后续命令: {cmd.intent}")
            return False, "ignored in post-stop guard"
        if cmd.confidence < self.cfg.cmd_confidence_th:
            self._log("warn", f"忽略低置信度 task_cmd: {cmd.confidence}")
            self._queue_tts("命令置信度过低")
            return False, "low confidence"
        if cmd.intent == "FIND":
            self._start_find_task(cmd)
            return True, f"FIND accepted: {cmd.target}"
        if cmd.intent == "RETURN":
            self._start_return_task(cmd)
            return True, "RETURN accepted"
        return False, "unsupported intent"

    def handle_table_obs(self, obs: TableEdgeObs):
        self.ctx.last_table_obs = obs

    def handle_target_obs(self, obs: TargetObs):
        self.ctx.last_target_obs = obs

    def handle_home_obs(self, obs: HomeTagObs):
        self.ctx.last_home_obs = obs

    def handle_grasp_obs(self, obs: Dict[str, Any]):
        self.ctx.grasp_status = str(obs.get("status") or "")
        self.ctx.grasp_result = obs.get("grasp") if isinstance(obs.get("grasp"), dict) else None
        self.ctx.grasp_reason = str(obs.get("reason") or "")
        proposal = obs.get("reposition_proposal")
        self.ctx.grasp_reposition_proposal = proposal if isinstance(proposal, dict) else None

    def handle_arm_response(self, resp: ArmResponse):
        self.ctx.arm_response = resp

    def handle_car_state(self, state: CarState):
        self.ctx.last_car_state = state
        self.ctx.last_car_state_mono = monotonic_ts()
        if state.estop and self.cfg.car_estop_to_stop:
            reason = f"底盘急停: {state.message or state.state}"
            self.ctx.last_safety_reason = reason
            self._interrupt_to_idle(reason, tts_text="底盘急停，已停止", interrupt_tts=True)
        elif state.fault and self.cfg.car_fault_to_fail:
            reason = f"底盘故障: {state.message or state.state}"
            self.ctx.last_fail_reason = reason
            self._enter_error_recovery(reason, tts_text="底盘故障，请检查小车", interrupt_tts=True)
        elif state.timeout and self.cfg.car_timeout_to_stop and self.ctx.state != State.IDLE:
            reason = f"底盘超时: {state.message or state.state}"
            self.ctx.last_fail_reason = reason
            self._enter_error_recovery(reason, tts_text="底盘通信超时，已停止", interrupt_tts=True)

    def handle_vision_req_send_result(self, sent: bool, payload: Dict, error: str = ""):
        if sent:
            self.ctx.vision_req_fail_streak = 0
            self.ctx.active_req_id = str(payload.get("req_id", "") or self.ctx.active_req_id)
            return
        self.ctx.vision_req_fail_streak += 1
        if not self.cfg.vision_req_fail_to_stop:
            return
        if self.ctx.state in {State.IDLE, State.ERROR_RECOVERY}:
            return
        if self.ctx.vision_req_fail_streak < int(self.cfg.vision_req_fail_threshold):
            return
        reason = f"vision_req_out 发送失败 {self.ctx.vision_req_fail_streak} 次"
        if error:
            reason += f": {error}"
        self.ctx.last_fail_reason = reason
        self._enter_error_recovery(reason, tts_text="视觉链路异常，已停车", interrupt_tts=True)

    def drain_vision_msgs(self) -> List[Dict]:
        out = list(self.ctx.pending_vision_msgs)
        self.ctx.pending_vision_msgs.clear()
        return out

    def drain_tts_msgs(self) -> List[Dict]:
        out = list(self.ctx.pending_tts_msgs)
        self.ctx.pending_tts_msgs.clear()
        return out

    def tick(self) -> MotionDecision:
        safety_override = self._check_safety_interlock()
        if safety_override is not None:
            return safety_override
        dispatch = {
            State.IDLE: self._tick_idle,
            State.TABLE_APPROACH_WARMUP: self._tick_table_approach_warmup,
            State.SEARCH_TABLE: self._tick_search_table,
            State.COARSE_ALIGN: self._tick_coarse_align,
            State.CONTROLLED_APPROACH: self._tick_controlled_approach,
            State.FINAL_LOCK: self._tick_final_lock,
            State.DOCK_RETRY: self._tick_dock_retry,
            State.AT_TABLE_EDGE: self._tick_at_table_edge,
            State.SEARCH_TARGET_INIT: self._tick_search_target_init,
            State.EDGE_SLIDE_SEARCH: self._tick_edge_slide_search,
            State.TARGET_CONFIRM: self._tick_target_confirm,
            State.TARGET_LOCKED: self._tick_target_locked,
            State.FREEZE_BASE: self._tick_freeze_base,
            State.LEAVE_EDGE: self._tick_leave_edge,
            State.RELOCATE_TO_EDGE: self._tick_relocate_to_edge,
            State.REACQUIRE_EDGE: self._tick_reacquire_edge,
            State.NEXT_TABLE: self._tick_next_table,
            State.AVOID_OBSTACLE: self._tick_avoid_obstacle,
            State.RETURN_HOME: self._tick_return_home,
            State.ERROR_RECOVERY: self._tick_error_recovery,
            State.DONE: self._tick_done,
            State.GRASP: self._tick_grasp,
        }
        return dispatch.get(self.ctx.state, self._tick_idle)()

    def export_state_block(self) -> Dict:
        table_obs = self.ctx.last_table_obs
        target_obs = self.ctx.last_target_obs
        lock_status = self._final_lock_status(table_obs, stable_count=self.ctx.table_lock_frames)
        target_window = self._target_window_stats()
        return {
            "ts": time.time(),
            "state": self.ctx.state.value,
            "prev_state": self.ctx.prev_state.value if self.ctx.prev_state else "",
            "resume_state": self.ctx.resume_state.value if self.ctx.resume_state else "",
            "task_intent": self.ctx.task_intent,
            "active_target": self.ctx.active_target,
            "session_id": self.ctx.active_session_id,
            "epoch": self.ctx.active_epoch,
            "req_id": self.ctx.active_req_id,
            "vision_stage": self.ctx.active_vision_stage,
            "vision_mode": self.ctx.active_vision_mode,
            "current_edge_id": self.ctx.current_edge_id,
            "edge_visit_index": self.ctx.edge_visit_index,
            "edge_transition_count": self.ctx.edge_transition_count,
            "table_cycle_count": self.ctx.table_cycle_count,
            "last_enter_reason": self.ctx.last_enter_reason,
            "last_fail_reason": self.ctx.last_fail_reason,
            "last_safety_reason": self.ctx.last_safety_reason,
            "vision_req_fail_streak": self.ctx.vision_req_fail_streak,
            "table_found_frames": self.ctx.table_found_frames,
            "table_lost_frames": self.ctx.table_lost_frames,
            "table_lock_frames": self.ctx.table_lock_frames,
            "table_approach_warmup_active": self.ctx.state == State.TABLE_APPROACH_WARMUP,
            "warmup_elapsed_s": self._state_elapsed() if self.ctx.state == State.TABLE_APPROACH_WARMUP else 0.0,
            "warmup_plane_seen_count": int(self.ctx.table_approach_warmup_plane_seen_count),
            "warmup_fresh_obs_count": int(self.ctx.table_approach_warmup_fresh_obs_count),
            "warmup_reason": "waiting_for_initial_vision" if self.ctx.state == State.TABLE_APPROACH_WARMUP else "",
            "table_dock_phase": str(self.ctx.table_dock_phase or ""),
            "table_micro_adjust_count": int(self.ctx.table_micro_adjust_count),
            "approach_aligned_frames": self.ctx.approach_aligned_frames,
            "target_found_frames": self.ctx.target_found_frames,
            "target_lost_frames": self.ctx.target_lost_frames,
            "target_lock_frames": self.ctx.target_lock_frames,
            "target_stable_ms": self._target_stable_ms(),
            "center_jitter": float(self.ctx.target_last_center_jitter),
            "target_window_found_ratio": target_window.get("found_ratio"),
            "target_conf_median": target_window.get("conf_median"),
            "target_conf_max": target_window.get("conf_max"),
            "bbox_valid_ratio": target_window.get("bbox_valid_ratio"),
            "target_window_latest_matched_cls": target_window.get("latest_matched_cls"),
            "target_window_latest_matched_conf": target_window.get("latest_matched_conf"),
            "lost_reason": self.ctx.target_last_lost_reason,
            "tag_lost_frames": self.ctx.tag_lost_frames,
            "tag_arrived_frames": self.ctx.tag_arrived_frames,
            "dock_retry_count": self.ctx.dock_retry_count,
            "edge_slide_relock_attempts": self.ctx.edge_slide_relock_attempts,
            "avoid_retry_count": self.ctx.avoid_retry_count,
            "table_loss_elapsed_s": self._loss_elapsed(self.ctx.table_loss_since_mono),
            "target_loss_elapsed_s": self._loss_elapsed(self.ctx.target_loss_since_mono),
            "tag_loss_elapsed_s": self._loss_elapsed(self.ctx.tag_loss_since_mono),
            "has_table_edge_obs": table_obs is not None,
            "has_target_obs": target_obs is not None,
            "lock_ready": bool(lock_status["lock_ready"]),
            "lock_reason": str(lock_status["reason"]),
            "final_lock_debug": dict(lock_status),
            "final_lock_yaw_ok": bool(lock_status.get("yaw_ok", False)),
            "final_lock_dist_ok": bool(lock_status.get("dist_ok", False)),
            "final_lock_age_ok": bool(lock_status.get("age_ok", False)),
            "final_lock_confidence_ok": bool(lock_status.get("confidence_ok", False)),
            "stable_lock_count": int(lock_status.get("stable_lock_count", self.ctx.table_lock_frames) or 0),
            "required_lock_count": int(lock_status.get("required_lock_count", self._required_lock_count()) or 0),
            "lock_count_inc_reason": str(lock_status.get("lock_count_inc_reason", "") or ""),
            "lock_count_hold_reason": str(lock_status.get("lock_count_hold_reason", "") or ""),
            "lock_count_reset_reason": str(lock_status.get("lock_count_reset_reason", "") or ""),
            "lock_reset_reason": str(lock_status.get("lock_reset_reason", "") or ""),
            "vision_stale_reason": str(lock_status.get("vision_stale_reason", "") or ""),
            "table_found": bool(table_obs.table_found) if table_obs is not None else False,
            "edge_found": bool(table_obs.edge_found) if table_obs is not None else False,
            "edge_valid": bool(getattr(table_obs, "edge_valid", table_obs.edge_found)) if table_obs is not None else False,
            "valid_for_control": bool(getattr(table_obs, "valid_for_control", False)) if table_obs is not None else False,
            "raw_control_level": self._raw_control_level(table_obs),
            "normalized_control_level": self._control_level(table_obs),
            "control_level": self._control_level(table_obs),
            "reject_reason": getattr(table_obs, "reject_reason", None) if table_obs is not None else None,
            "control_reject_reason": getattr(table_obs, "control_reject_reason", None) if table_obs is not None else None,
            "usable_for_approach": bool(getattr(table_obs, "usable_for_approach", False)) if table_obs is not None else False,
            "usable_for_alignment": bool(getattr(table_obs, "usable_for_alignment", False)) if table_obs is not None else False,
            "usable_for_stop": bool(getattr(table_obs, "usable_for_stop", False)) if table_obs is not None else False,
            "pose_found": bool(getattr(table_obs, "pose_found", False)) if table_obs is not None else False,
            "table_confirmed_by_yolo": bool(getattr(table_obs, "table_confirmed_by_yolo", False)) if table_obs is not None else False,
            "yolo_gate_open": bool(getattr(table_obs, "yolo_gate_open", False)) if table_obs is not None else False,
            "table_approach_phase": self._table_approach_phase(table_obs),
            "view_source": getattr(table_obs, "view_source", None) if table_obs is not None else None,
            "view_err_norm": getattr(table_obs, "view_err_norm", None) if table_obs is not None else None,
            "view_reliable": bool(getattr(table_obs, "view_reliable", False)) if table_obs is not None else False,
            "fov_guard_active": bool(getattr(table_obs, "fov_guard_active", False)) if table_obs is not None else False,
            "fov_guard_reason": str(getattr(table_obs, "fov_guard_reason", "") or "") if table_obs is not None else "",
            "yolo_reliable": bool(getattr(table_obs, "yolo_reliable", False)) if table_obs is not None else False,
            "plane_cx_norm": getattr(table_obs, "plane_cx_norm", None) if table_obs is not None else None,
            "plane_width_norm": getattr(table_obs, "plane_width_norm", None) if table_obs is not None else None,
            "plane_touch_left": bool(getattr(table_obs, "plane_touch_left", False)) if table_obs is not None else False,
            "plane_touch_right": bool(getattr(table_obs, "plane_touch_right", False)) if table_obs is not None else False,
            "confidence": table_obs.confidence if table_obs is not None else None,
            "edge_conf": getattr(table_obs, "edge_conf", table_obs.confidence) if table_obs is not None else None,
            "yaw_err_rad": table_obs.yaw_err_rad if table_obs is not None else None,
            "dist_err_m": table_obs.dist_err_m if table_obs is not None else None,
            "target_dist_m": table_obs.target_dist_m if table_obs is not None else None,
            "table_edge_obs_ts": table_obs.obs_ts if table_obs is not None else None,
            "table_edge_obs_age_ms": self._table_obs_age_ms(table_obs),
            "obs_total_age_ms": self._table_obs_age_ms(table_obs),
            "vision_process_ms": getattr(table_obs, "vision_process_ms", getattr(table_obs, "edge_process_ms", None)) if table_obs is not None else None,
            "control_loop_age_ms": self._table_control_loop_age_ms(table_obs),
            "stale_level": self._table_obs_stale_level(table_obs),
            "stale_guard_active": self._table_obs_stale_level(table_obs) != "fresh",
            "stale_guard_reason": self._table_obs_stale_reason(table_obs),
            "table_edge_obs_frame_id": table_obs.frame_id if table_obs is not None else None,
            "table_edge_obs_seq": table_obs.seq if table_obs is not None else None,
            "obs_seq": table_obs.obs_seq if table_obs is not None else None,
            "camera_frame_seq": table_obs.camera_frame_seq if table_obs is not None else None,
            "camera_frame_ts_ms": table_obs.camera_frame_ts_ms if table_obs is not None else None,
            "vision_process_start_ts_ms": table_obs.vision_process_start_ts_ms if table_obs is not None else None,
            "vision_process_end_ts_ms": table_obs.vision_process_end_ts_ms if table_obs is not None else None,
            "vision_publish_ts_ms": table_obs.vision_publish_ts_ms if table_obs is not None else None,
            "obs_out_send_ts_ms": table_obs.obs_out_send_ts_ms if table_obs is not None else None,
            "orchestrator_recv_ts_ms": table_obs.orchestrator_recv_ts_ms if table_obs is not None else None,
            "state_machine_consume_ts_ms": table_obs.state_machine_consume_ts_ms if table_obs is not None else None,
            "cmd_publish_ts_ms": table_obs.cmd_publish_ts_ms if table_obs is not None else None,
            "edge_update_interval_ms": table_obs.edge_update_interval_ms if table_obs is not None else None,
            "edge_process_ms": table_obs.edge_process_ms if table_obs is not None else None,
            "camera_frame_interval_ms": table_obs.camera_frame_interval_ms if table_obs is not None else None,
            "camera_frame_hz": table_obs.camera_frame_hz if table_obs is not None else None,
            "vision_process_interval_ms": table_obs.vision_process_interval_ms if table_obs is not None else None,
            "vision_publish_interval_ms": table_obs.vision_publish_interval_ms if table_obs is not None else None,
            "obs_out_send_interval_ms": table_obs.obs_out_send_interval_ms if table_obs is not None else None,
            "obs_out_send_hz": table_obs.obs_out_send_hz if table_obs is not None else None,
            "table_edge_obs_recv_interval_ms": table_obs.table_edge_obs_recv_interval_ms if table_obs is not None else None,
            "orchestrator_recv_interval_ms": table_obs.orchestrator_recv_interval_ms if table_obs is not None else None,
            "table_edge_obs_recv_hz": table_obs.table_edge_obs_recv_hz if table_obs is not None else None,
            "state_machine_tick_interval_ms": table_obs.state_machine_tick_interval_ms if table_obs is not None else None,
            "state_machine_consume_interval_ms": table_obs.state_machine_consume_interval_ms if table_obs is not None else None,
            "same_obs_reuse_count": table_obs.same_obs_reuse_count if table_obs is not None else None,
            "obs_seq_gap": table_obs.obs_seq_gap if table_obs is not None else None,
            "obs_age_at_consume_ms": table_obs.obs_age_at_consume_ms if table_obs is not None else None,
            "vision_publish_to_orch_recv_ms": table_obs.vision_publish_to_orch_recv_ms if table_obs is not None else None,
            "orch_recv_to_state_consume_ms": table_obs.orch_recv_to_state_consume_ms if table_obs is not None else None,
            "table_edge_obs_source_mode": table_obs.source_mode if table_obs is not None else None,
            "edge_obs_is_stale": self._edge_obs_is_stale(table_obs),
            "edge_follow_stale": self._edge_obs_is_stale(table_obs) if self.ctx.state == State.EDGE_SLIDE_SEARCH else False,
            "locked_edge_conf": self.ctx.locked_edge_conf,
            "locked_yaw_err": self.ctx.locked_yaw_err,
            "locked_dist_err": self.ctx.locked_dist_err,
            "locked_roi": self.ctx.locked_roi,
            "locked_obs_seq": self.ctx.locked_obs_seq,
            "handoff_state": self.ctx.handoff_state,
            "handoff_samples_count": len(self.ctx.slide_ref_samples),
            "handoff_valid_samples_count": len(self.ctx.slide_ref_samples),
            "slide_ref_ready": bool(self.ctx.slide_ref_ready),
            "slide_ref_yaw_err": self.ctx.slide_ref_yaw_err,
            "slide_ref_dist_err": self.ctx.slide_ref_dist_err,
            "slide_ref_edge_conf": self.ctx.slide_ref_edge_conf,
            "slide_ref_roi": self.ctx.slide_ref_roi,
            "slide_ref_seq": self.ctx.slide_ref_seq,
            "full_locked_yaw_err": self.ctx.locked_yaw_err,
            "full_locked_dist_err": self.ctx.locked_dist_err,
            "full_vs_light_yaw_offset": self._full_vs_light_yaw_offset(),
            "full_vs_light_dist_offset": self._full_vs_light_dist_offset(),
            "edge_quality": dict(self.ctx.last_edge_quality or {}),
            "task_result": "success" if self.ctx.state == State.DONE and not self.ctx.last_fail_reason else ("failed" if self.ctx.state == State.ERROR_RECOVERY else ""),
            "task_total_time_s": max(0.0, time.time() - float(self.ctx.task_start_wall_ts or time.time())) if self.ctx.task_start_wall_ts else 0.0,
            "edge_retries": int(self.ctx.dock_retry_count + self.ctx.edge_transition_count),
            "slide_entries": int(self.ctx.task_slide_entries_count),
            "target_confirm_count": int(self.ctx.task_target_confirm_count),
            "target_locked_count": int(self.ctx.task_target_locked_count),
            "last_matched_cls": target_obs.matched_cls if target_obs is not None else None,
            "last_matched_conf": target_obs.matched_conf if target_obs is not None else None,
            "last_edge_conf": table_obs.confidence if table_obs is not None else None,
            "warnings": list(self.ctx.task_warning_history),
        }

    def _transition(self, new_state: State, reason: str):
        old_state = self.ctx.state
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
        if new_state == State.DONE and self.ctx.last_fail_reason:
            warning = str(self.ctx.last_fail_reason).strip()
            if warning and warning not in self.ctx.task_warning_history:
                self.ctx.task_warning_history.append(warning)
        self.ctx.clear_motion_counters()
        if preserve_target_debounce:
            self._restore_target_debounce_snapshot(target_debounce)
        if self.transition_observer is not None:
            try:
                self.transition_observer(old_state.value, new_state.value, reason)
            except Exception:
                pass
        self._on_enter_state(new_state)

    def _float_or_none(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _target_center(self, obs: Optional[TargetObs]) -> Optional[Dict[str, Optional[float]]]:
        if obs is None:
            return None
        full = self._target_center_full_norm(obs)
        offset = self._float_or_none(obs.cx_norm)
        if full is not None and full.get("cx") is not None:
            return full
        cy = self._float_or_none(obs.cy_norm)
        if full is not None and cy is None:
            cy = self._float_or_none(full.get("cy"))
        if offset is None and cy is None:
            return None
        cx = None
        if offset is not None:
            cx = max(0.0, min(1.0, 0.5 - (float(offset) / 2.0)))
        return {
            "cx": cx,
            "cy": cy,
        }

    def _target_center_full_norm(self, obs: Optional[TargetObs]) -> Optional[Dict[str, Optional[float]]]:
        if obs is None:
            return None
        source = obs.matched_center_full_norm if isinstance(obs.matched_center_full_norm, dict) else None
        if source is None and isinstance(obs.matched_center, dict):
            source = obs.matched_center
        cx = None
        cy = None
        if isinstance(source, dict):
            cx = self._float_or_none(source.get("cx", source.get("x_norm", source.get("cx_norm"))))
            cy = self._float_or_none(source.get("cy", source.get("y_norm", source.get("cy_norm"))))
            if cx is not None and not (0.0 <= float(cx) <= 1.0):
                cx = None
            if cy is not None and not (0.0 <= float(cy) <= 1.0):
                cy = None
        if cx is None:
            cx = self._float_or_none(getattr(obs, "x_norm", None))
        if cy is None:
            cy = self._float_or_none(getattr(obs, "y_norm", None))
        if cx is None and cy is None:
            return None
        if cx is not None:
            cx = max(0.0, min(1.0, float(cx)))
        if cy is not None:
            cy = max(0.0, min(1.0, float(cy)))
        return {"cx": cx, "cy": cy}

    def _target_center_offset_norm(self, obs: Optional[TargetObs]) -> Optional[Dict[str, Optional[float]]]:
        if obs is None:
            return None
        source = obs.matched_center_offset_norm if isinstance(obs.matched_center_offset_norm, dict) else None
        dx = None
        dy = None
        if isinstance(source, dict):
            dx = self._float_or_none(source.get("dx"))
            dy = self._float_or_none(source.get("dy"))
        if dx is None:
            dx = self._float_or_none(obs.cx_norm)
        if dy is None and self._target_center_full_norm(obs) is not None:
            full = self._target_center_full_norm(obs) or {}
            cy = self._float_or_none(full.get("cy"))
            if cy is not None:
                dy = max(-1.0, min(1.0, 1.0 - (2.0 * float(cy))))
        if dx is None and dy is None:
            return None
        if dx is not None:
            dx = max(-1.0, min(1.0, float(dx)))
        if dy is not None:
            dy = max(-1.0, min(1.0, float(dy)))
        return {
            "dx": dx,
            "dy": dy,
        }

    def _frames_to_ms(self, frames: int) -> int:
        try:
            tick_hz = max(1.0, float(getattr(self.cfg, "tick_hz", 10.0)))
        except Exception:
            tick_hz = 10.0
        return int(round((max(0, int(frames)) / tick_hz) * 1000.0))

    def _transition_stable_ms(self, old_state: State) -> int:
        if old_state == State.SEARCH_TABLE:
            return self._frames_to_ms(self.ctx.table_found_frames)
        if old_state == State.COARSE_ALIGN:
            return self._frames_to_ms(self.ctx.approach_aligned_frames)
        if old_state == State.FINAL_LOCK:
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
        target_obs = self.ctx.last_target_obs
        car_state = self.ctx.last_car_state
        edge_quality = dict(self.ctx.last_edge_quality or {})
        target_window = self._target_window_stats()
        return {
            "event": "state_transition",
            "previous_state": old_state.value,
            "next_state": new_state.value,
            "reason": str(reason or ""),
            "target": self.ctx.active_target,
            "session_id": self.ctx.active_session_id,
            "epoch": self.ctx.active_epoch,
            "req_id": self.ctx.active_req_id,
            "edge_id": self.ctx.current_edge_id,
            "edge_conf": self._float_or_none(table_obs.confidence if table_obs is not None else None),
            "yaw_err": self._float_or_none(table_obs.yaw_err_rad if table_obs is not None else None),
            "dist_err": self._float_or_none(table_obs.dist_err_m if table_obs is not None else None),
            "edge_obs_age_ms": self._table_obs_age_ms(table_obs),
            "edge_obs_is_stale": self._edge_obs_is_stale(table_obs),
            **self._handoff_trace_fields(),
            "yaw_delta_from_slide_ref": edge_quality.get("yaw_delta_from_slide_ref"),
            "dist_delta_from_slide_ref": edge_quality.get("dist_delta_from_slide_ref"),
            "edge_identity_basis": edge_quality.get("edge_identity_basis"),
            "stable_ms": self._transition_stable_ms(old_state),
            "lost_ms": self._transition_lost_ms(old_state),
            "table_approach_warmup_active": old_state == State.TABLE_APPROACH_WARMUP,
            "warmup_elapsed_s": self._state_elapsed() if old_state == State.TABLE_APPROACH_WARMUP else 0.0,
            "warmup_plane_seen_count": int(self.ctx.table_approach_warmup_plane_seen_count),
            "warmup_fresh_obs_count": int(self.ctx.table_approach_warmup_fresh_obs_count),
            "warmup_reason": "waiting_for_initial_vision" if old_state == State.TABLE_APPROACH_WARMUP else "",
            "target_found": bool(target_obs.found) if target_obs is not None else None,
            "target_conf": self._float_or_none(target_obs.confidence if target_obs is not None else None),
            "target_cls": (target_obs.matched_cls or target_obs.target if target_obs is not None else None),
            "matched_cls": (target_obs.matched_cls if target_obs is not None else None),
            "matched_conf": self._float_or_none(target_obs.matched_conf if target_obs is not None else None),
            "best_cls": (target_obs.best_cls if target_obs is not None else None),
            "best_conf": self._float_or_none(target_obs.best_conf if target_obs is not None else None),
            "target_center": self._target_center(target_obs),
            "matched_center": self._target_center(target_obs),
            "matched_center_full_norm": self._target_center_full_norm(target_obs),
            "matched_center_offset_norm": self._target_center_offset_norm(target_obs),
            "bbox_valid": target_obs.bbox_valid if target_obs is not None else None,
            "bbox_invalid_reason": target_obs.bbox_invalid_reason if target_obs is not None else None,
            "target_window_found_ratio": target_window.get("found_ratio"),
            "target_conf_median": target_window.get("conf_median"),
            "target_conf_max": target_window.get("conf_max"),
            "bbox_valid_ratio": target_window.get("bbox_valid_ratio"),
            "target_window_latest_matched_cls": target_window.get("latest_matched_cls"),
            "target_window_latest_matched_conf": target_window.get("latest_matched_conf"),
            "found_frames": int(self.ctx.target_found_frames),
            "lost_frames": int(self.ctx.target_lost_frames),
            "confirm_elapsed_ms": int(round(self._state_elapsed() * 1000.0)) if old_state == State.TARGET_CONFIRM else 0,
            "lock_elapsed_ms": int(round(self._state_elapsed() * 1000.0)) if old_state == State.TARGET_LOCKED else 0,
            "lost_hold_ms": self._transition_lost_ms(old_state),
            "lock_decision_reason": str(self.ctx.target_last_transition_reason or ""),
            "unlock_reason": str(self.ctx.target_last_lost_reason or "") if new_state == State.EDGE_SLIDE_SEARCH else "",
            "target_stable_ms": self._target_stable_ms(),
            "center_jitter": float(self.ctx.target_last_center_jitter),
            "lost_reason": str(self.ctx.target_last_lost_reason or ""),
            "transition_reason": str(reason or ""),
            "car_mode": car_state.mode if car_state is not None else None,
            "planned_cmd": {"vx": None, "vy": None, "wz": None},
            "condition": self._transition_condition_snapshot(old_state, new_state),
        }

    def _on_enter_state(self, state: State):
        if state == State.IDLE:
            self.reset_task_runtime("enter_idle", keep_session=False)
            self.ctx.active_vision_stage = ""
            self.ctx.active_vision_mode = ""
            self.ctx.resume_state = None
            return
        if state == State.SEARCH_TARGET_INIT:
            self._reset_slide_ref_handoff()
        if state == State.TABLE_APPROACH_WARMUP:
            self.ctx.table_approach_warmup_fresh_obs_count = 0
            self.ctx.table_approach_warmup_plane_seen_count = 0
            self.ctx.table_approach_warmup_last_obs_key = ""
            self.ctx.table_found_frames = 0
        if state == State.SEARCH_TABLE:
            self.reset_edge_tracking("enter_search_table")
            self.reset_target_tracking("enter_search_table")
        elif state in {State.DOCK_RETRY, State.LEAVE_EDGE, State.NEXT_TABLE}:
            reason = f"enter_{state.value.lower()}"
            self.reset_edge_tracking(reason)
            self.reset_target_tracking(reason)
            self.reset_slide_reference(reason)
        elif state == State.GRASP:
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

    def _target_debounce_snapshot(self) -> Dict[str, Any]:
        return {
            "target_found_frames": int(self.ctx.target_found_frames),
            "target_lost_frames": int(self.ctx.target_lost_frames),
            "target_lock_frames": int(self.ctx.target_lock_frames),
            "target_loss_since_mono": float(self.ctx.target_loss_since_mono),
            "target_stable_since_mono": float(self.ctx.target_stable_since_mono),
            "target_center_history": [dict(item) for item in self.ctx.target_center_history],
            "target_obs_window": [dict(item) for item in self.ctx.target_obs_window],
            "target_last_center_jitter": float(self.ctx.target_last_center_jitter),
            "target_last_lost_reason": str(self.ctx.target_last_lost_reason or ""),
            "target_last_transition_reason": str(self.ctx.target_last_transition_reason or ""),
        }

    def _restore_target_debounce_snapshot(self, snapshot: Dict[str, Any]) -> None:
        if not snapshot:
            return
        self.ctx.target_found_frames = int(snapshot.get("target_found_frames", 0) or 0)
        self.ctx.target_lost_frames = int(snapshot.get("target_lost_frames", 0) or 0)
        self.ctx.target_lock_frames = int(snapshot.get("target_lock_frames", 0) or 0)
        self.ctx.target_loss_since_mono = float(snapshot.get("target_loss_since_mono", 0.0) or 0.0)
        self.ctx.target_stable_since_mono = float(snapshot.get("target_stable_since_mono", 0.0) or 0.0)
        self.ctx.target_center_history = [dict(item) for item in snapshot.get("target_center_history", [])]
        self.ctx.target_last_center_jitter = float(snapshot.get("target_last_center_jitter", 0.0) or 0.0)
        self.ctx.target_obs_window = [dict(item) for item in snapshot.get("target_obs_window", [])]
        self.ctx.target_last_lost_reason = str(snapshot.get("target_last_lost_reason", "") or "")
        self.ctx.target_last_transition_reason = str(snapshot.get("target_last_transition_reason", "") or "")

    def _emit_reset_trace(self, reset_state: str, reason: str, cleared_fields: List[str]) -> None:
        self._pending_reset_traces.append(
            {
                "event": "reset_state",
                "reset_state": reset_state,
                "reset_reason": reason,
                "cleared_fields": list(cleared_fields),
                "session_id": self.ctx.active_session_id,
                "target": self.ctx.active_target,
                "state": self.ctx.state.value,
            }
        )

    def reset_edge_tracking(self, reason: str) -> None:
        cleared = [
            "last_table_obs",
            "locked_edge_id",
            "locked_edge_line",
            "locked_roi",
            "locked_yaw_err",
            "locked_dist_err",
            "locked_edge_conf",
            "locked_obs_seq",
            "slide_ref_ready",
            "slide_ref_yaw_err",
            "slide_ref_dist_err",
            "slide_ref_edge_conf",
            "slide_ref_roi",
            "slide_ref_seq",
            "slide_ref_samples",
            "slide_ref_last_sample_key",
            "handoff_state",
            "last_edge_quality",
            "table_found_frames",
            "table_lost_frames",
            "table_lock_frames",
            "edge_identity_state",
        ]
        self.ctx.last_table_obs = None
        self.ctx.locked_edge_id = ""
        self.ctx.locked_edge_line = None
        self.ctx.locked_roi = None
        self.ctx.locked_yaw_err = None
        self.ctx.locked_dist_err = None
        self.ctx.locked_edge_conf = None
        self.ctx.locked_obs_seq = None
        self.ctx.slide_ref_ready = False
        self.ctx.slide_ref_yaw_err = None
        self.ctx.slide_ref_dist_err = None
        self.ctx.slide_ref_edge_conf = None
        self.ctx.slide_ref_roi = None
        self.ctx.slide_ref_seq = None
        self.ctx.slide_ref_samples.clear()
        self.ctx.slide_ref_last_sample_key = ""
        self.ctx.handoff_state = ""
        self.ctx.last_edge_quality.clear()
        self.ctx.table_found_frames = 0
        self.ctx.table_lost_frames = 0
        self.ctx.table_lock_frames = 0
        self._emit_reset_trace("edge", reason, cleared)

    def reset_target_tracking(self, reason: str) -> None:
        cleared = [
            "last_target_obs",
            "target_found_frames",
            "target_lost_frames",
            "target_lock_frames",
            "target_loss_since_mono",
            "target_stable_since_mono",
            "target_center_history",
            "target_last_center_jitter",
            "target_obs_window",
            "target_last_lost_reason",
            "target_last_transition_reason",
        ]
        self.ctx.last_target_obs = None
        self.ctx.target_found_frames = 0
        self.ctx.target_lost_frames = 0
        self.ctx.target_lock_frames = 0
        self.ctx.target_loss_since_mono = 0.0
        self.ctx.target_stable_since_mono = 0.0
        self.ctx.target_center_history.clear()
        self.ctx.target_obs_window.clear()
        self.ctx.target_last_center_jitter = 0.0
        self.ctx.target_last_lost_reason = ""
        self.ctx.target_last_transition_reason = ""
        self._emit_reset_trace("target", reason, cleared)

    def reset_slide_reference(self, reason: str) -> None:
        cleared = [
            "slide_ref_ready",
            "slide_ref_yaw_err",
            "slide_ref_dist_err",
            "slide_ref_edge_conf",
            "slide_ref_roi",
            "slide_ref_seq",
            "slide_ref_samples",
            "slide_ref_last_sample_key",
            "handoff_state",
        ]
        self.ctx.slide_ref_ready = False
        self.ctx.slide_ref_yaw_err = None
        self.ctx.slide_ref_dist_err = None
        self.ctx.slide_ref_edge_conf = None
        self.ctx.slide_ref_roi = None
        self.ctx.slide_ref_seq = None
        self.ctx.slide_ref_samples.clear()
        self.ctx.slide_ref_last_sample_key = ""
        self.ctx.handoff_state = ""
        self._emit_reset_trace("slide_ref", reason, cleared)

    def reset_task_runtime(self, reason: str, keep_session: bool = False) -> None:
        cleared = [
            "task_intent",
            "active_target",
            "active_session_id",
            "active_epoch",
            "active_req_id",
            "active_vision_stage",
            "active_vision_mode",
            "current_edge_id",
            "edge_visit_index",
            "edge_transition_count",
            "table_cycle_count",
            "locked_edge_id",
            "locked_edge_line",
            "locked_roi",
            "locked_yaw_err",
            "locked_dist_err",
            "locked_edge_conf",
            "locked_obs_seq",
            "slide_ref_ready",
            "slide_ref_yaw_err",
            "slide_ref_dist_err",
            "slide_ref_edge_conf",
            "slide_ref_roi",
            "slide_ref_seq",
            "slide_ref_samples",
            "slide_ref_last_sample_key",
            "handoff_state",
            "last_edge_quality",
            "last_fail_reason",
            "last_enter_reason",
            "last_safety_reason",
            "vision_req_fail_streak",
            "task_slide_entries_count",
            "task_target_confirm_count",
            "task_target_locked_count",
            "edge_slide_relock_attempts",
            "task_warning_history",
            "task_done_summary_emitted",
        ]
        session_id = self.ctx.active_session_id
        self._emit_reset_trace("task", reason, cleared)
        self.ctx.clear_task_context()
        self.ctx.last_fail_reason = ""
        self.ctx.last_enter_reason = ""
        self.ctx.last_safety_reason = ""
        self.ctx.vision_req_fail_streak = 0
        self.ctx.task_slide_entries_count = 0
        self.ctx.task_target_confirm_count = 0
        self.ctx.task_target_locked_count = 0
        self.ctx.edge_slide_relock_attempts = 0
        self.ctx.task_warning_history.clear()
        self.ctx.task_done_summary_emitted = False
        self._reset_vision_request_dedupe()
        if keep_session:
            self.ctx.active_session_id = session_id

    def _interrupt_to_idle(self, reason: str, tts_text: Optional[str] = None, interrupt_tts: bool = False, send_vision_idle: bool = False):
        if send_vision_idle:
            self._queue_vision_req(make_vision_idle(session_id=self.ctx.active_session_id, epoch=self.ctx.active_epoch), force=True)
        self._transition(State.IDLE, reason)
        if tts_text:
            self._queue_tts(tts_text, interrupt=interrupt_tts)

    def _enter_error_recovery(self, reason: str, tts_text: Optional[str] = None, interrupt_tts: bool = False):
        self.ctx.resume_state = None
        self._transition(State.ERROR_RECOVERY, reason)
        if tts_text:
            self._queue_tts(tts_text, interrupt=interrupt_tts)

    def _maybe_resend_req(self, req: Optional[Dict]):
        self._queue_vision_req(req, force=False)

    def _queue_vision_req(self, payload: Dict, force: bool = False):
        if not isinstance(payload, dict) or not payload:
            return
        now_m = monotonic_ts()
        req_type = str(payload.get("req_type") or (payload.get("payload") or {}).get("req_type") or "").strip().lower()
        if not req_type:
            req_type = "mode_request" if str(payload.get("op") or "").strip().upper() in {"START", "STOP"} else "target_update"
            payload["req_type"] = req_type
        req_payload = dict(payload.get("payload") or {})
        req_payload["req_type"] = req_type
        payload["payload"] = req_payload

        request_key = self._vision_request_key(payload, req_type=req_type)
        if req_type == "mode_request":
            if request_key and request_key == self._last_mode_request_key:
                return
        elif req_type == "target_update":
            target_period_s = max(1.0, float(self.cfg.req_resend_period_s or 0.0))
            if request_key and request_key == self._last_target_update_key and (now_m - self._last_target_update_mono) < target_period_s:
                return
        elif not force and now_m - self._last_req_mono < self.cfg.req_resend_period_s:
            return

        if self.ctx.active_session_id and not payload.get("session_id"):
            payload["session_id"] = self.ctx.active_session_id
        if self.ctx.active_epoch and payload.get("epoch") in (None, 0):
            payload["epoch"] = self.ctx.active_epoch
        self.ctx.active_req_id = str(payload.get("req_id", self.ctx.active_req_id) or self.ctx.active_req_id)
        self.ctx.pending_vision_msgs.append(payload)
        self._last_req_mono = now_m
        if req_type == "mode_request":
            self._last_mode_request_key = request_key
        elif req_type == "target_update":
            self._last_target_update_key = request_key
            self._last_target_update_mono = now_m

    def _queue_tts(self, text: str, interrupt: bool = False):
        try:
            self.ctx.pending_tts_msgs.append(make_tts_event(text, interrupt=interrupt))
        except Exception:
            pass

    def _start_find_task(self, cmd: TaskCmd):
        target = str(cmd.target or "").strip()
        if not target:
            self._log("warn", "FIND target 为空，忽略")
            return
        self.ctx.clear_task_context()
        self._reset_vision_request_dedupe()
        self.ctx.task_intent = "FIND"
        self.ctx.active_target = target
        self.ctx.active_session_id = cmd.session_id
        self.ctx.active_epoch = cmd.epoch
        self.ctx.task_start_wall_ts = time.time()
        self._transition(State.TABLE_APPROACH_WARMUP, f"开始桌边任务，视觉warmup，目标 {target}")
        self._queue_tts(f"开始寻找 {target}")

    def _start_return_task(self, cmd: TaskCmd):
        self.ctx.clear_task_context()
        self._reset_vision_request_dedupe()
        self.ctx.task_intent = "RETURN"
        self.ctx.active_session_id = cmd.session_id
        self.ctx.active_epoch = cmd.epoch
        self.ctx.task_start_wall_ts = time.time()
        self._transition(State.RETURN_HOME, "开始返航")
        self._queue_tts("开始返航")

    def _tick_idle(self) -> MotionDecision:
        return self.controller.stop_cmd("IDLE")

    def _tick_table_approach_warmup(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        obs = self._fresh_table_obs()
        obs_key = self._table_obs_key(obs)
        if obs is not None and obs_key and obs_key != self.ctx.table_approach_warmup_last_obs_key:
            self.ctx.table_approach_warmup_last_obs_key = obs_key
            self.ctx.table_approach_warmup_fresh_obs_count += 1
            if self._table_visible(obs) and (self._control_level(obs) != "none" or self._table_plane_stable(obs)):
                self.ctx.table_approach_warmup_plane_seen_count += 1

        elapsed_s = self._state_elapsed()
        warmup_s = max(0.0, float(getattr(self.cfg, "table_approach_warmup_s", 2.0) or 0.0))
        min_fresh = max(0, int(getattr(self.cfg, "table_approach_warmup_min_fresh_obs", 1) or 0))
        if elapsed_s < warmup_s:
            return self.controller.stop_cmd("TABLE_APPROACH_WARMUP")

        has_min_fresh = self.ctx.table_approach_warmup_fresh_obs_count >= min_fresh
        level = self._control_level(obs)
        plane_ready = bool(has_min_fresh and self._table_visible(obs) and (level != "none" or self._table_plane_stable(obs)))
        if plane_ready:
            self.ctx.table_found_frames = max(int(self.ctx.table_found_frames), int(self.cfg.table_found_frames_to_approach))
            if level == "approach" and self._yaw_ready_for_controlled_approach(obs):
                self._transition(State.CONTROLLED_APPROACH, "warmup已有fresh plane obs，进入plane-only approach")
                return self.controller.fov_table_approach_cmd(obs, phase="PLANE_APPROACH")
            if level == "stop":
                return self._enter_final_lock_or_keep_approach(obs, "warmup已有fresh stop obs，进入最终停车")
            self._transition(State.COARSE_ALIGN, "warmup已有fresh plane obs，进入plane yaw/dist对齐")
            return self.controller.fov_table_approach_cmd(obs, phase="PLANE_FINAL_LOCK", mode="COARSE_ALIGN")

        self.ctx.table_found_frames = 0
        self._transition(State.SEARCH_TABLE, "warmup未获得稳定plane obs，进入搜索")
        return self.controller.stop_cmd("TABLE_APPROACH_WARMUP")

    def _tick_search_table(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        obs = self._fresh_table_obs()
        level = self._control_level(obs)
        if self._table_visible(obs) and (level != "none" or self._table_plane_stable(obs)):
            self.ctx.table_found_frames += 1
            if self.ctx.table_found_frames >= int(self.cfg.table_found_frames_to_approach):
                if level == "approach" and self._yaw_ready_for_controlled_approach(obs):
                    self._transition(State.CONTROLLED_APPROACH, "plane_confirmed_table_front，plane_only_approach")
                    return self.controller.fov_table_approach_cmd(obs, phase="PLANE_APPROACH")
                if level == "stop":
                    return self._enter_final_lock_or_keep_approach(obs, "plane-only stop 可用，进入最终停车")
                self._transition(State.COARSE_ALIGN, "plane_confirmed_table_front，进入plane yaw/dist对齐")
                return self.controller.fov_table_approach_cmd(obs, phase="PLANE_FINAL_LOCK", mode="COARSE_ALIGN")
        else:
            self.ctx.table_found_frames = 0
        if self._state_elapsed() >= float(self.cfg.search_table_timeout_s):
            self.ctx.last_fail_reason = "搜索桌边超时"
            self._transition(State.NEXT_TABLE, self.ctx.last_fail_reason)
            return self.controller.next_table_cmd(turn_sign=self.ctx.relocate_turn_sign)
        return self.controller.search_table_cmd(turn_sign=self.ctx.relocate_turn_sign)

    def _tick_coarse_align(self) -> MotionDecision:
        obs = self._fresh_table_obs()
        if not self._table_visible(obs):
            return self._handle_table_loss("桌边丢失，回到搜索", State.SEARCH_TABLE, "COARSE_ALIGN_HOLD")
        self._reset_table_loss()
        level = self._control_level(obs)
        if level == "none":
            return self.controller.stop_cmd("COARSE_ALIGN")
        if level == "stop":
            return self._enter_final_lock_or_keep_approach(obs, "plane-only stop 可用，进入停车确认")

        pending_reason = ""
        yaw_ready = self._yaw_ready_for_controlled_approach(obs)
        if self._count_table_motion_hysteresis_obs(
            obs,
            ok=yaw_ready,
            last_key_attr="align_hysteresis_last_obs_key",
            count_attr="approach_aligned_frames",
        ) >= self._align_to_approach_stable_obs():
            dwell_s = self._state_elapsed()
            min_dwell_s = self._coarse_align_min_dwell_s()
            if dwell_s >= min_dwell_s:
                self._transition(State.CONTROLLED_APPROACH, "plane yaw/dist稳定，开始plane-only接近")
                return self.controller.fov_table_approach_cmd(obs, phase="PLANE_APPROACH")
            pending_reason = f"align_to_approach_pending_min_dwell:{dwell_s:.2f}/{min_dwell_s:.2f}s"
            self.ctx.table_motion_pending_transition_reason = pending_reason
        else:
            self.ctx.table_motion_pending_transition_reason = ""
        if self._state_elapsed() >= float(self.cfg.approach_timeout_s):
            return self._enter_dock_retry_or_next("粗对齐超时")
        self._maybe_resend_req(self._active_req_payload())
        decision = self.controller.fov_table_approach_cmd(obs, phase="PLANE_FINAL_LOCK", mode="COARSE_ALIGN")
        return self._annotate_table_motion_hysteresis(decision, pending_reason=pending_reason)

    def _tick_controlled_approach(self) -> MotionDecision:
        obs = self._fresh_table_obs()
        if not self._table_visible(obs):
            return self._handle_table_loss("接近时桌边丢失，回到搜索", State.SEARCH_TABLE, "CONTROLLED_APPROACH_HOLD")
        self._reset_table_loss()
        pending_reason = ""
        yaw_needs_realign = self._yaw_needs_realign_from_approach(obs)
        if self._count_table_motion_hysteresis_obs(
            obs,
            ok=yaw_needs_realign,
            last_key_attr="approach_hysteresis_last_obs_key",
            count_attr="approach_realign_frames",
        ) >= self._approach_to_align_stable_obs():
            dwell_s = self._state_elapsed()
            min_dwell_s = self._controlled_approach_min_dwell_s()
            if dwell_s >= min_dwell_s:
                self._transition(State.COARSE_ALIGN, "approach yaw连续偏大，回到粗对齐")
                return self.controller.fov_table_approach_cmd(obs, phase="PLANE_FINAL_LOCK", mode="COARSE_ALIGN")
            pending_reason = f"approach_to_align_pending_min_dwell:{dwell_s:.2f}/{min_dwell_s:.2f}s"
            self.ctx.table_motion_pending_transition_reason = pending_reason
        else:
            self.ctx.table_motion_pending_transition_reason = ""
        level = self._control_level(obs)
        if level == "none":
            return self.controller.stop_cmd("CONTROLLED_APPROACH")
        if level == "approach":
            decision = self.controller.fov_table_approach_cmd(obs, phase="PLANE_APPROACH")
            return self._annotate_table_motion_hysteresis(decision, pending_reason=pending_reason)
        if level == "stop":
            return self._enter_final_lock_or_keep_approach(obs, "plane-only stop 可用，进入最终停车")
        if self._edge_ready(obs):
            return self._enter_final_lock_or_keep_approach(obs, "进入最终锁边")
        if self._state_elapsed() >= float(self.cfg.approach_timeout_s):
            return self._enter_dock_retry_or_next("受控接近超时")
        self._maybe_resend_req(self._active_req_payload())
        decision = self._table_approach_decision(obs, phase="PLANE_FINAL_LOCK")
        return self._annotate_table_motion_hysteresis(decision, pending_reason=pending_reason)

    def _tick_final_lock(self) -> MotionDecision:
        if not self._table_final_lock_enabled():
            obs = self._fresh_table_obs()
            if self._table_visible(obs):
                self._transition(State.CONTROLLED_APPROACH, "final_lock disabled，回到受控接近")
                return self._table_approach_decision(obs, phase="PLANE_APPROACH", stop_ready_ignored=True)
            return self._handle_table_loss("final_lock disabled 且桌边丢失，回到搜索", State.SEARCH_TABLE, "FINAL_LOCK_DISABLED_HOLD")
        obs = self._fresh_table_obs()
        if not self._table_visible(obs):
            stale_obs = self.ctx.last_table_obs if obs is None else obs
            status = self._update_final_lock_count(stale_obs if stale_obs is not None else obs)
            reason = str(status.get("reason") or "")
            self._log_final_lock_summary(stale_obs if stale_obs is not None else obs, lock_ready=False, reason=reason, stable_count=self.ctx.table_lock_frames, status=status)
            if str(status.get("lock_count_hold_reason") or ""):
                return self._annotate_final_lock_decision(self.controller.stop_cmd("FINAL_LOCK"), status)
            return self._handle_table_loss("最终锁边时桌边丢失，回到搜索", State.SEARCH_TABLE, "FINAL_LOCK_HOLD")
        self._reset_table_loss()

        phase = str(self.ctx.table_dock_phase or "APPROACH").upper()
        if phase == "APPROACH":
            status = self._update_final_lock_count(obs)
            self._log_final_lock_summary(
                obs,
                lock_ready=bool(status["lock_ready"]),
                reason=str(status["reason"]),
                stable_count=self.ctx.table_lock_frames,
                phase=phase,
                status=status,
            )
            level = str(status.get("normalized_control_level") or self._control_level(obs))
            if not self._table_micro_adjust_enabled() and str(status.get("reason") or "") == "distance_too_far":
                self._transition(State.CONTROLLED_APPROACH, "final_lock_distance_too_far_return_approach")
                return self._table_approach_decision(obs, phase="PLANE_APPROACH")
            if bool(status.get("final_lock_window_ready")):
                return self._final_lock_arrived_decision(obs, status)
            if bool(status["lock_ready"]) or level == "stop" or bool(getattr(obs, "usable_for_stop", False)):
                self.ctx.table_stop_sent = True
                self._enter_table_dock_phase("STOP_AND_SETTLE", "[TABLE_DOCK][STOP] final lock/stop condition reached")
                self._log("info", "[TABLE_DOCK][SETTLE] begin after STOP")
                return self._annotate_final_lock_decision(self.controller.stop_cmd("FINAL_LOCK"), status)
            if self._state_elapsed() >= float(self.cfg.approach_timeout_s):
                return self._enter_dock_retry_or_next(f"最终锁边超时:{status['reason']}")
            self._maybe_resend_req(self._active_req_payload())
            if level == "none":
                return self._annotate_final_lock_decision(self.controller.stop_cmd("FINAL_LOCK"), status)
            if level == "approach":
                return self.controller.plane_approach_cmd(obs, mode="FINAL_LOCK", reason="plane_final_approach")
            if level == "stop":
                return self.controller.fov_table_approach_cmd(obs, phase="PLANE_STOP", mode="FINAL_LOCK")
            return self.controller.fov_table_approach_cmd(obs, phase="PLANE_FINAL_LOCK", mode="FINAL_LOCK")

        if phase == "STOP_AND_SETTLE":
            settle_s = max(0.0, float(getattr(self.cfg, "table_settle_s", 0.30)))
            if monotonic_ts() - float(self.ctx.table_dock_phase_since_mono or 0.0) < settle_s:
                return self._annotate_final_lock_decision(
                    self.controller.stop_cmd("FINAL_LOCK"),
                    self._final_lock_status(obs, stable_count=self.ctx.table_lock_frames),
                )
            lock_status = self._update_final_lock_count(obs)
            self._log_final_lock_summary(
                obs,
                lock_ready=bool(lock_status["lock_ready"]),
                reason=str(lock_status["reason"]),
                stable_count=self.ctx.table_lock_frames,
                phase=phase,
                status=lock_status,
            )
            if bool(lock_status.get("final_lock_window_ready")):
                return self._final_lock_arrived_decision(obs, lock_status)
            if bool(lock_status["lock_ready"]):
                self._log(
                    "info",
                    "[TABLE_DOCK][STABLE] "
                    f"frames={self.ctx.table_lock_frames}/{self._required_lock_count()} "
                    f"dist_err={obs.dist_err_m} yaw_err={obs.yaw_err_rad}",
                )
                if self.ctx.table_lock_frames >= self._required_lock_count():
                    self.ctx.dock_retry_count = 0
                    self._capture_locked_edge(obs)
                    self._log("info", "[TABLE_DOCK][DONE] stable final lock confirmed")
                    if self._table_edge_only_test_enabled():
                        self._log("info", "[TABLE_EDGE_ONLY][DONE] table edge reached; stopping before target search")
                        self._transition(State.DONE, "table_edge_only_done")
                        self._queue_tts("桌边停靠测试完成")
                        return self._annotate_final_lock_decision(self.controller.stop_cmd("DONE"), lock_status)
                    self._transition(State.AT_TABLE_EDGE, "lock_ready")
                    self._queue_tts("已完成桌边停靠")
                    return self._annotate_final_lock_decision(self.controller.stop_cmd("AT_TABLE_EDGE"), lock_status)
                return self._annotate_final_lock_decision(self.controller.stop_cmd("FINAL_LOCK"), lock_status)

            hold_reason = str(lock_status.get("lock_count_hold_reason") or "")
            if hold_reason:
                return self._annotate_final_lock_decision(self.controller.stop_cmd("FINAL_LOCK"), lock_status)

            if not self._table_micro_adjust_enabled() and str(lock_status.get("reason") or "") == "distance_too_far":
                self._transition(State.CONTROLLED_APPROACH, "final_lock_distance_too_far_return_approach")
                return self._table_approach_decision(obs, phase="PLANE_APPROACH")

            self._enter_table_dock_phase("MICRO_ADJUST", f"[TABLE_DOCK][SETTLE] done reason={lock_status['reason']}")

        if str(self.ctx.table_dock_phase or "").upper() == "MICRO_ADJUST":
            decision = self._table_dock_micro_adjust(obs)
            if decision is not None:
                return decision

        if self._state_elapsed() >= float(self.cfg.approach_timeout_s):
            status = self._final_lock_status(obs, stable_count=self.ctx.table_lock_frames)
            self._log_final_lock_summary(obs, lock_ready=False, reason=str(status["reason"]), stable_count=self.ctx.table_lock_frames)
            return self._enter_dock_retry_or_next(f"最终锁边超时:{status['reason']}")
        self._maybe_resend_req(self._active_req_payload())
        return self._annotate_final_lock_decision(
            self.controller.stop_cmd("FINAL_LOCK"),
            self._final_lock_status(obs, stable_count=self.ctx.table_lock_frames),
        )

    def _final_lock_arrived_decision(self, obs: TableEdgeObs, status: Dict[str, object]) -> MotionDecision:
        status = dict(status or {})
        status.update(
            {
                "final_lock_transition_reason": "final_lock_window_ready",
                "final_lock_transition_block_reason": "",
                "ready_obs_count": int(status.get("lock_ready_obs_count", 0) or 0),
                "required_ready_obs": int(self._final_lock_required_ready_obs()),
                "window_ms": int(self._final_lock_window_ms()),
                "latest_yaw_err": getattr(obs, "yaw_err_rad", None),
                "latest_dist_err": getattr(obs, "dist_err_m", None),
                "latest_obs_age_ms": self._table_obs_age_ms(obs),
            }
        )
        self.ctx.dock_retry_count = 0
        self._capture_locked_edge(obs)
        self.ctx.final_lock_last_transition_reason = "final_lock_window_ready"
        self._log(
            "info",
            "[TABLE_DOCK][DONE] final_lock_window_ready "
            f"ready_obs_count={status['ready_obs_count']}/{status['required_ready_obs']} "
            f"window_ms={status['window_ms']} "
            f"latest_yaw_err={status.get('latest_yaw_err')} "
            f"latest_dist_err={status.get('latest_dist_err')} "
            f"latest_obs_age_ms={status.get('latest_obs_age_ms')}",
        )
        if self._table_edge_only_test_enabled():
            self._log("info", "[TABLE_EDGE_ONLY][DONE] table edge reached; stopping before target search")
            self._transition(State.DONE, "final_lock_window_ready")
            self._queue_tts("桌边停靠测试完成")
            return self._annotate_final_lock_decision(self.controller.stop_cmd("DONE"), status)
        self._transition(State.AT_TABLE_EDGE, "final_lock_window_ready")
        self._queue_tts("已完成桌边停靠")
        return self._annotate_final_lock_decision(self.controller.stop_cmd("AT_TABLE_EDGE"), status)

    def _tick_dock_retry(self) -> MotionDecision:
        if self._state_elapsed() < float(self.cfg.dock_retry_backoff_s):
            return self.controller.leave_edge_cmd()
        self._transition(State.SEARCH_TABLE, "停靠重试完成，重新搜索桌边")
        return self.controller.search_table_cmd(turn_sign=self.ctx.relocate_turn_sign)

    def _tick_at_table_edge(self) -> MotionDecision:
        if self._table_edge_only_test_enabled():
            if self._state_elapsed() < float(self.cfg.edge_settle_s):
                return self.controller.stop_cmd("AT_TABLE_EDGE")
            self._log("info", "[TABLE_EDGE_ONLY][DONE] table edge reached; stopping before target search")
            self._transition(State.DONE, "table_edge_only_done")
            self._queue_tts("桌边停靠测试完成")
            return self.controller.stop_cmd("DONE")
        if self._state_elapsed() < float(self.cfg.edge_settle_s):
            return self.controller.stop_cmd("AT_TABLE_EDGE")
        self._transition(State.SEARCH_TARGET_INIT, "桌边姿态稳定，初始化沿边搜索")
        return self.controller.stop_cmd("AT_TABLE_EDGE")

    def _tick_search_target_init(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        edge_obs = self._fresh_table_obs()
        if edge_obs is not None:
            self._maybe_add_slide_ref_sample(edge_obs)
        needed = max(1, int(getattr(self.cfg, "edge_handoff_samples", 3) or 3))
        min_s = max(float(self.cfg.search_target_init_hold_s), float(getattr(self.cfg, "edge_handoff_min_s", 0.5) or 0.5))
        max_s = max(min_s, float(getattr(self.cfg, "edge_handoff_max_s", 1.0) or 1.0))
        enough_samples = len(self.ctx.slide_ref_samples) >= needed
        if enough_samples and self._state_elapsed() >= min_s:
            self._finalize_slide_ref()
        if not self.ctx.slide_ref_ready:
            self.ctx.handoff_state = "collecting" if self._state_elapsed() < max_s or not enough_samples else "waiting_valid_light_edge"
            if self._state_elapsed() >= float(getattr(self.cfg, "reacquire_timeout_s", 8.0) or 8.0):
                if not self._table_final_lock_enabled():
                    self._transition(State.CONTROLLED_APPROACH, "slide_ref_handoff_timeout final_lock disabled")
                    return self._table_approach_decision(edge_obs, phase="PLANE_APPROACH", stop_ready_ignored=True)
                self._transition(State.FINAL_LOCK, "slide_ref_handoff_timeout")
                return self.controller.final_lock_cmd(edge_obs)
            return self.controller.stop_cmd("SEARCH_TARGET_INIT")
        if self._state_elapsed() < min_s:
            return self.controller.stop_cmd("SEARCH_TARGET_INIT")
        self._transition(State.EDGE_SLIDE_SEARCH, "开始沿桌边搜索目标 slide_ref_ready=1")
        self._queue_tts("开始沿桌边搜索目标")
        return self.controller.stop_cmd("SEARCH_TARGET_INIT")

    def _tick_edge_slide_search(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        target_obs = self._fresh_target_obs()
        candidate_ok, candidate_reason = self._target_candidate_status(
            target_obs,
            self.cfg.target_confirm_conf_th,
            min_area=self.cfg.target_confirm_min_bbox_area,
        )
        target_window = self._record_target_window_sample(target_obs, candidate_reason)
        if candidate_ok and target_obs is not None:
            self.ctx.target_found_frames += 1
            self.ctx.target_lost_frames = 0
            self._update_target_stability(target_obs)
            found_ratio_ok = (
                float(target_window.get("found_ratio", 0.0) or 0.0)
                >= float(getattr(self.cfg, "target_confirm_found_ratio_th", 0.5) or 0.5)
                and int(target_window.get("samples", 0) or 0) >= int(self.cfg.target_found_frames_to_confirm)
            )
            consecutive_ok = self.ctx.target_found_frames >= int(self.cfg.target_found_frames_to_confirm)
            if found_ratio_ok or consecutive_ok:
                self.ctx.target_last_transition_reason = (
                    f"confirm_enter found_ratio={float(target_window.get('found_ratio', 0.0) or 0.0):.2f} "
                    f"consecutive_frames={int(self.ctx.target_found_frames)} bbox_valid={int(self._target_bbox_valid(target_obs))}"
                )
                self._transition(
                    State.TARGET_CONFIRM,
                    self._format_target_transition_reason("target_found", target_obs),
                )
                return self.controller.stop_cmd("TARGET_CONFIRM")
        else:
            self.ctx.target_found_frames = 0
            self.ctx.target_last_lost_reason = candidate_reason
        if self._state_elapsed() >= float(self.cfg.target_search_timeout_s):
            self.ctx.last_fail_reason = "当前桌边未找到目标"
            if self._can_relocate_edge():
                self.ctx.advance_edge()
                self._transition(State.LEAVE_EDGE, f"{self.ctx.last_fail_reason}，切换到边 {self.ctx.current_edge_id}")
                self._queue_tts("当前边未找到目标，准备换边")
                return self.controller.leave_edge_cmd()
            self._transition(State.NEXT_TABLE, f"{self.ctx.last_fail_reason}，准备切换下一张桌")
            self._queue_tts("当前桌位未找到目标，尝试下一张桌")
            return self.controller.next_table_cmd(turn_sign=self.ctx.relocate_turn_sign)
        edge_obs = self._fresh_table_obs()
        if not self.ctx.slide_ref_ready:
            self._transition(State.SEARCH_TARGET_INIT, "slide_ref_missing")
            return self.controller.stop_cmd("SEARCH_TARGET_INIT")
        if edge_obs is None or not self._table_visible(edge_obs):
            return self._handle_edge_slide_edge_loss("edge_obs_missing" if edge_obs is None else "edge_not_visible")
        if self._edge_obs_is_stale(edge_obs):
            age_ms = self._table_obs_age_ms(edge_obs)
            age_text = "unknown" if age_ms is None else f"{age_ms:.0f}"
            self.ctx.last_fail_reason = f"edge_follow_stale age_ms={age_text}"
            self.ctx.last_edge_quality = {
                "mode": "stale",
                "reason": "edge_follow_stale",
                "fallback_candidate_state": self._edge_slide_stale_fallback_state().value,
                "fallback_decision": "stale_hold",
            }
            return self._handle_edge_slide_edge_loss(
                "edge_follow_stale",
                fallback_state=self._edge_slide_stale_fallback_state(),
                use_last_obs_for_fallback=False,
            )
        if not self._edge_valid_for_follow(edge_obs):
            reason = str(edge_obs.reason or "no_valid_edge").strip() or "no_valid_edge"
            self.ctx.last_fail_reason = reason
            quality = {
                "mode": "pause",
                "reason": reason,
                "fallback_candidate_state": self._edge_slide_fallback_state().value,
            }
            self.ctx.last_edge_quality = dict(quality)
            return self._edge_slide_pause_or_recover(edge_obs, quality)
        quality = self._edge_follow_quality(edge_obs)
        self.ctx.last_edge_quality = dict(quality)
        if str(quality.get("mode")) in {"identity_mismatch", "pause", "recover"}:
            return self._edge_slide_pause_or_recover(edge_obs, quality)
        if edge_obs.dist_err_m is None:
            return self._handle_edge_slide_edge_loss("dist_err_missing")
        if abs(float(edge_obs.dist_err_m)) > float(self.cfg.edge_slide_dist_tolerance_m):
            self._start_loss_timer("table_loss_since_mono")
            lost_s = self._loss_elapsed(self.ctx.table_loss_since_mono)
            dist = float(edge_obs.dist_err_m)
            tol = float(self.cfg.edge_slide_dist_tolerance_m)
            hold_s = float(getattr(self.cfg, "edge_slide_dist_out_of_range_hold_s", self.cfg.edge_slide_pause_hold_s) or self.cfg.edge_slide_pause_hold_s)
            warning = f"edge_distance_out_of_tolerance dist={dist:+.3f} tol={tol:.3f}"
            if warning not in self.ctx.task_warning_history:
                self.ctx.task_warning_history.append(warning)
            if lost_s >= hold_s:
                self.ctx.edge_slide_relock_attempts += 1
                max_attempts = max(0, int(getattr(self.cfg, "edge_slide_max_relock_attempts", 3) or 3))
                fatal = bool(getattr(self.cfg, "edge_slide_relock_failure_is_fatal", True))
                if fatal and self.ctx.edge_slide_relock_attempts > max_attempts:
                    reason = "edge_distance_out_of_tolerance_after_retries"
                    self.ctx.last_fail_reason = reason
                    self._enter_error_recovery(reason, tts_text="任务失败，无法稳定锁定桌边。", interrupt_tts=True)
                    return self.controller.stop_cmd("ERROR_RECOVERY", brake=True)
                fallback_state = self._edge_slide_fallback_state()
                self._transition(
                    fallback_state,
                    (
                        "edge_distance_out_of_tolerance "
                        f"dist={dist:+.3f} "
                        f"tol={tol:.3f} "
                        f"hold_s={lost_s:.2f} "
                        f"attempt={int(self.ctx.edge_slide_relock_attempts)}/{max_attempts} "
                        "recoverable=true severity=warning"
                    ),
                )
                return self._edge_slide_fallback_cmd(fallback_state, edge_obs)
            quality = dict(quality)
            pause_s = float(getattr(self.cfg, "edge_slide_pause_hold_s", 0.8) or 0.8)
            quality["mode"] = "pause" if lost_s < pause_s else "recover"
            quality["reason"] = "edge_distance_out_of_tolerance"
            quality["severity"] = "warning"
            quality["recoverable"] = True
            quality["dist_err_m"] = dist
            quality["dist_tolerance_m"] = tol
            quality["relock_attempts"] = int(self.ctx.edge_slide_relock_attempts)
            quality["max_relock_attempts"] = int(getattr(self.cfg, "edge_slide_max_relock_attempts", 3) or 3)
            self.ctx.last_edge_quality = dict(quality)
            decision = self.controller.edge_slide_hold_cmd(
                f"edge_distance_out_of_tolerance_{quality['mode']} elapsed_s={lost_s:.2f}",
                edge_obs=edge_obs,
            )
            return self._annotate_edge_slide_decision(
                decision,
                quality,
                stop_reason="edge_distance_out_of_tolerance",
                fallback_decision="relock_hold",
                pause_elapsed_s=min(lost_s, pause_s),
                recover_elapsed_s=max(0.0, lost_s - pause_s),
            )
        else:
            self._reset_table_loss()
        if self._obs_has_motion(target_obs) and self._target_quality_ok(target_obs, self.cfg.target_confirm_conf_th):
            return self.controller.target_track_cmd(target_obs)
        direction = self._edge_slide_direction()
        quality_mode = str(quality.get("mode") or "strong")
        vy_norm = None
        reason = "edge_slide"
        if quality_mode == "weak":
            vy_norm = float(getattr(self.controller.car_cfg, "edge_slide_weak_vy_norm", 0.05) or 0.05)
            reason = "weak_edge_slide"
        else:
            self._reset_table_loss()
        decision = self.controller.edge_slide_search_cmd(
            self._segment_elapsed(self.cfg.edge_slide_segment_s),
            direction_sign=direction,
            edge_obs=edge_obs,
            vy_norm=vy_norm,
            reason=reason,
        )
        return self._annotate_edge_slide_decision(decision, quality, fallback_decision="slide")

    def _edge_slide_fallback_state(self) -> State:
        if not self._table_final_lock_enabled():
            return State.CONTROLLED_APPROACH
        raw = str(getattr(self.cfg, "edge_slide_fallback_state", "") or "").strip().upper()
        direct = bool(getattr(self.cfg, "edge_slide_direct_fallback_to_controlled_approach", False))
        if direct and raw == State.CONTROLLED_APPROACH.value:
            return State.CONTROLLED_APPROACH
        return State.FINAL_LOCK

    def _edge_slide_stale_fallback_state(self) -> State:
        if not self._table_final_lock_enabled():
            return State.CONTROLLED_APPROACH
        raw = str(getattr(self.cfg, "edge_follow_stale_fallback_state", "") or "").strip().upper()
        direct = bool(getattr(self.cfg, "edge_slide_direct_fallback_to_controlled_approach", False))
        if direct and raw == State.CONTROLLED_APPROACH.value:
            return State.CONTROLLED_APPROACH
        return State.FINAL_LOCK

    def _edge_slide_fallback_cmd(self, state: State, edge_obs: Optional[TableEdgeObs]) -> MotionDecision:
        if state == State.FINAL_LOCK:
            return self.controller.final_lock_cmd(edge_obs)
        return self.controller.controlled_approach_cmd(edge_obs)

    def _annotate_edge_slide_decision(
        self,
        decision: MotionDecision,
        quality: Dict[str, Any],
        *,
        stop_reason: str = "",
        fallback_decision: str = "",
        pause_elapsed_s: Optional[float] = None,
        recover_elapsed_s: Optional[float] = None,
    ) -> MotionDecision:
        summary = dict(decision.control_summary or {})
        cmd = decision.cmd
        summary.update(
            {
                "edge_quality_mode": quality.get("mode"),
                "stop_reason": stop_reason or summary.get("stop_reason") or "",
                "slide_vy_norm": float(getattr(self.controller.car_cfg, "edge_slide_vy_norm", 0.0) or 0.0),
                "weak_slide_vy_norm": float(getattr(self.controller.car_cfg, "edge_slide_weak_vy_norm", 0.0) or 0.0),
                "final_vx": float(cmd.vx_norm),
                "final_vy": float(cmd.vy_norm),
                "final_wz": float(cmd.wz_norm),
                "pause_elapsed_ms": int(round(max(0.0, float(pause_elapsed_s or 0.0)) * 1000.0)),
                "recover_elapsed_ms": int(round(max(0.0, float(recover_elapsed_s or 0.0)) * 1000.0)),
                "fallback_candidate_state": quality.get("fallback_candidate_state", self._edge_slide_fallback_state().value),
                "fallback_decision": fallback_decision or ("none" if abs(float(cmd.vy_norm or 0.0)) > 0.0 else "hold"),
                "severity": quality.get("severity"),
                "recoverable": quality.get("recoverable"),
                "dist_tolerance_m": quality.get("dist_tolerance_m"),
                "relock_attempts": quality.get("relock_attempts"),
                "max_relock_attempts": quality.get("max_relock_attempts"),
            }
        )
        if "vx_from_dist" not in summary:
            summary["vx_from_dist"] = float(cmd.vx_norm)
        if "wz_from_yaw" not in summary:
            summary["wz_from_yaw"] = float(cmd.wz_norm)
        decision.control_summary = summary
        return decision

    def _edge_slide_pause_or_recover(self, edge_obs: TableEdgeObs, quality: Dict[str, Any]) -> MotionDecision:
        reason = str(quality.get("reason") or quality.get("mode") or "edge_uncertain")
        if reason in {"edge_identity_mismatch", "edge_follow_stale", "edge_conf_low", "conf_low", "target_lost"}:
            warning = f"{reason} recoverable=true severity=warning"
            if warning not in self.ctx.task_warning_history:
                self.ctx.task_warning_history.append(warning)
        else:
            self.ctx.last_fail_reason = reason
        self._start_loss_timer("table_loss_since_mono")
        elapsed_s = self._loss_elapsed(self.ctx.table_loss_since_mono)
        pause_s = float(getattr(self.cfg, "edge_slide_pause_hold_s", 0.8) or 0.8)
        recover_timeout_s = float(getattr(self.cfg, "edge_slide_recover_timeout_s", self.cfg.table_loss_hold_s) or self.cfg.table_loss_hold_s)
        quality = dict(quality)
        if elapsed_s >= recover_timeout_s:
            fallback_state = self._edge_slide_fallback_state()
            quality["mode"] = "recover"
            quality["fallback_decision"] = f"fallback_to_{fallback_state.value}"
            self.ctx.last_edge_quality = dict(quality)
            self._transition(fallback_state, f"{reason} recover_timeout_s={elapsed_s:.2f}")
            return self._edge_slide_fallback_cmd(fallback_state, edge_obs)
        if elapsed_s >= pause_s:
            quality["mode"] = "recover"
            control_reason = f"edge_recover stop_reason={reason} elapsed_s={elapsed_s:.2f}"
            fallback_decision = "recover_hold"
        else:
            quality["raw_mode"] = quality.get("mode")
            quality["mode"] = "pause"
            control_reason = f"edge_pause stop_reason={reason} elapsed_s={elapsed_s:.2f}"
            fallback_decision = "pause_hold"
        quality["pause_elapsed_ms"] = int(round(min(elapsed_s, pause_s) * 1000.0))
        quality["recover_elapsed_ms"] = int(round(max(0.0, elapsed_s - pause_s) * 1000.0))
        quality["fallback_decision"] = fallback_decision
        self.ctx.last_edge_quality = dict(quality)
        decision = self.controller.edge_slide_hold_cmd(control_reason, edge_obs=edge_obs)
        return self._annotate_edge_slide_decision(
            decision,
            quality,
            stop_reason=reason,
            fallback_decision=fallback_decision,
            pause_elapsed_s=min(elapsed_s, pause_s),
            recover_elapsed_s=max(0.0, elapsed_s - pause_s),
        )

    def _handle_edge_slide_edge_loss(
        self,
        reason: str,
        fallback_state: Optional[State] = None,
        use_last_obs_for_fallback: bool = True,
    ) -> MotionDecision:
        self._start_loss_timer("table_loss_since_mono")
        lost_s = self._loss_elapsed(self.ctx.table_loss_since_mono)
        hold_s = self._edge_slide_loss_hold_s(reason)
        if lost_s < hold_s:
            return self.controller.edge_slide_hold_cmd(f"{reason}_hold lost_s={lost_s:.2f}")
        fallback_state = fallback_state or self._edge_slide_fallback_state()
        self._transition(fallback_state, f"{reason} lost_s={lost_s:.2f}")
        fallback_obs = self.ctx.last_table_obs if use_last_obs_for_fallback else None
        return self._edge_slide_fallback_cmd(fallback_state, fallback_obs)

    def _edge_slide_loss_hold_s(self, reason: str) -> float:
        raw = str(reason or "").strip()
        if raw.startswith("edge_follow_stale") or self._edge_obs_is_stale(self.ctx.last_table_obs):
            return float(getattr(self.cfg, "edge_follow_stale_hold_s", self.cfg.table_loss_hold_s) or self.cfg.table_loss_hold_s)
        return float(self.cfg.table_loss_hold_s)

    def _tick_target_confirm(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        obs = self._fresh_target_obs()
        visible_ok, visible_reason = self._target_candidate_status(
            obs,
            self.cfg.target_confirm_conf_th,
            min_area=self.cfg.target_confirm_min_bbox_area,
        )
        target_window = self._record_target_window_sample(obs, visible_reason)
        lock_ok, lock_reason = self._target_candidate_status(obs, self.cfg.target_lock_conf_th, min_area=0.0)
        if visible_ok and obs is not None:
            self.ctx.target_found_frames += 1
            self.ctx.target_lost_frames = 0
            center_jitter = self._update_target_stability(obs)
            window_jitter = self._float_or_none(target_window.get("center_jitter"))
            if window_jitter is not None:
                center_jitter = window_jitter
                self.ctx.target_last_center_jitter = float(window_jitter)
            confirm_elapsed_s = self._state_elapsed()
            confirm_min_ok = confirm_elapsed_s >= float(self.cfg.target_confirm_min_s)
            found_ratio = float(target_window.get("found_ratio", 0.0) or 0.0)
            conf_median = self._float_or_none(target_window.get("conf_median"))
            stable_ok = self._target_stable_ms() >= int(round(float(self.cfg.target_lock_stable_s) * 1000.0))
            jitter_ok = center_jitter <= float(self.cfg.target_lock_center_jitter_th)
            ratio_ok = found_ratio >= float(getattr(self.cfg, "target_lock_found_ratio_th", 0.6) or 0.6)
            conf_ok = conf_median is not None and conf_median >= float(self.cfg.target_lock_conf_th or 0.0)
            if lock_ok and confirm_min_ok and stable_ok and jitter_ok and ratio_ok and conf_ok:
                self.ctx.target_last_transition_reason = (
                    f"lock_ok found_ratio={found_ratio:.2f} conf_median={float(conf_median):.3f} "
                    f"center_jitter={float(center_jitter):.3f} stable_ms={self._target_stable_ms()}"
                )
                self._transition(
                    State.TARGET_LOCKED,
                    self._format_target_transition_reason("target_confirmed", obs),
                )
                return self.controller.stop_cmd("TARGET_LOCKED")
            if self._state_elapsed() >= float(self.cfg.target_confirm_timeout_s):
                reasons = []
                if not lock_ok:
                    reasons.append(lock_reason)
                if not confirm_min_ok:
                    reasons.append("confirm_min_not_reached")
                if not stable_ok:
                    reasons.append("target_stable_not_enough")
                if not jitter_ok:
                    reasons.append("center_jitter_high")
                if not ratio_ok:
                    reasons.append(f"found_ratio_low ratio={found_ratio:.2f}")
                if not conf_ok:
                    reasons.append("conf_median_low")
                self.ctx.target_last_lost_reason = ",".join(reasons) or "lock_condition_timeout"
                self._transition(
                    State.EDGE_SLIDE_SEARCH,
                    self._format_target_transition_reason("confirm_timeout", obs),
                )
            return self.controller.stop_cmd("TARGET_CONFIRM")
        self.ctx.target_found_frames = 0
        self.ctx.target_lost_frames += 1
        self._start_loss_timer("target_loss_since_mono")
        lost_s = self._loss_elapsed(self.ctx.target_loss_since_mono)
        self.ctx.target_last_lost_reason = f"{visible_reason} lost_hold_ms={int(round(lost_s * 1000.0))}"
        if self._state_elapsed() < float(self.cfg.target_confirm_min_s):
            return self.controller.stop_cmd("TARGET_CONFIRM")
        if lost_s >= float(self.cfg.target_confirm_lost_hold_s):
            self._reset_target_stability(visible_reason)
            self._transition(
                State.EDGE_SLIDE_SEARCH,
                self._format_target_transition_reason("confirm_lost_hold_exceeded", obs),
            )
        return self.controller.stop_cmd("TARGET_CONFIRM")

    def _tick_target_locked(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        obs = self._fresh_target_obs()
        lock_ok, lock_reason = self._target_candidate_status(obs, self.cfg.target_lock_conf_th, min_area=0.0)
        target_window = self._record_target_window_sample(obs, lock_reason)
        if not lock_ok or obs is None:
            self.ctx.target_lost_frames += 1
            self._start_loss_timer("target_loss_since_mono")
            lost_s = self._loss_elapsed(self.ctx.target_loss_since_mono)
            self.ctx.target_last_lost_reason = f"{lock_reason} lost_hold_ms={int(round(lost_s * 1000.0))}"
            if lost_s >= float(self.cfg.target_lock_lost_hold_s):
                self._reset_target_stability(lock_reason)
                self._transition(
                    State.EDGE_SLIDE_SEARCH,
                    self._format_target_transition_reason("locked_lost_hold_exceeded", obs),
                )
                return self.controller.stop_cmd("EDGE_SLIDE_SEARCH")
            return self.controller.stop_cmd("TARGET_LOCKED")
        self.ctx.target_lost_frames = 0
        self.ctx.target_lock_frames += 1
        center_jitter = self._update_target_stability(obs)
        window_jitter = self._float_or_none(target_window.get("center_jitter"))
        if window_jitter is not None:
            center_jitter = window_jitter
            self.ctx.target_last_center_jitter = float(window_jitter)
        found_ratio = float(target_window.get("found_ratio", 0.0) or 0.0)
        conf_median = self._float_or_none(target_window.get("conf_median"))
        stable_ok = self._target_stable_ms() >= int(round(float(self.cfg.target_lock_stable_s) * 1000.0))
        jitter_ok = center_jitter <= float(self.cfg.target_lock_center_jitter_th)
        conf_stable = conf_median is not None and conf_median >= float(self.cfg.target_lock_conf_th or 0.0)
        ratio_ok = found_ratio >= float(getattr(self.cfg, "target_lock_found_ratio_th", 0.6) or 0.6)
        if (
            self._state_elapsed() >= float(self.cfg.target_locked_freeze_after_s)
            and stable_ok
            and jitter_ok
            and conf_stable
            and ratio_ok
            and self.ctx.target_loss_since_mono <= 0.0
        ):
            self.ctx.target_last_transition_reason = (
                f"freeze_ok found_ratio={found_ratio:.2f} conf_median={float(conf_median):.3f} "
                f"center_jitter={float(center_jitter):.3f} stable_ms={self._target_stable_ms()}"
            )
            self._transition(
                State.FREEZE_BASE,
                self._format_target_transition_reason("locked_stable_freeze", obs),
            )
            return self.controller.stop_cmd("FREEZE_BASE")
        return self.controller.stop_cmd("TARGET_LOCKED")

    def _tick_freeze_base(self) -> MotionDecision:
        if self._state_elapsed() < float(self.cfg.freeze_settle_s):
            return self.controller.stop_cmd("FREEZE_BASE")
        if self.ctx.task_intent == "FIND" and self.ctx.active_target:
            self._transition(State.GRASP, f"已锁定 {self.ctx.active_target}，开始抓取")
            self._queue_tts(f"已锁定 {self.ctx.active_target}，开始抓取")
            return self.controller.stop_cmd("GRASP")
        self._transition(State.DONE, f"已在桌边锁定 {self.ctx.active_target}")
        self._queue_tts(f"已在桌边锁定 {self.ctx.active_target}")
        return self.controller.stop_cmd("DONE")

    def _tick_grasp(self) -> MotionDecision:
        now_m = monotonic_ts()
        substate = str(self.ctx.grasp_substate or "")

        if substate == "AWAITING_RESPOND":
            return self._tick_grasp_awaiting_respond(now_m)
        if substate == "AWAITING_RESULT":
            return self._tick_grasp_awaiting_result(now_m)
        if substate == "REPOSITIONING":
            return self._tick_grasp_repositioning(now_m)
        if substate == "AWAITING_ARM":
            return self._tick_grasp_awaiting_arm(now_m)
        if substate == "GRASP_VERIFY":
            return self._tick_grasp_verify(now_m)
        return self.controller.stop_cmd("GRASP")

    def _tick_grasp_awaiting_respond(self, now_m: float) -> MotionDecision:
        if self._state_elapsed() < 0.3:
            return self.controller.stop_cmd("GRASP")
        if now_m > self.ctx.grasp_timeout_mono:
            self._enter_error_recovery("grasp respond timeout")
            return self.controller.stop_cmd("GRASP")
        if self.ctx.grasp_status == "WAITING_RESPONSE":
            self._queue_vision_req(
                make_grasp_req(
                    target=self.ctx.active_target or "",
                    class_id=target_to_class_id(self.ctx.active_target or ""),
                    session_id=self.ctx.active_session_id,
                    epoch=self.ctx.active_epoch,
                    op="RESPOND",
                ),
                force=True,
            )
            self.ctx.grasp_substate = "AWAITING_RESULT"
            self.ctx.grasp_timeout_mono = now_m + _GRASP_RESULT_TIMEOUT_S
        return self.controller.stop_cmd("GRASP")

    def _tick_grasp_awaiting_result(self, now_m: float) -> MotionDecision:
        if now_m > self.ctx.grasp_timeout_mono:
            self._enter_error_recovery("grasp result timeout")
            return self.controller.stop_cmd("GRASP")

        status = str(self.ctx.grasp_status or "").upper()

        if status == "RESULT_READY" and isinstance(self.ctx.grasp_result, dict):
            arm_cmd = grasp_to_pose_params(self.ctx.grasp_result, time_ms=500)
            self.ctx.grasp_substate = "AWAITING_ARM"
            self.ctx.grasp_timeout_mono = now_m + _GRASP_ARM_TIMEOUT_S
            return MotionDecision(cmd=self.controller.stop_cmd("GRASP").cmd, arm_cmd=arm_cmd)

        if status == "RUNNING" and self.ctx.grasp_reposition_proposal is not None:
            self.ctx.grasp_retry_count += 1
            if self.ctx.grasp_retry_count > _GRASP_RETRY_LIMIT:
                self._enter_error_recovery("grasp reposition retries exhausted")
                return self.controller.stop_cmd("GRASP")
            self.ctx.grasp_substate = "REPOSITIONING"
            self.ctx.grasp_timeout_mono = now_m + _GRASP_REPOSITION_TIMEOUT_S
            self.ctx.grasp_reposition_start_mono = now_m
            return self.controller.stop_cmd("GRASP")

        if status == "FAILED":
            reason = str(self.ctx.grasp_reason or "")
            if reason == "no_detection":
                self._transition(State.SEARCH_TARGET_INIT, "grasp failed: target not detected")
                return self.controller.stop_cmd("SEARCH_TARGET_INIT")
            self._enter_error_recovery(reason or "grasp failed")
            return self.controller.stop_cmd("GRASP")

        return self.controller.stop_cmd("GRASP")

    def _tick_grasp_repositioning(self, now_m: float) -> MotionDecision:
        if now_m > self.ctx.grasp_timeout_mono:
            self._enter_error_recovery("grasp reposition timeout")
            return self.controller.stop_cmd("GRASP")

        proposal = self.ctx.grasp_reposition_proposal
        if not isinstance(proposal, dict):
            self.ctx.grasp_substate = "AWAITING_RESPOND"
            self.ctx.grasp_timeout_mono = now_m + _GRASP_RESPOND_TIMEOUT_S
            return self.controller.stop_cmd("GRASP")

        dx = float(proposal.get("dx_cm", 0.0) or 0.0)
        dy = float(proposal.get("dy_cm", 0.0) or 0.0)
        distance = max(abs(dx), abs(dy))
        if distance < 0.5:
            self.ctx.grasp_reposition_proposal = None
            self.ctx.grasp_substate = "AWAITING_RESPOND"
            self.ctx.grasp_timeout_mono = now_m + _GRASP_RESPOND_TIMEOUT_S
            self._queue_vision_req(
                make_grasp_req(
                    target=self.ctx.active_target or "",
                    class_id=target_to_class_id(self.ctx.active_target or ""),
                    session_id=self.ctx.active_session_id,
                    epoch=self.ctx.active_epoch,
                    op="START",
                ),
                force=True,
            )
            return self.controller.stop_cmd("GRASP")

        duration = min(distance / _GRASP_REPOSITION_SPEED_CM_S, 2.0)
        elapsed = now_m - self.ctx.grasp_reposition_start_mono
        if elapsed < duration:
            vx = _GRASP_REPOSITION_SPEED if dx > 0 else (-_GRASP_REPOSITION_SPEED if dx < 0 else 0.0)
            vy = _GRASP_REPOSITION_SPEED if dy > 0 else (-_GRASP_REPOSITION_SPEED if dy < 0 else 0.0)
            cmd = self.controller._cmd("GRASP_REPOSITION", vx=vx, vy=vy, wz=0.0)
            return MotionDecision(cmd=cmd, control_summary=self.controller._summary(
                "GRASP_REPOSITION", cmd, reason=f"reposition dx={dx:.1f} dy={dy:.1f}"
            ))

        self.ctx.grasp_reposition_proposal = None
        self.ctx.grasp_substate = "AWAITING_RESPOND"
        self.ctx.grasp_timeout_mono = now_m + _GRASP_RESPOND_TIMEOUT_S
        self._queue_vision_req(
            make_grasp_req(
                target=self.ctx.active_target or "",
                class_id=target_to_class_id(self.ctx.active_target or ""),
                session_id=self.ctx.active_session_id,
                epoch=self.ctx.active_epoch,
                op="START",
            ),
            force=True,
        )
        return self.controller.stop_cmd("GRASP")

    def _tick_grasp_awaiting_arm(self, now_m: float) -> MotionDecision:
        if now_m > self.ctx.grasp_timeout_mono:
            self._enter_error_recovery("arm response timeout")
            return self.controller.stop_cmd("GRASP")

        resp = self.ctx.arm_response
        if resp is not None:
            if resp.ok:
                self.ctx.grasp_substate = "GRASP_VERIFY"
                self.ctx.grasp_timeout_mono = now_m + 3.0
                self.ctx.grasp_verify_reported = False
                self.ctx.arm_response = None
                return self.controller.stop_cmd("GRASP")
            self.ctx.grasp_retry_count += 1
            if self.ctx.grasp_retry_count > _GRASP_RETRY_LIMIT:
                self._enter_error_recovery("arm IK exhausted")
                return self.controller.stop_cmd("GRASP")
            self.ctx.grasp_substate = "AWAITING_RESPOND"
            self.ctx.grasp_timeout_mono = now_m + _GRASP_RESPOND_TIMEOUT_S
            self.ctx.arm_response = None

        return self.controller.stop_cmd("GRASP")

    def _tick_grasp_verify(self, now_m: float) -> MotionDecision:
        status = str(self.ctx.grasp_status or "").strip().upper()
        result = self.ctx.grasp_result if isinstance(self.ctx.grasp_result, dict) else {}
        explicit_success = result.get("verify_success")
        if explicit_success is None:
            explicit_success = result.get("grasp_success")

        if bool(getattr(self.cfg, "assume_grasp_success_for_test", False)):
            self._log("info", "[GRASP][VERIFY_ASSUMED_SUCCESS] ORCH_ASSUME_GRASP_SUCCESS_FOR_TEST=1")
            self._transition(State.RETURN_HOME, "grasp verify assumed success")
            self._queue_tts("抓取完成，开始返航")
            return self.controller.stop_cmd("RETURN_HOME")

        if status in {"VERIFY_OK", "VERIFIED", "GRASP_VERIFIED", "SUCCESS"} or explicit_success is True:
            self._transition(State.RETURN_HOME, "grasp verified successfully")
            self._queue_tts("抓取完成，开始返航")
            return self.controller.stop_cmd("RETURN_HOME")

        if status in {"VERIFY_FAILED", "GRASP_VERIFY_FAILED"} or explicit_success is False:
            return self._handle_grasp_verify_failed("grasp verify failed")

        if not self.ctx.grasp_verify_reported:
            self._log("warn", "[GRASP][VERIFY_UNAVAILABLE] no real grasp verification source; not assuming success")
            self.ctx.grasp_verify_reported = True

        if now_m > self.ctx.grasp_timeout_mono:
            return self._handle_grasp_verify_failed("grasp verification unavailable")

        return self.controller.stop_cmd("GRASP")

    def _handle_grasp_verify_failed(self, reason: str) -> MotionDecision:
        self.ctx.grasp_retry_count += 1
        if self.ctx.grasp_retry_count > _GRASP_RETRY_LIMIT:
            self._enter_error_recovery(reason)
            return self.controller.stop_cmd("GRASP")
        self.ctx.grasp_substate = "AWAITING_RESPOND"
        self.ctx.grasp_timeout_mono = monotonic_ts() + _GRASP_RESPOND_TIMEOUT_S
        self.ctx.grasp_status = ""
        self.ctx.grasp_result = None
        self.ctx.arm_response = None
        self.ctx.grasp_verify_reported = False
        return self.controller.stop_cmd("GRASP")

    def _tick_leave_edge(self) -> MotionDecision:
        if self._state_elapsed() < float(self.cfg.leave_edge_backoff_s):
            return self.controller.leave_edge_cmd()
        self._transition(State.RELOCATE_TO_EDGE, f"准备重定位到边 {self.ctx.current_edge_id}")
        return self.controller.relocate_cmd(turn_sign=self.ctx.relocate_turn_sign)

    def _tick_relocate_to_edge(self) -> MotionDecision:
        if self._state_elapsed() < float(self.cfg.relocate_turn_s):
            return self.controller.relocate_cmd(turn_sign=self.ctx.relocate_turn_sign)
        self._transition(State.REACQUIRE_EDGE, f"开始重捕获边 {self.ctx.current_edge_id}")
        return self.controller.search_table_cmd(turn_sign=self.ctx.relocate_turn_sign)

    def _tick_reacquire_edge(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        obs = self._fresh_table_obs()
        if self._table_visible(obs):
            self.ctx.table_found_frames += 1
            if self.ctx.table_found_frames >= int(self.cfg.table_found_frames_to_approach):
                self._transition(State.COARSE_ALIGN, f"已重捕获边 {self.ctx.current_edge_id}")
                return self.controller.fov_table_approach_cmd(obs, phase="PLANE_FINAL_LOCK", mode="COARSE_ALIGN")
        else:
            self.ctx.table_found_frames = 0
        if self._state_elapsed() >= float(self.cfg.reacquire_timeout_s):
            self.ctx.last_fail_reason = f"重捕获边 {self.ctx.current_edge_id} 超时"
            self._transition(State.NEXT_TABLE, self.ctx.last_fail_reason)
            return self.controller.next_table_cmd(turn_sign=self.ctx.relocate_turn_sign)
        return self.controller.search_table_cmd(turn_sign=self.ctx.relocate_turn_sign)

    def _tick_next_table(self) -> MotionDecision:
        if self._state_elapsed() >= float(self.cfg.next_table_dwell_s):
            self.ctx.table_cycle_count += 1
            self.ctx.reset_edge_plan()
            self._transition(State.SEARCH_TABLE, "切换到下一张桌后重新搜索")
            return self.controller.search_table_cmd(turn_sign=self.ctx.relocate_turn_sign)
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
                return self.controller.search_table_cmd(turn_sign=self.ctx.relocate_turn_sign)
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
            self._queue_vision_req(make_vision_idle(session_id=self.ctx.active_session_id, epoch=self.ctx.active_epoch), force=True)
            self._transition(State.IDLE, "错误恢复完成，回到空闲")
        return self.controller.stop_cmd("ERROR_RECOVERY", brake=True)

    def _tick_done(self) -> MotionDecision:
        if self._state_elapsed() >= float(self.cfg.done_hold_s):
            self._queue_vision_req(make_vision_idle(session_id=self.ctx.active_session_id, epoch=self.ctx.active_epoch), force=True)
            self._transition(State.IDLE, "任务完成，回到空闲")
        return self.controller.stop_cmd("DONE")

    def _table_edge_only_test_enabled(self) -> bool:
        return bool(getattr(self.cfg, "table_edge_only_test", False))

    def _active_req_payload(self) -> Optional[Dict]:
        binding = self._vision_binding_for_state(self.ctx.state)
        if binding is None:
            return None
        prev_stage = str(self.ctx.active_vision_stage or "").strip().upper()
        prev_mode = str(self.ctx.active_vision_mode or "").strip().upper()
        next_stage = str(binding.stage or "").strip().upper()
        next_mode = str(binding.mode_hint or "").strip().upper()
        changed_mode_level = (not prev_stage) or prev_stage != next_stage or prev_mode != next_mode
        req_type = "mode_request" if changed_mode_level else "target_update"
        op = "START" if req_type == "mode_request" else "UPDATE"
        self.ctx.active_vision_stage = binding.stage
        self.ctx.active_vision_mode = binding.mode_hint
        payload = dict(binding.payload or {})
        payload["req_type"] = req_type
        payload["request_reason"] = "vision_mode_changed" if req_type == "mode_request" else "target_or_stage_update"
        return make_vision_req(
            target=binding.target,
            session_id=self.ctx.active_session_id,
            epoch=self.ctx.active_epoch,
            op=op,
            stage=binding.stage,
            mode_hint=binding.mode_hint,
            req_type=req_type,
            payload=payload,
        )

    def _vision_request_key(self, payload: Dict, *, req_type: str) -> str:
        req_payload = dict(payload.get("payload") or {})
        stage = str(payload.get("stage") or "").strip().upper()
        mode = str(payload.get("mode_hint") or "").strip().upper()
        session_id = str(payload.get("session_id") or self.ctx.active_session_id or "").strip()
        target = str(payload.get("target") or self.ctx.active_target or "").strip()
        if req_type == "mode_request":
            return "|".join([req_type, session_id, target, stage, mode])
        roi = req_payload.get("locked_roi") or req_payload.get("roi") or req_payload.get("target_roi") or []
        return "|".join(
            [
                req_type,
                session_id,
                target,
                stage,
                mode,
                str(req_payload.get("search_kind") or "").strip().upper(),
                str(req_payload.get("current_edge_id") or "").strip(),
                str(req_payload.get("locked_edge_id") or "").strip(),
                repr(roi),
            ]
        )

    def _vision_binding_for_state(self, state: State) -> Optional[VisionStageBinding]:
        if state in TABLE_VISION_STATES:
            return VisionStageBinding(
                stage="SEARCH",
                mode_hint="FIND_EDGE",
                target=None,
                payload={
                    "search_kind": "TABLE_EDGE",
                    "need_depth": True,
                    "current_edge_id": self.ctx.current_edge_id,
                    "orchestrator_state": state.value,
                    "table_cycle_count": int(self.ctx.table_cycle_count),
                    "edge_visit_index": int(self.ctx.edge_visit_index),
                },
            )
        if state in TARGET_VISION_STATES or state == State.AT_TABLE_EDGE:
            return VisionStageBinding(
                stage="SEARCH",
                mode_hint="FIND_OBJECT",
                target=self.ctx.active_target,
                payload={
                    "search_kind": "TARGET",
                    "need_depth": True,
                    "edge_follow": True,
                    "track_local_edge_update_hz": float(getattr(self.cfg, "edge_follow_track_local_edge_update_hz", 5.0) or 5.0),
                    "current_edge_id": self.ctx.current_edge_id,
                    "locked_edge_id": self.ctx.locked_edge_id,
                    "locked_edge_line": dict(self.ctx.locked_edge_line or {}),
                    "locked_roi": list(self.ctx.locked_roi or []),
                    "locked_yaw_err": self.ctx.locked_yaw_err,
                    "locked_dist_err": self.ctx.locked_dist_err,
                    "locked_edge_conf": self.ctx.locked_edge_conf,
                    "locked_obs_seq": self.ctx.locked_obs_seq,
                    "orchestrator_state": state.value,
                    "edge_visit_index": int(self.ctx.edge_visit_index),
                },
            )
        if state == State.GRASP:
            class_id = target_to_class_id(self.ctx.active_target or "")
            return VisionStageBinding(
                stage="GRASP",
                mode_hint="GRASP_REMOTE",
                target=self.ctx.active_target,
                payload={
                    "class_id": class_id,
                    "remote_grasp": True,
                    "need_depth": True,
                    "orchestrator_state": state.value,
                },
            )
        if state == State.RETURN_HOME:
            return VisionStageBinding(
                stage="RETURN",
                mode_hint="FIND_OBJECT",
                target=None,
                payload={
                    "search_kind": "HOME_TAG",
                    "orchestrator_state": state.value,
                },
            )
        return None

    def _fresh_table_obs(self) -> Optional[TableEdgeObs]:
        obs = self.ctx.last_table_obs
        if obs is None or time.time() - obs.ts > self.cfg.table_obs_max_age_s:
            return None
        if self.ctx.task_start_wall_ts > 0 and obs.ts < self.ctx.task_start_wall_ts:
            return None
        if obs.session_id and self.ctx.active_session_id and obs.session_id != self.ctx.active_session_id:
            return None
        return obs

    @staticmethod
    def _table_obs_key(obs: Optional[TableEdgeObs]) -> str:
        if obs is None:
            return ""
        seq = getattr(obs, "seq", None)
        frame_id = getattr(obs, "frame_id", None)
        obs_ts = getattr(obs, "obs_ts", None) or getattr(obs, "ts", None)
        return f"{seq}:{frame_id}:{obs_ts}"

    def _table_obs_age_ms(self, obs: Optional[TableEdgeObs]) -> Optional[float]:
        if obs is None:
            return None
        age_ms: Optional[float] = None
        obs_ts = getattr(obs, "frame_capture_ts", None)
        if obs_ts is None:
            obs_ts = obs.obs_ts if obs.obs_ts is not None else obs.ts
        try:
            age_ms = max(0.0, (time.time() - float(obs_ts)) * 1000.0)
        except Exception:
            age_ms = None
        for candidate in (getattr(obs, "obs_total_age_ms", None), obs.age_ms):
            if candidate is None:
                continue
            try:
                age_ms = max(float(age_ms or 0.0), float(candidate))
            except Exception:
                pass
        return age_ms

    def _table_control_loop_age_ms(self, obs: Optional[TableEdgeObs]) -> Optional[float]:
        if obs is None or getattr(obs, "obs_recv_ts", None) is None:
            return None
        try:
            return max(0.0, (time.time() - float(obs.obs_recv_ts)) * 1000.0)
        except Exception:
            return None

    def _table_obs_stale_level(self, obs: Optional[TableEdgeObs]) -> str:
        if obs is None:
            return "dead"
        age_ms = self._table_obs_age_ms(obs)
        if age_ms is None or bool(getattr(obs, "is_stale", False)) or obs.depth_valid is False:
            return "hard_stale"
        soft = float(getattr(self.cfg, "table_obs_stale_soft_ms", 300) or 300)
        stop = float(getattr(self.cfg, "table_obs_stale_stop_ms", 500) or 500)
        hard = float(getattr(self.cfg, "table_obs_stale_hard_ms", 800) or 800)
        if float(age_ms) <= soft:
            return "fresh"
        if float(age_ms) <= stop:
            return "soft_stale"
        if float(age_ms) <= hard:
            return "hard_stale"
        return "dead"

    def _table_obs_stale_reason(self, obs: Optional[TableEdgeObs]) -> str:
        level = self._table_obs_stale_level(obs)
        if level == "fresh":
            return ""
        age_ms = self._table_obs_age_ms(obs)
        age_text = "unknown" if age_ms is None else f"{float(age_ms):.0f}"
        return f"{level}:obs_total_age_ms={age_text}"

    def _vision_stale_reason(self, obs: Optional[TableEdgeObs]) -> str:
        if obs is None:
            return "no_recent_obs"
        if bool(getattr(obs, "is_stale", False)) or obs.depth_valid is False:
            return "obs_invalid"
        if bool(getattr(obs, "fast_temporal_jump", False)):
            return "temporal_jump"
        reject_reason = str(getattr(obs, "control_reject_reason", "") or getattr(obs, "reject_reason", "") or getattr(obs, "reason", "") or "").strip().lower()
        if reject_reason in {"temporal_jump", "yaw_jump", "dist_jump"}:
            return reject_reason
        if "yaw_jump" in reject_reason:
            return "yaw_jump"
        if "dist_jump" in reject_reason or "far_jump" in reject_reason:
            return "dist_jump"
        age_ms = self._table_obs_age_ms(obs)
        if age_ms is None:
            return "age_over_limit"
        stale_level = self._table_obs_stale_level(obs)
        if stale_level == "soft_stale":
            return "soft_stale"
        if stale_level in {"hard_stale", "dead"}:
            return "hard_stale"
        if reject_reason and reject_reason.lower() not in {"none", "ok", "valid"}:
            return "reject_reason"
        control_level = self._control_level(obs)
        if control_level in {"none", ""} or not bool(getattr(obs, "usable_for_approach", False) or getattr(obs, "usable_for_alignment", False) or getattr(obs, "usable_for_stop", False)):
            return "control_level_not_usable"
        return "unknown"

    def _edge_obs_is_stale(self, obs: Optional[TableEdgeObs]) -> bool:
        if obs is None:
            return True
        if bool(getattr(obs, "is_stale", False)):
            return True
        if obs.depth_valid is False:
            return True
        age_ms = self._table_obs_age_ms(obs)
        if age_ms is None:
            return True
        return self._table_obs_stale_level(obs) != "fresh"

    def _fresh_target_obs(self) -> Optional[TargetObs]:
        obs = self.ctx.last_target_obs
        if obs is None or time.time() - obs.ts > self.cfg.target_obs_max_age_s:
            return None
        if self.ctx.task_start_wall_ts > 0 and obs.ts < self.ctx.task_start_wall_ts:
            return None
        if obs.session_id and self.ctx.active_session_id and obs.session_id != self.ctx.active_session_id:
            return None
        return obs

    def _fresh_home_obs(self) -> Optional[HomeTagObs]:
        obs = self.ctx.last_home_obs
        if obs is None or time.time() - obs.ts > self.cfg.home_obs_max_age_s:
            return None
        if self.ctx.task_start_wall_ts > 0 and obs.ts < self.ctx.task_start_wall_ts:
            return None
        if obs.session_id and self.ctx.active_session_id and obs.session_id != self.ctx.active_session_id:
            return None
        return obs

    def _state_elapsed(self) -> float:
        return monotonic_ts() - self.ctx.state_enter_mono

    def _start_loss_timer(self, attr_name: str):
        if getattr(self.ctx, attr_name, 0.0) <= 0.0:
            setattr(self.ctx, attr_name, monotonic_ts())

    def _loss_elapsed(self, started_mono: float) -> float:
        if started_mono <= 0.0:
            return 0.0
        return max(0.0, monotonic_ts() - started_mono)

    def _reset_table_loss(self):
        self.ctx.table_lost_frames = 0
        self.ctx.table_loss_since_mono = 0.0

    def _table_visible(self, obs: Optional[TableEdgeObs]) -> bool:
        return bool(obs is not None and (obs.table_found or self._table_plane_stable(obs) or self._table_yolo_reliable(obs)))

    @staticmethod
    def _median(values: List[float]) -> Optional[float]:
        vals = sorted(float(v) for v in values if v is not None)
        if not vals:
            return None
        mid = len(vals) // 2
        if len(vals) % 2:
            return vals[mid]
        return (vals[mid - 1] + vals[mid]) * 0.5

    def _reset_slide_ref_handoff(self) -> None:
        self.ctx.slide_ref_ready = False
        self.ctx.slide_ref_yaw_err = None
        self.ctx.slide_ref_dist_err = None
        self.ctx.slide_ref_edge_conf = None
        self.ctx.slide_ref_roi = None
        self.ctx.slide_ref_seq = None
        self.ctx.slide_ref_samples.clear()
        self.ctx.slide_ref_last_sample_key = ""
        self.ctx.handoff_state = "collecting"
        self.ctx.last_edge_quality.clear()

    def _slide_ref_sample_key(self, obs: TableEdgeObs) -> str:
        return f"{obs.source_mode or ''}:{obs.frame_id if obs.frame_id is not None else ''}:{obs.seq if obs.seq is not None else ''}:{obs.obs_ts or obs.ts:.6f}"

    def _slide_ref_obs_usable(self, obs: Optional[TableEdgeObs]) -> bool:
        if obs is None or not self._table_visible(obs):
            return False
        if str(obs.source_mode or "").strip().upper() != "FIND_OBJECT":
            return False
        if self._edge_obs_is_stale(obs):
            return False
        if not self._edge_valid_for_follow(obs):
            return False
        if obs.yaw_err_rad is None or obs.dist_err_m is None:
            return False
        conf = float(obs.confidence if obs.confidence is not None else (obs.edge_conf or 0.0))
        min_conf = float(getattr(self.cfg, "edge_follow_min_edge_conf_track_local", 0.20) or 0.20)
        return conf >= min_conf

    def _maybe_add_slide_ref_sample(self, obs: Optional[TableEdgeObs]) -> None:
        if not self._slide_ref_obs_usable(obs):
            return
        assert obs is not None
        key = self._slide_ref_sample_key(obs)
        if key == self.ctx.slide_ref_last_sample_key:
            return
        self.ctx.slide_ref_last_sample_key = key
        roi = obs.depth_edge_roi or obs.table_edge_roi or obs.edge_roi
        self.ctx.slide_ref_samples.append(
            {
                "yaw_err": float(obs.yaw_err_rad),
                "dist_err": float(obs.dist_err_m),
                "edge_conf": float(obs.confidence if obs.confidence is not None else (obs.edge_conf or 0.0)),
                "roi": list(roi) if isinstance(roi, list) else None,
                "seq": int(obs.seq) if obs.seq is not None else None,
                "frame_id": int(obs.frame_id) if obs.frame_id is not None else None,
            }
        )
        max_samples = max(5, int(getattr(self.cfg, "edge_handoff_samples", 3) or 3))
        if len(self.ctx.slide_ref_samples) > max_samples:
            self.ctx.slide_ref_samples[:] = self.ctx.slide_ref_samples[-max_samples:]
        self.ctx.handoff_state = "collecting"

    def _finalize_slide_ref(self) -> None:
        samples = list(self.ctx.slide_ref_samples)
        yaw = self._median([float(s["yaw_err"]) for s in samples if s.get("yaw_err") is not None])
        dist = self._median([float(s["dist_err"]) for s in samples if s.get("dist_err") is not None])
        conf = self._median([float(s["edge_conf"]) for s in samples if s.get("edge_conf") is not None])
        if yaw is None or dist is None or conf is None:
            return
        last = samples[-1] if samples else {}
        self.ctx.slide_ref_ready = True
        self.ctx.slide_ref_yaw_err = float(yaw)
        self.ctx.slide_ref_dist_err = float(dist)
        self.ctx.slide_ref_edge_conf = float(conf)
        self.ctx.slide_ref_roi = list(last.get("roi")) if isinstance(last.get("roi"), list) else None
        self.ctx.slide_ref_seq = int(last.get("seq")) if last.get("seq") is not None else None
        self.ctx.handoff_state = "ready"

    def _full_vs_light_yaw_offset(self) -> Optional[float]:
        if self.ctx.slide_ref_yaw_err is None or self.ctx.locked_yaw_err is None:
            return None
        return float(self.ctx.slide_ref_yaw_err) - float(self.ctx.locked_yaw_err)

    def _full_vs_light_dist_offset(self) -> Optional[float]:
        if self.ctx.slide_ref_dist_err is None or self.ctx.locked_dist_err is None:
            return None
        return float(self.ctx.slide_ref_dist_err) - float(self.ctx.locked_dist_err)

    def _handoff_trace_fields(self) -> Dict[str, Any]:
        return {
            "handoff_state": self.ctx.handoff_state,
            "handoff_samples_count": len(self.ctx.slide_ref_samples),
            "handoff_valid_samples_count": len(self.ctx.slide_ref_samples),
            "slide_ref_ready": bool(self.ctx.slide_ref_ready),
            "slide_ref_yaw_err": self.ctx.slide_ref_yaw_err,
            "slide_ref_dist_err": self.ctx.slide_ref_dist_err,
            "slide_ref_edge_conf": self.ctx.slide_ref_edge_conf,
            "slide_ref_roi": self.ctx.slide_ref_roi,
            "slide_ref_seq": self.ctx.slide_ref_seq,
            "full_locked_yaw_err": self.ctx.locked_yaw_err,
            "full_locked_dist_err": self.ctx.locked_dist_err,
            "full_vs_light_yaw_offset": self._full_vs_light_yaw_offset(),
            "full_vs_light_dist_offset": self._full_vs_light_dist_offset(),
        }

    def _capture_locked_edge(self, obs: Optional[TableEdgeObs]) -> None:
        if obs is None:
            return
        self.ctx.locked_edge_id = str(self.ctx.current_edge_id or "")
        line = {}
        if obs.edge_k is not None:
            line["edge_k"] = float(obs.edge_k)
        if obs.edge_b is not None:
            line["edge_b"] = float(obs.edge_b)
        self.ctx.locked_edge_line = line or None
        roi = obs.depth_edge_roi or obs.table_edge_roi or obs.edge_roi
        self.ctx.locked_roi = list(roi) if isinstance(roi, list) else None
        self.ctx.locked_yaw_err = float(obs.yaw_err_rad) if obs.yaw_err_rad is not None else None
        self.ctx.locked_dist_err = float(obs.dist_err_m) if obs.dist_err_m is not None else None
        self.ctx.locked_edge_conf = float(obs.confidence or 0.0)
        self.ctx.locked_obs_seq = int(obs.seq) if obs.seq is not None else None

    def _edge_valid_for_follow(self, obs: Optional[TableEdgeObs]) -> bool:
        if obs is None:
            return False
        level = self._control_level(obs)
        if level in {"alignment", "stop"}:
            return True
        if level == "approach":
            return False
        edge_valid = getattr(obs, "edge_valid", None)
        if edge_valid is not None:
            return bool(edge_valid)
        return bool(obs.edge_found)

    def _edge_follow_quality(self, obs: TableEdgeObs) -> Dict[str, Any]:
        source_mode = str(obs.source_mode or self.ctx.active_vision_mode or "").strip().upper()
        is_track_local = source_mode == "FIND_OBJECT" or str(self.ctx.active_vision_mode or "").strip().upper() == "FIND_OBJECT"
        min_conf = float(
            getattr(
                self.cfg,
                "edge_follow_min_edge_conf_track_local" if is_track_local else "edge_follow_min_edge_conf_table_edge_perception",
                getattr(self.cfg, "edge_follow_min_edge_conf", 0.60),
            )
            or 0.0
        )
        if is_track_local:
            weak_conf = float(getattr(self.cfg, "edge_follow_weak_edge_conf_track_local", min_conf) or min_conf)
            strong_conf = float(getattr(self.cfg, "edge_follow_strong_edge_conf_track_local", min_conf) or min_conf)
        else:
            weak_conf = min_conf
            strong_conf = min_conf
        conf = float(obs.confidence or 0.0)
        yaw = float(obs.yaw_err_rad) if obs.yaw_err_rad is not None else None
        dist = float(obs.dist_err_m) if obs.dist_err_m is not None else None
        locked_yaw = self.ctx.locked_yaw_err
        locked_dist = self.ctx.locked_dist_err
        slide_ref_yaw = self.ctx.slide_ref_yaw_err
        slide_ref_dist = self.ctx.slide_ref_dist_err
        yaw_delta = None if yaw is None or locked_yaw is None else float(yaw - float(locked_yaw))
        dist_delta = None if dist is None or locked_dist is None else float(dist - float(locked_dist))
        yaw_delta_from_slide_ref = None if yaw is None or slide_ref_yaw is None else float(yaw - float(slide_ref_yaw))
        dist_delta_from_slide_ref = None if dist is None or slide_ref_dist is None else float(dist - float(slide_ref_dist))
        identity_basis = "slide_ref" if is_track_local and self.ctx.slide_ref_ready else "full_locked_edge"
        basis_yaw_delta = yaw_delta_from_slide_ref if identity_basis == "slide_ref" else yaw_delta
        basis_dist_delta = dist_delta_from_slide_ref if identity_basis == "slide_ref" else dist_delta
        yaw_mismatch = basis_yaw_delta is not None and abs(basis_yaw_delta) > float(getattr(self.cfg, "edge_identity_yaw_mismatch_rad", 0.15))
        dist_mismatch = basis_dist_delta is not None and abs(basis_dist_delta) > float(getattr(self.cfg, "edge_identity_dist_mismatch_m", 0.04))
        identity_ok = not (yaw_mismatch or dist_mismatch)
        if not identity_ok:
            mode = "identity_mismatch"
            reason = "edge_identity_mismatch"
        elif conf >= strong_conf:
            mode = "strong"
            reason = "edge_slide"
        elif conf >= weak_conf:
            mode = "weak"
            reason = "weak_edge_slide"
        else:
            mode = "pause"
            reason = "edge_conf_low"
        return {
            "mode": mode,
            "reason": reason,
            "edge_conf_threshold_used": min_conf,
            "weak_conf": weak_conf,
            "strong_conf": strong_conf,
            "locked_edge_conf": self.ctx.locked_edge_conf,
            "locked_yaw_err": locked_yaw,
            "locked_dist_err": locked_dist,
            "yaw_delta_from_locked": yaw_delta,
            "dist_delta_from_locked": dist_delta,
            "slide_ref_ready": bool(self.ctx.slide_ref_ready),
            "slide_ref_yaw_err": slide_ref_yaw,
            "slide_ref_dist_err": slide_ref_dist,
            "slide_ref_edge_conf": self.ctx.slide_ref_edge_conf,
            "yaw_delta_from_slide_ref": yaw_delta_from_slide_ref,
            "dist_delta_from_slide_ref": dist_delta_from_slide_ref,
            "edge_identity_basis": identity_basis,
            "full_locked_yaw_err": locked_yaw,
            "full_locked_dist_err": locked_dist,
            "full_vs_light_yaw_offset": self._full_vs_light_yaw_offset(),
            "full_vs_light_dist_offset": self._full_vs_light_dist_offset(),
            "handoff_state": self.ctx.handoff_state,
            "handoff_samples_count": len(self.ctx.slide_ref_samples),
            "handoff_valid_samples_count": len(self.ctx.slide_ref_samples),
            "edge_identity_ok": identity_ok,
            "slide_vy_norm": float(getattr(self.controller.car_cfg, "edge_slide_vy_norm", 0.14) or 0.14),
            "weak_slide_vy": float(getattr(self.controller.car_cfg, "edge_slide_weak_vy_norm", 0.05) or 0.05),
            "weak_slide_vy_norm": float(getattr(self.controller.car_cfg, "edge_slide_weak_vy_norm", 0.05) or 0.05),
            "fallback_candidate_state": self._edge_slide_fallback_state().value,
            "fallback_suppressed_reason": "fresh_geometry_stable" if mode in {"weak", "strong"} else "",
        }

    @staticmethod
    def _raw_control_level(obs: Optional[TableEdgeObs]) -> str:
        if obs is None:
            return "none"
        return str(getattr(obs, "control_level", "none") or "none").strip().lower()

    @classmethod
    def _control_level(cls, obs: Optional[TableEdgeObs]) -> str:
        level = cls._raw_control_level(obs)
        aliases = {
            "approach_slow": "approach",
            "approach": "approach",
            "rotate_only": "alignment",
            "align": "alignment",
            "alignment": "alignment",
            "stop_ready": "stop",
            "stop": "stop",
            "none": "none",
            "": "none",
        }
        return aliases.get(level, "none")

    def _table_yolo_reliable(self, obs: Optional[TableEdgeObs]) -> bool:
        if obs is None:
            return False
        if hasattr(obs, "yolo_reliable"):
            return bool(getattr(obs, "yolo_reliable", False))
        return bool(getattr(obs, "table_confirmed_by_yolo", False)) and getattr(obs, "table_cx_norm", None) is not None

    def _table_plane_stable(self, obs: Optional[TableEdgeObs]) -> bool:
        if obs is None or obs.yaw_err_rad is None or obs.dist_err_m is None:
            return False
        if bool(getattr(obs, "usable_for_approach", False)):
            return True
        if bool(getattr(obs, "usable_for_alignment", False)) or bool(getattr(obs, "usable_for_stop", False)):
            return True
        level = self._control_level(obs)
        return level in {"approach", "alignment", "stop"} or bool(getattr(obs, "edge_found", False))

    def _table_approach_phase(self, obs: Optional[TableEdgeObs]) -> str:
        level = self._control_level(obs)
        if level == "stop":
            return "PLANE_STOP"
        if level == "alignment":
            return "PLANE_FINAL_LOCK"
        if level == "approach":
            return "PLANE_APPROACH"
        if self._table_plane_stable(obs):
            return "PLANE_FINAL_LOCK"
        return "PLANE_ACQUIRE"

    def _align_to_approach_yaw_rad(self) -> float:
        return max(0.0, float(getattr(self.cfg, "align_to_approach_yaw_rad", 0.08) or 0.08))

    def _approach_to_align_yaw_rad(self) -> float:
        align_yaw = self._align_to_approach_yaw_rad()
        return max(align_yaw + 1e-6, float(getattr(self.cfg, "approach_to_align_yaw_rad", 0.16) or 0.16))

    def _align_to_approach_stable_obs(self) -> int:
        return max(1, int(getattr(self.cfg, "align_to_approach_stable_obs", 2) or 2))

    def _approach_to_align_stable_obs(self) -> int:
        return max(1, int(getattr(self.cfg, "approach_to_align_stable_obs", 2) or 2))

    def _coarse_align_min_dwell_s(self) -> float:
        return max(0.0, float(getattr(self.cfg, "coarse_align_min_dwell_s", 0.8) or 0.8))

    def _controlled_approach_min_dwell_s(self) -> float:
        return max(0.0, float(getattr(self.cfg, "controlled_approach_min_dwell_s", 0.8) or 0.8))

    def _yaw_abs(self, obs: Optional[TableEdgeObs]) -> Optional[float]:
        if obs is None or obs.yaw_err_rad is None:
            return None
        try:
            return abs(float(obs.yaw_err_rad))
        except Exception:
            return None

    def _yaw_ready_for_controlled_approach(self, obs: Optional[TableEdgeObs]) -> bool:
        yaw_abs = self._yaw_abs(obs)
        return bool(yaw_abs is not None and yaw_abs <= self._align_to_approach_yaw_rad())

    def _yaw_needs_realign_from_approach(self, obs: Optional[TableEdgeObs]) -> bool:
        yaw_abs = self._yaw_abs(obs)
        return bool(yaw_abs is not None and yaw_abs >= self._approach_to_align_yaw_rad())

    def _count_table_motion_hysteresis_obs(self, obs: Optional[TableEdgeObs], *, ok: bool, last_key_attr: str, count_attr: str) -> int:
        obs_key = self._final_lock_obs_key(obs)
        if not ok:
            setattr(self.ctx, count_attr, 0)
            if obs_key:
                setattr(self.ctx, last_key_attr, obs_key)
            return 0
        if obs_key and obs_key != str(getattr(self.ctx, last_key_attr, "") or ""):
            setattr(self.ctx, last_key_attr, obs_key)
            setattr(self.ctx, count_attr, int(getattr(self.ctx, count_attr, 0) or 0) + 1)
        return int(getattr(self.ctx, count_attr, 0) or 0)

    def _annotate_table_motion_hysteresis(self, decision: MotionDecision, *, pending_reason: str = "") -> MotionDecision:
        if decision.control_summary is None:
            decision.control_summary = {}
        summary = decision.control_summary
        summary.update(
            {
                "align_to_approach_yaw_rad": float(self._align_to_approach_yaw_rad()),
                "approach_to_align_yaw_rad": float(self._approach_to_align_yaw_rad()),
                "align_to_approach_stable_obs": int(self._align_to_approach_stable_obs()),
                "approach_to_align_stable_obs": int(self._approach_to_align_stable_obs()),
                "coarse_align_min_dwell_s": float(self._coarse_align_min_dwell_s()),
                "controlled_approach_min_dwell_s": float(self._controlled_approach_min_dwell_s()),
                "state_dwell_s": float(self._state_elapsed()),
                "approach_aligned_frames": int(self.ctx.approach_aligned_frames),
                "approach_realign_frames": int(self.ctx.approach_realign_frames),
                "table_motion_pending_transition_reason": str(pending_reason or self.ctx.table_motion_pending_transition_reason or ""),
            }
        )
        return decision

    def _coarse_aligned(self, obs: Optional[TableEdgeObs]) -> bool:
        if obs is None:
            return False
        if self._edge_obs_is_stale(obs):
            return False
        if self._control_level(obs) == "stop":
            return True
        if obs.yaw_err_rad is not None:
            return abs(float(obs.yaw_err_rad)) <= self._align_to_approach_yaw_rad()
        if obs.table_cx_norm is not None:
            return abs(float(obs.table_cx_norm)) <= 0.12
        return False

    def _edge_ready(self, obs: Optional[TableEdgeObs]) -> bool:
        if obs is None:
            return False
        if self._edge_obs_is_stale(obs):
            return False
        if self._control_level(obs) == "stop" or bool(getattr(obs, "usable_for_stop", False)):
            return True
        if obs.edge_ready is not None:
            return bool(obs.edge_ready)
        if obs.dist_err_m is not None:
            return abs(float(obs.dist_err_m)) <= float(self.cfg.final_lock_dist_tol_m) * 2.0
        return bool(obs.edge_found)

    def _final_lock_enter_status(self, obs: Optional[TableEdgeObs]) -> Dict[str, object]:
        level = self._control_level(obs)
        yaw_th = abs(float(getattr(self.cfg, "final_lock_enter_yaw_th_rad", 0.10) or 0.10))
        dist_th = abs(float(getattr(self.cfg, "final_lock_enter_dist_th_m", 0.08) or 0.08))
        min_conf = float(getattr(self.controller.docking.cfg, "min_confidence", 0.0) or 0.0)
        status: Dict[str, object] = {
            "final_lock_enabled": self._table_final_lock_enabled(),
            "micro_adjust_enabled": self._table_micro_adjust_enabled(),
            "normalized_control_level": level,
            "yaw_err": getattr(obs, "yaw_err_rad", None) if obs is not None else None,
            "dist_err_m": getattr(obs, "dist_err_m", None) if obs is not None else None,
            "final_lock_enter_allowed": False,
            "final_lock_enter_block_reason": "",
            "final_lock_enter_dist_th_m": float(dist_th),
            "final_lock_enter_yaw_th_rad": float(yaw_th),
        }
        if not self._table_final_lock_enabled():
            status["final_lock_enter_block_reason"] = "final_lock_disabled"
            return status
        if obs is None:
            status["final_lock_enter_block_reason"] = "no_recent_obs"
            return status
        stale_level = self._table_obs_stale_level(obs)
        if stale_level in {"hard_stale", "dead"}:
            status["final_lock_enter_block_reason"] = "vision_stale"
            return status
        if bool(getattr(obs, "is_stale", False)) or getattr(obs, "depth_valid", True) is False:
            status["final_lock_enter_block_reason"] = "obs_invalid"
            return status
        if not self._table_visible(obs) or not bool(getattr(obs, "edge_found", False)):
            status["final_lock_enter_block_reason"] = "edge_invalid"
            return status
        if not bool(getattr(obs, "valid_for_control", False) or getattr(obs, "usable_for_stop", False)):
            status["final_lock_enter_block_reason"] = "not_valid_for_control"
            return status
        if level != "stop" and not bool(getattr(obs, "usable_for_stop", False)):
            status["final_lock_enter_block_reason"] = "not_stop_ready"
            return status
        confidence = float(getattr(obs, "confidence", 0.0) or 0.0)
        if confidence < min_conf:
            status["final_lock_enter_block_reason"] = "low_confidence"
            return status
        if obs.yaw_err_rad is None:
            status["final_lock_enter_block_reason"] = "yaw_missing"
            return status
        if abs(float(obs.yaw_err_rad)) > yaw_th:
            status["final_lock_enter_block_reason"] = "yaw_out_of_range"
            return status
        if obs.dist_err_m is None:
            status["final_lock_enter_block_reason"] = "dist_missing"
            return status
        if abs(float(obs.dist_err_m)) > dist_th:
            status["final_lock_enter_block_reason"] = "distance_too_far" if float(obs.dist_err_m) > 0.0 else "distance_too_close"
            return status
        status["final_lock_enter_allowed"] = True
        status["final_lock_enter_block_reason"] = "allowed"
        return status

    def _table_target_dist_m(self, obs: Optional[TableEdgeObs] = None) -> float:
        target = getattr(self.cfg, "table_target_dist_m", 0.015)
        if obs is not None and obs.target_dist_m is not None:
            target = obs.target_dist_m
        try:
            return max(0.0, float(target))
        except Exception:
            return 0.015

    def _table_measured_dist_m(self, obs: Optional[TableEdgeObs]) -> Optional[float]:
        if obs is None or obs.dist_err_m is None:
            return None
        return self._table_target_dist_m(obs) + float(obs.dist_err_m)

    def _table_dock_should_stop(self, obs: Optional[TableEdgeObs]) -> bool:
        if obs is None or self._edge_obs_is_stale(obs):
            return False
        if self._control_level(obs) != "stop" and not bool(getattr(obs, "usable_for_stop", False)):
            return False
        measured = self._table_measured_dist_m(obs)
        if measured is None:
            return False
        target = self._table_target_dist_m(obs)
        margin = max(0.0, float(getattr(self.cfg, "table_stop_margin_m", 0.05)))
        return float(measured) <= target + margin

    def _table_final_lock_enabled(self) -> bool:
        return bool(getattr(self.cfg, "enable_final_lock", False))

    def _table_micro_adjust_enabled(self) -> bool:
        return bool(getattr(self.cfg, "enable_micro_adjust", False))

    def _table_approach_decision(
        self,
        obs: Optional[TableEdgeObs],
        *,
        phase: str = "PLANE_APPROACH",
        mode: str = "CONTROLLED_APPROACH",
        stop_ready_ignored: bool = False,
    ) -> MotionDecision:
        decision = self.controller.fov_table_approach_cmd(obs, phase=phase, mode=mode)
        if decision.control_summary is not None:
            enter_status = self._final_lock_enter_status(obs)
            decision.control_summary.update(
                {
                    **enter_status,
                    "stop_ready_ignored_for_stage_transition": bool(stop_ready_ignored),
                }
            )
        return decision

    def _enter_final_lock_or_keep_approach(self, obs: Optional[TableEdgeObs], reason: str) -> MotionDecision:
        enter_status = self._final_lock_enter_status(obs)
        if bool(enter_status.get("final_lock_enter_allowed")):
            self._transition(State.FINAL_LOCK, reason)
            decision = self.controller.fov_table_approach_cmd(obs, phase="PLANE_FINAL_LOCK", mode="FINAL_LOCK")
            if decision.control_summary is not None:
                decision.control_summary.update(enter_status)
            return decision
        block_reason = str(enter_status.get("final_lock_enter_block_reason") or "blocked")
        decision = self._table_approach_decision(obs, phase="PLANE_APPROACH", stop_ready_ignored=True)
        if self.ctx.state != State.CONTROLLED_APPROACH:
            self._transition(State.CONTROLLED_APPROACH, f"final_lock_enter_blocked:{block_reason}")
        return decision

    def _annotate_final_lock_decision(self, decision: MotionDecision, status: Dict[str, object]) -> MotionDecision:
        if decision.control_summary is not None:
            decision.control_summary.update(
                {
                    "final_lock_enabled": self._table_final_lock_enabled(),
                    "micro_adjust_enabled": self._table_micro_adjust_enabled(),
                    "normalized_control_level": status.get("normalized_control_level"),
                    "yaw_err": status.get("yaw_err"),
                    "dist_err_m": status.get("dist_err"),
                    "stable_lock_count": status.get("stable_lock_count"),
                    "required_lock_count": status.get("required_lock_count"),
                    "lock_ready_obs_count": status.get("lock_ready_obs_count"),
                    "window_ready_count": status.get("window_ready_count"),
                    "required_ready_obs": status.get("required_ready_obs"),
                    "final_lock_window_ms": status.get("final_lock_window_ms"),
                    "same_obs_reuse_count": status.get("same_obs_reuse_count"),
                    "consecutive_lost_count": status.get("consecutive_lost_count"),
                    "lock_count_inc_reason": status.get("lock_count_inc_reason"),
                    "lock_count_hold_reason": status.get("lock_count_hold_reason"),
                    "lock_count_reset_reason": status.get("lock_count_reset_reason"),
                    "final_lock_transition_block_reason": status.get("final_lock_transition_block_reason"),
                    "final_lock_transition_reason": status.get("final_lock_transition_reason"),
                }
            )
        return decision

    def _enter_table_dock_phase(self, phase: str, log_line: str = "") -> None:
        phase = str(phase or "").upper()
        if self.ctx.table_dock_phase != phase:
            self.ctx.table_dock_phase = phase
            self.ctx.table_dock_phase_since_mono = monotonic_ts()
        if log_line:
            self._log("info", log_line)

    def _table_dock_micro_adjust(self, obs: Optional[TableEdgeObs]) -> Optional[MotionDecision]:
        if not self._table_micro_adjust_enabled():
            self._log("info", "[TABLE_DOCK][MICRO_ADJUST] skipped enable_micro_adjust=false")
            decision = self._table_approach_decision(obs, phase="PLANE_APPROACH", stop_ready_ignored=True)
            if decision.control_summary is not None:
                decision.control_summary.update({"micro_adjust_skipped": True, "table_dock_phase": "MICRO_ADJUST_DISABLED"})
            return decision
        max_adjust = max(0, int(getattr(self.cfg, "table_max_micro_adjust", 4)))
        if self.ctx.table_micro_adjust_count >= max_adjust:
            reason = f"[TABLE_DOCK][FAIL] max_micro_adjust={max_adjust}"
            self.ctx.last_fail_reason = reason
            self._log("error", reason)
            self._enter_error_recovery("最终锁边微调次数超限", tts_text="桌边停靠失败，请检查", interrupt_tts=True)
            return self.controller.stop_cmd("ERROR_RECOVERY", brake=True)

        status = self._final_lock_status(obs, stable_count=self.ctx.table_lock_frames)
        action = ""
        tag = ""
        dist_delta = None
        if obs is not None and obs.dist_err_m is not None:
            measured = self._table_measured_dist_m(obs)
            target = self._table_target_dist_m(obs)
            dist_delta = (float(measured) - target) if measured is not None else float(obs.dist_err_m)
        dist_tol = float(getattr(self.cfg, "table_dist_tol_m", self.cfg.final_lock_dist_tol_m))
        yaw_tol = float(getattr(self.cfg, "table_yaw_tol_rad", self.cfg.final_lock_yaw_tol_rad))

        if dist_delta is not None and dist_delta > dist_tol:
            action = "forward"
            tag = "[TABLE_DOCK][JOG_FORWARD]"
        elif dist_delta is not None and dist_delta < -dist_tol:
            action = "backward"
            tag = "[TABLE_DOCK][JOG_BACKWARD]"
        elif obs is not None and obs.yaw_err_rad is not None and abs(float(obs.yaw_err_rad)) > yaw_tol:
            action = "turn_left" if float(obs.yaw_err_rad) > 0.0 else "turn_right"
            tag = "[TABLE_DOCK][JOG_TURN]"
        else:
            action = "forward" if str(status["reason"]) == "distance_too_far" else ""
            tag = "[TABLE_DOCK][JOG_FORWARD]" if action else ""

        if not action:
            self._log("info", f"[TABLE_DOCK][SETTLE] no jog action reason={status['reason']}")
            self._enter_table_dock_phase("STOP_AND_SETTLE")
            return self.controller.stop_cmd("FINAL_LOCK")

        self.ctx.table_micro_adjust_count += 1
        reason = (
            f"{tag} action={action} count={self.ctx.table_micro_adjust_count}/{max_adjust} "
            f"reason={status['reason']} dist_delta={dist_delta} yaw_err={obs.yaw_err_rad if obs is not None else None}"
        )
        self._log("info", reason)
        self.ctx.table_lock_frames = 0
        self._enter_table_dock_phase("STOP_AND_SETTLE")
        self._log("info", "[TABLE_DOCK][SETTLE] begin after JOG")
        decision = self.controller.stop_cmd("FINAL_LOCK")
        decision.jog_action = action
        decision.jog_reason = reason
        if decision.control_summary is not None:
            decision.control_summary.update({"table_dock_phase": "MICRO_ADJUST", "jog_action": action, "reason": reason})
        return decision

    def _final_lock_ready(self, obs: Optional[TableEdgeObs]) -> bool:
        return bool(self._final_lock_status(obs, stable_count=self.ctx.table_lock_frames)["lock_ready"])

    def _required_lock_count(self) -> int:
        return max(1, int(getattr(self.cfg, "table_stable_frames", self.cfg.final_lock_frames_to_arrive)))

    def _final_lock_required_ready_obs(self) -> int:
        return max(1, int(getattr(self.cfg, "final_lock_required_ready_obs", 3) or 3))

    def _final_lock_window_ms(self) -> int:
        return max(100, int(getattr(self.cfg, "final_lock_window_ms", 1000) or 1000))

    def _final_lock_max_consecutive_lost(self) -> int:
        return max(0, int(getattr(self.cfg, "final_lock_max_consecutive_lost", 2) or 2))

    def _final_lock_obs_key(self, obs: Optional[TableEdgeObs]) -> str:
        if obs is None:
            return ""
        obs_seq = getattr(obs, "obs_seq", None)
        if obs_seq is not None:
            return f"obs_seq:{obs_seq}"
        return self._table_obs_key(obs)

    def _prune_final_lock_window(self, now_mono: Optional[float] = None) -> None:
        now = monotonic_ts() if now_mono is None else float(now_mono)
        window_s = float(self._final_lock_window_ms()) / 1000.0
        self.ctx.final_lock_ready_window = [
            item
            for item in list(self.ctx.final_lock_ready_window or [])
            if now - float(item.get("mono_ts", now) or now) <= window_s
        ]
        self.ctx.table_lock_frames = len(self.ctx.final_lock_ready_window)

    def _reset_final_lock_window(self, reason: str = "") -> None:
        self.ctx.final_lock_ready_window.clear()
        self.ctx.table_lock_frames = 0
        self.ctx.final_lock_consecutive_lost_count = 0
        if reason:
            self.ctx.final_lock_last_transition_reason = str(reason)

    def _final_lock_ready_jump_reason(self, obs: Optional[TableEdgeObs]) -> str:
        if obs is None or not self.ctx.final_lock_ready_window:
            return ""
        last = self.ctx.final_lock_ready_window[-1]
        yaw = getattr(obs, "yaw_err_rad", None)
        dist = getattr(obs, "dist_err_m", None)
        last_yaw = last.get("yaw_err")
        last_dist = last.get("dist_err")
        try:
            yaw_jump_th = max(0.35, float(getattr(self.cfg, "edge_identity_yaw_mismatch_rad", 0.15) or 0.15) * 2.0)
            if yaw is not None and last_yaw is not None and abs(float(yaw) - float(last_yaw)) > yaw_jump_th:
                return "yaw_jump"
        except Exception:
            pass
        try:
            dist_jump_th = max(0.20, float(getattr(self.cfg, "edge_identity_dist_mismatch_m", 0.04) or 0.04) * 3.0)
            if dist is not None and last_dist is not None and abs(float(dist) - float(last_dist)) > dist_jump_th:
                return "dist_jump"
        except Exception:
            pass
        return ""

    def _update_final_lock_count(self, obs: Optional[TableEdgeObs]) -> Dict[str, object]:
        status = self._final_lock_status(obs, stable_count=self.ctx.table_lock_frames)
        now = monotonic_ts()
        self._prune_final_lock_window(now)
        obs_key = self._final_lock_obs_key(obs)
        new_obs = bool(obs_key and obs_key != self.ctx.final_lock_last_obs_key)
        if obs_key:
            if new_obs:
                self.ctx.final_lock_same_obs_reuse_count = 0
                self.ctx.final_lock_last_obs_key = obs_key
            else:
                self.ctx.final_lock_same_obs_reuse_count += 1

        reason = str(status.get("reason") or "")
        stale_reason = str(status.get("vision_stale_reason") or "")
        reset_reason = str(status.get("lock_count_reset_reason") or stale_reason or reason or "lock_ready_false")
        lost_like = reason in {"table_lost", "no_edge", "edge_invalid"} or reset_reason in {"table_lost", "no_edge", "edge_invalid"}
        hard_reset = reset_reason in {"hard_stale", "obs_invalid", "temporal_jump", "no_recent_obs", "yaw_jump", "dist_jump", "age_over_limit"}
        hold_reason = str(status.get("lock_count_hold_reason") or "")
        if hold_reason == "soft_stale" and not bool(getattr(self.cfg, "final_lock_soft_stale_hold", True)):
            hold_reason = ""

        status["obs_seq"] = getattr(obs, "obs_seq", None) if obs is not None else None
        status["same_obs_reuse_count"] = int(self.ctx.final_lock_same_obs_reuse_count)
        status["required_ready_obs"] = int(self._final_lock_required_ready_obs())
        status["final_lock_window_ms"] = int(self._final_lock_window_ms())
        status["required_lock_count"] = int(self._final_lock_required_ready_obs())
        status["legacy_required_lock_count"] = int(self._required_lock_count())

        if bool(status["lock_ready"]):
            jump_reason = self._final_lock_ready_jump_reason(obs)
            if jump_reason:
                self._reset_final_lock_window(jump_reason)
                status["lock_ready"] = False
                status["lock_count_inc_reason"] = ""
                status["lock_count_hold_reason"] = ""
                status["lock_count_reset_reason"] = jump_reason
                status["lock_reset_reason"] = jump_reason
            elif new_obs:
                self.ctx.final_lock_consecutive_lost_count = 0
                self.ctx.final_lock_ready_window.append(
                    {
                        "key": obs_key,
                        "mono_ts": now,
                        "obs_seq": getattr(obs, "obs_seq", None) if obs is not None else None,
                        "yaw_err": getattr(obs, "yaw_err_rad", None) if obs is not None else None,
                        "dist_err": getattr(obs, "dist_err_m", None) if obs is not None else None,
                        "obs_age_ms": status.get("obs_age_ms"),
                    }
                )
                self._prune_final_lock_window(now)
                status["lock_count_inc_reason"] = "fresh_lock_ready"
                status["lock_count_hold_reason"] = ""
                status["lock_count_reset_reason"] = ""
                status["lock_reset_reason"] = ""
            else:
                status["lock_count_inc_reason"] = ""
                status["lock_count_hold_reason"] = "same_obs_reuse"
                status["lock_count_reset_reason"] = ""
                status["lock_reset_reason"] = ""
        elif hold_reason:
            status["lock_count_inc_reason"] = ""
            status["lock_count_hold_reason"] = hold_reason
            status["lock_count_reset_reason"] = ""
            status["lock_reset_reason"] = ""
        elif lost_like and not hard_reset:
            if new_obs:
                self.ctx.final_lock_consecutive_lost_count += 1
            max_lost = self._final_lock_max_consecutive_lost()
            if self.ctx.final_lock_consecutive_lost_count <= max_lost:
                status["lock_count_inc_reason"] = ""
                status["lock_count_hold_reason"] = f"{reason or reset_reason}_lost_hold"
                status["lock_count_reset_reason"] = ""
                status["lock_reset_reason"] = ""
            else:
                reset_reason = f"{reason or reset_reason}_lost_exceeded"
                self._reset_final_lock_window(reset_reason)
                status["lock_count_inc_reason"] = ""
                status["lock_count_hold_reason"] = ""
                status["lock_count_reset_reason"] = reset_reason
                status["lock_reset_reason"] = reset_reason
        else:
            self._reset_final_lock_window(reset_reason)
            status["lock_count_inc_reason"] = ""
            status["lock_count_hold_reason"] = ""
            status["lock_count_reset_reason"] = reset_reason
            status["lock_reset_reason"] = reset_reason

        ready_count = len(self.ctx.final_lock_ready_window)
        window_ready = ready_count >= self._final_lock_required_ready_obs()
        status["lock_ready_obs_count"] = int(ready_count)
        status["window_ready_count"] = int(ready_count)
        status["stable_count"] = int(ready_count)
        status["stable_lock_count"] = int(ready_count)
        status["consecutive_lost_count"] = int(self.ctx.final_lock_consecutive_lost_count)
        status["final_lock_window_ready"] = bool(window_ready)
        status["final_lock_transition_reason"] = "final_lock_window_ready" if window_ready else ""
        status["final_lock_transition_block_reason"] = "" if window_ready else str(
            status.get("lock_count_reset_reason")
            or status.get("lock_count_hold_reason")
            or reason
            or "ready_obs_count_not_enough"
        )
        return status

    def _final_lock_status(self, obs: Optional[TableEdgeObs], stable_count: int = 0) -> Dict[str, object]:
        required_count = self._final_lock_required_ready_obs()

        def _status(
            *,
            lock_ready: bool,
            reason: str,
            yaw_locked: bool = False,
            dist_locked: bool = False,
            lat_locked: bool = False,
            confidence_ok: bool = False,
            lock_count_inc_reason: str = "",
            lock_count_hold_reason: str = "",
            lock_count_reset_reason: str = "",
        ) -> Dict[str, object]:
            obs_age_ms = self._table_obs_age_ms(obs)
            stale_reason = self._vision_stale_reason(obs)
            valid_for_control = bool(getattr(obs, "valid_for_control", False)) if obs is not None else False
            usable_for_approach = bool(getattr(obs, "usable_for_approach", False)) if obs is not None else False
            usable_for_stop = bool(getattr(obs, "usable_for_stop", False)) if obs is not None else False
            raw_control_level = self._raw_control_level(obs)
            normalized_control_level = self._control_level(obs)
            out_reason = str(reason or "")
            hold_text = str(lock_count_hold_reason or "")
            reset_text = str(lock_count_reset_reason or ("" if (bool(lock_ready) or hold_text) else out_reason))
            return {
                "lock_ready": bool(lock_ready),
                "reason": out_reason,
                "yaw_locked": bool(yaw_locked),
                "dist_locked": bool(dist_locked),
                "lat_locked": bool(lat_locked),
                "stable_count": int(stable_count),
                "yaw_err": getattr(obs, "yaw_err_rad", None) if obs is not None else None,
                "dist_err": getattr(obs, "dist_err_m", None) if obs is not None else None,
                "obs_age_ms": obs_age_ms,
                "valid_for_control": bool(valid_for_control),
                "usable_for_approach": bool(usable_for_approach),
                "usable_for_stop": bool(usable_for_stop),
                "raw_control_level": raw_control_level,
                "normalized_control_level": normalized_control_level,
                "control_level": normalized_control_level,
                "confidence": float(getattr(obs, "confidence", 0.0) or 0.0) if obs is not None else 0.0,
                "yaw_ok": bool(yaw_locked),
                "dist_ok": bool(dist_locked),
                "age_ok": bool(obs is not None and self._table_obs_stale_level(obs) == "fresh"),
                "confidence_ok": bool(confidence_ok),
                "stable_lock_count": int(stable_count),
                "required_lock_count": int(required_count),
                "required_ready_obs": int(required_count),
                "final_lock_window_ms": int(self._final_lock_window_ms()),
                "lock_ready_obs_count": int(stable_count),
                "window_ready_count": int(stable_count),
                "same_obs_reuse_count": int(self.ctx.final_lock_same_obs_reuse_count),
                "consecutive_lost_count": int(self.ctx.final_lock_consecutive_lost_count),
                "final_lock_transition_block_reason": "",
                "final_lock_transition_reason": "",
                "lock_count_inc_reason": str(lock_count_inc_reason or ("lock_ready" if bool(lock_ready) else "")),
                "lock_count_hold_reason": hold_text,
                "lock_count_reset_reason": reset_text,
                "lock_reset_reason": reset_text,
                "vision_stale_reason": stale_reason,
            }

        if obs is None:
            reason = "vision_stale" if self.ctx.last_table_obs is not None else "table_lost"
            return _status(lock_ready=False, reason=reason, lock_count_reset_reason="no_recent_obs")
        if not bool(getattr(obs, "table_found", False)):
            return _status(lock_ready=False, reason="table_lost", lock_count_reset_reason="table_lost")
        if not bool(obs.edge_found):
            return _status(lock_ready=False, reason="no_edge", lock_count_reset_reason="no_edge")
        if not self._edge_valid_for_follow(obs):
            return _status(lock_ready=False, reason="edge_invalid", lock_count_reset_reason="edge_invalid")
        stale_reason = self._vision_stale_reason(obs)
        if stale_reason in {"hard_stale", "obs_invalid", "temporal_jump", "no_recent_obs", "yaw_jump", "dist_jump"}:
            return _status(lock_ready=False, reason="vision_stale", lock_count_reset_reason=stale_reason)
        min_confidence = float(getattr(self.controller.docking.cfg, "min_confidence", 0.0))
        confidence_ok = float(obs.confidence or 0.0) >= min_confidence
        if not confidence_ok:
            return _status(lock_ready=False, reason="low_confidence", confidence_ok=False, lock_count_reset_reason="low_confidence")
        if obs.depth_valid is False:
            return _status(lock_ready=False, reason="vision_stale", confidence_ok=confidence_ok, lock_count_reset_reason="obs_invalid")
        yaw_tol = float(getattr(self.cfg, "table_yaw_tol_rad", self.cfg.final_lock_yaw_tol_rad))
        dist_tol = float(getattr(self.cfg, "table_dist_tol_m", self.cfg.final_lock_dist_tol_m))
        yaw_ok = obs.yaw_err_rad is not None and abs(float(obs.yaw_err_rad)) <= yaw_tol
        measured_dist = self._table_measured_dist_m(obs)
        target_dist = self._table_target_dist_m(obs)
        dist_delta = (float(measured_dist) - target_dist) if measured_dist is not None else None
        dist_ok = dist_delta is not None and abs(float(dist_delta)) <= dist_tol
        lat_ok = obs.lateral_err_m is None or abs(float(obs.lateral_err_m)) <= float(self.cfg.final_lock_lateral_tol_m)
        reason = "stable_count_not_enough"
        if not yaw_ok:
            reason = "yaw_not_aligned"
        elif dist_delta is None:
            reason = "vision_stale"
        elif not dist_ok:
            reason = "distance_too_far" if float(dist_delta) > 0 else "distance_too_close"
        elif not lat_ok:
            reason = "yaw_not_aligned"
        lock_ready = bool(yaw_ok and dist_ok and lat_ok and stale_reason != "soft_stale")
        hold_reason = "soft_stale" if stale_reason == "soft_stale" else ""
        reset_reason = "" if (lock_ready or hold_reason) else reason
        return _status(
            lock_ready=lock_ready,
            reason="soft_stale" if hold_reason else reason,
            yaw_locked=bool(yaw_ok),
            dist_locked=bool(dist_ok),
            lat_locked=bool(lat_ok),
            confidence_ok=confidence_ok,
            lock_count_inc_reason="fresh_lock_ready" if lock_ready else "",
            lock_count_hold_reason=hold_reason,
            lock_count_reset_reason=reset_reason,
        )

    def _log_final_lock_summary(
        self,
        obs: Optional[TableEdgeObs],
        *,
        lock_ready: bool,
        reason: str,
        stable_count: int,
        phase: str = "",
        status: Optional[Dict[str, object]] = None,
    ) -> None:
        status = dict(status or self._final_lock_status(obs, stable_count=stable_count))
        measured_distance = None
        target_distance = None
        if obs is not None:
            target_distance = obs.target_dist_m
            if obs.dist_err_m is not None and target_distance is not None:
                measured_distance = float(target_distance) + float(obs.dist_err_m)
        reset_reason = str(status.get("lock_count_reset_reason") or status.get("lock_reset_reason") or "")
        stale_reason = str(status.get("vision_stale_reason") or "")
        reason_text = str(reason or status.get("reason") or "")
        should_emit = bool(
            reset_reason
            or status.get("lock_count_hold_reason")
            or status.get("lock_count_inc_reason")
            or status.get("final_lock_transition_block_reason")
            or status.get("final_lock_transition_reason")
            or "stale" in stale_reason
            or "stale" in reason_text
            or "lost" in reason_text
            or "reset" in reason_text
        )
        if not should_emit:
            return
        lines = [
            "FINAL_LOCK summary:",
            f"phase={phase or str(self.ctx.table_dock_phase or '').upper() or self.ctx.state.value}",
            f"state={self.ctx.state.value}",
            f"obs_seq={status.get('obs_seq')}",
            f"table_found={bool(obs.table_found) if obs is not None else False}",
            f"conf={float(obs.confidence or 0.0):.3f}" if obs is not None else "conf=0.000",
            f"yaw_err={obs.yaw_err_rad}" if obs is not None else "yaw_err=None",
            f"obs_age_ms={status.get('obs_age_ms')}",
            f"measured_distance={measured_distance}",
            f"target_distance={target_distance}",
            f"dist_err={obs.dist_err_m if obs is not None else None}",
            f"yaw_ok={bool(status.get('yaw_ok'))}",
            f"dist_ok={bool(status.get('dist_ok'))}",
            f"yaw_locked={bool(status['yaw_locked'])}",
            f"dist_locked={bool(status['dist_locked'])}",
            f"age_ok={bool(status.get('age_ok'))}",
            f"confidence_ok={bool(status.get('confidence_ok'))}",
            f"valid_for_control={bool(status.get('valid_for_control'))}",
            f"usable_for_approach={bool(status.get('usable_for_approach'))}",
            f"usable_for_stop={bool(status.get('usable_for_stop'))}",
            f"raw_control_level={status.get('raw_control_level')}",
            f"normalized_control_level={status.get('normalized_control_level')}",
            f"control_level={status.get('control_level')}",
            f"stable_lock_count={int(stable_count)}",
            f"stable_count={int(stable_count)}",
            f"required_lock_count={int(status.get('required_lock_count', 0) or 0)}",
            f"lock_ready_obs_count={int(status.get('lock_ready_obs_count', 0) or 0)}",
            f"window_ready_count={int(status.get('window_ready_count', 0) or 0)}",
            f"required_ready_obs={int(status.get('required_ready_obs', 0) or 0)}",
            f"final_lock_window_ms={int(status.get('final_lock_window_ms', 0) or 0)}",
            f"same_obs_reuse_count={int(status.get('same_obs_reuse_count', 0) or 0)}",
            f"consecutive_lost_count={int(status.get('consecutive_lost_count', 0) or 0)}",
            f"lock_count_inc_reason={status.get('lock_count_inc_reason')}",
            f"lock_count_hold_reason={status.get('lock_count_hold_reason')}",
            f"lock_count_reset_reason={status.get('lock_count_reset_reason')}",
            f"final_lock_transition_block_reason={status.get('final_lock_transition_block_reason')}",
            f"final_lock_transition_reason={status.get('final_lock_transition_reason')}",
            f"vision_stale_reason={status.get('vision_stale_reason')}",
            f"lock_reset_reason={status.get('lock_reset_reason')}",
            f"lock_ready={bool(lock_ready)}",
            f"reason={reason or status['reason']}",
        ]
        self._log("info", "\n".join(lines))

    def _target_matches_active(self, obs: TargetObs) -> bool:
        if not self.ctx.active_target or not obs.target:
            return True
        return str(obs.target).strip() == str(self.ctx.active_target).strip()

    def _target_cls_matches_active(self, obs: TargetObs) -> bool:
        active = str(self.ctx.active_target or "").strip()
        if not active:
            return True
        candidate_cls = str(obs.matched_cls or obs.target or "").strip()
        if not candidate_cls:
            return True
        return candidate_cls == active

    def _target_conf_value(self, obs: TargetObs) -> Optional[float]:
        value = obs.matched_conf if obs.matched_conf is not None else obs.confidence
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _target_bbox_valid(self, obs: Optional[TargetObs]) -> bool:
        if obs is None:
            return False
        if getattr(obs, "bbox_valid", None) is False:
            return False
        return True

    def _trim_target_obs_window(self, now_m: Optional[float] = None) -> None:
        now_m = monotonic_ts() if now_m is None else float(now_m)
        window_s = max(0.2, float(getattr(self.cfg, "target_confirm_window_s", 1.5) or 1.5))
        cutoff = now_m - window_s
        self.ctx.target_obs_window = [
            dict(item) for item in self.ctx.target_obs_window if float(item.get("t", 0.0) or 0.0) >= cutoff
        ][-80:]

    def _record_target_window_sample(self, obs: Optional[TargetObs], reason: str = "") -> Dict[str, Any]:
        now_m = monotonic_ts()
        basic_found = False
        conf = None
        center = None
        bbox_valid = False
        matched_cls = None
        if obs is not None:
            conf = self._target_conf_value(obs)
            center = self._target_center_pair(obs)
            bbox_valid = self._target_bbox_valid(obs)
            matched_cls = obs.matched_cls or obs.target
            basic_found = bool(obs.found) and self._target_cls_matches_active(obs) and self._target_matches_active(obs) and bbox_valid
        sample = {
            "t": now_m,
            "found": bool(basic_found),
            "conf": conf,
            "center": center,
            "bbox_valid": bool(bbox_valid),
            "matched_cls": matched_cls,
            "reason": str(reason or ""),
        }
        self.ctx.target_obs_window.append(sample)
        self._trim_target_obs_window(now_m)
        return self._target_window_stats(now_m=now_m)

    def _target_window_stats(self, now_m: Optional[float] = None) -> Dict[str, Any]:
        self._trim_target_obs_window(now_m)
        samples = list(self.ctx.target_obs_window)
        total = len(samples)
        found_samples = [item for item in samples if bool(item.get("found", False))]
        confs = sorted(
            float(item.get("conf"))
            for item in found_samples
            if item.get("conf") is not None
        )
        if confs:
            mid = len(confs) // 2
            conf_median = confs[mid] if len(confs) % 2 else (confs[mid - 1] + confs[mid]) / 2.0
            conf_max = max(confs)
        else:
            conf_median = None
            conf_max = None
        centers = [
            tuple(item.get("center"))
            for item in found_samples
            if isinstance(item.get("center"), (tuple, list)) and len(item.get("center")) >= 2
        ]
        center_jitter = 0.0
        if len(centers) >= 2:
            mean_cx = sum(float(item[0]) for item in centers) / float(len(centers))
            mean_cy = sum(float(item[1]) for item in centers) / float(len(centers))
            center_jitter = max(math.hypot(float(item[0]) - mean_cx, float(item[1]) - mean_cy) for item in centers)
        valid_bbox_count = sum(1 for item in samples if bool(item.get("bbox_valid", False)))
        latest_found = found_samples[-1] if found_samples else {}
        found_ratio = float(len(found_samples)) / float(total) if total else 0.0
        bbox_valid_ratio = float(valid_bbox_count) / float(total) if total else 0.0
        return {
            "samples": total,
            "found_samples": len(found_samples),
            "found_ratio": found_ratio,
            "conf_median": conf_median,
            "conf_max": conf_max,
            "center_jitter": float(center_jitter),
            "bbox_valid_ratio": bbox_valid_ratio,
            "latest_matched_cls": latest_found.get("matched_cls"),
            "latest_matched_conf": latest_found.get("conf"),
        }

    def _target_found_reason(self, obs: TargetObs) -> str:
        matched_cls = str(obs.matched_cls or obs.target or "").strip() or "n/a"
        conf = self._target_conf_value(obs)
        if conf is None:
            return f"target_found matched_cls={matched_cls} matched_conf=n/a"
        return f"target_found matched_cls={matched_cls} matched_conf={float(conf):.3f}"

    def _target_quality_ok(self, obs: TargetObs, conf_th: float) -> bool:
        return self._target_candidate_status(obs, conf_th, min_area=0.0)[0]

    def _target_stable_ms(self) -> int:
        if self.ctx.target_stable_since_mono <= 0.0:
            return 0
        return int(round(max(0.0, monotonic_ts() - self.ctx.target_stable_since_mono) * 1000.0))

    def _target_center_pair(self, obs: Optional[TargetObs]) -> Optional[Tuple[float, float]]:
        if obs is None:
            return None
        full = self._target_center_full_norm(obs)
        if full is None:
            full = self._target_center(obs)
        if full is None:
            return None
        cx = self._float_or_none(full.get("cx"))
        cy = self._float_or_none(full.get("cy"))
        if cx is None and cy is None:
            return None
        return (float(cx if cx is not None else 0.0), float(cy if cy is not None else 0.0))

    def _target_bbox_area(self, obs: TargetObs) -> Optional[float]:
        for value in (obs.matched_area, obs.size_norm, obs.mask_area_ratio):
            numeric = self._float_or_none(value)
            if numeric is not None and numeric > 0.0:
                return numeric
        bbox = obs.matched_bbox or obs.bbox or obs.mask_bbox
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            values = [self._float_or_none(item) for item in bbox[:4]]
            if all(item is not None for item in values):
                x1, y1, x2, y2 = [float(item) for item in values]  # type: ignore[arg-type]
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                if width > 1.0 or height > 1.0:
                    return None
                return width * height
        return None

    def _target_candidate_status(
        self,
        obs: Optional[TargetObs],
        conf_th: float,
        *,
        min_area: float = 0.0,
    ) -> Tuple[bool, str]:
        if obs is None:
            return False, "vision_stale"
        if not bool(obs.found):
            return False, "target_lost"
        if not self._target_cls_matches_active(obs):
            return False, "class_mismatch"
        if not self._target_matches_active(obs):
            return False, "target_mismatch"
        if not self._target_bbox_valid(obs):
            reason = str(getattr(obs, "bbox_invalid_reason", "") or "bbox_invalid")
            return False, reason
        conf = self._target_conf_value(obs)
        if conf is None:
            if float(conf_th or 0.0) > 0.0:
                return False, "conf_missing"
        elif conf < float(conf_th or 0.0):
            return False, f"conf_low conf={conf:.3f} th={float(conf_th or 0.0):.3f}"
        area_th = float(min_area or 0.0)
        area = self._target_bbox_area(obs)
        if area_th > 0.0 and area is not None and area < area_th:
            return False, f"bbox_area_low area={area:.4f} th={area_th:.4f}"
        return True, "target_visible"

    def _update_target_stability(self, obs: TargetObs) -> float:
        now_m = monotonic_ts()
        if self.ctx.target_stable_since_mono <= 0.0:
            self.ctx.target_stable_since_mono = now_m
        self.ctx.target_loss_since_mono = 0.0
        self.ctx.target_last_lost_reason = ""
        center = self._target_center_pair(obs)
        if center is not None:
            self.ctx.target_center_history.append({"t": now_m, "cx": center[0], "cy": center[1]})
        window_s = max(float(self.cfg.target_lock_stable_s), float(self.cfg.target_confirm_min_s), 0.5)
        cutoff = now_m - window_s
        self.ctx.target_center_history = [
            item for item in self.ctx.target_center_history if float(item.get("t", 0.0) or 0.0) >= cutoff
        ][-30:]
        self.ctx.target_last_center_jitter = self._target_center_jitter()
        return self.ctx.target_last_center_jitter

    def _target_center_jitter(self) -> float:
        points = self.ctx.target_center_history
        if len(points) < 2:
            return 0.0
        mean_cx = sum(float(item.get("cx", 0.0) or 0.0) for item in points) / float(len(points))
        mean_cy = sum(float(item.get("cy", 0.0) or 0.0) for item in points) / float(len(points))
        return max(
            math.hypot(float(item.get("cx", 0.0) or 0.0) - mean_cx, float(item.get("cy", 0.0) or 0.0) - mean_cy)
            for item in points
        )

    def _reset_target_stability(self, reason: str) -> None:
        self.ctx.target_stable_since_mono = 0.0
        self.ctx.target_center_history.clear()
        self.ctx.target_last_center_jitter = 0.0
        self.ctx.target_last_lost_reason = str(reason or "")

    def _format_target_transition_reason(self, reason: str, obs: Optional[TargetObs] = None) -> str:
        obs = obs or self.ctx.last_target_obs
        matched_cls = str(obs.matched_cls or obs.target or "").strip() if obs is not None else ""
        conf = self._target_conf_value(obs) if obs is not None else None
        center = self._target_center(obs)
        window = self._target_window_stats()
        parts = [
            str(reason or "target_transition"),
            f"found_frames={int(self.ctx.target_found_frames)}",
            f"lost_frames={int(self.ctx.target_lost_frames)}",
            f"target_stable_ms={self._target_stable_ms()}",
            f"found_ratio={float(window.get('found_ratio', 0.0) or 0.0):.2f}",
        ]
        if matched_cls:
            parts.append(f"matched_cls={matched_cls}")
        if conf is not None:
            parts.append(f"matched_conf={float(conf):.3f}")
        if window.get("conf_median") is not None:
            parts.append(f"conf_median={float(window.get('conf_median')):.3f}")
        if window.get("conf_max") is not None:
            parts.append(f"conf_max={float(window.get('conf_max')):.3f}")
        if center is not None:
            parts.append(f"matched_center_full_norm={center}")
        offset = self._target_center_offset_norm(obs)
        if offset is not None:
            parts.append(f"matched_center_offset_norm={offset}")
        parts.append(f"center_jitter={float(self.ctx.target_last_center_jitter):.3f}")
        parts.append(f"bbox_valid_ratio={float(window.get('bbox_valid_ratio', 0.0) or 0.0):.2f}")
        if self.ctx.target_last_lost_reason:
            parts.append(f"lost_reason={self.ctx.target_last_lost_reason}")
        if self.ctx.target_last_transition_reason:
            parts.append(f"decision={self.ctx.target_last_transition_reason}")
        return " ".join(parts)

    def _obs_has_motion(self, obs: Optional[TargetObs]) -> bool:
        if obs is None:
            return False
        return obs.vx_norm is not None or obs.vy_norm is not None or obs.wz_norm is not None

    def _edge_slide_direction(self) -> int:
        segment_s = max(0.2, float(self.cfg.edge_slide_segment_s))
        segment_index = int(self._state_elapsed() / segment_s)
        return self.ctx.slide_direction_sign if segment_index % 2 == 0 else -self.ctx.slide_direction_sign

    def _segment_elapsed(self, segment_s: float) -> float:
        segment_s = max(0.1, float(segment_s))
        return self._state_elapsed() % segment_s

    def _can_relocate_edge(self) -> bool:
        if not bool(self.cfg.edge_relocate_enabled):
            return False
        if self.ctx.edge_transition_count >= int(self.cfg.max_edge_transitions_per_task):
            return False
        return self.ctx.edge_visit_index + 1 < len(self.ctx.edge_visit_order)

    def _handle_table_loss(self, reason: str, fallback_state: State, hold_mode: str) -> MotionDecision:
        self.ctx.table_lost_frames += 1
        self._start_loss_timer("table_loss_since_mono")
        lost_frames_ok = self.ctx.table_lost_frames >= int(self.cfg.table_lost_frames_to_reacquire)
        lost_hold_ok = self._loss_elapsed(self.ctx.table_loss_since_mono) >= float(self.cfg.table_loss_hold_s)
        min_dwell_ok = self._state_elapsed() >= float(self.cfg.approach_min_dwell_s)
        if lost_frames_ok and lost_hold_ok and min_dwell_ok:
            self._transition(fallback_state, reason)
            return self.controller.search_table_cmd(turn_sign=self.ctx.relocate_turn_sign)
        return self.controller.stop_cmd(hold_mode)

    def _enter_dock_retry_or_next(self, reason: str) -> MotionDecision:
        self.ctx.last_fail_reason = reason
        if self.ctx.dock_retry_count < int(self.cfg.dock_retry_limit):
            self.ctx.dock_retry_count += 1
            self._transition(State.DOCK_RETRY, f"{reason}，准备重试第 {self.ctx.dock_retry_count} 次")
            return self.controller.leave_edge_cmd()
        self._transition(State.NEXT_TABLE, reason)
        return self.controller.next_table_cmd(turn_sign=self.ctx.relocate_turn_sign)

    def _extract_obstacle_signal(self) -> ObstacleSignal:
        table_obs = self._fresh_table_obs()
        if table_obs is not None and bool(table_obs.obstacle_flag):
            return ObstacleSignal(
                active=True,
                best_turn_dir=str(table_obs.best_turn_dir or ""),
                distance_m=table_obs.obstacle_distance_m,
                source="table_edge_obs",
            )
        target_obs = self._fresh_target_obs()
        if target_obs is not None and bool(target_obs.obstacle_flag):
            return ObstacleSignal(
                active=True,
                best_turn_dir=str(target_obs.best_turn_dir or ""),
                distance_m=target_obs.obstacle_distance_m,
                source="target_obs",
            )
        home_obs = self._fresh_home_obs()
        if home_obs is not None and bool(home_obs.obstacle_flag):
            return ObstacleSignal(
                active=True,
                best_turn_dir=str(home_obs.best_turn_dir or ""),
                distance_m=home_obs.obstacle_distance_m,
                source="home_tag_obs",
            )
        return ObstacleSignal(active=False)

    def _check_safety_interlock(self) -> Optional[MotionDecision]:
        obstacle = self._extract_obstacle_signal()
        if self.ctx.state == State.AVOID_OBSTACLE:
            return None
        if self.ctx.state in MOVING_STATES and obstacle.active:
            self.ctx.last_safety_reason = f"检测到障碍({obstacle.source})"
            self.ctx.resume_state = self.ctx.state
            self.ctx.avoid_retry_count += 1
            self._transition(State.AVOID_OBSTACLE, self.ctx.last_safety_reason)
            self._queue_tts("前方有障碍，开始避障")
            return self.controller.stop_cmd("AVOID_OBSTACLE", brake=True)
        return None
