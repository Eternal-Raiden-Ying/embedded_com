#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config.schema import CarMotionConfig, ControlThresholds
from ..control.types import DockingControlConfig
from ..ipc.protocol import (
    CarState,
    HomeTagObs,
    TableEdgeObs,
    TargetObs,
    TaskCmd,
    make_tts_event,
    make_vision_idle,
    make_vision_req,
)
from .common import monotonic_ts
from .context import RuntimeContext, State
from .controller import MotionController, MotionDecision


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
}


TABLE_VISION_STATES = {
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
        self._last_req_mono = 0.0
        self._last_stop_mono = 0.0

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
        }
        return dispatch.get(self.ctx.state, self._tick_idle)()

    def export_state_block(self) -> Dict:
        table_obs = self.ctx.last_table_obs
        target_obs = self.ctx.last_target_obs
        lock_status = self._final_lock_status(table_obs, stable_count=self.ctx.table_lock_frames)
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
            "approach_aligned_frames": self.ctx.approach_aligned_frames,
            "target_found_frames": self.ctx.target_found_frames,
            "target_lost_frames": self.ctx.target_lost_frames,
            "target_lock_frames": self.ctx.target_lock_frames,
            "tag_lost_frames": self.ctx.tag_lost_frames,
            "tag_arrived_frames": self.ctx.tag_arrived_frames,
            "dock_retry_count": self.ctx.dock_retry_count,
            "avoid_retry_count": self.ctx.avoid_retry_count,
            "table_loss_elapsed_s": self._loss_elapsed(self.ctx.table_loss_since_mono),
            "target_loss_elapsed_s": self._loss_elapsed(self.ctx.target_loss_since_mono),
            "tag_loss_elapsed_s": self._loss_elapsed(self.ctx.tag_loss_since_mono),
            "has_table_edge_obs": table_obs is not None,
            "has_target_obs": target_obs is not None,
            "lock_ready": bool(lock_status["lock_ready"]),
            "lock_reason": str(lock_status["reason"]),
            "table_found": bool(table_obs.table_found) if table_obs is not None else False,
            "edge_found": bool(table_obs.edge_found) if table_obs is not None else False,
            "confidence": table_obs.confidence if table_obs is not None else None,
            "yaw_err_rad": table_obs.yaw_err_rad if table_obs is not None else None,
            "dist_err_m": table_obs.dist_err_m if table_obs is not None else None,
            "target_dist_m": table_obs.target_dist_m if table_obs is not None else None,
        }

    def _transition(self, new_state: State, reason: str):
        old_state = self.ctx.state
        if old_state == new_state:
            self.ctx.last_enter_reason = reason
            return
        snapshot = self._build_transition_snapshot(old_state, new_state, reason)
        self._log("info", f"状态切换 {old_state.value} -> {new_state.value} ({reason})")
        self.ctx.prev_state = old_state
        self.ctx.state = new_state
        self.ctx.state_enter_mono = monotonic_ts()
        self.ctx.state_enter_wall_ts = time.time()
        self.ctx.last_enter_reason = reason
        self.last_transition_snapshot = snapshot
        self.ctx.clear_motion_counters()
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
        return {
            "cx": self._float_or_none(obs.cx_norm),
            "cy": self._float_or_none(obs.cy_norm),
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
            return self._frames_to_ms(self.ctx.target_found_frames or self.ctx.target_lock_frames)
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
            "target_found_frames": int(self.ctx.target_found_frames),
            "target_found_frames_to_confirm": int(self.cfg.target_found_frames_to_confirm),
            "target_lost_frames": int(self.ctx.target_lost_frames),
            "target_confirm_lost_frames": int(self.cfg.target_confirm_lost_frames),
            "edge_settle_ms": int(round(float(self.cfg.edge_settle_s) * 1000.0)),
            "search_target_init_hold_ms": int(round(float(self.cfg.search_target_init_hold_s) * 1000.0)),
            "target_lock_settle_ms": int(round(float(self.cfg.target_lock_settle_s) * 1000.0)),
            "freeze_settle_ms": int(round(float(self.cfg.freeze_settle_s) * 1000.0)),
            "approach_timeout_ms": int(round(float(self.cfg.approach_timeout_s) * 1000.0)),
            "target_search_timeout_ms": int(round(float(self.cfg.target_search_timeout_s) * 1000.0)),
        }

    def _build_transition_snapshot(self, old_state: State, new_state: State, reason: str) -> Dict[str, Any]:
        table_obs = self.ctx.last_table_obs
        target_obs = self.ctx.last_target_obs
        car_state = self.ctx.last_car_state
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
            "stable_ms": self._transition_stable_ms(old_state),
            "lost_ms": self._transition_lost_ms(old_state),
            "target_found": bool(target_obs.found) if target_obs is not None else None,
            "target_conf": self._float_or_none(target_obs.confidence if target_obs is not None else None),
            "target_cls": (target_obs.best_cls or target_obs.target if target_obs is not None else None),
            "target_center": self._target_center(target_obs),
            "target_stable_ms": self._frames_to_ms(self.ctx.target_found_frames or self.ctx.target_lock_frames),
            "car_mode": car_state.mode if car_state is not None else None,
            "planned_cmd": {"vx": None, "vy": None, "wz": None},
            "condition": self._transition_condition_snapshot(old_state, new_state),
        }

    def _on_enter_state(self, state: State):
        if state == State.IDLE:
            self.ctx.clear_task_context()
            self.ctx.active_vision_stage = ""
            self.ctx.active_vision_mode = ""
            self.ctx.resume_state = None
            return
        req = self._active_req_payload()
        if req is not None:
            self._queue_vision_req(req, force=True)

    def _interrupt_to_idle(self, reason: str, tts_text: Optional[str] = None, interrupt_tts: bool = False, send_vision_idle: bool = False):
        if send_vision_idle:
            self._queue_vision_req(make_vision_idle(session_id=self.ctx.active_session_id, epoch=self.ctx.active_epoch), force=True)
        self.ctx.clear_task_context()
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
        if not force and now_m - self._last_req_mono < self.cfg.req_resend_period_s:
            return
        if self.ctx.active_session_id and not payload.get("session_id"):
            payload["session_id"] = self.ctx.active_session_id
        if self.ctx.active_epoch and payload.get("epoch") in (None, 0):
            payload["epoch"] = self.ctx.active_epoch
        self.ctx.active_req_id = str(payload.get("req_id", self.ctx.active_req_id) or self.ctx.active_req_id)
        self.ctx.pending_vision_msgs.append(payload)
        self._last_req_mono = now_m

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
        self.ctx.task_intent = "FIND"
        self.ctx.active_target = target
        self.ctx.active_session_id = cmd.session_id
        self.ctx.active_epoch = cmd.epoch
        self.ctx.task_start_wall_ts = time.time()
        self._transition(State.SEARCH_TABLE, f"开始桌边任务，目标 {target}")
        self._queue_tts(f"开始寻找 {target}")

    def _start_return_task(self, cmd: TaskCmd):
        self.ctx.clear_task_context()
        self.ctx.task_intent = "RETURN"
        self.ctx.active_session_id = cmd.session_id
        self.ctx.active_epoch = cmd.epoch
        self.ctx.task_start_wall_ts = time.time()
        self._transition(State.RETURN_HOME, "开始返航")
        self._queue_tts("开始返航")

    def _tick_idle(self) -> MotionDecision:
        return self.controller.stop_cmd("IDLE")

    def _tick_search_table(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        obs = self._fresh_table_obs()
        if self._table_visible(obs):
            self.ctx.table_found_frames += 1
            if self.ctx.table_found_frames >= int(self.cfg.table_found_frames_to_approach):
                self._transition(State.COARSE_ALIGN, "稳定发现桌边")
                return self.controller.coarse_align_cmd(obs)
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
        if self._coarse_aligned(obs):
            self.ctx.approach_aligned_frames += 1
            if self.ctx.approach_aligned_frames >= int(self.cfg.coarse_align_frames_to_advance):
                self._transition(State.CONTROLLED_APPROACH, "完成粗对齐，开始受控接近")
                return self.controller.controlled_approach_cmd(obs)
        else:
            self.ctx.approach_aligned_frames = 0
        if self._state_elapsed() >= float(self.cfg.approach_timeout_s):
            return self._enter_dock_retry_or_next("粗对齐超时")
        self._maybe_resend_req(self._active_req_payload())
        return self.controller.coarse_align_cmd(obs)

    def _tick_controlled_approach(self) -> MotionDecision:
        obs = self._fresh_table_obs()
        if not self._table_visible(obs):
            return self._handle_table_loss("接近时桌边丢失，回到搜索", State.SEARCH_TABLE, "CONTROLLED_APPROACH_HOLD")
        self._reset_table_loss()
        if self._edge_ready(obs):
            self._transition(State.FINAL_LOCK, "进入最终锁边")
            return self.controller.final_lock_cmd(obs)
        if self._state_elapsed() >= float(self.cfg.approach_timeout_s):
            return self._enter_dock_retry_or_next("受控接近超时")
        self._maybe_resend_req(self._active_req_payload())
        return self.controller.controlled_approach_cmd(obs)

    def _tick_final_lock(self) -> MotionDecision:
        obs = self._fresh_table_obs()
        if not self._table_visible(obs):
            stale_obs = self.ctx.last_table_obs if obs is None else obs
            reason = str(self._final_lock_status(stale_obs if stale_obs is not None else obs, stable_count=self.ctx.table_lock_frames)["reason"])
            if obs is None and self.ctx.last_table_obs is not None:
                reason = "vision_stale"
            self._log_final_lock_summary(obs, lock_ready=False, reason=reason, stable_count=self.ctx.table_lock_frames)
            return self._handle_table_loss("最终锁边时桌边丢失，回到搜索", State.SEARCH_TABLE, "FINAL_LOCK_HOLD")
        self._reset_table_loss()
        lock_status = self._final_lock_status(obs, stable_count=self.ctx.table_lock_frames)
        if bool(lock_status["lock_ready"]):
            self.ctx.table_lock_frames += 1
            if self.ctx.table_lock_frames >= int(self.cfg.final_lock_frames_to_arrive):
                self.ctx.dock_retry_count = 0
                self._log_final_lock_summary(obs, lock_ready=True, reason="lock_ready", stable_count=self.ctx.table_lock_frames)
                self._transition(State.AT_TABLE_EDGE, "lock_ready")
                self._queue_tts("已完成桌边停靠")
                return self.controller.stop_cmd("AT_TABLE_EDGE")
        else:
            self.ctx.table_lock_frames = 0
        if self._state_elapsed() >= float(self.cfg.approach_timeout_s):
            reason = "stable_count_not_enough" if bool(lock_status["lock_ready"]) else str(lock_status["reason"])
            self._log_final_lock_summary(obs, lock_ready=False, reason=reason, stable_count=self.ctx.table_lock_frames)
            return self._enter_dock_retry_or_next(f"最终锁边超时:{reason}")
        self._maybe_resend_req(self._active_req_payload())
        return self.controller.final_lock_cmd(obs)

    def _tick_dock_retry(self) -> MotionDecision:
        if self._state_elapsed() < float(self.cfg.dock_retry_backoff_s):
            return self.controller.leave_edge_cmd()
        self._transition(State.SEARCH_TABLE, "停靠重试完成，重新搜索桌边")
        return self.controller.search_table_cmd(turn_sign=self.ctx.relocate_turn_sign)

    def _tick_at_table_edge(self) -> MotionDecision:
        if self._state_elapsed() < float(self.cfg.edge_settle_s):
            return self.controller.stop_cmd("AT_TABLE_EDGE")
        self._transition(State.SEARCH_TARGET_INIT, "桌边姿态稳定，初始化沿边搜索")
        return self.controller.stop_cmd("AT_TABLE_EDGE")

    def _tick_search_target_init(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        if self._state_elapsed() < float(self.cfg.search_target_init_hold_s):
            return self.controller.stop_cmd("SEARCH_TARGET_INIT")
        self._transition(State.EDGE_SLIDE_SEARCH, "开始沿桌边搜索目标")
        self._queue_tts("开始沿桌边搜索目标")
        return self.controller.stop_cmd("SEARCH_TARGET_INIT")

    def _tick_edge_slide_search(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        target_obs = self._fresh_target_obs()
        if (
            target_obs is not None
            and target_obs.found
            and self._target_matches_active(target_obs)
        ):
            self._transition(State.TARGET_CONFIRM, "target_found")
            return self.controller.stop_cmd("TARGET_CONFIRM")
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
        if edge_obs is None or not self._table_visible(edge_obs):
            return self._handle_edge_slide_edge_loss("edge_obs_missing" if edge_obs is None else "edge_not_visible")
        if edge_obs.dist_err_m is None:
            return self._handle_edge_slide_edge_loss("dist_err_missing")
        if abs(float(edge_obs.dist_err_m)) > float(self.cfg.edge_slide_dist_tolerance_m):
            self._start_loss_timer("table_loss_since_mono")
            lost_s = self._loss_elapsed(self.ctx.table_loss_since_mono)
            self.ctx.last_fail_reason = f"沿边距离偏差超过容忍: dist_err={float(edge_obs.dist_err_m):+.3f}m"
            if lost_s >= float(self.cfg.table_loss_hold_s):
                fallback_state = self._edge_slide_fallback_state()
                self._transition(
                    fallback_state,
                    (
                        "edge_distance_out_of_tolerance "
                        f"dist={float(edge_obs.dist_err_m):+.3f} "
                        f"tol={float(self.cfg.edge_slide_dist_tolerance_m):.3f} "
                        f"hold_s={lost_s:.2f}"
                    ),
                )
                return self._edge_slide_fallback_cmd(fallback_state, edge_obs)
        else:
            self._reset_table_loss()
        if self._obs_has_motion(target_obs):
            return self.controller.target_track_cmd(target_obs)
        direction = self._edge_slide_direction()
        return self.controller.edge_slide_search_cmd(
            self._segment_elapsed(self.cfg.edge_slide_segment_s),
            direction_sign=direction,
            edge_obs=edge_obs,
        )

    def _edge_slide_fallback_state(self) -> State:
        raw = str(getattr(self.cfg, "edge_slide_fallback_state", "") or "").strip().upper()
        if raw == State.FINAL_LOCK.value:
            return State.FINAL_LOCK
        return State.CONTROLLED_APPROACH

    def _edge_slide_fallback_cmd(self, state: State, edge_obs: Optional[TableEdgeObs]) -> MotionDecision:
        if state == State.FINAL_LOCK:
            return self.controller.final_lock_cmd(edge_obs)
        return self.controller.controlled_approach_cmd(edge_obs)

    def _handle_edge_slide_edge_loss(self, reason: str) -> MotionDecision:
        self._start_loss_timer("table_loss_since_mono")
        lost_s = self._loss_elapsed(self.ctx.table_loss_since_mono)
        if lost_s < float(self.cfg.table_loss_hold_s):
            return self.controller.edge_slide_hold_cmd(f"{reason}_hold lost_s={lost_s:.2f}")
        fallback_state = self._edge_slide_fallback_state()
        self._transition(fallback_state, f"{reason} lost_s={lost_s:.2f}")
        return self._edge_slide_fallback_cmd(fallback_state, self.ctx.last_table_obs)

    def _tick_target_confirm(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        obs = self._fresh_target_obs()
        if (
            obs is not None
            and obs.found
            and self._target_matches_active(obs)
        ):
            self.ctx.target_found_frames += 1
            self.ctx.target_lost_frames = 0
            if self.ctx.target_found_frames >= int(self.cfg.target_found_frames_to_confirm):
                self._transition(State.TARGET_LOCKED, "目标确认完成")
                return self.controller.stop_cmd("TARGET_LOCKED")
            return self.controller.stop_cmd("TARGET_CONFIRM")
        self.ctx.target_lost_frames += 1
        if self.ctx.target_lost_frames >= int(self.cfg.target_confirm_lost_frames):
            self._transition(State.EDGE_SLIDE_SEARCH, "目标确认失败，恢复沿边搜索")
        return self.controller.stop_cmd("TARGET_CONFIRM")

    def _tick_target_locked(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        obs = self._fresh_target_obs()
        if obs is None or not obs.found or not self._target_matches_active(obs):
            self._transition(State.EDGE_SLIDE_SEARCH, "目标锁定丢失，恢复沿边搜索")
            return self.controller.stop_cmd("EDGE_SLIDE_SEARCH")
        self.ctx.target_lock_frames += 1
        if self._state_elapsed() >= float(self.cfg.target_lock_settle_s):
            self._transition(State.FREEZE_BASE, "目标稳定，冻结底盘")
            return self.controller.stop_cmd("FREEZE_BASE")
        return self.controller.stop_cmd("TARGET_LOCKED")

    def _tick_freeze_base(self) -> MotionDecision:
        if self._state_elapsed() < float(self.cfg.freeze_settle_s):
            return self.controller.stop_cmd("FREEZE_BASE")
        self._transition(State.DONE, f"已在桌边锁定 {self.ctx.active_target}")
        self._queue_tts(f"已在桌边锁定 {self.ctx.active_target}")
        return self.controller.stop_cmd("DONE")

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
                return self.controller.coarse_align_cmd(obs)
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

    def _active_req_payload(self) -> Optional[Dict]:
        binding = self._vision_binding_for_state(self.ctx.state)
        if binding is None:
            return None
        same_stage = bool(self.ctx.active_vision_stage) and self.ctx.active_vision_stage == binding.stage
        op = "UPDATE" if same_stage else "START"
        self.ctx.active_vision_stage = binding.stage
        self.ctx.active_vision_mode = binding.mode_hint
        return make_vision_req(
            target=binding.target,
            session_id=self.ctx.active_session_id,
            epoch=self.ctx.active_epoch,
            op=op,
            stage=binding.stage,
            mode_hint=binding.mode_hint,
            payload=dict(binding.payload or {}),
        )

    def _vision_binding_for_state(self, state: State) -> Optional[VisionStageBinding]:
        if state in TABLE_VISION_STATES:
            return VisionStageBinding(
                stage="SEARCH",
                mode_hint="DEPTH_PERCEPTION",
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
                mode_hint="TRACK_LOCAL",
                target=self.ctx.active_target,
                payload={
                    "search_kind": "TARGET",
                    "need_depth": True,
                    "edge_follow": True,
                    "current_edge_id": self.ctx.current_edge_id,
                    "orchestrator_state": state.value,
                    "edge_visit_index": int(self.ctx.edge_visit_index),
                },
            )
        if state == State.RETURN_HOME:
            return VisionStageBinding(
                stage="RETURN",
                mode_hint="TRACK_LOCAL",
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
        return bool(obs is not None and obs.table_found)

    def _coarse_aligned(self, obs: Optional[TableEdgeObs]) -> bool:
        if obs is None:
            return False
        if obs.yaw_err_rad is not None:
            return abs(float(obs.yaw_err_rad)) <= float(self.cfg.coarse_align_done_rad)
        if obs.table_cx_norm is not None:
            return abs(float(obs.table_cx_norm)) <= 0.12
        return False

    def _edge_ready(self, obs: Optional[TableEdgeObs]) -> bool:
        if obs is None:
            return False
        if obs.edge_ready is not None:
            return bool(obs.edge_ready)
        if obs.dist_err_m is not None:
            return abs(float(obs.dist_err_m)) <= float(self.cfg.final_lock_dist_tol_m) * 2.0
        return bool(obs.edge_found)

    def _final_lock_ready(self, obs: Optional[TableEdgeObs]) -> bool:
        return bool(self._final_lock_status(obs, stable_count=self.ctx.table_lock_frames)["lock_ready"])

    def _final_lock_status(self, obs: Optional[TableEdgeObs], stable_count: int = 0) -> Dict[str, object]:
        if obs is None:
            reason = "vision_stale" if self.ctx.last_table_obs is not None else "table_lost"
            return {
                "lock_ready": False,
                "reason": reason,
                "yaw_locked": False,
                "dist_locked": False,
                "lat_locked": False,
                "stable_count": int(stable_count),
            }
        if not bool(getattr(obs, "table_found", False)):
            return {
                "lock_ready": False,
                "reason": "table_lost",
                "yaw_locked": False,
                "dist_locked": False,
                "lat_locked": False,
                "stable_count": int(stable_count),
            }
        if not bool(obs.edge_found):
            return {
                "lock_ready": False,
                "reason": "no_edge",
                "yaw_locked": False,
                "dist_locked": False,
                "lat_locked": False,
                "stable_count": int(stable_count),
            }
        if float(obs.confidence or 0.0) < float(getattr(self.controller.docking.cfg, "min_confidence", 0.0)):
            return {
                "lock_ready": False,
                "reason": "low_confidence",
                "yaw_locked": False,
                "dist_locked": False,
                "lat_locked": False,
                "stable_count": int(stable_count),
            }
        if obs.depth_valid is False:
            return {
                "lock_ready": False,
                "reason": "vision_stale",
                "yaw_locked": False,
                "dist_locked": False,
                "lat_locked": False,
                "stable_count": int(stable_count),
            }
        yaw_ok = obs.yaw_err_rad is not None and abs(float(obs.yaw_err_rad)) <= float(self.cfg.final_lock_yaw_tol_rad)
        dist_ok = obs.dist_err_m is not None and abs(float(obs.dist_err_m)) <= float(self.cfg.final_lock_dist_tol_m)
        lat_ok = obs.lateral_err_m is None or abs(float(obs.lateral_err_m)) <= float(self.cfg.final_lock_lateral_tol_m)
        reason = "stable_count_not_enough"
        if not yaw_ok:
            reason = "yaw_not_aligned"
        elif obs.dist_err_m is None:
            reason = "vision_stale"
        elif not dist_ok:
            reason = "distance_too_far" if float(obs.dist_err_m) > 0 else "distance_too_close"
        elif not lat_ok:
            reason = "yaw_not_aligned"
        return {
            "lock_ready": bool(yaw_ok and dist_ok and lat_ok),
            "reason": reason,
            "yaw_locked": bool(yaw_ok),
            "dist_locked": bool(dist_ok),
            "lat_locked": bool(lat_ok),
            "stable_count": int(stable_count),
        }

    def _log_final_lock_summary(
        self,
        obs: Optional[TableEdgeObs],
        *,
        lock_ready: bool,
        reason: str,
        stable_count: int,
    ) -> None:
        status = self._final_lock_status(obs, stable_count=stable_count)
        measured_distance = None
        target_distance = None
        if obs is not None:
            target_distance = obs.target_dist_m
            if obs.dist_err_m is not None and target_distance is not None:
                measured_distance = float(target_distance) + float(obs.dist_err_m)
        lines = [
            "FINAL_LOCK summary:",
            f"table_found={bool(obs.table_found) if obs is not None else False}",
            f"conf={float(obs.confidence or 0.0):.3f}" if obs is not None else "conf=0.000",
            f"yaw_err={obs.yaw_err_rad}" if obs is not None else "yaw_err=None",
            f"measured_distance={measured_distance}",
            f"target_distance={target_distance}",
            f"dist_err={obs.dist_err_m if obs is not None else None}",
            f"yaw_locked={bool(status['yaw_locked'])}",
            f"dist_locked={bool(status['dist_locked'])}",
            f"stable_count={int(stable_count)}",
            f"lock_ready={bool(lock_ready)}",
            f"reason={reason or status['reason']}",
        ]
        self._log("info", "\n".join(lines))

    def _target_matches_active(self, obs: TargetObs) -> bool:
        if not self.ctx.active_target or not obs.target:
            return True
        return str(obs.target).strip() == str(self.ctx.active_target).strip()

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
