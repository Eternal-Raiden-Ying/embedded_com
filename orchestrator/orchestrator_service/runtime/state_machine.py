#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from typing import Callable, Dict, List, Optional, Tuple

from ..config.schema import CarMotionConfig, ControlThresholds
from ..ipc.protocol import CarState, HomeTagObs, TargetObs, TaskCmd, make_home_tag_req, make_tts_event, make_vision_idle, make_vision_req
from .common import monotonic_ts
from .context import RuntimeContext, State
from .controller import MotionController, MotionDecision


class OrchestratorCore:
    def __init__(self, cfg: ControlThresholds, car_cfg: CarMotionConfig, logger: Optional[Callable] = None):
        self.cfg = cfg
        self.ctx = RuntimeContext()
        self._logger = logger
        self.controller = MotionController(cfg, car_cfg)
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
            # 只有显式语音 STOP 才让视觉进入 IDLE/HOT_STANDBY。
            self._stop_current_task("收到 STOP 命令", tts_text="已停止", interrupt_tts=True, send_vision_idle=True)
            return True, "STOP accepted"
        if self._last_stop_mono > 0 and (monotonic_ts() - self._last_stop_mono) < float(self.cfg.post_stop_ignore_s):
            self._log("info", f"忽略 STOP 后短窗口内的后续命令: {cmd.intent}")
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

    def handle_target_obs(self, obs: TargetObs):
        self.ctx.last_target_obs = obs
        try:
            if bool(obs.found):
                self.ctx.last_target_found_wall_ts = float(obs.ts)
        except Exception:
            pass

    def handle_home_obs(self, obs: HomeTagObs):
        self.ctx.last_home_obs = obs
        try:
            if bool(obs.found):
                self.ctx.last_home_found_wall_ts = float(obs.ts)
        except Exception:
            pass

    def handle_car_state(self, state: CarState):
        self.ctx.last_car_state = state
        self.ctx.last_car_state_mono = monotonic_ts()
        if state.estop and self.cfg.car_estop_to_stop:
            reason = f"底盘急停: {state.message or state.state}"
            self._stop_current_task(reason, tts_text="底盘急停，已停止", interrupt_tts=True)
        elif state.fault and self.cfg.car_fault_to_fail:
            reason = f"底盘故障: {state.message or state.state}"
            self._stop_current_task(reason, tts_text="底盘故障，请检查小车", interrupt_tts=True)
            self.ctx.last_fail_reason = reason
        elif state.timeout and self.cfg.car_timeout_to_stop and self.ctx.state != State.STOP:
            reason = f"底盘超时: {state.message or state.state}"
            self._stop_current_task(reason, tts_text="底盘通信超时，已停止", interrupt_tts=True)

    def handle_vision_req_send_result(self, sent: bool, payload: Dict, error: str = ""):
        if sent:
            self.ctx.vision_req_fail_streak = 0
            self.ctx.active_req_id = str(payload.get("req_id", "") or self.ctx.active_req_id)
            return
        self.ctx.vision_req_fail_streak += 1
        if not self.cfg.vision_req_fail_to_stop:
            return
        if self.ctx.state == State.STOP:
            return
        if self.ctx.vision_req_fail_streak < int(self.cfg.vision_req_fail_threshold):
            return
        reason = f"vision_req_out 发送失败 {self.ctx.vision_req_fail_streak} 次"
        if error:
            reason += f": {error}"
        self._stop_current_task(reason, tts_text="视觉链路异常，已停车", interrupt_tts=True)

    def drain_vision_msgs(self) -> List[Dict]:
        out = list(self.ctx.pending_vision_msgs)
        self.ctx.pending_vision_msgs.clear()
        return out

    def drain_tts_msgs(self) -> List[Dict]:
        out = list(self.ctx.pending_tts_msgs)
        self.ctx.pending_tts_msgs.clear()
        return out

    def tick(self) -> MotionDecision:
        st = self.ctx.state
        if st == State.STOP:
            return self.controller.stop_cmd("STOP")
        if st == State.AUTOEXPLORE:
            return self._tick_auto_explore()
        if st == State.AUTOSEARCH:
            return self._tick_auto_search()
        if st == State.SEARCH:
            return self._tick_search()
        if st == State.RETURN:
            return self._tick_return()
        return self.controller.stop_cmd("STOP")

    def export_state_block(self) -> Dict:
        return {
            "ts": time.time(),
            "state": self.ctx.state.value,
            "task_intent": self.ctx.task_intent,
            "active_target": self.ctx.active_target,
            "session_id": self.ctx.active_session_id,
            "epoch": self.ctx.active_epoch,
            "req_id": self.ctx.active_req_id,
            "last_enter_reason": self.ctx.last_enter_reason,
            "last_fail_reason": self.ctx.last_fail_reason,
            "vision_req_fail_streak": self.ctx.vision_req_fail_streak,
            "found_frames": self.ctx.found_frames,
            "lost_frames": self.ctx.lost_frames,
            "arrived_frames": self.ctx.arrived_frames,
            "tag_found_frames": self.ctx.tag_found_frames,
            "tag_lost_frames": self.ctx.tag_lost_frames,
            "tag_arrived_frames": self.ctx.tag_arrived_frames,
        }

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
        self._enter_state(State.AUTOEXPLORE, f"开始寻找 {target}")
        self._queue_vision_req(make_vision_req(target, session_id=cmd.session_id, epoch=cmd.epoch), force=True)
        self._queue_tts(f"开始寻找{target}")

    def _start_return_task(self, cmd: TaskCmd):
        self.ctx.clear_task_context()
        self.ctx.task_intent = "RETURN"
        self.ctx.active_session_id = cmd.session_id
        self.ctx.active_epoch = cmd.epoch
        self.ctx.task_start_wall_ts = time.time()
        self._enter_state(State.AUTOEXPLORE, "开始返回")
        self._queue_vision_req(make_home_tag_req(session_id=cmd.session_id, epoch=cmd.epoch), force=True)
        self._queue_tts("开始返回")

    def _tick_auto_explore(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        if self.ctx.task_intent == "RETURN":
            obs = self._fresh_home_obs()
            if obs is not None and obs.found:
                self.ctx.tag_found_frames += 1
                if self.ctx.tag_found_frames >= self.cfg.tag_found_frames_to_track:
                    self._enter_state(State.RETURN, "已发现回家标记")
                    self._queue_tts("已发现回家标记")
                    return self.controller.return_cmd(obs)
            else:
                self.ctx.tag_found_frames = 0
        else:
            obs = self._fresh_target_obs()
            if obs is not None and obs.found:
                self.ctx.found_frames += 1
                if self.ctx.found_frames >= self.cfg.found_frames_to_approach:
                    self._enter_state(State.SEARCH, "已发现目标")
                    self._queue_tts("已发现目标")
                    return self.controller.search_cmd(obs)
            else:
                self.ctx.found_frames = 0
        if self._task_elapsed() >= self._task_timeout_s():
            self.ctx.last_fail_reason = "搜索超时" if self.ctx.task_intent != "RETURN" else "返回搜索超时"
            self._stop_current_task(self.ctx.last_fail_reason, tts_text="搜索超时，已停止", interrupt_tts=True)
            return self.controller.stop_cmd("STOP")
        return self.controller.auto_explore_cmd()

    def _tick_auto_search(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        if self.ctx.task_intent == "RETURN":
            obs = self._fresh_home_obs()
            if obs is not None and obs.found:
                self.ctx.tag_found_frames += 1
                if self.ctx.tag_found_frames >= self.cfg.tag_found_frames_to_track:
                    self._enter_state(State.RETURN, "重新发现回家标记")
                    self._queue_tts("已重新发现回家标记")
                    return self.controller.return_cmd(obs)
            else:
                self.ctx.tag_found_frames = 0
        else:
            obs = self._fresh_target_obs()
            if obs is not None and obs.found:
                self.ctx.found_frames += 1
                if self.ctx.found_frames >= self.cfg.found_frames_to_approach:
                    self._enter_state(State.SEARCH, "重新发现目标")
                    self._queue_tts("已重新发现目标")
                    return self.controller.search_cmd(obs)
            else:
                self.ctx.found_frames = 0
        # 自动搜索阶段的 timeout 应该从进入 AUTOSEARCH 那一刻重新计时，
        # 而不是沿用整轮 FIND 的 task_start_wall_ts。
        if self._state_elapsed() >= self._task_timeout_s():
            self.ctx.last_fail_reason = "自动搜索超时"
            self._stop_current_task(self.ctx.last_fail_reason, tts_text="搜索超时，已停止", interrupt_tts=True)
            return self.controller.stop_cmd("STOP")
        return self.controller.auto_search_cmd()

    def _tick_search(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        obs = self._fresh_target_obs()
        now = time.time()
        if obs is None or not obs.found:
            self.ctx.lost_frames += 1
            lost_hold_ok = True
            if self.ctx.last_target_found_wall_ts > 0:
                lost_hold_ok = (now - self.ctx.last_target_found_wall_ts) >= float(self.cfg.search_lost_hold_s)
            dwell_ok = self._state_elapsed() >= float(self.cfg.search_min_dwell_s)
            if self.ctx.lost_frames >= self.cfg.lost_frames_to_search and lost_hold_ok and dwell_ok:
                self._enter_state(State.AUTOSEARCH, "目标丢失，自动搜索")
                self._queue_tts("目标丢失，自动搜索")
                self._queue_vision_req(self._active_req_payload(), force=True)
                return self.controller.auto_search_cmd()
            # 短时抖动/短时丢帧：保持 SEARCH，但先停住，不要立刻切到 AUTOSEARCH。
            return self.controller.stop_cmd("SEARCH_HOLD")
        self.ctx.lost_frames = 0
        self.ctx.last_target_found_wall_ts = now
        # 测试阶段：接近目标后不再自动 STOP，不再自动让视觉进入热待机。
        self.ctx.arrived_frames = 0
        return self.controller.search_cmd(obs)

    def _tick_return(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        obs = self._fresh_home_obs()
        now = time.time()
        if obs is None or not obs.found:
            self.ctx.tag_lost_frames += 1
            lost_hold_ok = True
            if self.ctx.last_home_found_wall_ts > 0:
                lost_hold_ok = (now - self.ctx.last_home_found_wall_ts) >= float(self.cfg.return_lost_hold_s)
            dwell_ok = self._state_elapsed() >= float(self.cfg.search_min_dwell_s)
            if self.ctx.tag_lost_frames >= self.cfg.tag_lost_frames_to_search and lost_hold_ok and dwell_ok:
                self._enter_state(State.AUTOSEARCH, "回家目标丢失，自动搜索")
                self._queue_tts("回家目标丢失，自动搜索")
                self._queue_vision_req(self._active_req_payload(), force=True)
                return self.controller.auto_search_cmd()
            return self.controller.stop_cmd("RETURN_HOLD")
        self.ctx.tag_lost_frames = 0
        self.ctx.last_home_found_wall_ts = now
        # 测试阶段：RETURN 接近目标后也不再自动 STOP，不再自动让视觉进入热待机。
        self.ctx.tag_arrived_frames = 0
        return self.controller.return_cmd(obs)

    def _active_req_payload(self) -> Dict:
        if self.ctx.task_intent == "RETURN":
            return make_home_tag_req(session_id=self.ctx.active_session_id, epoch=self.ctx.active_epoch)
        return make_vision_req(self.ctx.active_target or "", session_id=self.ctx.active_session_id, epoch=self.ctx.active_epoch)

    def _fresh_target_obs(self) -> Optional[TargetObs]:
        obs = self.ctx.last_target_obs
        if obs is None or time.time() - obs.ts > self.cfg.target_obs_max_age_s:
            return None
        if self.ctx.task_start_wall_ts > 0 and obs.ts < self.ctx.task_start_wall_ts:
            return None
        if self.ctx.active_target and obs.target and obs.target != self.ctx.active_target:
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

    def _task_elapsed(self) -> float:
        if self.ctx.task_start_wall_ts <= 0:
            return 0.0
        return time.time() - self.ctx.task_start_wall_ts

    def _task_timeout_s(self) -> float:
        return float(self.cfg.return_search_timeout_s if self.ctx.task_intent == "RETURN" else self.cfg.search_timeout_s)

    def _state_elapsed(self) -> float:
        return monotonic_ts() - self.ctx.state_enter_mono

    def _enter_state(self, new_state: State, reason: str):
        self._log("info", f"状态切换: {self.ctx.state.value} -> {new_state.value} ({reason})")
        self.ctx.state = new_state
        self.ctx.state_enter_mono = monotonic_ts()
        self.ctx.state_enter_wall_ts = time.time()
        self.ctx.last_enter_reason = reason
        self.ctx.clear_motion_counters()

    def _stop_current_task(self, reason: str, tts_text: Optional[str] = None, interrupt_tts: bool = False, send_vision_idle: bool = False):
        # 只有显式语音 STOP 才允许把视觉打到 IDLE/HOT_STANDBY。
        if send_vision_idle:
            self._queue_vision_req(make_vision_idle(session_id=self.ctx.active_session_id, epoch=self.ctx.active_epoch), force=True)
        self.ctx.clear_task_context()
        self._enter_state(State.STOP, reason)
        if tts_text:
            self._queue_tts(tts_text, interrupt=interrupt_tts)

    def _maybe_resend_req(self, req: Dict):
        self._queue_vision_req(req, force=False)

    def _queue_vision_req(self, payload: Dict, force: bool = False):
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
