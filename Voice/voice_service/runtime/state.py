#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    wake_key: str
    stop_key: str
    wake_th: float
    stop_th: float
    armed_secs: float
    followup_secs: float
    stop_followup_secs: float
    max_followup_turns: int
    max_reject_streak: int
    energy_th: float
    start_frames: int
    end_frames: int
    pre_frames: int
    max_frames: int
    post_wake_mute_secs: float
    stop_mute_secs: float
    stop_guard_secs: float
    stop_repeat_block_secs: float
    heartbeat_secs: float
    debug: bool


@dataclass
class RuntimeState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    current_state: str = "IDLE"
    armed_until: float = 0.0
    mute_until: float = 0.0
    busy: bool = False
    last_rms: float = 0.0
    last_text: str = ""
    session_turns: int = 0
    reject_streak: int = 0
    last_intent: str = ""
    session_reason: str = ""
    guard_until: float = 0.0
    stop_block_until: float = 0.0
    command_epoch: int = 0
    session_id: str = ""
    ipc_state: str = "CONNECTED"
    last_cmd_id: str = ""
    last_ack_cmd_id: str = ""
    last_ack_accepted: bool = False
    last_ack_reason: str = ""
    stop_hotword_count: int = 0
    stop_text_count: int = 0
    last_stop_source: str = ""
    last_stop_score: float = 0.0
    last_stop_text: str = ""
    last_stop_cmd_id: str = ""
    last_stop_ack_ok: bool = False
    last_stop_accepted: bool = False
    last_task_route: str = ""

    def _new_session_id(self) -> str:
        return f"sess_{uuid.uuid4().hex[:10]}"

    def set_state(self, state: str):
        with self.lock:
            self.current_state = state

    def ensure_session(self, reason: str = "wake") -> str:
        with self.lock:
            if not self.session_id:
                self.session_id = self._new_session_id()
                self.session_reason = reason
            return self.session_id

    def start_session(self, secs: float, reason: str = "wake"):
        with self.lock:
            if not self.session_id or time.time() >= self.armed_until:
                self.session_id = self._new_session_id()
            self.armed_until = time.time() + max(0.0, secs)
            self.session_turns = 0
            self.reject_streak = 0
            self.session_reason = reason
            if not self.busy:
                self.current_state = "ARMED_WAIT"

    def keep_session(self, secs: float, reason: str = "followup"):
        with self.lock:
            if not self.session_id:
                self.session_id = self._new_session_id()
            self.armed_until = max(self.armed_until, time.time() + max(0.0, secs))
            self.session_reason = reason
            if not self.busy and time.time() >= self.guard_until:
                self.current_state = "ARMED_WAIT"

    def enter_guard(self, secs: float, state: str = "POST_STOP_GUARD"):
        with self.lock:
            self.guard_until = max(self.guard_until, time.time() + max(0.0, secs))
            if not self.busy:
                self.current_state = state

    def in_guard(self) -> bool:
        with self.lock:
            return time.time() < self.guard_until

    def block_stop_retrigger(self, secs: float):
        with self.lock:
            self.stop_block_until = max(self.stop_block_until, time.time() + max(0.0, secs))

    def can_trigger_stop(self) -> bool:
        with self.lock:
            return time.time() >= self.stop_block_until

    def bump_epoch(self) -> int:
        with self.lock:
            self.command_epoch += 1
            return self.command_epoch

    def get_epoch(self) -> int:
        with self.lock:
            return self.command_epoch

    def clear_session(self):
        with self.lock:
            self.armed_until = 0.0
            self.guard_until = 0.0
            self.session_turns = 0
            self.reject_streak = 0
            self.session_reason = ""
            self.session_id = ""
            if not self.busy:
                self.current_state = "IDLE"

    def disarm(self):
        self.clear_session()

    def reset_after_stop(self, guard_secs: float = 0.0, mute_secs: float = 0.0, block_secs: float = 0.0,
                         state: str = "POST_STOP_GUARD"):
        with self.lock:
            now = time.time()
            self.armed_until = 0.0
            self.session_turns = 0
            self.reject_streak = 0
            self.session_reason = ""
            self.session_id = ""
            self.guard_until = max(self.guard_until, now + max(0.0, guard_secs))
            self.mute_until = max(self.mute_until, now + max(0.0, mute_secs))
            self.stop_block_until = max(self.stop_block_until, now + max(0.0, block_secs))
            if not self.busy:
                if time.time() < self.guard_until:
                    self.current_state = state
                elif self.ipc_state != "CONNECTED":
                    self.current_state = "IPC_DEGRADED"
                else:
                    self.current_state = "IDLE"

    def is_armed(self) -> bool:
        with self.lock:
            return time.time() < self.armed_until

    def set_busy(self, v: bool):
        with self.lock:
            self.busy = v
            if v:
                self.current_state = "BUSY"
            elif time.time() < self.guard_until:
                self.current_state = "POST_STOP_GUARD"
            elif time.time() < self.armed_until:
                self.current_state = "ARMED_WAIT"
            elif self.ipc_state != "CONNECTED":
                self.current_state = "IPC_DEGRADED"
            else:
                self.current_state = "IDLE"

    def set_mute(self, secs: float):
        with self.lock:
            self.mute_until = max(self.mute_until, time.time() + max(0.0, secs))

    def is_muted(self) -> bool:
        with self.lock:
            return time.time() < self.mute_until

    def set_ipc_state(self, state: str):
        with self.lock:
            self.ipc_state = state
            if state != "CONNECTED" and not self.busy and time.time() >= self.guard_until:
                self.current_state = "IPC_DEGRADED"

    def note_command(self, cmd_id: str):
        with self.lock:
            self.last_cmd_id = cmd_id

    def note_ack(self, cmd_id: str, accepted: bool, reason: str = ""):
        with self.lock:
            self.last_ack_cmd_id = cmd_id
            self.last_ack_accepted = bool(accepted)
            self.last_ack_reason = str(reason or "")

    def note_task_route(self, route: str):
        with self.lock:
            self.last_task_route = str(route or "")

    def note_stop_trigger(self, source: str, score: float = 0.0, text: str = ""):
        with self.lock:
            src = str(source or "")
            if src == "stop_hotword":
                self.stop_hotword_count += 1
            elif src == "stop_asr":
                self.stop_text_count += 1
            self.last_stop_source = src
            self.last_stop_score = float(score or 0.0)
            self.last_stop_text = str(text or "")

    def note_stop_result(self, cmd_id: str, ack_ok: bool, accepted: bool):
        with self.lock:
            self.last_stop_cmd_id = str(cmd_id or "")
            self.last_stop_ack_ok = bool(ack_ok)
            self.last_stop_accepted = bool(accepted)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "state": self.current_state,
                "armed": time.time() < self.armed_until,
                "mute": time.time() < self.mute_until,
                "busy": self.busy,
                "guard": time.time() < self.guard_until,
                "rms": round(self.last_rms, 2),
                "last_text": self.last_text,
                "session_turns": self.session_turns,
                "reject_streak": self.reject_streak,
                "last_intent": self.last_intent,
                "session_reason": self.session_reason,
                "session_id": self.session_id,
                "epoch": self.command_epoch,
                "ipc_state": self.ipc_state,
                "last_cmd_id": self.last_cmd_id,
                "last_ack_cmd_id": self.last_ack_cmd_id,
                "last_ack_accepted": self.last_ack_accepted,
                "last_ack_reason": self.last_ack_reason,
                "stop_hotword_count": self.stop_hotword_count,
                "stop_text_count": self.stop_text_count,
                "last_stop_source": self.last_stop_source,
                "last_stop_score": round(self.last_stop_score, 4),
                "last_stop_text": self.last_stop_text,
                "last_stop_cmd_id": self.last_stop_cmd_id,
                "last_stop_ack_ok": self.last_stop_ack_ok,
                "last_stop_accepted": self.last_stop_accepted,
                "last_task_route": self.last_task_route,
            }

    def set_rms(self, rms: float):
        with self.lock:
            self.last_rms = rms

    def set_last_text(self, text: str):
        with self.lock:
            self.last_text = text

    def mark_result(self, accepted: bool, intent: str = ""):
        with self.lock:
            self.last_intent = intent
            if accepted:
                self.session_turns += 1
                self.reject_streak = 0
            else:
                self.reject_streak += 1
