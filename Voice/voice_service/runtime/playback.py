#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phone TTS wake-prompt playback guard.

This module has no audio or socket side effects.  The service owns socket
construction; this controller only produces/validates protocol payloads and
performs deterministic state transitions, making it safe to unit test.
"""

import threading
import time
import uuid
from typing import Any, Callable, Dict, Optional

from .state import RuntimeState


class PhonePlaybackGuard:
    WAKE_TEXT = "我在，请说出要拿的物品。"

    def __init__(self, cfg: Any, rt: RuntimeState, sender: Any,
                 clock: Optional[Callable[[], float]] = None):
        self.cfg = cfg
        self.rt = rt
        self.sender = sender
        self.clock = clock or time.time
        self._lock = threading.Lock()
        self._event_id = ""
        self._session_id = ""
        self._epoch = -1
        self._phase = "IDLE"
        self._deadline = 0.0

    @property
    def enabled(self) -> bool:
        return self.sender is not None and str(self.cfg.mobile_feedback_transport) != "disabled"

    def waiting(self) -> bool:
        with self._lock:
            return self._phase in {"WAIT_START", "WAIT_FINISH", "POST_GUARD"}

    def request_wake_prompt(self) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            if self._phase in {"WAIT_START", "WAIT_FINISH", "POST_GUARD"}:
                return False
            now = self.clock()
            self._session_id = "sess_" + uuid.uuid4().hex[:10]
            self._event_id = "tts_" + uuid.uuid4().hex
            self._epoch = self.rt.get_epoch()
            self._phase = "WAIT_START"
            self._deadline = now + float(self.cfg.playback_start_timeout_s)
            payload = {
                "type": "tts_event",
                "schema_version": 1,
                "event_id": self._event_id,
                "source": "voice_gateway",
                "session_id": self._session_id,
                "cmd_id": "",
                "phrase_id": "WAKE_PROMPT",
                "text": self.WAKE_TEXT,
                "priority": "P3",
                "interrupt": False,
                "dedup_key": "wake_prompt:" + self._session_id,
                "epoch": self._epoch,
                "ts": now,
            }
            self.rt.begin_prompt_playback(self._session_id, self._event_id, self._epoch)
        if self.sender.send(payload):
            return True
        self._enter_guard("SEND_FAILED")
        return False

    def handle_playback(self, payload: Dict[str, Any]) -> bool:
        """Accept only ordered state for the current prompt/session/epoch."""
        state = str(payload.get("state", payload.get("playback_state", ""))).lower()
        with self._lock:
            if (str(payload.get("type", "")) != "tts_playback_state" or
                    str(payload.get("event_id", "")) != self._event_id or
                    str(payload.get("session_id", "")) != self._session_id or
                    int(payload.get("epoch", -1)) != self._epoch or
                    self._epoch != self.rt.get_epoch()):
                return False
            if state == "started" and self._phase == "WAIT_START":
                self._phase = "WAIT_FINISH"
                self._deadline = self.clock() + float(self.cfg.playback_finish_timeout_s)
                self.rt.note_playback_phase("WAIT_FINISH")
                return True
            if state == "finished" and self._phase == "WAIT_FINISH":
                self._phase = "POST_GUARD"
                self._deadline = self.clock() + float(self.cfg.post_playback_guard_s)
                self.rt.note_playback_phase("POST_GUARD")
                self.rt.set_state("POST_PLAYBACK_GUARD")
                return True
            return False

    def invalidate(self) -> None:
        with self._lock:
            self._event_id = ""
            self._session_id = ""
            self._epoch = -1
            self._phase = "IDLE"
            self._deadline = 0.0
        self.rt.clear_playback()

    def poll(self) -> None:
        with self._lock:
            phase = self._phase
            deadline = self._deadline
        if phase in {"WAIT_START", "WAIT_FINISH"} and self.clock() >= deadline:
            self._enter_guard("TIMEOUT")
            return
        if phase == "POST_GUARD" and self.clock() >= deadline:
            with self._lock:
                if self._phase != "POST_GUARD":
                    return
                self._phase = "IDLE"
                self._deadline = 0.0
            self.rt.arm_prompt_session(float(self.cfg.armed_secs))

    def _enter_guard(self, reason: str) -> None:
        with self._lock:
            if self._phase == "IDLE":
                return
            self._phase = "POST_GUARD"
            self._deadline = self.clock() + float(self.cfg.post_playback_guard_s)
        self.rt.note_playback_phase("POST_GUARD_" + reason)
        self.rt.set_state("POST_PLAYBACK_GUARD")
