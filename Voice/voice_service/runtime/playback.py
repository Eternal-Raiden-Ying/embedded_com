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
from .common import jlog, write_timeline


class PhonePlaybackGuard:
    def __init__(self, cfg: Any, rt: RuntimeState, sender: Any,
                 clock: Optional[Callable[[], float]] = None):
        self.cfg = cfg
        self.rt = rt
        self.sender = sender
        self.clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._event_id = ""
        self._session_id = ""
        self._epoch = -1
        self._phase = "IDLE"
        self._deadline = 0.0
        self._capture_arm_callback: Optional[Callable[[str], None]] = None
        self._requested_mono = 0.0
        self._started_mono = 0.0
        self._finished_mono = 0.0

    @property
    def finished_mono(self) -> float:
        with self._lock:
            return self._finished_mono

    @property
    def enabled(self) -> bool:
        return self.sender is not None and str(self.cfg.mobile_feedback_transport) != "disabled"

    def waiting(self) -> bool:
        with self._lock:
            return self._phase in {"WAIT_START", "WAIT_FINISH", "POST_GUARD"}

    def set_capture_arm_callback(self, callback: Callable[[str], None]) -> None:
        """Install the audio-thread transition used after the playback tail guard."""
        with self._lock:
            self._capture_arm_callback = callback

    def request_wake_prompt(self) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            if self._phase in {"WAIT_START", "WAIT_FINISH", "POST_GUARD"}:
                return False
            now = self.clock()
            self._session_id = self.rt.ensure_session("wake_hotword")
            self._event_id = "tts_" + uuid.uuid4().hex
            self._epoch = self.rt.get_epoch()
            self._phase = "WAIT_START"
            self._deadline = now + float(self.cfg.playback_start_timeout_s)
            self._requested_mono = now
            self._started_mono = 0.0
            self._finished_mono = 0.0
            wall_now = time.time()
            payload = {
                "type": "tts_event",
                "schema_version": 1,
                "version": 1,
                "event_id": self._event_id,
                "source": "voice_gateway",
                "session_id": self._session_id,
                "cmd_id": "",
                "phrase_id": "WAKE_PROMPT",
                "event_key": "WAKE_PROMPT",
                "state": "WAKE",
                "target": "",
                "text": str(self.cfg.wake_prompt_text),
                "priority": "P2",
                "interrupt": False,
                "dedup_key": "wake_prompt:" + self._session_id,
                "dedupe_key": "wake_prompt:" + self._session_id,
                "epoch": self._epoch,
                "created_at": wall_now,
                "expires_at": wall_now + 10.0,
                "ttl_s": 10.0,
                "ts": wall_now,
            }
            self.rt.begin_prompt_playback(self._session_id, self._event_id, self._epoch)
        if self.sender.send(payload):
            fields = self._log_fields()
            write_timeline("WAKE_TTS_REQUESTED", **fields)
            jlog({"level": "info", "src": "tts_event", "msg": "WAKE_TTS_REQUESTED", **fields})
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
                self._started_mono = self.clock()
                self._phase = "WAIT_FINISH"
                self._deadline = self.clock() + float(self.cfg.playback_finish_timeout_s)
                self.rt.note_playback_phase("WAIT_FINISH")
                fields = self._log_fields_locked()
                write_timeline("WAKE_TTS_STARTED", **fields)
                return True
            if state == "finished" and self._phase == "WAIT_FINISH":
                self._finished_mono = self.clock()
                self._phase = "POST_GUARD"
                self._deadline = self.clock() + float(self.cfg.post_playback_guard_s)
                self.rt.note_playback_phase("POST_GUARD")
                self.rt.set_state("POST_PLAYBACK_GUARD")
                fields = self._log_fields_locked()
                wake_mono = float(self.rt.wake_trigger_mono_ns or 0) / 1_000_000_000.0
                if wake_mono > 0.0:
                    fields["kws_to_tts_finished_ms"] = round((self._finished_mono - wake_mono) * 1000.0, 2)
                write_timeline("WAKE_TTS_FINISHED", **fields)
                return True
            return False

    def invalidate(self) -> None:
        with self._lock:
            self._event_id = ""
            self._session_id = ""
            self._epoch = -1
            self._phase = "IDLE"
            self._deadline = 0.0
            self._requested_mono = 0.0
            self._started_mono = 0.0
            self._finished_mono = 0.0
        self.rt.clear_playback()

    def poll(self) -> bool:
        with self._lock:
            phase = self._phase
            deadline = self._deadline
        if phase in {"WAIT_START", "WAIT_FINISH"} and self.clock() >= deadline:
            self._enter_guard("TIMEOUT")
            return False
        if phase == "POST_GUARD" and self.clock() >= deadline:
            with self._lock:
                if self._phase != "POST_GUARD":
                    return False
                self._phase = "IDLE"
                self._deadline = 0.0
                callback = self._capture_arm_callback
            if callback is not None:
                callback("wake_prompt_complete")
            else:
                # Keep the controller independently testable and preserve the
                # legacy local/state-only behavior for callers without audio.
                self.rt.arm_command_capture(float(self.cfg.armed_secs), reason="wake_prompt_complete")
            return True
        return False

    def _log_fields(self) -> Dict[str, Any]:
        with self._lock:
            return self._log_fields_locked()

    def _log_fields_locked(self) -> Dict[str, Any]:
        return {
            "session_id": self._session_id,
            "event_id": self._event_id,
            "epoch": self._epoch,
            "phase": self._phase,
        }

    def _enter_guard(self, reason: str) -> None:
        with self._lock:
            if self._phase == "IDLE":
                return
            self._phase = "POST_GUARD"
            self._deadline = self.clock() + float(self.cfg.post_playback_guard_s)
        self.rt.note_playback_phase("POST_GUARD_" + reason)
        self.rt.set_state("POST_PLAYBACK_GUARD")
