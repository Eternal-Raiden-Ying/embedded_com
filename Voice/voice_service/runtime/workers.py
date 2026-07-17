#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import queue
import math
import threading
import time
import wave
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..ipc import build_task_cmd, normalize_task_ack, normalize_tts_event
from ..ipc import JsonlAckInbox, InboundPollerThread, build_msgpack_client_sender, build_msgpack_inbound_server
from .asr_engine import AudioCommandPipeline
from .commands import CommandInterpreter
from .common import FRAME_MS, MIN_UTT_MS, SR, clean_asr_text, current_run_dir, jlog, kws_trigger, rms_int16, write_ipc_event, write_state_block, write_timeline
from .kws_engine import KwsBackendFatalError, create_kws_backend
from .mic_stream import RawMicStream, WavReplayAudioSource
from .state import AudioConfig, RuntimeState
from .playback import PhonePlaybackGuard
from .tts_engine import ThreadSafeTTS


def select_hotword_action(pred: Dict[str, float], wake_key: str, wake_th: float, stop_key: str, stop_th: float) -> str:
    """STOP wins deterministically when a classifier window crosses both thresholds."""
    if stop_key and kws_trigger(pred, stop_key, stop_th):
        return "STOP"
    if wake_key and kws_trigger(pred, wake_key, wake_th):
        return "WAKE"
    return ""


def dispatch_task_cmd(payload: Dict[str, Any], publisher: Any, ack_inbox: Optional[JsonlAckInbox], rt: RuntimeState,
                      ack_timeout_s: float, label: str = "TASK_CMD", suppress_dispatch: bool = False,
                      timing: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    out = build_task_cmd(payload)
    if suppress_dispatch:
        marker = "STOP" if out["intent"] == "STOP" else "TASK"
        jlog({"level": "info", "src": "debug_input", "msg": "[VOICE][{}] suppressed debug_input_only=true".format(marker), "intent": out["intent"]})
        write_timeline("{}_SUPPRESSED".format(label), intent=out["intent"], session_id=out.get("session_id"), epoch=out.get("epoch"))
        return {"sent": True, "ack": None, "ack_ok": True, "accepted": True, "cmd": out, "suppressed": True}
    cmd_id = out["cmd_id"]
    rt.note_command(cmd_id)

    # Register pending cmd_id in ack_inbox
    if ack_inbox is not None:
        ack_inbox.register_pending(cmd_id)

    write_timeline(f"{label}_SEND_ATTEMPT", cmd_id=cmd_id, intent=out.get("intent"), target=out.get("target"), session_id=out.get("session_id"), epoch=out.get("epoch"))
    jlog({"level": "info", "src": "ipc", "msg": f"{label} send", "cmd_id": cmd_id, "intent": out.get("intent"), "target": out.get("target")})

    t_send = time.monotonic()
    sent = publisher.send(out)
    if not sent:
        rt.set_ipc_state("DEGRADED")
        write_ipc_event("SEND_FAIL", cmd_id=cmd_id, intent=out.get("intent"), link_state=publisher.snapshot().get("link_state"))
        jlog({"level": "warn", "src": "ipc", "msg": f"{label} send failed", "cmd_id": cmd_id, "link_state": publisher.snapshot().get("link_state")})
        return {"sent": False, "ack": None, "ack_ok": False, "accepted": False, "cmd": out}

    write_ipc_event("SEND_OK", cmd_id=cmd_id, intent=out.get("intent"), link_state=publisher.snapshot().get("link_state"))
    timing_fields: Dict[str, Any] = {}
    sent_mono = time.monotonic()
    if timing:
        if timing.get("asr_final_mono", 0.0) > 0.0:
            timing_fields["asr_final_to_task_cmd_ms"] = round((sent_mono - timing["asr_final_mono"]) * 1000.0, 2)
        if timing.get("kws_mono", 0.0) > 0.0:
            timing_fields["kws_to_task_cmd_ms"] = round((sent_mono - timing["kws_mono"]) * 1000.0, 2)
    write_timeline("TASK_CMD_SENT", cmd_id=cmd_id, intent=out.get("intent"), target=out.get("target"), session_id=out.get("session_id"), epoch=out.get("epoch"), **timing_fields)

    ack_raw = None
    if ack_inbox is not None and ack_timeout_s > 0:
        ack_raw = ack_inbox.wait_ack(cmd_id, ack_timeout_s)

    if ack_raw is None:
        rt.set_ipc_state("ACK_TIMEOUT")
        write_ipc_event("ACK_TIMEOUT", cmd_id=cmd_id, timeout_s=ack_timeout_s)
        jlog({"level": "warn", "src": "ipc", "msg": f"{label} ack timeout", "cmd_id": cmd_id, "timeout_s": ack_timeout_s})
        write_timeline("INTENT_REJECTED", cmd_id=cmd_id, intent=out.get("intent"), target=out.get("target"), session_id=out.get("session_id"), epoch=out.get("epoch"), reason="ack_timeout")
        return {"sent": True, "ack": None, "ack_ok": False, "accepted": False, "cmd": out}

    ack = normalize_task_ack(ack_raw)
    task_ack_latency_ms = (time.monotonic() - t_send) * 1000.0
    rt.set_ipc_state("CONNECTED")
    rt.note_ack(ack["cmd_id"], ack["accepted"], ack.get("reason", ""))
    write_ipc_event("ACK_RECV", cmd_id=ack["cmd_id"], session_id=ack.get("session_id", ""), epoch=ack.get("epoch", 0), accepted=ack["accepted"], reason=ack.get("reason", ""), state=ack.get("state", ""))
    jlog({"level": "info", "src": "ipc", "msg": f"{label} ack", "cmd_id": ack["cmd_id"], "accepted": ack["accepted"], "reason": ack.get("reason", "")})

    write_timeline("TASK_ACK", cmd_id=ack["cmd_id"], session_id=ack.get("session_id", ""), epoch=ack.get("epoch", 0), accepted=ack["accepted"], reason=ack.get("reason", ""), task_ack_latency_ms=round(task_ack_latency_ms, 2))
    if ack["accepted"]:
        write_timeline("INTENT_ACCEPTED", cmd_id=ack["cmd_id"], session_id=ack.get("session_id", ""), epoch=ack.get("epoch", 0), intent=out.get("intent"), target=out.get("target"))
    else:
        write_timeline("INTENT_REJECTED", cmd_id=ack["cmd_id"], session_id=ack.get("session_id", ""), epoch=ack.get("epoch", 0), intent=out.get("intent"), target=out.get("target"), reason=ack.get("reason", ""))

    return {"sent": True, "ack": ack, "ack_ok": True, "accepted": ack["accepted"], "cmd": out}


class AudioKWSWorker(threading.Thread):
    def __init__(self, cfg: Any, rt: RuntimeState, stop_event: threading.Event, utter_q: queue.Queue,
                 task_sender: Any, ack_inbox: Optional[JsonlAckInbox],
                 phone_playback: Optional[PhonePlaybackGuard] = None,
                 pipeline: Optional[AudioCommandPipeline] = None):
        super().__init__(daemon=True, name="audio_kws")
        self.cfg_runtime = AudioConfig(
            wake_key=cfg.kws.wake_keyword,
            stop_key=cfg.kws.stop_keyword,
            armed_secs=cfg.armed_secs,
            followup_secs=cfg.followup_secs,
            stop_followup_secs=cfg.stop_followup_secs,
            max_followup_turns=cfg.max_followup_turns,
            max_reject_streak=cfg.max_reject_streak,
            energy_th=cfg.energy_th,
            start_frames=cfg.start_frames,
            end_frames=cfg.end_frames,
            pre_frames=cfg.pre_frames,
            max_frames=cfg.max_frames,
            post_wake_mute_secs=cfg.post_wake_mute_secs,
            stop_mute_secs=cfg.stop_mute_secs,
            stop_guard_secs=cfg.stop_guard_secs,
            stop_repeat_block_secs=cfg.stop_repeat_block_secs,
            heartbeat_secs=cfg.heartbeat_secs,
            debug=cfg.debug,
        )
        self.cfg_board = cfg
        self.rt = rt
        self.stop_event = stop_event
        self.utter_q = utter_q
        self.task_sender = task_sender
        self.ack_inbox = ack_inbox
        self.phone_playback = phone_playback
        self.pipeline = pipeline
        self.asr_mode = str(getattr(cfg, "asr_mode", "offline") or "offline").lower()
        self.dry_run_text = getattr(cfg, "dry_run_text", False)

        self.online_chunk_size = list(getattr(cfg, "asr_online_chunk_size", [5, 10, 5]))
        self.online_step_samples = max(1, int(self.online_chunk_size[1]) * 960)

        self.kws = create_kws_backend(cfg.kws, dry_run_text=self.dry_run_text)
        self.kws_trigger_cooldown_s = max(0.0, float(cfg.kws.trigger_cooldown_ms) / 1000.0)
        self._last_kws_trigger: Dict[str, float] = {}
        if str(getattr(cfg, "input_mode", "voice_only")) == "wav_replay":
            self.mic = WavReplayAudioSource(
                manifest_path=cfg.replay_manifest,
                realtime=bool(cfg.replay_realtime),
                repeat=int(cfg.replay_repeat),
            )
        else:
            self.mic = RawMicStream(
                device=cfg.arecord_device,
                sr=SR,
                channels=1,
                read_timeout_sec=cfg.mic_read_timeout,
                startup_delay_sec=cfg.mic_startup_delay,
                mic_debug=cfg.mic_debug,
                mic_debug_every=cfg.mic_debug_every,
                dry_run_text=self.dry_run_text,
            )
        self._last_mic_restart_count = int(self.mic.stats().get("restarts", 0))
        self._last_busy = False
        self.prebuf = deque(maxlen=self.cfg_runtime.pre_frames)
        self.state = "WAIT_WAKE"
        self.speech_up = 0
        self.speech_down = 0
        self.captured: List[np.ndarray] = []
        self.last_heartbeat = 0.0
        self.asr_sample_buf = np.zeros((0,), dtype=np.int16)
        self.asr_chunk_seq = 0
        self.online_vad_session = None
        self.online_session_started = False
        self.online_speech_started = False
        self.online_final_emitted = False
        self.online_session_started_at = 0.0
        self.online_speech_started_at = 0.0
        self.online_speech_samples = 0
        self.online_speech_content_samples = 0
        self.vad_end_mono = 0.0
        pre_roll_ms = int(getattr(cfg, "command_pre_roll_ms", 480))
        self.online_prebuf = deque(maxlen=max(1, int(round(pre_roll_ms / FRAME_MS))))
        self.capture_history = deque(maxlen=max(1, int(math.ceil(pre_roll_ms / FRAME_MS)) + 2))
        self.capture_armed_mono = 0.0
        self._last_record_drop_reason = ""
        self._noise_floor_rms = 0.0
        self._noise_floor_before_prompt: Optional[float] = None
        self._last_speech_gate_blocked_at = 0.0
        self._reset_kws("gateway_restart")

    def _reset_kws(self, reason: str) -> None:
        self.kws.reset(reason)

    def _emit_heartbeat(self):
        now = time.time()
        
        # Check and log resource usage every 5.0 seconds (without terminal printing)
        if not hasattr(self, "_last_resource_check") or now - self._last_resource_check >= 5.0:
            self._last_resource_check = now
            rss_bytes = 0
            try:
                with open("/proc/self/status", "r") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            parts = line.split()
                            if len(parts) >= 2:
                                rss_bytes = int(parts[1]) * 1024
                            break
            except Exception:
                try:
                    import psutil
                    import os
                    process = psutil.Process(os.getpid())
                    rss_bytes = process.memory_info().rss
                except Exception:
                    pass
            if rss_bytes > 0:
                import os
                from .common import write_named_jsonl
                write_named_jsonl("resource", {
                    "ts": time.time(),
                    "run_id": os.getenv("STACK_RUN_ID", ""),
                    "voice_rss": rss_bytes,
                    "timestamp": time.time()
                })

        if now - self.last_heartbeat < self.cfg_runtime.heartbeat_secs:
            return
        self.last_heartbeat = now
        snap = self.rt.snapshot()
        snap.update({
            "level": "info",
            "src": "heartbeat",
            "mic_restarts": self.mic.stats().get("restarts", 0),
        })
        jlog(snap)
        write_state_block(snap)

    def _reset_recording(self, next_state: str = "WAIT_WAKE", update_runtime: bool = True):
        self.state = next_state
        self.speech_up = 0
        self.speech_down = 0
        self.captured = []
        self.asr_sample_buf = np.zeros((0,), dtype=np.int16)
        self.asr_chunk_seq = 0
        self.prebuf.clear()
        self._reset_kws("recording_reset")
        if update_runtime:
            self.rt.set_state(next_state)
            write_state_block(self.rt.snapshot())

    def _reset_online_capture_state(self) -> None:
        """Clear command-local Online ASR/VAD state before a fresh wake turn."""
        old_vad_session = getattr(self, "online_vad_session", None)
        pipeline = getattr(self, "pipeline", None)
        if old_vad_session is not None and pipeline is not None:
            pipeline.abort_vad_stream_session(old_vad_session)
        self.online_vad_session = None
        self.online_session_started = False
        self.online_speech_started = False
        self.online_final_emitted = False
        self.online_session_started_at = 0.0
        self.online_speech_started_at = 0.0
        self.online_speech_samples = 0
        self.online_speech_content_samples = 0
        self.vad_end_mono = 0.0
        self.online_prebuf.clear()
        self.asr_sample_buf = np.zeros((0,), dtype=np.int16)
        self.asr_chunk_seq = 0

    def _start_online_session(self, reason: str) -> None:
        """Open FSMN-VAD at arm time; Paraformer START waits for VAD START."""
        if self.asr_mode != "online":
            return
        pipeline = getattr(self, "pipeline", None)
        if pipeline is None:
            raise RuntimeError("online capture requires the shared AudioCommandPipeline")
        self._reset_online_capture_state()
        self.online_vad_session = pipeline.start_vad_stream_session()
        self.online_session_started_at = time.monotonic()
        snap = self.rt.snapshot()
        fields = {
            "reason": reason,
            "session_id": snap.get("session_id", ""),
            "epoch": snap.get("epoch"),
            "pre_roll_ms": int(self.online_prebuf.maxlen * FRAME_MS),
            "step_samples": int(self.online_step_samples),
        }
        write_timeline("ONLINE_SESSION_STARTED", **fields)
        jlog({"level": "info", "src": "online_asr", "msg": "[ONLINE_SESSION_STARTED]", **fields})

    def _finish_online_session(self, rms: float, reason: str, too_long: bool = False) -> None:
        """Flush exactly one Online FINAL after VAD end or a bounded timeout."""
        if self.asr_mode != "online" or not self.online_session_started or self.online_final_emitted:
            return
        self.vad_end_mono = time.monotonic()
        self._flush_online_chunk(is_final=True, rms=rms, too_long=too_long)
        self.online_final_emitted = True
        pipeline = getattr(self, "pipeline", None)
        if pipeline is not None:
            pipeline.abort_vad_stream_session(self.online_vad_session)
        self.online_vad_session = None
        fields = {
            "reason": reason,
            "samples": int(self.online_speech_samples),
            "session_id": self.rt.snapshot().get("session_id", ""),
            "epoch": self.rt.get_epoch(),
        }
        write_timeline("VAD_SPEECH_ENDED", **fields)
        write_timeline("VAD_END", **fields)
        jlog({"level": "info", "src": "online_vad", "msg": "[VAD_SPEECH_ENDED]", **fields})

    def _advance_online_armed_capture(self, x: np.ndarray, rms: float) -> bool:
        """Feed every armed frame to FSMN-VAD; RMS never gates Online speech."""
        if self.online_vad_session is None:
            return False
        self.online_prebuf.append(x.copy())
        vad_meta = self.pipeline.stream_vad_feed(self.online_vad_session, x, is_final=False)
        if not vad_meta.get("speech_started"):
            return False
        self.online_session_started = self._emit_online_asr_event("START")
        if not self.online_session_started:
            raise RuntimeError("failed to enqueue Online ASR START")
        self.online_speech_started = True
        self.online_speech_started_at = time.monotonic()
        self.online_speech_samples = sum(len(frame) for frame in self.online_prebuf)
        self.online_speech_content_samples = 0
        self.captured = list(self.online_prebuf)
        self.state = "REC"
        self.rt.begin_recording()
        self._append_online_samples(np.concatenate(self.captured, axis=0).astype(np.int16))
        self._flush_online_chunk(is_final=False)
        fields = {
            "rms": round(float(rms), 2),
            "pre_roll_ms": int(self.online_prebuf.maxlen * FRAME_MS),
            "pre_roll_samples": int(sum(len(frame) for frame in self.online_prebuf)),
            "session_id": self.rt.snapshot().get("session_id", ""),
            "epoch": self.rt.get_epoch(),
        }
        if self.capture_armed_mono > 0.0:
            fields["capture_armed_to_vad_start_ms"] = round(
                (self.online_speech_started_at - self.capture_armed_mono) * 1000.0, 2
            )
        write_timeline("VAD_SPEECH_STARTED", **fields)
        write_timeline("VAD_START", **fields)
        write_timeline("REC_STARTED", reason="fsmn_vad_speech_start", **fields)
        jlog({"level": "info", "src": "online_vad", "msg": "[VAD_SPEECH_STARTED]", **fields})
        jlog({"level": "info", "src": "seg", "msg": "REC_STARTED", "reason": "fsmn_vad_speech_start", **fields})
        return True

    def _advance_online_recording(self, x: np.ndarray, rms: float) -> None:
        """Continue VAD and stream only VAD-bounded speech into Paraformer."""
        vad_meta = self.pipeline.stream_vad_feed(self.online_vad_session, x, is_final=False)
        self.captured.append(x.copy())
        self.online_speech_samples += int(len(x))
        self.online_speech_content_samples += int(len(x))
        self._append_online_samples(x)
        self._flush_online_chunk(is_final=False)

        max_utterance_s = min(
            float(getattr(self.cfg_board, "asr_stream_session_timeout_s", 8.0)),
            float(self.cfg_runtime.max_frames * FRAME_MS) / 1000.0,
        )
        too_long = self.online_speech_started_at > 0.0 and (time.monotonic() - self.online_speech_started_at) >= max_utterance_s
        if vad_meta.get("speech_ended") or too_long:
            if self.online_speech_content_samples < int(MIN_UTT_MS * SR / 1000):
                fields = {
                    "reason": "vad_speech_too_short",
                    "speech_samples": int(self.online_speech_content_samples),
                    "min_speech_samples": int(MIN_UTT_MS * SR / 1000),
                    "session_id": self.rt.snapshot().get("session_id", ""),
                    "epoch": self.rt.get_epoch(),
                }
                write_timeline("VAD_SPEECH_ENDED", **fields)
                jlog({"level": "info", "src": "online_vad", "msg": "[VAD_SPEECH_ENDED]", **fields})
                self._abort_online_stream("vad_speech_too_short")
                self._reset_recording("WAIT_WAKE")
                return
            end_reason = "vad_timeout_fallback" if too_long else "fsmn_vad"
            self._finish_online_session(rms, end_reason, too_long=too_long)
            record_duration_ms = len(self.captured) * FRAME_MS
            write_timeline(
                "REC_ENDED", reason=end_reason, fallback_reason="max_frames" if too_long else "",
                frames=len(self.captured), too_long=too_long, record_duration_ms=float(record_duration_ms),
            )
            self._reset_recording("WAIT_WAKE", update_runtime=False)

    @staticmethod
    def _effective_energy_threshold(static_threshold: float, noise_floor_rms: float) -> float:
        """Use a bounded noise-relative onset gate without losing speech capture.

        A phone prompt can be much louder than normal ambience.  The cap makes
        a transient unable to turn the command gate into a permanently closed
        high-energy gate; the pre-prompt floor is restored after phone playback.
        """
        static = max(1.0, float(static_threshold))
        noise = max(0.0, float(noise_floor_rms or 0.0))
        return max(static, min(noise * 1.5, static * 2.0))

    @staticmethod
    def _snr_db(rms: float, noise_floor_rms: float) -> Optional[float]:
        noise = float(noise_floor_rms or 0.0)
        if noise <= 0.0:
            return None
        return 20.0 * math.log10(max(float(rms), 1e-6) / max(noise, 1e-6))

    def _noise_floor_updates_allowed(self) -> bool:
        snap = self.rt.snapshot()
        phone_waiting = self.phone_playback is not None and self.phone_playback.waiting()
        return (
            self.state == "WAIT_WAKE"
            and not snap["armed"]
            and not snap["busy"]
            and not snap["mute"]
            and not snap["guard"]
            and not phone_waiting
        )

    def _arm_command_capture(self, reason: str) -> None:
        """Single local/phone-TTS command-capture transition.

        The playback listener only moves the PhonePlaybackGuard to IDLE; this
        method runs in the audio loop when that guard expires, so the worker and
        RuntimeState cannot disagree about whether recording is armed.
        """
        arm_mono = time.monotonic()
        seed_frames: List[np.ndarray] = []
        if self.asr_mode == "online" and reason == "wake_prompt_complete":
            finished_mono = float(getattr(self.phone_playback, "finished_mono", 0.0) or 0.0)
            before_ms = int(getattr(self.cfg_board, "pre_roll_before_tts_finished_ms", 160))
            lower_bound = finished_mono - before_ms / 1000.0 if finished_mono > 0.0 else arm_mono
            seed_frames = [frame.copy() for ts, frame in getattr(self, "capture_history", ()) if ts >= lower_bound]
            max_seed = int(getattr(self.cfg_board, "command_pre_roll_ms", 480) // FRAME_MS)
            seed_frames = seed_frames[-max(1, max_seed):]
        if self._noise_floor_before_prompt is not None:
            self._noise_floor_rms = self._noise_floor_before_prompt
        self._noise_floor_before_prompt = None
        self.rt.arm_command_capture(self.cfg_runtime.armed_secs, reason=reason)
        self.state = "ARMED_WAIT"
        self.speech_up = 0
        self.speech_down = 0
        self.captured = []
        self.asr_sample_buf = np.zeros((0,), dtype=np.int16)
        self.asr_chunk_seq = 0
        self.prebuf.clear()
        self._reset_kws("command_capture_armed")
        if self.asr_mode == "online":
            self._start_online_session(reason)
        self.capture_armed_mono = arm_mono
        static = float(self.cfg_runtime.energy_th)
        effective = self._effective_energy_threshold(static, self._noise_floor_rms)
        snap = self.rt.snapshot()
        fields = {
            "reason": reason,
            "state": self.state,
            "armed_until": snap["armed_until"],
            "static_threshold": round(static, 2),
            "noise_floor": round(float(self._noise_floor_rms), 2),
            "effective_threshold": round(effective, 2),
            "online_energy_gate": self.asr_mode != "online",
            "start_frames": int(self.cfg_runtime.start_frames),
            "session_id": snap.get("session_id", ""),
            "epoch": snap.get("epoch"),
            "pre_roll_seed_ms": len(seed_frames) * FRAME_MS,
        }
        finished_mono = float(getattr(self.phone_playback, "finished_mono", 0.0) or 0.0)
        if reason == "wake_prompt_complete" and finished_mono > 0.0:
            fields["tts_finished_to_capture_armed_ms"] = round((arm_mono - finished_mono) * 1000.0, 2)
        write_timeline("COMMAND_CAPTURE_ARMED", **fields)
        jlog({"level": "info", "src": "seg", "msg": "COMMAND_CAPTURE_ARMED", **fields})
        write_state_block(snap)
        for frame in seed_frames:
            if self.state == "ARMED_WAIT":
                self._advance_online_armed_capture(frame, rms_int16(frame))
            elif self.state == "REC":
                self._advance_online_recording(frame, rms_int16(frame))

    def _log_speech_gate_blocked(self, rms: float, effective: float, blocked_reason: str) -> None:
        now = time.monotonic()
        if now - self._last_speech_gate_blocked_at < 1.0:
            return
        self._last_speech_gate_blocked_at = now
        snr = self._snr_db(rms, self._noise_floor_rms)
        fields = {
            "rms": round(float(rms), 2),
            "effective_threshold": round(float(effective), 2),
            "snr_db": round(snr, 2) if snr is not None else None,
            "speech_up": int(self.speech_up),
            "blocked_reason": blocked_reason,
            "session_id": self.rt.snapshot().get("session_id", ""),
            "epoch": self.rt.get_epoch(),
        }
        write_timeline("SPEECH_GATE_BLOCKED", **fields)
        jlog({"level": "info", "src": "seg", "msg": "SPEECH_GATE_BLOCKED", **fields})

    def _advance_armed_capture(self, x: np.ndarray, rms: float) -> bool:
        """Advance the onset gate and return true when a recording starts."""
        static = float(self.cfg_runtime.energy_th)
        effective = self._effective_energy_threshold(static, self._noise_floor_rms)
        if rms >= effective:
            self.speech_up += 1
        else:
            if rms >= static:
                self._log_speech_gate_blocked(rms, effective, "below_effective_threshold")
            self.speech_up = 0
        if self.speech_up < self.cfg_runtime.start_frames:
            return False
        self.captured = list(self.prebuf)
        self.captured.append(x.copy())
        self.speech_down = 0
        self.state = "REC"
        self.rt.begin_recording()
        if self.asr_mode == "online":
            self.asr_sample_buf = np.zeros((0,), dtype=np.int16)
            self.asr_chunk_seq = 0
            self._append_online_samples(np.concatenate(self.captured, axis=0).astype(np.int16))
            self._emit_online_asr_event("START")
            self._flush_online_chunk(is_final=False)
        wake_to_record_ms = (time.time() - self.rt.wake_trigger_wall_ts) * 1000.0 if self.rt.wake_trigger_wall_ts > 0 else 0.0
        snr = self._snr_db(rms, self._noise_floor_rms)
        fields = {
            "reason": "speech_start",
            "rms": round(rms, 2),
            "static_threshold": round(static, 2),
            "noise_floor": round(float(self._noise_floor_rms), 2),
            "effective_threshold": round(effective, 2),
            "snr_db": round(snr, 2) if snr is not None else None,
            "speech_up": int(self.speech_up),
            "start_frames": int(self.cfg_runtime.start_frames),
            "session_id": self.rt.snapshot().get("session_id"),
            "epoch": self.rt.snapshot().get("epoch"),
            "wake_to_record_ms": round(wake_to_record_ms, 2),
        }
        jlog({"level": "info", "src": "seg", "msg": "REC_STARTED", **fields})
        write_timeline("REC_STARTED", **fields)
        return True

    @staticmethod
    def _armed_timeout_applies(state: str, rt: RuntimeState) -> bool:
        return state == "ARMED_WAIT" and not rt.is_armed()

    def _drop_pending_utterances(self) -> int:
        dropped = 0
        while True:
            try:
                self.utter_q.get_nowait()
                self.utter_q.task_done()
                dropped += 1
            except queue.Empty:
                break
            except ValueError:
                break
        return dropped

    def _push_q_item(self, item: dict):
        try:
            self.utter_q.put_nowait(item)
            return True
        except queue.Full:
            jlog({"level": "warn", "src": "queue", "msg": "utterance queue full", "kind": item.get("kind")})
            return False

    def _enqueue_utterance(self, captured: List[np.ndarray], rms: float) -> bool:
        if not captured:
            return False
        audio = np.concatenate(captured, axis=0).astype(np.int16)
        item = {
            "kind": "FINAL_UTT",
            "ts": time.time(),
            "audio": audio,
            "rms": float(rms),
            "epoch": self.rt.get_epoch(),
            "session_id": self.rt.snapshot().get("session_id", ""),
        }
        if self._push_q_item(item):
            run_dir = current_run_dir()
            if run_dir:
                utter_dir = Path(run_dir) / "utterances"
                utter_dir.mkdir(parents=True, exist_ok=True)
                utter_path = utter_dir / "utterance_{}_{}.wav".format(item["epoch"], int(item["ts"] * 1000))
                with wave.open(str(utter_path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(SR)
                    wav.writeframes(audio.tobytes())
                item["utterance_wav"] = str(utter_path)
            self.rt.set_busy(True)
            self.rt.set_state("ASR_PROCESSING")
            write_state_block(self.rt.snapshot())
            write_timeline("ASR_ENQUEUED", samples=len(audio), session_id=item["session_id"], epoch=item["epoch"])
            return True
        return False

    def _emit_online_asr_event(self, kind: str, audio: Optional[np.ndarray] = None, rms: float = 0.0, too_long: bool = False):
        item = {
            "kind": kind,
            "ts": time.time(),
            "epoch": self.rt.get_epoch(),
            "session_id": self.rt.snapshot().get("session_id"),
            "kws_mono": float(self.rt.wake_trigger_mono_ns or 0) / 1_000_000_000.0,
            "tts_finished_mono": float(getattr(self.phone_playback, "finished_mono", 0.0) or 0.0),
            "capture_armed_mono": float(getattr(self, "capture_armed_mono", 0.0)),
            "vad_start_mono": float(getattr(self, "online_speech_started_at", 0.0)),
            "vad_end_mono": float(getattr(self, "vad_end_mono", 0.0)),
        }
        if audio is not None:
            item["audio"] = audio.astype(np.int16)
            item["samples"] = int(len(item["audio"]))
        if kind == "FINAL":
            item["rms"] = float(rms)
            item["too_long"] = bool(too_long)
        if kind == "CHUNK":
            item["seq"] = self.asr_chunk_seq
            self.asr_chunk_seq += 1
        ok = self._push_q_item(item)
        if kind == "CHUNK" and ok:
            fields = {
                "seq": item["seq"], "samples": item.get("samples", 0),
                "session_id": item.get("session_id", ""), "epoch": item.get("epoch"),
            }
            write_timeline("ONLINE_CHUNK_SENT", **fields)
            jlog({"level": "info", "src": "online_asr", "msg": "[ONLINE_CHUNK_SENT]", **fields})
        if kind == "FINAL" and ok:
            self.rt.set_busy(True)
            self.rt.set_state("BUSY")
            write_state_block(self.rt.snapshot())
        return ok

    def _append_online_samples(self, audio: np.ndarray):
        if audio is None:
            return
        arr = np.asarray(audio, dtype=np.int16).reshape(-1)
        if arr.size == 0:
            return
        if self.asr_sample_buf.size == 0:
            self.asr_sample_buf = arr.copy()
        else:
            self.asr_sample_buf = np.concatenate([self.asr_sample_buf, arr])

    def _flush_online_chunk(self, is_final: bool, rms: float = 0.0, too_long: bool = False):
        if self.asr_sample_buf.size == 0:
            if is_final:
                self._emit_online_asr_event("FINAL", audio=np.zeros((0,), dtype=np.int16), rms=rms, too_long=too_long)
            return
        if is_final:
            audio = self.asr_sample_buf.astype(np.int16, copy=False)
            self.asr_sample_buf = np.zeros((0,), dtype=np.int16)
            self._emit_online_asr_event("FINAL", audio=audio, rms=rms, too_long=too_long)
            return
        while self.asr_sample_buf.size >= self.online_step_samples:
            audio = self.asr_sample_buf[:self.online_step_samples].astype(np.int16, copy=False)
            self.asr_sample_buf = self.asr_sample_buf[self.online_step_samples:]
            self._emit_online_asr_event("CHUNK", audio=audio)

    def _classify_kws_hit(self, keyword: str, muted: bool, armed: bool, busy: bool, in_guard: bool) -> str:
        now_mono = time.monotonic()
        now_wall = time.time()
        snap = self.rt.snapshot()
        current_state = str(snap.get("state", self.state))
        last = self._last_kws_trigger.get(keyword, 0.0)
        cooldown_remaining = max(0.0, self.kws_trigger_cooldown_s - (now_mono - last))
        jlog({
            "level": "info", "src": "kws", "event": "KWS_HIT", "keyword": keyword,
            "current_state": current_state, "cooldown": cooldown_remaining > 0.0,
            "cooldown_ms": int(round(cooldown_remaining * 1000.0)), "timestamp": now_wall,
        })

        reason = ""
        if keyword not in {self.cfg_runtime.wake_key, self.cfg_runtime.stop_key}:
            reason = "unknown_keyword"
        elif cooldown_remaining > 0.0:
            reason = "trigger_cooldown"
        elif keyword == self.cfg_runtime.stop_key:
            if not self.rt.can_trigger_stop():
                reason = "stop_authority_guard"
            else:
                self._last_kws_trigger[keyword] = now_mono
                return "STOP"
        elif self.phone_playback is not None and self.phone_playback.waiting():
            reason = "phone_tts_playback"
        elif muted:
            reason = "phone_tts_or_mute_guard"
        elif current_state != "WAIT_WAKE" or self.state != "WAIT_WAKE":
            reason = "not_wait_wake"
        elif armed or busy:
            reason = "voice_session_active"
        elif in_guard or bool(snap.get("cooldown")):
            reason = "interaction_cooldown"
        else:
            self._last_kws_trigger[keyword] = now_mono
            return "WAKE"

        jlog({
            "level": "info", "src": "kws", "event": "KWS_IGNORED", "keyword": keyword,
            "reason": reason, "current_state": current_state,
        })
        return ""

    def _emit_stop_hotword(self, stop_score: float):
        snap = self.rt.snapshot()
        prev_state = snap.get("state", "WAIT_WAKE")
        write_timeline("STOP_DETECTED", prev_state=prev_state, score=round(float(stop_score), 4), session_id=snap.get("session_id"), epoch=snap.get("epoch"))
        self.rt.bump_epoch()
        if self.phone_playback is not None:
            self.phone_playback.invalidate()
        dropped = self._drop_pending_utterances()
        self._abort_online_stream("stop_hotword")
        self.rt.set_busy(False)
        session_id = self.rt.ensure_session("stop_hotword")
        payload = {
            "ts": float(time.time()),
            "intent": "STOP",
            "confidence": 0.985,
            "source": "voice_gateway",
            "text": "小车停止",
            "wake_score": round(float(stop_score), 4),
            "high_priority": True,
            "state": prev_state,
            "session_id": session_id,
            "epoch": self.rt.get_epoch(),
            "wake_trigger_wall_ts": self.rt.wake_trigger_wall_ts,
            "wake_trigger_mono_ns": self.rt.wake_trigger_mono_ns,
        }
        result = dispatch_task_cmd(
            payload, self.task_sender, self.ack_inbox, self.rt,
            self.cfg_board.task_ack_timeout_s, label="STOP",
            suppress_dispatch=bool(getattr(self.cfg_board, "debug_input_only", False)),
        )
        self.rt.reset_after_stop(
            guard_secs=self.cfg_runtime.stop_guard_secs,
            mute_secs=self.cfg_runtime.stop_mute_secs,
            block_secs=self.cfg_runtime.stop_repeat_block_secs,
            state="POST_STOP_GUARD",
        )
        if self.state == "REC":
            self._abort_recording("stop_hotword", "POST_STOP_GUARD", abort_online=False)
        else:
            self._reset_recording("POST_STOP_GUARD")
        event = "STOP_ACKED" if result.get("ack_ok") else "STOP_ACK_TIMEOUT"
        write_timeline(event, cmd_id=result["cmd"]["cmd_id"], accepted=result.get("accepted", False), dropped_utts=dropped)
        jlog({
            "level": "info",
            "src": "kws",
            "msg": "STOP hotword triggered",
            "score": round(float(stop_score), 4),
            "sent": bool(result.get("sent")),
            "ack_ok": bool(result.get("ack_ok")),
            "accepted": bool(result.get("accepted")),
            "prev_state": prev_state,
            "dropped_utts": dropped,
        })

    def _abort_online_stream(self, reason: str = "abort") -> None:
        if self.asr_mode != "online":
            return
        active = getattr(self, "online_vad_session", None) is not None
        pipeline = getattr(self, "pipeline", None)
        if pipeline is not None:
            pipeline.abort_vad_stream_session(getattr(self, "online_vad_session", None))
        self.online_vad_session = None
        if active and not getattr(self, "online_final_emitted", False):
            if getattr(self, "online_session_started", False):
                self._emit_online_asr_event("ABORT")
            fields = {
                "reason": reason,
                "session_id": self.rt.snapshot().get("session_id", ""),
                "epoch": self.rt.get_epoch(),
            }
            write_timeline("ONLINE_SESSION_ABORTED", **fields)
            write_timeline("UTTERANCE_ABORTED", **fields)
            jlog({"level": "info", "src": "online_asr", "msg": "[ONLINE_SESSION_ABORTED]", **fields})
        self.online_session_started = False
        self.online_speech_started = False

    def _abort_recording(self, reason: str, next_state: str = "WAIT_WAKE", abort_online: bool = True) -> None:
        if abort_online:
            self._abort_online_stream(reason)
        jlog({"level": "info", "src": "seg", "msg": "REC abort reason={}".format(reason), "frames": len(self.captured)})
        write_timeline("REC_ABORTED", reason=reason, frames=len(self.captured), epoch=self.rt.get_epoch())
        self._reset_recording(next_state)

    def run(self):
        jlog({"level": "info", "src": "loop", "msg": "audio/kws thread started"})

        if self.dry_run_text:
            while not self.stop_event.is_set():
                time.sleep(0.5)
            self.mic.close()
            return

        try:
            while not self.stop_event.is_set():
                b = self.mic.read_frame()
                restart_count = int(self.mic.stats().get("restarts", 0))
                if restart_count != self._last_mic_restart_count:
                    self._last_mic_restart_count = restart_count
                    self._reset_kws("audio_reconnect")
                if b is None:
                    if getattr(self.mic, "completed", False) and bool(getattr(self.cfg_board, "replay_exit_after_complete", False)):
                        jlog({"level": "info", "src": "loop", "msg": "WAV replay complete", "stats": self.mic.stats()})
                        write_timeline("REPLAY_COMPLETE", **self.mic.stats())
                        self.stop_event.set()
                        break
                    continue
                x = np.frombuffer(b, dtype=np.int16)
                r = rms_int16(x)
                self.rt.set_rms(r)
                self.capture_history.append((time.monotonic(), x.copy()))
                if self._noise_floor_updates_allowed():
                    self._noise_floor_rms = r if self._noise_floor_rms <= 0 else (0.95 * self._noise_floor_rms + 0.05 * r)
                self.prebuf.append(x.copy())
                self._emit_heartbeat()

                muted = self.rt.is_muted()
                armed = self.rt.is_armed()
                busy = self.rt.snapshot()["busy"]
                in_guard = self.rt.in_guard()

                if busy and not self._last_busy:
                    self._reset_kws("new_task_started")
                self._last_busy = bool(busy)

                try:
                    audio_float32 = np.ascontiguousarray(x.astype(np.float32) / 32768.0)
                    keyword = self.kws.process(audio_float32)
                except KwsBackendFatalError as exc:
                    self.rt.set_state("KWS_ERROR")
                    jlog({"level": "error", "src": "kws", "event": "KWS_BACKEND_ERROR", "stage": "runtime", "exception": str(exc)})
                    write_timeline("KWS_BACKEND_ERROR", stage="runtime", exception=str(exc))
                    self.stop_event.set()
                    break
                hotword_action = self._classify_kws_hit(keyword, muted, armed, busy, in_guard) if keyword else ""
                if hotword_action == "STOP":
                    snap = self.rt.snapshot()
                    write_timeline("KWS_HIT", keyword=keyword, current_state=snap.get("state"), cooldown=False,
                                   session_id=snap.get("session_id", ""), epoch=snap.get("epoch"))
                    self._emit_stop_hotword(0.0)
                    continue

                if (self.state == "WAIT_WAKE" and not muted and not armed and not busy and not in_guard and
                        hotword_action == "WAKE"):
                    self.rt.wake_trigger_wall_ts = time.time()
                    self.rt.wake_trigger_mono_ns = time.monotonic_ns()
                    session_id = self.rt.ensure_session("wake_hotword")
                    write_timeline("KWS_HIT", keyword=keyword, current_state="WAIT_WAKE", cooldown=False,
                                   session_id=session_id, epoch=self.rt.get_epoch())
                    phone_prompt_sent = False
                    if self.phone_playback is not None and self.phone_playback.enabled:
                        self._noise_floor_before_prompt = self._noise_floor_rms
                        phone_prompt_sent = self.phone_playback.request_wake_prompt()
                    if not phone_prompt_sent:
                        self._arm_command_capture("wake_hotword")
                    self.rt.set_mute(self.cfg_runtime.post_wake_mute_secs)
                    self.state = "WAIT_PROMPT_PLAYBACK" if phone_prompt_sent else "ARMED_WAIT"
                    if phone_prompt_sent:
                        self.speech_up = 0
                        self.speech_down = 0
                        self.captured = []
                        self.asr_sample_buf = np.zeros((0,), dtype=np.int16)
                        self.asr_chunk_seq = 0
                    jlog({"level": "info", "src": "kws", "msg": "WAKE triggered -> phone prompt" if phone_prompt_sent else "WAKE triggered -> armed", "session_id": self.rt.snapshot().get("session_id")})
                    noise = float(self._noise_floor_rms or 0.0)
                    snr = 20.0 * math.log10(max(r, 1e-6) / max(noise, 1e-6)) if noise > 0 else None
                    write_timeline("WAKE_TRIGGERED", reason="wake_hotword", keyword=self.cfg_runtime.wake_key, session_id=self.rt.snapshot().get("session_id"), epoch=self.rt.snapshot().get("epoch"), audio_rms=round(r, 2), noise_floor_rms=round(noise, 2), snr_db=round(snr, 2) if snr is not None else None)
                    continue

                if self.phone_playback is not None:
                    armed_now = self.phone_playback.poll()
                    if self.phone_playback.waiting():
                        self.state = self.rt.snapshot().get("state", "WAIT_PROMPT_PLAYBACK")
                        continue
                    if armed_now:
                        continue

                armed = self.rt.is_armed()
                if self._armed_timeout_applies(self.state, self.rt):
                    if self.asr_mode == "online" and self.online_vad_session is not None:
                        fields = {
                            "reason": "armed_wait_deadline",
                            "session_id": self.rt.snapshot().get("session_id", ""),
                            "epoch": self.rt.get_epoch(),
                        }
                        write_timeline("ONLINE_NO_SPEECH_TIMEOUT", **fields)
                        jlog({"level": "info", "src": "online_asr", "msg": "[ONLINE_NO_SPEECH_TIMEOUT]", **fields})
                        self._abort_online_stream("no_speech_timeout")
                    write_timeline("ARM_TIMEOUT", reason="armed_wait_deadline", session_id=self.rt.snapshot().get("session_id"), epoch=self.rt.get_epoch())
                    jlog({"level": "info", "src": "seg", "msg": "ARM timeout reason=armed_wait_deadline"})
                    self.rt.disarm()
                    self._reset_recording("WAIT_WAKE")
                    continue
                if self.state != "REC" and not armed:
                    if r >= self.cfg_runtime.energy_th:
                        reason = self.rt.recording_gate_reason()
                        if reason and reason != self._last_record_drop_reason:
                            write_timeline("REC_DROPPED", reason=reason, epoch=self.rt.get_epoch())
                            self._last_record_drop_reason = reason
                    if self.rt.in_guard():
                        if self.state != "POST_STOP_GUARD":
                            self._reset_recording("POST_STOP_GUARD")
                        continue
                    self.rt.disarm()
                    if self.state != "WAIT_WAKE":
                        self._reset_recording("WAIT_WAKE")
                    continue
                self._last_record_drop_reason = ""
                if busy:
                    continue
                if self.rt.in_guard():
                    if self.state != "POST_STOP_GUARD":
                        self._reset_recording("POST_STOP_GUARD")
                    continue

                if self.state in {"WAIT_WAKE", "POST_STOP_GUARD"}:
                    self.state = "ARMED_WAIT"
                    self.rt.set_state("ARMED_WAIT")

                if self.state == "ARMED_WAIT":
                    if self.asr_mode == "online":
                        idle_timeout_s = float(getattr(self.cfg_board, "asr_stream_idle_timeout_s", self.cfg_runtime.armed_secs))
                        if self.online_session_started_at > 0.0 and time.monotonic() - self.online_session_started_at >= idle_timeout_s:
                            fields = {
                                "reason": "asr_stream_idle_timeout",
                                "timeout_s": idle_timeout_s,
                                "session_id": self.rt.snapshot().get("session_id", ""),
                                "epoch": self.rt.get_epoch(),
                            }
                            write_timeline("ONLINE_NO_SPEECH_TIMEOUT", **fields)
                            jlog({"level": "info", "src": "online_asr", "msg": "[ONLINE_NO_SPEECH_TIMEOUT]", **fields})
                            self._abort_online_stream("no_speech_timeout")
                            self.rt.disarm()
                            self._reset_recording("WAIT_WAKE")
                            continue
                        try:
                            self._advance_online_armed_capture(x, r)
                        except Exception as exc:
                            jlog({"level": "error", "src": "online_vad", "msg": "online VAD feed failed", "error": str(exc)})
                            self._abort_online_stream("vad_feed_error")
                            self.rt.disarm()
                            self._reset_recording("WAIT_WAKE")
                    else:
                        self._advance_armed_capture(x, r)
                    continue

                if self.state == "REC":
                    if self.asr_mode == "online":
                        try:
                            self._advance_online_recording(x, r)
                        except Exception as exc:
                            jlog({"level": "error", "src": "online_vad", "msg": "online VAD feed failed", "error": str(exc)})
                            self._abort_recording("vad_feed_error")
                        continue
                    self.captured.append(x.copy())
                    if r < self.cfg_runtime.energy_th:
                        self.speech_down += 1
                    else:
                        self.speech_down = 0

                    enough = len(self.captured) * FRAME_MS >= 200
                    end_now = enough and self.speech_down >= self.cfg_runtime.end_frames
                    too_long = len(self.captured) >= self.cfg_runtime.max_frames
                    if end_now or too_long:
                        queued = self._enqueue_utterance(self.captured, r)
                        if not queued:
                            self._abort_recording("utterance_queue_full")
                            continue
                        record_duration_ms = len(self.captured) * FRAME_MS
                        jlog({
                            "level": "info", "src": "seg", "msg": "REC end",
                            "frames": len(self.captured), "too_long": too_long,
                        })
                        write_timeline("REC_ENDED", reason="max_frames" if too_long else "silence_end", frames=len(self.captured), too_long=too_long, record_duration_ms=float(record_duration_ms))
                        self._reset_recording("WAIT_WAKE", update_runtime=False)
        finally:
            try:
                self.mic.close()
            except Exception:
                pass


class ASRDecisionWorker(threading.Thread):
    def __init__(self, cfg: Any, rt: RuntimeState, stop_event: threading.Event, utter_q: queue.Queue,
                 publisher: Any, ack_inbox: Optional[JsonlAckInbox], tts: Optional[ThreadSafeTTS], pipeline: AudioCommandPipeline):
        super().__init__(daemon=True, name="asr_decision")
        self.cfg = cfg
        self.rt = rt
        self.stop_event = stop_event
        self.utter_q = utter_q
        self.publisher = publisher
        self.ack_inbox = ack_inbox
        self.tts = tts
        self.pipeline = pipeline
        self.interpreter: CommandInterpreter = pipeline.interpreter
        self.dry_run_text = getattr(cfg, "dry_run_text", False)
        self.stream_session = None
        self.stream_epoch: Optional[int] = None
        self.last_partial_text = ""

    def say_text(self, text: str):
        if self.tts and text:
            out = self.tts.say(text)
            if out is not None and self.cfg.debug:
                jlog({"level": "info", "src": "tts", "saved": str(out)})

    def _compose_ack(self, intent: str, target: Optional[str]) -> str:
        if intent == "FIND":
            spoken = self.interpreter.target_display_name(target)
            return f"好，开始找{spoken}" if target and target != "unknown" else "好，请再说一次目标"
        if intent == "RETURN":
            return "好，开始返回"
        if intent == "STOP":
            return "好，已停止"
        return ""

    def emit_action(self, intent: str, target: Optional[str], conf: float, text: str, wake_score: float = 0.0,
                    timing: Optional[Dict[str, float]] = None):
        session_id = self.rt.ensure_session("command_turn")
        payload = {
            "ts": float(time.time()),
            "intent": intent,
            "confidence": conf,
            "source": "voice_gateway",
            "text": text,
            "state": self.rt.snapshot().get("state", "WAIT_WAKE"),
            "session_id": session_id,
            "epoch": self.rt.get_epoch(),
        }
        if wake_score > 0:
            payload["wake_score"] = round(float(wake_score), 4)
        if intent == "FIND":
            payload["target"] = target or "unknown"
            payload["slots"] = {"target": payload["target"]}
        return dispatch_task_cmd(
            payload, self.publisher, self.ack_inbox, self.rt, self.cfg.task_ack_timeout_s,
            label="TASK_CMD", suppress_dispatch=bool(getattr(self.cfg, "debug_input_only", False)),
            timing=timing,
        )

    def _handle_result(self, result: dict, timing: Optional[Dict[str, float]] = None):
        if timing is None:
            timing = getattr(self, "_active_timing", None)
        status = result.get("status")
        if status == "DROP_SHORT":
            jlog({"level": "info", "src": "decision", "msg": "drop too-short utterance", "samples": result.get("samples")})
            self.rt.mark_result(False, intent="DROP_SHORT")
            return {"keep_alive": True, "tts": ""}
        if status == "IGNORE_WAKE":
            jlog({"level": "info", "src": "decision", "msg": "ignore wake phrase as command", "text": result.get("text", "")})
            self.rt.mark_result(False, intent="IGNORE_WAKE")
            return {"keep_alive": True, "tts": ""}
        if status == "IGNORE_NOISE":
            jlog({"level": "info", "src": "decision", "msg": "ignore residual/noise utterance", "text": result.get("text", "")})
            self.rt.mark_result(False, intent="IGNORE_NOISE")
            return {"keep_alive": True, "tts": ""}

        text = clean_asr_text(result.get("text", ""))
        snap = self.rt.snapshot()
        asr_latency_ms = round(float(result.get("latency_ms", 0.0)), 2)
        intent_latency_ms = round(float(result.get("intent_latency_ms", 0.0)), 2)
        final_fields: Dict[str, Any] = {}
        if timing and timing.get("vad_end_mono", 0.0) > 0.0:
            final_fields["vad_end_to_asr_final_ms"] = round(
                (timing.get("asr_final_mono", time.monotonic()) - timing["vad_end_mono"]) * 1000.0, 2
            )
        write_timeline("ASR_FINAL", raw_text=str(result.get("text", "")), normalized_text=text, inference_ms=asr_latency_ms, asr_latency_ms=asr_latency_ms, intent_latency_ms=intent_latency_ms, session_id=snap.get("session_id", ""), epoch=snap.get("epoch", 0), **final_fields)
        if text:
            self.rt.set_last_text(text)
        jlog({
            "level": "info", "src": "decision", "text": text,
            "intent": result.get("intent"), "target": result.get("target"),
            "confidence": result.get("confidence"),
            "latency_ms": round(float(result.get("latency_ms", 0.0)), 2),
            "asr_confidence": result.get("asr_confidence"),
        })

        if status == "REJECT":
            jlog({"level": "info", "src": "decision", "msg": "reject / no action", "text": text})
            self.rt.mark_result(False, intent="REJECT")
            return {"keep_alive": True, "tts": ""}

        intent = result.get("intent")
        target = result.get("target")
        conf = float(result.get("confidence", 0.0))
        write_timeline("INTENT", intent=intent, target=target, confidence=conf, session_id=self.rt.snapshot().get("session_id", ""), epoch=self.rt.get_epoch())
        dispatch = self.emit_action(intent, target, conf, text=text, timing=timing)
        if dispatch.get("suppressed"):
            self.rt.mark_result(True, intent=intent or "")
            return {"keep_alive": True, "tts": "", "intent": intent, "suppressed": True}
        if not dispatch.get("sent"):
            self.rt.mark_result(False, intent="IPC_SEND_FAIL")
            return {"keep_alive": False, "tts": "通信异常，请检查状态机", "intent": intent}
        if not dispatch.get("ack_ok"):
            self.rt.mark_result(False, intent="ACK_TIMEOUT")
            return {"keep_alive": False, "tts": "通信异常，请检查状态机", "intent": intent}
        if not dispatch.get("accepted"):
            self.rt.mark_result(False, intent=f"REJECTED_{intent}")
            reason = dispatch.get("ack", {}).get("reason", "")
            return {"keep_alive": True, "tts": "", "intent": intent, "reason": reason}
        
        # Apply command cooldown guard if successfully sent and accepted
        if intent != "STOP":
            cooldown_secs = float(getattr(getattr(self.cfg, "interaction", None), "post_command_cooldown_ms", 1500)) / 1000.0
            self.rt.consume_interaction(cooldown_secs)
            # The audio worker owns the pending queue; bumping epoch makes
            # already queued tail utterances stale without touching STOP.
            self.rt.bump_epoch()

        self.rt.mark_result(True, intent=intent or "")
        return {"keep_alive": True, "tts": self._compose_ack(intent, target), "intent": intent}


    def _apply_post_turn_policy(self, handle_meta: dict):
        snap = self.rt.snapshot()
        turns = int(snap.get("session_turns", 0))
        reject_streak = int(snap.get("reject_streak", 0))
        intent = str(handle_meta.get("intent", ""))

        if reject_streak >= int(self.cfg.max_reject_streak):
            self.rt.disarm()
            return
        if turns >= int(self.cfg.max_followup_turns):
            self.rt.disarm()
            return
        if intent == "STOP":
            self.rt.reset_after_stop(
                guard_secs=self.cfg.stop_guard_secs,
                mute_secs=self.cfg.stop_mute_secs,
                block_secs=self.cfg.stop_repeat_block_secs,
                state="POST_STOP_GUARD",
            )
            return
        if intent and intent != "STOP" and self.rt.snapshot().get("interaction_consumed"):
            self.rt.disarm()
        elif handle_meta.get("keep_alive", False):
            self.rt.keep_session(self.cfg.followup_secs, reason="post_turn_followup")
        else:
            self.rt.disarm()

    def _reset_stream_session(self):
        self.pipeline.abort_stream_session(self.stream_session)
        self.stream_session = None
        self.stream_epoch = None
        self.last_partial_text = ""

    def _handle_online_event(self, item: dict):
        kind = str(item.get("kind", ""))
        item_epoch = int(item.get("epoch", -1))

        if kind == "START":
            if item_epoch != self.rt.get_epoch():
                write_timeline("UTTERANCE_DROPPED_STALE", kind=kind, item_epoch=item_epoch,
                               epoch=self.rt.get_epoch(), session_id=item.get("session_id", ""))
                return {"finalize_turn": False, "handle_meta": {"keep_alive": False, "tts": ""}}
            self._reset_stream_session()
            self.stream_session = self.pipeline.start_stream_session()
            self.stream_epoch = item_epoch
            self.last_partial_text = ""
            write_timeline("ASR_STREAM_START", session_id=item.get("session_id"), epoch=item_epoch)
            return {"finalize_turn": False, "handle_meta": {"keep_alive": False, "tts": ""}}

        if kind == "CHUNK":
            if item_epoch != self.rt.get_epoch() or self.stream_session is None or self.stream_epoch != item_epoch:
                write_timeline("UTTERANCE_DROPPED_STALE", kind=kind, item_epoch=item_epoch,
                               epoch=self.rt.get_epoch(), session_id=item.get("session_id", ""))
                return {"finalize_turn": False, "handle_meta": {"keep_alive": False, "tts": ""}}
            feed_meta = self.pipeline.stream_feed(self.stream_session, item["audio"], is_final=False)
            merged = clean_asr_text(str(feed_meta.get("merged_text", "") or ""))
            raw_fields = {
                "raw_text": str(feed_meta.get("raw_text", feed_meta.get("text", "")) or ""),
                "previous_text": str(feed_meta.get("previous_text", "") or ""),
                "merged_text": merged,
                "chunk_seq": item.get("seq"),
                "session_id": item.get("session_id", ""), "epoch": item_epoch,
            }
            write_timeline("ONLINE_RAW_CHUNK", **raw_fields)
            jlog({"level": "info", "src": "asr_partial", "msg": "[ONLINE_RAW_CHUNK]", **raw_fields})
            write_timeline("ONLINE_MERGED_PARTIAL", **raw_fields)
            jlog({"level": "info", "src": "asr_partial", "msg": "[ONLINE_MERGED_PARTIAL]", **raw_fields})
            if self.cfg.asr_emit_partial and merged and merged != self.last_partial_text and not self.interpreter.is_residual_text(merged):
                self.last_partial_text = merged
                fields = {
                    "text": merged,
                    "chunk_seq": item.get("seq"),
                    "feed_latency_ms": round(float(feed_meta.get("feed_latency_ms", 0.0)), 2),
                    "session_id": item.get("session_id", ""), "epoch": item_epoch,
                }
                write_timeline("ONLINE_PARTIAL", **fields)
                write_timeline("ASR_PARTIAL", **fields)
                jlog({"level": "info", "src": "asr_partial", "msg": "[ONLINE_PARTIAL]", **fields})
            return {"finalize_turn": False, "handle_meta": {"keep_alive": False, "tts": ""}}

        if kind == "FINAL":
            if item_epoch != self.rt.get_epoch() or self.stream_session is None or self.stream_epoch != item_epoch or self.stream_session.finalized:
                write_timeline("UTTERANCE_DROPPED_STALE", kind=kind, item_epoch=item_epoch,
                               epoch=self.rt.get_epoch(), session_id=item.get("session_id", ""))
                return {"finalize_turn": False, "handle_meta": {"keep_alive": False, "tts": ""}}
            finalize_turn = True
            feed_meta = self.pipeline.stream_feed(self.stream_session, item["audio"], is_final=True)
            raw_fields = {
                "raw_text": str(feed_meta.get("raw_text", feed_meta.get("text", "")) or ""),
                "previous_text": str(feed_meta.get("previous_text", "") or ""),
                "merged_text": str(feed_meta.get("merged_text", "") or ""),
                "session_id": item.get("session_id", ""), "epoch": item_epoch,
            }
            write_timeline("ONLINE_RAW_FINAL", **raw_fields)
            jlog({"level": "info", "src": "asr", "msg": "[ONLINE_RAW_FINAL]", **raw_fields})
            if item_epoch != self.rt.get_epoch():
                jlog({"level": "info", "src": "decision", "msg": "drop stale final result", "item_epoch": item_epoch, "current_epoch": self.rt.get_epoch()})
                self.rt.mark_result(False, intent="DROP_STALE")
                write_timeline("UTTERANCE_DROPPED_STALE", kind=kind, item_epoch=item_epoch,
                               epoch=self.rt.get_epoch(), session_id=item.get("session_id", ""))
                self._reset_stream_session()
                return {"finalize_turn": finalize_turn, "handle_meta": {"keep_alive": False, "tts": ""}}
            result = self.pipeline.finalize_stream_result(self.stream_session)
            timing = {
                "kws_mono": float(item.get("kws_mono", 0.0) or 0.0),
                "vad_end_mono": float(item.get("vad_end_mono", 0.0) or 0.0),
                "asr_final_mono": time.monotonic(),
            }
            fields = {
                "text": result.get("text", ""), "status": result.get("status", ""),
                "samples": getattr(self.stream_session, "samples", 0),
                **raw_fields,
            }
            self._active_timing = timing
            handle_meta = self._handle_result(result)
            self._active_timing = None
            self.say_text(handle_meta.get("tts", ""))
            write_timeline("ONLINE_FINAL", **fields)
            jlog({"level": "info", "src": "asr", "msg": "[ONLINE_FINAL]", **fields})
            write_state_block(self.rt.snapshot())
            self._reset_stream_session()
            return {"finalize_turn": finalize_turn, "handle_meta": handle_meta}

        if kind == "ABORT":
            self._reset_stream_session()
            return {"finalize_turn": False, "handle_meta": {"keep_alive": False, "tts": ""}}

        raise RuntimeError(f"unknown online event kind: {kind}")

    def run(self):
        jlog({"level": "info", "src": "loop", "msg": "asr/decision thread started"})
        while not self.stop_event.is_set():
            try:
                item = self.utter_q.get(timeout=0.2)
            except queue.Empty:
                continue
            handle_meta = {"keep_alive": False, "tts": ""}
            finalize_turn = False
            try:
                # Support direct TEXT command injection for Windows / dry-run-text mode
                if item.get("kind") == "TEXT":
                    text = str(item.get("text", "")).strip()
                    item_epoch = int(item.get("epoch", -1))
                    if item_epoch == -1:
                        item_epoch = self.rt.get_epoch()

                    if item_epoch != self.rt.get_epoch():
                        jlog({"level": "info", "src": "decision", "msg": "drop stale text command", "item_epoch": item_epoch, "current_epoch": self.rt.get_epoch()})
                        self.rt.mark_result(False, intent="DROP_STALE")
                        finalize_turn = True
                    else:
                        jlog({"level": "info", "src": "decision", "msg": f"processing text injection: {text}"})
                        result = self.pipeline._interpret_text(text, 1.0, 0.0, 4000)

                        # Set busy state while processing command turn
                        self.rt.set_busy(True)
                        self.rt.set_state("BUSY")
                        write_state_block(self.rt.snapshot())

                        handle_meta = self._handle_result(result)
                        self.say_text(handle_meta.get("tts", ""))
                        write_state_block(self.rt.snapshot())
                        finalize_turn = True

                elif self.pipeline.is_online():
                    outcome = self._handle_online_event(item)
                    finalize_turn = bool(outcome.get("finalize_turn", False))
                    handle_meta = outcome.get("handle_meta", handle_meta)
                else:
                    finalize_turn = True
                    item_epoch = int(item.get("epoch", -1))
                    if item_epoch != self.rt.get_epoch():
                        jlog({"level": "info", "src": "decision", "msg": "drop stale utterance", "item_epoch": item_epoch, "current_epoch": self.rt.get_epoch()})
                        self.rt.mark_result(False, intent="DROP_STALE")
                    else:
                        result = self.pipeline.process_audio(item["audio"])
                        if item_epoch != self.rt.get_epoch():
                            jlog({"level": "info", "src": "decision", "msg": "drop stale result", "item_epoch": item_epoch, "current_epoch": self.rt.get_epoch()})
                            self.rt.mark_result(False, intent="DROP_STALE")
                        else:
                            handle_meta = self._handle_result(result)
                            self.say_text(handle_meta.get("tts", ""))
                            write_state_block(self.rt.snapshot())
            except Exception as e:
                jlog({"level": "error", "src": "worker", "msg": f"process utterance failed: {e}"})
                self.rt.mark_result(False, intent="ERROR")
                finalize_turn = True if item.get("kind") in {"FINAL", "FINAL_UTT", "ABORT", "TEXT"} or not self.pipeline.is_online() else finalize_turn
                self._reset_stream_session()
            finally:
                if finalize_turn:
                    self.rt.set_busy(False)
                    self._apply_post_turn_policy(handle_meta)
                    if self.cfg.tts_mode == "play" and handle_meta.get("tts"):
                        self.rt.set_mute(self.cfg.post_tts_mute_secs)
                self.utter_q.task_done()


class TTSEventListenerWrapper:
    def __init__(self, server: Any, poller: InboundPollerThread):
        self.server = server
        self.poller = poller

    def start(self) -> None:
        self.server.start()
        self.poller.start()

    def close(self) -> None:
        self.poller.stop()
        self.server.close()


class TTSEventListenerFactory:
    @staticmethod
    def build(cfg: Any, tts: Optional[ThreadSafeTTS]) -> Optional[TTSEventListenerWrapper]:
        if tts is None or cfg.tts_event_transport == "disabled":
            return None

        def _handle(payload: Dict[str, Any]) -> None:
            try:
                evt = normalize_tts_event(payload)
                jlog({"level": "info", "src": "tts_event", "event": "tts_event_received", "text": evt["text"], "interrupt": evt.get("interrupt", False)})
                out = tts.say(evt["text"])
                if out is not None and cfg.debug:
                    jlog({"level": "info", "src": "tts_event", "saved": str(out)})
            except Exception as e:
                jlog({"level": "error", "src": "tts_event", "msg": f"TTS speak failed: {e}"})

        server = build_msgpack_inbound_server(
            mode=cfg.tts_event_transport,
            host=cfg.tts_event_host,
            port=cfg.tts_event_port,
            uds_path=cfg.tts_event_uds_path,
            name="tts_event_in",
            logger=jlog,
        )
        poller = InboundPollerThread(server, _handle, name="tts_event_poller")
        return TTSEventListenerWrapper(server, poller)


class TaskAckListenerWrapper:
    def __init__(self, server: Any, poller: InboundPollerThread):
        self.server = server
        self.poller = poller

    def start(self) -> None:
        self.server.start()
        self.poller.start()

    def close(self) -> None:
        self.poller.stop()
        self.server.close()


def build_task_ack_listener(cfg: Any, inbox: JsonlAckInbox) -> Optional[TaskAckListenerWrapper]:
    if cfg.task_ack_transport == "disabled":
        return None
    server = build_msgpack_inbound_server(
        mode=cfg.task_ack_transport,
        host=cfg.task_ack_tcp_host,
        port=cfg.task_ack_tcp_port,
        uds_path=cfg.task_ack_uds_path,
        name="task_ack_in",
        logger=jlog,
    )
    poller = InboundPollerThread(server, inbox.handle_message, name="task_ack_poller")
    return TaskAckListenerWrapper(server, poller)
