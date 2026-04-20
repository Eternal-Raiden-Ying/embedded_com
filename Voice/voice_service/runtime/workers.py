#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import queue
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

from ..ipc.protocol import build_task_cmd, normalize_task_ack, normalize_tts_event
from ..ipc.transport import JsonlAckInbox, JsonlClientSender, JsonlInboundListener
from .asr_engine import AudioCommandPipeline
from .commands import CommandInterpreter
from .common import FRAME_MS, SR, clean_asr_text, jlog, kws_trigger, rms_int16, write_ipc_event, write_state_block, write_timeline
from .kws_engine import FlexibleWakeWord
from .mic_stream import RawMicStream
from .state import AudioConfig, RuntimeState
from .tts_engine import ThreadSafeTTS


def dispatch_task_cmd(payload, publisher: JsonlClientSender, ack_inbox: Optional[JsonlAckInbox], rt: RuntimeState,
                      ack_timeout_s: float, label: str = "TASK_CMD") -> dict:
    out = build_task_cmd(payload)
    cmd_id = out["cmd_id"]
    rt.note_command(cmd_id)
    write_timeline(f"{label}_SEND_ATTEMPT", cmd_id=cmd_id, intent=out.get("intent"), target=out.get("target"), session_id=out.get("session_id"), epoch=out.get("epoch"))
    jlog({"level": "info", "src": "ipc", "msg": f"{label} send", "cmd_id": cmd_id, "intent": out.get("intent"), "target": out.get("target")})
    sent = publisher.send(out)
    if not sent:
        rt.set_ipc_state("DEGRADED")
        write_ipc_event("SEND_FAIL", cmd_id=cmd_id, intent=out.get("intent"), link_state=publisher.snapshot().get("link_state"))
        jlog({"level": "warn", "src": "ipc", "msg": f"{label} send failed", "cmd_id": cmd_id, "link_state": publisher.snapshot().get("link_state")})
        return {"sent": False, "ack": None, "ack_ok": False, "accepted": False, "cmd": out}

    write_ipc_event("SEND_OK", cmd_id=cmd_id, intent=out.get("intent"), link_state=publisher.snapshot().get("link_state"))
    ack_raw = None
    if ack_inbox is not None and ack_timeout_s > 0:
        ack_raw = ack_inbox.wait_ack(cmd_id, ack_timeout_s)
    if ack_raw is None:
        rt.set_ipc_state("ACK_TIMEOUT")
        write_ipc_event("ACK_TIMEOUT", cmd_id=cmd_id, timeout_s=ack_timeout_s)
        jlog({"level": "warn", "src": "ipc", "msg": f"{label} ack timeout", "cmd_id": cmd_id, "timeout_s": ack_timeout_s})
        return {"sent": True, "ack": None, "ack_ok": False, "accepted": False, "cmd": out}

    ack = normalize_task_ack(ack_raw)
    rt.set_ipc_state("CONNECTED")
    rt.note_ack(ack["cmd_id"], ack["accepted"], ack.get("reason", ""))
    write_ipc_event("ACK_RECV", cmd_id=ack["cmd_id"], accepted=ack["accepted"], reason=ack.get("reason", ""), state=ack.get("state", ""))
    jlog({"level": "info", "src": "ipc", "msg": f"{label} ack", "cmd_id": ack["cmd_id"], "accepted": ack["accepted"], "reason": ack.get("reason", "")})
    return {"sent": True, "ack": ack, "ack_ok": True, "accepted": ack["accepted"], "cmd": out}


class AudioKWSWorker(threading.Thread):
    def __init__(self, cfg, rt: RuntimeState, stop_event: threading.Event, utter_q: queue.Queue,
                 task_sender: JsonlClientSender, ack_inbox: Optional[JsonlAckInbox]):
        super().__init__(daemon=True)
        self.cfg_runtime = AudioConfig(
            wake_key=cfg.wake_key,
            stop_key=cfg.stop_key,
            wake_th=cfg.wake_th,
            stop_th=cfg.stop_th,
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
        self.asr_mode = str(getattr(cfg, "asr_mode", "offline") or "offline").lower()
        self.online_chunk_size = list(getattr(cfg, "asr_online_chunk_size", [5, 10, 5]))
        if len(self.online_chunk_size) < 3 or self.online_chunk_size == [0, 8, 4]:
            self.online_chunk_size = [5, 10, 5]
        self.online_step_samples = max(1, int(self.online_chunk_size[1]) * 960)
        models = [cfg.wake_tflite]
        if cfg.stop_tflite:
            models.append(cfg.stop_tflite)
        self.oww = FlexibleWakeWord(models, vad_threshold=cfg.oww_vad_th, ncpu=1)
        self.mic = RawMicStream(
            device=cfg.arecord_device,
            sr=SR,
            channels=1,
            read_timeout_sec=cfg.mic_read_timeout,
            startup_delay_sec=cfg.mic_startup_delay,
            mic_debug=cfg.mic_debug,
            mic_debug_every=cfg.mic_debug_every,
        )
        self.prebuf = deque(maxlen=self.cfg_runtime.pre_frames)
        self.state = "IDLE"
        self.speech_up = 0
        self.speech_down = 0
        self.captured = []
        self.last_heartbeat = 0.0
        self.asr_sample_buf = np.zeros((0,), dtype=np.int16)
        self.asr_chunk_seq = 0

    def _emit_heartbeat(self):
        now = time.time()
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

    def _reset_recording(self, next_state: str = "IDLE"):
        self.state = next_state
        self.speech_up = 0
        self.speech_down = 0
        self.captured = []
        self.asr_sample_buf = np.zeros((0,), dtype=np.int16)
        self.asr_chunk_seq = 0
        self.prebuf.clear()
        self.oww.reset()
        self.rt.set_state(next_state)
        write_state_block(self.rt.snapshot())

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

    def _enqueue_utterance(self, captured, rms: float):
        if not captured:
            return
        audio = np.concatenate(captured, axis=0).astype(np.int16)
        item = {
            "kind": "FINAL_UTT",
            "ts": time.time(),
            "audio": audio,
            "rms": float(rms),
            "epoch": self.rt.get_epoch(),
        }
        if self._push_q_item(item):
            self.rt.set_busy(True)
            self.rt.set_state("BUSY")
            write_state_block(self.rt.snapshot())

    def _emit_online_asr_event(self, kind: str, audio: Optional[np.ndarray] = None, rms: float = 0.0, too_long: bool = False):
        item = {
            "kind": kind,
            "ts": time.time(),
            "epoch": self.rt.get_epoch(),
            "session_id": self.rt.snapshot().get("session_id"),
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

    def _predict_subset(self, armed: bool, busy: bool):
        only = [self.cfg_runtime.wake_key]
        if self.cfg_runtime.stop_key:
            only.append(self.cfg_runtime.stop_key)
        if armed or busy:
            only = [self.cfg_runtime.stop_key] if self.cfg_runtime.stop_key else [self.cfg_runtime.wake_key]
        return [x for x in only if x]

    def _emit_stop_hotword(self, stop_score: float):
        snap = self.rt.snapshot()
        prev_state = snap.get("state", "IDLE")
        write_timeline("STOP_DETECTED", prev_state=prev_state, score=round(float(stop_score), 4), session_id=snap.get("session_id"), epoch=snap.get("epoch"))
        self.rt.bump_epoch()
        dropped = self._drop_pending_utterances()
        self.rt.set_busy(False)
        session_id = self.rt.ensure_session("stop_hotword")
        payload = {
            "ts": float(time.time()),
            "intent": "STOP",
            "confidence": 0.985,
            "source": "voice",
            "text": "小车停止",
            "wake_score": round(float(stop_score), 4),
            "high_priority": True,
            "state": prev_state,
            "session_id": session_id,
            "epoch": self.rt.get_epoch(),
        }
        result = dispatch_task_cmd(payload, self.task_sender, self.ack_inbox, self.rt, self.cfg_board.task_ack_timeout_s, label="STOP")
        self.rt.reset_after_stop(
            guard_secs=self.cfg_runtime.stop_guard_secs,
            mute_secs=self.cfg_runtime.stop_mute_secs,
            block_secs=self.cfg_runtime.stop_repeat_block_secs,
            state="POST_STOP_GUARD",
        )
        self._reset_recording("POST_STOP_GUARD")
        event = "STOP_ACKED" if result.get("ack_ok") else "STOP_ACK_TIMEOUT"
        write_timeline(event, cmd_id=result["cmd"]["cmd_id"], accepted=result.get("accepted", False), dropped_utts=dropped)
        jlog({
            "level": "info",
            "src": "oww",
            "msg": "STOP hotword triggered",
            "score": round(float(stop_score), 4),
            "sent": bool(result.get("sent")),
            "ack_ok": bool(result.get("ack_ok")),
            "accepted": bool(result.get("accepted")),
            "prev_state": prev_state,
            "dropped_utts": dropped,
        })

    def run(self):
        jlog({"level": "info", "src": "loop", "msg": "audio/kws thread started"})
        try:
            while not self.stop_event.is_set():
                b = self.mic.read_frame()
                if b is None:
                    continue
                x = np.frombuffer(b, dtype=np.int16)
                r = rms_int16(x)
                self.rt.set_rms(r)
                self.prebuf.append(x.copy())
                self._emit_heartbeat()

                muted = self.rt.is_muted()
                armed = self.rt.is_armed()
                busy = self.rt.snapshot()["busy"]
                in_guard = self.rt.in_guard()

                pred = {}
                if self.cfg_runtime.stop_key and self.rt.can_trigger_stop():
                    pred = self.oww.predict(x, only=self._predict_subset(armed=armed, busy=busy))
                    if self.cfg_runtime.debug:
                        jlog({"level": "debug", "src": "oww", "pred": pred})
                    if kws_trigger(pred, self.cfg_runtime.stop_key, self.cfg_runtime.stop_th):
                        self._emit_stop_hotword(pred.get(self.cfg_runtime.stop_key, 0.0))
                        continue
                elif not muted:
                    pred = self.oww.predict(x, only=[self.cfg_runtime.wake_key] if self.cfg_runtime.wake_key else [])

                if (not muted) and (not armed) and (not busy) and (not in_guard) and pred and kws_trigger(pred, self.cfg_runtime.wake_key, self.cfg_runtime.wake_th):
                    self.rt.start_session(self.cfg_runtime.armed_secs, reason="wake_hotword")
                    self.rt.set_mute(self.cfg_runtime.post_wake_mute_secs)
                    self.state = "ARMED_WAIT"
                    self.speech_up = 0
                    self.speech_down = 0
                    self.captured = []
                    self.asr_sample_buf = np.zeros((0,), dtype=np.int16)
                    self.asr_chunk_seq = 0
                    jlog({"level": "info", "src": "oww", "msg": "WAKE triggered -> armed", "session_id": self.rt.snapshot().get("session_id")})
                    write_timeline("WAKE_TRIGGERED", session_id=self.rt.snapshot().get("session_id"), epoch=self.rt.snapshot().get("epoch"))
                    continue

                armed = self.rt.is_armed()
                if not armed:
                    if self.rt.in_guard():
                        if self.state != "POST_STOP_GUARD":
                            self._reset_recording("POST_STOP_GUARD")
                        continue
                    self.rt.disarm()
                    if self.state != "IDLE":
                        self._reset_recording("IDLE")
                    continue
                if busy:
                    continue
                if self.rt.in_guard():
                    if self.state != "POST_STOP_GUARD":
                        self._reset_recording("POST_STOP_GUARD")
                    continue

                if self.state in {"IDLE", "POST_STOP_GUARD"}:
                    self.state = "ARMED_WAIT"
                    self.rt.set_state("ARMED_WAIT")

                if self.state == "ARMED_WAIT":
                    if r >= self.cfg_runtime.energy_th:
                        self.speech_up += 1
                    else:
                        self.speech_up = 0
                    if self.speech_up >= self.cfg_runtime.start_frames:
                        self.captured = list(self.prebuf)
                        self.captured.append(x.copy())
                        self.speech_down = 0
                        self.state = "REC"
                        self.rt.set_state("REC")
                        if self.asr_mode == "online":
                            self.asr_sample_buf = np.zeros((0,), dtype=np.int16)
                            self.asr_chunk_seq = 0
                            self._append_online_samples(np.concatenate(self.captured, axis=0).astype(np.int16))
                            self._emit_online_asr_event("START")
                            self._flush_online_chunk(is_final=False)
                        jlog({"level": "info", "src": "seg", "msg": "REC start", "rms": round(r, 2)})
                        write_timeline("REC_START", rms=round(r, 2), session_id=self.rt.snapshot().get("session_id"), epoch=self.rt.snapshot().get("epoch"))
                    continue

                if self.state == "REC":
                    self.captured.append(x.copy())
                    if self.asr_mode == "online":
                        self._append_online_samples(x)
                        self._flush_online_chunk(is_final=False)
                    if r < self.cfg_runtime.energy_th:
                        self.speech_down += 1
                    else:
                        self.speech_down = 0

                    enough = len(self.captured) * FRAME_MS >= 200
                    end_now = enough and self.speech_down >= self.cfg_runtime.end_frames
                    too_long = len(self.captured) >= self.cfg_runtime.max_frames
                    if end_now or too_long:
                        if self.asr_mode == "online":
                            self._flush_online_chunk(is_final=True, rms=r, too_long=too_long)
                        else:
                            self._enqueue_utterance(self.captured, r)
                        jlog({
                            "level": "info", "src": "seg", "msg": "REC end",
                            "frames": len(self.captured), "too_long": too_long,
                        })
                        write_timeline("REC_END", frames=len(self.captured), too_long=too_long)
                        self._reset_recording("IDLE")
        finally:
            try:
                self.mic.close()
            except Exception:
                pass


class ASRDecisionWorker(threading.Thread):
    def __init__(self, cfg, rt: RuntimeState, stop_event: threading.Event, utter_q: queue.Queue,
                 publisher: JsonlClientSender, ack_inbox: Optional[JsonlAckInbox], tts: Optional[ThreadSafeTTS], pipeline: AudioCommandPipeline):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.rt = rt
        self.stop_event = stop_event
        self.utter_q = utter_q
        self.publisher = publisher
        self.ack_inbox = ack_inbox
        self.tts = tts
        self.pipeline = pipeline
        self.interpreter: CommandInterpreter = pipeline.interpreter
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

    def emit_action(self, intent: str, target: Optional[str], conf: float, text: str, wake_score: float = 0.0):
        session_id = self.rt.ensure_session("command_turn")
        payload = {
            "ts": float(time.time()),
            "intent": intent,
            "confidence": conf,
            "source": "voice",
            "text": text,
            "state": self.rt.snapshot().get("state", "IDLE"),
            "session_id": session_id,
            "epoch": self.rt.get_epoch(),
        }
        if wake_score > 0:
            payload["wake_score"] = round(float(wake_score), 4)
        if intent == "FIND":
            payload["target"] = target or "unknown"
            payload["slots"] = {"target": payload["target"]}
        return dispatch_task_cmd(payload, self.publisher, self.ack_inbox, self.rt, self.cfg.task_ack_timeout_s, label="TASK_CMD")

    def _handle_result(self, result: dict):
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
        dispatch = self.emit_action(intent, target, conf, text=text)
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
        if handle_meta.get("keep_alive", False):
            self.rt.keep_session(self.cfg.followup_secs, reason="post_turn_followup")
        else:
            self.rt.disarm()

    def _reset_stream_session(self):
        self.stream_session = None
        self.stream_epoch = None
        self.last_partial_text = ""

    def _handle_online_event(self, item: dict):
        kind = str(item.get("kind", ""))
        item_epoch = int(item.get("epoch", -1))

        if kind == "START":
            if item_epoch != self.rt.get_epoch():
                return {"finalize_turn": False, "handle_meta": {"keep_alive": False, "tts": ""}}
            self.stream_session = self.pipeline.start_stream_session()
            self.stream_epoch = item_epoch
            self.last_partial_text = ""
            write_timeline("ASR_STREAM_START", session_id=item.get("session_id"), epoch=item_epoch)
            return {"finalize_turn": False, "handle_meta": {"keep_alive": False, "tts": ""}}

        if kind == "CHUNK":
            if self.stream_session is None or self.stream_epoch != item_epoch:
                return {"finalize_turn": False, "handle_meta": {"keep_alive": False, "tts": ""}}
            feed_meta = self.pipeline.stream_feed(self.stream_session, item["audio"], is_final=False)
            merged = clean_asr_text(str(feed_meta.get("merged_text", "") or ""))
            if self.cfg.asr_emit_partial and merged and merged != self.last_partial_text and not self.interpreter.is_residual_text(merged):
                self.last_partial_text = merged
                jlog({
                    "level": "info", "src": "asr_partial", "text": merged,
                    "chunk_seq": item.get("seq"),
                    "feed_latency_ms": round(float(feed_meta.get("feed_latency_ms", 0.0)), 2),
                })
            return {"finalize_turn": False, "handle_meta": {"keep_alive": False, "tts": ""}}

        if kind == "FINAL":
            finalize_turn = True
            if self.stream_session is None or self.stream_epoch != item_epoch:
                self.rt.mark_result(False, intent="DROP_STALE")
                return {"finalize_turn": finalize_turn, "handle_meta": {"keep_alive": False, "tts": ""}}
            self.pipeline.stream_feed(self.stream_session, item["audio"], is_final=True)
            if item_epoch != self.rt.get_epoch():
                jlog({"level": "info", "src": "decision", "msg": "drop stale final result", "item_epoch": item_epoch, "current_epoch": self.rt.get_epoch()})
                self.rt.mark_result(False, intent="DROP_STALE")
                self._reset_stream_session()
                return {"finalize_turn": finalize_turn, "handle_meta": {"keep_alive": False, "tts": ""}}
            result = self.pipeline.finalize_stream_result(self.stream_session)
            handle_meta = self._handle_result(result)
            self.say_text(handle_meta.get("tts", ""))
            write_state_block(self.rt.snapshot())
            self._reset_stream_session()
            return {"finalize_turn": finalize_turn, "handle_meta": handle_meta}

        if kind == "ABORT":
            self._reset_stream_session()
            return {"finalize_turn": True, "handle_meta": {"keep_alive": False, "tts": ""}}

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
                if self.pipeline.is_online():
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
                finalize_turn = True if item.get("kind") in {"FINAL", "FINAL_UTT", "ABORT"} or not self.pipeline.is_online() else finalize_turn
                self._reset_stream_session()
            finally:
                if finalize_turn:
                    self.rt.set_busy(False)
                    self._apply_post_turn_policy(handle_meta)
                    if self.cfg.tts_mode == "play" and handle_meta.get("tts"):
                        self.rt.set_mute(self.cfg.post_tts_mute_secs)
                self.utter_q.task_done()


class TTSEventListenerFactory:
    @staticmethod
    def build(cfg, tts: Optional[ThreadSafeTTS]) -> Optional[JsonlInboundListener]:
        if tts is None or cfg.tts_event_transport == "disabled":
            return None

        def _handle(payload):
            evt = normalize_tts_event(payload)
            out = tts.say(evt["text"])
            if out is not None and cfg.debug:
                jlog({"level": "info", "src": "tts_event", "saved": str(out), "text": evt["text"]})

        listener = JsonlInboundListener(
            mode=cfg.tts_event_transport,
            tcp_host=cfg.tts_event_host,
            tcp_port=cfg.tts_event_port,
            uds_path=cfg.tts_event_uds_path,
            on_message=_handle,
            name="tts_event_in",
            logger=jlog,
        )
        return listener
