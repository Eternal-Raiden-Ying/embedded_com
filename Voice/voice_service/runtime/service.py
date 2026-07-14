#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import queue
import os
import signal
import subprocess
import threading
import time
from typing import Optional, Any

from ..config.schema import VoiceServiceConfig
from ..ipc import JsonlAckInbox, InboundPollerThread, build_msgpack_client_sender, build_msgpack_inbound_server
from .asr_engine import AudioCommandPipeline
from .commands import CommandInterpreter
from .common import (
    configure_artifact_logging,
    configure_logging,
    current_run_dir,
    jlog,
    write_config_snapshot,
    write_ipc_event,
    write_named_jsonl,
    write_stop_trace,
    write_timeline,
)
from .state import RuntimeState
from .playback import PhonePlaybackGuard
from .tts_engine import PiperTTS, ThreadSafeTTS
from .workers import (
    ASRDecisionWorker,
    AudioKWSWorker,
    TTSEventListenerFactory,
    build_task_ack_listener,
)


def list_audio_devices() -> int:
    print("===== arecord -l =====")
    subprocess.run(["arecord", "-l"], check=False)
    print("\n===== arecord -L =====")
    subprocess.run(["arecord", "-L"], check=False)
    print("\n===== aplay -l =====")
    subprocess.run(["aplay", "-l"], check=False)
    print("\n===== aplay -L =====")
    subprocess.run(["aplay", "-L"], check=False)
    return 0


class DebugInputOnlySender:
    """Log-only sender: debug_input_only never creates robot control IPC."""
    def send(self, payload: Any) -> bool:
        jlog({"level": "info", "src": "debug_input", "msg": "TaskCmd suppressed", "payload": payload})
        return False
    def close(self) -> None:
        return None

    def snapshot(self) -> dict:
        return {"name": "debug_input_only", "link_state": "DISABLED"}

def build_task_sender(cfg: VoiceServiceConfig) -> Any:
    return build_msgpack_client_sender(
        mode=cfg.task_transport,
        host=cfg.task_tcp_host,
        port=cfg.task_tcp_port,
        uds_path=cfg.task_uds_path,
        name="task_cmd_sender",
        logger=jlog,
        send_mode=cfg.task_send_mode,
    )


def build_mobile_feedback_sender(cfg: VoiceServiceConfig) -> Any:
    return build_msgpack_client_sender(
        mode=cfg.mobile_feedback_transport,
        host="127.0.0.1",
        port=0,
        uds_path=cfg.mobile_tts_event_uds_path,
        name="mobile_tts_event_sender",
        logger=jlog,
        send_mode="oneshot",
    )


class PlaybackListener:
    def __init__(self, server: Any, poller: InboundPollerThread):
        self.server = server
        self.poller = poller

    def start(self) -> None:
        self.server.start()
        self.poller.start()

    def close(self) -> None:
        self.poller.stop()
        self.server.close()


def build_playback_listener(cfg: VoiceServiceConfig, guard: PhonePlaybackGuard) -> Optional[PlaybackListener]:
    if cfg.playback_transport == "disabled":
        return None
    server = build_msgpack_inbound_server(
        mode=cfg.playback_transport,
        host="127.0.0.1",
        port=0,
        uds_path=cfg.playback_uds_path,
        name="tts_playback_in",
        logger=jlog,
    )
    poller = InboundPollerThread(server, guard.handle_playback, name="tts_playback_poller")
    return PlaybackListener(server, poller)


def build_tts(cfg: VoiceServiceConfig) -> Optional[ThreadSafeTTS]:
    if cfg.disable_tts:
        return None
    base_tts = PiperTTS(
        cfg.piper_model,
        cfg.tts_cache,
        cfg.tts_out_dir,
        mode=cfg.tts_mode,
        play_cmd=cfg.play_cmd,
        dry_run_text=cfg.dry_run_text
    )
    tts = ThreadSafeTTS(base_tts)
    tts.warmup_phrases(["好，已停止", "好，开始返回", "好，请再说一次目标", "好，开始找苹果", "好，开始找水杯", "通信异常，请检查状态机"])
    return tts


def run_voice_service(cfg: VoiceServiceConfig, stop_event: Optional[threading.Event] = None, utter_q: Optional[queue.Queue] = None):
    configure_logging(
        "full" if (cfg.debug or cfg.mic_read_timeout > 5.0 or cfg.log_mode == "full") else "concise",
        quiet_mic_info=not cfg.show_mic_info,
    )
    run_dir = configure_artifact_logging(cfg.runs_dir)

    cfg.wake_key = cfg.wake_key or (cfg.wake_tflite.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].split(".")[0] if cfg.wake_tflite else "")
    cfg.stop_key = cfg.stop_key or (cfg.stop_tflite.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].split(".")[0] if cfg.stop_tflite else "")

    interpreter = CommandInterpreter.from_json(cfg.commands_json)
    if stop_event is None:
        stop_event = threading.Event()
    rt = RuntimeState()
    if utter_q is None:
        utter_q = queue.Queue(maxsize=64 if str(cfg.asr_mode).lower() == "online" else 2)

    import sys
    if cfg.debug_input_only:
        task_sender = DebugInputOnlySender()
        jlog({"level": "info", "src": "debug_input", "msg": "TaskCmd sender disabled"})
    else:
        try:
            task_sender = build_task_sender(cfg)
        except Exception as e:
            print(f"[VOICE][ERROR] component=task_sender reason={e}")
            sys.exit(1)

    ack_inbox = None
    task_ack_listener = None
    shared_tts = None
    tts_listener = None
    mobile_feedback_sender = None
    phone_playback = None
    playback_listener = None
    if cfg.debug_input_only:
        jlog({"level": "info", "src": "debug_input", "msg": "TaskAck/mobile feedback/local TTS disabled"})
    else:
        ack_inbox = JsonlAckInbox(logger=jlog)
        try:
            task_ack_listener = build_task_ack_listener(cfg, ack_inbox)
        except Exception as e:
            print(f"[VOICE][ERROR] component=task_ack_listener reason={e}")
            sys.exit(1)
        try:
            shared_tts = build_tts(cfg)
        except Exception as e:
            print(f"[VOICE][ERROR] component=tts reason={e}")
            sys.exit(1)
        try:
            tts_listener = TTSEventListenerFactory.build(cfg, shared_tts)
        except Exception as e:
            print(f"[VOICE][ERROR] component=tts_listener reason={e}")
            sys.exit(1)
        if cfg.mobile_feedback_transport != "disabled":
            try:
                mobile_feedback_sender = build_mobile_feedback_sender(cfg)
                phone_playback = PhonePlaybackGuard(cfg, rt, mobile_feedback_sender)
                playback_listener = build_playback_listener(cfg, phone_playback)
            except Exception as e:
                print(f"[VOICE][ERROR] component=phone_playback reason={e}")
                sys.exit(1)

    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/voice_numba_cache")
    try:
        os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
    except OSError:
        pass
    try:
        pipeline = AudioCommandPipeline(cfg, interpreter)
    except Exception as e:
        print(f"[VOICE][ERROR] component=asr reason={e}")
        sys.exit(1)
    if bool(getattr(cfg, "asr_warmup_enabled", False)):
        try:
            warmup_ms = pipeline.warmup(int(getattr(cfg, "asr_warmup_samples", 1600)))
            jlog({"level": "info", "src": "asr", "msg": "ASR warmup complete", "warmup_latency_ms": round(warmup_ms, 2)})
            write_timeline("ASR_WARMUP", reason="startup_silence", latency_ms=round(warmup_ms, 2))
        except Exception as e:
            jlog({"level": "error", "src": "asr", "msg": "ASR warmup failed", "error": str(e)})
            write_timeline("ASR_WARMUP_FAILED", reason="startup_silence", error=str(e))

    tts_enabled = shared_tts is not None
    tts_listener_enabled = tts_listener is not None
    task_ack_enabled = task_ack_listener is not None
    stop_hotword_enabled = bool(cfg.stop_tflite and cfg.stop_key)
    _effective_online_chunk_size = list(getattr(cfg, "asr_online_chunk_size", [5, 10, 5]))

    config_payload = {
        "ts": time.time(),
        "run_dir": run_dir,
        "wake_model": cfg.wake_tflite,
        "asr_mode": cfg.asr_mode,
        "asr_backend": pipeline.asr.name,
        "asr_model_path": cfg.asr_dir,
        "asr_quantized": bool(cfg.asr_quant),
        "asr_online_chunk_size": _effective_online_chunk_size,
        "asr_online_step_samples": int(_effective_online_chunk_size[1]) * 960,
        "asr_online_encoder_chunk_look_back": getattr(cfg, "asr_online_encoder_chunk_look_back", None),
        "asr_online_decoder_chunk_look_back": getattr(cfg, "asr_online_decoder_chunk_look_back", None),
        "stop_model": cfg.stop_tflite,
        "wake_key": cfg.wake_key,
        "stop_key": cfg.stop_key,
        "wake_th": cfg.wake_th,
        "stop_th": cfg.stop_th,
        "arecord_device": cfg.arecord_device,
        "task_transport": cfg.task_transport,
        "task_send_mode": cfg.task_send_mode,
        "task_ack_transport": cfg.task_ack_transport,
        "task_ack_timeout_s": cfg.task_ack_timeout_s,
        "tts_event_transport": cfg.tts_event_transport,
        "mobile_feedback_transport": cfg.mobile_feedback_transport,
        "playback_transport": cfg.playback_transport,
        "disable_tts": bool(cfg.disable_tts),
        "tts_enabled": tts_enabled,
        "tts_listener_enabled": tts_listener_enabled,
        "task_ack_listener_enabled": task_ack_enabled,
        "stop_hotword_enabled": stop_hotword_enabled,
        "followup_secs": cfg.followup_secs,
        "stop_followup_secs": cfg.stop_followup_secs,
        "max_followup_turns": cfg.max_followup_turns,
        "log_mode": cfg.log_mode,
        "input_mode": cfg.input_mode,
        "dry_run_text": cfg.dry_run_text,
    }
    write_config_snapshot(config_payload)
    write_named_jsonl("boot", config_payload)
    write_stop_trace("STOP_BOOT_CONFIG", stop_key=cfg.stop_key, stop_hotword_enabled=stop_hotword_enabled, stop_th=cfg.stop_th)

    jlog({
        "level": "info", "src": "boot",
        "msg": "voice service boot",
        **config_payload,
    })
    jlog({
        "level": "info", "src": "boot",
        "msg": "asr_mode={} asr_backend={} model_path={} quantized={} chunk_size={} look_back=[{},{}] step_samples={}".format(
            cfg.asr_mode, pipeline.asr.name, cfg.asr_dir, bool(cfg.asr_quant),
            _effective_online_chunk_size, getattr(cfg, "asr_online_encoder_chunk_look_back", None),
            getattr(cfg, "asr_online_decoder_chunk_look_back", None), int(_effective_online_chunk_size[1]) * 960),
    })
    write_timeline("BOOT", run_dir=run_dir, task_transport=cfg.task_transport, task_ack_transport=cfg.task_ack_transport)
    write_ipc_event("CONFIG", task_tcp_port=cfg.task_tcp_port, task_ack_tcp_port=cfg.task_ack_tcp_port)

    if cfg.disable_tts:
        jlog({"level": "info", "src": "tts", "msg": "local TTS disabled by config"})
        write_timeline("TTS_LOCAL_DISABLED")
    if cfg.tts_event_transport == "disabled":
        jlog({"level": "info", "src": "tts_event", "msg": "tts_event listener disabled by config"})
        write_timeline("TTS_EVENT_LISTENER_DISABLED")
    elif not tts_listener_enabled:
        jlog({"level": "info", "src": "tts_event", "msg": "tts_event listener skipped because local TTS is disabled"})
        write_timeline("TTS_EVENT_LISTENER_SKIPPED", reason="local_tts_disabled")

    def handle_sig(signum, frame):
        jlog({"level": "info", "src": "signal", "msg": f"got signal {signum}, stopping"})
        write_timeline("SIGNAL", signum=int(signum))
        stop_event.set()

    # Signals are not fully supported when running multithreaded non-main on Windows sometimes, so wrap
    try:
        signal.signal(signal.SIGINT, handle_sig)
        signal.signal(signal.SIGTERM, handle_sig)
    except ValueError:
        pass

    try:
        audio_thread = AudioKWSWorker(cfg, rt, stop_event, utter_q, task_sender=task_sender, ack_inbox=ack_inbox, phone_playback=phone_playback)
    except Exception as e:
        print(f"[VOICE][ERROR] component=kws reason={e}")
        sys.exit(1)
    if phone_playback is not None:
        phone_playback.set_capture_arm_callback(audio_thread._arm_command_capture)
    worker_thread = ASRDecisionWorker(cfg, rt, stop_event, utter_q, publisher=task_sender, ack_inbox=ack_inbox, tts=shared_tts, pipeline=pipeline)

    if task_ack_listener is not None:
        task_ack_listener.start()
    if tts_listener is not None:
        tts_listener.start()
    if playback_listener is not None:
        playback_listener.start()
    audio_thread.start()
    worker_thread.start()

    ready_payload = {
        "ts": time.time(),
        "event": "READY",
        "threads": ["audio_kws", "asr_decision"],
        "task_ack_listener_enabled": task_ack_enabled,
        "tts_listener_enabled": tts_listener_enabled,
        "tts_enabled": tts_enabled,
        "playback_listener_enabled": playback_listener is not None,
        "stop_hotword_enabled": stop_hotword_enabled,
    }
    write_timeline(**ready_payload)
    write_named_jsonl("heartbeat", ready_payload)
    jlog({"level": "info", "src": "boot", "msg": "voice service ready", **ready_payload})

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    finally:
        stop_event.set()
        if task_ack_listener is not None:
            task_ack_listener.close()
        if tts_listener is not None:
            tts_listener.close()
        if playback_listener is not None:
            playback_listener.close()
        audio_thread.join(timeout=1.0)
        worker_thread.join(timeout=1.0)
        task_sender.close()
        if mobile_feedback_sender is not None:
            mobile_feedback_sender.close()
