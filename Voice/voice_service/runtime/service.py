#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import queue
import signal
import subprocess
import threading
import time
from typing import Optional

from ..config.schema import VoiceServiceConfig
from ..ipc.transport import JsonlAckInbox, JsonlClientSender, JsonlInboundListener
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
from .tts_engine import PiperTTS, ThreadSafeTTS
from .workers import ASRDecisionWorker, AudioKWSWorker, TTSEventListenerFactory


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


def build_task_sender(cfg: VoiceServiceConfig) -> JsonlClientSender:
    return JsonlClientSender(
        mode=cfg.task_transport,
        tcp_host=cfg.task_tcp_host,
        tcp_port=cfg.task_tcp_port,
        uds_path=cfg.task_uds_path,
        reconnect_interval=cfg.task_reconnect_secs,
        send_timeout=cfg.task_send_timeout,
        name="task_cmd_sender",
        logger=jlog,
        send_mode=cfg.task_send_mode,
    )


def build_task_ack_listener(cfg: VoiceServiceConfig, inbox: JsonlAckInbox) -> Optional[JsonlInboundListener]:
    if cfg.task_ack_transport == "disabled":
        return None
    return JsonlInboundListener(
        mode=cfg.task_ack_transport,
        tcp_host=cfg.task_ack_tcp_host,
        tcp_port=cfg.task_ack_tcp_port,
        uds_path=cfg.task_ack_uds_path,
        on_message=inbox.handle_message,
        name="task_ack_in",
        logger=jlog,
    )


def build_tts(cfg: VoiceServiceConfig) -> Optional[ThreadSafeTTS]:
    if cfg.disable_tts:
        return None
    base_tts = PiperTTS(cfg.piper_model, cfg.tts_cache, cfg.tts_out_dir, mode=cfg.tts_mode, play_cmd=cfg.play_cmd)
    tts = ThreadSafeTTS(base_tts)
    tts.warmup_phrases(["好，已停止", "好，开始返回", "好，请再说一次目标", "好，开始找苹果", "好，开始找水杯", "通信异常，请检查状态机"])
    return tts


def run_voice_service(cfg: VoiceServiceConfig):
    configure_logging(
        "full" if (cfg.debug or cfg.mic_debug or cfg.log_mode == "full") else "concise",
        quiet_mic_info=not cfg.show_mic_info,
    )
    run_dir = configure_artifact_logging(cfg.runs_dir)

    cfg.wake_key = cfg.wake_key or cfg.wake_tflite.rsplit("/", 1)[-1].split(".")[0]
    cfg.stop_key = cfg.stop_key or (cfg.stop_tflite.rsplit("/", 1)[-1].split(".")[0] if cfg.stop_tflite else "")

    interpreter = CommandInterpreter.from_json(cfg.commands_json)
    stop_event = threading.Event()
    rt = RuntimeState()
    utter_q: queue.Queue = queue.Queue(maxsize=64 if str(cfg.asr_mode).lower() == "online" else 2)

    task_sender = build_task_sender(cfg)
    ack_inbox = JsonlAckInbox(logger=jlog)
    task_ack_listener = build_task_ack_listener(cfg, ack_inbox)
    shared_tts = build_tts(cfg)
    tts_listener = TTSEventListenerFactory.build(cfg, shared_tts)
    pipeline = AudioCommandPipeline(cfg, interpreter)

    tts_enabled = shared_tts is not None
    tts_listener_enabled = tts_listener is not None
    task_ack_enabled = task_ack_listener is not None
    stop_hotword_enabled = bool(cfg.stop_tflite and cfg.stop_key)
    _effective_online_chunk_size = list(getattr(cfg, "asr_online_chunk_size", [5, 10, 5]))
    if len(_effective_online_chunk_size) < 3 or _effective_online_chunk_size == [0, 8, 4]:
        _effective_online_chunk_size = [5, 10, 5]

    config_payload = {
        "ts": time.time(),
        "run_dir": run_dir,
        "wake_model": cfg.wake_tflite,
        "asr_mode": cfg.asr_mode,
        "asr_online_chunk_frames": getattr(cfg, "asr_online_chunk_frames", None),
        "asr_online_chunk_size": _effective_online_chunk_size,
        "asr_online_step_samples": int(_effective_online_chunk_size[1]) * 960,
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
        "disable_tts": bool(cfg.disable_tts),
        "tts_enabled": tts_enabled,
        "tts_listener_enabled": tts_listener_enabled,
        "task_ack_listener_enabled": task_ack_enabled,
        "stop_hotword_enabled": stop_hotword_enabled,
        "followup_secs": cfg.followup_secs,
        "stop_followup_secs": cfg.stop_followup_secs,
        "max_followup_turns": cfg.max_followup_turns,
        "log_mode": cfg.log_mode,
    }
    write_config_snapshot(config_payload)
    write_named_jsonl("boot", config_payload)
    write_stop_trace("STOP_BOOT_CONFIG", stop_key=cfg.stop_key, stop_hotword_enabled=stop_hotword_enabled, stop_th=cfg.stop_th)

    jlog({
        "level": "info", "src": "boot",
        "msg": "voice service boot",
        **config_payload,
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

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    audio_thread = AudioKWSWorker(cfg, rt, stop_event, utter_q, task_sender=task_sender, ack_inbox=ack_inbox)
    worker_thread = ASRDecisionWorker(cfg, rt, stop_event, utter_q, publisher=task_sender, ack_inbox=ack_inbox, tts=shared_tts, pipeline=pipeline)

    if task_ack_listener is not None:
        task_ack_listener.start()
    if tts_listener is not None:
        tts_listener.start()
    audio_thread.start()
    worker_thread.start()

    ready_payload = {
        "ts": time.time(),
        "event": "READY",
        "threads": ["audio_kws", "asr_decision"],
        "task_ack_listener_enabled": task_ack_enabled,
        "tts_listener_enabled": tts_listener_enabled,
        "tts_enabled": tts_enabled,
        "stop_hotword_enabled": stop_hotword_enabled,
    }
    write_timeline(**ready_payload)
    write_named_jsonl("heartbeat", ready_payload)
    jlog({"level": "info", "src": "boot", "msg": "voice service ready", **ready_payload})

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    finally:
        stop_event.set()
        if task_ack_listener is not None:
            task_ack_listener.close()
        if tts_listener is not None:
            tts_listener.close()
        task_sender.close()
        audio_thread.join(timeout=3.0)
        worker_thread.join(timeout=3.0)
        write_timeline("STOP", run_dir=current_run_dir())
