#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from pathlib import Path


_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RUNS_DIR = _DEFAULT_PROJECT_ROOT / "runs"
_DEFAULT_TTS_CACHE_DIR = _DEFAULT_PROJECT_ROOT / "tts_cache"
_DEFAULT_TTS_OUT_DIR = _DEFAULT_PROJECT_ROOT / "tts_out"


@dataclass
class VoiceServiceConfig:
    # 项目路径 / runs
    project_root: str = field(default_factory=lambda: str(_DEFAULT_PROJECT_ROOT))
    runs_dir: str = field(default_factory=lambda: str(_DEFAULT_RUNS_DIR))

    # 模型与资源路径
    asr_dir: str = ""
    vad_dir: str = ""
    wake_tflite: str = ""
    stop_tflite: str = ""
    piper_model: str = ""
    commands_json: str = ""

    # 音频设备
    arecord_device: str = "plughw:1,0"

    # 量化与模型开关
    asr_quant: bool = False
    vad_quant: bool = False
    asr_mode: str = "online"  # offline / online
    asr_online_chunk_frames: int = 6
    asr_online_chunk_size: list = None
    asr_online_encoder_chunk_look_back: int = 4
    asr_online_decoder_chunk_look_back: int = 1
    asr_emit_partial: bool = True
    wake_key: str = ""
    stop_key: str = ""

    # KWS / VAD / 分段
    wake_th: float = 0.60
    stop_th: float = 0.58
    armed_secs: float = 6.0
    followup_secs: float = 4.0
    stop_followup_secs: float = 5.5
    max_followup_turns: int = 3
    max_reject_streak: int = 2
    oww_vad_th: float = 0.0
    wake_phrases: str = "你好小车,你好 小车,小车你好"
    energy_th: float = 450.0
    start_frames: int = 2
    end_frames: int = 4
    pre_frames: int = 3
    max_frames: int = 80

    # TTS
    disable_tts: bool = False
    tts_cache: str = field(default_factory=lambda: str(_DEFAULT_TTS_CACHE_DIR))
    tts_out_dir: str = field(default_factory=lambda: str(_DEFAULT_TTS_OUT_DIR))
    tts_mode: str = "play"  # save / play
    play_cmd: str = "aplay -q"

    # task_cmd 输出通道
    task_transport: str = "tcp"  # stdout / tcp / uds / tcp+stdout / uds+stdout
    task_tcp_host: str = "127.0.0.1"
    task_tcp_port: int = 9001
    task_uds_path: str = "/tmp/robot_stack/task_cmd.sock"
    task_reconnect_secs: float = 1.0
    task_send_timeout: float = 1.0
    task_send_mode: str = "oneshot"  # oneshot / persistent

    # task_ack 输入通道
    task_ack_transport: str = "tcp"  # disabled / tcp / uds
    task_ack_tcp_host: str = "127.0.0.1"
    task_ack_tcp_port: int = 9012
    task_ack_uds_path: str = "/tmp/robot_stack/task_ack.sock"
    task_ack_timeout_s: float = 0.60

    # tts_event 输入通道
    tts_event_transport: str = "tcp"  # disabled / tcp / uds
    tts_event_host: str = "127.0.0.1"
    tts_event_port: int = 9011
    tts_event_uds_path: str = "/tmp/robot_stack/tts_event.sock"

    # 麦克风读流
    mic_read_timeout: float = 2.0
    mic_startup_delay: float = 0.15
    mic_debug: bool = False
    mic_debug_every: int = 50

    # 静音、心跳、日志
    post_tts_mute_secs: float = 1.2
    post_wake_mute_secs: float = 0.35
    stop_mute_secs: float = 0.18
    stop_guard_secs: float = 0.80
    stop_repeat_block_secs: float = 1.20
    heartbeat_secs: float = 10.0
    debug: bool = False
    debug_ipc: bool = True
    debug_state: bool = True
    debug_stop: bool = True
    debug_timeline: bool = True
    log_mode: str = "concise"  # concise / full

    # STOP 后重唤醒 / 在线 ASR watchdog
    asr_stream_idle_timeout_s: float = 2.5
    asr_stream_session_timeout_s: float = 8.0
    rearm_after_stream_timeout_s: float = 0.0
    show_mic_info: bool = False
