#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional

@dataclass
class VoiceConsoleConfig:
    events: List[str] = field(default_factory=lambda: ["AUDIO_READY", "WAKE_TRIGGERED", "COMMAND_CAPTURE_ARMED", "SPEECH_GATE_BLOCKED", "REC_STARTED", "REC_ENDED", "ASR_FINAL", "TASK_CMD_SENT", "TASK_ACK", "INTENT_ACCEPTED", "INTENT_REJECTED", "WARN", "ERROR"])
    fields: List[str] = field(default_factory=lambda: ["run_id", "session_id", "cmd_id", "epoch", "timestamp"])
    periodic_health_s: float = 5.0

@dataclass
class VoiceLexiconConfig:
    intents: List[str] = field(default_factory=lambda: ["FIND", "RETURN", "STOP"])
    targets: List[str] = field(default_factory=lambda: ["apple", "banana", "bottle", "key", "mouse"])
    asr_hotwords: List[str] = field(default_factory=list)

@dataclass
class VoiceInteractionConfig:
    post_command_cooldown_ms: int = 1500
    reject_commands_while_busy: bool = True
    allow_stop_while_busy: bool = True

@dataclass
class VoiceServiceConfig:
    # Repository & runs directories
    project_root: str = ""
    runs_dir: str = ""
    logs_dir: str = ""

    # Models & configuration resources
    asr_dir: str = ""
    vad_dir: str = ""
    wake_tflite: str = ""
    stop_tflite: str = ""
    piper_model: str = ""
    piper_config: str = ""
    commands_json: str = ""

    # Audio device parameters
    arecord_device: str = "plughw:CARD=UACDemoV10,DEV=0"

    # Optimization toggles
    asr_quant: bool = False
    vad_quant: bool = False

    # ASR configurations
    asr_mode: str = "online"  # online / offline
    asr_online_chunk_frames: int = 6
    asr_online_chunk_size: List[int] = field(default_factory=lambda: [0, 8, 4])
    asr_online_encoder_chunk_look_back: int = 4
    asr_online_decoder_chunk_look_back: int = 1
    asr_emit_partial: bool = True
    asr_warmup_enabled: bool = True
    asr_warmup_samples: int = 1600

    # Wake & Stop Hotword parameters
    wake_key: str = ""
    stop_key: str = ""
    wake_th: float = 0.60
    stop_th: float = 0.58
    armed_secs: float = 6.0
    followup_secs: float = 4.0
    stop_followup_secs: float = 5.5
    max_followup_turns: int = 3
    max_reject_streak: int = 2
    oww_vad_th: float = 0.0
    frontend_backend: str = "tflite"
    classifier_backend: str = "onnx"
    wake_phrases: str = "你好小车,你好 小车,小车你好"

    # Energy-based VAD / segmenting
    energy_th: float = 450.0
    start_frames: int = 2
    end_frames: int = 4
    pre_frames: int = 3
    max_frames: int = 80

    # Local TTS configurations
    disable_tts: bool = False
    tts_cache: str = ""
    tts_out_dir: str = ""
    tts_mode: str = "play"  # save / play
    play_cmd: str = "aplay -q"

    # Orchestrator TaskCmd connection (client)
    task_transport: str = "disabled"  # disabled / tcp / uds
    task_tcp_host: str = "127.0.0.1"
    task_tcp_port: int = 19101
    task_uds_path: str = "/tmp/robot_stack/task_cmd.sock"
    task_reconnect_secs: float = 1.0
    task_send_timeout: float = 1.0
    task_send_mode: str = "persistent"  # oneshot / persistent

    # Orchestrator TaskAck binding (server)
    task_ack_transport: str = "disabled"  # disabled / tcp / uds
    task_ack_tcp_host: str = "127.0.0.1"
    task_ack_tcp_port: int = 19102
    task_ack_uds_path: str = "/tmp/robot_stack/task_ack.sock"
    task_ack_timeout_s: float = 0.60

    # Orchestrator TTSEvent binding (server)
    tts_event_transport: str = "disabled"  # disabled / tcp / uds
    tts_event_host: str = "127.0.0.1"
    tts_event_port: int = 19111
    tts_event_uds_path: str = "/tmp/robot_stack/tts_event.sock"

    # Phone-TTS bridge: Voice is a client for events and server for playback state.
    mobile_feedback_transport: str = "disabled"
    mobile_tts_event_uds_path: str = "/tmp/robot_stack/mobile_tts_event.sock"
    playback_transport: str = "disabled"
    playback_uds_path: str = "/tmp/robot_stack/tts_playback.sock"
    playback_start_timeout_s: float = 1.5
    playback_finish_timeout_s: float = 6.0
    post_playback_guard_s: float = 0.35

    # Mic recording loop tunables
    mic_read_timeout: float = 2.0
    mic_startup_delay: float = 0.15
    mic_debug: bool = False
    mic_debug_every: int = 50

    # Timing, Mutes, Heartbeats & Debugging
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

    # Online stream timeouts
    asr_stream_idle_timeout_s: float = 2.5
    asr_stream_session_timeout_s: float = 8.0
    rearm_after_stream_timeout_s: float = 0.0
    show_mic_info: bool = False

    # Input modes & dry-run switches
    input_mode: str = "voice_only"  # voice_only / hybrid / mobile_only
    dry_run_text: bool = False
    debug_input_only: bool = False
    replay_manifest: str = ""
    replay_realtime: bool = True
    replay_exit_after_complete: bool = True
    replay_repeat: int = 1
    replay_fail_fast: bool = True

    # Voice helper subsections
    console: VoiceConsoleConfig = field(default_factory=VoiceConsoleConfig)
    lexicon: VoiceLexiconConfig = field(default_factory=VoiceLexiconConfig)
    interaction: VoiceInteractionConfig = field(default_factory=VoiceInteractionConfig)
