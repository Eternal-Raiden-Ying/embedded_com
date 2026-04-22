#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
板端默认配置。

这份文件的目的就是把你比赛/上板阶段真正会改的内容收口在一个地方，
避免每次都在命令行里写很长一串参数。
"""

import os
from pathlib import Path

from .schema import VoiceServiceConfig


_DEFAULT_VOICE_ROOT = Path(__file__).resolve().parents[2]
VOICE_ROOT = Path(os.getenv("VOICE_ROOT", str(_DEFAULT_VOICE_ROOT)))
VOICE_SERVICE_ROOT = Path(os.getenv("VOICE_SERVICE_ROOT", str(VOICE_ROOT)))

VOICE_TEST_PROFILE = os.getenv("VOICE_TEST_PROFILE", "").strip().lower()
VOICE_NO_SPEAKER_DEFAULT = VOICE_TEST_PROFILE in {"nospeaker", "mic_cam_only", "silent"}
TTS_EVENT_TRANSPORT_DEFAULT = "disabled" if VOICE_NO_SPEAKER_DEFAULT else "tcp"


def _pick_model(candidates, default: str = "") -> str:
    for p in candidates:
        path = Path(p)
        if path.exists():
            return str(path)
    return default


WAKE_MODEL = os.getenv(
    "VOICE_WAKE_MODEL",
    _pick_model([
        VOICE_ROOT / "kws" / "wake_nihao_xiaoche_v2.onnx",
        VOICE_ROOT / "kws" / "wake_nihao_xiaoche.onnx",
        VOICE_ROOT / "kws" / "wake_nihao_xiaoche_clean.onnx",
    ], str(VOICE_ROOT / "kws" / "wake_nihao_xiaoche_v2.onnx")),
)

STOP_MODEL = os.getenv(
    "VOICE_STOP_MODEL",
    _pick_model([
        VOICE_ROOT / "kws" / "stop_smallcar_v1.onnx",
        VOICE_ROOT / "kws" / "stop_xiaochetingzhi_v1.onnx",
        VOICE_ROOT / "kws" / "stop_smallcar.onnx",
    ], str(VOICE_ROOT / "kws" / "stop_smallcar_v1.onnx")),
)

PIPER_MODEL = os.getenv(
    "VOICE_PIPER_MODEL",
    str(VOICE_ROOT / "tts" / "zh_CN-huayan-x_low" / "zh_CN-huayan-x_low.onnx"),
)

CONFIG = VoiceServiceConfig(
    project_root=os.getenv("VOICE_SERVICE_ROOT", str(VOICE_SERVICE_ROOT)),
    runs_dir=os.getenv("VOICE_RUNS_DIR", str(VOICE_SERVICE_ROOT / "runs")),
    asr_dir=os.getenv("VOICE_ASR_DIR", str(VOICE_ROOT / "thirdparty" / "paraformer_asr_online")),
    # asr_dir=os.getenv("VOICE_ASR_DIR", str(VOICE_ROOT / "thirdparty" / "paraformer_asr")),
    vad_dir=os.getenv("VOICE_VAD_DIR", str(VOICE_ROOT / "thirdparty" / "fsmn_vad")),
    wake_tflite=WAKE_MODEL,
    stop_tflite=STOP_MODEL,
    piper_model=PIPER_MODEL,
    commands_json=os.getenv("VOICE_COMMANDS_JSON", str(VOICE_ROOT / "config" / "commands.json")),
    arecord_device=os.getenv("VOICE_ARECORD_DEVICE", "plughw:1,0"),

    asr_mode=os.getenv("VOICE_ASR_MODE", "online"),
    asr_online_chunk_frames=int(os.getenv("VOICE_ASR_ONLINE_CHUNK_FRAMES", "6")),
    asr_online_chunk_size=[int(x.strip()) for x in os.getenv("VOICE_ASR_ONLINE_CHUNK_SIZE", "0,8,4").split(",") if x.strip()],
    asr_online_encoder_chunk_look_back=int(os.getenv("VOICE_ASR_ONLINE_ENCODER_LOOK_BACK", "4")),
    asr_online_decoder_chunk_look_back=int(os.getenv("VOICE_ASR_ONLINE_DECODER_LOOK_BACK", "1")),
    asr_emit_partial=os.getenv("VOICE_ASR_EMIT_PARTIAL", "1") == "1",

    disable_tts=os.getenv("VOICE_DISABLE_TTS", "1" if VOICE_NO_SPEAKER_DEFAULT else "0") == "1",
    tts_mode=os.getenv("VOICE_TTS_MODE", "play"),
    play_cmd=os.getenv("VOICE_PLAY_CMD", "aplay -q"),

    task_transport=os.getenv("VOICE_TASK_TRANSPORT", "tcp"),
    task_tcp_host=os.getenv("VOICE_TASK_TCP_HOST", "127.0.0.1"),
    task_tcp_port=int(os.getenv("VOICE_TASK_TCP_PORT", "9001")),
    task_uds_path=os.getenv("VOICE_TASK_UDS_PATH", "/tmp/robot_stack/task_cmd.sock"),
    task_send_mode=os.getenv("VOICE_TASK_SEND_MODE", "oneshot"),
    task_reconnect_secs=float(os.getenv("VOICE_TASK_RECONNECT_SECS", "1.0")),
    task_send_timeout=float(os.getenv("VOICE_TASK_SEND_TIMEOUT", "1.0")),

    task_ack_transport=os.getenv("VOICE_TASK_ACK_TRANSPORT", "tcp"),
    task_ack_tcp_host=os.getenv("VOICE_TASK_ACK_TCP_HOST", "127.0.0.1"),
    task_ack_tcp_port=int(os.getenv("VOICE_TASK_ACK_TCP_PORT", "9012")),
    task_ack_uds_path=os.getenv("VOICE_TASK_ACK_UDS_PATH", "/tmp/robot_stack/task_ack.sock"),
    task_ack_timeout_s=float(os.getenv("VOICE_TASK_ACK_TIMEOUT_S", "0.60")),

    tts_event_transport=os.getenv("VOICE_TTS_EVENT_TRANSPORT", TTS_EVENT_TRANSPORT_DEFAULT),
    tts_event_host=os.getenv("VOICE_TTS_EVENT_HOST", "127.0.0.1"),
    tts_event_port=int(os.getenv("VOICE_TTS_EVENT_PORT", "9011")),
    tts_event_uds_path=os.getenv("VOICE_TTS_EVENT_UDS_PATH", "/tmp/robot_stack/tts_event.sock"),

    wake_th=float(os.getenv("VOICE_WAKE_TH", "0.60")),
    stop_th=float(os.getenv("VOICE_STOP_TH", "0.58")),
    armed_secs=float(os.getenv("VOICE_ARMED_SECS", "6.0")),
    followup_secs=float(os.getenv("VOICE_FOLLOWUP_SECS", "4.0")),
    stop_followup_secs=float(os.getenv("VOICE_STOP_FOLLOWUP_SECS", "5.5")),
    max_followup_turns=int(os.getenv("VOICE_MAX_FOLLOWUP_TURNS", "3")),
    max_reject_streak=int(os.getenv("VOICE_MAX_REJECT_STREAK", "2")),
    post_tts_mute_secs=float(os.getenv("VOICE_POST_TTS_MUTE_SECS", "1.2")),
    post_wake_mute_secs=float(os.getenv("VOICE_POST_WAKE_MUTE_SECS", "0.35")),
    stop_mute_secs=float(os.getenv("VOICE_STOP_MUTE_SECS", "0.18")),
    stop_guard_secs=float(os.getenv("VOICE_STOP_GUARD_SECS", "0.80")),
    stop_repeat_block_secs=float(os.getenv("VOICE_STOP_REPEAT_BLOCK_SECS", "1.20")),
    asr_stream_idle_timeout_s=float(os.getenv("VOICE_ASR_STREAM_IDLE_TIMEOUT_S", "2.5")),
    asr_stream_session_timeout_s=float(os.getenv("VOICE_ASR_STREAM_SESSION_TIMEOUT_S", "8.0")),
    rearm_after_stream_timeout_s=float(os.getenv("VOICE_REARM_AFTER_STREAM_TIMEOUT_S", "0.0")),
    heartbeat_secs=float(os.getenv("VOICE_HEARTBEAT_SECS", "10.0")),
    show_mic_info=os.getenv("VOICE_SHOW_MIC_INFO", "0") == "1",
    debug=os.getenv("VOICE_DEBUG", "0") == "1",
    debug_ipc=os.getenv("VOICE_DEBUG_IPC", "1") == "1",
    debug_state=os.getenv("VOICE_DEBUG_STATE", "1") == "1",
    debug_stop=os.getenv("VOICE_DEBUG_STOP", "1") == "1",
    debug_timeline=os.getenv("VOICE_DEBUG_TIMELINE", "1") == "1",
    log_mode=os.getenv("VOICE_LOG_MODE", "concise"),
)
