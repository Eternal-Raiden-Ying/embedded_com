#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from types import SimpleNamespace
import time
import wave
import pytest

from voice_service.runtime.playback import PhonePlaybackGuard
from voice_service.runtime.state import RuntimeState
from voice_service.config.loader import load_voice_config
from voice_service.runtime.service import DebugInputOnlySender
from voice_service.runtime.workers import AudioKWSWorker, dispatch_task_cmd, select_hotword_action
from voice_service.runtime.mic_stream import WavReplayAudioSource
from orchestrator_service.ipc.protocol import TaskCmd, make_task_ack


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FakeSender:
    def __init__(self, result=True):
        self.result = result
        self.messages = []

    def send(self, payload):
        self.messages.append(dict(payload))
        return self.result


def make_cfg():
    return SimpleNamespace(
        mobile_feedback_transport="uds",
        playback_start_timeout_s=1.5,
        playback_finish_timeout_s=6.0,
        post_playback_guard_s=0.35,
        armed_secs=6.0,
    )


def ack_for(event, state, **changes):
    payload = {
        "type": "tts_playback_state",
        "event_id": event["event_id"],
        "session_id": event["session_id"],
        "epoch": event["epoch"],
        "state": state,
    }
    payload.update(changes)
    return payload


def test_wake_waits_for_ordered_phone_playback_then_arms():
    clock, sender, rt = FakeClock(), FakeSender(), RuntimeState()
    guard = PhonePlaybackGuard(make_cfg(), rt, sender, clock=clock)

    assert guard.request_wake_prompt()
    event = sender.messages[-1]
    assert event["phrase_id"] == "WAKE_PROMPT"
    assert rt.snapshot()["state"] == "WAIT_PROMPT_PLAYBACK"
    assert not rt.is_armed()

    assert guard.handle_playback(ack_for(event, "started"))
    assert rt.snapshot()["state"] == "WAIT_PROMPT_PLAYBACK"
    assert not rt.is_armed()
    assert guard.handle_playback(ack_for(event, "finished"))
    assert rt.snapshot()["state"] == "POST_PLAYBACK_GUARD"
    clock.now += 0.36
    guard.poll()
    assert rt.snapshot()["state"] == "ARMED_WAIT"
    assert rt.is_armed()


def test_stale_wrong_and_duplicate_playback_ack_are_ignored():
    clock, sender, rt = FakeClock(), FakeSender(), RuntimeState()
    guard = PhonePlaybackGuard(make_cfg(), rt, sender, clock=clock)
    assert guard.request_wake_prompt()
    event = sender.messages[-1]
    assert not guard.handle_playback(ack_for(event, "finished"))
    assert not guard.handle_playback(ack_for(event, "started", event_id="wrong"))
    assert not guard.handle_playback(ack_for(event, "started", session_id="wrong"))
    assert not guard.handle_playback(ack_for(event, "started", epoch=99))
    assert guard.handle_playback(ack_for(event, "started"))
    assert guard.handle_playback(ack_for(event, "finished"))
    assert not guard.handle_playback(ack_for(event, "finished"))


def test_playback_timeout_recovers_through_guard_without_arming_early():
    clock, sender, rt = FakeClock(), FakeSender(), RuntimeState()
    guard = PhonePlaybackGuard(make_cfg(), rt, sender, clock=clock)
    assert guard.request_wake_prompt()
    clock.now += 1.6
    guard.poll()
    assert rt.snapshot()["state"] == "POST_PLAYBACK_GUARD"
    assert not rt.is_armed()
    clock.now += 0.36
    guard.poll()
    assert rt.snapshot()["state"] == "ARMED_WAIT"


def test_stop_epoch_invalidates_old_playback_ack():
    clock, sender, rt = FakeClock(), FakeSender(), RuntimeState()
    guard = PhonePlaybackGuard(make_cfg(), rt, sender, clock=clock)
    assert guard.request_wake_prompt()
    event = sender.messages[-1]
    rt.bump_epoch()
    guard.invalidate()
    assert not guard.handle_playback(ack_for(event, "started"))
    assert not rt.is_armed()


def test_phone_profile_disables_local_piper_and_enables_phone_endpoints():
    cfg = load_voice_config(["--profile", "configs/profiles/sc171_voice_phone_tts.yaml"])
    assert cfg.disable_tts
    assert cfg.mobile_feedback_transport == "uds"
    assert cfg.mobile_tts_event_uds_path == "/tmp/robot_stack/mobile_tts_event.sock"
    assert cfg.playback_transport == "uds"
    assert cfg.playback_uds_path == "/tmp/robot_stack/tts_playback.sock"
    assert cfg.frontend_backend == "onnx"
    assert cfg.classifier_backend == "onnx"
    assert cfg.asr_quant is True
    assert cfg.vad_quant is True


def test_phone_dryrun_profile_uses_onnx_quantized_voice_configuration():
    cfg = load_voice_config(["--profile", "configs/profiles/sc171_voice_phone_tts_dryrun.yaml"])
    assert cfg.disable_tts
    assert cfg.task_transport == "uds"
    assert cfg.mobile_feedback_transport == "uds"
    assert cfg.playback_transport == "uds"
    assert cfg.frontend_backend == "onnx"
    assert cfg.classifier_backend == "onnx"
    assert cfg.asr_quant is True
    assert cfg.vad_quant is True


def test_debug_profile_disables_all_robot_and_phone_ipc():
    cfg = load_voice_config(["--profile", "configs/profiles/sc171_voice_input_debug.yaml"])
    assert cfg.debug_input_only
    assert cfg.task_transport == "disabled"
    assert cfg.task_ack_transport == "disabled"
    assert cfg.tts_event_transport == "disabled"
    assert cfg.mobile_feedback_transport == "disabled"
    assert cfg.playback_transport == "disabled"
    assert cfg.disable_tts
    assert cfg.frontend_backend == "onnx"
    assert cfg.classifier_backend == "onnx"
    assert cfg.asr_quant is True
    assert cfg.vad_quant is True


def test_debug_sender_suppresses_find_and_stop_without_socket_io():
    sender, rt = DebugInputOnlySender(), RuntimeState()
    find_result = dispatch_task_cmd(
        {"intent": "FIND", "target": "apple", "session_id": "debug", "epoch": 0},
        sender, None, rt, 0.0,
    )
    stop_result = dispatch_task_cmd(
        {"intent": "STOP", "session_id": "debug", "epoch": 1},
        sender, None, rt, 0.0,
    )
    assert not find_result["sent"]
    assert not stop_result["sent"]
    assert stop_result["cmd"]["high_priority"] is True
    assert sender.snapshot()["link_state"] == "DISABLED"


class CountingSender:
    def __init__(self):
        self.calls = 0

    def send(self, payload):
        self.calls += 1
        return False

    def snapshot(self):
        return {"link_state": "SHOULD_NOT_BE_USED"}


def test_debug_dispatch_does_not_call_sender_or_degrade_ipc_for_find_or_stop():
    sender, rt = CountingSender(), RuntimeState()
    find = dispatch_task_cmd(
        {"intent": "FIND", "target": "apple", "session_id": "debug", "epoch": 0},
        sender, None, rt, 0.0, suppress_dispatch=True,
    )
    stop = dispatch_task_cmd(
        {"intent": "STOP", "session_id": "debug", "epoch": 1},
        sender, None, rt, 0.0, label="STOP", suppress_dispatch=True,
    )
    assert find["suppressed"] and stop["suppressed"]
    assert sender.calls == 0
    assert rt.snapshot()["ipc_state"] == "CONNECTED"
    assert rt.snapshot()["last_intent"] != "IPC_SEND_FAIL"


def test_recording_freezes_armed_deadline_but_armed_wait_still_times_out():
    rt = RuntimeState()
    rt.start_session(0.0, reason="test")
    assert AudioKWSWorker._armed_timeout_applies("ARMED_WAIT", rt)
    rt.start_session(0.0, reason="test")
    rt.begin_recording()
    assert rt.snapshot()["state"] == "REC"
    assert not AudioKWSWorker._armed_timeout_applies("REC", rt)


def test_stop_wins_when_wake_and_stop_cross_threshold_together():
    pred = {"wake": 0.836, "stop": 0.991}
    assert select_hotword_action(pred, "wake", 0.90, "stop", 0.58) == "STOP"


def _write_pcm_wav(path, samples=1280):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x01\x00" * samples)


def test_wav_replay_matches_pcm_frame_contract_and_step_order(tmp_path):
    wav_path = tmp_path / "input.wav"
    _write_pcm_wav(wav_path)
    manifest = tmp_path / "scenario.yaml"
    manifest.write_text("steps:\n  - silence_s: 0.08\n  - wav: {}\n".format(wav_path), encoding="utf-8")
    source = WavReplayAudioSource(str(manifest), realtime=False)
    first, second = source.read_frame(), source.read_frame()
    assert len(first) == len(second) == 2560
    assert first == b"\x00" * 2560
    assert second == b"\x01\x00" * 1280
    assert source.read_frame() is None
    assert source.completed


def test_wav_replay_realtime_does_not_inject_all_frames_at_once(tmp_path):
    manifest = tmp_path / "realtime.yaml"
    manifest.write_text("steps:\n  - silence_s: 0.16\n", encoding="utf-8")
    source = WavReplayAudioSource(str(manifest), realtime=True)
    start = time.monotonic()
    source.read_frame()
    source.read_frame()
    assert time.monotonic() - start >= 0.07


def test_wav_replay_missing_manifest_or_invalid_wav_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        WavReplayAudioSource(str(tmp_path / "missing.yaml"))
    bad = tmp_path / "bad.yaml"
    bad.write_text("steps:\n  - wav: missing.wav\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        WavReplayAudioSource(str(bad))


def test_task_ack_preserves_execution_status_for_replay_orchestrator_chain():
    cmd = TaskCmd.from_dict({"type": "task_cmd", "intent": "FIND", "target": "apple", "cmd_id": "replay-cmd", "session_id": "replay-session"}, {"apple"})
    ack = make_task_ack(cmd, accepted=True, state="SEARCH_TABLE", reason="accepted", execution_status="executable")
    assert ack["cmd_id"] == "replay-cmd"
    assert ack["execution_status"] == "executable"
