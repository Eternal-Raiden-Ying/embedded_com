#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from types import SimpleNamespace

from voice_service.runtime.playback import PhonePlaybackGuard
from voice_service.runtime.state import RuntimeState
from voice_service.config.loader import load_voice_config
from voice_service.runtime.service import DebugInputOnlySender
from voice_service.runtime.workers import dispatch_task_cmd


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
