"""Regression coverage for the shell launcher's safe mode decisions.

These tests source the launcher only; they never start a service or touch a
hardware device.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "start_robot_stack.sh"


def shell(script: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(["bash", "-c", script], cwd=str(ROOT), env=env,
                          text=True, capture_output=True, check=False)


def test_disabled_optional_endpoint_skips_immediately():
    result = shell(
        "source ./start_robot_stack.sh; STACK_PROFILE=dryrun; "
        "SYSTEM_CONFIG_PROFILE=dry_run; apply_profile_defaults; "
        "wait_for_endpoint mobile_gateway gateway_tts_event_in 1 0 0"
    )
    assert result.returncode == 0, result.stderr
    assert "[SKIP] gateway_tts_event_in disabled by configuration" in result.stdout


def test_disabled_required_endpoint_fails_immediately():
    result = shell(
        "source ./start_robot_stack.sh; STACK_PROFILE=dryrun; "
        "SYSTEM_CONFIG_PROFILE=dry_run; apply_profile_defaults; "
        "wait_for_endpoint mobile_gateway gateway_tts_event_in 1 0 1"
    )
    assert result.returncode != 0
    assert "required endpoint=gateway_tts_event_in is disabled" in result.stdout


def test_legacy_input_modes_normalize_without_hybrid():
    result = shell(
        "source ./start_robot_stack.sh; STACK_PROFILE=dryrun; "
        "ROBOT_INPUT_MODE=voice_only; apply_profile_defaults; printf '%s' \"$ROBOT_INPUT_MODE\""
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.endswith("voice")


def test_no_argument_prints_explicit_usage_without_starting_stack():
    result = subprocess.run(["bash", str(LAUNCHER)], cwd=str(ROOT), text=True,
                            capture_output=True, check=False)
    assert result.returncode != 0
    assert "start-mobile" in result.stdout
    assert "start-voice" in result.stdout


def test_voice_dryrun_profile_is_real_arecord_with_tts_disabled():
    result = subprocess.run(
        ["python3", "-m", "voice_service.app.main", "--profile",
         "configs/profiles/sc171_voice_orchestrator_dryrun.yaml", "--inspect-config"],
        cwd=str(ROOT), env={**os.environ, "PYTHONPATH": f"{ROOT}:{ROOT / 'Voice'}"},
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout
    for expected in (
        "arecord_device=plughw:CARD=UACDemoV10,DEV=0",
        "frontend_backend=onnx",
        "classifier_backend=onnx",
        "asr_quant=True",
        "vad_quant=True",
        "disable_tts=True",
        "task_transport=uds",
        "task_ack_transport=uds",
        "mobile_feedback_transport=disabled",
        "playback_transport=disabled",
    ):
        assert expected in output


def test_robot_input_role_does_not_override_voice_audio_source():
    result = subprocess.run(
        ["python3", "-m", "voice_service.app.main", "--profile",
         "configs/profiles/sc171_voice_replay_orchestrator_dryrun.yaml", "--inspect-config"],
        cwd=str(ROOT), env={**os.environ, "ROBOT_INPUT_MODE": "voice", "PYTHONPATH": f"{ROOT}:{ROOT / 'Voice'}"},
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "input_mode=wav_replay" in result.stdout
