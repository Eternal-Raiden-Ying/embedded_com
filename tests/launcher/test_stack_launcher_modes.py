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
    assert "start-voice-tts" in result.stdout


def test_dedicated_phone_tts_dryrun_preflight_allows_only_its_ipc_contract():
    result = shell(
        "source ./start_robot_stack.sh; STACK_PROFILE=dryrun; apply_profile_defaults; "
        "configure_voice_phone_tts_dryrun; normalize_input_mode; assert_dryrun_safety; "
        "phone_tts_route_enabled; "
        "printf '%s|%s|%s|%s' \"$SYSTEM_CONFIG_PROFILE\" \"$ROBOT_INPUT_MODE\" \"$FEEDBACK_OUTPUT_MODE\" \"$ORCH_SERIAL_DRY_RUN\""
    )
    assert result.returncode == 0, result.stderr
    assert "dedicated phone-TTS IPC enabled" in result.stdout
    assert result.stdout.endswith("sc171_voice_phone_tts_dryrun|voice|phone_tts|1")


def test_voice_phone_tts_dryrun_banner_reports_selected_effective_modes():
    result = shell(
        "source ./start_robot_stack.sh; STACK_PROFILE=dryrun; apply_profile_defaults; "
        "configure_voice_phone_tts_dryrun; normalize_input_mode; show_banner"
    )
    assert result.returncode == 0, result.stderr
    assert "config profile : sc171_voice_phone_tts_dryrun" in result.stdout
    assert "input mode     : voice" in result.stdout
    assert "feedback mode  : phone_tts" in result.stdout
    assert "ORCH_SERIAL_DRY_RUN=1" in result.stdout
    assert "serial dry-run : true" in result.stdout
    assert "arm dry-run    : true" in result.stdout


def test_full_phone_tts_preflight_uses_full_contract_not_generic_dryrun_gate():
    result = shell(
        "source ./start_robot_stack.sh; STACK_PROFILE=full; apply_profile_defaults; "
        "configure_voice_phone_tts_full; normalize_input_mode; assert_launcher_safety; "
        "printf '%s|%s|%s|%s' \"$SYSTEM_CONFIG_PROFILE\" \"$ROBOT_INPUT_MODE\" \"$FEEDBACK_OUTPUT_MODE\" \"$ORCH_SERIAL_DRY_RUN\""
    )
    assert result.returncode == 0, result.stderr
    assert "full phone-TTS safety gate" in result.stdout
    assert "unsafe dry-run profile" not in result.stderr
    assert result.stdout.endswith("sc171_voice_phone_tts|voice|phone_tts|0")


def test_full_phone_tts_banner_reports_real_actuators_and_selected_modes():
    result = shell(
        "source ./start_robot_stack.sh; STACK_PROFILE=full; apply_profile_defaults; "
        "configure_voice_phone_tts_full; normalize_input_mode; show_banner"
    )
    assert result.returncode == 0, result.stderr
    assert "profile        : full" in result.stdout
    assert "config profile : sc171_voice_phone_tts" in result.stdout
    assert "input mode     : voice" in result.stdout
    assert "feedback mode  : phone_tts" in result.stdout
    assert "serial dry-run : false" in result.stdout
    assert "arm dry-run    : false" in result.stdout


def test_main_full_start_voice_tts_maps_to_production_phone_profile_without_starting_services():
    result = shell(
        "source ./start_robot_stack.sh; "
        "start_stack(){ printf '%s|%s|%s|%s|%s' \"$STACK_PROFILE\" \"$SYSTEM_CONFIG_PROFILE\" \"$ROBOT_INPUT_MODE\" \"$FEEDBACK_OUTPUT_MODE\" \"$VOICE_PROFILE\"; }; "
        "main full start-voice-tts"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        "full|sc171_voice_phone_tts|voice_only|phone_tts|"
        f"{ROOT}/configs/profiles/sc171_voice_phone_tts.yaml"
    )


def test_full_phone_tts_blocks_residual_serial_dryrun_override():
    result = shell(
        "source ./start_robot_stack.sh; STACK_PROFILE=full; ORCH_SERIAL_DRY_RUN=1; "
        "apply_profile_defaults; configure_voice_phone_tts_full; assert_launcher_safety"
    )
    assert result.returncode != 0
    assert "full profile unexpectedly keeps actuator dry-run enabled" in result.stderr


def test_full_phone_tts_requires_complete_feedback_route():
    result = shell(
        "source ./start_robot_stack.sh; STACK_PROFILE=full; apply_profile_defaults; "
        "configure_voice_phone_tts_full; export MOBILE_GATEWAY_TTS_PLAYBACK_OUT_TRANSPORT=disabled; "
        "assert_launcher_safety"
    )
    assert result.returncode != 0
    assert "full phone-TTS profile has incomplete feedback route" in result.stderr


def test_full_phone_tts_requires_nonempty_vista_uds_path():
    result = shell(
        "source ./start_robot_stack.sh; STACK_PROFILE=full; apply_profile_defaults; "
        "configure_voice_phone_tts_full; export VISION_REQ_IN_SOCKET_PATH=''; "
        "assert_launcher_safety"
    )
    assert result.returncode != 0
    assert "vision.req_in must be enabled with a UDS socket path" in result.stderr


def test_full_phone_tts_profile_configures_vista_uds_endpoints():
    code = """
from common.config.loader import get_config

cfg = get_config()
assert cfg.vision.req_in.transport == 'uds'
assert cfg.vision.req_in.ipc_socket_path == '/tmp/robot_stack/vision_req.sock'
assert cfg.vision.obs_out.transport == 'uds'
assert cfg.vision.obs_out.ipc_socket_path == '/tmp/robot_stack/vision_obs.sock'
assert cfg.orchestrator.vision_req_out.transport == 'uds'
assert cfg.orchestrator.vision_req_out.ipc_socket_path == '/tmp/robot_stack/vision_req.sock'
assert cfg.orchestrator.vision_obs_in.transport == 'uds'
assert cfg.orchestrator.vision_obs_in.ipc_socket_path == '/tmp/robot_stack/vision_obs.sock'
"""
    result = subprocess.run(
        ["python3", "-c", code],
        cwd=str(ROOT),
        env={
            **os.environ,
            "SYSTEM_CONFIG_PROFILE": "sc171_voice_phone_tts",
            "PYTHONPATH": str(ROOT),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_online_dryrun_profile_inherits_vista_ipc_and_launcher_prints_effective_summary():
    result = shell(
        "source ./start_robot_stack.sh; STACK_PROFILE=dryrun; apply_profile_defaults; "
        "configure_voice_phone_tts_online; normalize_input_mode; validate_profile_ipc"
    )
    assert result.returncode == 0, result.stderr
    assert "name=sc171_voice_phone_tts_online_asr_dryrun" in result.stdout
    assert "vision_req mode=uds path=/tmp/robot_stack/vision_req.sock" in result.stdout
    assert "vision_obs mode=uds path=/tmp/robot_stack/vision_obs.sock" in result.stdout
    assert "asr mode=online" in result.stdout
    assert "serial_dry_run=True arm_dry_run=True" in result.stdout


def test_online_profile_ipc_preflight_blocks_empty_uds_path_before_service_start():
    result = shell(
        "source ./start_robot_stack.sh; STACK_PROFILE=dryrun; apply_profile_defaults; "
        "configure_voice_phone_tts_online; normalize_input_mode; "
        "export VISION_REQ_IN_SOCKET_PATH=''; validate_profile_ipc"
    )
    assert result.returncode != 0
    assert "invalid profile IPC" in result.stderr
    assert "vision.req_in uses uds with an empty socket path" in result.stderr


def test_main_dryrun_start_voice_tts_online_maps_to_online_dryrun_profile_without_starting_services():
    result = shell(
        "source ./start_robot_stack.sh; "
        "start_stack(){ printf '%s|%s|%s|%s|%s' \"$STACK_PROFILE\" \"$SYSTEM_CONFIG_PROFILE\" \"$ROBOT_INPUT_MODE\" \"$FEEDBACK_OUTPUT_MODE\" \"$VOICE_PROFILE\"; }; "
        "main dryrun start-voice-tts-online"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        "dryrun|sc171_voice_phone_tts_online_asr_dryrun|voice_only|phone_tts|"
        f"{ROOT}/configs/profiles/sc171_voice_phone_tts_online_asr_dryrun.yaml"
    )


def test_phone_tts_dryrun_profile_keeps_vista_ipc_and_actuator_dryrun_contract():
    code = """
from common.config.loader import get_config
from voice_service.config.loader import load_voice_config

cfg = get_config()
voice = load_voice_config(['--profile', 'configs/profiles/sc171_voice_phone_tts_dryrun.yaml'])
assert cfg.vision.req_in.transport == 'uds'
assert cfg.vision.req_in.ipc_socket_path == '/tmp/robot_stack/vision_req.sock'
assert cfg.vision.obs_out.transport == 'uds'
assert cfg.vision.obs_out.ipc_socket_path == '/tmp/robot_stack/vision_obs.sock'
assert cfg.orchestrator.vision_req_out.transport == 'uds'
assert cfg.orchestrator.vision_req_out.ipc_socket_path == '/tmp/robot_stack/vision_req.sock'
assert cfg.orchestrator.vision_obs_in.transport == 'uds'
assert cfg.orchestrator.vision_obs_in.ipc_socket_path == '/tmp/robot_stack/vision_obs.sock'
assert not hasattr(cfg.orchestrator, 'vista')
assert cfg.orchestrator.serial.dry_run is True
assert cfg.orchestrator.serial.port == 'DRY_RUN'
assert cfg.orchestrator.arm_serial.dry_run is True
assert cfg.gateway.runtime.feedback_output_mode == 'phone_tts'
assert voice.input_mode == 'voice_only'
assert voice.dry_run_text is False
"""
    result = subprocess.run(
        ["python3", "-c", code],
        cwd=str(ROOT),
        env={
            **os.environ,
            "SYSTEM_CONFIG_PROFILE": "sc171_voice_phone_tts_dryrun",
            "PYTHONPATH": f"{ROOT}:{ROOT / 'Voice'}",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_generic_dryrun_keeps_phone_tts_transports_blocked():
    result = shell(
        "source ./start_robot_stack.sh; STACK_PROFILE=dryrun; SYSTEM_CONFIG_PROFILE=dry_run; "
        "ROBOT_INPUT_MODE=voice; FEEDBACK_OUTPUT_MODE=phone_tts; "
        "VOICE_PROFILE=configs/profiles/sc171_voice_phone_tts_dryrun.yaml; apply_profile_defaults; "
        "assert_dryrun_safety"
    )
    assert result.returncode != 0
    assert "voice.mobile_feedback_transport must be disabled" in result.stderr
    assert "voice.playback_transport must be disabled" in result.stderr


def test_non_dedicated_profile_with_real_actuators_remains_blocked():
    result = shell(
        "source ./start_robot_stack.sh; STACK_PROFILE=dryrun; SYSTEM_CONFIG_PROFILE=sc171_voice_phone_tts; "
        "ROBOT_INPUT_MODE=voice; FEEDBACK_OUTPUT_MODE=phone_tts; "
        "VOICE_PROFILE=configs/profiles/sc171_voice_phone_tts.yaml; apply_profile_defaults; "
        "ORCH_SERIAL_DRY_RUN=0; "
        "assert_dryrun_safety"
    )
    assert result.returncode != 0
    assert "orchestrator.serial.dry_run must be true" in result.stderr
    assert "orchestrator.arm_serial.dry_run must be true" in result.stderr


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
