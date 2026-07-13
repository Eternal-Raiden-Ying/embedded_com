import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ORCH_ROOT = ROOT / "orchestrator"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from common.config.schema import MobileGatewayConfig, SystemGlobalConfig
from common.config.loader import load_global_config
from common.config.validators import validate_config
from orchestrator_service.mobile_gateway.config.board_config import build_config
from orchestrator_service.mobile_gateway.runtime.service import MobileGatewayService, TaskTemplate


class _BackendSpy:
    def __init__(self):
        self.submitted = []

    def submit(self, payload):
        self.submitted.append(dict(payload))
        return True, "accepted"


class _MqttSpy:
    def __init__(self):
        self.statuses = []
        self.heartbeats = []
        self.acks = []

    def publish_status(self, payload):
        self.statuses.append(dict(payload))

    def publish_heartbeat(self, payload):
        self.heartbeats.append(dict(payload))

    def publish_ack(self, payload):
        self.acks.append(dict(payload))


def _service(mode="mobile"):
    cfg = SystemGlobalConfig().gateway
    cfg.runtime.log_enabled = False
    cfg.runtime.status_stdout = False
    cfg.runtime.task_input_mode = mode
    cfg.command_in.transport = "disabled"
    cfg.status_out.transport = "disabled"
    cfg.orchestrator_task_cmd_out.transport = "disabled"
    cfg.orchestrator_task_ack_in.transport = "disabled"
    cfg.tts_event_in.transport = "disabled"
    cfg.tts_playback_out.transport = "disabled"
    cfg.backend.mode = "mock"
    cfg.backend.observer_enabled = False
    service = MobileGatewayService(cfg)
    service.backend = _BackendSpy()
    service.mqtt_adapter = _MqttSpy()
    return service


def _fetch(cmd_id="fetch-1"):
    return {"cmd": "fetch_object", "target": "apple", "cmd_id": cmd_id, "session_id": "s-1"}


def test_mobile_mode_allows_fetch_object_and_forwards_task_cmd():
    service = _service("mobile")

    result = service.handle_mobile_command_payload(_fetch())

    assert result["accepted"] is True
    assert service.backend.submitted[-1]["intent"] == "FIND"
    assert service._active_template.command == "fetch_object"


@pytest.mark.parametrize("command", ["fetch_object", "go_home", "resume", "retry_search"])
def test_voice_mode_rejects_all_ordinary_mobile_tasks_before_side_effects(command):
    service = _service("voice")
    service._active_template = TaskTemplate("go_home", None, "existing-session")
    service._last_stop_ts = time.time()
    payload = _fetch(command) if command == "fetch_object" else {
        "cmd": command,
        "cmd_id": f"{command}-1",
        "session_id": "s-1",
    }

    result = service.handle_mobile_command_payload(payload)

    assert result == {
        "ok": False,
        "accepted": False,
        "reason": "mobile_task_disabled_in_current_mode",
        "message": "当前模式不允许手机发起普通机器人任务",
    }
    assert service.backend.submitted == []
    assert service._pending_after_stop_command is None
    assert service._active_template.session_id == "existing-session"
    assert payload["cmd_id"] not in service._recent_cmd_id_set
    assert service.mqtt_adapter.acks[-1]["accepted"] is False
    assert service.mqtt_adapter.acks[-1]["reason"] == "mobile_task_disabled_in_current_mode"


@pytest.mark.parametrize("command", ["manual_drive", "manual_stop"])
def test_voice_mode_allows_manual_controls(command):
    service = _service("voice")

    result = service.handle_mobile_command_payload({"cmd": command, "cmd_id": f"{command}-1", "session_id": "s-1"})

    assert result["accepted"] is True
    assert service.backend.submitted[-1]["cmd"] == command


def test_voice_mode_allows_stop_and_emergency_stop_compatibility():
    service = _service("voice")

    result = service.handle_mobile_command_payload({"cmd": "emergency_stop", "cmd_id": "stop-1", "session_id": "s-1"})

    assert result["accepted"] is True
    assert service.backend.submitted[-1]["intent"] == "STOP"


def test_voice_mode_http_fallback_is_filtered_before_task_cmd_forwarding():
    service = _service("voice")
    service._last_stop_ts = time.time()

    result = service._handle_http_task_payload({"action": "fetch", "target": "apple", "cmd_id": "http-fetch"})

    assert result["reason"] == "mobile_task_disabled_in_current_mode"
    assert service.backend.submitted == []
    assert service._pending_after_stop_command is None


def test_voice_mode_allows_core_control_without_spawning_stack_script(monkeypatch):
    service = _service("voice")
    seen = []

    def _core_handler(payload, action, raw_cmd):
        seen.append((action, raw_cmd))
        return {"ok": True, "accepted": True, "message": "core accepted"}

    monkeypatch.setattr(service, "_handle_core_control_payload", _core_handler)
    result = service.handle_mobile_command_payload({"cmd": "core_start", "cmd_id": "core-1", "session_id": "s-1"})

    assert result["accepted"] is True
    assert seen == [("core_start", "core_start")]


@pytest.mark.parametrize("mode", ["mobile", "voice"])
def test_status_queries_are_allowed_in_both_modes(monkeypatch, mode):
    service = _service(mode)
    seen = []
    monkeypatch.setattr(service, "_handle_core_status_payload", lambda payload, raw: seen.append("core") or {"ok": True, "accepted": True, "message": "core"})
    monkeypatch.setattr(service, "_handle_task_status_payload", lambda payload, raw: seen.append("task") or {"ok": True, "accepted": True, "message": "task"})

    assert service.handle_mobile_command_payload({"cmd": "core_status", "cmd_id": "core-status"})["accepted"] is True
    assert service.handle_mobile_command_payload({"cmd": "task_status", "cmd_id": "task-status"})["accepted"] is True
    assert service.handle_mobile_command_payload({"cmd": "query_status", "cmd_id": "query-status"})["accepted"] is True
    assert seen == ["core", "task"]


def test_capability_fields_are_attached_to_status_heartbeat_and_gateway_ack():
    service = _service("voice_only")

    service._publish_status({"state": "idle", "session_id": "s-1"}, force=True)
    service._emit_heartbeat_if_needed()
    service._publish_gateway_ack({"cmd": "query_status", "cmd_id": "ack-1"}, accepted=True, message="ok")

    expected = {
        "task_input_mode": "voice",
        "feedback_output_mode": "optional",
        "mobile_permissions": {
            "task_commands": False,
            "manual_control": True,
            "core_control": True,
            "emergency_stop": True,
        },
    }
    for payload in (service.mqtt_adapter.statuses[-1], service.mqtt_adapter.heartbeats[-1], service.mqtt_adapter.acks[-1]):
        for key, value in expected.items():
            assert payload[key] == value


@pytest.mark.parametrize(("raw_mode", "normalized_mode"), [("mobile_only", "mobile"), ("voice_only", "voice")])
def test_legacy_task_input_mode_aliases_are_normalized_for_mobile(raw_mode, normalized_mode):
    service = _service(raw_mode)

    assert service._mobile_capability_fields()["task_input_mode"] == normalized_mode


def test_invalid_gateway_mode_is_rejected_by_validator():
    cfg = SystemGlobalConfig()
    cfg.gateway.runtime.task_input_mode = "keyboard"

    with pytest.raises(ValueError, match="task_input_mode"):
        validate_config(cfg)


@pytest.mark.parametrize(
    ("profile", "task_input_mode", "feedback_output_mode", "task_commands_allowed", "voice_input_mode"),
    [
        ("sc171_voice_phone_tts", "voice", "phone_tts", False, "voice_only"),
        ("sc171_voice_phone_tts_dryrun", "voice", "phone_tts", False, "voice_only"),
        ("sc171_mobile_control", "mobile", "optional", True, "mobile_only"),
        ("sc171_mobile_control_dryrun", "mobile", "optional", True, "mobile_only"),
    ],
)
def test_dual_mode_profiles_define_gateway_policy_and_topics(
    monkeypatch, profile, task_input_mode, feedback_output_mode, task_commands_allowed, voice_input_mode
):
    monkeypatch.setenv("SYSTEM_CONFIG_PROFILE", profile)
    cfg = load_global_config(str(ROOT / "configs" / "system_config.yaml"))
    voice_profile = ROOT / "configs" / "profiles" / f"{profile}.yaml"

    assert cfg.gateway.runtime.task_input_mode == task_input_mode
    assert cfg.gateway.runtime.feedback_output_mode == feedback_output_mode
    assert cfg.gateway.runtime.mobile_task_commands_allowed is task_commands_allowed
    assert cfg.gateway.runtime.mobile_manual_control_allowed is True
    assert cfg.gateway.runtime.mobile_core_control_allowed is True
    assert cfg.gateway.runtime.mobile_emergency_stop_allowed is True
    assert cfg.gateway.mqtt.accept_commands is True
    assert cfg.gateway.mqtt.retain_status is True
    assert cfg.gateway.mqtt.retain_heartbeat is False
    assert vars(cfg.gateway.mqtt.topics) == {
        "cmd": "robot/v1/SC171/mobile/cmd",
        "ack": "robot/v1/SC171/mobile/ack",
        "status": "robot/v1/SC171/mobile/status",
        "heartbeat": "robot/v1/SC171/heartbeat",
        "tts": "robot/v1/SC171/mobile/tts",
        "tts_ack": "robot/v1/SC171/mobile/tts_ack",
    }
    assert voice_input_mode in voice_profile.read_text(encoding="utf-8")


def test_gateway_uses_common_schema_after_local_schema_removal():
    cfg = build_config()
    local_schema = ROOT / "orchestrator" / "orchestrator_service" / "mobile_gateway" / "config" / "schema.py"
    gateway_sources = (ROOT / "orchestrator" / "orchestrator_service" / "mobile_gateway").rglob("*.py")

    assert isinstance(cfg, MobileGatewayConfig)
    assert not local_schema.exists()
    assert all(
        "from ..config.schema import" not in path.read_text(encoding="utf-8")
        and "mobile_gateway.config.schema" not in path.read_text(encoding="utf-8")
        for path in gateway_sources
    )
