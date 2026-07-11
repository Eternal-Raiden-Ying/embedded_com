import json
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
ORCH_ROOT = ROOT / "orchestrator"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from common.config.schema import SystemGlobalConfig
from common.config.loader import load_global_config
from orchestrator_service.ipc.transport import JsonlClientSender, JsonlInboundServer
from orchestrator_service.ipc.protocol import make_tts_event
from orchestrator_service.mobile_gateway.adapters.mqtt_adapter import MqttAdapter
from orchestrator_service.mobile_gateway.runtime.service import MobileGatewayService


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _wait_for_items(server: JsonlInboundServer, expected: int = 1):
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        items = server.drain()
        if len(items) >= expected:
            return items
        time.sleep(0.02)
    return []


class _MqttPublishSpy:
    def __init__(self):
        self.events = []

    def publish_tts(self, payload):
        self.events.append(dict(payload))

    def start(self):
        pass

    def stop(self):
        pass

    def publish_status(self, payload):
        pass

    def publish_ack(self, payload):
        pass

    def publish_heartbeat(self, payload):
        pass


@pytest.mark.parametrize("transport", ["tcp", "uds"])
def test_tts_bridge_forwards_framed_event_and_playback_state(tmp_path: Path, transport: str):
    cfg = SystemGlobalConfig().gateway
    cfg.runtime.log_enabled = False
    cfg.runtime.status_stdout = False
    cfg.command_in.transport = "disabled"
    cfg.status_out.transport = "disabled"
    cfg.orchestrator_task_cmd_out.transport = "disabled"
    cfg.orchestrator_task_ack_in.transport = "disabled"
    cfg.backend.mode = "mock"
    cfg.tts_event_in.transport = transport
    cfg.tts_playback_out.transport = transport

    if transport == "tcp":
        cfg.tts_event_in.tcp_host = "127.0.0.1"
        cfg.tts_event_in.tcp_port = _free_port()
        cfg.tts_playback_out.tcp_host = "127.0.0.1"
        cfg.tts_playback_out.tcp_port = _free_port()
    else:
        cfg.tts_event_in.ipc_socket_path = str(tmp_path / "mobile_tts_event.sock")
        cfg.tts_playback_out.ipc_socket_path = str(tmp_path / "tts_playback.sock")

    playback_server = JsonlInboundServer(
        mode=transport,
        tcp_host=cfg.tts_playback_out.tcp_host,
        tcp_port=cfg.tts_playback_out.tcp_port,
        uds_path=cfg.tts_playback_out.ipc_socket_path,
        name="test_tts_playback",
    )
    playback_server.start()
    service = MobileGatewayService(cfg)
    service.mqtt_adapter = _MqttPublishSpy()
    sender = JsonlClientSender(
        mode=transport,
        tcp_host=cfg.tts_event_in.tcp_host,
        tcp_port=cfg.tts_event_in.tcp_port,
        uds_path=cfg.tts_event_in.ipc_socket_path,
        name="test_tts_event",
        send_mode="oneshot",
    )
    try:
        service.start()
        orch_tts_event = make_tts_event("开始寻找 apple")
        orch_tts_event["event_id"] = "evt-1"
        assert sender.send(orch_tts_event)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and service.tts_event_server.snapshot()["total_recv_count"] < 1:
            time.sleep(0.02)
        service._drain_tts_events()
        assert len(service.mqtt_adapter.events) == 1
        event = service.mqtt_adapter.events[0]
        assert event["type"] == "tts_event"
        assert event["event_id"] == "evt-1"
        assert event["text"] == "开始寻找 apple"
        assert isinstance(event["ts"], float)

        service._handle_tts_event({"text": "generated id"})
        assert len(service.mqtt_adapter.events) == 2
        assert service.mqtt_adapter.events[-1]["event_id"].startswith("tts_")

        service._handle_tts_event({"event_id": "evt-1", "text": "duplicate"})
        assert len(service.mqtt_adapter.events) == 2

        service._handle_tts_ack({"type": "tts_ack", "event_id": "evt-1", "state": "finished"})
        playback = _wait_for_items(playback_server)
        assert playback[0]["payload"] == {
            "type": "tts_ack", "event_id": "evt-1", "state": "finished", "kind": "tts_playback"
        }
    finally:
        sender.close()
        service.stop()
        playback_server.close()


class _FakeMqttClient:
    def __init__(self):
        self.subscriptions = []
        self.published = []

    def subscribe(self, topic, qos):
        self.subscriptions.append((topic, qos))

    def publish(self, topic, payload, qos, retain):
        self.published.append((topic, json.loads(payload), qos, retain))
        return SimpleNamespace(rc=0)


def test_mqtt_adapter_playback_only_routes_tts_ack_and_rejects_cmd_subscription():
    cfg = SystemGlobalConfig().gateway.mqtt
    cfg.accept_commands = False
    command_events = []
    tts_ack_events = []
    adapter = MqttAdapter(
        cfg,
        command_events.append,
        tts_ack_handler=tts_ack_events.append,
    )
    client = _FakeMqttClient()
    adapter._client = client
    adapter._started = True

    adapter._on_connect(client, None, None, 0)
    assert client.subscriptions == [(cfg.topics.tts_ack, cfg.ack_qos)]

    adapter._on_message(client, None, SimpleNamespace(
        topic=cfg.topics.cmd, payload=b'{"cmd":"stop"}',
    ))
    adapter._on_message(client, None, SimpleNamespace(
        topic=cfg.topics.tts_ack, payload=b'{"event_id":"evt-2","state":"started"}',
    ))
    adapter.publish_tts({"event_id": "evt-2", "text": "hello"})

    assert command_events == []
    assert tts_ack_events == [{"event_id": "evt-2", "state": "started"}]
    assert client.published == [(cfg.topics.tts, {"event_id": "evt-2", "text": "hello"}, cfg.ack_qos, False)]


def test_sc171_voice_phone_tts_profile_routes_orchestrator_to_gateway(monkeypatch):
    monkeypatch.setenv("SYSTEM_CONFIG_PROFILE", "sc171_voice_phone_tts")
    cfg = load_global_config(str(ROOT / "configs" / "system_config.yaml"))

    assert cfg.profile == "sc171_voice_phone_tts"
    assert cfg.gateway.mqtt.accept_commands is True
    assert cfg.orchestrator.tts_event_out.transport == "uds"
    assert cfg.orchestrator.tts_event_out.ipc_socket_path == "/tmp/robot_stack/mobile_tts_event.sock"
    assert cfg.gateway.tts_event_in.transport == "uds"
    assert cfg.gateway.tts_event_in.ipc_socket_path == "/tmp/robot_stack/mobile_tts_event.sock"
    assert cfg.gateway.tts_playback_out.transport == "uds"
    assert cfg.gateway.tts_playback_out.ipc_socket_path == "/tmp/robot_stack/tts_playback.sock"
