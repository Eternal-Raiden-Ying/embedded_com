#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
ORCH_ROOT = ROOT / "orchestrator"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from orchestrator_service.ipc.protocol import TaskCmd  # noqa: E402
from orchestrator_service.mobile_gateway.config.board_config import build_config  # noqa: E402
from orchestrator_service.mobile_gateway.config.schema import MobileGatewayConfig  # noqa: E402
from orchestrator_service.mobile_gateway.protocol import ROBOT_ID  # noqa: E402
from orchestrator_service.mobile_gateway.runtime.service import MobileGatewayService  # noqa: E402


class _CaptureBackend:
    def __init__(self) -> None:
        self.payloads: List[Dict[str, Any]] = []

    def start(self, *args, **kwargs) -> None:
        return None

    def stop(self) -> None:
        return None

    def submit(self, payload: Dict[str, Any]):
        self.payloads.append(dict(payload))
        return True, "captured"


class _CaptureMqttAdapter:
    def __init__(self) -> None:
        self.acks: List[Dict[str, Any]] = []
        self.statuses: List[Dict[str, Any]] = []
        self.heartbeats: List[Dict[str, Any]] = []

    def publish_ack(self, payload: Dict[str, Any]) -> None:
        self.acks.append(dict(payload))

    def publish_status(self, payload: Dict[str, Any]) -> None:
        self.statuses.append(dict(payload))

    def publish_heartbeat(self, payload: Dict[str, Any]) -> None:
        self.heartbeats.append(dict(payload))


class RealProtocolMappingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._services: List[MobileGatewayService] = []

    def tearDown(self) -> None:
        for service in self._services:
            service.stop()

    def _make_service(self) -> MobileGatewayService:
        cfg = MobileGatewayConfig()
        cfg.backend.mode = "orchestrator_tcp"
        cfg.runtime.mode = "production"
        cfg.runtime.status_stdout = False
        cfg.status_out.transport = "disabled"
        cfg.orchestrator_task_ack_in.transport = "disabled"
        service = MobileGatewayService(cfg)
        service.backend = _CaptureBackend()
        self._services.append(service)
        return service

    def _capture_status(self, service: MobileGatewayService) -> List[Dict[str, Any]]:
        published: List[Dict[str, Any]] = []
        service._publish_status = lambda payload, force=False: published.append(dict(payload))
        return published

    def _capture_gateway_ack(self, service: MobileGatewayService) -> List[Dict[str, Any]]:
        published: List[Dict[str, Any]] = []
        service._publish_gateway_ack = lambda payload, accepted, message, error_code=None: published.append({
            "payload": dict(payload),
            "accepted": accepted,
            "message": message,
            "error_code": error_code,
        })
        return published

    def _assert_task_cmd_valid(
        self,
        service: MobileGatewayService,
        payload: Dict[str, Any],
        expected_intent: str,
        expected_target: str = "",
    ) -> None:
        parsed = TaskCmd.from_dict(payload, set(service.cfg.backend.default_robot_id and {"apple", "banana", "bottle", "orange"}))
        self.assertEqual(parsed.intent, expected_intent)
        if expected_target:
            self.assertEqual(parsed.target, expected_target)
        else:
            self.assertIsNone(parsed.target)

    def test_fetch_object_maps_to_find(self) -> None:
        service = self._make_service()
        gateway_ack = self._capture_gateway_ack(service)
        service._handle_command_payload({
            "cmd": "fetch_object",
            "target": "apple",
            "cmd_id": "cmd_fetch",
            "session_id": "sess_1",
            "ts": time.time(),
        })
        self.assertEqual(len(service.backend.payloads), 1)
        payload = service.backend.payloads[-1]
        self._assert_task_cmd_valid(service, payload, "FIND", "apple")
        self.assertEqual(payload["cmd_id"], "cmd_fetch")
        self.assertEqual(payload["source"], "wechat_miniprogram")
        self.assertEqual(gateway_ack[-1]["accepted"], True)
        self.assertEqual(gateway_ack[-1]["payload"]["cmd_id"], "cmd_fetch")

    def test_stop_maps_to_stop(self) -> None:
        service = self._make_service()
        gateway_ack = self._capture_gateway_ack(service)
        service._active_template = service._last_fetch_template = service._paused_template = None
        service._active_template = type("T", (), {"command": "fetch_object", "target": "apple", "session_id": "sess_2", "text": None})()
        service._handle_command_payload({"cmd": "stop", "cmd_id": "cmd_stop", "session_id": "sess_2", "ts": time.time()})
        payload = service.backend.payloads[-1]
        self._assert_task_cmd_valid(service, payload, "STOP")
        self.assertEqual(payload["source"], "wechat_miniprogram")
        self.assertEqual(gateway_ack[-1]["accepted"], True)

    def test_go_home_maps_to_return(self) -> None:
        service = self._make_service()
        service._handle_command_payload({"cmd": "go_home", "cmd_id": "cmd_home", "session_id": "sess_3", "ts": time.time()})
        payload = service.backend.payloads[-1]
        self._assert_task_cmd_valid(service, payload, "RETURN")

    def test_invalid_target_rejected(self) -> None:
        service = self._make_service()
        published = self._capture_status(service)
        gateway_ack = self._capture_gateway_ack(service)
        service._handle_command_payload({"cmd": "fetch_object", "target": "cup", "session_id": "sess_4", "ts": time.time()})
        self.assertEqual(service.backend.payloads, [])
        self.assertEqual(published[-1]["state"], "error")
        self.assertEqual(gateway_ack[-1]["accepted"], False)

    def test_invalid_cmd_rejected(self) -> None:
        service = self._make_service()
        published = self._capture_status(service)
        service._handle_command_payload({"cmd": "dance", "session_id": "sess_5", "ts": time.time()})
        self.assertEqual(service.backend.payloads, [])
        self.assertEqual(published[-1]["state"], "error")

    def test_stop_priority_bypasses_busy_guard(self) -> None:
        service = self._make_service()
        published = self._capture_status(service)
        service._snapshot["state"] = "searching"
        service._active_template = type("T", (), {"command": "fetch_object", "target": "apple", "session_id": "sess_busy", "text": None})()
        service._handle_command_payload({"cmd": "go_home", "session_id": "sess_busy", "ts": time.time()})
        self.assertEqual(service.backend.payloads, [])
        self.assertEqual(published[-1]["state"], "error")
        service._handle_command_payload({"cmd": "stop", "session_id": "sess_busy", "ts": time.time()})
        self.assertEqual(service.backend.payloads[-1]["intent"], "STOP")

    def test_duplicate_cmd_id_is_not_forwarded_twice(self) -> None:
        service = self._make_service()
        statuses = self._capture_status(service)
        gateway_ack = self._capture_gateway_ack(service)
        payload = {
            "cmd": "fetch_object",
            "target": "apple",
            "cmd_id": "cmd_dup_1",
            "session_id": "sess_dup",
            "ts": time.time(),
        }
        service._handle_command_payload(payload)
        service._handle_command_payload(payload)
        self.assertEqual(len(service.backend.payloads), 1)
        self.assertEqual(len(gateway_ack), 2)
        self.assertEqual(len(statuses), 1)

    def test_task_ack_maps_to_mobile_status(self) -> None:
        service = self._make_service()
        published = self._capture_status(service)
        mqtt = _CaptureMqttAdapter()
        service.mqtt_adapter = mqtt
        service._snapshot["command"] = "fetch_object"
        service._snapshot["target"] = "apple"
        service._handle_task_ack({
            "type": "task_ack",
            "cmd_id": "cmd_1",
            "session_id": "sess_ack",
            "epoch": 2,
            "accepted": True,
            "state": "SEARCH_TABLE",
            "reason": "FIND accepted",
        })
        self.assertEqual(published[-1]["state"], "accepted")
        self.assertEqual(published[-1]["kind"], "status")
        self.assertEqual(published[-1]["robot_id"], ROBOT_ID)
        self.assertEqual(mqtt.acks[-1]["kind"], "task_ack")
        self.assertEqual(mqtt.acks[-1]["message"], "FIND accepted: apple")
        self.assertNotIn("backend_state", published[-1])

    def test_state_block_maps_to_mobile_status(self) -> None:
        service = self._make_service()
        published = self._capture_status(service)
        service._snapshot["command"] = "fetch_object"
        service._handle_state_block({
            "state": "SEARCH_TABLE",
            "session_id": "sess_state",
            "epoch": 3,
            "active_target": "apple",
            "last_enter_reason": "开始桌边任务",
        })
        self.assertEqual(published[-1]["state"], "searching_table")
        self.assertEqual(published[-1]["target"], "apple")
        self.assertEqual(published[-1]["message"], "正在寻找桌边")
        self.assertEqual(published[-1]["backend_state"], "SEARCH_TABLE")
        service._handle_state_block({
            "state": "DONE",
            "session_id": "sess_state",
            "epoch": 3,
            "active_target": "apple",
            "last_enter_reason": "任务完成",
            "warnings": ["distance_too_far"],
        })
        self.assertEqual(published[-1]["state"], "idle")
        self.assertEqual(published[-1]["kind"], "status")
        self.assertEqual(published[-1]["message"], "任务完成，已锁定目标 apple")

    def test_vision_connect_failure_maps_to_error_status(self) -> None:
        service = self._make_service()
        published = self._capture_status(service)
        service._snapshot["command"] = "fetch_object"
        service._snapshot["target"] = "apple"
        service._handle_task_ack({
            "type": "task_ack",
            "cmd_id": "cmd_err",
            "session_id": "sess_err",
            "epoch": 4,
            "accepted": False,
            "state": "ERROR_RECOVERY",
            "reason": "vision_req_out connect_failed: Connection refused",
            "link_state": "DEGRADED",
        })
        self.assertEqual(published[-1]["state"], "error")
        self.assertEqual(published[-1]["error_code"], 1007)
        self.assertEqual(published[-1]["message"], "视觉模块未连接，任务暂时无法继续")

    def test_heartbeat_payload_uses_formal_kind(self) -> None:
        service = self._make_service()
        mqtt = _CaptureMqttAdapter()
        service.mqtt_adapter = mqtt
        service._last_heartbeat_emit_ts = 0.0
        service._emit_heartbeat_if_needed()
        self.assertTrue(mqtt.heartbeats)
        self.assertEqual(mqtt.heartbeats[-1]["kind"], "heartbeat")
        self.assertEqual(mqtt.heartbeats[-1]["robot_id"], ROBOT_ID)
        self.assertNotIn("recent_states", mqtt.heartbeats[-1])

    def test_production_mode_filters_mqtt_publish_noise(self) -> None:
        service = self._make_service()
        captured: List[Dict[str, Any]] = []
        service.log = lambda level, src, msg, data=None: captured.append({  # type: ignore[assignment]
            "level": level,
            "src": src,
            "msg": msg,
            "data": dict(data or {}),
        })
        service._log_mqtt_event("info", "mqtt_publish", {"topic": "robot/v1/SC171/heartbeat"})
        self.assertEqual(captured, [])

    def test_debug_mode_keeps_diagnostic_logs_and_fields(self) -> None:
        cfg = MobileGatewayConfig()
        cfg.backend.mode = "orchestrator_tcp"
        cfg.runtime.mode = "debug"
        cfg.runtime.enable_raw_mqtt_debug = True
        cfg.runtime.status_stdout = False
        cfg.status_out.transport = "disabled"
        cfg.orchestrator_task_ack_in.transport = "disabled"
        service = MobileGatewayService(cfg)
        service.backend = _CaptureBackend()
        self._services.append(service)

        captured: List[Dict[str, Any]] = []
        service.log = lambda level, src, msg, data=None: captured.append({  # type: ignore[assignment]
            "level": level,
            "src": src,
            "msg": msg,
            "data": dict(data or {}),
        })
        published = self._capture_status(service)

        service._log_mqtt_event("info", "mqtt_publish", {"topic": "robot/v1/SC171/heartbeat", "payload": {"kind": "heartbeat"}})
        self.assertTrue(captured)
        self.assertEqual(captured[-1]["msg"], "mqtt publish")

        service._handle_task_ack({
            "type": "task_ack",
            "cmd_id": "cmd_debug",
            "session_id": "sess_debug",
            "epoch": 5,
            "accepted": True,
            "state": "SEARCH_TABLE",
            "reason": "FIND accepted",
        })
        self.assertEqual(published[-1]["backend_state"], "SEARCH_TABLE")

    def test_orchestrator_tcp_config_enables_ack_listener(self) -> None:
        config_obj = {
            "robot_id": "SC171",
            "backend": "orchestrator_tcp",
            "runtime": {
                "mode": "debug",
                "heartbeat_log_interval_s": 12,
                "suppress_heartbeat_success_log": False,
                "enable_raw_mqtt_debug": True,
                "enable_legacy_command_compat": False,
                "cmd_dedup_cache_size": 32,
            },
            "orchestrator": {
                "task_cmd_host": "127.0.0.1",
                "task_cmd_port": 9001,
                "task_ack_host": "127.0.0.1",
                "task_ack_port": 9012,
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fp:
            json.dump(config_obj, fp)
            path = fp.name
        cfg = build_config(config_file=path)
        self.assertEqual(cfg.orchestrator_task_ack_in.transport, "tcp")
        self.assertEqual(cfg.orchestrator_task_ack_in.host, "127.0.0.1")
        self.assertEqual(cfg.orchestrator_task_ack_in.port, 9012)
        self.assertEqual(cfg.runtime.mode, "debug")
        self.assertEqual(cfg.runtime.heartbeat_log_interval_s, 12.0)
        self.assertEqual(cfg.runtime.enable_raw_mqtt_debug, True)
        self.assertEqual(cfg.runtime.enable_legacy_command_compat, False)
        self.assertEqual(cfg.runtime.cmd_dedup_cache_size, 32)


if __name__ == "__main__":
    unittest.main()
