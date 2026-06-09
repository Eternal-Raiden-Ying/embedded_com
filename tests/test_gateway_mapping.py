#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORCH_ROOT = ROOT / "orchestrator"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from orchestrator_service.mobile_gateway.config.schema import MobileGatewayConfig  # noqa: E402
from orchestrator_service.mobile_gateway.runtime.service import MobileGatewayService  # noqa: E402


class GatewayMappingTest(unittest.TestCase):
    def _make_service(self) -> MobileGatewayService:
        cfg = MobileGatewayConfig()
        cfg.backend.mode = "mock"
        cfg.runtime.status_stdout = False
        cfg.status_out.transport = "disabled"
        cfg.command_in.ipc_socket_path = "/tmp/robot_stack/test_gateway_mapping_cmd.sock"
        return MobileGatewayService(cfg)

    def test_stop_promotes_paused_template(self) -> None:
        service = self._make_service()
        try:
            service.backend.start(service._handle_backend_event)
            service._active_template = service._last_fetch_template = service._paused_template = None
            service._handle_command_payload({"cmd": "fetch_object", "target": "apple", "session_id": "sess_fetch"})
            self.assertIsNotNone(service._active_template)
            service._handle_command_payload({"cmd": "stop", "session_id": "sess_fetch"})
            self.assertIsNotNone(service._paused_template)
            self.assertEqual(service._paused_template.command, "fetch_object")
        finally:
            service.stop()

    def test_resume_without_paused_task_returns_error(self) -> None:
        service = self._make_service()
        try:
            published = []
            service._publish_status = lambda payload, force=False: published.append(dict(payload))
            service._handle_command_payload({"cmd": "resume", "session_id": "sess_none"})
            self.assertTrue(published)
            self.assertEqual(published[-1]["state"], "error")
        finally:
            service.stop()


if __name__ == "__main__":
    unittest.main()
