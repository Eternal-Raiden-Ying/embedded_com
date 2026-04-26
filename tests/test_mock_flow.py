#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import unittest
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
ORCH_ROOT = ROOT / "orchestrator"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from orchestrator_service.mobile_gateway.config.schema import MobileGatewayConfig  # noqa: E402
from orchestrator_service.mobile_gateway.runtime.service import MobileGatewayService  # noqa: E402


def wait_for_state(records: List[Dict[str, object]], expected: str, timeout_s: float = 4.0) -> Dict[str, object]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for payload in reversed(records):
            if str(payload.get("state")) == expected:
                return payload
        time.sleep(0.05)
    raise AssertionError(f"state {expected!r} not observed; got={records}")


class MockFlowTest(unittest.TestCase):
    def test_fetch_then_stop_flow(self) -> None:
        cfg = MobileGatewayConfig()
        cfg.backend.mode = "mock"
        cfg.runtime.status_stdout = False
        cfg.backend.mock_step_interval_s = 0.10
        cfg.status_out.transport = "disabled"

        service = MobileGatewayService(cfg)
        published: List[Dict[str, object]] = []
        original_publish = service._publish_status

        def _capture(payload: Dict[str, object], force: bool = False) -> None:
            published.append(dict(payload))
            original_publish(payload, force=force)

        service._publish_status = _capture
        try:
            service.backend.start(service._handle_backend_event)
            service._handle_command_payload({
                "cmd": "fetch_object",
                "target": "apple",
                "session_id": "sess_demo",
                "robot_id": "robot_demo",
                "ts": time.time(),
            })
            searching = wait_for_state(published, "searching")
            self.assertEqual(searching["target"], "apple")

            service._handle_command_payload({
                "cmd": "stop",
                "session_id": "sess_demo",
                "robot_id": "robot_demo",
                "ts": time.time(),
            })
            stopped = wait_for_state(published, "stopped")
            self.assertEqual(stopped["session_id"], "sess_demo")
        finally:
            service.stop()


if __name__ == "__main__":
    unittest.main()
