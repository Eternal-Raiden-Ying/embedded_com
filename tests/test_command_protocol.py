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

from orchestrator_service.mobile_gateway.protocol import (  # noqa: E402
    ERROR_CODES,
    MobileCommand,
    MobileProtocolError,
)


class CommandProtocolTest(unittest.TestCase):
    def test_fetch_object_requires_supported_target(self) -> None:
        with self.assertRaises(MobileProtocolError) as ctx:
            MobileCommand.from_dict(
                {"cmd": "fetch_object", "target": "orange"},
                default_robot_id="robot_01",
            )
        self.assertEqual(ctx.exception.error_code, ERROR_CODES["invalid_target"])

    def test_fetch_object_defaults_robot_and_session(self) -> None:
        cmd = MobileCommand.from_dict(
            {"cmd": "fetch_object", "target": "apple"},
            default_robot_id="robot_01",
        )
        self.assertEqual(cmd.robot_id, "robot_01")
        self.assertEqual(cmd.target, "apple")
        self.assertTrue(cmd.session_id.startswith("sess_"))

    def test_query_status_does_not_require_target(self) -> None:
        cmd = MobileCommand.from_dict(
            {"cmd": "query_status", "session_id": "sess_a"},
            default_robot_id="robot_01",
        )
        self.assertEqual(cmd.cmd, "query_status")
        self.assertIsNone(cmd.target)


if __name__ == "__main__":
    unittest.main()

