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
    ROBOT_ID,
)
from orchestrator_service.utils.target_utils import target_to_class_id  # noqa: E402


class CommandProtocolTest(unittest.TestCase):
    def test_fetch_object_requires_supported_target(self) -> None:
        with self.assertRaises(MobileProtocolError) as ctx:
            MobileCommand.from_dict(
                {"cmd": "fetch_object", "target": "cup"},
                default_robot_id=ROBOT_ID,
            )
        self.assertEqual(ctx.exception.error_code, ERROR_CODES["invalid_target"])

    def test_fetch_object_accepts_finetune_target(self) -> None:
        cmd = MobileCommand.from_dict(
            {"cmd": "fetch_object", "target": "orange"},
            default_robot_id=ROBOT_ID,
        )
        self.assertEqual(cmd.cmd, "fetch_object")
        self.assertEqual(cmd.target, "orange")

    def test_finetune_target_class_ids(self) -> None:
        self.assertEqual(target_to_class_id("apple"), 1)
        self.assertEqual(target_to_class_id("bottle"), 4)
        self.assertEqual(target_to_class_id("keys"), 6)
        self.assertEqual(target_to_class_id("kiwi"), 7)
        self.assertEqual(target_to_class_id("starfruit"), 13)

    def test_fetch_object_defaults_robot_and_session(self) -> None:
        cmd = MobileCommand.from_dict(
            {"cmd": "fetch_object", "target": "apple"},
            default_robot_id=ROBOT_ID,
        )
        self.assertEqual(cmd.robot_id, ROBOT_ID)
        self.assertEqual(cmd.target, "apple")
        self.assertTrue(cmd.session_id.startswith("sess_"))
        self.assertTrue(cmd.cmd_id.startswith("cmd_"))

    def test_query_status_does_not_require_target(self) -> None:
        cmd = MobileCommand.from_dict(
            {"cmd": "query_status", "session_id": "sess_a"},
            default_robot_id=ROBOT_ID,
        )
        self.assertEqual(cmd.cmd, "query_status")
        self.assertIsNone(cmd.target)

    def test_legacy_find_and_pick_is_accepted_for_compatibility(self) -> None:
        cmd = MobileCommand.from_dict(
            {"type": "FIND_AND_PICK", "target": "apple"},
            default_robot_id=ROBOT_ID,
        )
        self.assertEqual(cmd.cmd, "fetch_object")
        self.assertEqual(cmd.robot_id, ROBOT_ID)

    def test_legacy_find_and_pick_can_be_disabled(self) -> None:
        with self.assertRaises(MobileProtocolError) as ctx:
            MobileCommand.from_dict(
                {"type": "FIND_AND_PICK", "target": "apple"},
                default_robot_id=ROBOT_ID,
                allow_legacy_command_compat=False,
            )
        self.assertEqual(ctx.exception.error_code, ERROR_CODES["invalid_command"])


if __name__ == "__main__":
    unittest.main()
