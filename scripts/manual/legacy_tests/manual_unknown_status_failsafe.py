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

from orchestrator_service.config.schema import OrchestratorConfig
from orchestrator_service.runtime.context import State
from orchestrator_service.runtime.state_machine import OrchestratorCore
from orchestrator_service.ipc.protocol import TableEdgeObs, TargetObs

class UnknownStatusFailsafeTest(unittest.TestCase):
    def setUp(self):
        self.cfg = OrchestratorConfig()
        self.core = OrchestratorCore(self.cfg.control, self.cfg.car, self.cfg.docking)

    def test_unknown_table_obs_status_failsafe(self):
        self.core.ctx.state = State.SEARCH_TABLE
        obs = TableEdgeObs(ts=123.0, table_found=True, edge_found=True, vision_status="INVALID_STATUS")
        self.core.handle_table_obs(obs)
        self.assertEqual(self.core.ctx.state, State.ERROR_RECOVERY)

    def test_unknown_target_obs_status_failsafe(self):
        self.core.ctx.state = State.EDGE_SLIDE_SEARCH
        obs = TargetObs(ts=123.0, found=True, vision_status="DODGY_STATUS")
        self.core.handle_target_obs(obs)
        self.assertEqual(self.core.ctx.state, State.ERROR_RECOVERY)

    def test_unknown_grasp_obs_status_failsafe(self):
        self.core.ctx.state = State.GRASP
        self.core.handle_grasp_obs({"status": "STRANGE_STATUS"})
        self.assertEqual(self.core.ctx.state, State.ERROR_RECOVERY)

if __name__ == "__main__":
    unittest.main()
