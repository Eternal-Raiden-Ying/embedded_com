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
from orchestrator_service.ipc.protocol import TableEdgeObs, TargetObs, now_ts


class VisionStateSyncTest(unittest.TestCase):
    def setUp(self):
        self.cfg = OrchestratorConfig()
        self.core = OrchestratorCore(self.cfg.control, self.cfg.car, self.cfg.docking)

    def test_property_backward_compatibility(self):
        """Verify active_vision_stage/mode properties point to confirmed stage/mode."""
        self.core.ctx.confirmed_vision_stage = "STAGE_A"
        self.core.ctx.confirmed_vision_mode = "MODE_A"

        self.assertEqual(self.core.ctx.active_vision_stage, "STAGE_A")
        self.assertEqual(self.core.ctx.active_vision_mode, "MODE_A")

        self.core.ctx.active_vision_stage = "STAGE_B"
        self.core.ctx.active_vision_mode = "MODE_B"

        self.assertEqual(self.core.ctx.confirmed_vision_stage, "STAGE_B")
        self.assertEqual(self.core.ctx.confirmed_vision_mode, "MODE_B")

    def test_mode_request_failure_and_retry(self):
        """Verify failure to send mode_request rolls back desired state and allows retry."""
        # Start in IDLE
        self.assertEqual(self.core.ctx.state, State.IDLE)
        self.assertEqual(self.core.ctx.confirmed_vision_stage, "")
        self.assertEqual(self.core.ctx.confirmed_vision_mode, "")

        # Set state to SEARCH_TABLE which binds to SEARCH/FIND_EDGE
        self.core.ctx.state = State.SEARCH_TABLE

        # Tick 1: active request is generated because prev (confirmed) is empty, next is SEARCH/FIND_EDGE
        req = self.core._active_req_payload()
        self.assertIsNotNone(req)
        self.assertEqual(req["req_type"], "mode_request")
        self.assertEqual(req["op"], "START")
        self.assertEqual(req["stage"], "SEARCH")
        self.assertEqual(req["mode_hint"], "FIND_EDGE")

        # In _active_req_payload, desired was updated but confirmed was not
        self.assertEqual(self.core.ctx.desired_vision_stage, "SEARCH")
        self.assertEqual(self.core.ctx.desired_vision_mode, "FIND_EDGE")
        self.assertEqual(self.core.ctx.confirmed_vision_stage, "")
        self.assertEqual(self.core.ctx.confirmed_vision_mode, "")

        # Queue the request
        self.core._queue_vision_req(req)
        self.assertEqual(len(self.core.ctx.pending_vision_msgs), 1)

        # Simulate send failure
        self.core.handle_vision_req_send_result(False, req, error="mock error")

        # After failure, desired rolls back to confirmed (which was empty)
        self.assertEqual(self.core.ctx.desired_vision_stage, "")
        self.assertEqual(self.core.ctx.desired_vision_mode, "")
        self.assertEqual(self.core.ctx.confirmed_vision_stage, "")
        self.assertEqual(self.core.ctx.confirmed_vision_mode, "")

        # Since it failed, _last_mode_request_key was not set, allowing subsequent tick to regenerate and send again
        self.core.ctx.pending_vision_msgs.clear()
        req2 = self.core._active_req_payload()
        self.assertIsNotNone(req2)
        self.assertEqual(req2["req_type"], "mode_request") # Still a mode_request!
        self.assertEqual(req2["stage"], "SEARCH")
        self.assertEqual(req2["mode_hint"], "FIND_EDGE")

        self.core._queue_vision_req(req2)
        self.assertEqual(len(self.core.ctx.pending_vision_msgs), 1)

    def test_mode_request_success_and_subsequent_updates(self):
        """Verify successful mode_request send updates confirmed, and subsequent are target_update."""
        self.core.ctx.state = State.SEARCH_TABLE

        # Generate mode request
        req = self.core._active_req_payload()
        self.assertEqual(req["req_type"], "mode_request")
        self.core._queue_vision_req(req)

        # Simulate send success
        self.core.handle_vision_req_send_result(True, req)

        # confirmed transitions to new mode, and desired remains there
        self.assertEqual(self.core.ctx.confirmed_vision_stage, "SEARCH")
        self.assertEqual(self.core.ctx.confirmed_vision_mode, "FIND_EDGE")
        self.assertEqual(self.core.ctx.desired_vision_stage, "SEARCH")
        self.assertEqual(self.core.ctx.desired_vision_mode, "FIND_EDGE")

        # Subsequent requests generated should be target_update, not mode_request
        req2 = self.core._active_req_payload()
        self.assertIsNotNone(req2)
        self.assertEqual(req2["req_type"], "target_update")
        self.assertEqual(req2["op"], "UPDATE")

    def test_stale_observation_protection(self):
        """Verify stale old observations do not overwrite or roll back confirmed mode."""
        self.core.ctx.state = State.SEARCH_TABLE
        
        # We start by sending mode request for FIND_EDGE, and it succeeds
        req = self.core._active_req_payload()
        self.core.handle_vision_req_send_result(True, req)
        self.assertEqual(self.core.ctx.confirmed_vision_mode, "FIND_EDGE")

        # Now, we transit state to EDGE_SLIDE_SEARCH which binds to SEARCH/FIND_OBJECT (desired becomes FIND_OBJECT)
        self.core.ctx.state = State.EDGE_SLIDE_SEARCH
        req2 = self.core._active_req_payload()
        self.assertEqual(self.core.ctx.desired_vision_mode, "FIND_OBJECT")
        self.assertEqual(self.core.ctx.confirmed_vision_mode, "FIND_EDGE")

        # Before req2 is confirmed, a stale observation from VISTA with source_mode="FIND_EDGE" arrives
        obs = TableEdgeObs(ts=now_ts(), table_found=True, edge_found=True, source_mode="FIND_EDGE")
        
        # Calling handle_table_obs or confirm_vision_state with this stale obs should ignore it, not roll anything back
        confirmed = self.core.confirm_vision_state("SEARCH", obs.source_mode, source="vision_obs")
        self.assertFalse(confirmed)
        
        self.assertEqual(self.core.ctx.desired_vision_mode, "FIND_OBJECT")
        self.assertEqual(self.core.ctx.confirmed_vision_mode, "FIND_EDGE")

        # Now confirm the new desired mode FIND_OBJECT via ack/send success
        self.core.handle_vision_req_send_result(True, req2)
        self.assertEqual(self.core.ctx.confirmed_vision_mode, "FIND_OBJECT")


if __name__ == "__main__":
    unittest.main()
