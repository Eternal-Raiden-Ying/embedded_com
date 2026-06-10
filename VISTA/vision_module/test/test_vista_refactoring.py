#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from vision_module.ipc.protocol import VisionReq
from vision_module.app.stage_controller import StageController
from vision_module.app.stages.search import SearchStagePlan

class VistaRefactoringTest(unittest.TestCase):
    def setUp(self):
        self.stage_controller = StageController()
        self.search_plan = SearchStagePlan()
        self.stage_controller.register_plan(self.search_plan)

    def test_legacy_mode_aliases_mapping(self):
        # 1. TRACK_LOCAL maps to FIND_OBJECT
        req1 = VisionReq(ts=123.0, op="START", stage="SEARCH", mode_hint="TRACK_LOCAL")
        self.assertEqual(req1.mode_hint, "FIND_OBJECT")

        # 2. DEPTH_PERCEPTION maps to FIND_EDGE
        req2 = VisionReq(ts=123.0, op="START", stage="SEARCH", mode_hint="DEPTH_PERCEPTION")
        self.assertEqual(req2.mode_hint, "FIND_EDGE")

        # 3. TABLE_EDGE_PERCEPTION maps to FIND_EDGE
        req3 = VisionReq(ts=123.0, op="START", stage="SEARCH", mode_hint="TABLE_EDGE_PERCEPTION")
        self.assertEqual(req3.mode_hint, "FIND_EDGE")

    def test_invalid_search_kind_returns_failed_with_reason(self):
        # Create search req with invalid search_kind
        req = VisionReq(
            ts=123.0,
            op="START",
            stage="SEARCH",
            mode_hint="FIND_OBJECT",
            payload={"search_kind": "INVALID_KIND"},
        )
        # Calling handle_request should route to SEARCH and on_enter should return StageOutput with status="FAILED" and result with reason
        output = self.stage_controller.handle_request(req)
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.vision_obs)
        self.assertEqual(output.vision_obs.get("status"), "FAILED")
        self.assertIn("reason", output.vision_obs.get("result", {}))
        self.assertIn("invalid search_kind", output.vision_obs["result"]["reason"])

    def test_activate_stage_returns_tuple_and_compatibility(self):
        # activate_stage should return a tuple (plan, on_enter_output)
        # 1. Successful transition
        plan, on_enter_output = self.stage_controller.activate_stage("SEARCH")
        self.assertIs(plan, self.search_plan)
        self.assertIsNone(on_enter_output)

        # 2. Transition that returns an enter output (invalid request)
        req = VisionReq(
            ts=123.0,
            op="START",
            stage="SEARCH",
            payload={"search_kind": "INVALID_KIND"},
        )
        plan, on_enter_output = self.stage_controller.activate_stage("SEARCH", req=req)
        self.assertIsNone(plan)
        self.assertIsNotNone(on_enter_output)
        self.assertEqual(on_enter_output.vision_obs.get("status"), "FAILED")

if __name__ == "__main__":
    unittest.main()
