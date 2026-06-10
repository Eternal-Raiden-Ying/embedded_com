#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import unittest
from unittest.mock import MagicMock

try:
    from .test_support import build_test_config
except ImportError:
    from test_support import build_test_config

from vision_module.app.app import VistaApp
from vision_module.app.stages.base import StageContext, StageOutput, build_vision_obs


class MockSender:
    def __init__(self, name):
        self.name = name
        self.sent_payloads = []
        self.should_succeed = True

    def send(self, payload):
        if self.should_succeed:
            self.sent_payloads.append(payload)
            return True
        return False

    def close(self):
        pass


class TestObservationSending(unittest.TestCase):
    def setUp(self):
        self.app = VistaApp()
        self.app.obs_sender = MockSender("obs_out")
        self.app.diag_sender = MockSender("obs_diag")
        self.ctx = StageContext(current_stage="SEARCH", current_mode="FIND_OBJECT")
        self.ctx.session_id = "test_sess"
        self.ctx.req_id = "test_req"
        self.ctx.epoch = 1

    def test_obs_splitting_and_latency_injection(self):
        # 1. Prepare stage output with table_edge_obs, target_obs and local_perception
        table_edge_obs = {
            "type": "table_edge_obs",
            "edge_found": True,
            "confidence": 0.9,
            "dist_err_m": 0.05,
            "yaw_err_rad": 0.01,
            "frame_id": 42,
            "obs_ts": time.time() - 0.05,  # 50ms ago
        }
        target_obs = {
            "type": "target_obs",
            "target": "bottle",
            "found": True,
            "obs_ts": time.time() - 0.05,
        }
        local_perception = {
            "box_count": 2,
            "infer_boxes": [[10, 20, 30, 40, 0.9, 0]],
            "yolo_infer_ms": 35.0,
        }

        perception = {
            "table_edge_obs": table_edge_obs,
            "target_obs": target_obs,
            "local_perception": local_perception,
        }

        raw_obs = build_vision_obs(
            self.ctx,
            status="RUNNING",
            perception=perception,
            proposal={"test_prop": 1},
            result={"test_res": 2},
        )

        output = StageOutput(vision_obs=raw_obs)

        # Mock scheduler.read_result to return custom frame_meta
        self.app.scheduler.read_result = MagicMock(return_value={
            "frame_seq": 42,
            "frame_capture_ts": time.time() - 0.05,
            "camera_frame_ts_ms": int((time.time() - 0.05) * 1000.0)
        })

        # 2. Run _apply_stage_output
        now = time.time()
        queued = self.app._apply_stage_output(output, now=now, force_send=True)

        self.assertTrue(queued)
        self.assertEqual(len(self.app.obs_sender.sent_payloads), 1)
        self.assertEqual(len(self.app.diag_sender.sent_payloads), 1)

        control_obs = self.app.obs_sender.sent_payloads[0]
        diag_obs = self.app.diag_sender.sent_payloads[0]

        # 3. Check Control Obs Properties
        self.assertEqual(control_obs["obs_class"], "control")
        self.assertIn("table_edge_obs", control_obs["perception"])
        self.assertIn("target_obs", control_obs["perception"])
        # Should NOT contain local_perception in control obs
        self.assertNotIn("local_perception", control_obs["perception"])
        self.assertNotIn("proposal", control_obs)
        self.assertNotIn("result", control_obs)

        # Check Latency Injection at top-level
        self.assertEqual(control_obs["frame_id"], 42)
        self.assertGreater(control_obs["process_latency_ms"], 0.0)
        self.assertGreater(control_obs["obs_total_age_ms"], 0.0)

        # Check Latency Injection inside children
        control_edge = control_obs["perception"]["table_edge_obs"]
        self.assertEqual(control_edge["frame_id"], 42)
        self.assertEqual(control_edge["camera_frame_seq"], 42)
        self.assertGreater(control_edge["process_latency_ms"], 0.0)
        self.assertGreater(control_edge["obs_total_age_ms"], 0.0)

        # 4. Check Diagnostic Obs Properties
        self.assertEqual(diag_obs["obs_class"], "diagnostic")
        self.assertIn("local_perception", diag_obs["perception"])
        self.assertNotIn("table_edge_obs", diag_obs["perception"])
        self.assertNotIn("target_obs", diag_obs["perception"])
        self.assertIn("proposal", diag_obs)
        self.assertIn("result", diag_obs)

    def test_same_frame_reuse_and_metrics(self):
        # Trigger sending control obs twice with same frame ID
        table_edge_obs = {
            "type": "table_edge_obs",
            "frame_id": 100,
            "obs_ts": time.time(),
        }
        raw_obs = build_vision_obs(self.ctx, status="RUNNING", perception={"table_edge_obs": table_edge_obs})
        output = StageOutput(vision_obs=raw_obs)

        self.app.scheduler.read_result = MagicMock(return_value={
            "frame_seq": 100,
            "frame_capture_ts": time.time()
        })

        # First Send
        self.app._apply_stage_output(output, now=time.time(), force_send=True)
        self.assertEqual(self.app.same_frame_reuse_count, 0)

        # Second Send (same frame ID)
        self.app._apply_stage_output(output, now=time.time(), force_send=True)
        self.assertEqual(self.app.same_frame_reuse_count, 1)

    def test_rate_limiting_diagnostics(self):
        table_edge_obs = {"type": "table_edge_obs", "frame_id": 200, "obs_ts": time.time()}
        raw_obs = build_vision_obs(self.ctx, status="RUNNING", perception={"table_edge_obs": table_edge_obs})
        output = StageOutput(vision_obs=raw_obs)

        self.app.scheduler.read_result = MagicMock(return_value={
            "frame_seq": 200,
            "frame_capture_ts": time.time()
        })

        # Send once: both control and diagnostic sent
        self.app._apply_stage_output(output, now=time.time(), force_send=True)
        self.assertEqual(len(self.app.obs_sender.sent_payloads), 1)
        self.assertEqual(len(self.app.diag_sender.sent_payloads), 1)

        # Send again immediately with force_send=False: control is subject to interval (skips or sends),
        # but diagnostic is rate limited to 1.0s interval
        self.app.last_send_ts = 0.0  # bypass control send interval
        self.app._apply_stage_output(output, now=time.time(), force_send=False)
        self.assertEqual(len(self.app.obs_sender.sent_payloads), 2)
        # Diagnostic should NOT have been sent again because 1.0s hasn't passed
        self.assertEqual(len(self.app.diag_sender.sent_payloads), 1)

    def test_drop_tracking_on_failure(self):
        # Mock sender to fail enqueuing
        self.app.obs_sender.should_succeed = False
        table_edge_obs = {"type": "table_edge_obs", "frame_id": 300, "obs_ts": time.time()}
        raw_obs = build_vision_obs(self.ctx, status="RUNNING", perception={"table_edge_obs": table_edge_obs})
        output = StageOutput(vision_obs=raw_obs)

        self.app.scheduler.read_result = MagicMock(return_value={
            "frame_seq": 300,
            "frame_capture_ts": time.time()
        })

        queued = self.app._apply_stage_output(output, now=time.time(), force_send=True)
        self.assertFalse(queued)
        self.assertEqual(self.app.obs_drop_count, 1)


if __name__ == "__main__":
    unittest.main()
