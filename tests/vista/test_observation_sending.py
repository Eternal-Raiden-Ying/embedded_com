#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import unittest
from unittest.mock import MagicMock

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
        self.app.scheduler.read_result = MagicMock(
            return_value={
                "frame_seq": 42,
                "frame_capture_ts": time.time() - 0.05,
                "camera_frame_ts_ms": int((time.time() - 0.05) * 1000.0),
            }
        )

    def _running_output(self, frame_id=42):
        raw_obs = build_vision_obs(
            self.ctx,
            status="RUNNING",
            perception={
                "table_edge_obs": {
                    "type": "table_edge_obs",
                    "edge_found": True,
                    "confidence": 0.9,
                    "dist_err_m": 0.05,
                    "yaw_err_rad": 0.01,
                    "frame_id": frame_id,
                    "obs_ts": time.time() - 0.05,
                },
                "target_obs": {
                    "type": "target_obs",
                    "target": "bottle",
                    "found": True,
                    "obs_ts": time.time() - 0.05,
                },
                "local_perception": {
                    "box_count": 2,
                    "infer_boxes": [[10, 20, 30, 40, 0.9, 0]],
                    "yolo_infer_ms": 35.0,
                },
            },
            proposal={"test_prop": 1},
            result={"test_res": 2},
        )
        return StageOutput(vision_obs=raw_obs)

    def test_obs_splitting_and_latency_injection(self):
        output = self._running_output(frame_id=42)

        queued = self.app._apply_stage_output(output, now=time.time(), force_send=False)

        self.assertTrue(queued)
        self.assertEqual(len(self.app.obs_sender.sent_payloads), 1)
        self.assertEqual(len(self.app.diag_sender.sent_payloads), 1)

        control_obs = self.app.obs_sender.sent_payloads[0]
        diag_obs = self.app.diag_sender.sent_payloads[0]

        self.assertEqual(control_obs["obs_class"], "control")
        self.assertIn("table_edge_obs", control_obs["perception"])
        self.assertIn("target_obs", control_obs["perception"])
        self.assertNotIn("local_perception", control_obs["perception"])
        self.assertNotIn("proposal", control_obs)
        self.assertNotIn("result", control_obs)

        self.assertEqual(control_obs["frame_id"], 42)
        self.assertGreater(control_obs["process_latency_ms"], 0.0)
        self.assertGreater(control_obs["obs_total_age_ms"], 0.0)

        control_edge = control_obs["perception"]["table_edge_obs"]
        self.assertEqual(control_edge["frame_id"], 42)
        self.assertEqual(control_edge["camera_frame_seq"], 42)
        self.assertGreater(control_edge["process_latency_ms"], 0.0)
        self.assertGreater(control_edge["obs_total_age_ms"], 0.0)

        self.assertEqual(diag_obs["obs_class"], "diagnostic")
        self.assertIn("local_perception", diag_obs["perception"])
        self.assertNotIn("table_edge_obs", diag_obs["perception"])
        self.assertNotIn("target_obs", diag_obs["perception"])
        self.assertIn("proposal", diag_obs)
        self.assertIn("result", diag_obs)

    def test_running_diagnostic_is_limited_to_one_hz_while_control_runs_fast(self):
        output = self._running_output(frame_id=200)
        start = time.time()

        for idx in range(10):
            self.app.scheduler.read_result = MagicMock(
                return_value={"frame_seq": 200 + idx, "frame_capture_ts": start + idx * 0.11 - 0.05}
            )
            self.app._apply_stage_output(output, now=start + idx * 0.11, force_send=False)

        self.assertGreaterEqual(len(self.app.obs_sender.sent_payloads), 8)
        self.assertEqual(len(self.app.diag_sender.sent_payloads), 1)

        self.app.scheduler.read_result = MagicMock(
            return_value={"frame_seq": 300, "frame_capture_ts": start + 1.10 - 0.05}
        )
        self.app._apply_stage_output(output, now=start + 1.10, force_send=False)
        self.assertEqual(len(self.app.diag_sender.sent_payloads), 2)

    def test_same_frame_reuse_and_metrics_without_force(self):
        output = self._running_output(frame_id=100)
        now = time.time()
        self.app.scheduler.read_result = MagicMock(return_value={"frame_seq": 100, "frame_capture_ts": now - 0.05})

        self.app._apply_stage_output(output, now=now, force_send=False)
        self.assertEqual(self.app.same_frame_reuse_count, 0)

        self.app._apply_stage_output(output, now=now + 0.11, force_send=False)
        self.assertEqual(self.app.same_frame_reuse_count, 1)

    def test_force_send_predicate_is_false_for_running_and_true_for_urgent(self):
        running_output = self._running_output(frame_id=400)
        self.assertFalse(self.app._should_force_send_stage_output(running_output))

        failed_obs = build_vision_obs(self.ctx, status="FAILED", perception={})
        failed_output = StageOutput(vision_obs=failed_obs)
        self.assertTrue(self.app._should_force_send_stage_output(failed_output))

        urgent_obs = build_vision_obs(self.ctx, status="RUNNING", perception={})
        urgent_output = StageOutput(vision_obs=urgent_obs, signals={"urgent": True})
        self.assertTrue(self.app._should_force_send_stage_output(urgent_output))

        self.app.last_send_ts = time.time()
        queued = self.app._apply_stage_output(urgent_output, now=time.time(), force_send=True)
        self.assertTrue(queued)

    def test_skip_and_drop_tracking(self):
        output = self._running_output(frame_id=500)
        now = time.time()
        self.app.scheduler.read_result = MagicMock(return_value={"frame_seq": 500, "frame_capture_ts": now - 0.05})

        self.assertTrue(self.app._apply_stage_output(output, now=now, force_send=False))
        self.assertFalse(self.app._apply_stage_output(output, now=now + 0.01, force_send=False))
        self.assertEqual(self.app.obs_skip_count, 1)

        self.app.obs_sender.should_succeed = False
        queued = self.app._apply_stage_output(output, now=now + 0.20, force_send=False)
        self.assertFalse(queued)
        self.assertEqual(self.app.obs_drop_count, 1)

    def test_control_send_interval_targets_ten_hz(self):
        interval = self.app._control_send_interval_s()
        self.assertLessEqual(interval, 0.125)
        self.assertLessEqual(interval, 0.10)


if __name__ == "__main__":
    unittest.main()
