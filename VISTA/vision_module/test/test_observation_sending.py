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
        # 1. 准备包含 table_edge_obs、target_obs 和 local_perception 的阶段输出
        table_edge_obs = {
            "type": "table_edge_obs",
            "edge_found": True,
            "confidence": 0.9,
            "dist_err_m": 0.05,
            "yaw_err_rad": 0.01,
            "frame_id": 42,
            "obs_ts": time.time() - 0.05,  # 50 毫秒前
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

        # 模拟 scheduler.read_result 返回自定义的 frame_meta
        self.app.scheduler.read_result = MagicMock(return_value={
            "frame_seq": 42,
            "frame_capture_ts": time.time() - 0.05,
            "camera_frame_ts_ms": int((time.time() - 0.05) * 1000.0)
        })

        # 2. 运行 _apply_stage_output
        now = time.time()
        queued = self.app._apply_stage_output(output, now=now, force_send=True)

        self.assertTrue(queued)
        self.assertEqual(len(self.app.obs_sender.sent_payloads), 1)
        self.assertEqual(len(self.app.diag_sender.sent_payloads), 1)

        control_obs = self.app.obs_sender.sent_payloads[0]
        diag_obs = self.app.diag_sender.sent_payloads[0]

        # 3. 检查控制观测属性
        self.assertEqual(control_obs["obs_class"], "control")
        self.assertIn("table_edge_obs", control_obs["perception"])
        self.assertIn("target_obs", control_obs["perception"])
        # 在控制观测中不应包含 local_perception
        self.assertNotIn("local_perception", control_obs["perception"])
        self.assertNotIn("proposal", control_obs)
        self.assertNotIn("result", control_obs)

        # 检查顶层延迟注入
        self.assertEqual(control_obs["frame_id"], 42)
        self.assertGreater(control_obs["process_latency_ms"], 0.0)
        self.assertGreater(control_obs["obs_total_age_ms"], 0.0)

        # 检查子观测内部延迟注入
        control_edge = control_obs["perception"]["table_edge_obs"]
        self.assertEqual(control_edge["frame_id"], 42)
        self.assertEqual(control_edge["camera_frame_seq"], 42)
        self.assertGreater(control_edge["process_latency_ms"], 0.0)
        self.assertGreater(control_edge["obs_total_age_ms"], 0.0)

        # 4. 检查诊断观测属性
        self.assertEqual(diag_obs["obs_class"], "diagnostic")
        self.assertIn("local_perception", diag_obs["perception"])
        self.assertNotIn("table_edge_obs", diag_obs["perception"])
        self.assertNotIn("target_obs", diag_obs["perception"])
        self.assertIn("proposal", diag_obs)
        self.assertIn("result", diag_obs)

    def test_same_frame_reuse_and_metrics(self):
        # 触发使用相同帧 ID 发送两次控制观测
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

        # 首次发送
        self.app._apply_stage_output(output, now=time.time(), force_send=True)
        self.assertEqual(self.app.same_frame_reuse_count, 0)

        # 二次发送 (相同帧 ID)
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

        # 发送一次：控制和诊断均被发送
        self.app._apply_stage_output(output, now=time.time(), force_send=True)
        self.assertEqual(len(self.app.obs_sender.sent_payloads), 1)
        self.assertEqual(len(self.app.diag_sender.sent_payloads), 1)

        # 立即再次发送，force_send=False：控制观测受到时间间隔限制（可能跳过或发送），
        # 但诊断观测严格受 1.0 秒间隔限制
        self.app.last_send_ts = 0.0  # 绕过控制发送间隔限制
        self.app._apply_stage_output(output, now=time.time(), force_send=False)
        self.assertEqual(len(self.app.obs_sender.sent_payloads), 2)
        # 诊断观测不应再次发送，因为 1.0 秒的间隔尚未过去
        self.assertEqual(len(self.app.diag_sender.sent_payloads), 1)

    def test_drop_tracking_on_failure(self):
        # 模拟发送器入队失败
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

    def test_search_stage_end_to_end_payload(self):
        from vision_module.app.stages.search.stage import SearchStagePlan
        from vision_module.app.stages.base import StageTickInput

        results = {
            "table_edge_obs": {
                "frame_id": 33,
                "edge_found": True,
                "edge_valid": True,
                "edge_trusted": True,
                "point_count": 91,
                "support_count": 85,
                "inlier_count": 13,
                "yaw_err_rad": 0.0791,
                "dist_err_m": 0.4733,
                "obs_ts": time.time() - 0.05,
            },
            "local_perception": {
                "frame_seq": 33,
                "table_bbox": [10, 20, 30, 40],
                "rgb_shape": (480, 640, 3),
            }
        }

        plan = SearchStagePlan()
        tick_input = StageTickInput(
            ts=time.time(),
            generation=1,
            results=results,
        )

        self.ctx.current_mode = "FIND_EDGE"
        self.ctx.stage_state = {}

        table_edge_obs, source, force_send = plan._process_table_edge_obs(
            results=results,
            ctx=self.ctx,
            tick_input=tick_input,
            local_perception=results["local_perception"],
        )

        self.assertTrue(table_edge_obs["edge_found"])
        self.assertTrue(table_edge_obs["edge_valid"])
        self.assertTrue(table_edge_obs["edge_trusted"])
        self.assertEqual(table_edge_obs["point_count"], 91)
        self.assertEqual(table_edge_obs["table_point_count"], 91)
        self.assertEqual(table_edge_obs["reason"], "edge_trusted")

        raw_obs = plan.build_obs(
            self.ctx,
            status="RUNNING",
            perception={"table_edge_obs": table_edge_obs},
        )
        output = StageOutput(vision_obs=raw_obs)

        self.app.scheduler.read_result = MagicMock(return_value={
            "frame_seq": 33,
            "frame_capture_ts": results["table_edge_obs"]["obs_ts"],
        })

        self.app.obs_sender.sent_payloads = []
        queued = self.app._apply_stage_output(output, now=time.time(), force_send=force_send)
        self.assertTrue(queued)

        sent_payload = self.app.obs_sender.sent_payloads[0]
        final_edge_obs = sent_payload["perception"]["table_edge_obs"]

        self.assertTrue(final_edge_obs["edge_found"])
        self.assertTrue(final_edge_obs["edge_valid"])
        self.assertTrue(final_edge_obs["edge_trusted"])
        self.assertEqual(final_edge_obs["point_count"], 91)
        self.assertEqual(final_edge_obs["table_point_count"], 91)
        self.assertEqual(final_edge_obs["reason"], "edge_trusted")


if __name__ == "__main__":
    unittest.main()
