#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
import time
import unittest
from collections import deque
from types import SimpleNamespace

try:
    from .test_support import PrintLogger, build_test_config, patch_engine_backends
except ImportError:
    from test_support import PrintLogger, build_test_config, patch_engine_backends

from vision_module.app.stage_controller import StageController
from vision_module.app.stages.base import StageContext, StageTickInput
from vision_module.app.stages.grasp import GraspStagePlan
from vision_module.backend.mode_controller import ModeController
from vision_module.backend.preview.base import PreviewFrame, PreviewSink
from vision_module.backend.preview.manager import PreviewManager
from vision_module.backend.scheduler import Scheduler
from vision_module.config.mode_defaults import build_default_mode_profiles
from vision_module.ipc.protocol import VisionReq


def build_runtime_stack(engine_module, cfg, logger, event_sink=None):
    runtime = engine_module.VisionEngine(cfg, logger=logger, event_sink=event_sink)
    mode_controller = ModeController(
        logger=logger,
        backend_event_sink=(lambda event, **fields: event_sink(event, fields)) if event_sink is not None else None,
        preview_allowed=bool(cfg.debug.preview),
    )
    mode_controller.register_profiles(build_default_mode_profiles(cfg.model.active_model).values())
    stage_controller = StageController(
        logger=logger,
        mode_controller=mode_controller,
        runtime_service=runtime,
    )
    return runtime, mode_controller, stage_controller


class GraspStageRemoteFlowTest(unittest.TestCase):
    def test_grasp_stage_uses_remote_effects_and_result_slot(self):
        plan = GraspStagePlan()
        ctx = StageContext()
        start_req = VisionReq(
            ts=time.time(),
            op="START",
            stage="GRASP",
            target="cup",
            payload={"remote_grasp": True, "need_depth": True},
        )
        plan.on_enter(start_req, ctx)

        first_tick = plan.tick(
            StageTickInput(
                ts=time.time(),
                results={"local_perception": {"target_obs": {"found": True, "target": "cup"}}},
            ),
            ctx,
        )
        self.assertIsNotNone(first_tick)
        self.assertEqual(first_tick.vision_obs["status"], "WAITING_RESPONSE")

        respond_req = VisionReq(
            ts=time.time(),
            op="RESPOND",
            stage="GRASP",
            target="cup",
            interaction_id=ctx.interaction_id,
            response={"decision": "ACCEPT"},
        )
        respond_out = plan.on_respond(respond_req, ctx)
        self.assertIsNotNone(respond_out)
        self.assertEqual(len(respond_out.effects), 2)
        self.assertEqual(respond_out.effects[0]["route"], "remote_cmd")
        self.assertEqual(respond_out.effects[1]["payload"]["op"], "PREDICT")

        request_id = ctx.stage_state["remote_request_id"]
        final_tick = plan.tick(
            StageTickInput(
                ts=time.time(),
                results={
                    "remote_result": {
                        "request_id": request_id,
                        "last_action": "predict",
                        "last_ok": True,
                        "has_result": True,
                        "result": {"grasps": [{"x": 1.0}]},
                        "sequence": 1,
                    }
                },
            ),
            ctx,
        )
        self.assertIsNotNone(final_tick)
        self.assertEqual(final_tick.vision_obs["status"], "RESULT_READY")
        self.assertTrue(any(effect["payload"]["op"] == "RELEASE" for effect in final_tick.effects))


class RuntimeSupervisorModeApplyTest(unittest.TestCase):
    def test_runtime_supervisor_reconciles_managers_from_mode(self):
        args = SimpleNamespace(
            rgb_device="mock_rgb",
            depth_device="mock_depth",
            ir_device="mock_ir",
            rgb_in_w=1280,
            rgb_in_h=720,
            rgb_out_w=640,
            rgb_out_h=640,
            rgb_fps=30,
            depth_width=424,
            depth_height=240,
            depth_fps=15,
            ir_in_w=640,
            ir_in_h=480,
            ir_out_w=640,
            ir_out_h=480,
            ir_fps=30,
            model_path="",
            model_width=640,
            model_height=640,
            conf_thres=0.25,
            iou_thres=0.15,
            class_num=20,
        )
        cfg = build_test_config(args)
        cfg.runtime.capability_placeholder = True
        engine_module = importlib.import_module("vision_module.backend.vision_engine")
        patch_engine_backends(engine_module, "mock", "mock")

        runtime, _, stage_controller = build_runtime_stack(engine_module, cfg, PrintLogger("arch"))
        try:
            runtime.init()
            runtime.start()
            self.assertTrue(stage_controller.set_runtime_mode("TRACK_LOCAL", reason="arch_test", force=True))
            snapshot = runtime.runtime_snapshot()
            self.assertTrue(snapshot["runtime_supervisor"]["camera"]["runtime_running"])
            self.assertTrue(snapshot["runtime_supervisor"]["predictor"]["runtime_running"])
            self.assertEqual(snapshot["runtime_supervisor"]["predictor"]["active_model_name"], "test_model")

            self.assertTrue(stage_controller.set_runtime_mode("IDLE", reason="arch_test_idle", force=True))
            snapshot = runtime.runtime_snapshot()
            self.assertEqual(snapshot["runtime_supervisor"]["camera"]["enabled_cameras"], [])
            self.assertIsNone(snapshot["runtime_supervisor"]["predictor"]["active_model_name"])
        finally:
            runtime.stop()

    def test_reconcile_failure_updates_mode_snapshot(self):
        args = SimpleNamespace(
            rgb_device="mock_rgb",
            depth_device="mock_depth",
            ir_device="mock_ir",
            rgb_in_w=1280,
            rgb_in_h=720,
            rgb_out_w=640,
            rgb_out_h=640,
            rgb_fps=30,
            depth_width=424,
            depth_height=240,
            depth_fps=15,
            ir_in_w=640,
            ir_in_h=480,
            ir_out_w=640,
            ir_out_h=480,
            ir_fps=30,
            model_path="",
            model_width=640,
            model_height=640,
            conf_thres=0.25,
            iou_thres=0.15,
            class_num=20,
        )
        cfg = build_test_config(args)
        cfg.runtime.capability_placeholder = True
        events = []

        def event_sink(name, fields):
            events.append((name, dict(fields or {})))

        engine_module = importlib.import_module("vision_module.backend.vision_engine")
        patch_engine_backends(engine_module, "mock", "mock")
        runtime, mode_controller, stage_controller = build_runtime_stack(
            engine_module,
            cfg,
            PrintLogger("arch_fail"),
            event_sink=event_sink,
        )
        try:
            runtime.init()
            runtime.start()
            original_reconcile = runtime.runtime_supervisor.reconcile
            runtime.runtime_supervisor.reconcile = lambda plan, generation: False
            try:
                self.assertFalse(stage_controller.set_runtime_mode("TRACK_LOCAL", reason="force_fail", force=True))
            finally:
                runtime.runtime_supervisor.reconcile = original_reconcile
            snapshot = mode_controller.snapshot()
            last_switch = snapshot["last_switch_result"]
            self.assertFalse(last_switch["ok"])
            self.assertEqual(last_switch["reason"], "runtime_apply_failed")
            self.assertEqual(last_switch["requested_mode"], "TRACK_LOCAL")
            self.assertEqual(last_switch["active_mode"], "IDLE")
            self.assertTrue(any(name == "BACKEND_FAILURE" and fields.get("failure_type") == "mode_apply_incomplete" for name, fields in events))
        finally:
            runtime.stop()


class SchedulerIsolationTest(unittest.TestCase):
    def setUp(self):
        self.scheduler = Scheduler()
        self.scheduler.start_runtime()
        self.plan = {
            "mode": "TRACK_LOCAL",
            "routes": {
                "local_perception": {"policy": "slot", "scope": "stage"},
                "remote_result": {"policy": "slot", "scope": "stage"},
                "camera_frames": {"policy": "slot", "scope": "backend"},
                "remote_ack": {"policy": "event", "scope": "backend"},
            },
        }

    def tearDown(self):
        self.scheduler.stop_runtime()

    def test_collect_tick_input_skips_stale_generation_results(self):
        self.scheduler.configure(self.plan, generation=2)
        self.scheduler.result_slots["local_perception"] = {
            "generation": 1,
            "ts": time.time(),
            "seq": 1,
            "payload": {"stale": True},
        }
        tick_input = self.scheduler.collect_tick_input(ts=time.time())
        self.assertNotIn("local_perception", tick_input.results)
        self.assertIsNone(self.scheduler.read_slot("local_perception"))

    def test_consume_event_skips_stale_generation_events(self):
        self.scheduler.configure(self.plan, generation=2)
        self.scheduler.event_latches["remote_ack"] = deque()
        self.scheduler.event_latches["remote_ack"].append(
            {"generation": 1, "ts": time.time(), "payload": {"stale": True}}
        )
        self.scheduler.event_latches["remote_ack"].append(
            {"generation": 2, "ts": time.time(), "payload": {"fresh": True}}
        )
        self.assertEqual(self.scheduler.consume_event("remote_ack"), {"fresh": True})
        self.assertIsNone(self.scheduler.consume_event("remote_ack"))

    def test_configure_clears_old_slots_and_events(self):
        self.scheduler.configure(self.plan, generation=1)
        self.scheduler.publish_result("local_perception", {"v": 1}, generation=1)
        self.scheduler.publish_event("remote_ack", {"v": 1}, generation=1)
        self.scheduler.configure(self.plan, generation=2)
        tick_input = self.scheduler.collect_tick_input(ts=time.time())
        self.assertEqual(tick_input.results, {})
        self.assertIsNone(self.scheduler.consume_event("remote_ack"))

    def test_rejects_unregistered_and_policy_mismatched_publishes(self):
        self.scheduler.configure(self.plan, generation=1)
        self.assertFalse(self.scheduler.publish_result("remote_ack", {"bad": True}, generation=1))
        self.assertFalse(self.scheduler.publish_event("local_perception", {"bad": True}, generation=1))
        self.assertFalse(self.scheduler.publish_result("unknown_route", {"bad": True}, generation=1))


class PreviewBehaviorTest(unittest.TestCase):
    class _ExitSink(PreviewSink):
        sink_name = "exit"

        def __init__(self):
            self.render_count = 0
            self.opened = False

        def open(self) -> None:
            self.opened = True

        def render(self, frame: PreviewFrame) -> bool:
            self.render_count += 1
            return False

        def close(self) -> None:
            self.opened = False

    def test_preview_close_disables_preview_without_event(self):
        scheduler = Scheduler()
        scheduler.start_runtime()
        scheduler.configure(
            {
                "mode": "TRACK_LOCAL",
                "routes": {
                    "camera_frames": {"policy": "slot", "scope": "backend"},
                    "local_perception": {"policy": "slot", "scope": "stage"},
                    "runtime_status": {"policy": "slot", "scope": "backend"},
                },
            },
            generation=1,
        )
        sink = self._ExitSink()
        manager = PreviewManager(sink=sink, logger=PrintLogger("preview"))
        manager.bind_runtime(scheduler, lambda: 1)
        scheduler.publish_result("camera_frames", {"rgb": [[0, 0], [0, 0]]}, generation=1)
        scheduler.publish_result("runtime_status", {"stage": "SEARCH", "mode": "TRACK_LOCAL", "epoch": 1}, generation=1)
        scheduler.publish_result("local_perception", {"box_count": 0}, generation=1)
        try:
            manager.enable()
            manager.start_runtime()
            time.sleep(0.1)
            self.assertFalse(manager.enabled)
            self.assertGreaterEqual(sink.render_count, 1)
            self.assertIsNone(scheduler.consume_event("preview_exit"))
        finally:
            manager.stop_runtime()
            scheduler.stop_runtime()


if __name__ == "__main__":
    unittest.main()
