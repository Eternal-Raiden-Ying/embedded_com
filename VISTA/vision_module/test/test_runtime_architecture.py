#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
import time
import unittest
from collections import deque
from types import SimpleNamespace

import numpy as np

try:
    from .test_support import PrintLogger, build_test_config, patch_engine_backends
except ImportError:
    from test_support import PrintLogger, build_test_config, patch_engine_backends

from vision_module.app.stage_controller import StageController
from vision_module.app.stages.base import StageContext, StageTickInput
from vision_module.app.stages.grasp import GraspStagePlan
from vision_module.app.stages.return_home import ReturnStagePlan
from vision_module.backend.camera_manager import CameraManager
from vision_module.backend.mode_controller import ModeController
from vision_module.backend.preview.base import PreviewFrame, PreviewSink
from vision_module.backend.preview.manager import PreviewManager
from vision_module.backend.predictor_manager import PredictorManager
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
    mode_controller.register_profiles(build_default_mode_profiles(cfg.model.active_model, cfg).values())
    stage_controller = StageController(
        logger=logger,
        mode_controller=mode_controller,
        runtime_service=runtime,
    )
    return runtime, mode_controller, stage_controller


class GraspStageRemoteFlowTest(unittest.TestCase):
    def test_grasp_stage_retries_service_init_then_waits_for_fresh_frames_before_predict(self):
        plan = GraspStagePlan()
        ctx = StageContext()
        start_req = VisionReq(
            ts=time.time(),
            op="START",
            stage="GRASP",
            target="cup",
            payload={"remote_grasp": True, "need_depth": True, "class_id": 41},
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
        self.assertEqual(respond_out.effects, [])

        request_id = ctx.stage_state["remote_request_id"]
        retry_tick = plan.tick(
            StageTickInput(
                ts=time.time(),
                results={
                    "remote_result": {
                        "service_init_confirmed": False,
                        "service_init_state": "failed",
                        "service_init_attempts": 1,
                        "service_init_last_error": "startup_failed",
                        "last_action": "init",
                        "last_ok": False,
                        "sequence": 1,
                    }
                },
            ),
            ctx,
        )
        self.assertIsNotNone(retry_tick)
        self.assertEqual(retry_tick.vision_obs["status"], "RUNNING")
        self.assertEqual(retry_tick.vision_obs["result"]["remote_state"], "retrying_init")
        self.assertEqual(retry_tick.effects[0]["payload"]["op"], "INIT")
        self.assertNotIn("base_url", retry_tick.effects[0]["payload"])

        init_ready_tick = plan.tick(
            StageTickInput(
                ts=time.time(),
                results={
                    "remote_result": {
                        "service_init_confirmed": True,
                        "service_init_state": "ready",
                        "service_init_attempts": 2,
                        "last_action": "init",
                        "last_ok": True,
                        "sequence": 2,
                    },
                },
            ),
            ctx,
        )
        self.assertIsNotNone(init_ready_tick)
        self.assertEqual(init_ready_tick.vision_obs["status"], "RUNNING")
        self.assertEqual(init_ready_tick.vision_obs["result"]["remote_state"], "awaiting_fresh_frames")

        predict_tick = plan.tick(
            StageTickInput(
                ts=time.time(),
                results={
                    "frame_meta": {
                        "has_frames": True,
                        "cameras": ["rgb", "depth"],
                        "frame_seq": 3,
                    },
                    "remote_result": {
                        "service_init_confirmed": True,
                        "service_init_state": "ready",
                        "service_init_attempts": 2,
                        "last_action": "init",
                        "last_ok": True,
                        "sequence": 2,
                    },
                },
            ),
            ctx,
        )
        self.assertIsNotNone(predict_tick)
        self.assertEqual(predict_tick.vision_obs["status"], "RUNNING")
        self.assertEqual(len(predict_tick.effects), 1)
        self.assertEqual(predict_tick.effects[0]["payload"]["op"], "PREDICT")
        self.assertNotIn("base_url", predict_tick.effects[0]["payload"])

        final_tick = plan.tick(
            StageTickInput(
                ts=time.time(),
                results={
                    "remote_result": {
                        "service_init_confirmed": True,
                        "service_init_state": "ready",
                        "service_init_attempts": 2,
                        "request_id": request_id,
                        "last_action": "predict",
                        "last_ok": True,
                        "has_result": True,
                        "result": {"grasps": [{"x": 1.0}]},
                        "sequence": 2,
                    }
                },
            ),
            ctx,
        )
        self.assertIsNotNone(final_tick)
        self.assertEqual(final_tick.vision_obs["status"], "RESULT_READY")
        self.assertEqual(final_tick.effects, [])

    def test_grasp_stage_fails_after_three_init_retries(self):
        plan = GraspStagePlan()
        ctx = StageContext()
        start_req = VisionReq(
            ts=time.time(),
            op="START",
            stage="GRASP",
            target="cup",
            payload={"remote_grasp": True, "need_depth": True, "class_id": 41},
        )
        plan.on_enter(start_req, ctx)
        plan.tick(
            StageTickInput(
                ts=time.time(),
                results={"local_perception": {"target_obs": {"found": True, "target": "cup"}}},
            ),
            ctx,
        )
        respond_req = VisionReq(
            ts=time.time(),
            op="RESPOND",
            stage="GRASP",
            target="cup",
            interaction_id=ctx.interaction_id,
            response={"decision": "ACCEPT"},
        )
        plan.on_respond(respond_req, ctx)

        for service_attempts in (1, 2, 3):
            tick = plan.tick(
                StageTickInput(
                    ts=time.time(),
                    results={
                        "remote_result": {
                            "service_init_confirmed": False,
                            "service_init_state": "failed",
                            "service_init_attempts": service_attempts,
                            "service_init_last_error": f"init_failed_{service_attempts}",
                            "last_action": "init",
                            "last_ok": False,
                            "sequence": service_attempts,
                        }
                    },
                ),
                ctx,
            )
            self.assertIsNotNone(tick)
            self.assertEqual(tick.vision_obs["status"], "RUNNING")
            self.assertEqual(tick.effects[0]["payload"]["op"], "INIT")

        failed_tick = plan.tick(
            StageTickInput(
                ts=time.time(),
                results={
                    "remote_result": {
                        "service_init_confirmed": False,
                        "service_init_state": "failed",
                        "service_init_attempts": 4,
                        "service_init_last_error": "init_failed_4",
                        "last_action": "init",
                        "last_ok": False,
                        "sequence": 4,
                    }
                },
            ),
            ctx,
        )
        self.assertIsNotNone(failed_tick)
        self.assertEqual(failed_tick.vision_obs["status"], "FAILED")
        self.assertEqual(failed_tick.vision_obs["result"]["reason"], "remote_init_failed")
        self.assertEqual(failed_tick.vision_obs["result"]["init_attempts"], 3)
        self.assertFalse(failed_tick.vision_obs["result"]["init_confirmed"])
        self.assertEqual(failed_tick.effects, [])


class ReturnStageDetectContractTest(unittest.TestCase):
    def test_return_stage_consumes_detect_mainline(self):
        plan = ReturnStagePlan()
        ctx = StageContext()
        start_req = VisionReq(
            ts=time.time(),
            op="START",
            stage="RETURN",
            target="cup",
        )
        plan.on_enter(start_req, ctx)

        tick = plan.tick(
            StageTickInput(
                ts=time.time(),
                results={
                    "local_perception": {
                        "infer_boxes": [[160.0, 120.0, 320.0, 360.0, 0.9, 1.0]],
                        "class_names": ["person", "cup"],
                        "rgb_shape": [480, 640, 3],
                        "contract_ok": True,
                    }
                },
            ),
            ctx,
        )
        self.assertIsNotNone(tick)
        home_tag_obs = tick.vision_obs["perception"]["home_tag_obs"]
        self.assertTrue(home_tag_obs["found"])
        self.assertEqual(home_tag_obs["source"], "detect")
        self.assertEqual(home_tag_obs["target"], "cup")
        self.assertNotIn("tag_id", home_tag_obs)
        self.assertIn("area_norm", home_tag_obs)

    def test_return_stage_requires_explicit_target_for_detect_path(self):
        plan = ReturnStagePlan()
        ctx = StageContext()
        start_req = VisionReq(
            ts=time.time(),
            op="START",
            stage="RETURN",
        )
        plan.on_enter(start_req, ctx)

        tick = plan.tick(
            StageTickInput(
                ts=time.time(),
                results={
                    "local_perception": {
                        "infer_boxes": [[160.0, 120.0, 320.0, 360.0, 0.9, 1.0]],
                        "class_names": ["person", "cup"],
                        "rgb_shape": [480, 640, 3],
                        "contract_ok": True,
                    }
                },
            ),
            ctx,
        )
        self.assertIsNotNone(tick)
        home_tag_obs = tick.vision_obs["perception"]["home_tag_obs"]
        self.assertFalse(home_tag_obs["found"])
        self.assertEqual(home_tag_obs["reason"], "missing_return_target")


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


class BackendSelectionContractTest(unittest.TestCase):
    class _SentinelCamera:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def read_frame(self):
            return np.zeros((4, 4, 3), dtype=np.uint8)

        def release(self) -> None:
            return None

    def test_camera_manager_backend_selection_does_not_follow_capability_placeholder(self):
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
        manager_module = importlib.import_module("vision_module.backend.camera_manager")
        original_cls = manager_module.ColorCamera
        manager_module.ColorCamera = self._SentinelCamera
        try:
            manager = CameraManager(cfg=cfg, logger=PrintLogger("camera_backend"))
            camera = manager._build_camera(
                "rgb",
                {
                    "device": "mock_rgb",
                    "in_format": "YUY2",
                    "format": "BGR",
                    "fps": 30,
                    "in_w": 1280,
                    "in_h": 720,
                    "out_w": 640,
                    "out_h": 640,
                },
            )
            self.assertIsInstance(camera, self._SentinelCamera)
        finally:
            manager_module.ColorCamera = original_cls


class ModeProfileCameraContractTest(unittest.TestCase):
    def test_default_mode_profiles_expose_distinct_bgr_camera_overrides(self):
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
        profiles = build_default_mode_profiles("test_model", cfg)

        track_rgb = dict(profiles["TRACK_LOCAL"].camera_overrides["rgb"])
        micro_rgb = dict(profiles["MICRO_ADJUST"].camera_overrides["rgb"])
        grasp_rgb = dict(profiles["GRASP_REMOTE"].camera_overrides["rgb"])

        self.assertEqual(track_rgb["format"], "BGR")
        self.assertEqual(micro_rgb["format"], "BGR")
        self.assertEqual(grasp_rgb["format"], "BGR")
        self.assertEqual(track_rgb["fps"], 24)
        self.assertEqual(micro_rgb["fps"], 30)
        self.assertEqual(grasp_rgb["fps"], 15)
        self.assertNotEqual(
            (track_rgb["crop_x"], track_rgb["crop_y"], track_rgb["crop_w"], track_rgb["crop_h"]),
            (micro_rgb["crop_x"], micro_rgb["crop_y"], micro_rgb["crop_w"], micro_rgb["crop_h"]),
        )
        self.assertIn("depth", profiles["GRASP_REMOTE"].camera_overrides)


class GenerationAwareFrameConsumptionTest(unittest.TestCase):
    class _DummyPredictor:
        def __init__(self, profile):
            self.profile = profile

        def is_ready(self) -> bool:
            return True

        def predict_frame(self, frame):
            _ = frame
            return [[10.0, 20.0, 110.0, 220.0, 0.95, 41.0]], []

        def release(self) -> None:
            return None

    class _CountingSink(PreviewSink):
        sink_name = "counting"

        def __init__(self):
            self.render_count = 0
            self.frames = []

        def open(self) -> None:
            return None

        def render(self, frame: PreviewFrame) -> bool:
            self.render_count += 1
            self.frames.append(dict(frame.overlay.metadata or {}))
            return True

        def close(self) -> None:
            return None

    def _test_args(self):
        return SimpleNamespace(
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
            model_path="dummy.ctx.bin",
            model_width=640,
            model_height=640,
            conf_thres=0.25,
            iou_thres=0.15,
            class_num=80,
        )

    def test_predictor_resumes_after_generation_reset(self):
        cfg = build_test_config(self._test_args())
        profile = cfg.model.profiles["test_model"]
        profile.class_num = 80
        profile.classes = ("person", "cup")
        profile.predictor_type = "detect"
        scheduler = Scheduler()
        scheduler.start_runtime()
        plan = {
            "mode": "TRACK_LOCAL",
            "routes": {
                "camera_frames": {"policy": "slot", "scope": "backend"},
                "local_perception": {"policy": "slot", "scope": "stage"},
            },
        }
        scheduler.configure(plan, generation=1)
        generation = {"value": 1}
        manager = PredictorManager(cfg=cfg, logger=PrintLogger("predictor_generation"))
        manager.bind_runtime(scheduler, lambda: generation["value"])
        manager_module = importlib.import_module("vision_module.backend.predictor_manager")
        original_cls = manager_module.QNN_YOLO_Dectec_Predictor
        manager_module.QNN_YOLO_Dectec_Predictor = self._DummyPredictor
        try:
            self.assertTrue(manager.ensure_model("test_model"))
            manager.set_inference_enabled(True)
            manager.start_runtime()
            scheduler.publish_result("camera_frames", {"rgb": np.zeros((64, 64, 3), dtype=np.uint8)}, generation=1)
            scheduler.publish_result("camera_frames", {"rgb": np.zeros((64, 64, 3), dtype=np.uint8)}, generation=1)
            deadline = time.time() + 1.0
            while time.time() < deadline:
                if manager.snapshot()["last_camera_seq"] >= 2:
                    break
                time.sleep(0.05)
            self.assertGreaterEqual(manager.snapshot()["last_camera_seq"], 2)

            generation["value"] = 2
            scheduler.configure(plan, generation=2)
            scheduler.publish_result("camera_frames", {"rgb": np.zeros((64, 64, 3), dtype=np.uint8)}, generation=2)
            payload = None
            deadline = time.time() + 1.0
            while time.time() < deadline:
                payload = scheduler.read_result("local_perception", default=None)
                snapshot = manager.snapshot()
                if isinstance(payload, dict) and payload.get("box_count") == 1 and snapshot["last_camera_generation"] == 2:
                    break
                time.sleep(0.05)
            self.assertIsInstance(payload, dict)
            self.assertEqual(payload["box_count"], 1)
            self.assertEqual(manager.snapshot()["last_camera_generation"], 2)
        finally:
            manager_module.QNN_YOLO_Dectec_Predictor = original_cls
            manager.release_all()
            scheduler.stop_runtime()

    def test_preview_resumes_after_generation_reset(self):
        scheduler = Scheduler()
        scheduler.start_runtime()
        plan = {
            "mode": "TRACK_LOCAL",
            "routes": {
                "camera_frames": {"policy": "slot", "scope": "backend"},
                "local_perception": {"policy": "slot", "scope": "stage"},
                "runtime_status": {"policy": "slot", "scope": "backend"},
            },
        }
        scheduler.configure(plan, generation=1)
        generation = {"value": 1}
        sink = self._CountingSink()
        manager = PreviewManager(sink=sink, logger=PrintLogger("preview_generation"))
        manager.bind_runtime(scheduler, lambda: generation["value"])
        try:
            manager.enable()
            manager.start_runtime()
            scheduler.publish_result("runtime_status", {"stage": "SEARCH", "mode": "TRACK_LOCAL", "epoch": 1}, generation=1)
            scheduler.publish_result("local_perception", {"box_count": 0}, generation=1)
            scheduler.publish_result("camera_frames", {"rgb": np.zeros((16, 16, 3), dtype=np.uint8)}, generation=1)
            deadline = time.time() + 1.0
            while time.time() < deadline:
                if sink.render_count >= 1:
                    break
                time.sleep(0.05)
            self.assertGreaterEqual(sink.render_count, 1)

            generation["value"] = 2
            scheduler.configure(plan, generation=2)
            scheduler.publish_result("runtime_status", {"stage": "SEARCH", "mode": "TRACK_LOCAL", "epoch": 2}, generation=2)
            scheduler.publish_result("local_perception", {"box_count": 0}, generation=2)
            previous_count = sink.render_count
            scheduler.publish_result("camera_frames", {"rgb": np.zeros((16, 16, 3), dtype=np.uint8)}, generation=2)
            deadline = time.time() + 1.0
            while time.time() < deadline:
                if sink.render_count > previous_count and manager.snapshot()["last_frame_generation"] == 2:
                    break
                time.sleep(0.05)
            self.assertGreater(sink.render_count, previous_count)
            self.assertEqual(manager.snapshot()["last_frame_generation"], 2)
        finally:
            manager.stop_runtime()
            scheduler.stop_runtime()


if __name__ == "__main__":
    unittest.main()
