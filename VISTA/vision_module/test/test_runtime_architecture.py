#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
import time
import unittest
from collections import deque
from types import SimpleNamespace

import numpy as np

try:
    from .test_support import PrintLogger, build_test_config, import_camera_classes, import_predictor_class
except ImportError:
    from test_support import PrintLogger, build_test_config, import_camera_classes, import_predictor_class

from vision_module.app.stage_controller import StageController
from vision_module.app.stages.base import StageContext, StageTickInput
from vision_module.app.stages.grasp import GraspStagePlan
from vision_module.app.stages.return_home import ReturnStagePlan
from vision_module.app.stages.search import SearchStagePlan
from vision_module.backend.camera_manager import CameraManager
from vision_module.backend.mode_controller import ModeController
from vision_module.backend.preview import NullPreviewSink, PreviewFrame, PreviewSink
from vision_module.backend.preview.manager import PreviewManager
from vision_module.backend.remote.client import RemoteGraspClient
from vision_module.backend.remote.manager import RemoteManager
from vision_module.backend.runtime_supervisor import RuntimeSupervisor
from vision_module.backend.table_edge_manager import TableEdgeManager
from vision_module.backend.predictor_manager import PredictorManager
from vision_module.backend.scheduler import Scheduler
from vision_module.config.mode_defaults import build_default_mode_profiles
from vision_module.ipc.protocol import VisionReq
try:
    from common.runtime_logging import OperatorConsole
except ImportError:
    from vision_module.diagnostics.operator_console import OperatorConsole


def _patch_managers(backend: str) -> None:
    import importlib as _imp
    _cam = _imp.import_module("vision_module.backend.camera_manager")
    _pred = _imp.import_module("vision_module.backend.predictor_manager")
    color_cls, ir_cls, depth_cls = import_camera_classes(backend)
    predictor_cls = import_predictor_class(backend)
    _cam.ColorCamera = color_cls
    _cam.IRCamera = ir_cls
    _cam.RealSenseDepthCamera = depth_cls
    _pred.QNN_YOLO_Detect_Predictor = predictor_cls
    _pred.QNN_YOLO_Segment_Predictor = predictor_cls


def build_runtime_stack(cfg, logger, event_sink=None):
    scheduler = Scheduler()
    supervisor = RuntimeSupervisor(
        scheduler=scheduler,
        camera_manager=CameraManager(cfg=cfg, logger=logger),
        predictor_manager=PredictorManager(cfg=cfg, logger=logger),
        remote_manager=RemoteManager(client=RemoteGraspClient(logger=logger), logger=logger),
        table_edge_manager=TableEdgeManager(logger=logger),
        preview_manager=PreviewManager(sink=NullPreviewSink(), logger=logger),
        logger=logger,
        backend_event_sink=event_sink,
    )
    mode_controller = ModeController(
        scheduler=scheduler,
        supervisor=supervisor,
        logger=logger,
        backend_event_sink=(lambda event, **fields: event_sink(event, fields)) if event_sink is not None else None,
        preview_allowed=bool(cfg.debug.preview),
    )
    mode_controller.register_profiles(build_default_mode_profiles(cfg.model.active_model, cfg).values())
    stage_controller = StageController(
        logger=logger,
        mode_controller=mode_controller,
        scheduler=scheduler,
    )
    return mode_controller, mode_controller, stage_controller


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


class SearchStagePlanKindTest(unittest.TestCase):
    def _enter(self, search_kind, target="cup"):
        plan = SearchStagePlan()
        ctx = StageContext()
        req = VisionReq(
            ts=time.time(),
            op="START",
            stage="SEARCH",
            target=target,
            payload={"search_kind": search_kind},
        )
        plan.on_enter(req, ctx)
        return plan, ctx

    def test_target_search_outputs_target_and_table_edge_obs(self):
        plan, ctx = self._enter("TARGET")
        self.assertEqual(ctx.current_mode, "FIND_OBJECT")
        output = plan.tick(
            StageTickInput(
                ts=time.time(),
                generation=1,
                results={
                    "local_perception": {"target_obs": {"found": True, "target": "cup"}},
                    "table_edge_obs": {"edge_found": True, "obs_ts": time.time()},
                },
            ),
            ctx,
        )
        perception = output.vision_obs["perception"]
        self.assertEqual(sorted(perception.keys()), ["table_edge_obs", "target_obs"])
        self.assertTrue(perception["target_obs"]["found"])
        self.assertTrue(perception["table_edge_obs"]["edge_found"])
        self.assertTrue(perception["table_edge_obs"]["edge_valid"])
        self.assertIn("edge_conf", perception["table_edge_obs"])
        self.assertIn("yaw_err", perception["table_edge_obs"])
        self.assertIn("dist_err", perception["table_edge_obs"])
        self.assertIn("obs_ts", perception["table_edge_obs"])
        self.assertIn("age_ms", perception["table_edge_obs"])
        self.assertEqual(perception["table_edge_obs"]["source_mode"], "FIND_OBJECT")
        self.assertFalse(perception["table_edge_obs"]["is_stale"])

    def test_target_search_builds_obs_from_yolo_boxes(self):
        plan, ctx = self._enter("TARGET", target="apple")
        output = plan.tick(
            StageTickInput(
                ts=time.time(),
                generation=1,
                results={
                    "local_perception": {
                        "has_infer": True,
                        "rgb_shape": [640, 640, 3],
                        "class_names": ["apple"],
                        "box_count": 1,
                        "infer_boxes": [[120, 180, 260, 340, 0.88, 0]],
                    },
                },
            ),
            ctx,
        )
        target_obs = output.vision_obs["perception"]["target_obs"]
        self.assertTrue(target_obs["found"])
        self.assertEqual(target_obs["target"], "apple")
        self.assertEqual(target_obs["boxes_count"], 1)
        self.assertEqual(target_obs["best_cls"], "apple")
        self.assertEqual(target_obs["bbox"], [120, 180, 260, 340])

    def test_target_search_reports_empty_yolo_results(self):
        plan, ctx = self._enter("TARGET", target="apple")
        output = plan.tick(
            StageTickInput(
                ts=time.time(),
                generation=1,
                results={
                    "local_perception": {
                        "has_infer": True,
                        "rgb_shape": [640, 640, 3],
                        "class_names": ["apple"],
                        "box_count": 0,
                        "infer_boxes": [],
                    },
                },
            ),
            ctx,
        )
        target_obs = output.vision_obs["perception"]["target_obs"]
        self.assertFalse(target_obs["found"])
        self.assertEqual(target_obs["boxes_count"], 0)
        self.assertEqual(target_obs["reason"], "no_boxes")

    def test_target_search_update_does_not_clobber_latest_target_obs(self):
        plan, ctx = self._enter("TARGET", target="apple")
        first = plan.tick(
            StageTickInput(
                ts=time.time(),
                generation=1,
                results={
                    "local_perception": {
                        "has_infer": True,
                        "rgb_shape": [640, 640, 3],
                        "class_names": ["apple"],
                        "box_count": 1,
                        "infer_boxes": [[120, 180, 260, 340, 0.88, 0]],
                    },
                },
            ),
            ctx,
        )
        self.assertTrue(first.vision_obs["perception"]["target_obs"]["found"])

        update_req = VisionReq(
            ts=time.time(),
            op="UPDATE",
            stage="SEARCH",
            target="apple",
            mode_hint="FIND_OBJECT",
            payload={"search_kind": "TARGET", "orchestrator_state": "EDGE_SLIDE_SEARCH"},
        )
        plan.on_update(update_req, ctx)
        self.assertTrue(ctx.stage_state["target_obs"]["found"])
        self.assertEqual(ctx.stage_state["target_obs"]["best_cls"], "apple")

    def test_table_edge_search_outputs_only_table_edge_obs(self):
        plan, ctx = self._enter("TABLE_EDGE")
        self.assertEqual(ctx.current_mode, "FIND_EDGE")
        output = plan.tick(
            StageTickInput(
                ts=time.time(),
                generation=1,
                results={
                    "local_perception": {"target_obs": {"found": True, "target": "cup"}},
                    "table_edge_obs": {"edge_found": True, "confidence": 0.8},
                },
            ),
            ctx,
        )
        perception = output.vision_obs["perception"]
        self.assertEqual(sorted(perception.keys()), ["table_edge_obs"])
        self.assertTrue(perception["table_edge_obs"]["edge_found"])

    def test_edge_follow_target_outputs_both_and_keeps_partial_results(self):
        plan, ctx = self._enter("EDGE_FOLLOW_TARGET")
        self.assertEqual(ctx.current_mode, "FIND_EDGE")
        output = plan.tick(
            StageTickInput(
                ts=time.time(),
                generation=1,
                results={
                    "local_perception": {"target_obs": {"found": True, "target": "cup"}},
                },
            ),
            ctx,
        )
        perception = output.vision_obs["perception"]
        self.assertEqual(sorted(perception.keys()), ["table_edge_obs", "target_obs"])
        self.assertTrue(perception["target_obs"]["found"])
        self.assertFalse(perception["table_edge_obs"]["edge_found"])
        self.assertTrue(perception["table_edge_obs"]["is_stale"])

        alias_plan, alias_ctx = self._enter("TARGET_ON_EDGE")
        self.assertEqual(alias_ctx.current_mode, "FIND_EDGE")
        self.assertEqual(alias_ctx.stage_state["search_kind"], "TARGET_ON_EDGE")


class StageControllerStatusTest(unittest.TestCase):
    def test_runtime_status_includes_active_target(self):
        controller = StageController()
        controller._ctx.current_stage = "SEARCH"
        controller._ctx.current_mode = "FIND_OBJECT"
        controller._ctx.target_name = "apple"
        status = controller._runtime_status_payload()
        self.assertEqual(status["target"], "apple")


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
        _patch_managers("mock")

        mode_controller, _, stage_controller = build_runtime_stack(cfg, PrintLogger("arch"))
        try:
            mode_controller.start_runtime()
            self.assertTrue(stage_controller.set_runtime_mode("FIND_OBJECT", reason="arch_test", force=True))
            snapshot = mode_controller.runtime_snapshot()
            self.assertTrue(snapshot["runtime_supervisor"]["camera"]["runtime_running"])
            self.assertTrue(snapshot["runtime_supervisor"]["predictor"]["runtime_running"])
            self.assertEqual(snapshot["runtime_supervisor"]["predictor"]["active_model_name"], "test_model")

            self.assertTrue(stage_controller.set_runtime_mode("SILENT", reason="arch_test_idle", force=True))
            snapshot = mode_controller.runtime_snapshot()
            self.assertEqual(snapshot["runtime_supervisor"]["camera"]["enabled_cameras"], [])
            self.assertIsNone(snapshot["runtime_supervisor"]["predictor"]["active_model_name"])
        finally:
            mode_controller.stop_runtime()

    def test_table_edge_search_falls_back_when_preferred_mode_unregistered(self):
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
            class_num=80,
        )
        cfg = build_test_config(args)
        cfg.runtime.capability_placeholder = True
        _patch_managers("mock")
        mode_controller, _, stage_controller = build_runtime_stack(cfg, PrintLogger("table_edge_fallback"))
        profiles = build_default_mode_profiles(cfg.model.active_model)
        mode_controller.register_profile(profiles["SILENT"])
        mode_controller.register_profile(profiles["FIND_EDGE"])
        stage_controller.register_plan(SearchStagePlan())
        try:
            mode_controller.start_runtime()
            out = stage_controller.handle_request(
                VisionReq(
                    ts=time.time(),
                    op="START",
                    stage="SEARCH",
                    payload={"search_kind": "TABLE_EDGE"},
                )
            )
            self.assertIsNotNone(out)
            self.assertEqual(stage_controller.context().current_mode, "FIND_EDGE")
            self.assertEqual(mode_controller.current_mode(), "FIND_EDGE")
        finally:
            mode_controller.stop_runtime()

    def test_table_edge_perception_starts_rgb_depth_predictor_and_table_edge(self):
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
            class_num=80,
        )
        cfg = build_test_config(args)
        cfg.runtime.capability_placeholder = True
        _patch_managers("mock")

        mode_controller, _, stage_controller = build_runtime_stack(cfg, PrintLogger("table_edge_mode"))
        try:
            mode_controller.start_runtime()
            self.assertTrue(stage_controller.set_runtime_mode("FIND_EDGE", reason="table_edge_test", force=True))
            snapshot = mode_controller.runtime_snapshot()
            supervisor = snapshot["runtime_supervisor"]
            plan = snapshot["active_runtime_plan"]
            self.assertEqual(set(supervisor["camera"]["enabled_cameras"]), {"rgb", "depth"})
            self.assertTrue(supervisor["camera"]["runtime_running"])
            self.assertTrue(supervisor["predictor"]["runtime_running"])
            self.assertTrue(supervisor["predictor"]["inference_enabled"])
            self.assertEqual(supervisor["predictor"]["active_model_name"], "test_model")
            self.assertTrue(supervisor["table_edge"]["runtime_running"])
            self.assertEqual(
                plan["contract"]["capability"]["perception"],
                ["local_perception", "table_edge_obs"],
            )
        finally:
            mode_controller.stop_runtime()

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

        mode_controller, _, stage_controller = build_runtime_stack(cfg, PrintLogger("arch_fail"), event_sink=lambda name, fields: None)
        try:
            mode_controller.start_runtime()
            original_reconcile = mode_controller.supervisor.reconcile
            mode_controller.supervisor.reconcile = lambda plan, generation: False
            try:
                self.assertFalse(stage_controller.set_runtime_mode("FIND_OBJECT", reason="force_fail", force=True))
            finally:
                mode_controller.supervisor.reconcile = original_reconcile
            snapshot = mode_controller.snapshot()
            last_switch = snapshot["last_switch_result"]
            self.assertFalse(last_switch["ok"])
            self.assertEqual(last_switch["reason"], "runtime_apply_failed")
            self.assertEqual(last_switch["requested_mode"], "FIND_OBJECT")
            self.assertEqual(last_switch["active_mode"], "SILENT")
        finally:
            mode_controller.stop_runtime()


class SchedulerIsolationTest(unittest.TestCase):
    def setUp(self):
        self.scheduler = Scheduler()
        self.scheduler.start_runtime()
        self.plan = {
            "mode": "FIND_OBJECT",
            "routes": {
                "local_perception": {"policy": "slot", "scope": "stage"},
                "remote_result": {"policy": "slot", "scope": "stage"},
                "camera_frames": {"policy": "slot", "scope": "backend"},
                "test_event": {"policy": "event", "scope": "backend"},
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
        self.scheduler.event_latches["test_event"] = deque()
        self.scheduler.event_latches["test_event"].append(
            {"generation": 1, "ts": time.time(), "payload": {"stale": True}}
        )
        self.scheduler.event_latches["test_event"].append(
            {"generation": 2, "ts": time.time(), "payload": {"fresh": True}}
        )
        self.assertEqual(self.scheduler.consume_event("test_event"), {"fresh": True})
        self.assertIsNone(self.scheduler.consume_event("test_event"))

    def test_configure_clears_old_slots_and_events(self):
        self.scheduler.configure(self.plan, generation=1)
        self.scheduler.publish_result("local_perception", {"v": 1}, generation=1)
        self.scheduler.publish_event("test_event", {"v": 1}, generation=1)
        self.scheduler.configure(self.plan, generation=2)
        tick_input = self.scheduler.collect_tick_input(ts=time.time())
        self.assertEqual(tick_input.results, {})
        self.assertIsNone(self.scheduler.consume_event("test_event"))

    def test_rejects_unregistered_and_policy_mismatched_publishes(self):
        self.scheduler.configure(self.plan, generation=1)
        self.assertFalse(self.scheduler.publish_result("test_event", {"bad": True}, generation=1))
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
                "mode": "FIND_OBJECT",
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
        scheduler.publish_result("runtime_status", {"stage": "SEARCH", "mode": "FIND_OBJECT", "epoch": 1}, generation=1)
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

    def test_preview_render_exception_is_reported_without_killing_preview(self):
        class FailingSink(PreviewSink):
            sink_name = "failing"

            def __init__(self):
                self.render_count = 0

            def open(self) -> None:
                return None

            def render(self, frame: PreviewFrame) -> bool:
                self.render_count += 1
                raise RuntimeError("metadata render failure")

            def close(self) -> None:
                return None

        scheduler = Scheduler()
        scheduler.start_runtime()
        scheduler.configure(
            {
                "mode": "FIND_OBJECT",
                "routes": {
                    "camera_frames": {"policy": "slot", "scope": "backend"},
                    "local_perception": {"policy": "slot", "scope": "stage"},
                    "runtime_status": {"policy": "slot", "scope": "backend"},
                },
            },
            generation=1,
        )
        lines = []
        console = OperatorConsole(mode="operator", default_interval_s=1.0, sink=lines.append)
        sink = FailingSink()
        manager = PreviewManager(sink=sink, logger=None, operator_console=console)
        manager.bind_runtime(scheduler, lambda: 1)
        scheduler.publish_result("camera_frames", {"rgb": np.zeros((16, 16, 3), dtype=np.uint8)}, generation=1)
        scheduler.publish_result("runtime_status", {"stage": "SEARCH", "mode": "FIND_OBJECT", "target": "apple"}, generation=1)
        scheduler.publish_result("local_perception", {"has_infer": True, "box_count": 0, "infer_boxes": []}, generation=1)
        try:
            manager.enable()
            manager.start_runtime()
            time.sleep(0.08)
            self.assertTrue(manager.enabled)
            self.assertGreaterEqual(sink.render_count, 1)
            self.assertTrue(any("[VISTA] WARN PREVIEW render_failed" in line for line in lines))
        finally:
            manager.stop_runtime()
            scheduler.stop_runtime()

    def test_table_edge_operator_summary_is_rate_limited(self):
        lines = []
        console = OperatorConsole(mode="operator", default_interval_s=1.0, sink=lines.append)
        manager = PreviewManager(sink=self._ExitSink(), logger=None, operator_console=console)
        status = {"stage": "SEARCH", "mode": "FIND_EDGE"}
        table_edge = {
            "table_found": True,
            "edge_found": True,
            "confidence": 0.76,
            "yaw_err_rad": 0.02,
            "dist_err_m": 0.034,
            "roi_source": "static_fallback",
            "point_count": 418,
            "reason": "ok",
        }
        line = manager._table_edge_summary_line(status, table_edge)
        manager._emit_operator("preview:table_edge_obs", line)
        manager._emit_operator("preview:table_edge_obs", line)
        self.assertEqual(lines, [line])
        self.assertIn("[VISTA] EDGE stage=SEARCH mode=DEPTH_PERCEPTION", line)

    def test_track_local_preview_renders_without_depth_frame(self):
        class RecordingSink(PreviewSink):
            sink_name = "recording"

            def __init__(self):
                self.frames = []

            def open(self) -> None:
                return None

            def render(self, frame: PreviewFrame) -> bool:
                self.frames.append(frame)
                return True

            def close(self) -> None:
                return None

        scheduler = Scheduler()
        scheduler.start_runtime()
        scheduler.configure(
            {
                "mode": "FIND_OBJECT",
                "routes": {
                    "camera_frames": {"policy": "slot", "scope": "backend"},
                    "frame_meta": {"policy": "slot", "scope": "stage"},
                    "local_perception": {"policy": "slot", "scope": "stage"},
                    "runtime_status": {"policy": "slot", "scope": "backend"},
                    "target_obs": {"policy": "slot", "scope": "stage"},
                    "table_edge_obs": {"policy": "slot", "scope": "stage"},
                },
            },
            generation=1,
        )
        sink = RecordingSink()
        manager = PreviewManager(sink=sink, logger=None)
        manager.bind_runtime(scheduler, lambda: 1)
        scheduler.publish_result("camera_frames", {"rgb": np.zeros((16, 16, 3), dtype=np.uint8)}, generation=1)
        scheduler.publish_result("runtime_status", {"stage": "SEARCH", "mode": "FIND_OBJECT", "target": "apple"}, generation=1)
        scheduler.publish_result("local_perception", {"has_infer": True, "box_count": 0, "infer_boxes": []}, generation=1)
        try:
            manager.enable()
            manager.start_runtime()
            time.sleep(0.08)
        finally:
            manager.stop_runtime()
            scheduler.stop_runtime()
        self.assertGreaterEqual(len(sink.frames), 1)
        metadata = sink.frames[-1].overlay.metadata
        self.assertEqual(metadata["source_cameras"], ["rgb"])
        self.assertEqual(metadata["table_edge_obs"], {})
        self.assertEqual(metadata["preview_layout"], "rgb_yolo_edge_overlay")

    def test_preview_mode_switch_logs_layout_and_updates_sink_without_rebuild(self):
        class LayoutSink(PreviewSink):
            sink_name = "layout"

            def __init__(self):
                self.layouts = []
                self.open_count = 0
                self.close_count = 0

            def open(self) -> None:
                self.open_count += 1

            def render(self, frame: PreviewFrame) -> bool:
                return True

            def close(self) -> None:
                self.close_count += 1

            def set_layout(self, layout: str, reason: str = "") -> None:
                self.layouts.append((layout, reason))

        lines = []
        console = OperatorConsole(mode="operator", default_interval_s=0.0, sink=lines.append)
        sink = LayoutSink()
        manager = PreviewManager(sink=sink, logger=None, operator_console=console)
        metadata = {
            "mode_layouts": {
                "FIND_EDGE": "rgb_depth_edge",
                "FIND_OBJECT": "rgb_yolo_edge_overlay",
                "IDLE_HOT": "rgb_hot_preview",
            },
            "clear_overlay_on_mode_switch": True,
        }
        manager.configure_preview_mode("FIND_EDGE", metadata=metadata, reason="unit")
        manager.configure_preview_mode("FIND_OBJECT", metadata=metadata, reason="unit")
        manager.configure_preview_mode("FIND_OBJECT", metadata=metadata, reason="target_confirm")

        self.assertIn(("rgb_depth_edge", "unit"), sink.layouts)
        self.assertIn(("rgb_yolo_edge_overlay", "unit"), sink.layouts)
        self.assertEqual(sink.open_count, 0)
        self.assertEqual(sink.close_count, 0)
        joined = "\n".join(lines)
        self.assertIn("old_mode=TABLE_EDGE_PERCEPTION new_mode=TRACK_LOCAL", joined)
        self.assertIn("old_layout=rgb_depth_edge new_layout=rgb_yolo_edge_overlay", joined)

    def test_preview_unsupported_layout_falls_back_to_rgb_minimal(self):
        class LayoutSink(PreviewSink):
            sink_name = "layout"

            def __init__(self):
                self.layouts = []

            def open(self) -> None:
                return None

            def render(self, frame: PreviewFrame) -> bool:
                return True

            def close(self) -> None:
                return None

            def set_layout(self, layout: str, reason: str = "") -> None:
                self.layouts.append(layout)

        lines = []
        console = OperatorConsole(mode="operator", default_interval_s=0.0, sink=lines.append)
        sink = LayoutSink()
        manager = PreviewManager(sink=sink, logger=None, operator_console=console)
        manager.configure_preview_mode("FIND_OBJECT", metadata={"layout": "unknown_panel"}, reason="unit")
        self.assertEqual(sink.layouts[-1], "rgb_minimal")
        self.assertTrue(any("layout_unsupported" in line for line in lines))

    def test_track_local_target_summary_found_zero_with_boxes(self):
        manager = PreviewManager(sink=self._ExitSink(), logger=None)
        status = {"mode": "FIND_OBJECT", "target": "apple"}
        local = {
            "has_infer": True,
            "box_count": 3,
            "class_names": ["apple", "banana", "bottle"],
            "infer_boxes": [
                [0, 0, 10, 10, 0.31, 0],
                [0, 0, 10, 10, 0.62, 2],
                [0, 0, 10, 10, 0.44, 1],
            ],
        }
        target = manager._target_overlay(status, local, {"found": False, "target": "apple"})
        line = manager._target_summary_line(status, target)
        self.assertIn("found=0", line)
        self.assertIn("boxes=3", line)
        self.assertIn("best_cls=bottle", line)

    def test_track_local_target_summary_found_one(self):
        manager = PreviewManager(sink=self._ExitSink(), logger=None)
        status = {"mode": "FIND_OBJECT", "target": "apple"}
        local = {"has_infer": True, "rgb_shape": [100, 100, 3], "infer_boxes": [[10, 20, 40, 60, 0.81, 0]], "class_names": ["apple"]}
        target = manager._target_overlay(status, local, {"found": True, "target": "apple", "confidence": 0.81, "cx_norm": 0.54, "bbox": [10, 20, 40, 60]})
        line = manager._target_summary_line(status, target)
        self.assertIn("found=1", line)
        self.assertIn("boxes=1", line)
        self.assertIn("conf=0.81", line)
        self.assertIn("cx=0.54", line)

    def test_target_summary_found_zero_no_boxes_has_reason(self):
        manager = PreviewManager(sink=self._ExitSink(), logger=None)
        status = {"mode": "FIND_OBJECT", "target": "apple"}
        target = manager._target_overlay(status, {"has_infer": True, "box_count": 0, "infer_boxes": []}, {"found": False, "target": "apple"})
        line = manager._target_summary_line(status, target)
        self.assertIn("found=0", line)
        self.assertIn("boxes=0", line)
        self.assertIn("reason=no_boxes", line)

    def test_opencv_preview_same_window_reuses_sink(self):
        class FakeSink:
            sink_name = "opencv"

            def __init__(self):
                self.window_name = "VISTA App Dashboard"

        class FakePreviewManager:
            def __init__(self):
                self.sink = FakeSink()
                self.set_sink_calls = 0
                self.enabled = False

            def set_sink(self, sink):
                self.set_sink_calls += 1
                self.sink = sink

            def enable(self):
                self.enabled = True

            def start_runtime(self):
                return None

        preview = FakePreviewManager()
        supervisor = RuntimeSupervisor(Scheduler(), preview_manager=preview)
        ok = supervisor._configure_preview(
            {"enabled": True, "sink_name": "opencv", "window_name": "VISTA App Dashboard"}
        )
        self.assertTrue(ok)
        self.assertEqual(preview.set_sink_calls, 0)
        self.assertTrue(preview.enabled)

    def test_opencv_preview_different_window_replaces_sink(self):
        class FakeSink:
            sink_name = "opencv"

            def __init__(self):
                self.window_name = "old"

        class FakePreviewManager:
            def __init__(self):
                self.sink = FakeSink()
                self.set_sink_calls = 0

            def set_sink(self, sink):
                self.set_sink_calls += 1
                self.sink = sink

            def enable(self):
                return None

            def start_runtime(self):
                return None

        preview = FakePreviewManager()
        supervisor = RuntimeSupervisor(Scheduler(), preview_manager=preview)
        ok = supervisor._configure_preview(
            {"enabled": True, "sink_name": "opencv", "window_name": "VISTA App Dashboard"}
        )
        self.assertTrue(ok)
        self.assertEqual(preview.set_sink_calls, 1)
        self.assertEqual(getattr(preview.sink, "window_name", ""), "VISTA App Dashboard")


class OperatorConsoleIpcPolicyTest(unittest.TestCase):
    class _RunLogger:
        def __init__(self):
            self.ipc = []

        def write_ipc_record(self, **payload):
            self.ipc.append(dict(payload))

    def _build_app_shell(self, mode="operator"):
        app_module = importlib.import_module("vision_module.app.app")
        app = app_module.VistaApp.__new__(app_module.VistaApp)
        app.run_logger = self._RunLogger()
        app.operator_console_lines = []
        app.operator_console = OperatorConsole(mode=mode, default_interval_s=1.0, sink=app.operator_console_lines.append)
        app.log_lines = []
        app.log = lambda level, src, msg, data=None: app.log_lines.append((level, src, msg, data))
        app.log_info = lambda src, msg, data=None: app.log_lines.append(("info", src, msg, data))
        app.log_warn = lambda src, msg, data=None: app.log_lines.append(("warn", src, msg, data))
        app.log_error = lambda src, msg, data=None: app.log_lines.append(("error", src, msg, data))
        app.current_stage = "IDLE"
        app.current_mode = "IDLE"
        app.current_session_id = None
        app.current_req_id = None
        app.current_epoch = 0
        app.active_interaction_id = None
        app._last_runtime_reconciled_console = ""
        return app, app_module.CONFIG

    def test_target_request_kind_and_sync_reason(self):
        app, _cfg = self._build_app_shell(mode="operator")
        req = VisionReq(
            ts=time.time(),
            op="START",
            stage="SEARCH",
            mode_hint="FIND_OBJECT",
            target="apple",
            payload={"search_kind": "TARGET"},
        )
        self.assertEqual(app._request_kind(req, "SEARCH"), "TARGET")
        self.assertEqual(app._request_sync_reason(req, "SEARCH", "TARGET"), "target_search")

    def test_operator_mode_suppresses_ipc_success_console(self):
        app, cfg = self._build_app_shell(mode="operator")
        old_console_mode, old_ipc_console, old_log_mode, old_debug = (
            cfg.runtime.console_mode,
            cfg.runtime.ipc_console,
            cfg.runtime.log_mode,
            cfg.runtime.debug,
        )
        try:
            cfg.runtime.console_mode = "operator"
            cfg.runtime.ipc_console = False
            cfg.runtime.log_mode = "concise"
            cfg.runtime.debug = False
            for event in ("recv_ok", "send_ok", "enqueue_ok"):
                app._log_ipc_event({"level": "info", "name": "obs_out", "event": event})
            self.assertEqual(app.operator_console_lines, [])
            self.assertEqual(app.log_lines, [])
            self.assertEqual([item["event"] for item in app.run_logger.ipc], ["recv_ok", "send_ok", "enqueue_ok"])
        finally:
            cfg.runtime.console_mode = old_console_mode
            cfg.runtime.ipc_console = old_ipc_console
            cfg.runtime.log_mode = old_log_mode
            cfg.runtime.debug = old_debug

    def test_operator_mode_reports_ipc_connectivity_events(self):
        app, cfg = self._build_app_shell(mode="operator")
        old_console_mode, old_ipc_console, old_log_mode, old_debug = (
            cfg.runtime.console_mode,
            cfg.runtime.ipc_console,
            cfg.runtime.log_mode,
            cfg.runtime.debug,
        )
        try:
            cfg.runtime.console_mode = "operator"
            cfg.runtime.ipc_console = False
            cfg.runtime.log_mode = "concise"
            cfg.runtime.debug = False
            app._log_ipc_event({"level": "info", "name": "obs_out", "event": "connected", "transport": "tcp"})
            app._log_ipc_event({"level": "warn", "name": "obs_out", "event": "connect_failed", "error": "refused"})
            self.assertTrue(any("obs_out connected" in line for line in app.operator_console_lines))
            self.assertTrue(any("obs_out connect_failed" in line for line in app.operator_console_lines))
        finally:
            cfg.runtime.console_mode = old_console_mode
            cfg.runtime.ipc_console = old_ipc_console
            cfg.runtime.log_mode = old_log_mode
            cfg.runtime.debug = old_debug

    def test_full_mode_allows_ipc_success_console(self):
        app, cfg = self._build_app_shell(mode="full")
        old_console_mode, old_ipc_console, old_log_mode, old_debug = (
            cfg.runtime.console_mode,
            cfg.runtime.ipc_console,
            cfg.runtime.log_mode,
            cfg.runtime.debug,
        )
        try:
            cfg.runtime.console_mode = "full"
            cfg.runtime.ipc_console = False
            cfg.runtime.log_mode = "full"
            cfg.runtime.debug = False
            app._log_ipc_event({"level": "info", "name": "obs_out", "event": "send_ok"})
            self.assertEqual(app.log_lines[0][2], "obs_out send_ok")
        finally:
            cfg.runtime.console_mode = old_console_mode
            cfg.runtime.ipc_console = old_ipc_console
            cfg.runtime.log_mode = old_log_mode
            cfg.runtime.debug = old_debug


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

    def test_camera_manager_restores_rgb_after_shared_depth_is_disabled(self):
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
        manager_module = importlib.import_module("vision_module.backend.camera_manager")
        original_cls = manager_module.ColorCamera
        manager_module.ColorCamera = self._SentinelCamera
        try:
            manager = CameraManager(cfg=cfg, logger=PrintLogger("camera_restore"))
            rgb_params = manager._resolve_params("rgb", None)
            depth_params = manager._resolve_params("depth", None)
            manager.cams["rgb"] = manager_module._SharedRgbdStreamProxy(manager, "rgb")
            manager.cams["depth"] = manager_module._SharedRgbdStreamProxy(manager, "depth")
            manager._params["rgb"] = dict(rgb_params or {})
            manager._params["depth"] = dict(depth_params or {})

            self.assertTrue(manager.disable_camera("depth"))

            restored = manager.cams.get("rgb")
            self.assertIsInstance(restored, self._SentinelCamera)
            frame = restored.read_frame()
            self.assertEqual(tuple(frame.shape), (4, 4, 3))
        finally:
            manager_module.ColorCamera = original_cls

    def test_camera_manager_keeps_shared_rgb_alive_when_depth_is_disabled(self):
        class _DummyShared:
            def __init__(self):
                self.released = False

            def release(self):
                self.released = True

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
        manager_module = importlib.import_module("vision_module.backend.camera_manager")
        manager = CameraManager(cfg=cfg, logger=PrintLogger("camera_shared_keepalive"))
        rgb_params = manager._resolve_params("rgb", None)
        depth_params = manager._resolve_params("depth", None)
        shared = _DummyShared()
        manager.cams["rgb"] = manager_module._SharedRgbdStreamProxy(manager, "rgb")
        manager.cams["depth"] = manager_module._SharedRgbdStreamProxy(manager, "depth")
        manager._params["rgb"] = dict(rgb_params or {})
        manager._params["depth"] = dict(depth_params or {})
        manager._specs["rgb"] = manager_module.CameraSpec("rgb", manager_module._freeze_params(rgb_params or {}))
        manager._specs["depth"] = manager_module.CameraSpec("depth", manager_module._freeze_params(depth_params or {}))
        manager._shared_rgbd = shared
        manager._shared_rgbd_signature = manager._shared_signature_for_params(rgb_params or {}, depth_params or {})

        self.assertTrue(manager.disable_camera("depth"))
        manager._ensure_shared_rgbd_if_needed()

        self.assertIs(manager._shared_rgbd, shared)
        self.assertFalse(shared.released)
        self.assertEqual(type(manager.cams.get("rgb")).__name__, "_SharedRgbdStreamProxy")

        self.assertTrue(manager.ensure_camera("depth"))
        self.assertEqual(type(manager.cams.get("depth")).__name__, "_SharedRgbdStreamProxy")


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

        track_rgb = dict(profiles["FIND_OBJECT"].camera_overrides["rgb"])
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
            "mode": "FIND_OBJECT",
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
        original_cls = manager_module.QNN_YOLO_Detect_Predictor
        manager_module.QNN_YOLO_Detect_Predictor = self._DummyPredictor
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
            manager_module.QNN_YOLO_Detect_Predictor = original_cls
            manager.release_all()
            scheduler.stop_runtime()

    def test_table_edge_resumes_after_generation_reset(self):
        scheduler = Scheduler()
        scheduler.start_runtime()
        plan = {
            "mode": "FIND_OBJECT",
            "routes": {
                "camera_frames": {"policy": "slot", "scope": "backend"},
                "local_perception": {"policy": "slot", "scope": "stage"},
                "table_edge_obs": {"policy": "slot", "scope": "stage"},
                "runtime_status": {"policy": "slot", "scope": "backend"},
            },
        }
        scheduler.configure(plan, generation=1)
        generation = {"value": 1}
        manager = TableEdgeManager(logger=PrintLogger("table_edge_generation"))
        manager.bind_runtime(scheduler, lambda: generation["value"])
        try:
            manager.start_runtime()
            scheduler.publish_result("runtime_status", {"stage": "SEARCH", "mode": "FIND_OBJECT", "epoch": 1}, generation=1)
            scheduler.publish_result("camera_frames", {"depth": np.zeros((16, 16), dtype=np.uint16)}, generation=1)
            scheduler.publish_result("camera_frames", {"depth": np.zeros((16, 16), dtype=np.uint16)}, generation=1)
            deadline = time.time() + 1.0
            while time.time() < deadline:
                if manager.snapshot()["last_camera_seq"] >= 2:
                    break
                time.sleep(0.05)
            self.assertGreaterEqual(manager.snapshot()["last_camera_seq"], 2)

            generation["value"] = 2
            scheduler.configure(plan, generation=2)
            scheduler.publish_result("runtime_status", {"stage": "SEARCH", "mode": "FIND_OBJECT", "epoch": 2}, generation=2)
            scheduler.publish_result("camera_frames", {"depth": np.zeros((16, 16), dtype=np.uint16)}, generation=2)
            payload = None
            deadline = time.time() + 1.0
            while time.time() < deadline:
                payload = scheduler.read_result("table_edge_obs", default=None)
                snapshot = manager.snapshot()
                if isinstance(payload, dict) and snapshot["last_camera_generation"] == 2:
                    break
                time.sleep(0.05)
            self.assertIsInstance(payload, dict)
            self.assertEqual(payload["source_mode"], "FIND_OBJECT")
            self.assertEqual(manager.snapshot()["last_camera_generation"], 2)
        finally:
            manager.release_all()
            scheduler.stop_runtime()

    def test_table_edge_latest_frame_slot_drops_old_frames(self):
        scheduler = Scheduler()
        scheduler.start_runtime()
        plan = {
            "mode": "FIND_EDGE",
            "routes": {
                "camera_frames": {"policy": "slot", "scope": "backend"},
                "table_edge_obs": {"policy": "slot", "scope": "stage"},
                "runtime_status": {"policy": "slot", "scope": "backend"},
                "local_perception": {"policy": "slot", "scope": "stage"},
            },
        }
        scheduler.configure(plan, generation=1)
        manager = TableEdgeManager(logger=PrintLogger("table_edge_latest"))
        manager._worker_interval_s = 0.10
        manager._default_interval_s = 0.10
        manager.bind_runtime(scheduler, lambda: 1)
        try:
            manager.start_runtime()
            scheduler.publish_result("camera_frames", {"depth": np.zeros((16, 16), dtype=np.uint16)}, generation=1)
            deadline = time.time() + 1.0
            while time.time() < deadline and manager.snapshot()["processed_frame_count"] < 1:
                time.sleep(0.02)
            for _ in range(5):
                scheduler.publish_result("camera_frames", {"depth": np.zeros((16, 16), dtype=np.uint16)}, generation=1)
            deadline = time.time() + 1.0
            while time.time() < deadline:
                payload = scheduler.read_result("table_edge_obs", default=None)
                if isinstance(payload, dict) and int(payload.get("dropped_frame_count", 0) or 0) >= 1:
                    break
                time.sleep(0.02)
            snapshot = manager.snapshot()
            self.assertGreaterEqual(snapshot["dropped_frame_count"], 1)
            self.assertGreaterEqual(snapshot["processed_frame_count"], 2)
        finally:
            manager.release_all()
            scheduler.stop_runtime()

    def test_plane_only_yolo_gate_does_not_block_table_edge_obs(self):
        from vision_module.config.schema import VisionServiceConfig

        cfg = VisionServiceConfig()
        cfg.table_edge.require_yolo_table_confirm = True
        cfg.table_edge.enable_yolo_in_plane_only = False
        manager = TableEdgeManager(cfg=cfg, logger=PrintLogger("table_edge_yolo_gate"))
        try:
            manager._detector_cfg = SimpleNamespace(plane_only_mode=True)
            gate = manager._yolo_table_confirmation(local={"table_roi_source": "", "rgb_shape": (10, 10, 3)})
            self.assertTrue(gate["yolo_gate_open"])
            self.assertEqual(gate["yolo_gate_reason"], "not_required_plane_only")
        finally:
            manager.release_all()

    def test_table_edge_preview_rate_config_is_low_frequency(self):
        from vision_module.config.schema import VisionServiceConfig

        cfg = VisionServiceConfig()
        self.assertLessEqual(cfg.table_edge.preview_hz, 5.0)
        self.assertLess(cfg.table_edge.preview_hz, cfg.table_edge.target_hz)

    def test_preview_resumes_after_generation_reset(self):
        scheduler = Scheduler()
        scheduler.start_runtime()
        plan = {
            "mode": "FIND_OBJECT",
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
            scheduler.publish_result("runtime_status", {"stage": "SEARCH", "mode": "FIND_OBJECT", "epoch": 1}, generation=1)
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
            scheduler.publish_result("runtime_status", {"stage": "SEARCH", "mode": "FIND_OBJECT", "epoch": 2}, generation=2)
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
