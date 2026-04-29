#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

try:
    from .test_support import PrintLogger, build_test_config
except ImportError:
    from test_support import PrintLogger, build_test_config

from vision_module.backend.predictor_manager import (
    DETECT_BOX_FORMAT,
    LOCAL_PERCEPTION_CONTRACT,
    PredictorManager,
)
from vision_module.backend.predictor.detect_utils import preprocess_img
from vision_module.app.stages.search import _target_obs_from_results as search_target_obs_from_results
from vision_module.backend.scheduler import Scheduler
from vision_module.utils.detect import compute_target_obs


class DetectClassVocabularyTest(unittest.TestCase):
    def test_compute_target_obs_prefers_model_class_names(self):
        boxes = [[20.0, 30.0, 180.0, 260.0, 0.92, 1.0]]
        obs = compute_target_obs(
            frame_shape=(640, 640, 3),
            target="cup",
            det_pred=boxes,
            class_names=("person", "cup"),
        )
        self.assertIsNotNone(obs)
        self.assertEqual(obs["target"], "cup")
        self.assertEqual(obs["bbox"], [20, 30, 180, 260])

    def test_detect_preprocess_preserves_bgr_channel_order(self):
        bgr = np.array([[[11, 22, 33]]], dtype=np.uint8)
        processed = preprocess_img(bgr, target_shape=(1, 1))
        self.assertEqual(processed.shape, (1, 1, 1, 3))
        np.testing.assert_allclose(
            processed[0, 0, 0],
            np.array([11.0, 22.0, 33.0], dtype=np.float32) / 255.0,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_compute_target_obs_falls_back_to_coco80_when_class_names_missing(self):
        boxes = [[20.0, 30.0, 180.0, 260.0, 0.92, 41.0]]
        obs = compute_target_obs(
            frame_shape=(640, 640, 3),
            target="cup",
            det_pred=boxes,
            class_names=None,
        )
        self.assertIsNotNone(obs)
        self.assertEqual(obs["target"], "cup")
        self.assertEqual(obs["bbox"], [20, 30, 180, 260])

    def test_search_target_obs_warns_when_class_name_not_supported(self):
        payload = {
            "local_perception": {
                "has_infer": True,
                "contract_ok": True,
                "rgb_shape": [100, 100, 3],
                "class_names": ["person", "bottle"],
                "box_count": 1,
                "infer_boxes": [[10, 10, 30, 30, 0.62, 1]],
            }
        }
        obs = search_target_obs_from_results(payload, "apple")
        self.assertIsInstance(obs, dict)
        self.assertFalse(obs["found"])
        self.assertTrue(obs["class_not_supported"])
        self.assertIn("class_not_supported target=apple", "\n".join(obs["contract_warnings"]))


class PredictorManagerContractTest(unittest.TestCase):
    class _DummyPredictor:
        def __init__(self, profile):
            self.profile = profile

        def is_ready(self) -> bool:
            return True

        def predict_frame(self, frame):
            _ = frame
            boxes = np.array([[10.0, 20.0, 110.0, 220.0, 0.95, 1.0]], dtype=np.float32)
            masks = np.ones((1, 4, 4), dtype=np.uint8)
            return boxes, masks

        def release(self) -> None:
            return None

    class _BadDetectPredictor:
        def __init__(self, profile):
            self.profile = profile

        def is_ready(self) -> bool:
            return True

        def predict_frame(self, frame):
            _ = frame
            return [[10.0, 20.0, 30.0]], []

        def release(self) -> None:
            return None

    def test_manager_normalizes_numpy_outputs_to_stable_local_perception(self):
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
            model_path="dummy.ctx.bin",
            model_width=640,
            model_height=640,
            conf_thres=0.25,
            iou_thres=0.15,
            class_num=2,
        )
        cfg = build_test_config(args)
        cfg.runtime.capability_placeholder = False
        profile = cfg.model.profiles["test_model"]
        profile.class_num = 2
        profile.classes = ("person", "cup")
        profile.predictor_type = "detect"

        scheduler = Scheduler()
        scheduler.start_runtime()
        scheduler.configure(
            {
                "mode": "TRACK_LOCAL",
                "routes": {
                    "camera_frames": {"policy": "slot", "scope": "backend"},
                    "local_perception": {"policy": "slot", "scope": "stage"},
                },
            },
            generation=1,
        )

        manager = PredictorManager(cfg=cfg, logger=PrintLogger("detect_contract"))
        manager.bind_runtime(scheduler, lambda: 1)
        manager_module = importlib.import_module("vision_module.backend.predictor_manager")
        original_cls = manager_module.QNN_YOLO_Dectec_Predictor
        manager_module.QNN_YOLO_Dectec_Predictor = self._DummyPredictor
        try:
            self.assertTrue(manager.ensure_model("test_model"))
            manager.set_inference_enabled(True)
            manager.start_runtime()
            scheduler.publish_result(
                "camera_frames",
                {"rgb": np.zeros((64, 64, 3), dtype=np.uint8)},
                generation=1,
            )

            payload = None
            deadline = time.time() + 1.0
            while time.time() < deadline:
                payload = scheduler.read_result("local_perception", default=None)
                if isinstance(payload, dict) and payload.get("box_count") == 1:
                    break
                time.sleep(0.05)

            self.assertIsInstance(payload, dict)
            self.assertEqual(payload["contract"], LOCAL_PERCEPTION_CONTRACT)
            self.assertEqual(payload["infer_box_format"], DETECT_BOX_FORMAT)
            self.assertEqual(payload["class_names"], ["person", "cup"])
            self.assertEqual(payload["box_count"], 1)
            self.assertIsInstance(payload["infer_boxes"], list)
            self.assertIsInstance(payload["infer_boxes"][0], list)
            self.assertEqual(int(payload["infer_boxes"][0][5]), 1)
            self.assertIsInstance(payload["infer_masks"], list)
        finally:
            manager_module.QNN_YOLO_Dectec_Predictor = original_cls
            manager.release_all()
            scheduler.stop_runtime()

    def test_manager_surfaces_contract_error_for_malformed_detect_rows(self):
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
            model_path="dummy.ctx.bin",
            model_width=640,
            model_height=640,
            conf_thres=0.25,
            iou_thres=0.15,
            class_num=80,
        )
        cfg = build_test_config(args)
        profile = cfg.model.profiles["test_model"]
        profile.class_num = 80
        profile.classes = None
        profile.predictor_type = "detect"

        scheduler = Scheduler()
        scheduler.start_runtime()
        scheduler.configure(
            {
                "mode": "TRACK_LOCAL",
                "routes": {
                    "camera_frames": {"policy": "slot", "scope": "backend"},
                    "local_perception": {"policy": "slot", "scope": "stage"},
                },
            },
            generation=1,
        )

        manager = PredictorManager(cfg=cfg, logger=PrintLogger("detect_bad_contract"))
        manager.bind_runtime(scheduler, lambda: 1)
        manager_module = importlib.import_module("vision_module.backend.predictor_manager")
        original_cls = manager_module.QNN_YOLO_Dectec_Predictor
        manager_module.QNN_YOLO_Dectec_Predictor = self._BadDetectPredictor
        try:
            self.assertTrue(manager.ensure_model("test_model"))
            manager.set_inference_enabled(True)
            manager.start_runtime()
            scheduler.publish_result(
                "camera_frames",
                {"rgb": np.zeros((64, 64, 3), dtype=np.uint8)},
                generation=1,
            )

            payload = None
            deadline = time.time() + 1.0
            while time.time() < deadline:
                payload = scheduler.read_result("local_perception", default=None)
                if isinstance(payload, dict) and payload.get("has_infer"):
                    break
                time.sleep(0.05)

            self.assertIsInstance(payload, dict)
            self.assertFalse(payload["contract_ok"])
            self.assertEqual(payload["box_count"], 0)
            self.assertIn("expected >=6 values", payload["contract_error"])
            self.assertEqual(payload["class_names_source"], "fallback_coco80")

            target_obs = search_target_obs_from_results({"local_perception": payload}, "cup")
            self.assertIsInstance(target_obs, dict)
            self.assertFalse(target_obs["found"])
            self.assertIn("contract_error", target_obs)
        finally:
            manager_module.QNN_YOLO_Dectec_Predictor = original_cls
            manager.release_all()
            scheduler.stop_runtime()


class PredictorBackendSelectionContractTest(unittest.TestCase):
    class _SentinelPredictor:
        def __init__(self, profile):
            self.profile = profile

        def is_ready(self) -> bool:
            return True

        def predict_frame(self, frame):
            _ = frame
            return [], []

        def release(self) -> None:
            return None

    def test_manager_backend_selection_does_not_follow_capability_placeholder(self):
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
            model_path="dummy.ctx.bin",
            model_width=640,
            model_height=640,
            conf_thres=0.25,
            iou_thres=0.15,
            class_num=80,
        )
        cfg = build_test_config(args)
        cfg.runtime.capability_placeholder = True
        profile = cfg.model.profiles["test_model"]
        profile.predictor_type = "detect"

        manager_module = importlib.import_module("vision_module.backend.predictor_manager")
        original_cls = manager_module.QNN_YOLO_Dectec_Predictor
        manager_module.QNN_YOLO_Dectec_Predictor = self._SentinelPredictor
        try:
            manager = PredictorManager(cfg=cfg, logger=PrintLogger("backend_contract"))
            self.assertIs(manager._predictor_class_for_profile(profile), self._SentinelPredictor)
        finally:
            manager_module.QNN_YOLO_Dectec_Predictor = original_cls


class PredictorBackendStatusTest(unittest.TestCase):
    def test_auto_backend_status_is_explicit_on_windows_resolution(self):
        predictor_module = importlib.import_module("vision_module.backend.predictor")
        try:
            with patch.dict(os.environ, {"VISTA_BACKEND": "auto"}, clear=False):
                with patch("platform.system", return_value="Windows"):
                    predictor_module = importlib.reload(predictor_module)
                    status = predictor_module.predictor_backend_status()
            self.assertEqual(status["requested_backend"], "auto")
            self.assertEqual(status["resolved_backend"], "mock")
            self.assertIn("Windows", status["note"])
        finally:
            importlib.reload(predictor_module)


if __name__ == "__main__":
    unittest.main()
