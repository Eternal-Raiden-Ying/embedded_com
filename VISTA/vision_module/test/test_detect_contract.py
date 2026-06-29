#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
import os
import sys
import time
import unittest
from pathlib import Path
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
from vision_module.app.stages.search.target_obs_builder import target_obs_from_results
from vision_module.backend.scheduler import Scheduler
from vision_module.utils.detect import compute_target_obs


FINETUNE_CLASSES = (
    "table1",
    "apple",
    "banana",
    "basket",
    "bottle",
    "grape",
    "key",
    "kiwi fruit",
    "lemon",
    "mango",
    "mouse",
    "orange",
    "peach",
    "star fruit",
    "strawberry",
)


def _import_yolo26_module():
    sys.modules.setdefault("aidlite", SimpleNamespace())
    return importlib.import_module("vision_module.backend.predictor.QNN_YOLO26_Detect_Predictor")


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

    def test_board_config_uses_finetune_bgr15_model(self):
        from vision_module.config.board_config import CONFIG

        profile = CONFIG.model.profiles["yolo26s_detect"]
        self.assertEqual(profile.class_num, 15)
        self.assertEqual(tuple(profile.classes), FINETUNE_CLASSES)
        self.assertTrue(profile.target_model.endswith(
            "VISTA/vision_module/model/yolo26s/models/finetune/"
            "yolo26s-cutoff-bgr_qcs6490_w8a8.qnn236.ctx.bin"
        ))
        self.assertTrue(Path(profile.target_model).is_file())

    def test_yolo26_target_class_map_keeps_apple_at_finetune_index(self):
        from vision_module.config.board_config import CONFIG

        profile = CONFIG.model.profiles["yolo26s_detect"]
        class_names = tuple(profile.classes)
        self.assertEqual(profile.class_num, len(class_names))
        self.assertIn("apple", class_names)
        self.assertEqual(class_names.index("apple"), 1)
        self.assertNotIn("tag_home", class_names)
        self.assertNotIn("tag_station", class_names)

        local = {
            "has_infer": True,
            "rgb_shape": [640, 640, 3],
            "infer_boxes": [[20.0, 30.0, 180.0, 260.0, 0.92, 1.0]],
            "class_names": list(class_names),
        }
        obs = target_obs_from_results({"local_perception": local}, "apple")
        self.assertIsNotNone(obs)
        self.assertTrue(obs["target_found"])
        self.assertEqual(obs["matched_cls"], "apple")
        self.assertEqual(local["class_names"], list(class_names))

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

    def test_yolo26_preprocess_preserves_bgr_channel_order(self):
        module = _import_yolo26_module()
        bgr = np.array([[[11, 22, 33]]], dtype=np.uint8)
        processed, meta = module._yolo26s_preprocess(bgr, input_size=1)
        self.assertEqual(processed.shape, (1, 1, 1, 3))
        self.assertAlmostEqual(meta["scale"], 1.0)
        self.assertEqual(meta["draw_space"], "crop")
        np.testing.assert_allclose(
            processed[0, 0, 0],
            np.array([11.0, 22.0, 33.0], dtype=np.float32) / 255.0,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_yolo26_bbox_map_normalized_cxcywh(self):
        module = _import_yolo26_module()
        meta = {
            "crop_w": 640,
            "crop_h": 640,
            "model_w": 640,
            "model_h": 640,
            "resize_scale": 1.0,
            "pad_x": 0.0,
            "pad_y": 0.0,
            "draw_space": "crop",
        }
        mapped = module.map_bbox_from_model_to_image(
            [0.5, 0.5, 0.5, 0.5],
            model_input_shape=(640, 640),
            preprocess_meta=meta,
            draw_image_shape=(640, 640, 3),
            output_format="cxcywh",
            normalized=True,
        )
        np.testing.assert_allclose(mapped, np.array([160, 160, 480, 480], dtype=np.float32), atol=1e-5)

    def test_yolo26_bbox_map_normalized_crop_basis_non_square_640x480(self):
        module = _import_yolo26_module()
        meta = {
            "crop_w": 640,
            "crop_h": 480,
            "model_w": 640,
            "model_h": 640,
            "resize_scale": 1.0,
            "pad_x": 0.0,
            "pad_y": 0.0,
            "draw_space": "crop",
        }
        mapped = module.map_bbox_from_model_to_image(
            [0.5, 0.5, 0.5, 0.5],
            model_input_shape=(640, 640),
            preprocess_meta=meta,
            draw_image_shape=(480, 640, 3),
            output_format="cxcywh",
            normalized=True,
            normalized_basis="crop",
        )
        np.testing.assert_allclose(mapped, np.array([160, 120, 480, 360], dtype=np.float32), atol=1e-5)

    def test_yolo26_bbox_map_normalized_crop_basis_non_square_640x360(self):
        module = _import_yolo26_module()
        meta = {
            "crop_w": 640,
            "crop_h": 360,
            "model_w": 640,
            "model_h": 640,
            "resize_scale": 1.0,
            "pad_x": 0.0,
            "pad_y": 0.0,
            "draw_space": "crop",
        }
        mapped = module.map_bbox_from_model_to_image(
            [0.5, 0.5, 0.5, 0.5],
            model_input_shape=(640, 640),
            preprocess_meta=meta,
            draw_image_shape=(360, 640, 3),
            output_format="cxcywh",
            normalized=True,
            normalized_basis="crop",
        )
        np.testing.assert_allclose(mapped, np.array([160, 90, 480, 270], dtype=np.float32), atol=1e-5)

    def test_yolo26_bbox_map_removes_letterbox_padding(self):
        module = _import_yolo26_module()
        meta = {
            "crop_w": 640,
            "crop_h": 480,
            "model_w": 640,
            "model_h": 640,
            "resize_scale": 1.0,
            "pad_x": 0.0,
            "pad_y": 80.0,
            "draw_space": "crop",
        }
        mapped = module.map_bbox_from_model_to_image(
            [0.5, 0.5, 0.5, 0.5],
            model_input_shape=(640, 640),
            preprocess_meta=meta,
            draw_image_shape=(480, 640, 3),
            output_format="cxcywh",
            normalized=True,
        )
        np.testing.assert_allclose(mapped, np.array([160, 80, 480, 400], dtype=np.float32), atol=1e-5)

    def test_yolo26_bbox_map_crop_offset_only_for_full_frame(self):
        module = _import_yolo26_module()
        meta = {
            "crop_x0": 100,
            "crop_y0": 50,
            "crop_w": 320,
            "crop_h": 240,
            "full_w": 800,
            "full_h": 600,
            "model_w": 640,
            "model_h": 640,
            "resize_scale": 2.0,
            "pad_x": 0.0,
            "pad_y": 0.0,
        }
        crop = module.map_bbox_from_model_to_image(
            [160, 160, 480, 480],
            model_input_shape=(640, 640),
            preprocess_meta={**meta, "draw_space": "crop"},
            draw_image_shape=(240, 320, 3),
            output_format="xyxy",
            normalized=False,
            draw_space="crop",
        )
        full = module.map_bbox_from_model_to_image(
            [160, 160, 480, 480],
            model_input_shape=(640, 640),
            preprocess_meta={**meta, "draw_space": "full_frame"},
            draw_image_shape=(600, 800, 3),
            output_format="xyxy",
            normalized=False,
            draw_space="full_frame",
        )
        np.testing.assert_allclose(crop, np.array([80, 80, 240, 240], dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(full, np.array([180, 130, 340, 290], dtype=np.float32), atol=1e-5)

    def test_yolo26_bbox_map_distinguishes_xyxy_and_cxcywh(self):
        module = _import_yolo26_module()
        meta = {
            "crop_w": 640,
            "crop_h": 640,
            "model_w": 640,
            "model_h": 640,
            "resize_scale": 1.0,
            "pad_x": 0.0,
            "pad_y": 0.0,
            "draw_space": "crop",
        }
        xyxy = module.map_bbox_from_model_to_image(
            [160, 160, 480, 480],
            model_input_shape=(640, 640),
            preprocess_meta=meta,
            draw_image_shape=(640, 640, 3),
            output_format="xyxy",
            normalized=False,
        )
        cxcywh = module.map_bbox_from_model_to_image(
            [320, 320, 320, 320],
            model_input_shape=(640, 640),
            preprocess_meta=meta,
            draw_image_shape=(640, 640, 3),
            output_format="cxcywh",
            normalized=False,
        )
        np.testing.assert_allclose(xyxy, cxcywh, atol=1e-5)

    def test_yolo26_decode_modes_for_same_raw_bbox(self):
        module = _import_yolo26_module()
        meta = {
            "crop_w": 640,
            "crop_h": 480,
            "model_w": 640,
            "model_h": 640,
            "resize_scale": 1.0,
            "resize_scale_x": 1.0,
            "resize_scale_y": 1.0,
            "pad_x": 0.0,
            "pad_y": 0.0,
            "draw_space": "crop",
        }
        raw = [0.5, 0.5, 0.5, 0.5]
        crop = module.decode_bbox_with_mode(raw, "cxcywh_norm_crop", (640, 640), meta, (480, 640, 3))
        model = module.decode_bbox_with_mode(raw, "cxcywh_norm_model_square", (640, 640), meta, (480, 640, 3))
        xyxy = module.decode_bbox_with_mode(raw, "xyxy_norm_crop", (640, 640), meta, (480, 640, 3))
        xywh = module.decode_bbox_with_mode(raw, "xywh_norm_crop", (640, 640), meta, (480, 640, 3))
        np.testing.assert_allclose(crop, np.array([160, 120, 480, 360], dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(model, np.array([160, 160, 480, 480], dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(xyxy, np.array([320, 240, 320, 240], dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(xywh, np.array([320, 240, 640, 480], dtype=np.float32), atol=1e-5)

    def test_yolo26_direct_resize_preprocess_inverse(self):
        module = _import_yolo26_module()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        processed, meta = module._yolo26s_preprocess(image, input_size=640, preprocess_mode="direct_resize")
        self.assertEqual(processed.shape, (1, 640, 640, 3))
        self.assertEqual(meta["preprocess_mode"], "direct_resize")
        self.assertAlmostEqual(meta["resize_scale_x"], 1.0)
        self.assertAlmostEqual(meta["resize_scale_y"], 640.0 / 480.0)
        mapped = module.decode_bbox_with_mode(
            [160, 160, 480, 480],
            "xyxy_pixel_model_square",
            (640, 640),
            meta,
            (480, 640, 3),
        )
        np.testing.assert_allclose(mapped, np.array([160, 120, 480, 360], dtype=np.float32), atol=1e-5)

    def test_yolo26_square_fill_top_left_preprocess_inverse(self):
        module = _import_yolo26_module()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        processed, meta = module._yolo26s_preprocess(image, input_size=640, preprocess_mode="square_fill_top_left")
        self.assertEqual(processed.shape, (1, 640, 640, 3))
        self.assertEqual(meta["preprocess_mode"], "square_fill_top_left")
        self.assertAlmostEqual(meta["resize_scale_x"], 1.0)
        self.assertAlmostEqual(meta["resize_scale_y"], 1.0)
        mapped = module.decode_bbox_with_mode(
            [160, 160, 480, 480],
            "xyxy_pixel_model_square",
            (640, 640),
            meta,
            (480, 640, 3),
        )
        np.testing.assert_allclose(mapped, np.array([160, 160, 480, 480], dtype=np.float32), atol=1e-5)

    def test_yolo26_unclipped_and_clipped_bbox_are_separate(self):
        module = _import_yolo26_module()
        meta = {
            "crop_w": 640,
            "crop_h": 480,
            "model_w": 640,
            "model_h": 640,
            "resize_scale": 1.0,
            "resize_scale_x": 1.0,
            "resize_scale_y": 1.0,
            "pad_x": 0.0,
            "pad_y": 0.0,
            "draw_space": "crop",
        }
        unclipped = module.decode_bbox_with_mode(
            [-100, -50, 800, 700],
            "xyxy_pixel_model_square",
            (640, 640),
            meta,
            (480, 640, 3),
            clip=False,
        )
        clipped = module.decode_bbox_with_mode(
            [-100, -50, 800, 700],
            "xyxy_pixel_model_square",
            (640, 640),
            meta,
            (480, 640, 3),
            clip=True,
        )
        np.testing.assert_allclose(unclipped, np.array([-100, -50, 800, 700], dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(clipped, np.array([0, 0, 640, 480], dtype=np.float32), atol=1e-5)

    def test_yolo26_clip_ratio_detects_out_of_frame_bbox(self):
        module = _import_yolo26_module()
        unclipped = np.array([-100, -50, 800, 700], dtype=np.float32)
        clipped = np.array([0, 0, 640, 480], dtype=np.float32)
        self.assertGreater(module._clip_ratio(unclipped, clipped, (480, 640, 3)), 0.0)
        self.assertEqual(module._clip_ratio(clipped, clipped, (480, 640, 3)), 0.0)

    def test_yolo26_merge_outputs_uses_finetune_class_num(self):
        module = _import_yolo26_module()
        bbox = np.zeros((1, 4, 8400), dtype=np.float32)
        scores = np.zeros((1, 15, 8400), dtype=np.float32)
        scores[0, 14, 3] = 0.91
        merged = module._yolo26s_merge_outputs(bbox, scores, class_num=15)
        self.assertEqual(merged.shape, (8400, 19))
        self.assertAlmostEqual(float(merged[3, 18]), 0.91, places=6)

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

    def test_compute_target_obs_matches_finetune_targets_and_rejects_removed_aliases(self):
        boxes = [
            [20.0, 30.0, 180.0, 260.0, 0.92, 4.0],
            [200.0, 40.0, 300.0, 180.0, 0.88, 6.0],
        ]
        bottle = compute_target_obs(
            frame_shape=(640, 640, 3),
            target="bottle",
            det_pred=boxes,
            class_names=FINETUNE_CLASSES,
        )
        key = compute_target_obs(
            frame_shape=(640, 640, 3),
            target="keys",
            det_pred=boxes,
            class_names=FINETUNE_CLASSES,
        )
        cup = compute_target_obs(
            frame_shape=(640, 640, 3),
            target="cup",
            det_pred=boxes,
            class_names=FINETUNE_CLASSES,
        )
        self.assertIsNotNone(bottle)
        self.assertEqual(bottle["matched_cls"], "bottle")
        self.assertIsNotNone(key)
        self.assertEqual(key["matched_cls"], "key")
        self.assertIsNone(cup)

    def test_compute_target_obs_matches_target_not_top1_box(self):
        boxes = [
            [10.0, 20.0, 220.0, 260.0, 0.92, 0.0],
            [250.0, 240.0, 360.0, 420.0, 0.78, 1.0],
            [380.0, 120.0, 520.0, 300.0, 0.60, 2.0],
        ]
        obs = compute_target_obs(
            frame_shape=(640, 640, 3),
            target="apple",
            det_pred=boxes,
            class_names=("keyboard", "apple", "mouse"),
        )
        self.assertIsNotNone(obs)
        self.assertTrue(obs["target_found"])
        self.assertEqual(obs["matched_cls"], "apple")
        self.assertAlmostEqual(obs["matched_conf"], 0.78)
        self.assertEqual(obs["matched_rank_in_all_boxes"], 2)
        self.assertEqual(obs["num_target_candidates"], 1)
        self.assertEqual(obs["best_cls"], "keyboard")
        self.assertAlmostEqual(obs["best_conf"], 0.92)

    def test_compute_target_obs_separates_full_center_from_offset(self):
        boxes = [[1200.0, 340.0, 1270.0, 388.0, 0.90, 1.0]]
        obs = compute_target_obs(
            frame_shape=(720, 1280, 3),
            target="apple",
            det_pred=boxes,
            class_names=("keyboard", "apple"),
        )
        self.assertIsNotNone(obs)
        full = obs["matched_center_full_norm"]
        offset = obs["matched_center_offset_norm"]
        self.assertGreaterEqual(full["cx"], 0.0)
        self.assertLessEqual(full["cx"], 1.0)
        self.assertGreaterEqual(full["cy"], 0.0)
        self.assertLessEqual(full["cy"], 1.0)
        self.assertLess(offset["dx"], 0.0)
        self.assertAlmostEqual(obs["matched_center"]["cx"], full["cx"])
        self.assertAlmostEqual(obs["cx_norm"], offset["dx"])

    def test_compute_target_obs_marks_out_of_frame_bbox(self):
        boxes = [[-10.0, 10.0, 40.0, 60.0, 0.90, 1.0]]
        obs = compute_target_obs(
            frame_shape=(100, 100, 3),
            target="apple",
            det_pred=boxes,
            class_names=("keyboard", "apple"),
        )
        self.assertIsNotNone(obs)
        self.assertFalse(obs["bbox_valid"])
        self.assertEqual(obs["bbox_invalid_reason"], "bbox_out_of_frame")

    def test_search_target_obs_reports_best_when_target_absent(self):
        payload = {
            "local_perception": {
                "has_infer": True,
                "contract_ok": True,
                "rgb_shape": [640, 640, 3],
                "class_names": ["keyboard", "apple", "mouse"],
                "box_count": 2,
                "infer_boxes": [
                    [10, 20, 220, 260, 0.92, 0],
                    [380, 120, 520, 300, 0.60, 2],
                ],
            }
        }
        obs = search_target_obs_from_results(payload, "apple")
        self.assertIsInstance(obs, dict)
        self.assertFalse(obs["found"])
        self.assertFalse(obs["target_found"])
        self.assertEqual(obs["best_cls"], "keyboard")
        self.assertAlmostEqual(obs["best_conf"], 0.92)
        self.assertIsNone(obs["matched_cls"])
        self.assertEqual(obs["num_target_candidates"], 0)
        self.assertEqual(obs["reason"], "no_target_candidate")

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
                "mode": "FIND_OBJECT",
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
        original_cls = manager_module.QNN_YOLO_Detect_Predictor
        manager_module.QNN_YOLO_Detect_Predictor = self._DummyPredictor
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
            self.assertTrue(payload["yolo_has_infer"])
            self.assertIsInstance(payload["yolo_infer_ms"], float)
            self.assertGreaterEqual(payload["yolo_infer_ms"], 0.0)
            self.assertIsInstance(payload["yolo_roi_ms"], float)
            self.assertGreaterEqual(payload["yolo_roi_ms"], 0.0)
        finally:
            manager_module.QNN_YOLO_Detect_Predictor = original_cls
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
                "mode": "FIND_OBJECT",
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
        original_cls = manager_module.QNN_YOLO_Detect_Predictor
        manager_module.QNN_YOLO_Detect_Predictor = self._BadDetectPredictor
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
            manager_module.QNN_YOLO_Detect_Predictor = original_cls
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
        original_cls = manager_module.QNN_YOLO_Detect_Predictor
        manager_module.QNN_YOLO_Detect_Predictor = self._SentinelPredictor
        try:
            manager = PredictorManager(cfg=cfg, logger=PrintLogger("backend_contract"))
            self.assertIs(manager._predictor_class_for_profile(profile), self._SentinelPredictor)
        finally:
            manager_module.QNN_YOLO_Detect_Predictor = original_cls


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
