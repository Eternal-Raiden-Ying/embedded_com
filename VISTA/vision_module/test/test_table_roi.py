#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from types import SimpleNamespace

import numpy as np

from VISTA.vision_module.backend.table_edge_manager import TableEdgeManager
from VISTA.vision_module.utils.table_roi import (
    bbox_to_quadrant,
    build_table_roi,
    find_table_bbox,
    quadrant_to_roi,
)


class TableRoiTest(unittest.TestCase):
    def test_find_table_bbox_from_class_name_and_id(self):
        self.assertEqual(
            find_table_bbox({"infer_boxes": [[1, 2, 30, 40, 0.9, 12, "apple"], [10, 20, 110, 120, 0.8, 5, "diningtable"]]}),
            [10, 20, 110, 120],
        )
        self.assertEqual(
            find_table_bbox({"infer_boxes": [[100, 120, 420, 360, 0.7, 60]]}),
            [100, 120, 420, 360],
        )
        self.assertEqual(
            find_table_bbox({"infer_boxes": [[100, 120, 420, 360, 0.7, "bad", "desk"]]}),
            [100, 120, 420, 360],
        )

    def test_bbox_center_to_quadrant_and_roi(self):
        self.assertEqual(bbox_to_quadrant([100, 120, 420, 360], (640, 640, 3)), "LT")
        self.assertEqual(bbox_to_quadrant([360, 120, 620, 360], (640, 640, 3)), "RT")
        self.assertEqual(bbox_to_quadrant([100, 360, 420, 620], (640, 640, 3)), "LB")
        self.assertEqual(bbox_to_quadrant([360, 360, 620, 620], (640, 640, 3)), "RB")
        self.assertEqual(quadrant_to_roi("LT", 640, 480), [0, 0, 320, 240])
        self.assertEqual(quadrant_to_roi("RB", 640, 480), [320, 240, 640, 480])

    def test_build_table_roi_and_fallback(self):
        roi = build_table_roi(
            {"mock_table_bbox": "100,120,420,360"},
            (640, 640, 3),
            (480, 640),
            [11, 22, 33, 44],
        )
        self.assertEqual(roi["table_bbox"], [100, 120, 420, 360])
        self.assertEqual(roi["table_center_norm"], [0.40625, 0.375])
        self.assertEqual(roi["table_quadrant"], "LT")
        self.assertEqual(roi["rgb_search_roi"], [0, 0, 320, 320])
        self.assertEqual(roi["depth_edge_roi"], [0, 0, 320, 240])
        self.assertEqual(roi["table_roi_source"], "yolo_table_bbox")

        fallback = build_table_roi({"infer_boxes": []}, (640, 640, 3), (480, 640), [11, 22, 33, 44])
        self.assertIsNone(fallback["table_bbox"])
        self.assertIsNone(fallback["table_quadrant"])
        self.assertIsNone(fallback["rgb_search_roi"])
        self.assertEqual(fallback["depth_edge_roi"], [11, 22, 33, 44])


class TableEdgeManagerDynamicRoiTest(unittest.TestCase):
    class _Scheduler:
        def __init__(self):
            self.local = {}

        def read_result(self, route, default=None):
            if route == "local_perception":
                return self.local
            return default

    def setUp(self):
        self.manager = TableEdgeManager()
        self.manager._detector_cfg = SimpleNamespace(roi_x0=10, roi_y0=20, roi_x1=110, roi_y1=120)
        self.scheduler = self._Scheduler()
        self.manager.bind_runtime(self.scheduler, lambda: 1)
        self.depth = np.zeros((480, 640), dtype=np.uint16)

    def test_table_bbox_generates_dynamic_depth_roi(self):
        self.scheduler.local = {
            "rgb_shape": (640, 640, 3),
            "infer_boxes": [[400, 380, 620, 620, 0.9, 60]],
        }
        roi = self.manager._select_roi(self.depth)
        self.assertEqual(roi["table_bbox"], [400, 380, 620, 620])
        self.assertEqual(roi["table_center_norm"], [0.796875, 0.78125])
        self.assertEqual(roi["table_quadrant"], "RB")
        self.assertEqual(roi["depth_edge_roi"], [320, 240, 640, 480])
        self.assertEqual(roi["roi_source"], "local_perception_table_bbox")
        self.assertEqual(roi["roi_reason"], "table_bbox_detected")

    def test_missing_bbox_uses_last_valid_quadrant(self):
        self.scheduler.local = {
            "rgb_shape": (640, 640, 3),
            "infer_boxes": [[20, 350, 260, 620, 0.9, "x", "diningtable"]],
        }
        first = self.manager._select_roi(self.depth)
        self.assertEqual(first["table_quadrant"], "LB")
        self.scheduler.local = {"rgb_shape": (640, 640, 3), "infer_boxes": []}
        roi = self.manager._select_roi(self.depth)
        self.assertEqual(roi["table_quadrant"], "LB")
        self.assertEqual(roi["depth_edge_roi"], [0, 240, 320, 480])
        self.assertEqual(roi["roi_source"], "last_valid_quadrant")
        self.assertEqual(roi["roi_reason"], "table_bbox_lost_using_history")

    def test_no_bbox_and_no_history_falls_back_to_static_roi(self):
        self.scheduler.local = {"rgb_shape": (640, 640, 3), "infer_boxes": []}
        roi = self.manager._select_roi(self.depth)
        self.assertIsNone(roi["table_bbox"])
        self.assertIsNone(roi["table_quadrant"])
        self.assertEqual(roi["depth_edge_roi"], [10, 20, 110, 120])
        self.assertEqual(roi["roi_source"], "static_fallback")
        self.assertEqual(roi["roi_reason"], "table_bbox_unavailable")


if __name__ == "__main__":
    unittest.main()
