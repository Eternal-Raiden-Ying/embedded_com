#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

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
        self.assertEqual(roi["table_quadrant"], "LT")
        self.assertEqual(roi["rgb_search_roi"], [0, 0, 320, 320])
        self.assertEqual(roi["depth_edge_roi"], [0, 0, 320, 240])
        self.assertEqual(roi["table_roi_source"], "yolo_table_bbox")

        fallback = build_table_roi({"infer_boxes": []}, (640, 640, 3), (480, 640), [11, 22, 33, 44])
        self.assertIsNone(fallback["table_bbox"])
        self.assertIsNone(fallback["table_quadrant"])
        self.assertIsNone(fallback["rgb_search_roi"])
        self.assertEqual(fallback["depth_edge_roi"], [11, 22, 33, 44])


if __name__ == "__main__":
    unittest.main()
