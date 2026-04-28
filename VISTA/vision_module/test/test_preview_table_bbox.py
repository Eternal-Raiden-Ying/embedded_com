#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import types
import unittest

import numpy as np


class _FakeCV2(types.SimpleNamespace):
    WINDOW_NORMAL = 0
    COLOR_GRAY2BGR = 1
    COLOR_RGB2BGR = 2
    COLORMAP_TURBO = 3
    COLORMAP_JET = 4
    FONT_HERSHEY_SIMPLEX = 5
    INTER_AREA = 6


sys.modules.setdefault("cv2", _FakeCV2())

from VISTA.vision_module.backend.preview.opencv_sink import CYAN, YELLOW, OpenCVPreviewSink


class PreviewTableBboxTest(unittest.TestCase):
    def setUp(self):
        self._old_env = {
            "VISTA_TABLE_BBOX_ENABLE": os.environ.get("VISTA_TABLE_BBOX_ENABLE"),
            "VISTA_MOCK_TABLE_BBOX": os.environ.get("VISTA_MOCK_TABLE_BBOX"),
        }

    def tearDown(self):
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_table_bbox_disable_uses_unavailable_fallback(self):
        os.environ["VISTA_TABLE_BBOX_ENABLE"] = "0"
        os.environ["VISTA_MOCK_TABLE_BBOX"] = "10,20,110,220"
        sink = OpenCVPreviewSink()
        self.assertIsNone(sink._find_table_bbox({"table_bbox": [1, 2, 3, 4]}))
        self.assertIsNone(sink._quadrant_roi(None, {"table_quadrant": "LT"}, {}))

    def test_mock_table_bbox_and_quadrant_roi_change(self):
        os.environ["VISTA_TABLE_BBOX_ENABLE"] = "1"
        os.environ["VISTA_MOCK_TABLE_BBOX"] = "1"
        sink = OpenCVPreviewSink()
        bbox = sink._find_table_bbox({"rgb_shape": (240, 320, 3)})
        self.assertEqual(bbox, [80, 120, 240, 216])
        self.assertEqual(sink._find_table_bbox({}, (240, 320, 3)), [80, 120, 240, 216])
        self.assertEqual(sink._quadrant_roi(bbox, {"table_quadrant": "LT"}, {}), [80, 120, 160, 168])
        self.assertEqual(sink._quadrant_roi(bbox, {"table_quadrant": "top_left"}, {}), [80, 120, 160, 168])
        self.assertEqual(sink._quadrant_roi(bbox, {"table_quadrant": "RB"}, {}), [160, 168, 240, 216])

    def test_preview_colors_for_table_and_roi_overlays(self):
        self.assertEqual(YELLOW, (0, 255, 255))
        self.assertEqual(CYAN, (255, 255, 0))

    def test_rgb_panel_overlay_styles(self):
        class RecordingSink(OpenCVPreviewSink):
            def __init__(self):
                super().__init__()
                self.draws = []
                self.notes = []

            def _fit_with_transform(self, image, size):
                return np.zeros((size[1], size[0], 3), dtype=np.uint8), 1.0, (0, 0)

            def _title(self, panel, title):
                return None

            def _draw_roi(self, panel, roi, scale, offset, label, color, dashed=False):
                self.draws.append((label, list(roi), color, dashed))

            def _corner_note(self, panel, text, fg=(230, 230, 230)):
                self.notes.append((text, fg))

        os.environ["VISTA_TABLE_BBOX_ENABLE"] = "1"
        os.environ.pop("VISTA_MOCK_TABLE_BBOX", None)
        sink = RecordingSink()
        sink._make_rgb_panel(
            np.zeros((100, 100, 3), dtype=np.uint8),
            {
                "local_perception": {
                    "table_bbox": [10, 20, 80, 90],
                    "search_roi": [1, 2, 30, 40],
                    "table_quadrant": "RB",
                },
                "target_obs": {"bbox": [5, 6, 7, 8]},
            },
            (120, 120),
        )
        self.assertIn(("table_quadrant=RB", YELLOW), sink.notes)
        self.assertIn(("table_bbox", [10, 20, 80, 90], YELLOW, False), sink.draws)
        self.assertIn(("search_roi", [1, 2, 30, 40], YELLOW, True), sink.draws)
        self.assertIn(("quadrant_roi", [45, 55, 80, 90], CYAN, True), sink.draws)
        self.assertIn(("target_bbox", [5, 6, 7, 8], (90, 180, 255), False), sink.draws)

    def test_rgb_panel_notes_unavailable_without_bbox(self):
        class RecordingSink(OpenCVPreviewSink):
            def __init__(self):
                super().__init__()
                self.notes = []

            def _fit_with_transform(self, image, size):
                return np.zeros((size[1], size[0], 3), dtype=np.uint8), 1.0, (0, 0)

            def _title(self, panel, title):
                return None

            def _corner_note(self, panel, text, fg=(230, 230, 230)):
                self.notes.append((text, fg))

        os.environ["VISTA_TABLE_BBOX_ENABLE"] = "1"
        os.environ.pop("VISTA_MOCK_TABLE_BBOX", None)
        sink = RecordingSink()
        sink._make_rgb_panel(np.zeros((20, 20, 3), dtype=np.uint8), {}, (40, 40))
        self.assertIn(("table_bbox unavailable", YELLOW), sink.notes)


if __name__ == "__main__":
    unittest.main()
