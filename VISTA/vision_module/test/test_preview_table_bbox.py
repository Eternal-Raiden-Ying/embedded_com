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


_FAKE_CV2 = sys.modules.setdefault("cv2", _FakeCV2())

try:
    from VISTA.vision_module.backend.preview.opencv_sink import CYAN, YELLOW, OpenCVPreviewSink
except ImportError:
    from vision_module.backend.preview.opencv_sink import CYAN, YELLOW, OpenCVPreviewSink


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

    def test_track_local_rgb_panel_draws_yolo_boxes_not_table_overlay(self):
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

        sink = RecordingSink()
        sink._make_rgb_panel(
            np.zeros((100, 100, 3), dtype=np.uint8),
            {
                "runtime_status": {"mode": "TRACK_LOCAL"},
                "target": "apple",
                "local_perception": {
                    "table_bbox": [10, 20, 80, 90],
                    "class_names": ["apple", "bottle"],
                    "infer_boxes": [[1, 2, 30, 40, 0.81, 0], [4, 5, 50, 60, 0.62, 1]],
                },
            },
            (120, 120),
        )
        labels = [item[0] for item in sink.draws]
        self.assertIn("apple:0.81", labels)
        self.assertIn("bottle:0.62", labels)
        self.assertNotIn("table_bbox", labels)
        self.assertIn(("target=apple boxes=2", (0, 255, 80)), sink.notes)

    def test_status_panel_uses_metadata_without_crashing(self):
        calls = []
        cv2_obj = OpenCVPreviewSink._draw_status_sections.__globals__["cv2"]
        old_put_text = getattr(cv2_obj, "putText", None)
        cv2_obj.putText = lambda _panel, text, *_args, **_kwargs: calls.append(str(text))
        try:
            sink = OpenCVPreviewSink(window_name="VISTA App Dashboard")
            panel = np.zeros((420, 360, 3), dtype=np.uint8)
            sink._draw_status_sections(
                panel,
                {"preview_layout": "rgb_yolo_edge_overlay", "window_id": "VISTA App Dashboard#1"},
                {"stage": "SEARCH", "mode": "TRACK_LOCAL", "req_id": "req-test"},
                {"edge_found": True, "table_found": True, "confidence": 0.8},
                {"target": "apple", "found": False, "boxes_count": 0},
                {"has_infer": True, "box_count": 0},
                ["rgb"],
                "apple",
                0.02,
            )
        finally:
            if old_put_text is None:
                delattr(cv2_obj, "putText")
            else:
                cv2_obj.putText = old_put_text
        self.assertIn("preview_layout=rgb_yolo_edge_overlay", calls)
        self.assertIn("window_id=VISTA App Dashboard#1", calls)
        self.assertIn("mode=TRACK_LOCAL", calls)

    def test_close_cleans_preview_windows(self):
        calls = []
        cv2_obj = OpenCVPreviewSink.close.__globals__["cv2"]
        old_destroy_window = getattr(cv2_obj, "destroyWindow", None)
        old_destroy_all = getattr(cv2_obj, "destroyAllWindows", None)
        old_wait_key = getattr(cv2_obj, "waitKey", None)
        cv2_obj.destroyWindow = lambda name: calls.append(("destroyWindow", name))
        cv2_obj.destroyAllWindows = lambda: calls.append(("destroyAllWindows", None))
        cv2_obj.waitKey = lambda delay=0: calls.append(("waitKey", delay)) or 0
        try:
            sink = OpenCVPreviewSink(window_name="VISTA App Dashboard")
            sink._opened = True
            sink.close()
        finally:
            if old_destroy_window is None:
                delattr(cv2_obj, "destroyWindow")
            else:
                cv2_obj.destroyWindow = old_destroy_window
            if old_destroy_all is None:
                delattr(cv2_obj, "destroyAllWindows")
            else:
                cv2_obj.destroyAllWindows = old_destroy_all
            if old_wait_key is None:
                delattr(cv2_obj, "waitKey")
            else:
                cv2_obj.waitKey = old_wait_key
        self.assertIn(("destroyWindow", "VISTA App Dashboard"), calls)
        self.assertIn(("destroyAllWindows", None), calls)
        self.assertFalse(sink._opened)


if __name__ == "__main__":
    unittest.main()
