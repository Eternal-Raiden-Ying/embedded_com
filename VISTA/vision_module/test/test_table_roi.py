#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from types import SimpleNamespace

import numpy as np

try:
    from VISTA.vision_module.backend.table_edge_manager import TableEdgeManager
    from VISTA.vision_module.backend.table_edge_roi import (
        choose_depth_roi,
        choose_table_quadrant,
        normalize_table_bbox,
        preset_to_roi,
        quadrant_to_depth_roi,
    )
    from VISTA.vision_module.config.schema import VisionServiceConfig
    from VISTA.vision_module.diagnostics.summaries import (
        format_runtime_summary,
        format_table_edge_summary,
        format_target_summary,
    )
    from VISTA.vision_module.diagnostics.operator_console import OperatorConsole
    from VISTA.vision_module.utils.table_roi import (
        bbox_to_quadrant,
        build_table_roi,
        find_table_bbox,
        quadrant_to_roi,
        table_class_available,
        table_detection_debug,
    )
except ImportError:
    from vision_module.backend.table_edge_manager import TableEdgeManager
    from vision_module.backend.table_edge_roi import (
        choose_depth_roi,
        choose_table_quadrant,
        normalize_table_bbox,
        preset_to_roi,
        quadrant_to_depth_roi,
    )
    from vision_module.config.schema import VisionServiceConfig
    from vision_module.diagnostics.summaries import (
        format_runtime_summary,
        format_table_edge_summary,
        format_target_summary,
    )
    from vision_module.diagnostics.operator_console import OperatorConsole
    from vision_module.utils.table_roi import (
        bbox_to_quadrant,
        build_table_roi,
        find_table_bbox,
        quadrant_to_roi,
        table_class_available,
        table_detection_debug,
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
            find_table_bbox({"class_names": ["table1", "apple"], "infer_boxes": [[100, 120, 420, 360, 0.7, 0]]}),
            [100, 120, 420, 360],
        )
        self.assertEqual(
            find_table_bbox({"infer_boxes": [[100, 120, 420, 360, 0.7, "bad", "desk"]]}),
            [100, 120, 420, 360],
        )
        self.assertEqual(
            find_table_bbox({"class_names": ["person", "dining table"], "infer_boxes": [[100, 120, 420, 360, 0.7, 1]]}),
            [100, 120, 420, 360],
        )

    def test_table_detection_debug_reports_direction_or_missing_class(self):
        det = table_detection_debug(
            {
                "rgb_shape": (480, 640, 3),
                "class_names": ["person", "dining table"],
                "infer_boxes": [[20, 80, 220, 260, 0.82, 1]],
            },
            min_conf=0.30,
            center_tol=0.10,
        )
        self.assertTrue(det["found"])
        self.assertEqual(det["bbox"], [20, 80, 220, 260])
        self.assertEqual(det["direction"], "left")
        self.assertAlmostEqual(det["cx"], 0.1875)

        self.assertFalse(table_class_available({"class_names": ["apple", "bottle"]}))
        missing = table_detection_debug({"class_names": ["apple", "bottle"], "infer_boxes": []})
        self.assertFalse(missing["found"])
        self.assertTrue(missing["no_table_class"])
        self.assertEqual(missing["reason"], "no_table_class")

    def test_bbox_center_to_quadrant_and_roi(self):
        self.assertEqual(bbox_to_quadrant([100, 120, 420, 360], (640, 640, 3)), "LT")
        self.assertEqual(bbox_to_quadrant([360, 120, 620, 360], (640, 640, 3)), "RT")
        self.assertEqual(bbox_to_quadrant([100, 360, 420, 620], (640, 640, 3)), "LB")
        self.assertEqual(bbox_to_quadrant([360, 360, 620, 620], (640, 640, 3)), "RB")
        self.assertEqual(quadrant_to_roi("LT", 640, 480), [0, 0, 320, 240])
        self.assertEqual(quadrant_to_roi("RB", 640, 480), [320, 240, 640, 480])

    def test_center_roi_presets_are_explicit_and_shape_based(self):
        self.assertEqual(preset_to_roi("full_frame", (480, 640)), [0, 0, 640, 480])
        self.assertEqual(preset_to_roi("center_mid", (480, 640)), [160, 168, 480, 312])
        self.assertEqual(preset_to_roi("center_lower", (480, 640)), [160, 240, 480, 408])
        self.assertEqual(preset_to_roi("full_width_lower", (480, 640)), [0, 240, 640, 456])
        self.assertEqual(preset_to_roi("right_lower", (480, 640)), [320, 240, 640, 456])
        self.assertIsNone(preset_to_roi("", (480, 640)))

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

    def test_table_edge_roi_helpers_keep_roi_selection_pure(self):
        self.assertEqual(normalize_table_bbox("100,120,420,360", (640, 640, 3)), [100, 120, 420, 360])
        self.assertEqual(choose_table_quadrant([400, 380, 620, 620], (640, 640, 3)), "RB")
        self.assertEqual(quadrant_to_depth_roi("RB", (480, 640)), [320, 240, 640, 480])
        roi = choose_depth_roi(
            {"rgb_shape": (640, 640, 3), "infer_boxes": [[400, 380, 620, 620, 0.9, 60]]},
            depth_shape=(480, 640),
            fallback_depth_roi=[10, 20, 110, 120],
        )
        self.assertEqual(roi["roi_source"], "local_perception_table_bbox")
        self.assertEqual(roi["roi_reason"], "table_bbox_detected")
        self.assertEqual(roi["table_quadrant"], "RB")
        self.assertEqual(roi["depth_edge_roi"], [320, 240, 640, 480])

        preset = choose_depth_roi(
            {"rgb_shape": (640, 640, 3), "infer_boxes": [[400, 380, 620, 620, 0.9, 60]]},
            depth_shape=(480, 640),
            fallback_depth_roi=[10, 20, 110, 120],
            roi_preset="center_lower",
        )
        self.assertEqual(preset["roi_source"], "preset:center_lower")
        self.assertEqual(preset["roi_reason"], "debug_roi_preset")
        self.assertEqual(preset["roi_preset"], "center_lower")
        self.assertEqual(preset["depth_edge_roi"], [160, 240, 480, 408])

    def test_yolo_dynamic_roi_uses_lower_band_when_bbox_touches_bottom(self):
        roi = choose_depth_roi(
            {
                "rgb_shape": (640, 640, 3),
                "class_names": ["table1"],
                "infer_boxes": [[250, 410, 610, 640, 0.86, 0]],
            },
            depth_shape=(480, 640),
            fallback_depth_roi=[160, 240, 480, 408],
            yolo_dynamic_enable=True,
            yolo_table_class_id=0,
            near_distance=True,
        )
        self.assertTrue(roi["bbox_valid"])
        self.assertTrue(roi["yolo_table_roi_valid"])
        self.assertTrue(roi["table_bbox_touch_bottom"])
        self.assertTrue(roi["table_bbox_boundary_allowed"])
        self.assertEqual(roi["roi_source"], "yolo_table_lower_band")
        self.assertEqual(roi["roi_phase"], "near_yolo_assist")
        self.assertNotEqual(roi["depth_edge_roi"], [160, 240, 480, 408])
        self.assertGreaterEqual(roi["depth_edge_roi"][1], 300)

    def test_table_bbox_touch_boundary_remains_reliable_for_yolo_control(self):
        metrics = TableEdgeManager._bbox_view_metrics([0, 380, 640, 640], (640, 640, 3))
        self.assertTrue(metrics["touch_left"])
        self.assertTrue(metrics["touch_right"])
        self.assertTrue(metrics["touch_bottom"])
        self.assertTrue(metrics["boundary_allowed"])
        self.assertTrue(metrics["reliable"])

    def test_diagnostics_summary_formatters_are_short_lines(self):
        edge = format_table_edge_summary(
            {"stage": "SEARCH", "mode": "FIND_EDGE"},
            {"edge_found": True, "confidence": 0.86, "yaw_err_rad": -0.02, "dist_err_m": -0.01, "roi_source": "local_perception_table_bbox", "table_quadrant": "LT"},
        )
        target = format_target_summary(
            {"stage": "SEARCH", "mode": "FIND_OBJECT"},
            {"found": True, "target": "bottle", "confidence": 0.82, "cx_norm": 0.54, "cy_norm": 0.47},
        )
        runtime = format_runtime_summary({"stage": "SEARCH", "mode": "FIND_OBJECT", "req_id": "r1", "epoch": 2})
        self.assertIn("[VISTA] EDGE stage=SEARCH mode=TABLE_EDGE_PERCEPTION", edge)
        self.assertIn("roi=local_perception_table_bbox q=LT", edge)
        self.assertIn("[VISTA] TARGET stage=SEARCH mode=TRACK_LOCAL found=1 cls=bottle", target)
        self.assertIn("[VISTA] RUNTIME stage=SEARCH mode=TRACK_LOCAL req=r1 epoch=2", runtime)

    def test_operator_console_supports_dedupe_rate_limit_and_error(self):
        lines = []
        console = OperatorConsole(mode="operator", default_interval_s=10.0, sink=lines.append)
        self.assertTrue(console.emit_change("mode", "[VISTA] MODE IDLE -> SEARCH"))
        self.assertFalse(console.emit_change("mode", "[VISTA] MODE IDLE -> SEARCH"))
        self.assertTrue(console.emit_rate_limited("edge", "[VISTA] EDGE edge=1"))
        self.assertFalse(console.emit_rate_limited("edge", "[VISTA] EDGE edge=1"))
        self.assertTrue(console.emit_error("ipc", "[VISTA] IPC connect_failed"))
        self.assertFalse(console.emit_error("ipc", "[VISTA] IPC connect_failed"))
        self.assertEqual(len(lines), 3)


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
        self.assertEqual(roi["depth_edge_roi"], [460, 20, 560, 120])
        self.assertEqual(roi["roi_source"], "yolo_table_bbox")
        self.assertEqual(roi["roi_reason"], "table_bbox_center_follow")
        self.assertTrue(roi["bbox_valid"])
        self.assertTrue(roi["yolo_table_roi_valid"])

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
        self.assertEqual(roi["roi_source"], "last_valid_table_bbox")
        self.assertEqual(roi["roi_reason"], "table_bbox_lost_using_history")

    def test_no_bbox_and_no_history_falls_back_to_static_roi(self):
        self.scheduler.local = {"rgb_shape": (640, 640, 3), "infer_boxes": []}
        roi = self.manager._select_roi(self.depth)
        self.assertIsNone(roi["table_bbox"])
        self.assertIsNone(roi["table_quadrant"])
        self.assertEqual(roi["depth_edge_roi"], [10, 20, 110, 120])
        self.assertEqual(roi["roi_source"], "static_fallback")
        self.assertEqual(roi["roi_reason"], "table_bbox_unavailable")

    def test_file_config_roi_preset_overrides_dynamic_roi_selection(self):
        cfg = VisionServiceConfig()
        cfg.table_edge.roi_preset = "center_lower"
        manager = TableEdgeManager(cfg=cfg)
        manager._detector_cfg = SimpleNamespace(roi_x0=10, roi_y0=20, roi_x1=110, roi_y1=120)
        manager.bind_runtime(self.scheduler, lambda: 1)
        self.scheduler.local = {"rgb_shape": (640, 640, 3), "infer_boxes": []}
        roi = manager._select_roi(self.depth)
        self.assertEqual(roi["roi_source"], "preset:center_lower")
        self.assertEqual(roi["roi_preset"], "center_lower")
        self.assertEqual(roi["depth_edge_roi"], [160, 240, 480, 408])

    def test_process_camera_frame_emits_unified_timing_and_plane_aliases(self):
        cfg = VisionServiceConfig()
        cfg.table_edge.roi_preset = "center_lower"
        manager = TableEdgeManager(cfg=cfg)
        manager._detector_cfg = SimpleNamespace(
            roi_x0=10,
            roi_y0=20,
            roi_x1=110,
            roi_y1=120,
            plane_only_mode=True,
            enable_crease_line=False,
        )
        obs = manager.process_camera_frame(
            {"depth": np.zeros((480, 640, 1), dtype=np.uint16), "frame_capture_ts": 1000.0},
            frame_seq=7,
            frame_slot={"seq": 7, "ts": 1000.0},
            source_mode="OFFLINE_BAG",
            count_dropped=False,
        )
        self.assertEqual(obs["type"], "table_edge_obs")
        self.assertEqual(obs["source_mode"], "OFFLINE_BAG")
        self.assertEqual(obs["frame_seq"], 7)
        self.assertIn("process_ms", obs)
        self.assertIn("obs_total_age_ms", obs)
        self.assertIn("update_interval_ms", obs)
        self.assertEqual(obs["plane_roi"], obs["depth_edge_roi"])
        self.assertEqual(obs["roi_source"], "preset:center_lower")


if __name__ == "__main__":
    unittest.main()
