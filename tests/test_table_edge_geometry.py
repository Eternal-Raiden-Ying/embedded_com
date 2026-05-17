import unittest

import numpy as np

from VISTA.Online_Edge_Detect.detector import CameraCalib, EdgeDetectResult, OnlineTableEdgeDetector
from VISTA.Online_Edge_Detect.schema import DetectorConfig
from VISTA.vision_module.backend.preview.opencv_sink import OpenCVPreviewSink
from VISTA.vision_module.examples.offline_bag_edge_debug import build_parser


class TableEdgeGeometryTest(unittest.TestCase):
    def test_result_schema_keeps_legacy_and_geometry_fields(self) -> None:
        result = EdgeDetectResult(False, 0.0, 0.0, 0.0)
        self.assertFalse(result.edge_found)
        self.assertIsNone(result.line_k)
        self.assertEqual(result.pose_source, "none")
        self.assertFalse(result.valid_for_control)
        self.assertEqual(result.stable_count, 0)

    def test_detector_returns_geometry_diagnostics(self) -> None:
        height, width = 120, 160
        depth_m = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            upper = 0.0005 * (y - 60)
            lower = 0.0035 * max(0, y - 60)
            trend = upper if y < 60 else lower
            for x in range(width):
                depth_m[y, x] = 0.68 + trend + 0.00025 * (x - width / 2)
        depth = np.clip(depth_m / 0.001, 0, 65535).astype(np.uint16)
        cfg = DetectorConfig(
            roi_x0=0,
            roi_y0=0,
            roi_x1=width,
            roi_y1=height,
            z_min=0.2,
            z_max=1.4,
            min_all_points=100,
            min_table_points=10,
            trend_window_px=8,
            trend_col_step_px=10,
            trend_min_slope_delta=0.001,
            trend_min_candidate_count=5,
            line_min_x_span_m=0.04,
            line_max_residual_m=0.04,
            plane_min_inliers=50,
            control_min_stable_frames=1,
        )
        detector = OnlineTableEdgeDetector(CameraCalib(120.0, 120.0, 80.0, 60.0, 0.001), cfg, 0.55)
        result, debug = detector.process_depth(depth)
        self.assertGreater(result.point_count, 0)
        self.assertTrue(hasattr(result, "raw_found"))
        self.assertTrue(hasattr(result, "pose_found"))
        self.assertTrue(hasattr(result, "valid_for_control"))
        self.assertIn("front_plane_candidate_pixels", debug)
        self.assertIn("crease_candidate_pixels", debug)
        self.assertIn("fused_pose", debug)

    def test_offline_parser_accepts_save_csv(self) -> None:
        args = build_parser().parse_args(["--save-csv", "/tmp/table_edge_geometry.csv", "--max-frames", "1"])
        self.assertEqual(str(args.save_csv), "/tmp/table_edge_geometry.csv")
        self.assertEqual(args.max_frames, 1)

    def test_preview_edge_panel_handles_legacy_payload(self) -> None:
        sink = OpenCVPreviewSink("test")
        depth = np.full((64, 96), 700, dtype=np.uint16)
        panel = sink._make_edge_panel(depth, {"edge_found": False, "edge_roi": [10, 10, 80, 50]}, (240, 160))
        self.assertEqual(panel.shape, (160, 240, 3))


if __name__ == "__main__":
    unittest.main()
