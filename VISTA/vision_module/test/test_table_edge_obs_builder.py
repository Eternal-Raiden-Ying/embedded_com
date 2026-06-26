import sys
from pathlib import Path
TEST_DIR = Path(__file__).resolve().parent
VISION_ROOT = TEST_DIR.parent
VISTA_ROOT = VISION_ROOT.parent
STACK_ROOT = VISTA_ROOT.parent
if str(STACK_ROOT) not in sys.path:
    sys.path.insert(0, str(STACK_ROOT))
if str(VISTA_ROOT) not in sys.path:
    sys.path.insert(0, str(VISTA_ROOT))

import unittest
from vision_module.app.stages.search.table_edge_obs_builder import (
    annotate_table_edge_obs,
    merge_table_bbox_from_local_perception,
    default_table_edge_obs,
)

class TestTableEdgeObsBuilder(unittest.TestCase):

    def test_annotate_priority_trusted(self):
        # Case 1: edge_trusted is True
        obs = {
            "edge_found": True,
            "edge_valid": True,
            "edge_trusted": True,
            "point_count": 50,
            "table_point_count": 30,
            "frame_id": 42,
        }
        annotated = annotate_table_edge_obs(obs, tick_ts=100.0, source="results", source_mode="FIND_EDGE")
        self.assertEqual(annotated["reason"], "edge_trusted")
        self.assertTrue(annotated["edge_found"])
        self.assertTrue(annotated["edge_valid"])
        self.assertTrue(annotated["edge_trusted"])
        self.assertTrue(annotated["valid_for_control"])
        self.assertTrue(annotated["edge_control_allowed"])
        self.assertEqual(annotated["reject_reason"], "")
        self.assertEqual(annotated["point_count"], 50)
        self.assertEqual(annotated["table_point_count"], 30)

    def test_annotate_priority_valid(self):
        # Case 2: edge_valid is True, edge_trusted is False
        obs = {
            "edge_found": True,
            "edge_valid": True,
            "edge_trusted": False,
            "point_count": 50,
            "table_point_count": 30,
            "frame_id": 42,
        }
        annotated = annotate_table_edge_obs(obs, tick_ts=100.0, source="results", source_mode="FIND_EDGE")
        self.assertEqual(annotated["reason"], "edge_valid")
        self.assertTrue(annotated["edge_found"])
        self.assertTrue(annotated["edge_valid"])
        self.assertFalse(annotated["edge_trusted"])
        self.assertEqual(annotated["point_count"], 50)

    def test_annotate_priority_candidate_rejected(self):
        # Case 3: edge_candidate is True, edge_valid is False
        obs = {
            "candidate": True,
            "edge_found": False,
            "edge_valid": False,
            "point_count": 50,
            "table_point_count": 30,
            "frame_id": 42,
        }
        annotated = annotate_table_edge_obs(obs, tick_ts=100.0, source="results", source_mode="FIND_EDGE")
        self.assertEqual(annotated["reason"], "edge_candidate_rejected")
        self.assertTrue(annotated["edge_found"])
        self.assertFalse(annotated["edge_valid"])
        self.assertFalse(annotated["edge_trusted"])
        self.assertTrue(annotated["edge_candidate_found"])
        self.assertEqual(annotated["point_count"], 50)
        self.assertEqual(annotated["table_point_count"], 30)

    def test_annotate_priority_no_edge(self):
        # Case 4: No candidate, not valid
        obs = {
            "candidate": False,
            "edge_found": False,
            "edge_valid": False,
            "point_count": 50,
            "table_point_count": 30,
            "frame_id": 42,
        }
        annotated = annotate_table_edge_obs(obs, tick_ts=100.0, source="results", source_mode="FIND_EDGE")
        self.assertEqual(annotated["reason"], "table_bbox_from_local_perception_no_edge_result")
        self.assertFalse(annotated["edge_found"])
        self.assertFalse(annotated["edge_valid"])
        self.assertFalse(annotated["edge_trusted"])
        self.assertEqual(annotated["point_count"], 0)
        self.assertEqual(annotated["table_point_count"], 0)

    def test_annotate_aliases_mapping(self):
        obs = {
            "detector_candidate_line_present": True,
            "yaw_err_rad": 0.1,
            "dist_err_m": 0.2,
            "support_count": 15,
            "inlier_count": 8,
            "frame_seq": 100,
        }
        annotated = annotate_table_edge_obs(obs, tick_ts=100.0, source="results", source_mode="FIND_EDGE")
        self.assertEqual(annotated["frame_id"], 100)
        self.assertEqual(annotated["seq"], 100)
        self.assertEqual(annotated["yaw_err"], 0.1)
        self.assertEqual(annotated["yaw"], 0.1)
        self.assertEqual(annotated["dist_err"], 0.2)
        self.assertEqual(annotated["dist"], 0.2)
        self.assertEqual(annotated["lateral"], 0.2)
        self.assertEqual(annotated["support_count"], 15)
        self.assertEqual(annotated["fast_support_point_count"], 15)
        self.assertEqual(annotated["inlier_count"], 8)
        self.assertEqual(annotated["edge_inlier_count"], 8)
        self.assertEqual(annotated["valid_edge_points"], 8)
        self.assertTrue(annotated["edge_candidate_found"])

    def test_merge_current_frame(self):
        # Case: is_current_frame is True (frame IDs match)
        # Verify edge results are not overwritten/reset
        obs = {
            "edge_found": True,
            "edge_valid": True,
            "edge_trusted": True,
            "point_count": 99,
            "table_point_count": 70,
            "frame_id": 31,
            "reason": "edge_trusted",
        }
        local_perception = {
            "frame_id": 31,
            "table_bbox": [100, 200, 300, 400],
            "rgb_shape": (480, 640, 3),
        }
        merged = merge_table_bbox_from_local_perception(obs, local_perception, tick_ts=100.0)
        self.assertTrue(merged["table_found"])
        self.assertTrue(merged["edge_found"])
        self.assertTrue(merged["edge_valid"])
        self.assertTrue(merged["edge_trusted"])
        self.assertEqual(merged["point_count"], 99)
        self.assertEqual(merged["table_point_count"], 70)
        self.assertEqual(merged["reason"], "edge_trusted")
        self.assertEqual(merged["table_bbox"], [100, 200, 300, 400])

    def test_merge_stale_frame(self):
        # Case: is_current_frame is False (frame IDs mismatch by > 1, no timestamp)
        # Verify edge result is reset to no edge
        obs = {
            "edge_found": True,
            "edge_valid": True,
            "edge_trusted": True,
            "point_count": 99,
            "table_point_count": 70,
            "frame_id": 30, # Old frame!
            "reason": "edge_trusted",
        }
        local_perception = {
            "frame_id": 32, # New frame (32 - 30 = 2, so stale)
            "table_bbox": [100, 200, 300, 400],
            "rgb_shape": (480, 640, 3),
        }
        merged = merge_table_bbox_from_local_perception(obs, local_perception, tick_ts=100.0)
        self.assertTrue(merged["table_found"])
        self.assertFalse(merged["edge_found"])
        self.assertFalse(merged["edge_valid"])
        self.assertFalse(merged["edge_trusted"])
        self.assertEqual(merged["point_count"], 0)
        self.assertEqual(merged["table_point_count"], 0)
        self.assertEqual(merged["reason"], "table_bbox_from_local_perception_no_edge_result")
        self.assertEqual(merged["table_bbox"], [100, 200, 300, 400])
        self.assertFalse(merged["depth_valid"])

    def test_merge_non_stale_frame_id_diff_1(self):
        # Case: Frame IDs mismatch by 1, no timestamp
        # Verify edge is NOT reset
        obs = {
            "edge_found": True,
            "edge_valid": True,
            "edge_trusted": True,
            "point_count": 99,
            "table_point_count": 70,
            "frame_id": 30,
            "reason": "edge_trusted",
        }
        local_perception = {
            "frame_id": 31, # Diff of 1 (not stale)
            "table_bbox": [100, 200, 300, 400],
            "rgb_shape": (480, 640, 3),
        }
        merged = merge_table_bbox_from_local_perception(obs, local_perception, tick_ts=100.0)
        self.assertTrue(merged["table_found"])
        self.assertTrue(merged["edge_found"])
        self.assertTrue(merged["edge_valid"])
        self.assertTrue(merged["edge_trusted"])
        self.assertEqual(merged["point_count"], 99)
        self.assertEqual(merged["table_point_count"], 70) # aligned
        self.assertEqual(merged["reason"], "edge_trusted")

    def test_merge_timestamp_gating(self):
        from unittest.mock import patch
        class MockTableEdge:
            edge_sync_threshold_s = 0.15
        class MockVision:
            table_edge = MockTableEdge()
        class MockConfig:
            vision = MockVision()

        with patch("common.config_loader.get_config", return_value=MockConfig()):
            # Case: Frame IDs mismatch by 2, but timestamps match within <= 0.15s
            obs = {
                "edge_found": True,
                "edge_valid": True,
                "edge_trusted": True,
                "point_count": 99,
                "table_point_count": 70,
                "frame_id": 30,
                "obs_ts": 100.0,
                "reason": "edge_trusted",
            }
            local_perception = {
                "frame_id": 32, # Frame ID diff is 2, but...
                "obs_ts": 100.12, # Timestamp diff is 0.12s (<= 0.15s, so not stale)
                "table_bbox": [100, 200, 300, 400],
                "rgb_shape": (480, 640, 3),
            }
            merged = merge_table_bbox_from_local_perception(obs, local_perception, tick_ts=100.2)
            self.assertTrue(merged["table_found"])
            self.assertTrue(merged["edge_found"])
            self.assertTrue(merged["edge_trusted"])
        
            # Case: Timestamps mismatch > 0.15s
            local_perception["obs_ts"] = 100.16 # 0.16s diff (stale)
            merged_stale = merge_table_bbox_from_local_perception(obs, local_perception, tick_ts=100.2)
            self.assertFalse(merged_stale["edge_found"])


if __name__ == "__main__":
    unittest.main()
