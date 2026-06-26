#!/usr/bin/env python3
"""Pure-Python synthetic checks for table docking authority and ROI depth."""
from __future__ import annotations

from types import SimpleNamespace
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "VISTA"))

import numpy as np

from orchestrator.orchestrator_service.runtime.control_authority import decide_table_control_authority
from orchestrator.orchestrator_service.runtime.perception_semantics import TablePerceptionSemantics
from vision_module.backend.table_roi_depth import table_roi_depth_statistics


def auth(state: str, *, bbox: bool, edge: bool = False, error: float = 0.0, depth_stop: bool = False):
    return decide_table_control_authority(
        state,
        TablePerceptionSemantics(table_bbox_current_found=bbox, table_bbox_control_valid=bbox, edge_trusted=edge),
        SimpleNamespace(yolo_forward_center_hard_limit=0.25),
        depth_roi_stop_active=depth_stop,
        bbox_center_error=error,
    )


def main() -> None:
    a = auth("SEARCH_TABLE", bbox=False)
    assert (a.control_source, a.allow_forward, a.allow_rotate) == ("local_rotate_search", False, True)
    a = auth("YOLO_APPROACH", bbox=True, error=0.3)
    assert (a.control_source, a.allow_forward, a.allow_rotate) == ("yolo_acquire_align", False, True)
    a = auth("YOLO_APPROACH", bbox=True, edge=False)
    assert a.control_source == "yolo_track_forward" and a.allow_forward and a.yaw_source == "yolo"
    a = auth("YOLO_APPROACH", bbox=True, edge=True)
    assert a.control_source == "edge_guided_forward" and a.yaw_source == "edge"
    a = auth("YOLO_APPROACH", bbox=True, depth_stop=True)
    assert a.control_source == "depth_roi_stop" and not a.allow_forward
    a = auth("YOLO_APPROACH", bbox=True, edge=True)
    assert a.stop_source == "none", "edge trust alone must not become a depth stop"

    depth = np.full((100, 100), 1000, dtype=np.uint16)
    stats = table_roi_depth_statistics(depth, 0.001, [10, 10, 90, 90])
    assert stats["table_roi_depth_valid"] and stats["table_roi_depth_p10"] == 1.0
    assert stats["table_roi_depth_bbox"] and stats["table_roi_depth_coord_space"] == "depth_frame_xyxy"
    depth[:] = 0
    invalid = table_roi_depth_statistics(depth, 0.001, [10, 10, 90, 90])
    assert not invalid["table_roi_depth_valid"]
    print("docking static verification: PASS")


if __name__ == "__main__":
    main()
