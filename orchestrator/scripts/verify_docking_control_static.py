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
from orchestrator.orchestrator_service.runtime.states.table_docking import TableDockingMixin
from orchestrator.orchestrator_service.runtime.common import monotonic_ts
from orchestrator.orchestrator_service.ipc.protocol import TableEdgeObs
from vision_module.backend.table_roi_depth import table_roi_depth_statistics
from vision_module.app.stages.search.table_edge_obs_builder import merge_table_bbox_from_local_perception


def auth(state: str, *, bbox: bool, edge: bool = False, error: float = 0.0, depth_stop: bool = False, phase: str = ""):
    return decide_table_control_authority(
        state,
        TablePerceptionSemantics(table_bbox_current_found=bbox, table_bbox_control_valid=bbox, edge_trusted=edge),
        SimpleNamespace(yolo_forward_center_hard_limit=0.25),
        depth_roi_stop_active=depth_stop,
        bbox_center_error=error,
        control_phase=phase,
    )


def main() -> None:
    vista_obs = TableEdgeObs.from_dict({
        "ts": 1.0, "table_found": True, "edge_found": False,
        "yolo_table_visible": True, "yolo_table_fresh": True, "yolo_table_control_valid": True,
        "table_cx_norm": 0.70, "table_bbox_touch_right": True,
    })
    assert vista_obs.table_bbox_current_found and vista_obs.table_bbox_control_valid
    assert vista_obs.table_cx_norm == 0.70 and vista_obs.table_bbox_touch_right
    # Edge/depth facts without a current bbox must remain a rotate-only search.
    a = auth("YOLO_APPROACH", bbox=False, edge=True, depth_stop=True)
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
    assert auth("YOLO_APPROACH", bbox=True, edge=True, phase="BBOX_ACQUIRE").yaw_source == "bbox"
    assert auth("YOLO_APPROACH", bbox=True, edge=True, phase="EDGE_HANDOFF_CONFIRM").allow_forward is False
    assert auth("YOLO_APPROACH", bbox=True, edge=True, phase="EDGE_GUIDED_APPROACH").yaw_source == "edge"

    depth = np.full((100, 100), 1000, dtype=np.uint16)
    stats = table_roi_depth_statistics(depth, 0.001, [10, 10, 90, 90])
    assert stats["table_roi_depth_valid"] and stats["table_roi_depth_p10"] == 1.0
    assert stats["table_roi_depth_bbox"] and stats["table_roi_depth_coord_space"] == "depth_frame_xyxy"
    depth[:] = 0
    invalid = table_roi_depth_statistics(depth, 0.001, [10, 10, 90, 90])
    assert not invalid["table_roi_depth_valid"]
    no_bbox = table_roi_depth_statistics(depth + 1000, 0.001, [10, 10, 90, 90], current_table_bbox_found=False)
    assert not no_bbox["table_roi_depth_valid"] and no_bbox["table_roi_depth_p10"] is None
    merged = merge_table_bbox_from_local_perception(
        {"table_roi_depth_valid": True, "table_roi_depth_p10": 0.2, "table_roi_depth_sample_count": 100},
        {"table_bbox_current_found": False, "table_found": False}, tick_ts=1.0,
    )
    assert not merged["table_roi_depth_valid"] and merged["table_roi_depth_p10"] is None

    class ProgressProbe(TableDockingMixin):
        def __init__(self):
            self.cfg = SimpleNamespace(table_target_dist_m=0.5)  # deliberately lacks progress_window_ms
            self.ctx = SimpleNamespace(min_dist_seen=999.0, dist_progress_last_refreshed_mono=0.0, dist_missing_started_mono=0.0)
        def _log(self, *_args):
            pass

    probe = ProgressProbe()
    assert not probe._check_approach_progress(None)
    assert probe.ctx.dist_missing_started_mono > 0.0
    probe.ctx.dist_missing_started_mono = monotonic_ts() - 5.1
    assert probe._check_approach_progress(None)
    print("docking static verification: PASS")


if __name__ == "__main__":
    main()
