#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORCH_ROOT = ROOT / "orchestrator"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from orchestrator_service.config.schema import CarMotionConfig, ControlThresholds
from orchestrator_service.control.motion_controller import MotionController
from orchestrator_service.ipc.protocol import TableEdgeObs


def test_fov_approach_uses_yolo_forward_when_bbox_exists_but_edge_not_trusted():
    controller = MotionController(ControlThresholds(), CarMotionConfig())
    obs = TableEdgeObs.from_dict(
        {
            "ts": 1.0,
            "table_found": True,
            "edge_found": True,
            "table_bbox_xyxy": [0.25, 0.35, 0.75, 0.90],
            "table_bbox_found": True,
            "table_bbox_control_valid": True,
            "table_cx_norm": 0.0,
            "edge_trusted": False,
            "valid_for_control": False,
            "usable_for_approach": False,
        }
    )

    decision = controller.fov_table_approach_cmd(obs, mode="CONTROLLED_APPROACH")

    assert decision.cmd.vx_mps > 0.0
    assert decision.cmd.wz_radps == 0.0
    assert decision.control_summary["control_source"] == "yolo_forward"
    assert decision.control_summary["reason"] == "table_bbox_found_edge_not_trusted_yolo_forward"
