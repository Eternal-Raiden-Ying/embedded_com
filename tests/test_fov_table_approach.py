#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORCH_ROOT = ROOT / "orchestrator"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from orchestrator_service.config.schema import CarMotionConfig, ControlThresholds  # noqa: E402
from orchestrator_service.control.motion_adapter import Stm32MotionAdapter  # noqa: E402
from orchestrator_service.ipc.protocol import CmdVel, TableEdgeObs, now_ts  # noqa: E402
from orchestrator_service.runtime.controller import MotionController  # noqa: E402


def _controller() -> MotionController:
    cfg = ControlThresholds()
    car = CarMotionConfig()
    car.table_vx_slew_per_s = 10.0
    car.table_vy_slew_per_s = 10.0
    car.table_wz_slew_per_s = 10.0
    return MotionController(cfg, car)


def _obs(**kwargs) -> TableEdgeObs:
    payload = {
        "ts": now_ts(),
        "table_found": True,
        "edge_found": True,
        "edge_valid": True,
        "confidence": 0.9,
        "yaw_err_rad": 0.02,
        "dist_err_m": 0.12,
        "target_dist_m": 0.50,
        "control_level": "alignment",
        "usable_for_approach": True,
        "usable_for_alignment": True,
        "view_reliable": True,
        "view_source": "plane",
        "view_err_norm": 0.10,
        "plane_cx_norm": 0.10,
    }
    payload.update(kwargs)
    return TableEdgeObs.from_dict(payload)


class FovTableApproachTest(unittest.TestCase):
    def test_table_edge_obs_missing_new_fields_does_not_crash(self) -> None:
        obs = TableEdgeObs.from_dict({
            "ts": now_ts(),
            "table_found": True,
            "edge_found": True,
            "confidence": 0.8,
        })
        self.assertFalse(obs.view_reliable)
        self.assertFalse(obs.fov_guard_active)
        self.assertIsNone(obs.plane_cx_norm)

    def test_stage_a_yolo_only_turns_without_forward(self) -> None:
        controller = _controller()
        obs = _obs(
            table_found=False,
            edge_found=False,
            edge_valid=False,
            yaw_err_rad=None,
            dist_err_m=None,
            control_level="none",
            table_confirmed_by_yolo=True,
            yolo_reliable=True,
            table_cx_norm=0.50,
            view_reliable=True,
            view_source="yolo",
            view_err_norm=0.50,
        )
        decision = controller.fov_table_approach_cmd(obs, phase="STAGE_A")
        self.assertEqual(decision.control_summary["table_approach_phase"], "STAGE_A")
        self.assertEqual(decision.cmd.vx_norm, 0.0)
        self.assertNotEqual(decision.cmd.wz_norm, 0.0)

    def test_stage_c_plane_only_servo_when_yolo_unreliable(self) -> None:
        controller = _controller()
        obs = _obs(table_confirmed_by_yolo=False, yolo_reliable=False, view_source="plane")
        decision = controller.fov_table_approach_cmd(obs)
        self.assertEqual(decision.control_summary["table_approach_phase"], "STAGE_C")
        self.assertGreaterEqual(decision.cmd.vx_norm, 0.0)
        self.assertNotEqual(decision.cmd.wz_norm, 0.0)

    def test_hard_view_error_forces_vx_zero(self) -> None:
        controller = _controller()
        obs = _obs(view_err_norm=0.60, plane_cx_norm=0.60)
        decision = controller.fov_table_approach_cmd(obs)
        self.assertEqual(decision.cmd.vx_norm, 0.0)
        self.assertTrue(decision.control_summary["fov_guard_active"])

    def test_plane_touch_boundary_forces_vx_zero(self) -> None:
        controller = _controller()
        obs = _obs(plane_touch_left=True, view_err_norm=0.10)
        decision = controller.fov_table_approach_cmd(obs)
        self.assertEqual(decision.cmd.vx_norm, 0.0)
        self.assertTrue(decision.control_summary["fov_guard_active"])

    def test_approach_level_outputs_view_correction(self) -> None:
        controller = _controller()
        obs = _obs(control_level="approach", usable_for_alignment=False, view_err_norm=0.30, plane_cx_norm=0.30)
        decision = controller.plane_approach_cmd(obs)
        self.assertGreaterEqual(decision.cmd.vx_norm, 0.0)
        self.assertTrue(abs(decision.cmd.vy_norm) > 0.0 or abs(decision.cmd.wz_norm) > 0.0)

    def test_alignment_docking_output_is_fov_guarded(self) -> None:
        controller = _controller()
        obs = _obs(view_err_norm=0.55, plane_cx_norm=0.55)
        decision = controller.controlled_approach_cmd(obs)
        self.assertEqual(decision.cmd.vx_norm, 0.0)
        self.assertTrue(decision.control_summary["fov_guard_active"])

    def test_stop_level_at_distance_outputs_stop(self) -> None:
        controller = _controller()
        obs = _obs(control_level="stop", usable_for_stop=True, dist_err_m=0.0, view_err_norm=0.0, plane_cx_norm=0.0)
        decision = controller.final_lock_cmd(obs)
        self.assertEqual(decision.cmd.vx_norm, 0.0)

    def test_motion_adapter_wheel_mapping_unchanged(self) -> None:
        adapter = Stm32MotionAdapter(uart=object(), logger=lambda _line: None, vx_scale=100, vy_scale=100, wz_scale=100)
        cmd = CmdVel(ts=time.time(), mode="TEST", vx_norm=0.10, vy_norm=0.02, wz_norm=0.03)
        self.assertEqual(adapter.cmd_vel_to_wheels(cmd), (5, -9, 9, -5))


if __name__ == "__main__":
    unittest.main()
