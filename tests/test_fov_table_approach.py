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
from orchestrator_service.runtime.context import State  # noqa: E402
from orchestrator_service.runtime.state_machine import OrchestratorCore  # noqa: E402


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
        "pose_found": True,
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

    def test_plane_acquire_holds_when_plane_missing(self) -> None:
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
            view_err_norm=0.10,
        )
        decision = controller.fov_table_approach_cmd(obs, phase="PLANE_ACQUIRE")
        self.assertEqual(decision.control_summary["table_approach_phase"], "PLANE_ACQUIRE")
        self.assertEqual(decision.cmd.vx_norm, 0.0)
        self.assertEqual(decision.cmd.wz_norm, 0.0)

    def test_plane_final_lock_servo_when_yolo_unreliable(self) -> None:
        controller = _controller()
        obs = _obs(table_confirmed_by_yolo=False, yolo_reliable=False, view_source="plane")
        decision = controller.fov_table_approach_cmd(obs)
        self.assertEqual(decision.control_summary["table_approach_phase"], "PLANE_FINAL_LOCK")
        self.assertEqual(decision.control_summary["table_approach_reason"], "plane_confirmed_table_front")
        self.assertEqual(decision.control_summary["approach_source"], "plane_only")
        self.assertGreaterEqual(decision.cmd.vx_norm, 0.0)
        self.assertEqual(decision.control_summary["speed_profile"], "controlled_approach")
        self.assertLessEqual(abs(decision.control_summary["vx_mps"]), 0.030)
        self.assertLessEqual(abs(decision.control_summary["wz_radps"]), 0.050)

    def test_hard_view_error_forces_vx_zero(self) -> None:
        controller = _controller()
        obs = _obs(view_err_norm=0.60, plane_cx_norm=0.60)
        decision = controller.fov_table_approach_cmd(obs)
        self.assertEqual(decision.cmd.vx_norm, 0.0)
        self.assertTrue(decision.control_summary["fov_guard_active"])
        self.assertEqual(decision.control_summary["fov_guard_reason"], "view_err_hard")

    def test_plane_touch_left_forces_vx_zero_with_reason(self) -> None:
        controller = _controller()
        obs = _obs(plane_touch_left=True, view_err_norm=0.10)
        decision = controller.fov_table_approach_cmd(obs)
        self.assertEqual(decision.cmd.vx_norm, 0.0)
        self.assertTrue(decision.control_summary["fov_guard_active"])
        self.assertEqual(decision.control_summary["fov_guard_reason"], "plane_touch_left")
        self.assertTrue(abs(decision.cmd.vy_norm) > 0.0 or abs(decision.cmd.wz_norm) > 0.0)

    def test_plane_touch_right_forces_vx_zero_with_reason(self) -> None:
        controller = _controller()
        obs = _obs(plane_touch_right=True, view_err_norm=0.10)
        decision = controller.fov_table_approach_cmd(obs)
        self.assertEqual(decision.cmd.vx_norm, 0.0)
        self.assertTrue(decision.control_summary["fov_guard_active"])
        self.assertEqual(decision.control_summary["fov_guard_reason"], "plane_touch_right")
        self.assertTrue(abs(decision.cmd.vy_norm) > 0.0 or abs(decision.cmd.wz_norm) > 0.0)

    def test_unreliable_or_stale_view_does_not_advance(self) -> None:
        controller = _controller()
        unreliable = _obs(view_reliable=False)
        unreliable_decision = controller.fov_table_approach_cmd(unreliable)
        self.assertEqual(unreliable_decision.cmd.vx_norm, 0.0)

        stale = _obs(is_stale=True, plane_touch_right=True)
        stale_decision = controller.fov_table_approach_cmd(stale)
        self.assertEqual(stale_decision.cmd.vx_norm, 0.0)
        self.assertEqual(stale_decision.cmd.vy_norm, 0.0)
        self.assertEqual(stale_decision.cmd.wz_norm, 0.0)

    def test_fresh_obs_under_soft_threshold_allows_normal_control(self) -> None:
        controller = _controller()
        obs = _obs(frame_capture_ts=now_ts() - 0.05)
        decision = controller.fov_table_approach_cmd(obs)
        self.assertEqual(decision.control_summary["stale_level"], "fresh")
        self.assertGreater(decision.cmd.vx_norm, 0.0)
        self.assertTrue(decision.control_summary["forward_allowed"])

    def test_pose_missing_uses_safe_vx_then_times_out(self) -> None:
        controller = _controller()
        obs = _obs(pose_found=False, dist_err_m=0.30, control_level="approach")
        decision = controller.fov_table_approach_cmd(obs)
        self.assertAlmostEqual(decision.control_summary["vx_mps"], 0.010, places=3)
        self.assertFalse(decision.control_summary["forward_allowed"])
        self.assertEqual(decision.control_summary["forward_block_reason"], "pose_missing_safe_vx")
        self.assertTrue(decision.control_summary["pose_missing_safe_vx_active"])

        controller._pose_missing_since_mono = time.monotonic() - 3.5
        timed_out = controller.fov_table_approach_cmd(obs)
        self.assertEqual(timed_out.cmd.vx_norm, 0.0)
        self.assertEqual(timed_out.control_summary["forward_block_reason"], "pose_missing_timeout")

    def test_forward_release_uses_low_speed_clamp(self) -> None:
        controller = _controller()
        obs = _obs(dist_err_m=0.30, control_level="approach", yaw_err_rad=0.0, view_err_norm=0.0, plane_cx_norm=0.0)
        decision = controller.fov_table_approach_cmd(obs)
        self.assertTrue(decision.control_summary["forward_allowed"])
        self.assertGreaterEqual(decision.control_summary["vx_mps"], 0.012)
        self.assertLessEqual(decision.control_summary["vx_mps"], 0.030)
        self.assertLessEqual(decision.cmd.vx_norm, controller.car_cfg.table_vx_norm_max)

    def test_forward_release_respects_min_distance(self) -> None:
        controller = _controller()
        obs = _obs(dist_err_m=0.05, control_level="approach")
        decision = controller.fov_table_approach_cmd(obs)
        self.assertEqual(decision.cmd.vx_norm, 0.0)
        self.assertEqual(decision.control_summary["forward_block_reason"], "dist_below_min_forward")

    def test_forward_release_slows_for_yaw_and_fov_soft_gate(self) -> None:
        controller = _controller()
        yaw_obs = _obs(dist_err_m=0.30, yaw_err_rad=0.30, control_level="approach", view_err_norm=0.0, plane_cx_norm=0.0)
        yaw_decision = controller.fov_table_approach_cmd(yaw_obs)
        self.assertTrue(0.0 < yaw_decision.control_summary["yaw_gate"] < 1.0)
        self.assertLess(yaw_decision.control_summary["vx_mps_raw"], yaw_decision.control_summary["vx_from_dist"] * controller.car_cfg.vx_mps_per_norm)

        controller = _controller()
        fov_obs = _obs(dist_err_m=0.30, yaw_err_rad=0.0, control_level="approach", view_err_norm=0.30, plane_cx_norm=0.30)
        fov_decision = controller.fov_table_approach_cmd(fov_obs)
        self.assertTrue(0.0 < fov_decision.control_summary["fov_gate"] < 1.0)
        self.assertLess(fov_decision.control_summary["vx_mps_raw"], fov_decision.control_summary["vx_from_dist"] * controller.car_cfg.vx_mps_per_norm)

    def test_stop_level_does_not_release_forward(self) -> None:
        controller = _controller()
        obs = _obs(control_level="stop", usable_for_stop=True, dist_err_m=0.30)
        decision = controller.fov_table_approach_cmd(obs)
        self.assertEqual(decision.cmd.vx_norm, 0.0)
        self.assertFalse(decision.control_summary["forward_allowed"])

    def test_search_table_default_turn_is_conservative(self) -> None:
        controller = _controller()
        decision = controller.search_table_cmd()
        self.assertLessEqual(abs(decision.cmd.wz_norm), 0.12)

    def test_soft_stale_blocks_forward_only(self) -> None:
        controller = _controller()
        obs = _obs(frame_capture_ts=now_ts() - 0.35)
        decision = controller.fov_table_approach_cmd(obs)
        self.assertEqual(decision.control_summary["stale_level"], "soft_stale")
        self.assertEqual(decision.cmd.vx_norm, 0.0)

    def test_hard_stale_outputs_hold(self) -> None:
        controller = _controller()
        obs = _obs(frame_capture_ts=now_ts() - 0.60)
        decision = controller.fov_table_approach_cmd(obs)
        self.assertEqual(decision.control_summary["stale_level"], "hard_stale")
        self.assertEqual((decision.cmd.vx_norm, decision.cmd.vy_norm, decision.cmd.wz_norm), (0.0, 0.0, 0.0))

    def test_dead_stale_outputs_hold(self) -> None:
        controller = _controller()
        obs = _obs(frame_capture_ts=now_ts() - 1.20)
        decision = controller.fov_table_approach_cmd(obs)
        self.assertEqual(decision.control_summary["stale_level"], "dead")
        self.assertEqual((decision.cmd.vx_norm, decision.cmd.vy_norm, decision.cmd.wz_norm), (0.0, 0.0, 0.0))

    def test_stale_obs_cannot_trigger_final_stop_success(self) -> None:
        cfg = ControlThresholds()
        cfg.table_obs_stale_soft_ms = 300
        core = OrchestratorCore(cfg, CarMotionConfig())
        obs = _obs(frame_capture_ts=now_ts() - 0.35, dist_err_m=0.0, yaw_err_rad=0.0, usable_for_stop=True)
        status = core._final_lock_status(obs, stable_count=10)
        self.assertFalse(status["lock_ready"])
        self.assertEqual(status["reason"], "soft_stale")
        self.assertEqual(status["lock_count_hold_reason"], "soft_stale")

    def test_fast_control_level_aliases_normalize(self) -> None:
        core = OrchestratorCore(ControlThresholds(), CarMotionConfig())
        self.assertEqual(core._control_level(_obs(control_level="approach_slow")), "approach")
        self.assertEqual(core._control_level(_obs(control_level="rotate_only")), "alignment")
        self.assertEqual(core._control_level(_obs(control_level="align")), "alignment")
        self.assertEqual(core._control_level(_obs(control_level="stop_ready")), "stop")
        self.assertEqual(core._control_level(_obs(control_level="bad_level")), "none")

    def test_final_lock_count_inc_hold_and_reset(self) -> None:
        cfg = ControlThresholds()
        cfg.table_obs_stale_soft_ms = 300
        cfg.table_obs_stale_stop_ms = 500
        cfg.table_obs_stale_hard_ms = 800
        core = OrchestratorCore(cfg, CarMotionConfig())

        fresh = _obs(control_level="stop_ready", usable_for_stop=True, usable_for_alignment=True, dist_err_m=0.0, yaw_err_rad=0.0)
        status = core._update_final_lock_count(fresh)
        self.assertTrue(status["lock_ready"])
        self.assertEqual(core.ctx.table_lock_frames, 1)
        self.assertEqual(status["lock_count_inc_reason"], "fresh_lock_ready")

        soft = _obs(frame_capture_ts=now_ts() - 0.35, control_level="stop_ready", usable_for_stop=True, usable_for_alignment=True, dist_err_m=0.0, yaw_err_rad=0.0)
        status = core._update_final_lock_count(soft)
        self.assertFalse(status["lock_ready"])
        self.assertEqual(core.ctx.table_lock_frames, 1)
        self.assertEqual(status["lock_count_hold_reason"], "soft_stale")

        hard = _obs(frame_capture_ts=now_ts() - 0.65, control_level="stop_ready", usable_for_stop=True, usable_for_alignment=True, dist_err_m=0.0, yaw_err_rad=0.0)
        status = core._update_final_lock_count(hard)
        self.assertEqual(core.ctx.table_lock_frames, 0)
        self.assertEqual(status["lock_count_reset_reason"], "hard_stale")

    def test_final_lock_approach_enters_stop_and_settle_on_lock_ready(self) -> None:
        cfg = ControlThresholds()
        cfg.enable_final_lock = True
        cfg.enable_micro_adjust = True
        cfg.table_settle_s = 0.0
        cfg.table_stable_frames = 2
        core = OrchestratorCore(cfg, CarMotionConfig())
        core.ctx.state = State.FINAL_LOCK
        core.ctx.table_dock_phase = "APPROACH"
        core.handle_table_obs(_obs(control_level="stop_ready", usable_for_stop=True, usable_for_alignment=True, dist_err_m=0.0, yaw_err_rad=0.0))

        decision = core.tick()

        self.assertEqual(core.ctx.state, State.FINAL_LOCK)
        self.assertEqual(core.ctx.table_dock_phase, "STOP_AND_SETTLE")
        self.assertEqual(core.ctx.table_lock_frames, 1)
        self.assertEqual(core._control_level(core.ctx.last_table_obs), "stop")
        self.assertEqual(decision.cmd.vx_norm, 0.0)

        core.handle_table_obs(_obs(control_level="align", usable_for_stop=False, usable_for_alignment=True, dist_err_m=0.0, yaw_err_rad=0.0))
        core.tick()
        self.assertEqual(core.ctx.state, State.AT_TABLE_EDGE)

    def test_final_lock_disabled_keeps_stop_ready_in_controlled_approach(self) -> None:
        cfg = ControlThresholds()
        cfg.enable_final_lock = False
        core = OrchestratorCore(cfg, CarMotionConfig())
        core.ctx.state = State.CONTROLLED_APPROACH
        core.ctx.task_start_wall_ts = now_ts() - 1.0
        core.handle_table_obs(_obs(control_level="stop_ready", usable_for_stop=True, usable_for_alignment=True, dist_err_m=0.0, yaw_err_rad=0.0))

        decision = core.tick()

        self.assertEqual(core.ctx.state, State.CONTROLLED_APPROACH)
        self.assertNotEqual(decision.cmd.mode, "FINAL_LOCK")
        self.assertFalse(decision.control_summary["final_lock_enabled"])
        self.assertTrue(decision.control_summary["stop_ready_ignored_for_stage_transition"])

    def test_stop_ready_far_distance_stays_in_controlled_approach(self) -> None:
        cfg = ControlThresholds()
        cfg.enable_final_lock = True
        cfg.final_lock_enter_dist_th_m = 0.08
        core = OrchestratorCore(cfg, CarMotionConfig())
        core.ctx.state = State.CONTROLLED_APPROACH
        core.ctx.task_start_wall_ts = now_ts() - 1.0
        core.handle_table_obs(_obs(control_level="stop_ready", usable_for_stop=True, usable_for_alignment=True, dist_err_m=0.12, yaw_err_rad=0.02))

        decision = core.tick()

        self.assertEqual(core.ctx.state, State.CONTROLLED_APPROACH)
        self.assertFalse(decision.control_summary["final_lock_enter_allowed"])
        self.assertEqual(decision.control_summary["final_lock_enter_block_reason"], "distance_too_far")

    def test_stop_ready_near_distance_enters_final_lock(self) -> None:
        cfg = ControlThresholds()
        cfg.enable_final_lock = True
        cfg.final_lock_enter_dist_th_m = 0.08
        cfg.final_lock_enter_yaw_th_rad = 0.10
        core = OrchestratorCore(cfg, CarMotionConfig())
        core.ctx.state = State.CONTROLLED_APPROACH
        core.ctx.task_start_wall_ts = now_ts() - 1.0
        core.handle_table_obs(_obs(control_level="stop_ready", usable_for_stop=True, usable_for_alignment=True, dist_err_m=0.02, yaw_err_rad=0.02))

        decision = core.tick()

        self.assertEqual(core.ctx.state, State.FINAL_LOCK)
        self.assertTrue(decision.control_summary["final_lock_enter_allowed"])
        self.assertEqual(decision.cmd.mode, "FINAL_LOCK")

    def test_missing_frame_capture_ts_falls_back_safely(self) -> None:
        controller = _controller()
        obs = _obs(ts=now_ts(), frame_capture_ts=None)
        decision = controller.fov_table_approach_cmd(obs)
        self.assertIn(decision.control_summary["stale_level"], {"fresh", "soft_stale", "hard_stale", "dead"})
        self.assertIn("obs_total_age_ms", decision.control_summary)

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

    def test_motion_adapter_outputs_physical_velocity(self) -> None:
        adapter = Stm32MotionAdapter(uart=object(), logger=lambda _line: None, vx_scale=100, vy_scale=100, wz_scale=100)
        cmd = CmdVel(ts=time.time(), mode="TEST", vx_norm=0.10, vy_norm=0.02, wz_norm=0.03)
        self.assertEqual(adapter.cmd_vel_to_velocity(cmd), (10.0, 2.0, 3.0))


if __name__ == "__main__":
    unittest.main()
