#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORCH_ROOT = ROOT / "orchestrator"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from orchestrator_service.config.schema import OrchestratorConfig
from orchestrator_service.runtime.context import RuntimeContext, State
from orchestrator_service.runtime.state_machine import OrchestratorCore
from orchestrator_service.bridge.uart_bridge import UartBridge
from orchestrator_service.control.motion_adapter import Stm32MotionAdapter
from orchestrator_service.runtime.safety.base_motion_safety import apply_base_motion_safety


class GraspRepositionTest(unittest.TestCase):
    def setUp(self):
        self.cfg = OrchestratorConfig()
        self.cfg.car.grasp_reposition_speed_cm_s = 10.0
        self.cfg.car.pre_arm_stop_settle_ms = 150
        self.core = OrchestratorCore(self.cfg.control, self.cfg.car, self.cfg.docking)
        self.core.ctx.active_target = "apple"

    def test_dx_only_reposition(self):
        """dx-only reposition uses the correct original coordinate mapping and duration."""
        self.core.ctx.state = State.GRASP
        self.core.ctx.grasp_substate = "REPOSITIONING"
        self.core.ctx.grasp_reposition_proposal = {"dx_cm": 10.0, "dy_cm": 0.0}
        
        now = 100.0
        self.core.ctx.grasp_timeout_mono = now + 5.0
        self.core.ctx.grasp_reposition_start_mono = now
        
        # Trigger first tick
        decision = self.core._tick_grasp_repositioning(now)
        self.assertEqual(decision.cmd.mode, "GRASP_REPOSITION")
        
        # Elapsed 0.2s
        decision = self.core._tick_grasp_repositioning(now + 0.2)
        self.assertEqual(decision.cmd.mode, "GRASP_REPOSITION")
        self.assertAlmostEqual(decision.cmd.vx_mps, 0.10)
        self.assertAlmostEqual(decision.cmd.vy_mps, 0.0)
        self.assertAlmostEqual(decision.cmd.wz_radps, 0.0)

    def test_dy_only_reposition(self):
        """dy-only reposition uses the correct original coordinate mapping and duration."""
        self.core.ctx.state = State.GRASP
        self.core.ctx.grasp_substate = "REPOSITIONING"
        self.core.ctx.grasp_reposition_proposal = {"dx_cm": 0.0, "dy_cm": -10.0}
        
        now = 100.0
        self.core.ctx.grasp_timeout_mono = now + 5.0
        self.core.ctx.grasp_reposition_start_mono = now
        
        # Trigger first tick
        self.core._tick_grasp_repositioning(now)
        
        # Elapsed 0.2s
        decision = self.core._tick_grasp_repositioning(now + 0.2)
        self.assertEqual(decision.cmd.mode, "GRASP_REPOSITION")
        self.assertAlmostEqual(decision.cmd.vx_mps, 0.0)
        self.assertAlmostEqual(decision.cmd.vy_mps, -0.10)
        self.assertAlmostEqual(decision.cmd.wz_radps, 0.0)

    def test_diagonal_reposition_normalization(self):
        """diagonal reposition resultant velocity magnitude does not exceed the configured speed."""
        self.core.ctx.state = State.GRASP
        self.core.ctx.grasp_substate = "REPOSITIONING"
        self.core.ctx.grasp_reposition_proposal = {"dx_cm": 10.0, "dy_cm": 10.0}
        
        now = 100.0
        self.core.ctx.grasp_timeout_mono = now + 5.0
        self.core.ctx.grasp_reposition_start_mono = now
        
        # Trigger first tick
        self.core._tick_grasp_repositioning(now)
        
        # Elapsed 0.2s
        decision = self.core._tick_grasp_repositioning(now + 0.2)
        
        # Resultant speed should be grasp_reposition_speed_cm_s / 100.0 = 0.10 m/s
        resultant_speed = math.hypot(decision.cmd.vx_mps, decision.cmd.vy_mps)
        self.assertAlmostEqual(resultant_speed, 0.10)
        self.assertAlmostEqual(decision.cmd.vx_mps, decision.cmd.vy_mps)
        self.assertTrue(decision.cmd.vx_mps > 0)
        
        # Duration = hypot(10, 10)/10 = 1.4142s
        # If we tick after 1.5s, it should stop
        decision_stop = self.core._tick_grasp_repositioning(now + 1.5)
        self.assertEqual(decision_stop.cmd.vx_mps, 0.0)
        self.assertEqual(decision_stop.cmd.vy_mps, 0.0)
        self.assertEqual(self.core.ctx.grasp_substate, "AWAITING_RESPOND")

    def test_reposition_completion_emits_stop(self):
        """reposition completion emits STOP or zero velocity, not SSTOP."""
        self.core.ctx.state = State.GRASP
        self.core.ctx.grasp_substate = "REPOSITIONING"
        self.core.ctx.grasp_reposition_proposal = {"dx_cm": 10.0, "dy_cm": 0.0}
        
        now = 100.0
        self.core.ctx.grasp_timeout_mono = now + 5.0
        self.core.ctx.grasp_reposition_start_mono = now
        
        # Trigger first tick
        self.core._tick_grasp_repositioning(now)
        
        # Tick at 101.1 (elapsed 1.1s > 1.0s duration)
        decision = self.core._tick_grasp_repositioning(now + 1.1)
        self.assertEqual(decision.cmd.vx_mps, 0.0)
        self.assertEqual(decision.cmd.vy_mps, 0.0)
        self.assertEqual(decision.cmd.wz_radps, 0.0)
        self.assertEqual(decision.cmd.mode, "GRASP")
        
        # Verify it maps to a hard stop
        is_soft = decision.cmd.mode in {"IDLE", "DONE", "AT_TABLE_EDGE"}
        self.assertFalse(is_soft)

    def test_tick_overrun_guard(self):
        """tick-overrun guard triggers early stop if remaining duration is <= 0.08s."""
        self.core.ctx.state = State.GRASP
        self.core.ctx.grasp_substate = "REPOSITIONING"
        self.core.ctx.grasp_reposition_proposal = {"dx_cm": 10.0, "dy_cm": 0.0}
        
        now = 100.0
        self.core.ctx.grasp_timeout_mono = now + 5.0
        self.core.ctx.grasp_reposition_start_mono = now
        
        # Trigger first tick
        self.core._tick_grasp_repositioning(now)
        
        # Duration = 1.0s. Remaining = 0.05s <= 0.08s when ticking at 100.95.
        decision = self.core._tick_grasp_repositioning(now + 0.95)
        self.assertEqual(decision.cmd.vx_mps, 0.0)
        self.assertEqual(decision.cmd.vy_mps, 0.0)
        self.assertEqual(decision.cmd.mode, "GRASP")
        self.assertEqual(self.core.ctx.grasp_substate, "AWAITING_RESPOND")
        self.assertIsNone(self.core.ctx.grasp_reposition_proposal)

    def test_grasp_non_repositioning_forces_zero_velocity(self):
        """GRASP non-REPOSITIONING substate forces zero base velocity."""
        self.core.ctx.state = State.GRASP
        self.core.ctx.grasp_substate = "AWAITING_RESULT"
        
        decision = self.core.controller.stop_cmd("GRASP")
        decision.cmd.vx_mps = 0.1
        decision.cmd.vy_mps = 0.1
        decision.cmd.wz_radps = 0.1
        
        # Use self.core.cfg (which is cfg.control threshold object)
        decision = apply_base_motion_safety(decision, ctx=self.core.ctx, cfg=self.core.cfg)
        self.assertEqual(decision.cmd.vx_mps, 0.0)
        self.assertEqual(decision.cmd.vy_mps, 0.0)
        self.assertEqual(decision.cmd.wz_radps, 0.0)

    def test_arm_command_safety_stop_sequence(self):
        """arm command after previous non-zero motion writes STOP before arm command."""
        written = []
        def tx_callback(line, dry_run, meta):
            written.append(line)
            
        uart = UartBridge("COM1", 9600, 1.0, dry_run=True, tx_callback=tx_callback)
        uart.start()
        
        uart.send_velocity(0.10, 0.0, 0.0)
        time.sleep(0.05)
        
        uart.send_arm_command("POSE 10 10 10 0 0 0 500\r\n")
        
        self.assertTrue(len(written) >= 2)
        self.assertEqual(written[-2], "STOP\r\n")
        self.assertEqual(written[-1], "POSE 10 10 10 0 0 0 500\r\n")
        uart.close()

    def test_stale_velocity_suppression(self):
        """pending stale V commands are not written after the pre-arm STOP."""
        written = []
        def tx_callback(line, dry_run, meta):
            written.append(line)
            
        uart = UartBridge("COM1", 9600, 1.0, dry_run=True, tx_callback=tx_callback)
        uart.start()
        
        uart.send_motion_line("V 0.200 0.000 0.000\r\n", latest_override=False)
        uart.send_arm_command("POSE 10 10 10 0 0 0 500\r\n")
        
        for line in written:
            self.assertFalse(line.startswith("V "))
        uart.close()


if __name__ == "__main__":
    unittest.main()
