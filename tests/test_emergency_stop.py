#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import queue
import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORCH_ROOT = ROOT / "orchestrator"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

# Mock serial module to allow full mode UartBridge start
from unittest.mock import MagicMock
sys.modules["serial"] = MagicMock()

from orchestrator_service.bridge.uart_bridge import UartBridge
from orchestrator_service.control.motion_adapter import Stm32MotionAdapter
from orchestrator_service.ipc.protocol import CmdVel


class FakeSerial:
    def __init__(self):
        self.written = []
        self.lock = threading.Lock()
        self.write_delay = 0.001

    def write(self, data: bytes):
        with self.lock:
            # Simulate slight delay to expose potential race conditions
            time.sleep(self.write_delay)
            self.written.append(data.decode("utf-8"))

    def readline(self):
        time.sleep(0.1)
        return b""

    def close(self):
        pass


class EmergencyStopTest(unittest.TestCase):
    def setUp(self):
        self.ser = FakeSerial()
        import serial
        serial.Serial = lambda *args, **kwargs: self.ser

    def test_non_interleaving_concurrent_writes(self) -> None:
        """Verify that concurrent writes of V, arm commands, and STOP do not interleave.

        All commands written to FakeSerial must be complete lines ending in \r\n,
        with no partial commands mixed together.
        """
        bridge = UartBridge("/dev/null", 115200, 0.1, readback_enabled=False)
        bridge.dry_run = False
        bridge.start()

        stop_threads = False

        def arm_writer():
            while not stop_threads:
                bridge.send_arm_command("POSE 10.0 20.0 30.0 0.0 0.0 90.0 1000\n")
                time.sleep(0.001)

        def vel_writer():
            while not stop_threads:
                bridge.send_motion_line("MODE SEARCH\r\nV 0.100 0.200 0.300\r\n", latest_override=False)
                time.sleep(0.001)

        t1 = threading.Thread(target=arm_writer, daemon=True)
        t2 = threading.Thread(target=vel_writer, daemon=True)
        t1.start()
        t2.start()

        # Let them write concurrently for a bit
        time.sleep(0.2)

        # Trigger emergency stop
        bridge.send_emergency_stop()

        time.sleep(0.1)
        stop_threads = True
        t1.join(timeout=1.0)
        t2.join(timeout=1.0)
        bridge.close()

        # Check all written items end with \r\n and do not have interleaving parts
        self.assertTrue(len(self.ser.written) > 0)
        for line in self.ser.written:
            self.assertTrue(line.endswith("\r\n"))
            # Count internal newlines to make sure they are not corrupted or mixed
            # Normal single-line or multi-line commands are okay, but partial lines are not.
            parts = line.split("\r\n")[:-1]
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                # Each part should either be a complete V cmd, POSE cmd, MODE cmd, or STOP/SSTOP
                is_valid = (
                    part.startswith("V ") or 
                    part.startswith("POSE ") or 
                    part.startswith("MODE ") or 
                    part in {"STOP", "SSTOP"}
                )
                self.assertTrue(is_valid, f"Corrupted/Interleaved line detected: {part}")

    def test_first_command_after_estop_is_stop_and_cleared_pending(self) -> None:
        """Verify that STOP is the first command written and pending V commands are cleared."""
        bridge = UartBridge("/dev/null", 115200, 0.1, readback_enabled=False)
        bridge.dry_run = False
        bridge.start()

        # Queue multiple V commands
        bridge.send_motion_line("V 0.1 0.2 0.3\r\n", latest_override=False)
        bridge.send_motion_line("MODE SEARCH\r\nV 0.4 0.5 0.6\r\n", latest_override=False)

        # Immediately call emergency stop
        bridge.send_emergency_stop()

        # Wait for the queue to be processed
        time.sleep(0.1)
        bridge.close()

        # The first written command must be STOP
        self.assertTrue(len(self.ser.written) > 0)
        self.assertEqual(self.ser.written[0], "STOP\r\n")

        # The queued V commands must have been cleared/suppressed
        for line in self.ser.written[1:]:
            self.assertNotIn("V ", line)

    def test_stale_v_and_cooldown_suppression(self) -> None:
        """Verify that stale V commands or V commands during cooldown window are suppressed.

        This includes both single-line "V ..." and multi-line "MODE SEARCH\r\nV ..." commands.
        """
        bridge = UartBridge("/dev/null", 115200, 0.1, readback_enabled=False)
        bridge.dry_run = False
        bridge.start()

        # Trigger estop
        bridge.send_emergency_stop()
        self.assertEqual(self.ser.written[-1], "STOP\r\n")

        # 1. Try sending single-line V command immediately during cooldown
        bridge.send_motion_line("V 0.100 0.000 0.000\r\n")
        
        # 2. Try sending multi-line command immediately during cooldown
        bridge.send_motion_line("MODE SEARCH\r\nV 0.200 0.000 0.000\r\n")

        time.sleep(0.1)

        # Verify that no V commands were written during the cooldown
        for line in self.ser.written[1:]:
            self.assertNotIn("V ", line)

        # Wait out the cooldown period (0.5 seconds)
        time.sleep(0.45)

        # Send V command after cooldown
        bridge.send_motion_line("V 0.500 0.000 0.000\r\n")
        time.sleep(0.1)
        bridge.close()

        # It should now be written
        self.assertIn("V 0.500 0.000 0.000\r\n", self.ser.written)

    def test_jog_velocity_non_blocking_and_preemption(self) -> None:
        """Verify that jog_velocity returns immediately and can be preempted.

        Also verify that a cancelled/preempted jog does not send a final STOP/SSTOP.
        """
        class MockUart:
            def __init__(self):
                self.written = []
                self._stop = threading.Event()
                self._last_estop_mono = 0.0

            def send_motion_line(self, line, tx_meta=None, latest_override=True):
                self.written.append(line)
                return True

            def send_soft_stop(self, tx_meta=None):
                self.written.append("SSTOP\r\n")
                return True

            def send_emergency_stop_mcu(self, tx_meta=None):
                self.written.append("STOP\r\n")
                return True

        uart = MockUart()
        adapter = Stm32MotionAdapter(uart)
        adapter.jog_duration_ms = 100

        # Start a jog velocity command
        t_start = time.monotonic()
        seq = adapter.jog_velocity(0.02, 0.0, 0.0, reason="test_jog")
        t_elapsed = time.monotonic() - t_start

        # jog_velocity must return immediately (non-blocking)
        self.assertLess(t_elapsed, 0.01)

        # Cancel the jog by calling stop()
        adapter.stop(reason="estop_test")

        # Wait for the original jog duration
        time.sleep(0.15)

        # The written commands should contain the initial motion and the STOP from adapter.stop,
        # but NOT the final STOP command from the cancelled jog worker.
        self.assertIn("MODE SEARCH\r\nV 0.020 0.000 0.000\r\n", uart.written)
        self.assertIn("STOP\r\n", uart.written)
        # Check that we only have 2 commands (initial motion + manual stop)
        self.assertEqual(len(uart.written), 2)


if __name__ == "__main__":
    unittest.main()
