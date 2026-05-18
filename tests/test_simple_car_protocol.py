#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORCH_ROOT = ROOT / "orchestrator"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from orchestrator_service.bridge.simple_car_protocol import (  # noqa: E402
    encode_jog,
    encode_mode,
    encode_status,
    encode_stm32_jog,
    encode_stm32_status,
    encode_stm32_stop,
    encode_stm32_vel,
    encode_stop,
    encode_vel,
    parse_car_state_line,
)
from orchestrator_service.bridge.uart_bridge import UartBridge  # noqa: E402
from orchestrator_service.control.motion_adapter import Stm32MotionAdapter  # noqa: E402
from orchestrator_service.ipc.protocol import CmdVel  # noqa: E402
from orchestrator_service.runtime.service import OrchestratorService  # noqa: E402


class SimpleCarProtocolTest(unittest.TestCase):
    def test_stm32_command_encoding(self) -> None:
        self.assertEqual(encode_mode("SEARCH_TABLE"), "MODE SEARCH")
        self.assertEqual(encode_mode("RETURN_HOME"), "MODE RETURN")
        self.assertEqual(encode_mode("AUTOSEARCH"), "MODE AUTOSEARCH")
        self.assertEqual(encode_vel(0.1, 0, 0.5), "V 0.100 0.000 0.500")
        self.assertEqual(encode_stop(13), "STOP")
        self.assertEqual(encode_jog(0.02, 0, 0), "V 0.020 0.000 0.000")
        self.assertEqual(encode_status(), "")
        self.assertEqual(encode_stm32_vel(0.1, -0.1, 0.05, 0, 12), "V 0.100 -0.100 0.050")
        self.assertEqual(encode_stm32_stop(13), "STOP")
        self.assertEqual(encode_stm32_jog(0.02, 0, 0, 100), "V 0.020 0.000 0.000")
        self.assertEqual(encode_stm32_status(), "")

    def test_stm32_feedback_parse(self) -> None:
        fb_mode = parse_car_state_line("FB MODE SEARCH")
        self.assertIsNotNone(fb_mode)
        self.assertEqual(fb_mode.state, "FB")
        self.assertTrue(fb_mode.ok)
        self.assertEqual(fb_mode.mode, "SEARCH")
        self.assertEqual(fb_mode.message, "MODE SEARCH")

        fb_v = parse_car_state_line("FB V 0.100 0.000 0.500")
        self.assertIsNotNone(fb_v)
        self.assertEqual(fb_v.state, "FB")
        self.assertTrue(fb_v.ok)
        self.assertEqual(fb_v.vx, 0.1)
        self.assertEqual(fb_v.wz, 0.5)

        fb_stop = parse_car_state_line("FB STOP")
        self.assertIsNotNone(fb_stop)
        self.assertEqual(fb_stop.mode, "STOP")

        ack = parse_car_state_line("ACK_START seq=12")
        self.assertIsNotNone(ack)
        self.assertEqual(ack.state, "ACK_START")
        self.assertTrue(ack.ok)
        self.assertEqual(ack.message, "seq=12")

        done = parse_car_state_line("ACK_DONE seq=12")
        self.assertIsNotNone(done)
        self.assertEqual(done.state, "DONE")
        self.assertTrue(done.ok)

        busy = parse_car_state_line("BUSY seq=12")
        self.assertIsNotNone(busy)
        self.assertEqual(busy.state, "BUSY")
        self.assertTrue(busy.ok)

        status = parse_car_state_line("STATUS target=1 applied=0 jog=0")
        self.assertIsNotNone(status)
        self.assertEqual(status.state, "STATUS")
        self.assertTrue(status.ok)
        self.assertEqual(status.message, "target=1 applied=0 jog=0")

        jog_start = parse_car_state_line("[CAR][JOG_START] seq=12")
        self.assertIsNotNone(jog_start)
        self.assertEqual(jog_start.state, "ACK_START")
        self.assertTrue(jog_start.ok)
        self.assertEqual(jog_start.message, "seq=12")

        jog_done = parse_car_state_line("[CAR][JOG_DONE] seq=12")
        self.assertIsNotNone(jog_done)
        self.assertEqual(jog_done.state, "DONE")
        self.assertTrue(jog_done.ok)
        self.assertEqual(jog_done.message, "seq=12")

        jog_busy = parse_car_state_line("[CAR][JOG_BUSY] seq=12")
        self.assertIsNotNone(jog_busy)
        self.assertEqual(jog_busy.state, "BUSY")
        self.assertTrue(jog_busy.ok)

        timeout = parse_car_state_line("[CAR][TIMEOUT] auto stop")
        self.assertIsNotNone(timeout)
        self.assertEqual(timeout.state, "TIMEOUT")
        self.assertTrue(timeout.timeout)
        self.assertFalse(timeout.ok)
        self.assertEqual(timeout.message, "auto stop")

    def test_uart_bridge_dry_run_mode_v_stop(self) -> None:
        captured = []
        bridge = UartBridge(
            "/dev/null",
            115200,
            0.01,
            dry_run=True,
            tx_callback=lambda line, dry_run, meta: captured.append((line, dry_run, meta)),
        )
        out = StringIO()
        with redirect_stdout(out):
            bridge.start()
            self.assertTrue(bridge.send_motion_line("MODE SEARCH\nV 0.100 0.000 0.500", tx_meta={"kind": "stm32_vel"}, latest_override=False))
            self.assertTrue(bridge.send_motion_line("STOP", tx_meta={"kind": "stm32_stop"}, latest_override=False))
            deadline = time.time() + 1.0
            while len(captured) < 2 and time.time() < deadline:
                time.sleep(0.01)
            bridge.close()
        self.assertGreaterEqual(len(captured), 2)
        self.assertEqual(captured[0][0], "MODE SEARCH\nV 0.100 0.000 0.500\n")
        self.assertEqual(captured[-1][0], "STOP\n")
        self.assertTrue(captured[-1][1])
        self.assertIn("[MOTION][DRYRUN_TX] MODE SEARCH", out.getvalue())
        self.assertIn("[MOTION][DRYRUN_TX] V 0.100 0.000 0.500", out.getvalue())

    def test_service_tracks_stm32_feedback(self) -> None:
        service = object.__new__(OrchestratorService)
        service.motion_status = {
            "last_seq": 12,
            "last_ack_seq": None,
            "last_done_seq": None,
            "jog_running": False,
            "stm32_timeout_seen": False,
            "last_rx_time": None,
        }

        start = parse_car_state_line("[CAR][JOG_START] seq=12")
        service._update_stm32_motion_status(start)
        self.assertEqual(service.motion_status["last_ack_seq"], 12)
        self.assertTrue(service.motion_status["jog_running"])
        self.assertFalse(service.motion_status["stm32_timeout_seen"])

        done = parse_car_state_line("ACK_DONE seq=12")
        service._update_stm32_motion_status(done)
        self.assertEqual(service.motion_status["last_done_seq"], 12)
        self.assertFalse(service.motion_status["jog_running"])

        timeout = parse_car_state_line("[CAR][TIMEOUT] auto stop")
        service._update_stm32_motion_status(timeout)
        self.assertTrue(service.motion_status["stm32_timeout_seen"])
        self.assertFalse(service.motion_status["jog_running"])

    def test_stm32_motion_adapter_clamps_logs_and_sends(self) -> None:
        class FakeUart:
            def __init__(self) -> None:
                self.calls = []

            def send_stm32_vel(self, *args, **kwargs):
                self.calls.append(("vel", args, kwargs))
                return True

            def send_stm32_stop(self, *args, **kwargs):
                self.calls.append(("stop", args, kwargs))
                return True

            def send_stm32_jog(self, *args, **kwargs):
                self.calls.append(("jog", args, kwargs))
                return True

            def send_stm32_status(self, *args, **kwargs):
                self.calls.append(("status", args, kwargs))
                return True

            def send_mode(self, *args, **kwargs):
                self.calls.append(("mode", args, kwargs))
                return True

            def send_motion_line(self, *args, **kwargs):
                self.calls.append(("line", args, kwargs))
                return True

        logs = []
        uart = FakeUart()
        adapter = Stm32MotionAdapter(uart, logger=logs.append)

        self.assertEqual(adapter.set_velocity(0.1, -0.1, 0.5, reason="track"), 1)
        self.assertEqual(adapter.stop(reason="halt"), 2)
        self.assertEqual(adapter.jog_velocity(0.02, 0.0, 0.0, reason="nudge"), 3)
        adapter.query_status()

        self.assertEqual(uart.calls[0][0], "line")
        self.assertEqual(uart.calls[0][1][0], "MODE SEARCH\nV 0.100 -0.100 0.500")
        self.assertEqual(uart.calls[1][0], "stop")
        self.assertEqual(uart.calls[2][0], "line")
        self.assertEqual(uart.calls[2][1][0], "MODE SEARCH\nV 0.020 0.000 0.000")
        self.assertEqual(uart.calls[3][0], "line")
        self.assertEqual(uart.calls[3][1][0], "STOP")

        self.assertEqual(logs[0], "[MOTION][V] seq=1 mode=SEARCH vx=0.100 vy=-0.100 wz=0.500 reason=track")
        self.assertEqual(logs[1], "[MOTION][STOP] seq=2 reason=halt")
        self.assertEqual(logs[2], "[MOTION][PULSE] seq=3 mode=SEARCH vx=0.020 vy=0.000 wz=0.000 duration_ms=100 reason=nudge")
        self.assertEqual(logs[3], "[MOTION][STATUS] skipped: current STM32 protocol uses FB echoes only")

    def test_stm32_motion_adapter_maps_cmd_vel_to_wheels(self) -> None:
        class FakeUart:
            def __init__(self) -> None:
                self.calls = []

            def send_stm32_vel(self, *args, **kwargs):
                self.calls.append(("vel", args, kwargs))
                return True

            def send_stm32_stop(self, *args, **kwargs):
                self.calls.append(("stop", args, kwargs))
                return True

            def send_motion_line(self, *args, **kwargs):
                self.calls.append(("line", args, kwargs))
                return True

        uart = FakeUart()
        adapter = Stm32MotionAdapter(uart, logger=lambda _line: None, vx_scale=1, vy_scale=1, wz_scale=1)
        cmd = CmdVel(ts=0.0, mode="TRACK", vx_norm=0.20, vy_norm=0.10, wz_norm=0.05)

        self.assertEqual(adapter.cmd_vel_to_wheels(cmd), (0.20, 0.10, 0.05))
        self.assertEqual(adapter.send_cmd_vel(cmd, reason="track"), 1)
        self.assertEqual(uart.calls[-1][0], "line")
        self.assertEqual(uart.calls[-1][1][0], "MODE SEARCH\nV 0.200 0.100 0.050")

        stop_cmd = CmdVel(ts=0.0, mode="STOP", vx_norm=0.0, vy_norm=0.0, wz_norm=0.0)
        self.assertEqual(adapter.send_cmd_vel(stop_cmd, reason="stop"), 2)
        self.assertEqual(uart.calls[-1][0], "stop")
        self.assertEqual(uart.calls[-1][1][0], 2)

        limited = Stm32MotionAdapter(uart, logger=lambda _line: None, vx_scale=1, vy_scale=1, wz_scale=1)
        fast_cmd = CmdVel(ts=0.0, mode="TRACK", vx_norm=1.0, vy_norm=1.0, wz_norm=1.0)
        self.assertEqual(limited.cmd_vel_to_wheels(fast_cmd), (1.0, 1.0, 1.0))

    def test_stm32_motion_adapter_small_jogs(self) -> None:
        class FakeUart:
            def __init__(self) -> None:
                self.calls = []

            def send_stm32_jog(self, *args, **kwargs):
                self.calls.append(("jog", args, kwargs))
                return True

            def send_motion_line(self, *args, **kwargs):
                self.calls.append(("line", args, kwargs))
                return True

        logs = []
        uart = FakeUart()
        adapter = Stm32MotionAdapter(
            uart,
            logger=logs.append,
            jog_forward_speed=0.03,
            jog_turn_speed=0.02,
            jog_duration_ms=100,
        )

        self.assertEqual(adapter.jog_forward_small(reason="final_forward"), 1)
        self.assertEqual(uart.calls[-2][1][0], "MODE SEARCH\nV 0.030 0.000 0.000")
        self.assertEqual(uart.calls[-1][1][0], "STOP")
        self.assertEqual(logs[-1], "[MOTION][PULSE] seq=1 mode=SEARCH vx=0.030 vy=0.000 wz=0.000 duration_ms=100 reason=final_forward")

        self.assertEqual(adapter.jog_backward_small(reason="final_back"), 2)
        self.assertEqual(uart.calls[-2][1][0], "MODE SEARCH\nV -0.030 0.000 0.000")

        self.assertEqual(adapter.jog_turn_left_small(reason="align_left"), 3)
        self.assertEqual(uart.calls[-2][1][0], "MODE SEARCH\nV 0.000 0.000 0.020")

        self.assertEqual(adapter.jog_turn_right_small(reason="align_right"), 4)
        self.assertEqual(uart.calls[-2][1][0], "MODE SEARCH\nV 0.000 0.000 -0.020")


if __name__ == "__main__":
    unittest.main()
