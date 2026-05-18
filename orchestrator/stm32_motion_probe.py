#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone SC171 -> STM32 motion protocol probe.

This script intentionally does not import or start the orchestrator state
machine, VISTA, or mobile gateway. Do not run it at the same time as the
full orchestrator on the same serial port.

STM32 quick-test protocol:

    MODE SEARCH
    MODE RETURN
    MODE AUTOSEARCH
    MODE AUTOEXPLORE
    STOP
    V <vx_mps> <vy_mps> <wz_radps>

V is only effective on STM32 while in SEARCH or RETURN mode. STM32 echoes
accepted input as:

    FB <original command>
"""

import argparse
import sys
import time
from typing import List, Optional, Sequence

try:
    import serial  # type: ignore
except Exception:  # pragma: no cover - dry-run does not need pyserial.
    serial = None


DEFAULT_PERIOD_MS = 100
SWEEP_VX_MPS = (0.01, 0.03, 0.05, 0.08, 0.10)
SWEEP_DURATION_MS = 1000


def encode_mode_search() -> str:
    return "MODE SEARCH"


def encode_mode_return() -> str:
    return "MODE RETURN"


def encode_mode_autosearch() -> str:
    return "MODE AUTOSEARCH"


def encode_mode_autoexplore() -> str:
    return "MODE AUTOEXPLORE"


def encode_v(vx_mps: float, vy_mps: float, wz_radps: float) -> str:
    return f"V {float(vx_mps):.6g} {float(vy_mps):.6g} {float(wz_radps):.6g}"


def encode_bare_stop() -> str:
    return "STOP"


class MotionProbe:
    def __init__(self, ser, dry_run: bool):
        self.ser = ser
        self.dry_run = bool(dry_run)

    def send(self, line: str) -> None:
        line = str(line).strip()
        print(f"[PROBE][TX] {line}", flush=True)
        if self.dry_run:
            return
        payload = (line + "\n").encode("utf-8")
        self.ser.write(payload)
        self.ser.flush()

    def read_for(self, timeout_s: float) -> List[str]:
        lines: List[str] = []
        if self.dry_run or self.ser is None:
            return lines

        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue
            text = raw.decode("utf-8", errors="ignore").strip()
            if not text:
                continue
            print(f"[PROBE][RX] {text}", flush=True)
            if text.startswith("FB "):
                print(f"[PROBE][FB] echo={text[3:].strip()}", flush=True)
            lines.append(text)
        return lines

    def send_and_read(self, line: str, rx_window_s: float) -> List[str]:
        self.send(line)
        return self.read_for(rx_window_s)


def send_v_for(probe: MotionProbe, vx_mps: float, vy_mps: float, wz_radps: float, duration_ms: int, period_ms: int) -> None:
    duration_s = max(0.0, float(duration_ms) / 1000.0)
    period_s = max(0.001, float(period_ms) / 1000.0)
    deadline = time.monotonic() + duration_s
    line = encode_v(vx_mps, vy_mps, wz_radps)

    while True:
        now = time.monotonic()
        if now >= deadline:
            return
        probe.send(line)
        wait_s = min(period_s, max(0.0, deadline - time.monotonic()))
        if probe.dry_run:
            time.sleep(wait_s)
        else:
            probe.read_for(wait_s)


def run_command(args, probe: MotionProbe) -> int:
    rx_window_s = float(args.rx_window)

    if args.cmd == "mode-search":
        probe.send_and_read(encode_mode_search(), rx_window_s)
        return 0

    if args.cmd == "mode-return":
        probe.send_and_read(encode_mode_return(), rx_window_s)
        return 0

    if args.cmd == "mode-autosearch":
        probe.send_and_read(encode_mode_autosearch(), rx_window_s)
        return 0

    if args.cmd == "mode-autoexplore":
        probe.send_and_read(encode_mode_autoexplore(), rx_window_s)
        return 0

    if args.cmd == "stop":
        probe.send_and_read(encode_bare_stop(), rx_window_s)
        return 0

    if args.cmd == "v":
        probe.send_and_read(encode_v(args.vx, args.vy, args.wz), rx_window_s)
        return 0

    if args.cmd == "pulse":
        probe.send_and_read(encode_mode_search(), rx_window_s)
        send_v_for(probe, args.vx, args.vy, args.wz, args.duration_ms, args.period_ms)
        probe.send_and_read(encode_bare_stop(), rx_window_s)
        return 0

    if args.cmd == "sweep":
        probe.send_and_read(encode_mode_search(), rx_window_s)
        for index, vx_mps in enumerate(SWEEP_VX_MPS):
            send_v_for(probe, vx_mps, args.vy, args.wz, SWEEP_DURATION_MS, args.period_ms)
            probe.send_and_read(encode_bare_stop(), rx_window_s)
            if index != len(SWEEP_VX_MPS) - 1:
                probe.send_and_read(encode_mode_search(), rx_window_s)
        return 0

    if args.cmd == "sample":
        sample_lines = (
            encode_mode_search(),
            "V 0.10 0 0",
            "V 0 0.10 0",
            "V 0 0 0.50",
            encode_bare_stop(),
        )
        for line in sample_lines:
            probe.send_and_read(line, rx_window_s)
        return 0

    print(f"[PROBE][ERROR] unsupported command: {args.cmd}", file=sys.stderr)
    return 2


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Standalone probe for the SC171/STM32 motion protocol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", default="/dev/ttyHS1", help="STM32 UART device")
    parser.add_argument("--baud", type=int, default=115200, help="UART baudrate")
    parser.add_argument("--timeout", type=float, default=0.1, help="serial readline/write timeout in seconds")
    parser.add_argument("--dry-run", action="store_true", help="print TX lines without opening the serial port")
    parser.add_argument(
        "--cmd",
        choices=(
            "mode-search",
            "mode-return",
            "mode-autosearch",
            "mode-autoexplore",
            "stop",
            "v",
            "pulse",
            "sweep",
            "sample",
        ),
        default="stop",
        help="probe command to run",
    )
    parser.add_argument("--duration-ms", type=int, default=100, help="pulse duration in ms")
    parser.add_argument("--period-ms", type=int, default=DEFAULT_PERIOD_MS, help="V resend period for pulse/sweep")
    parser.add_argument("--vx", type=float, default=0.10, help="forward velocity in m/s for V/pulse")
    parser.add_argument("--vy", type=float, default=0.0, help="lateral velocity in m/s for V/pulse/sweep")
    parser.add_argument("--wz", type=float, default=0.0, help="yaw velocity in rad/s for V/pulse/sweep")
    parser.add_argument("--rx-window", type=float, default=0.25, help="short RX drain window after each TX")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.duration_ms = max(0, int(args.duration_ms))
    args.period_ms = max(1, int(args.period_ms))
    args.rx_window = max(0.01, float(args.rx_window))
    print("[PROBE][WARN] do not run this probe and the full orchestrator on the same serial port at the same time", flush=True)

    ser = None
    if not args.dry_run:
        if serial is None:
            print("[PROBE][ERROR] pyserial is not installed; use --dry-run for offline checks", file=sys.stderr)
            return 2
        ser = serial.Serial(args.port, args.baud, timeout=args.timeout, write_timeout=args.timeout)
        print(f"[PROBE][INFO] opened {args.port} @ {args.baud}", flush=True)

    try:
        return run_command(args, MotionProbe(ser, args.dry_run))
    finally:
        if ser is not None:
            ser.close()
            print("[PROBE][INFO] serial closed", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
