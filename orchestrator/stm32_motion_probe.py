#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone SC171 -> STM32 motion protocol probe.

This script intentionally does not import or start the orchestrator state
machine, VISTA, or mobile gateway. Protocol fields follow
ROBOT_MOTION_CONTRACT.md:

    VEL <s006> <s007> <s008> <s009> <seq>
    STOP <seq>
    JOG <s006> <s007> <s008> <s009> <duration_ms> <seq>
    STATUS
"""

import argparse
import sys
import time
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import serial  # type: ignore
except Exception:  # pragma: no cover - dry-run does not need pyserial.
    serial = None


WHEEL_MIN = -100
WHEEL_MAX = 100
DURATION_MIN_MS = 20
DURATION_MAX_MS = 1000

JOG_DONE_MARKER = "[CAR][JOG_DONE]"
JOG_BUSY_MARKER = "[CAR][JOG_BUSY]"


def clamp_int(value, lo: int, hi: int) -> int:
    try:
        number = int(round(float(value)))
    except Exception:
        number = 0
    return max(lo, min(hi, number))


def normalize_wheels(values: Sequence[int]) -> Tuple[int, int, int, int]:
    if len(values) != 4:
        raise ValueError("exactly four wheel values are required for s006/s007/s008/s009")
    return tuple(clamp_int(v, WHEEL_MIN, WHEEL_MAX) for v in values)  # type: ignore[return-value]


def format_seq(seq) -> str:
    try:
        return str(int(seq))
    except Exception:
        return "0"


def encode_vel(wheels: Sequence[int], seq: int) -> str:
    s006, s007, s008, s009 = normalize_wheels(wheels)
    return f"VEL {s006} {s007} {s008} {s009} {format_seq(seq)}"


def encode_stop(seq: int) -> str:
    return f"STOP {format_seq(seq)}"


def encode_jog(wheels: Sequence[int], duration_ms: int, seq: int) -> str:
    s006, s007, s008, s009 = normalize_wheels(wheels)
    duration = clamp_int(duration_ms, DURATION_MIN_MS, DURATION_MAX_MS)
    return f"JOG {s006} {s007} {s008} {s009} {duration} {format_seq(seq)}"


def encode_status() -> str:
    return "STATUS"


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
            lines.append(text)
        return lines

    def send_and_read(self, line: str, rx_window_s: float) -> List[str]:
        self.send(line)
        return self.read_for(rx_window_s)


def has_marker(lines: Iterable[str], marker: str) -> bool:
    marker_upper = marker.upper()
    return any(marker_upper in line.upper() for line in lines)


def wait_for_jog_done(probe: MotionProbe, timeout_s: float, poll_s: float) -> bool:
    if probe.dry_run:
        return True

    deadline = time.monotonic() + max(0.0, float(timeout_s))
    saw_busy = False
    while time.monotonic() < deadline:
        lines = probe.read_for(min(max(0.01, poll_s), max(0.0, deadline - time.monotonic())))
        if has_marker(lines, JOG_DONE_MARKER):
            return True
        if has_marker(lines, JOG_BUSY_MARKER):
            saw_busy = True

    if saw_busy:
        print("[PROBE][WARN] saw JOG_BUSY but did not see JOG_DONE before timeout", flush=True)
    else:
        print("[PROBE][WARN] did not see JOG_DONE before timeout", flush=True)
    return False


def run_command(args, probe: MotionProbe) -> int:
    seq = int(args.seq_start)
    wheels = tuple(args.wheels)
    rx_window_s = float(args.rx_window)

    if args.cmd == "status":
        probe.send_and_read(encode_status(), rx_window_s)
        return 0

    if args.cmd == "stop":
        probe.send_and_read(encode_stop(seq), rx_window_s)
        return 0

    if args.cmd == "vel":
        probe.send_and_read(encode_vel(wheels, seq), rx_window_s)
        return 0

    if args.cmd == "jog":
        probe.send_and_read(encode_jog(wheels, args.duration_ms, seq), rx_window_s)
        wait_for_jog_done(probe, args.jog_timeout, args.rx_window)
        return 0

    if args.cmd == "sequence":
        probe.send_and_read(encode_stop(seq), rx_window_s)
        seq += 1
        probe.send_and_read(encode_status(), rx_window_s)
        probe.send_and_read(encode_jog(wheels, args.duration_ms, seq), rx_window_s)
        wait_for_jog_done(probe, args.jog_timeout, args.rx_window)
        seq += 1
        probe.send_and_read(encode_stop(seq), rx_window_s)
        probe.send_and_read(encode_status(), rx_window_s)
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
        choices=("status", "stop", "vel", "jog", "sequence"),
        default="sequence",
        help="probe command to run",
    )
    parser.add_argument("--seq-start", type=int, default=1, help="first sequence number for seq commands")
    parser.add_argument(
        "--wheels",
        nargs=4,
        type=int,
        metavar=("S006", "S007", "S008", "S009"),
        default=(30, 30, 30, 30),
        help="wheel low-speed values, clamped to -100..100",
    )
    parser.add_argument("--duration-ms", type=int, default=100, help="JOG duration, clamped to 20..1000 ms")
    parser.add_argument("--rx-window", type=float, default=0.25, help="short RX drain window after each TX")
    parser.add_argument("--jog-timeout", type=float, default=2.0, help="maximum wait for [CAR][JOG_DONE]")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.wheels = normalize_wheels(args.wheels)
    args.duration_ms = clamp_int(args.duration_ms, DURATION_MIN_MS, DURATION_MAX_MS)
    args.rx_window = max(0.01, float(args.rx_window))
    args.jog_timeout = max(args.rx_window, float(args.jog_timeout))

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
