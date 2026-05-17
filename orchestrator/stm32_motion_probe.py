#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone SC171 -> STM32 motion protocol probe.

This script intentionally does not import or start the orchestrator state
machine, VISTA, or mobile gateway. Do not run it at the same time as the
full orchestrator on the same serial port.

Current mature STM32 protocol:

    MODE SEARCH
    MODE RETURN
    V <vx_mps> <vy_mps> <wz_radps>
    STOP

The old seq-based commands are kept for compatibility but are no longer the
default:

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
DEFAULT_PERIOD_MS = 100
SWEEP_VX_MPS = (0.005, 0.010, 0.015, 0.020, 0.030, 0.050)
SWEEP_DURATION_MS = 1000

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


def encode_mode_search() -> str:
    return "MODE SEARCH"


def encode_mode_return() -> str:
    return "MODE RETURN"


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
    seq = int(args.seq_start)
    wheels = tuple(args.wheels)
    rx_window_s = float(args.rx_window)

    if args.cmd == "mode-search":
        probe.send_and_read(encode_mode_search(), rx_window_s)
        return 0

    if args.cmd == "mode-return":
        probe.send_and_read(encode_mode_return(), rx_window_s)
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
        for vx_mps in SWEEP_VX_MPS:
            send_v_for(probe, vx_mps, args.vy, args.wz, SWEEP_DURATION_MS, args.period_ms)
            probe.send_and_read(encode_bare_stop(), rx_window_s)
        return 0

    if args.cmd == "status":
        probe.send_and_read(encode_status(), rx_window_s)
        return 0

    if args.cmd == "legacy-stop":
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
        choices=(
            "mode-search",
            "mode-return",
            "stop",
            "v",
            "pulse",
            "sweep",
            "status",
            "legacy-stop",
            "vel",
            "jog",
            "sequence",
        ),
        default="stop",
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
    parser.add_argument("--duration-ms", type=int, default=100, help="JOG/pulse duration in ms; JOG is clamped to 20..1000 ms")
    parser.add_argument("--period-ms", type=int, default=DEFAULT_PERIOD_MS, help="V resend period for pulse/sweep")
    parser.add_argument("--vx", type=float, default=0.010, help="forward velocity in m/s for V/pulse")
    parser.add_argument("--vy", type=float, default=0.0, help="lateral velocity in m/s for V/pulse/sweep")
    parser.add_argument("--wz", type=float, default=0.0, help="yaw velocity in rad/s for V/pulse/sweep")
    parser.add_argument("--rx-window", type=float, default=0.25, help="short RX drain window after each TX")
    parser.add_argument("--jog-timeout", type=float, default=2.0, help="maximum wait for [CAR][JOG_DONE]")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.wheels = normalize_wheels(args.wheels)
    args.duration_ms = max(0, int(args.duration_ms))
    args.period_ms = max(1, int(args.period_ms))
    args.rx_window = max(0.01, float(args.rx_window))
    args.jog_timeout = max(args.rx_window, float(args.jog_timeout))
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
