#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone SC171 -> STM32 car motion sweep.

Do not run this script while the orchestrator is using the same UART.
"""

import sys
import time

try:
    import serial  # type: ignore
except Exception:
    serial = None


PORT = "/dev/ttyHS1"
BAUD = 115200
TX_HZ = 10.0
TX_PERIOD_S = 1.0 / TX_HZ
STEP = 0.001
DEFAULT_MAX_VALUE = 0.200
HOLD_S = 0.2
STOP_KEEPALIVE_S = 1.0
VALID_AXES = {"vx", "vy", "wz"}


def _send_line(ser, line: str) -> None:
    ser.write((str(line).strip() + "\r\n").encode("utf-8"))
    ser.flush()


def _velocity_for_axis(axis: str, value: float):
    if axis == "vx":
        return value, 0.0, 0.0
    if axis == "vy":
        return 0.0, value, 0.0
    return 0.0, 0.0, value


def _send_stop_keepalive(ser) -> None:
    deadline = time.monotonic() + STOP_KEEPALIVE_S
    while time.monotonic() < deadline:
        _send_line(ser, "STOP")
        time.sleep(TX_PERIOD_S)


def _run_sweep(ser, axis: str, target_val: float) -> None:
    _send_line(ser, "MODE SEARCH")
    next_print = time.monotonic()
    
    # 计算需要执行的总步数，以目标值的绝对值为准
    steps = int(round(abs(target_val) / STEP))
    
    # 动态判断正负方向
    actual_step = STEP if target_val >= 0 else -STEP

    for index in range(steps + 1):
        value = index * actual_step
        vx, vy, wz = _velocity_for_axis(axis, value)
        line = f"V {vx:.3f} {vy:.3f} {wz:.3f}"
        step_deadline = time.monotonic() + HOLD_S
        while time.monotonic() < step_deadline:
            _send_line(ser, line)
            now = time.monotonic()
            if now >= next_print:
                print(f"[SWEEP] axis={axis} value={value:.3f} tx_hz={TX_HZ:.1f}", flush=True)
                next_print = now + 1.0
            time.sleep(min(TX_PERIOD_S, max(0.0, step_deadline - time.monotonic())))


def main() -> int:
    axis = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "vx"
    
    # 允许接收可选的第三个参数：目标速度
    if len(sys.argv) > 3 or axis not in VALID_AXES:
        print("usage: python3 scripts/car_motion_sweep.py [vx|vy|wz] [target_velocity]", file=sys.stderr)
        return 2

    target_val = DEFAULT_MAX_VALUE
    if len(sys.argv) == 3:
        try:
            target_val = float(sys.argv[2])
        except ValueError:
            print("[SWEEP][ERROR] target_velocity must be a valid number", file=sys.stderr)
            return 2

    if serial is None:
        print("[SWEEP][ERROR] pyserial is not installed; cannot open STM32 UART", file=sys.stderr)
        return 2

    ser = None
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1, write_timeout=0.1)
    except Exception as exc:
        print(f"[SWEEP][ERROR] failed to open {PORT} @ {BAUD}: {exc}", file=sys.stderr)
        return 2

    print(f"[SWEEP] opened {PORT} @ {BAUD}, axis={axis}, target_val={target_val}", flush=True)
    try:
        _run_sweep(ser, axis, target_val)
    except KeyboardInterrupt:
        print("\n[SWEEP] interrupted, sending STOP keepalive", flush=True)
    finally:
        if ser is not None:
            try:
                _send_stop_keepalive(ser)
                print("[SWEEP] STOP keepalive sent for 1.0s", flush=True)
            finally:
                ser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())