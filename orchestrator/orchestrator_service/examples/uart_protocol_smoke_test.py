#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
直接对 STM32 发送分行协议（归一化速度），脱离状态机验证底盘响应。

示例：
  python3 -m orchestrator_service.examples.uart_protocol_smoke_test --port /dev/ttyHS1 --baud 115200
  python3 -m orchestrator_service.examples.uart_protocol_smoke_test --dry-run
"""

import argparse
import sys
import time

try:
    import serial  # type: ignore
except Exception:
    serial = None


def send_line(ser, line: str, dry_run: bool):
    shown = line.rstrip("\n")
    print(f"[TX] {shown}")
    if dry_run:
        return
    ser.write(line.encode("utf-8"))
    ser.flush()


def maybe_read(ser, dry_run: bool, wait_s: float = 0.25):
    if dry_run or ser is None:
        return
    deadline = time.time() + wait_s
    while time.time() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        try:
            line = raw.decode("utf-8", errors="ignore").strip()
        except Exception:
            continue
        if line:
            print(f"[RX] {line}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyHS1")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=0.1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sleep", type=float, default=2.0, help="每段动作保持时间")
    args = ap.parse_args()

    if not args.dry_run and serial is None:
        print("pyserial 未安装", file=sys.stderr)
        sys.exit(2)

    ser = None
    if not args.dry_run:
        ser = serial.Serial(args.port, args.baud, timeout=args.timeout)
        print(f"[INFO] opened {args.port} @ {args.baud}")

    seq = [
        ("MODE AUTOEXPLORE\n", args.sleep),
        ("MODE AUTOSEARCH\n", args.sleep),
        ("MODE SEARCH\n", 0.1),
        ("V 0.350 0.000\n", args.sleep),
        ("V 0.000 0.650\n", args.sleep),
        ("MODE RETURN\n", 0.1),
        ("V 0.280 0.000\n", args.sleep),
        ("V 0.000 -0.550\n", args.sleep),
        ("MODE STOP\n", 0.05),
        ("STOP\n", 0.5),
    ]

    try:
        for line, hold in seq:
            send_line(ser, line, args.dry_run)
            maybe_read(ser, args.dry_run, wait_s=0.25)
            time.sleep(max(0.0, hold))
        print("[INFO] done")
    finally:
        if ser is not None:
            ser.close()
            print("[INFO] serial closed")


if __name__ == "__main__":
    main()
