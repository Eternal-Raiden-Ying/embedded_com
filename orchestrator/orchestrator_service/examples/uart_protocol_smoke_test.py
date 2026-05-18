#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    ap = argparse.ArgumentParser(description="直接验证底盘 TXT 串口协议是否接受 MODE/V/STOP")
    ap.add_argument("--port", default="/dev/ttyHS1")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=0.1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sleep", type=float, default=1.2, help="每段动作保持时长")
    args = ap.parse_args()

    if not args.dry_run and serial is None:
        print("pyserial 未安装", file=sys.stderr)
        sys.exit(2)

    ser = None
    if not args.dry_run:
        ser = serial.Serial(args.port, args.baud, timeout=args.timeout, write_timeout=args.timeout)
        print(f"[INFO] opened {args.port} @ {args.baud}")

    seq = [
        ("MODE SEARCH\r\n", 0.05),
        ("V 0.000 0.000 0.220\r\n", args.sleep),
        ("V 0.000 0.000 -0.180\r\n", args.sleep),
        ("V 0.120 0.040 -0.080\r\n", args.sleep),
        ("V 0.000 0.140 0.000\r\n", args.sleep),
        ("MODE RETURN\r\n", 0.05),
        ("V 0.100 0.000 0.000\r\n", args.sleep),
        ("STOP\r\n", 0.4),
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
