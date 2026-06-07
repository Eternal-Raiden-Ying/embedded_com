#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open-loop serial command generator for STM32 base dead-zone and speed mapping.

Default is dry-run and writes the exact MODE/VEL/STOP lines to a jsonl file.
Use --real only when the car is lifted or placed in a safe low-speed test area.

Typical dry run:
  python3 tools/serial_open_loop_test.py --out-dir /tmp/open_loop

Real serial test:
  python3 tools/serial_open_loop_test.py --real --port /dev/ttyHS1 --baud 115200 --phase-s 2.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from common_offline import write_json, write_jsonl


def build_default_profile() -> List[Tuple[str, float, float, float, float]]:
    profile: List[Tuple[str, float, float, float, float]] = []
    for v in [0.030, 0.050, 0.080, 0.100, 0.120]:
        profile.append((f"VX_{v:.3f}", v, 0.0, 0.0, 2.0))
    for w in [0.040, 0.060, 0.080, 0.120, 0.160]:
        profile.append((f"WZ_POS_{w:.3f}", 0.0, 0.0, w, 2.0))
        profile.append((f"WZ_NEG_{w:.3f}", 0.0, 0.0, -w, 2.0))
    return profile


def parse_profile(s: str) -> List[Tuple[str, float, float, float, float]]:
    if not s:
        return build_default_profile()
    out: List[Tuple[str, float, float, float, float]] = []
    # Format: name:vx,vy,wz,dur;name:vx,vy,wz,dur
    for item in s.split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, vals = item.split(":", 1)
        else:
            name, vals = f"STEP_{len(out)}", item
        parts = [float(x.strip()) for x in vals.split(",")]
        if len(parts) != 4:
            raise ValueError(f"bad profile item {item!r}; expected vx,vy,wz,duration")
        out.append((name.strip(), parts[0], parts[1], parts[2], parts[3]))
    return out


def fmt(v: float, digits: int = 3) -> str:
    return f"{float(v):.{int(digits)}f}"


def make_lines(mode: str, vx: float, vy: float, wz: float, hold_ms: int, digits: int) -> List[str]:
    return [f"MODE {mode}\n", f"VEL {fmt(vx, digits)} {fmt(vy, digits)} {fmt(wz, digits)} {int(hold_ms)}\n"]


def open_serial(args: argparse.Namespace):
    if not args.real:
        return None
    try:
        import serial  # type: ignore
    except Exception as exc:
        raise SystemExit(f"pyserial is required for --real: {exc}")
    return serial.Serial(args.port, args.baud, timeout=args.timeout)


def tx(ser, line: str, real: bool) -> None:
    if real:
        ser.write(line.encode("ascii", errors="ignore"))
        ser.flush()
    print("[TX]", line.rstrip())


def main() -> int:
    ap = argparse.ArgumentParser(description="Open-loop base speed/dead-zone test command generator")
    ap.add_argument("--real", action="store_true", help="Actually send to serial. Default is dry-run.")
    ap.add_argument("--port", default="/dev/ttyHS1")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=0.10)
    ap.add_argument("--hold-ms", type=int, default=180)
    ap.add_argument("--digits", type=int, default=3)
    ap.add_argument("--phase-s", type=float, default=0.0, help="Override duration of every profile step if >0")
    ap.add_argument("--between-stop-s", type=float, default=0.60)
    ap.add_argument("--profile", default="", help="name:vx,vy,wz,dur;... If empty, use default VX/WZ sweep")
    ap.add_argument("--out-dir", default="runs/open_loop_test")
    args = ap.parse_args()

    profile = parse_profile(args.profile)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    ser = open_serial(args)
    try:
        t0 = time.time()
        for idx, (name, vx, vy, wz, dur) in enumerate(profile):
            dur = float(args.phase_s) if args.phase_s > 0 else float(dur)
            mode = f"OPEN_LOOP_{idx:02d}"
            for line in make_lines(mode, vx, vy, wz, args.hold_ms, args.digits):
                tx(ser, line, args.real)
                rows.append({"ts": time.time(), "t_s": time.time() - t0, "idx": idx, "name": name, "line": line.rstrip(), "real": args.real})
                time.sleep(0.05)
            if args.real:
                time.sleep(dur)
            else:
                # Do not make dry runs unnecessarily slow.
                time.sleep(min(0.10, dur))
            tx(ser, "STOP\n", args.real)
            rows.append({"ts": time.time(), "t_s": time.time() - t0, "idx": idx, "name": name, "line": "STOP", "real": args.real})
            if args.real:
                time.sleep(max(0.0, args.between_stop_s))
            else:
                time.sleep(min(0.05, max(0.0, args.between_stop_s)))
    finally:
        if ser is not None:
            try:
                tx(ser, "STOP\n", args.real)
                ser.close()
            except Exception:
                pass

    summary = {
        "real": args.real,
        "port": args.port,
        "baud": args.baud,
        "steps": len(profile),
        "profile": [
            {"idx": i, "name": name, "vx": vx, "vy": vy, "wz": wz, "duration_s": (args.phase_s if args.phase_s > 0 else dur)}
            for i, (name, vx, vy, wz, dur) in enumerate(profile)
        ],
        "notes": [
            "真实测试时记录每个速度档是否起步、是否抖动、是否明显延迟。",
            "用该结果反推 ORCH_DOCKING_*_MAX_* 和 PID min_abs_output。",
        ],
    }
    write_jsonl(out_dir / "open_loop_cmds.jsonl", rows)
    write_json(out_dir / "open_loop_summary.json", summary)
    print(f"[OK] wrote open-loop command log to {out_dir}")
    if not args.real:
        print("[DRY-RUN] no serial data was sent. Add --real only in a safe test setup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
