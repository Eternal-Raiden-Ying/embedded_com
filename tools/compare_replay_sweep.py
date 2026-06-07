#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter sweep wrapper for replay_table_edge_log.py.

This lets you replay the same table_edge_obs log with several filter_alpha /
coarse hysteresis / speed settings and rank candidates by jitter risk and final
state.  It is designed for fast parameter selection before doing a real car run.

Example:
  python3 tools/compare_replay_sweep.py --input table_edge_obs.jsonl \
      --alphas 0.25,0.35,0.45 --approach-vxs 0.08,0.10,0.12
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from common_offline import read_jsonl, write_csv, write_json
from replay_table_edge_log import run_replay


def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep replay parameters and rank candidates")
    ap.add_argument("--input", default="", help="table_edge_obs.jsonl")
    ap.add_argument("--run-dir", default="", help="Run directory if --input is omitted")
    ap.add_argument("--out-dir", default="runs/offline_replay_sweep")
    ap.add_argument("--mode", choices=["state", "coarse", "controlled", "final"], default="state")
    ap.add_argument("--alphas", default="0.25,0.35,0.45")
    ap.add_argument("--approach-vxs", default="0.08,0.10,0.12")
    ap.add_argument("--approach-wzs", default="0.12,0.16")
    ap.add_argument("--coarse-enters", default="0.14,0.16")
    ap.add_argument("--coarse-exits", default="0.07,0.08")
    ap.add_argument("--max-cases", type=int, default=200)
    ap.add_argument("--default-rate", type=float, default=5.0)
    ap.add_argument("--start-time", type=float, default=1000.0)
    ap.add_argument("--ignore-obs-ts", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    combos = list(itertools.product(
        parse_floats(args.alphas),
        parse_floats(args.approach_vxs),
        parse_floats(args.approach_wzs),
        parse_floats(args.coarse_enters),
        parse_floats(args.coarse_exits),
    ))[: max(1, args.max_cases)]

    for case_idx, (alpha, avx, awz, enter, exit_) in enumerate(combos):
        if enter <= exit_:
            continue
        case_dir = out_root / f"case_{case_idx:03d}_a{alpha:.2f}_vx{avx:.2f}_wz{awz:.2f}_h{enter:.2f}_{exit_:.2f}"
        ns = argparse.Namespace(**vars(args))
        ns.out_dir = str(case_dir)
        ns.filter_alpha = alpha
        ns.filter_window = None
        ns.coarse_enter = enter
        ns.coarse_exit = exit_
        ns.approach_vx = avx
        ns.approach_wz = awz
        ns.final_vx = None
        ns.final_wz = None
        ns.vx_slew = None
        ns.wz_slew = None
        ns.min_confidence = None
        ns.max_rows = 0
        summary = run_replay(ns)
        pred = summary["jitter_prediction"]
        rows.append({
            "case_idx": case_idx,
            "filter_alpha": alpha,
            "approach_vx": avx,
            "approach_wz": awz,
            "coarse_enter": enter,
            "coarse_exit": exit_,
            "arrived": summary.get("arrived"),
            "final_state": summary.get("final_state"),
            "risk_score": pred.get("risk_score"),
            "risk_level": pred.get("risk_level"),
            "yaw_jitter_term": pred.get("terms", {}).get("yaw_jitter"),
            "wz_flip_term": pred.get("terms", {}).get("wz_flip"),
            "wz_rate_term": pred.get("terms", {}).get("wz_rate"),
            "case_dir": str(case_dir),
        })
        print(f"[{case_idx+1}/{len(combos)}] risk={pred.get('risk_score')} arrived={summary.get('arrived')} {case_dir.name}")

    rows.sort(key=lambda r: (not bool(r.get("arrived")), float(r.get("risk_score") or 999.0)))
    write_csv(out_root / "sweep_results.csv", rows)
    write_json(out_root / "sweep_results.json", {"cases": rows, "best": rows[:5]})
    print(f"[OK] sweep done -> {out_root / 'sweep_results.csv'}")
    if rows:
        b = rows[0]
        print(f"best: risk={b['risk_score']} arrived={b['arrived']} alpha={b['filter_alpha']} vx={b['approach_vx']} wz={b['approach_wz']} h=({b['coarse_enter']},{b['coarse_exit']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
