#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic offline tests for ctrl_code/control/docking_controller.py.

Typical use:
  python3 tools/sim_docking_controller.py --scenario all --out-dir /tmp/dock_sim
  python3 tools/sim_docking_controller.py --scenario yaw_jitter --filter-alpha 0.25

Outputs:
  sim_<scenario>.jsonl       per-frame observation + DockingController command
  sim_<scenario>_summary.json
  sim_all_summary.json       when --scenario all
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from common_offline import (
    add_common_docking_args,
    apply_common_docking_overrides,
    basic_jitter_prediction,
    command_to_row,
    make_docking_cfg_from_env,
    write_csv,
    write_json,
    write_jsonl,
)

from ctrl_code.control.docking_controller import DockingController
from ctrl_code.control.types import EdgeControlObservation


def _noise(rng: random.Random, amp: float) -> float:
    return rng.uniform(-amp, amp)


def gen_clean_approach(duration_s: float, rate_hz: float, rng: random.Random) -> Iterable[Dict]:
    n = int(duration_s * rate_hz)
    for i in range(n):
        p = i / max(1, n - 1)
        yield {
            "ts": i / rate_hz,
            "edge_found": True,
            "confidence": 0.78 + _noise(rng, 0.03),
            "yaw_err_rad": 0.24 * (1 - p) + 0.015 * math.sin(8 * p) + _noise(rng, 0.005),
            "dist_err_m": 0.22 * (1 - p) + 0.010 + _noise(rng, 0.004),
            "lateral_err_m": 0.04 * math.sin(2.5 * p) + _noise(rng, 0.003),
            "edge_ready": p > 0.72,
            "depth_valid": True,
            "source": "synthetic_clean_approach",
        }


def gen_yaw_jitter(duration_s: float, rate_hz: float, rng: random.Random) -> Iterable[Dict]:
    n = int(duration_s * rate_hz)
    for i in range(n):
        p = i / max(1, n - 1)
        base = 0.13 * (1 - p) + 0.035
        yield {
            "ts": i / rate_hz,
            "edge_found": True,
            "confidence": 0.70 + _noise(rng, 0.08),
            "yaw_err_rad": base + 0.04 * math.sin(i * 1.7) + _noise(rng, 0.018),
            "dist_err_m": 0.12 * (1 - p) + 0.025 + _noise(rng, 0.006),
            "lateral_err_m": _noise(rng, 0.012),
            "edge_ready": p > 0.70,
            "depth_valid": True,
            "source": "synthetic_yaw_jitter",
        }


def gen_dist_jitter(duration_s: float, rate_hz: float, rng: random.Random) -> Iterable[Dict]:
    n = int(duration_s * rate_hz)
    for i in range(n):
        p = i / max(1, n - 1)
        yield {
            "ts": i / rate_hz,
            "edge_found": True,
            "confidence": 0.72 + _noise(rng, 0.05),
            "yaw_err_rad": 0.10 * (1 - p) + 0.02 + _noise(rng, 0.008),
            "dist_err_m": 0.08 * (1 - p) + 0.04 + 0.035 * math.sin(i * 1.3) + _noise(rng, 0.012),
            "lateral_err_m": _noise(rng, 0.008),
            "edge_ready": False,
            "depth_valid": True,
            "source": "synthetic_dist_jitter",
        }


def gen_jump(duration_s: float, rate_hz: float, rng: random.Random) -> Iterable[Dict]:
    n = int(duration_s * rate_hz)
    for i in range(n):
        p = i / max(1, n - 1)
        yaw = 0.16 * (1 - p) + 0.03 + _noise(rng, 0.006)
        dist = 0.16 * (1 - p) + 0.025 + _noise(rng, 0.005)
        if i in {int(n * 0.30), int(n * 0.55), int(n * 0.72)}:
            yaw += rng.choice([-1.0, 1.0]) * 0.25
            dist += rng.choice([-1.0, 1.0]) * 0.12
        yield {
            "ts": i / rate_hz,
            "edge_found": True,
            "confidence": 0.74 + _noise(rng, 0.04),
            "yaw_err_rad": yaw,
            "dist_err_m": dist,
            "lateral_err_m": _noise(rng, 0.01),
            "edge_ready": p > 0.75,
            "depth_valid": True,
            "source": "synthetic_jump",
        }


def gen_dropout(duration_s: float, rate_hz: float, rng: random.Random) -> Iterable[Dict]:
    n = int(duration_s * rate_hz)
    for i in range(n):
        p = i / max(1, n - 1)
        bad = (i % 11 == 0 and i > 0) or (int(n * 0.45) <= i <= int(n * 0.45) + 2)
        yield {
            "ts": i / rate_hz,
            "edge_found": not bad,
            "confidence": 0.20 if bad else 0.75 + _noise(rng, 0.04),
            "yaw_err_rad": None if bad else 0.18 * (1 - p) + 0.025 + _noise(rng, 0.008),
            "dist_err_m": None if bad else 0.18 * (1 - p) + 0.030 + _noise(rng, 0.007),
            "lateral_err_m": None if bad else _noise(rng, 0.01),
            "edge_ready": False,
            "depth_valid": False if bad else True,
            "source": "synthetic_dropout",
        }


def gen_stale(duration_s: float, rate_hz: float, rng: random.Random) -> Iterable[Dict]:
    n = int(duration_s * rate_hz)
    stale_start = int(n * 0.42)
    stale_end = stale_start + int(1.0 * rate_hz)
    held_ts: Optional[float] = None
    for i in range(n):
        p = i / max(1, n - 1)
        ts = i / rate_hz
        if i == stale_start:
            held_ts = ts
        obs_ts = held_ts if held_ts is not None and stale_start <= i <= stale_end else ts
        yield {
            "ts": obs_ts,
            "edge_found": True,
            "confidence": 0.76 + _noise(rng, 0.04),
            "yaw_err_rad": 0.18 * (1 - p) + 0.020 + _noise(rng, 0.006),
            "dist_err_m": 0.15 * (1 - p) + 0.030 + _noise(rng, 0.006),
            "lateral_err_m": _noise(rng, 0.01),
            "edge_ready": p > 0.80,
            "depth_valid": True,
            "source": "synthetic_stale",
        }


SCENARIOS = {
    "clean_approach": gen_clean_approach,
    "yaw_jitter": gen_yaw_jitter,
    "dist_jitter": gen_dist_jitter,
    "jump": gen_jump,
    "dropout": gen_dropout,
    "stale": gen_stale,
}


def obs_from_dict(d: Mapping[str, object]) -> Optional[EdgeControlObservation]:
    try:
        return EdgeControlObservation(
            ts=float(d.get("ts", 0.0) or 0.0),
            edge_found=bool(d.get("edge_found", False)),
            confidence=float(d.get("confidence", 0.0) or 0.0),
            yaw_err_rad=None if d.get("yaw_err_rad") is None else float(d.get("yaw_err_rad")),
            dist_err_m=None if d.get("dist_err_m") is None else float(d.get("dist_err_m")),
            lateral_err_m=None if d.get("lateral_err_m") is None else float(d.get("lateral_err_m")),
            edge_ready=bool(d.get("edge_ready", False)),
            depth_valid=d.get("depth_valid", None),
            source=str(d.get("source", "synthetic")),
        )
    except Exception:
        return None


def run_scenario(name: str, args: argparse.Namespace) -> Dict[str, object]:
    rng = random.Random(args.seed)
    cfg = apply_common_docking_overrides(make_docking_cfg_from_env(), args)
    ctrl = DockingController(cfg)
    gen = SCENARIOS[name]
    obs_rows = list(gen(args.duration, args.rate, rng))
    out_rows: List[Dict[str, object]] = []
    now0 = float(args.start_time)
    mode = args.mode.upper()

    for idx, obs_row0 in enumerate(obs_rows):
        now = now0 + idx / float(args.rate)
        # Scenario generators use relative observation timestamps.  Offset them
        # to the controller clock so normal synthetic runs are not accidentally
        # classified as stale.  The stale scenario intentionally holds its
        # relative ts constant for several frames, which is preserved here.
        obs_row = dict(obs_row0)
        obs_row["replay_t_s"] = float(obs_row0.get("ts", idx / float(args.rate)) or 0.0)
        obs_row["ts"] = now0 + float(obs_row["replay_t_s"])
        obs = obs_from_dict(obs_row)
        cmd = ctrl.update(mode, obs, now_s=now)
        row = command_to_row(cmd, ts=now - now0, idx=idx, state=mode, obs=obs_row)
        row["scenario"] = name
        out_rows.append(row)

    pred = basic_jitter_prediction(out_rows, source=f"synthetic:{name}")
    summary = {
        "scenario": name,
        "mode": mode,
        "duration_s": args.duration,
        "rate_hz": args.rate,
        "rows": len(out_rows),
        "config": {
            "filter_alpha": cfg.filter_ewma_alpha,
            "filter_window": cfg.filter_window,
            "coarse_enter_rad": cfg.coarse_align_enter_rad,
            "coarse_exit_rad": cfg.coarse_align_exit_rad,
            "approach_max_vx_norm": cfg.approach_max_vx_norm,
            "approach_max_wz_norm": cfg.approach_max_wz_norm,
            "final_max_vx_norm": cfg.final_max_vx_norm,
            "final_max_wz_norm": cfg.final_max_wz_norm,
            "vx_slew_per_s": cfg.vx_slew_per_s,
            "wz_slew_per_s": cfg.wz_slew_per_s,
            "min_confidence": cfg.min_confidence,
        },
        "phase_counts": {p: sum(1 for r in out_rows if r.get("phase") == p) for p in sorted({str(r.get("phase", "")) for r in out_rows})},
        "valid_rate": sum(1 for r in out_rows if r.get("valid")) / max(1, len(out_rows)),
        "jitter_prediction": pred,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / f"sim_{name}.jsonl", out_rows)
    write_csv(out_dir / f"sim_{name}.csv", out_rows)
    write_json(out_dir / f"sim_{name}_summary.json", summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Synthetic tests for DockingController")
    ap.add_argument("--scenario", choices=["all"] + sorted(SCENARIOS), default="all")
    ap.add_argument("--mode", choices=["COARSE_ALIGN", "CONTROLLED_APPROACH", "FINAL_LOCK"], default="CONTROLLED_APPROACH")
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--rate", type=float, default=5.0)
    ap.add_argument("--start-time", type=float, default=1000.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default="runs/offline_sim")
    add_common_docking_args(ap)
    args = ap.parse_args()

    names = sorted(SCENARIOS) if args.scenario == "all" else [args.scenario]
    summaries = [run_scenario(name, args) for name in names]
    out_dir = Path(args.out_dir)
    write_json(out_dir / "sim_all_summary.json", {"generated_at": time.time(), "summaries": summaries})

    print(f"[OK] wrote synthetic docking simulations to {out_dir}")
    for s in summaries:
        pred = s["jitter_prediction"]
        print(f"  - {s['scenario']}: risk={pred['risk_score']} {pred['risk_level']} phase={s['phase_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
