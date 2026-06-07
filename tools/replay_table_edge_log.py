#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replay real table_edge_obs logs through an offline docking-state subset and the
real DockingController implementation.

Typical use:
  python3 tools/replay_table_edge_log.py --run-dir runs/2026_xxx --out-dir /tmp/replay
  python3 tools/replay_table_edge_log.py --input table_edge_obs.jsonl --mode controlled --filter-alpha 0.25

Outputs:
  replay_cmd_vel.jsonl       simulated velocity commands per observation frame
  replay_state_trace.jsonl   state transitions, counters, raw/filtered errors
  replay_summary.json        lock result, transition counts, jitter prediction
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from common_offline import (
    add_common_docking_args,
    apply_common_docking_overrides,
    basic_jitter_prediction,
    command_to_row,
    extract_table_obs,
    find_log_file,
    get_path,
    make_docking_cfg_from_env,
    normalized_time,
    read_jsonl,
    to_bool,
    to_float,
    write_csv,
    write_json,
    write_jsonl,
)

from ctrl_code.control.docking_controller import DockingController
from ctrl_code.control.types import EdgeControlObservation


@dataclass
class ReplayThresholds:
    min_confidence: float = 0.55
    table_found_frames_to_approach: int = 2
    table_lost_frames_to_reacquire: int = 4
    table_loss_hold_s: float = 1.20
    approach_timeout_s: float = 14.0
    approach_min_dwell_s: float = 0.80
    coarse_align_min_dwell_s: float = 0.50
    coarse_align_frames_to_advance: int = 3
    coarse_align_done_rad: float = 0.08
    edge_ready_frames_to_final_lock: int = 2
    edge_ready_yaw_tol_rad: float = 0.12
    edge_ready_dist_tol_m: float = 0.06
    final_lock_frames_to_arrive: int = 3
    final_lock_yaw_tol_rad: float = 0.04
    final_lock_dist_tol_m: float = 0.03
    final_lock_lateral_tol_m: float = 0.03


class TableDockingReplayState:
    """Small offline subset of the real docking states.

    It mirrors only the table-docking path. It is deliberately deterministic so
    that one table_edge_obs.jsonl can be replayed repeatedly while sweeping
    control/filter parameters.
    """

    SEARCH_TABLE = "SEARCH_TABLE"
    COARSE_ALIGN = "COARSE_ALIGN"
    CONTROLLED_APPROACH = "CONTROLLED_APPROACH"
    FINAL_LOCK = "FINAL_LOCK"
    AT_TABLE_EDGE = "AT_TABLE_EDGE"
    DOCK_RETRY = "DOCK_RETRY"

    def __init__(self, th: ReplayThresholds):
        self.th = th
        self.state = self.SEARCH_TABLE
        self.enter_ts = 0.0
        self.table_found_frames = 0
        self.table_lost_frames = 0
        self.approach_aligned_frames = 0
        self.approach_ready_frames = 0
        self.table_lock_frames = 0
        self.transitions: List[Dict] = []

    @staticmethod
    def _obs_ok(obs: Mapping[str, object], min_conf: float) -> bool:
        if not to_bool(obs.get("edge_found"), False):
            return False
        if obs.get("depth_valid") is False:
            return False
        if float(obs.get("confidence") or 0.0) < min_conf:
            return False
        if obs.get("yaw_err_rad") is None or obs.get("dist_err_m") is None:
            return False
        return True

    def _set_state(self, new_state: str, now_s: float, reason: str) -> None:
        if new_state == self.state:
            return
        self.transitions.append({
            "ts": now_s,
            "from": self.state,
            "to": new_state,
            "reason": reason,
        })
        self.state = new_state
        self.enter_ts = now_s
        self.table_found_frames = 0
        self.table_lost_frames = 0
        self.approach_aligned_frames = 0
        self.approach_ready_frames = 0
        self.table_lock_frames = 0

    def _lost_update(self, obs_ok: bool, now_s: float) -> bool:
        if obs_ok:
            self.table_lost_frames = 0
            return False
        self.table_lost_frames += 1
        if self.table_lost_frames >= self.th.table_lost_frames_to_reacquire:
            self._set_state(self.SEARCH_TABLE, now_s, "edge_lost_frames")
            return True
        return False

    def _edge_ready(self, obs: Mapping[str, object]) -> bool:
        if to_bool(obs.get("edge_ready"), False):
            # Still require basic geometry; a stale/over-optimistic ready flag
            # should not force FINAL_LOCK alone.
            pass
        yaw = abs(float(obs.get("yaw_err_rad") or 0.0))
        dist = abs(float(obs.get("dist_err_m") or 0.0))
        return yaw <= self.th.edge_ready_yaw_tol_rad and dist <= self.th.edge_ready_dist_tol_m

    def step(self, obs: Mapping[str, object], now_s: float) -> Tuple[str, str]:
        reason = "stay"
        ok = self._obs_ok(obs, self.th.min_confidence)
        dwell = now_s - self.enter_ts

        if self.state == self.SEARCH_TABLE:
            if ok:
                self.table_found_frames += 1
                if self.table_found_frames >= self.th.table_found_frames_to_approach:
                    self._set_state(self.COARSE_ALIGN, now_s, "table_found_frames")
                    reason = "table_found_frames"
            else:
                self.table_found_frames = 0
            return self.state, reason

        if self.state == self.COARSE_ALIGN:
            if self._lost_update(ok, now_s):
                return self.state, "edge_lost"
            yaw = abs(float(obs.get("yaw_err_rad") or 0.0))
            if dwell >= self.th.coarse_align_min_dwell_s and yaw <= self.th.coarse_align_done_rad:
                self.approach_aligned_frames += 1
                if self.approach_aligned_frames >= self.th.coarse_align_frames_to_advance:
                    self._set_state(self.CONTROLLED_APPROACH, now_s, "coarse_align_done")
                    reason = "coarse_align_done"
            else:
                self.approach_aligned_frames = 0
            if dwell > self.th.approach_timeout_s:
                self._set_state(self.DOCK_RETRY, now_s, "coarse_align_timeout")
                reason = "coarse_align_timeout"
            return self.state, reason

        if self.state == self.CONTROLLED_APPROACH:
            if self._lost_update(ok, now_s):
                return self.state, "edge_lost"
            if dwell >= self.th.approach_min_dwell_s and self._edge_ready(obs):
                self.approach_ready_frames += 1
                if self.approach_ready_frames >= self.th.edge_ready_frames_to_final_lock:
                    self._set_state(self.FINAL_LOCK, now_s, "edge_ready")
                    reason = "edge_ready"
            else:
                self.approach_ready_frames = 0
            if dwell > self.th.approach_timeout_s:
                self._set_state(self.DOCK_RETRY, now_s, "approach_timeout")
                reason = "approach_timeout"
            return self.state, reason

        if self.state == self.FINAL_LOCK:
            if self._lost_update(ok, now_s):
                return self.state, "edge_lost"
            yaw_ok = abs(float(obs.get("yaw_err_rad") or 0.0)) <= self.th.final_lock_yaw_tol_rad
            dist_ok = abs(float(obs.get("dist_err_m") or 0.0)) <= self.th.final_lock_dist_tol_m
            lat = obs.get("lateral_err_m")
            lat_ok = True if lat is None else abs(float(lat)) <= self.th.final_lock_lateral_tol_m
            if yaw_ok and dist_ok and lat_ok:
                self.table_lock_frames += 1
                if self.table_lock_frames >= self.th.final_lock_frames_to_arrive:
                    self._set_state(self.AT_TABLE_EDGE, now_s, "final_lock_arrive")
                    reason = "final_lock_arrive"
            else:
                self.table_lock_frames = 0
            if dwell > self.th.approach_timeout_s:
                self._set_state(self.DOCK_RETRY, now_s, "final_lock_timeout")
                reason = "final_lock_timeout"
            return self.state, reason

        if self.state == self.DOCK_RETRY:
            self._set_state(self.SEARCH_TABLE, now_s, "replay_retry_reset")
            return self.state, "replay_retry_reset"

        return self.state, reason

    def snapshot(self) -> Dict[str, object]:
        return {
            "state": self.state,
            "table_found_frames": self.table_found_frames,
            "table_lost_frames": self.table_lost_frames,
            "approach_aligned_frames": self.approach_aligned_frames,
            "approach_ready_frames": self.approach_ready_frames,
            "table_lock_frames": self.table_lock_frames,
        }


def obs_to_edge(obs: Mapping[str, object], now_s: float, use_replay_ts_as_obs_ts: bool = True) -> EdgeControlObservation:
    ts = float(obs.get("ts") or now_s)
    if not use_replay_ts_as_obs_ts:
        ts = now_s
    return EdgeControlObservation(
        ts=ts,
        edge_found=to_bool(obs.get("edge_found"), False),
        confidence=float(obs.get("confidence") or 0.0),
        yaw_err_rad=None if obs.get("yaw_err_rad") is None else float(obs.get("yaw_err_rad")),
        dist_err_m=None if obs.get("dist_err_m") is None else float(obs.get("dist_err_m")),
        lateral_err_m=None if obs.get("lateral_err_m") is None else float(obs.get("lateral_err_m")),
        edge_ready=to_bool(obs.get("edge_ready"), False),
        depth_valid=obs.get("depth_valid", None),
        source=str(obs.get("source", "replay")),
    )


def load_table_rows(args: argparse.Namespace) -> Tuple[Path, List[Dict[str, object]]]:
    if args.input:
        path = Path(args.input)
    else:
        if not args.run_dir:
            raise SystemExit("need --input or --run-dir")
        path = find_log_file(args.run_dir, "table_edge_obs")
        if path is None:
            raise SystemExit(f"cannot find table_edge_obs jsonl under {args.run_dir}")
    raw_rows = read_jsonl(path)
    rows = [extract_table_obs(r) for r in raw_rows]
    # Use relative monotonic time for replay while preserving the original timestamp.
    rel_t = normalized_time(rows, rate_hz=args.default_rate)
    for r, t in zip(rows, rel_t):
        r["orig_ts"] = r.get("ts")
        r["ts"] = float(args.start_time) + float(t)
        r["replay_t_s"] = float(t)
    return path, rows


def run_replay(args: argparse.Namespace) -> Dict[str, object]:
    input_path, rows = load_table_rows(args)
    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]

    cfg = apply_common_docking_overrides(make_docking_cfg_from_env(), args)
    ctrl = DockingController(cfg)
    th = ReplayThresholds(min_confidence=cfg.min_confidence)
    sm = TableDockingReplayState(th)
    sm.enter_ts = float(args.start_time)

    fixed_mode = None
    if args.mode == "coarse":
        fixed_mode = "COARSE_ALIGN"
    elif args.mode == "controlled":
        fixed_mode = "CONTROLLED_APPROACH"
    elif args.mode == "final":
        fixed_mode = "FINAL_LOCK"

    cmd_rows: List[Dict[str, object]] = []
    trace_rows: List[Dict[str, object]] = []

    for idx, obs in enumerate(rows):
        now_s = float(obs.get("ts") or (args.start_time + idx / args.default_rate))
        if fixed_mode is not None:
            state = fixed_mode
            reason = "fixed_mode"
        else:
            state, reason = sm.step(obs, now_s)

        if state in {"COARSE_ALIGN", "CONTROLLED_APPROACH", "FINAL_LOCK"}:
            cmd = ctrl.update(state, obs_to_edge(obs, now_s, use_replay_ts_as_obs_ts=not args.ignore_obs_ts), now_s=now_s)
            cmd_row = command_to_row(cmd, ts=float(obs.get("replay_t_s", now_s - args.start_time)), idx=idx, state=state, obs=obs)
        else:
            # SEARCH/AT_TABLE are represented as zero commands in replay.  Real
            # search turn is handled in runtime/controller.py, but the focus here
            # is docking control stability after edge observation exists.
            from ctrl_code.control.types import DockingCommand
            cmd = DockingCommand(vx=0.0, vy=0.0, wz=0.0, valid=True, mode=state, reason=reason, phase=state)
            cmd_row = command_to_row(cmd, ts=float(obs.get("replay_t_s", now_s - args.start_time)), idx=idx, state=state, obs=obs)

        cmd_row["transition_reason"] = reason
        cmd_rows.append(cmd_row)
        snap = sm.snapshot() if fixed_mode is None else {}
        trace = dict(cmd_row)
        trace.update({f"counter_{k}": v for k, v in snap.items() if k != "state"})
        trace_rows.append(trace)

    pred = basic_jitter_prediction(cmd_rows, source=str(input_path))
    final_state = sm.state if fixed_mode is None else fixed_mode
    summary = {
        "input": str(input_path),
        "mode": args.mode,
        "rows": len(rows),
        "final_state": final_state,
        "arrived": final_state == "AT_TABLE_EDGE" or any(r.get("state") == "AT_TABLE_EDGE" for r in trace_rows),
        "transitions": sm.transitions if fixed_mode is None else [],
        "state_counts": {s: sum(1 for r in trace_rows if r.get("state") == s) for s in sorted({str(r.get("state")) for r in trace_rows})},
        "phase_counts": {s: sum(1 for r in trace_rows if r.get("phase") == s) for s in sorted({str(r.get("phase")) for r in trace_rows})},
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
        "jitter_prediction": pred,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "replay_cmd_vel.jsonl", cmd_rows)
    write_jsonl(out_dir / "replay_state_trace.jsonl", trace_rows)
    write_csv(out_dir / "replay_cmd_vel.csv", cmd_rows)
    write_json(out_dir / "replay_summary.json", summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay table_edge_obs logs through offline docking control")
    ap.add_argument("--input", default="", help="Path to table_edge_obs.jsonl")
    ap.add_argument("--run-dir", default="", help="Run directory containing table_edge_obs.jsonl")
    ap.add_argument("--out-dir", default="runs/offline_replay")
    ap.add_argument("--mode", choices=["state", "coarse", "controlled", "final"], default="state",
                    help="state=offline docking state subset; otherwise fixed controller mode")
    ap.add_argument("--default-rate", type=float, default=5.0, help="Rate used when log timestamps are missing")
    ap.add_argument("--start-time", type=float, default=1000.0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--ignore-obs-ts", action="store_true", help="Treat obs timestamp as current replay time to avoid stale testing")
    add_common_docking_args(ap)
    args = ap.parse_args()

    summary = run_replay(args)
    print(f"[OK] replay wrote files to {args.out_dir}")
    print(f"  input={summary['input']}")
    print(f"  final_state={summary['final_state']} arrived={summary['arrived']}")
    pred = summary["jitter_prediction"]
    print(f"  jitter_risk={pred['risk_score']} {pred['risk_level']}")
    for w in pred.get("warnings", []):
        print(f"  warning: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
