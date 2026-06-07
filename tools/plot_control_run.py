#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot and summarize real/replay control runs.

Typical use:
  python3 tools/plot_control_run.py --run-dir runs/2026_xxx --out-dir /tmp/plot
  python3 tools/plot_control_run.py --table table_edge_obs.jsonl --cmd replay_cmd_vel.jsonl

Outputs:
  merged_timeline.csv/jsonl
  obs_timeseries.png          raw/filtered yaw/dist + confidence
  cmd_timeseries.png          vx/vy/wz commands
  state_timeline.png          state/phase as a timeline
  plot_summary.json           signal statistics + jitter prediction
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from common_offline import (
    basic_jitter_prediction,
    extract_table_obs,
    find_log_file,
    get_path,
    normalized_time,
    read_jsonl,
    summarize_signal,
    to_float,
    write_csv,
    write_json,
    write_jsonl,
)


def load_file(path: str) -> List[Dict]:
    if not path:
        return []
    return read_jsonl(path)


def load_inputs(args: argparse.Namespace) -> Dict[str, List[Dict]]:
    files: Dict[str, Optional[Path]] = {}
    if args.run_dir:
        root = Path(args.run_dir)
        files["table"] = Path(args.table) if args.table else find_log_file(root, "table_edge_obs")
        files["cmd"] = Path(args.cmd) if args.cmd else (find_log_file(root, "replay_cmd_vel") or find_log_file(root, "cmd_vel"))
        files["car"] = Path(args.car) if args.car else find_log_file(root, "car_cmd")
        files["state"] = Path(args.state) if args.state else (find_log_file(root, "replay_state_trace") or find_log_file(root, "state_block"))
    else:
        files["table"] = Path(args.table) if args.table else None
        files["cmd"] = Path(args.cmd) if args.cmd else None
        files["car"] = Path(args.car) if args.car else None
        files["state"] = Path(args.state) if args.state else None

    out: Dict[str, List[Dict]] = {}
    for k, p in files.items():
        if p and p.exists():
            rows = read_jsonl(p)
            for r in rows:
                r["_source_file"] = str(p)
                r["_source_kind"] = k
            out[k] = rows
        else:
            out[k] = []
    return out


def normalize_cmd_rows(rows: Sequence[Mapping]) -> List[Dict]:
    rel_t = normalized_time(rows)
    out: List[Dict] = []
    for idx, (row, t) in enumerate(zip(rows, rel_t)):
        out.append({
            "idx": idx,
            "ts": t,
            "state": get_path(row, "state", "mode", default=""),
            "mode": get_path(row, "mode", "state", default=""),
            "phase": get_path(row, "phase", default=""),
            "valid": get_path(row, "valid", default=True),
            "vx_norm": to_float(get_path(row, "vx_norm", "vx"), None),
            "vy_norm": to_float(get_path(row, "vy_norm", "vy"), None),
            "wz_norm": to_float(get_path(row, "wz_norm", "wz"), None),
            "raw_yaw_err_rad": to_float(get_path(row, "raw_yaw_err_rad"), None),
            "raw_dist_err_m": to_float(get_path(row, "raw_dist_err_m"), None),
            "filtered_yaw_err_rad": to_float(get_path(row, "filtered_yaw_err_rad", "obs_yaw_err_rad"), None),
            "filtered_dist_err_m": to_float(get_path(row, "filtered_dist_err_m", "obs_dist_err_m"), None),
            "confidence": to_float(get_path(row, "confidence"), None),
            "edge_found": get_path(row, "edge_found", default=None),
            "_source_file": get_path(row, "_source_file", default=""),
        })
    return out


def normalize_table_rows(rows: Sequence[Mapping]) -> List[Dict]:
    rel_t = normalized_time(rows)
    out: List[Dict] = []
    for idx, (row, t) in enumerate(zip(rows, rel_t)):
        obs = extract_table_obs(row)
        obs.update({
            "idx": idx,
            "ts": t,
            "state": "VISION",
            "mode": "VISION",
            "phase": obs.get("roi_source", ""),
            "vx_norm": None,
            "vy_norm": None,
            "wz_norm": None,
            "filtered_yaw_err_rad": obs.get("yaw_err_rad"),
            "filtered_dist_err_m": obs.get("dist_err_m"),
            "_source_file": get_path(row, "_source_file", default=""),
        })
        out.append(obs)
    return out


def merge_for_csv(table_rows: List[Dict], cmd_rows: List[Dict]) -> List[Dict]:
    # For offline analysis a full nearest-neighbour merge is unnecessary.  Put
    # all rows on one relative timeline and let plotting scripts use each source.
    merged: List[Dict] = []
    for r in table_rows:
        x = dict(r)
        x["kind"] = "table"
        merged.append(x)
    for r in cmd_rows:
        x = dict(r)
        x["kind"] = "cmd"
        merged.append(x)
    merged.sort(key=lambda r: (float(r.get("ts") or 0.0), str(r.get("kind", ""))))
    return merged


def build_summary(table_rows: List[Dict], cmd_rows: List[Dict]) -> Dict:
    rows_for_pred = cmd_rows if cmd_rows else table_rows
    pred = basic_jitter_prediction(rows_for_pred, source="plot_control_run")
    summary = {
        "n_table_rows": len(table_rows),
        "n_cmd_rows": len(cmd_rows),
        "jitter_prediction": pred,
        "table_signals": {
            "yaw": summarize_signal([r.get("yaw_err_rad") for r in table_rows], [r.get("ts") for r in table_rows]),
            "dist": summarize_signal([r.get("dist_err_m") for r in table_rows], [r.get("ts") for r in table_rows]),
            "edge_k": summarize_signal([r.get("edge_k") for r in table_rows], [r.get("ts") for r in table_rows]),
            "confidence": summarize_signal([r.get("confidence") for r in table_rows], [r.get("ts") for r in table_rows]),
        },
        "cmd_signals": {
            "vx_norm": summarize_signal([r.get("vx_norm") for r in cmd_rows], [r.get("ts") for r in cmd_rows]),
            "vy_norm": summarize_signal([r.get("vy_norm") for r in cmd_rows], [r.get("ts") for r in cmd_rows]),
            "wz_norm": summarize_signal([r.get("wz_norm") for r in cmd_rows], [r.get("ts") for r in cmd_rows]),
        },
    }
    return summary


def try_plot(out_dir: Path, table_rows: List[Dict], cmd_rows: List[Dict], state_rows: List[Dict]) -> List[str]:
    written: List[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out_dir / "PLOT_SKIPPED.txt").write_text(
            f"matplotlib is not available: {exc}\nCSV and JSON summaries were still generated.\n",
            encoding="utf-8",
        )
        return written

    if table_rows:
        t = [float(r.get("ts") or 0.0) for r in table_rows]
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(211)
        ax1.plot(t, [r.get("raw_yaw_err_rad") for r in table_rows], label="raw_yaw")
        ax1.plot(t, [r.get("yaw_err_rad") for r in table_rows], label="filtered_yaw")
        ax1.set_ylabel("yaw_err_rad")
        ax1.grid(True)
        ax1.legend(loc="best")
        ax2 = fig.add_subplot(212)
        ax2.plot(t, [r.get("raw_dist_err_m") for r in table_rows], label="raw_dist")
        ax2.plot(t, [r.get("dist_err_m") for r in table_rows], label="filtered_dist")
        ax2.plot(t, [r.get("confidence") for r in table_rows], label="confidence")
        ax2.set_xlabel("time_s")
        ax2.set_ylabel("dist_m / confidence")
        ax2.grid(True)
        ax2.legend(loc="best")
        fig.tight_layout()
        p = out_dir / "obs_timeseries.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(str(p))

    if cmd_rows:
        t = [float(r.get("ts") or 0.0) for r in cmd_rows]
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.plot(t, [r.get("vx_norm") for r in cmd_rows], label="vx_norm")
        ax.plot(t, [r.get("vy_norm") for r in cmd_rows], label="vy_norm")
        ax.plot(t, [r.get("wz_norm") for r in cmd_rows], label="wz_norm")
        ax.set_xlabel("time_s")
        ax.set_ylabel("normalized command")
        ax.grid(True)
        ax.legend(loc="best")
        fig.tight_layout()
        p = out_dir / "cmd_timeseries.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(str(p))

        # Discrete state/phase timeline.
        states = [str(r.get("state") or r.get("mode") or "") for r in cmd_rows]
        phases = [str(r.get("phase") or "") for r in cmd_rows]
        labels = sorted(set(states + phases))
        label_to_y = {s: i for i, s in enumerate(labels)}
        fig = plt.figure(figsize=(12, max(4, 0.25 * len(labels) + 2)))
        ax = fig.add_subplot(111)
        ax.step(t, [label_to_y.get(s, 0) for s in states], where="post", label="state")
        if any(phases):
            ax.step(t, [label_to_y.get(s, 0) for s in phases], where="post", label="phase")
        ax.set_yticks(list(range(len(labels))))
        ax.set_yticklabels(labels)
        ax.set_xlabel("time_s")
        ax.grid(True)
        ax.legend(loc="best")
        fig.tight_layout()
        p = out_dir / "state_timeline.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(str(p))

    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot table-edge control logs")
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--table", default="", help="table_edge_obs.jsonl")
    ap.add_argument("--cmd", default="", help="cmd_vel.jsonl or replay_cmd_vel.jsonl")
    ap.add_argument("--car", default="", help="car_cmd.jsonl")
    ap.add_argument("--state", default="", help="state_block.jsonl or replay_state_trace.jsonl")
    ap.add_argument("--out-dir", default="runs/offline_plot")
    args = ap.parse_args()

    inputs = load_inputs(args)
    table_rows = normalize_table_rows(inputs.get("table", []))
    cmd_rows = normalize_cmd_rows(inputs.get("cmd", []))
    if not cmd_rows and inputs.get("car"):
        cmd_rows = normalize_cmd_rows(inputs.get("car", []))
    merged = merge_for_csv(table_rows, cmd_rows)
    summary = build_summary(table_rows, cmd_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "merged_timeline.jsonl", merged)
    write_csv(out_dir / "merged_timeline.csv", merged)
    write_json(out_dir / "plot_summary.json", summary)
    written_plots = try_plot(out_dir, table_rows, cmd_rows, inputs.get("state", []))

    print(f"[OK] analysis written to {out_dir}")
    print(f"  table_rows={len(table_rows)} cmd_rows={len(cmd_rows)}")
    print(f"  jitter_risk={summary['jitter_prediction']['risk_score']} {summary['jitter_prediction']['risk_level']}")
    for p in written_plots:
        print(f"  plot={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
