#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate visible shaking / jerk risk from real or replay logs.

This is a heuristic diagnostic tool, not a physics-accurate vehicle model.  It
is meant to quickly answer: "If these observation and command traces are used
on the car, is the run likely to look shaky? Which signal is the main cause?"

Typical use:
  python3 tools/predict_motion_jitter.py --run-dir runs/xxx --out-dir /tmp/report
  python3 tools/predict_motion_jitter.py --input replay_cmd_vel.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Optional

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


def load_any(args: argparse.Namespace) -> tuple[Path, List[Dict]]:
    if args.input:
        p = Path(args.input)
        return p, read_jsonl(p)
    if not args.run_dir:
        raise SystemExit("need --input or --run-dir")
    root = Path(args.run_dir)
    for stem in ["replay_cmd_vel", "cmd_vel", "car_cmd", "table_edge_obs"]:
        p = find_log_file(root, stem)
        if p is not None:
            return p, read_jsonl(p)
    raise SystemExit(f"cannot find replay_cmd_vel/cmd_vel/car_cmd/table_edge_obs under {root}")


def normalize_rows(rows: List[Mapping]) -> List[Dict]:
    out: List[Dict] = []
    rel_t = normalized_time(rows)
    for idx, (row, t) in enumerate(zip(rows, rel_t)):
        # If this is table_edge_obs, map obs fields; if it is cmd/replay, keep cmds.
        obs = extract_table_obs(row)
        cmd_like = {
            "idx": idx,
            "ts": t,
            "state": get_path(row, "state", "mode", default=""),
            "mode": get_path(row, "mode", "state", default=""),
            "phase": get_path(row, "phase", default=""),
            "valid": get_path(row, "valid", default=True),
            "vx_norm": to_float(get_path(row, "vx_norm", "vx"), None),
            "vy_norm": to_float(get_path(row, "vy_norm", "vy"), None),
            "wz_norm": to_float(get_path(row, "wz_norm", "wz"), None),
            "filtered_yaw_err_rad": to_float(get_path(row, "filtered_yaw_err_rad", "yaw_err_rad"), obs.get("yaw_err_rad")),
            "filtered_dist_err_m": to_float(get_path(row, "filtered_dist_err_m", "dist_err_m"), obs.get("dist_err_m")),
            "edge_found": obs.get("edge_found"),
            "confidence": obs.get("confidence"),
            "roi_source": obs.get("roi_source"),
            "raw": get_path(row, "raw", default=""),
        }
        out.append(cmd_like)
    return out


def build_detailed_report(rows: List[Mapping], source: str) -> Dict:
    rel_t = normalized_time(rows)
    yaw = [get_path(r, "filtered_yaw_err_rad", "yaw_err_rad") for r in rows]
    dist = [get_path(r, "filtered_dist_err_m", "dist_err_m") for r in rows]
    vx = [get_path(r, "vx_norm", "vx") for r in rows]
    vy = [get_path(r, "vy_norm", "vy") for r in rows]
    wz = [get_path(r, "wz_norm", "wz") for r in rows]
    conf = [get_path(r, "confidence") for r in rows]
    pred = basic_jitter_prediction(rows, source=source)
    report = {
        "source": source,
        "jitter_prediction": pred,
        "signals": {
            "yaw": summarize_signal(yaw, rel_t, deadband=0.004),
            "dist": summarize_signal(dist, rel_t, deadband=0.002),
            "vx_norm": summarize_signal(vx, rel_t, deadband=0.010),
            "vy_norm": summarize_signal(vy, rel_t, deadband=0.010),
            "wz_norm": summarize_signal(wz, rel_t, deadband=0.010),
            "confidence": summarize_signal(conf, rel_t, deadband=0.001),
        },
    }
    # Rank causes based on normalized terms.
    terms = pred.get("terms", {})
    ranked = sorted(terms.items(), key=lambda kv: kv[1], reverse=True)
    cause_map = {
        "yaw_jitter": "视觉角度观测抖动较大，优先检查相机固定、桌边 ROI、yaw 滤波",
        "dist_jitter": "距离观测抖动较大，优先检查 depth ROI、桌面/桌边深度点稳定性",
        "wz_flip": "转向命令频繁换向，优先调大滞回或降低 yaw PID kd/min_abs_output",
        "vx_flip": "前进命令频繁换向，优先检查 dist_err 正负号和 final lock 门限",
        "wz_rate": "转向命令变化率过大，优先降低 wz_slew/速度上限",
        "vx_rate": "前进命令变化率过大，优先降低 vx_slew/速度上限",
        "dropout": "观测丢失/无效比例高，先修视觉稳定性，不要继续加 PID",
    }
    report["top_causes"] = [
        {"term": k, "score": v, "meaning": cause_map.get(k, k)}
        for k, v in ranked[:4]
    ]
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Predict visible car/camera jitter from logs")
    ap.add_argument("--input", default="", help="A jsonl file: replay_cmd_vel/cmd_vel/car_cmd/table_edge_obs")
    ap.add_argument("--run-dir", default="", help="Run dir containing logs")
    ap.add_argument("--out-dir", default="runs/offline_jitter_report")
    args = ap.parse_args()

    src, raw = load_any(args)
    rows = normalize_rows(raw)
    report = build_detailed_report(rows, str(src))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "jitter_input_normalized.jsonl", rows)
    write_csv(out_dir / "jitter_input_normalized.csv", rows)
    write_json(out_dir / "jitter_prediction.json", report)
    pred = report["jitter_prediction"]
    print(f"[OK] jitter report -> {out_dir / 'jitter_prediction.json'}")
    print(f"  risk={pred['risk_score']} {pred['risk_level']}")
    for cause in report.get("top_causes", []):
        print(f"  cause {cause['term']}: {cause['score']:.2f} | {cause['meaning']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
