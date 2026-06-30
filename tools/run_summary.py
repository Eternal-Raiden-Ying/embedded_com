#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot run summary generator.

Usage:
  python3 tools/run_summary.py logs/runs/latest
  python3 tools/run_summary.py logs/runs/run_YYYY... --no-plots

Outputs into <run_dir>/summary/:
  run_summary_auto.md
  run_summary_auto.json
  speed_timeseries.csv
  plots/*.png   (if matplotlib is available)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics as stats
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

EPS = 1e-9


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        if math.isfinite(float(v)):
            return float(v)
        return None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            x = float(s)
            return x if math.isfinite(x) else None
        except Exception:
            return None
    return None


def _get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict):
            return default
        if key not in cur:
            return default
        cur = cur[key]
    return cur


def _first_num(d: Dict[str, Any], paths: Sequence[Tuple[str, ...]]) -> Optional[float]:
    for p in paths:
        x = _safe_float(_get(d, *p))
        if x is not None:
            return x
    return None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                # Keep script robust; old logs may contain damaged partial lines.
                continue
    return out


def find_run_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_symlink():
        path = path.resolve()
    if (path / "orchestrator").is_dir():
        return path
    if path.name == "orchestrator" and path.is_dir():
        return path.parent
    raise SystemExit(f"Cannot find run dir from: {path}")


def percentile(xs: Sequence[float], p: float) -> Optional[float]:
    vals = sorted(x for x in xs if x is not None and math.isfinite(x))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] * (c - k) + vals[c] * (k - f)


def num_stats(xs: Sequence[Optional[float]]) -> Dict[str, Any]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not vals:
        return {"count": 0}
    abs_vals = [abs(x) for x in vals]
    return {
        "count": len(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": sum(vals) / len(vals),
        "mean_abs": sum(abs_vals) / len(abs_vals),
        "p50": percentile(vals, 50),
        "p95_abs": percentile(abs_vals, 95),
        "std": stats.pstdev(vals) if len(vals) >= 2 else 0.0,
    }


def fmt(x: Any, nd: int = 3, unit: str = "") -> str:
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return str(x)
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(xf):
        return "n/a"
    return f"{xf:.{nd}f}{unit}"


def normalize_ts(records: Sequence[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    ts = [_safe_float(r.get("ts")) for r in records]
    vals = [x for x in ts if x is not None]
    return (min(vals), max(vals)) if vals else (None, None)


def intervals_hz(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ts = sorted(x for x in (_safe_float(r.get("ts")) for r in records) if x is not None)
    if len(ts) < 2:
        return {"count": len(ts)}
    dts = [b - a for a, b in zip(ts, ts[1:]) if b >= a]
    if not dts:
        return {"count": len(ts)}
    med = percentile(dts, 50)
    mean = sum(dts) / len(dts)
    return {
        "count": len(ts),
        "dt_mean_s": mean,
        "dt_p50_s": med,
        "dt_p90_s": percentile(dts, 90),
        "hz_mean": (1.0 / mean) if mean and mean > 0 else None,
        "hz_p50": (1.0 / med) if med and med > 0 else None,
    }


def transitions_summary(state_trace: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    trans = [r for r in state_trace if r.get("event") == "state_transition"]
    trans.sort(key=lambda r: _safe_float(r.get("ts")) or 0.0)
    chain: List[str] = []
    for r in trans:
        prev = r.get("previous_state")
        nxt = r.get("next_state")
        if not chain and prev:
            chain.append(str(prev))
        if nxt:
            chain.append(str(nxt))
    # compress immediate duplicates
    compact: List[str] = []
    for s in chain:
        if not compact or compact[-1] != s:
            compact.append(s)
    return trans, compact


def has_subsequence(chain: Sequence[str], required: Sequence[str]) -> bool:
    i = 0
    for s in chain:
        if i < len(required) and s == required[i]:
            i += 1
    return i == len(required)


def state_durations_from_control(control: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    rows = sorted([r for r in control if _safe_float(r.get("ts")) is not None], key=lambda r: _safe_float(r.get("ts")) or 0.0)
    durations: Dict[str, float] = defaultdict(float)
    for a, b in zip(rows, rows[1:]):
        s = str(a.get("state") or "UNKNOWN")
        ta = _safe_float(a.get("ts"))
        tb = _safe_float(b.get("ts"))
        if ta is None or tb is None:
            continue
        dt = max(0.0, min(tb - ta, 1.0))  # cap gaps after Ctrl+C / shutdown
        durations[s] += dt
    return dict(durations)


def extract_velocity_rows(uart: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in sorted(uart, key=lambda x: _safe_float(x.get("ts")) or 0.0):
        vx = _first_num(r, [("vx_mps",), ("uart_tx_cmd", "vx_mps"), ("effective_cmd", "vx_mps"), ("original_cmd", "vx_mps")])
        vy = _first_num(r, [("vy_mps",), ("uart_tx_cmd", "vy_mps"), ("effective_cmd", "vy_mps"), ("original_cmd", "vy_mps")])
        wz = _first_num(r, [("wz_radps",), ("uart_tx_cmd", "wz_radps"), ("effective_cmd", "wz_radps"), ("original_cmd", "wz_radps")])
        if vx is None and vy is None and wz is None:
            raw = str(r.get("raw") or "")
            m = re.search(r"V\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)", raw)
            if m:
                vx, vy, wz = map(float, m.groups())
        rows.append({
            "ts": _safe_float(r.get("ts")),
            "state": r.get("state") or r.get("mode") or "UNKNOWN",
            "raw": str(r.get("raw") or "").replace("\r", ""),
            "vx_mps": vx or 0.0,
            "vy_mps": vy or 0.0,
            "wz_radps": wz or 0.0,
            "reason": r.get("uart_emit_reason") or r.get("reason") or "",
        })
    return rows


def active_window(
    state_trace: Sequence[Dict[str, Any]],
    control: Sequence[Dict[str, Any]],
    velocity_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    trans = [r for r in state_trace if r.get("event") == "state_transition"]
    trans.sort(key=lambda r: _safe_float(r.get("ts")) or 0.0)
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None
    end_reason = ""
    for r in trans:
        if r.get("previous_state") == "IDLE" and r.get("next_state") == "SEARCH_TABLE":
            start_ts = _safe_float(r.get("ts"))
            break
    if start_ts is None:
        candidates = [_safe_float(r.get("ts")) for r in list(control) + list(velocity_rows)]
        candidates = [x for x in candidates if x is not None]
        start_ts = min(candidates) if candidates else None
    if start_ts is not None:
        for prev_state, next_state, reason in (
            ("GRASP", "ERROR_RECOVERY", "grasp_to_error_recovery"),
            ("GRASP", "IDLE", "grasp_to_idle"),
            ("ERROR_RECOVERY", "IDLE", "error_recovery_to_idle"),
        ):
            for r in trans:
                ts = _safe_float(r.get("ts"))
                if ts is None or ts < start_ts:
                    continue
                if r.get("previous_state") == prev_state and r.get("next_state") == next_state:
                    end_ts = ts
                    end_reason = reason
                    break
            if end_ts is not None:
                break
    if end_ts is None:
        non_idle_ts: List[float] = []
        for r in list(control) + list(velocity_rows):
            ts = _safe_float(r.get("ts"))
            if ts is None or (start_ts is not None and ts < start_ts):
                continue
            if str(r.get("state") or "").upper() != "IDLE":
                non_idle_ts.append(ts)
        for r in trans:
            ts = _safe_float(r.get("ts"))
            if ts is None or (start_ts is not None and ts < start_ts):
                continue
            if str(r.get("next_state") or "").upper() != "IDLE":
                non_idle_ts.append(ts)
        if non_idle_ts:
            end_ts = max(non_idle_ts)
            end_reason = "last_non_idle_state"
    return {"start_ts": start_ts, "end_ts": end_ts, "end_reason": end_reason}


def in_window(row: Dict[str, Any], window: Dict[str, Any]) -> bool:
    ts = _safe_float(row.get("ts"))
    if ts is None:
        return False
    start_ts = window.get("start_ts")
    end_ts = window.get("end_ts")
    if start_ts is not None and ts < float(start_ts):
        return False
    if end_ts is not None and ts > float(end_ts):
        return False
    return True


def moving_velocity_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        r for r in rows
        if abs(float(r.get("vx_mps", 0.0) or 0.0)) + abs(float(r.get("vy_mps", 0.0) or 0.0)) + abs(float(r.get("wz_radps", 0.0) or 0.0)) > 1e-4
    ]


def velocity_stats_for(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "count": len(rows),
        "vx_mps": num_stats([r.get("vx_mps") for r in rows]),
        "vy_mps": num_stats([r.get("vy_mps") for r in rows]),
        "wz_radps": num_stats([r.get("wz_radps") for r in rows]),
    }


def velocity_stats_by_state(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[str(r.get("state") or "UNKNOWN")].append(r)
    return {state: velocity_stats_for(vals) for state, vals in sorted(grouped.items())}


def final_close_summary(control: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    for r in control:
        state = r.get("state")
        reason = r.get("final_distance_servo_reason") or r.get("docking_reason") or ""
        if (
            state in {"YOLO_APPROACH", "FINAL_SLOW_STOP", "AT_TABLE_EDGE"}
            and (
                r.get("final_phase_active")
                or r.get("final_depth_valid") is not None
                or r.get("final_depth_m") is not None
                or r.get("final_distance_servo_reason") is not None
                or r.get("near_table_latched")
                or r.get("final_depth_latched")
                or "final" in str(reason).lower()
                or "close_range" in str(reason).lower()
            )
        ):
            rows.append(r)
    reasons = Counter(str(r.get("final_distance_servo_reason") or r.get("docking_reason") or "") for r in rows)
    has_final_depth_field = any("final_depth_m" in r or "final_depth_valid" in r for r in rows)
    valid = [bool(r.get("final_depth_valid")) for r in rows if r.get("final_depth_valid") is not None]
    depth_values = [_safe_float(r.get("final_depth_m")) for r in rows]
    source_counts = Counter(str(r.get("final_depth_source") or "missing") for r in rows)
    final_vx = [_safe_float(r.get("vx_mps")) for r in rows]
    return {
        "count": len(rows),
        "has_final_depth_field": bool(has_final_depth_field),
        "final_depth_valid_count": sum(1 for x in valid if x),
        "final_depth_valid_ratio": (sum(1 for x in valid if x) / len(valid)) if valid else None,
        "final_depth_source_distribution": dict(source_counts.most_common()),
        "final_depth_m_stats": num_stats(depth_values),
        "final_distance_servo_reason_distribution": dict(reasons.most_common(12)),
        "final_vx_stats": num_stats(final_vx),
        "reason_counts": dict(reasons.most_common(12)),
        "warning": "" if has_final_depth_field else "WARN: final_depth_m missing; control still uses old ROI debug fields",
        "rows": rows,
    }


def target_summary(target_obs: Sequence[Dict[str, Any]], control: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [r for r in target_obs if r.get("target_found") or r.get("matched_cls") or r.get("matched_conf")]
    if not rows:
        # fallback from control_summary if target fields are copied there in future.
        rows = [r for r in control if r.get("target_found") or r.get("target_center_x_norm") is not None]
    centers: List[float] = []
    confs: List[float] = []
    for r in rows:
        cx = _first_num(r, [("matched_center_full_norm", "cx"), ("matched_center", "cx"), ("target_center", "cx"), ("target_center_x_norm",), ("cx_norm",)])
        cf = _first_num(r, [("matched_conf",), ("target_conf",), ("best_conf",)])
        if cx is not None:
            centers.append(cx)
        if cf is not None:
            confs.append(cf)
    return {
        "count": len(rows),
        "center_start": centers[0] if centers else None,
        "center_end": centers[-1] if centers else None,
        "center_min": min(centers) if centers else None,
        "center_max": max(centers) if centers else None,
        "center_mean": sum(centers) / len(centers) if centers else None,
        "conf_stats": num_stats(confs),
        "centers": centers,
        "rows": rows,
    }


def grasp_summary(state_trace: Sequence[Dict[str, Any]], vision_req: Sequence[Dict[str, Any]], vision_dir: Path) -> Dict[str, Any]:
    reqs = []
    for r in vision_req:
        if r.get("mode_hint") == "GRASP_REMOTE" or r.get("stage") == "GRASP":
            reqs.append(r)
    grasp_trans = [r for r in state_trace if r.get("previous_state") == "GRASP" or r.get("next_state") == "GRASP"]
    last_reason = None
    for r in reversed(state_trace):
        if r.get("previous_state") == "GRASP" or r.get("state") == "GRASP":
            last_reason = r.get("reason") or r.get("transition_reason")
            break
    remote_lines: List[str] = []
    for fname in ["vision.out", "vision.log"]:
        p = vision_dir / fname
        if p.exists():
            try:
                with p.open("r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if "GRASP" in line or "remote" in line.lower() or "grasp" in line.lower():
                            remote_lines.append(line.strip())
            except Exception:
                pass
    return {
        "grasp_remote_requested": bool(reqs),
        "request_count": len(reqs),
        "last_request": reqs[-1] if reqs else None,
        "grasp_transition_count": len(grasp_trans),
        "last_grasp_reason": last_reason,
        "remote_log_tail": remote_lines[-20:],
    }


def conclusion_summary(summary: Dict[str, Any]) -> Dict[str, str]:
    chain_ok = bool(summary.get("chain_complete"))
    final_close = summary.get("final_close") or {}
    final_ratio = final_close.get("final_depth_valid_ratio")
    if not final_close.get("has_final_depth_field"):
        final_depth = "MISSING_FIELD"
    elif final_ratio is None:
        final_depth = "BAD"
    elif float(final_ratio) >= 0.80:
        final_depth = "GOOD"
    elif float(final_ratio) >= 0.30:
        final_depth = "WEAK"
    else:
        final_depth = "BAD"
    final_safety = "PASS" if final_depth == "GOOD" else ("WARN" if final_depth in {"WEAK", "MISSING_FIELD"} else "FAIL")
    target = summary.get("target") or {}
    target_slide = "PASS" if target.get("count", 0) > 0 else "FAIL"
    grasp = summary.get("grasp") or {}
    if not grasp.get("grasp_remote_requested"):
        grasp_state = "NOT_TRIGGERED"
    else:
        last_reason = str(grasp.get("last_grasp_reason") or "").lower()
        if "success" in last_reason or "done" in last_reason:
            grasp_state = "SUCCESS"
        elif "timeout" in last_reason:
            grasp_state = "TIMEOUT"
        elif "stop" in last_reason or "idle" in last_reason:
            grasp_state = "STOPPED"
        else:
            grasp_state = "TRIGGERED"
    return {
        "Main chain": "PASS" if chain_ok else "FAIL",
        "Final safety": final_safety,
        "Target slide": target_slide,
        "Grasp remote": grasp_state,
        "Final depth": final_depth,
    }


def write_velocity_csv(path: Path, rows: Sequence[Dict[str, Any]], t0: Optional[float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t_s", "ts", "state", "vx_mps", "vy_mps", "wz_radps", "reason", "raw"])
        w.writeheader()
        for r in rows:
            ts = r.get("ts")
            t = (float(ts) - t0) if ts is not None and t0 is not None else ""
            w.writerow({
                "t_s": f"{t:.3f}" if isinstance(t, float) else "",
                "ts": ts,
                "state": r.get("state", ""),
                "vx_mps": r.get("vx_mps", 0.0),
                "vy_mps": r.get("vy_mps", 0.0),
                "wz_radps": r.get("wz_radps", 0.0),
                "reason": r.get("reason", ""),
                "raw": r.get("raw", ""),
            })


def make_plots(out_dir: Path, summary: Dict[str, Any], velocity_rows: Sequence[Dict[str, Any]], control: Sequence[Dict[str, Any]], target: Dict[str, Any], final_info: Dict[str, Any]) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"plot_error": f"matplotlib unavailable: {exc}"}

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    t0 = summary.get("t0")
    old_global = plot_dir / "uart_velocity.png"
    if old_global.exists():
        try:
            old_global.unlink()
        except Exception:
            pass

    def plot_velocity(rows: Sequence[Dict[str, Any]], filename: str, title: str, key: str) -> None:
        if not rows:
            return
        xs = [r["ts"] - t0 for r in rows if r.get("ts") is not None and t0 is not None]
        vx = [r.get("vx_mps", 0.0) for r in rows if r.get("ts") is not None and t0 is not None]
        vy = [r.get("vy_mps", 0.0) for r in rows if r.get("ts") is not None and t0 is not None]
        wz = [r.get("wz_radps", 0.0) for r in rows if r.get("ts") is not None and t0 is not None]
        if xs:
            plt.figure(figsize=(12, 5))
            plt.step(xs, vx, where="post", label="vx_mps")
            plt.step(xs, vy, where="post", label="vy_mps")
            plt.step(xs, wz, where="post", label="wz_radps")
            plt.xlabel("time since active start (s)")
            plt.ylabel("command")
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            p = plot_dir / filename
            plt.tight_layout()
            plt.savefig(p, dpi=150)
            plt.close()
            paths[key] = str(p)

    active_rows = list(velocity_rows)
    plot_velocity(active_rows, "uart_velocity_active.png", "UART velocity command timeline (active window)", "uart_velocity_active")
    plot_velocity([r for r in active_rows if str(r.get("state") or "") == "YOLO_APPROACH"], "yolo_approach_velocity.png", "YOLO_APPROACH velocity commands", "yolo_approach_velocity")
    plot_velocity([r for r in active_rows if str(r.get("state") or "") in {"FINAL_SLOW_STOP", "AT_TABLE_EDGE"} or "final" in str(r.get("reason") or "").lower()], "final_velocity.png", "Final velocity commands", "final_velocity")
    plot_velocity([r for r in active_rows if str(r.get("state") or "") in {"EDGE_SLIDE_SEARCH", "TARGET_CONFIRM"}], "target_slide_velocity.png", "Target slide velocity commands", "target_slide_velocity")

    # 2) State duration bar.
    durations = summary.get("state_durations", {})
    if durations:
        items = sorted(durations.items(), key=lambda kv: kv[1], reverse=True)
        names = [k for k, _ in items]
        vals = [v for _, v in items]
        plt.figure(figsize=(10, max(4, 0.35 * len(names))))
        plt.barh(names, vals)
        plt.xlabel("duration (s)")
        plt.title("State duration summary")
        plt.gca().invert_yaxis()
        plt.grid(True, axis="x", alpha=0.3)
        p = plot_dir / "state_durations.png"
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        paths["state_durations"] = str(p)

    # 3) Target center plot.
    rows = target.get("rows") or []
    tx: List[float] = []
    tc: List[float] = []
    for r in rows:
        ts = _safe_float(r.get("ts"))
        cx = _first_num(r, [("matched_center_full_norm", "cx"), ("matched_center", "cx"), ("target_center", "cx"), ("target_center_x_norm",), ("cx_norm",)])
        if ts is not None and cx is not None and t0 is not None:
            tx.append(ts - t0)
            tc.append(cx)
    if tx:
        plt.figure(figsize=(12, 4))
        plt.plot(tx, tc, marker=".", linewidth=1.0)
        plt.axhline(0.5, linestyle="--", linewidth=1.0)
        plt.axhline(0.44, linestyle=":", linewidth=1.0)
        plt.axhline(0.56, linestyle=":", linewidth=1.0)
        plt.xlabel("time since run start (s)")
        plt.ylabel("target center x norm")
        plt.title("Target center convergence")
        plt.grid(True, alpha=0.3)
        p = plot_dir / "target_center_x.png"
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        paths["target_center_x"] = str(p)

    # 4) Final abstract depth plot.
    frows = final_info.get("rows") or []
    fx: List[float] = []
    fdepth: List[float] = []
    fv: List[int] = []
    for r in frows:
        ts = _safe_float(r.get("ts"))
        depth_m = _safe_float(r.get("final_depth_m"))
        if ts is not None and t0 is not None:
            fx.append(ts - t0)
            fdepth.append(depth_m if depth_m is not None else float("nan"))
            fv.append(1 if r.get("final_depth_valid") else 0)
    if fx:
        plt.figure(figsize=(12, 4))
        plt.step(fx, fdepth, where="post", linewidth=1.0, label="final_depth_m")
        plt.scatter(fx, [0.05 if v == 0 else 0.0 for v in fv], s=8, label="invalid marker")
        plt.xlabel("time since active start (s)")
        plt.ylabel("depth (m)")
        plt.title("Final abstract depth")
        plt.grid(True, alpha=0.3)
        plt.legend()
        p = plot_dir / "final_depth_m.png"
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        paths["final_depth_m"] = str(p)

    return paths


def md_table(rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return ""
    # all cells to str
    srows = [[str(c) for c in row] for row in rows]
    widths = [max(len(row[i]) for row in srows) for i in range(len(srows[0]))]
    lines = []
    header = "| " + " | ".join(srows[0][i].ljust(widths[i]) for i in range(len(widths))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    lines.append(header)
    lines.append(sep)
    for row in srows[1:]:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(widths))) + " |")
    return "\n".join(lines)


def render_markdown(run_dir: Path, summary: Dict[str, Any], plots: Dict[str, str]) -> str:
    lines: List[str] = []
    lines.append(f"# Robot Run Summary: `{run_dir.name}`")
    lines.append("")
    lines.append("## 1. 一句话结论")
    conclusions = summary.get("conclusions") or {}
    for name in ("Main chain", "Final safety", "Target slide", "Grasp remote", "Final depth"):
        if name in conclusions:
            lines.append(f"- {name}: {conclusions[name]}")
    complete = summary["chain_complete"]
    grasp_req = summary["grasp"].get("grasp_remote_requested")
    final_ratio = summary["final_close"].get("final_depth_valid_ratio")
    target = summary["target"]
    lines.append(f"- 状态链完整性：{'✅ 完整到 GRASP' if complete else '⚠️ 未完整到 GRASP'}")
    lines.append(f"- GRASP_REMOTE 触发：{'✅ 已触发' if grasp_req else '⚠️ 未触发'}")
    if final_ratio is not None:
        lines.append(f"- final_depth 有效率：{fmt(final_ratio * 100, 1, '%')} ({summary['final_close'].get('final_depth_valid_count')}/{summary['final_close'].get('count')})")
    if summary["final_close"].get("warning"):
        lines.append(f"- {summary['final_close'].get('warning')}")
    if target.get("center_start") is not None:
        lines.append(f"- target center：{fmt(target.get('center_start'))} → {fmt(target.get('center_end'))}")
    lines.append("")

    lines.append("## 2. 状态切换总览")
    chain = summary.get("state_chain") or []
    lines.append("```text")
    lines.append(" -> ".join(chain) if chain else "<no state_transition found>")
    lines.append("```")
    trans_rows = [["time_s", "from", "to", "reason"]]
    t0 = summary.get("t0")
    for tr in summary.get("transitions", []):
        ts = tr.get("ts")
        trans_rows.append([
            fmt((ts - t0) if ts is not None and t0 is not None else None, 2),
            tr.get("previous_state"),
            tr.get("next_state"),
            tr.get("reason") or tr.get("transition_reason") or "",
        ])
    lines.append(md_table(trans_rows))
    lines.append("")

    lines.append("## 3. 速度统计（UART 实际发送）")
    vstats = summary["velocity_stats_moving_only"]
    rows = [["axis", "mean", "mean_abs", "min", "max", "p95_abs", "std"]]
    for axis in ["vx_mps", "vy_mps", "wz_radps"]:
        st = vstats.get(axis, {})
        rows.append([axis, fmt(st.get("mean")), fmt(st.get("mean_abs")), fmt(st.get("min")), fmt(st.get("max")), fmt(st.get("p95_abs")), fmt(st.get("std"))])
    lines.append(md_table(rows))
    hz = summary.get("uart_hz", {})
    lines.append(f"- active moving_only count={vstats.get('count', 0)}")
    lines.append(f"- UART 频率：mean={fmt(hz.get('hz_mean'), 2, 'Hz')}, p50={fmt(hz.get('hz_p50'), 2, 'Hz')}, count={hz.get('count', 0)}")
    lines.append("")

    lines.append("## 4. Final / Close 阶段")
    fc = summary["final_close"]
    lines.append(f"- final/close entries: {fc.get('count', 0)}")
    lines.append(f"- final_depth valid: {fc.get('final_depth_valid_count', 0)}/{fc.get('count', 0)} = {fmt((fc.get('final_depth_valid_ratio') or 0) * 100, 1, '%')}")
    ds = fc.get("final_depth_m_stats", {})
    lines.append(f"- final_depth_m: mean={fmt(ds.get('mean'))}, min={fmt(ds.get('min'))}, max={fmt(ds.get('max'))}")
    if fc.get("final_depth_source_distribution"):
        lines.append("- final_depth source counts:")
        for name, count in fc.get("final_depth_source_distribution", {}).items():
            lines.append(f"  - `{name or '<empty>'}`: {count}")
    lines.append("- reason counts:")
    for reason, count in fc.get("final_distance_servo_reason_distribution", {}).items():
        lines.append(f"  - `{reason or '<empty>'}`: {count}")
    lines.append("")

    lines.append("## 5. Target / Slide / Lock")
    tsu = summary["target"]
    lines.append(f"- target obs count: {tsu.get('count', 0)}")
    lines.append(f"- center start/end: {fmt(tsu.get('center_start'))} -> {fmt(tsu.get('center_end'))}")
    lines.append(f"- center min/max: {fmt(tsu.get('center_min'))} / {fmt(tsu.get('center_max'))}")
    cf = tsu.get("conf_stats", {})
    lines.append(f"- confidence mean/max: {fmt(cf.get('mean'))} / {fmt(cf.get('max'))}")
    lines.append("")

    lines.append("## 6. Grasp / Remote")
    gs = summary["grasp"]
    lines.append(f"- GRASP_REMOTE requested: {gs.get('grasp_remote_requested')}")
    lines.append(f"- request count: {gs.get('request_count')}")
    if gs.get("last_request"):
        req = gs["last_request"]
        lines.append(f"- last request: stage={req.get('stage')} mode_hint={req.get('mode_hint')} class_id={_get(req, 'payload', 'class_id', default=req.get('class_id'))}")
    lines.append(f"- last grasp reason: {gs.get('last_grasp_reason')}")
    if gs.get("remote_log_tail"):
        lines.append("- remote/grasp log tail:")
        for l in gs["remote_log_tail"][-8:]:
            lines.append(f"  - `{l[:220]}`")
    lines.append("")

    lines.append("## 7. 状态耗时")
    dur_rows = [["state", "duration_s"]]
    for s, d in sorted(summary.get("state_durations", {}).items(), key=lambda kv: kv[1], reverse=True):
        dur_rows.append([s, fmt(d, 2)])
    lines.append(md_table(dur_rows))
    lines.append("")

    lines.append("## 8. 图表")
    if plots:
        for name, p in plots.items():
            if name == "plot_error":
                lines.append(f"- {p}")
            else:
                rel = os.path.relpath(p, start=run_dir)
                lines.append(f"- {name}: `{rel}`")
    else:
        lines.append("- 未生成图表。")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate robot run summary from JSONL logs.")
    ap.add_argument("run_dir", nargs="?", default="logs/runs/latest", help="run dir or orchestrator dir. Default: logs/runs/latest")
    ap.add_argument("--no-plots", action="store_true", help="do not generate PNG plots")
    ap.add_argument("--quiet", action="store_true", help="do not print markdown summary to stdout")
    args = ap.parse_args(argv)

    run_dir = find_run_dir(Path(args.run_dir))
    orch = run_dir / "orchestrator"
    vision = run_dir / "vision"
    out_dir = run_dir / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    state_trace = read_jsonl(orch / "state_trace.jsonl")
    control = read_jsonl(orch / "control_summary.jsonl")
    uart = read_jsonl(orch / "uart_tx.jsonl")
    vision_req = read_jsonl(orch / "vision_req.jsonl")
    target_obs = read_jsonl(orch / "target_obs.jsonl")
    table_edge_obs = read_jsonl(orch / "table_edge_obs.jsonl")

    all_records: List[Dict[str, Any]] = []
    for rs in [state_trace, control, uart, vision_req, target_obs, table_edge_obs]:
        all_records.extend(rs)
    t0, t1 = normalize_ts(all_records)

    vrows = extract_velocity_rows(uart)
    window = active_window(state_trace, control, vrows)
    active_control = [r for r in control if in_window(r, window)]
    active_vrows = [r for r in vrows if in_window(r, window)]
    active_target_obs = [r for r in target_obs if in_window(r, window)]
    active_table_edge_obs = [r for r in table_edge_obs if in_window(r, window)]
    active_t0 = window.get("start_ts") if window.get("start_ts") is not None else t0
    active_t1 = window.get("end_ts") if window.get("end_ts") is not None else t1

    trans, chain = transitions_summary(state_trace)
    state_durations = state_durations_from_control(active_control)
    write_velocity_csv(out_dir / "speed_timeseries.csv", active_vrows, active_t0)

    moving_rows = moving_velocity_rows(active_vrows)
    yolo_rows = [r for r in active_vrows if str(r.get("state") or "") == "YOLO_APPROACH"]
    final_rows = [r for r in active_vrows if str(r.get("state") or "") in {"FINAL_SLOW_STOP", "AT_TABLE_EDGE"} or "final" in str(r.get("reason") or "").lower()]
    target_slide_rows = [r for r in active_vrows if str(r.get("state") or "") in {"EDGE_SLIDE_SEARCH", "TARGET_CONFIRM"}]

    required_chain = ["SEARCH_TABLE", "YOLO_APPROACH", "AT_TABLE_EDGE", "SEARCH_TARGET_INIT", "EDGE_SLIDE_SEARCH", "TARGET_CONFIRM", "TARGET_LOCKED", "FREEZE_BASE", "GRASP"]
    # Allow YOLO_ACQUIRE_ALIGN between SEARCH_TABLE and YOLO_APPROACH.
    chain_complete = has_subsequence(chain, required_chain)

    final_info = final_close_summary(active_control)
    target_info = target_summary(active_target_obs, active_control)
    grasp_info = grasp_summary(state_trace, vision_req, vision)

    summary: Dict[str, Any] = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "t0": active_t0,
        "t1": active_t1,
        "duration_s": (active_t1 - active_t0) if active_t0 is not None and active_t1 is not None else None,
        "active_window": window,
        "state_chain": chain,
        "transitions": [
            {
                "ts": _safe_float(r.get("ts")),
                "previous_state": r.get("previous_state"),
                "next_state": r.get("next_state"),
                "reason": r.get("reason") or r.get("transition_reason"),
            }
            for r in trans
        ],
        "chain_complete": chain_complete,
        "required_chain": required_chain,
        "state_durations": state_durations,
        "velocity_stats_active_all": velocity_stats_for(active_vrows),
        "velocity_stats_moving_only": velocity_stats_for(moving_rows),
        "velocity_stats_by_state": velocity_stats_by_state(active_vrows),
        "velocity_stats_yolo_approach": velocity_stats_for(yolo_rows),
        "velocity_stats_final_only": velocity_stats_for(final_rows),
        "velocity_stats_target_slide_only": velocity_stats_for(target_slide_rows),
        "uart_hz": intervals_hz(active_vrows),
        "control_hz": intervals_hz(active_control),
        "vision_table_edge_hz": intervals_hz(active_table_edge_obs),
        "target_obs_hz": intervals_hz(active_target_obs),
        "final_close": {k: v for k, v in final_info.items() if k != "rows"},
        "target": {k: v for k, v in target_info.items() if k not in {"rows", "centers"}},
        "grasp": grasp_info,
        "files": {
            "markdown": str(out_dir / "run_summary_auto.md"),
            "json": str(out_dir / "run_summary_auto.json"),
            "speed_csv": str(out_dir / "speed_timeseries.csv"),
        },
    }
    summary["conclusions"] = conclusion_summary(summary)

    plots: Dict[str, str] = {}
    if not args.no_plots:
        plots = make_plots(out_dir, summary, active_vrows, active_control, target_info, final_info)
    summary["plots"] = plots

    md = render_markdown(run_dir, summary, plots)
    (out_dir / "run_summary_auto.md").write_text(md, encoding="utf-8")
    with (out_dir / "run_summary_auto.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not args.quiet:
        print(md)
        print("\n[OK] summary written to:")
        print(f"  {out_dir / 'run_summary_auto.md'}")
        print(f"  {out_dir / 'run_summary_auto.json'}")
        print(f"  {out_dir / 'speed_timeseries.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
