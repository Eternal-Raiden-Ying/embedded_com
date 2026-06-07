#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline analysis helpers for the SC171 table-edge docking stack.

The tools under this directory are intentionally lightweight and can run on a
laptop without camera, STM32, ipc/protocol.py, or the full orchestrator process.
They consume jsonl logs and/or generate synthetic table-edge observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


Number = Optional[float]


def read_jsonl(path: os.PathLike | str) -> List[Dict[str, Any]]:
    p = Path(path)
    rows: List[Dict[str, Any]] = []
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception as exc:
                rows.append({"_parse_error": str(exc), "_line_no": line_no, "raw_line": s})
                continue
            if isinstance(obj, dict):
                obj.setdefault("_line_no", line_no)
                rows.append(obj)
            else:
                rows.append({"value": obj, "_line_no": line_no})
    return rows


def write_jsonl(path: os.PathLike | str, rows: Iterable[Mapping[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False, sort_keys=False) + "\n")


def write_json(path: os.PathLike | str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: os.PathLike | str, rows: Sequence[Mapping[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: flatten_scalar(row.get(k)) for k in keys})


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return to_jsonable(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def flatten_scalar(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return v
    return json.dumps(to_jsonable(v), ensure_ascii=False)


def to_float(v: Any, default: Number = None) -> Number:
    if v is None or v == "":
        return default
    try:
        x = float(v)
    except Exception:
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return x


def to_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def get_path(row: Mapping[str, Any], *candidates: str, default: Any = None) -> Any:
    for cand in candidates:
        cur: Any = row
        ok = True
        for part in cand.split("."):
            if isinstance(cur, Mapping) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    return default


def first_existing_file(run_dir: os.PathLike | str, names: Sequence[str]) -> Optional[Path]:
    root = Path(run_dir)
    for name in names:
        p = root / name
        if p.exists():
            return p
    return None


def find_log_file(run_dir: os.PathLike | str, stem: str) -> Optional[Path]:
    root = Path(run_dir)
    candidates = [
        root / f"{stem}.jsonl",
        root / f"orch_{stem}.jsonl",
        root / "orch" / f"{stem}.jsonl",
        root / "logs" / f"{stem}.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = sorted(root.rglob(f"*{stem}*.jsonl")) if root.exists() else []
    return matches[0] if matches else None


def normalized_time(rows: Sequence[Mapping[str, Any]], rate_hz: float = 5.0) -> List[float]:
    out: List[float] = []
    last_t: Optional[float] = None
    step = 1.0 / max(1e-6, float(rate_hz))
    for i, row in enumerate(rows):
        t = to_float(get_path(row, "ts", "timestamp", "time", "wall_ts", "mono_ts"), None)
        if t is None:
            t = (last_t + step) if last_t is not None else i * step
        if last_t is not None and t <= last_t:
            t = last_t + step
        out.append(float(t))
        last_t = float(t)
    if out:
        t0 = out[0]
        return [t - t0 for t in out]
    return out


def robust_mad(values: Sequence[Any]) -> Number:
    xs = [float(x) for x in values if to_float(x, None) is not None]
    if not xs:
        return None
    med = statistics.median(xs)
    mad = statistics.median([abs(x - med) for x in xs])
    return 1.4826 * mad


def safe_stdev(values: Sequence[Any]) -> Number:
    xs = [float(x) for x in values if to_float(x, None) is not None]
    if len(xs) < 2:
        return None
    return statistics.stdev(xs)


def safe_mean(values: Sequence[Any]) -> Number:
    xs = [float(x) for x in values if to_float(x, None) is not None]
    return statistics.mean(xs) if xs else None


def safe_max_abs(values: Sequence[Any]) -> Number:
    xs = [abs(float(x)) for x in values if to_float(x, None) is not None]
    return max(xs) if xs else None


def sign_changes(values: Sequence[Any], deadband: float = 1e-6) -> int:
    last = 0
    count = 0
    for v0 in values:
        v = to_float(v0, None)
        if v is None or abs(v) <= deadband:
            continue
        s = 1 if v > 0 else -1
        if last and s != last:
            count += 1
        last = s
    return count


def diff_rate(times: Sequence[float], values: Sequence[Any]) -> List[float]:
    rates: List[float] = []
    prev_t: Optional[float] = None
    prev_v: Optional[float] = None
    for t, v0 in zip(times, values):
        v = to_float(v0, None)
        if v is None:
            continue
        if prev_t is not None and prev_v is not None:
            dt = max(1e-6, float(t) - float(prev_t))
            rates.append((float(v) - float(prev_v)) / dt)
        prev_t, prev_v = float(t), float(v)
    return rates


def summarize_signal(values: Sequence[Any], times: Optional[Sequence[float]] = None, deadband: float = 1e-6) -> Dict[str, Any]:
    xs = [to_float(x, None) for x in values]
    valid = [float(x) for x in xs if x is not None]
    summary: Dict[str, Any] = {
        "n": len(xs),
        "valid_n": len(valid),
        "mean": safe_mean(valid),
        "std": safe_stdev(valid),
        "mad_std": robust_mad(valid),
        "max_abs": safe_max_abs(valid),
        "sign_changes": sign_changes(valid, deadband=deadband),
    }
    if times is not None and len(times) == len(values):
        rates = diff_rate(times, values)
        summary["rate_std"] = safe_stdev(rates)
        summary["rate_max_abs"] = safe_max_abs(rates)
    return summary


def extract_table_obs(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize a table_edge_obs row from real logs or replay logs."""
    return {
        "ts": to_float(get_path(row, "ts", "timestamp", "wall_ts"), 0.0),
        "edge_found": to_bool(get_path(row, "edge_found", "table_found", "found"), False),
        "confidence": to_float(get_path(row, "confidence", "conf", "score"), 0.0),
        "yaw_err_rad": to_float(get_path(row, "yaw_err_rad", "filtered_yaw_err_rad", "edge_yaw_rad"), None),
        "dist_err_m": to_float(get_path(row, "dist_err_m", "filtered_dist_err_m", "distance_err_m"), None),
        "lateral_err_m": to_float(get_path(row, "lateral_err_m", "lat_err_m"), None),
        "edge_ready": to_bool(get_path(row, "edge_ready", "ready"), False),
        "depth_valid": get_path(row, "depth_valid", default=None),
        "source": str(get_path(row, "source", default="log")),
        "edge_k": to_float(get_path(row, "edge_k", "filtered_edge_k"), None),
        "raw_yaw_err_rad": to_float(get_path(row, "raw_yaw_err_rad"), None),
        "raw_dist_err_m": to_float(get_path(row, "raw_dist_err_m"), None),
        "roi_source": get_path(row, "roi_source", "debug.roi_source", default=""),
    }


def env_float(name: str, default: float) -> float:
    return float(to_float(os.environ.get(name), default))


def env_int(name: str, default: int) -> int:
    v = to_float(os.environ.get(name), None)
    return int(v) if v is not None else int(default)


def env_bool(name: str, default: bool) -> bool:
    return to_bool(os.environ.get(name), default)


def make_docking_cfg_from_env():
    """Create DockingControlConfig without importing the full orchestrator."""
    from ctrl_code.control.types import DockingControlConfig

    cfg = DockingControlConfig()
    cfg.min_confidence = env_float("ORCH_DOCKING_MIN_CONFIDENCE", cfg.min_confidence)
    cfg.obs_timeout_s = env_float("ORCH_DOCKING_OBS_TIMEOUT_S", cfg.obs_timeout_s)
    cfg.obs_grace_s = env_float("ORCH_DOCKING_OBS_GRACE_S", cfg.obs_grace_s)
    cfg.obs_grace_cmd_scale = env_float("ORCH_DOCKING_OBS_GRACE_CMD_SCALE", cfg.obs_grace_cmd_scale)
    cfg.dt_min_s = env_float("ORCH_DOCKING_DT_MIN_S", cfg.dt_min_s)
    cfg.reset_on_mode_change = env_bool("ORCH_DOCKING_RESET_ON_MODE_CHANGE", cfg.reset_on_mode_change)
    cfg.coarse_align_enter_rad = env_float("ORCH_DOCKING_COARSE_ENTER_RAD", cfg.coarse_align_enter_rad)
    cfg.coarse_align_exit_rad = env_float("ORCH_DOCKING_COARSE_EXIT_RAD", cfg.coarse_align_exit_rad)
    cfg.spin_only_yaw_rad = env_float("ORCH_DOCKING_SPIN_ONLY_YAW_RAD", cfg.spin_only_yaw_rad)
    cfg.precise_yaw_tol_rad = env_float("ORCH_DOCKING_PRECISE_YAW_RAD", cfg.precise_yaw_tol_rad)
    cfg.precise_dist_tol_m = env_float("ORCH_DOCKING_PRECISE_DIST_M", cfg.precise_dist_tol_m)
    cfg.precise_lateral_tol_m = env_float("ORCH_DOCKING_PRECISE_LAT_M", cfg.precise_lateral_tol_m)
    cfg.precise_stable_s = env_float("ORCH_DOCKING_PRECISE_STABLE_S", cfg.precise_stable_s)
    cfg.coarse_max_wz_norm = env_float("ORCH_DOCKING_COARSE_MAX_WZ", cfg.coarse_max_wz_norm)
    cfg.approach_max_vx_norm = env_float("ORCH_DOCKING_APPROACH_MAX_VX", cfg.approach_max_vx_norm)
    cfg.approach_max_vy_norm = env_float("ORCH_DOCKING_APPROACH_MAX_VY", cfg.approach_max_vy_norm)
    cfg.approach_max_wz_norm = env_float("ORCH_DOCKING_APPROACH_MAX_WZ", cfg.approach_max_wz_norm)
    cfg.final_max_vx_norm = env_float("ORCH_DOCKING_FINAL_MAX_VX", cfg.final_max_vx_norm)
    cfg.final_max_vy_norm = env_float("ORCH_DOCKING_FINAL_MAX_VY", cfg.final_max_vy_norm)
    cfg.final_max_wz_norm = env_float("ORCH_DOCKING_FINAL_MAX_WZ", cfg.final_max_wz_norm)
    cfg.vx_slew_per_s = env_float("ORCH_DOCKING_VX_SLEW", cfg.vx_slew_per_s)
    cfg.vy_slew_per_s = env_float("ORCH_DOCKING_VY_SLEW", cfg.vy_slew_per_s)
    cfg.wz_slew_per_s = env_float("ORCH_DOCKING_WZ_SLEW", cfg.wz_slew_per_s)
    cfg.enable_lateral_control = env_bool("ORCH_DOCKING_ENABLE_LATERAL", cfg.enable_lateral_control)
    cfg.filter_window = env_int("ORCH_DOCKING_FILTER_WINDOW", cfg.filter_window)
    cfg.filter_ewma_alpha = env_float("ORCH_DOCKING_FILTER_ALPHA", cfg.filter_ewma_alpha)
    cfg.filter_reject_yaw_jump_rad = env_float("ORCH_DOCKING_FILTER_REJECT_YAW", cfg.filter_reject_yaw_jump_rad)
    cfg.filter_reject_dist_jump_m = env_float("ORCH_DOCKING_FILTER_REJECT_DIST", cfg.filter_reject_dist_jump_m)
    cfg.filter_reject_lateral_jump_m = env_float("ORCH_DOCKING_FILTER_REJECT_LAT", cfg.filter_reject_lateral_jump_m)
    return cfg


def apply_common_docking_overrides(cfg: Any, args: argparse.Namespace) -> Any:
    for attr, arg_name in [
        ("filter_ewma_alpha", "filter_alpha"),
        ("filter_window", "filter_window"),
        ("coarse_align_enter_rad", "coarse_enter"),
        ("coarse_align_exit_rad", "coarse_exit"),
        ("approach_max_vx_norm", "approach_vx"),
        ("approach_max_wz_norm", "approach_wz"),
        ("final_max_vx_norm", "final_vx"),
        ("final_max_wz_norm", "final_wz"),
        ("vx_slew_per_s", "vx_slew"),
        ("wz_slew_per_s", "wz_slew"),
        ("min_confidence", "min_confidence"),
    ]:
        if hasattr(args, arg_name):
            v = getattr(args, arg_name)
            if v is not None:
                setattr(cfg, attr, type(getattr(cfg, attr))(v) if not isinstance(getattr(cfg, attr), bool) else bool(v))
    return cfg


def add_common_docking_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--filter-alpha", type=float, default=None, help="Override ORCH_DOCKING_FILTER_ALPHA")
    parser.add_argument("--filter-window", type=int, default=None, help="Override ORCH_DOCKING_FILTER_WINDOW")
    parser.add_argument("--coarse-enter", type=float, default=None, help="Override coarse spin enter yaw rad")
    parser.add_argument("--coarse-exit", type=float, default=None, help="Override coarse spin exit yaw rad")
    parser.add_argument("--approach-vx", type=float, default=None, help="Override approach max vx norm")
    parser.add_argument("--approach-wz", type=float, default=None, help="Override approach max wz norm")
    parser.add_argument("--final-vx", type=float, default=None, help="Override final max vx norm")
    parser.add_argument("--final-wz", type=float, default=None, help="Override final max wz norm")
    parser.add_argument("--vx-slew", type=float, default=None, help="Override vx slew per second")
    parser.add_argument("--wz-slew", type=float, default=None, help="Override wz slew per second")
    parser.add_argument("--min-confidence", type=float, default=None, help="Override docking min confidence")


def command_to_row(cmd: Any, ts: float, idx: int, state: str, obs: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    d = asdict(cmd) if is_dataclass(cmd) else dict(cmd)
    row: Dict[str, Any] = {
        "idx": idx,
        "ts": ts,
        "state": state,
        "mode": d.get("mode"),
        "phase": d.get("phase", ""),
        "valid": d.get("valid", False),
        "pose_locked": d.get("pose_locked", False),
        "reason": d.get("reason", ""),
        "vx_norm": d.get("vx", 0.0),
        "vy_norm": d.get("vy", 0.0),
        "wz_norm": d.get("wz", 0.0),
        "raw_yaw_err_rad": d.get("raw_yaw_err_rad"),
        "raw_dist_err_m": d.get("raw_dist_err_m"),
        "filtered_yaw_err_rad": d.get("filtered_yaw_err_rad"),
        "filtered_dist_err_m": d.get("filtered_dist_err_m"),
    }
    if obs:
        row.update({
            "edge_found": obs.get("edge_found"),
            "confidence": obs.get("confidence"),
            "obs_yaw_err_rad": obs.get("yaw_err_rad"),
            "obs_dist_err_m": obs.get("dist_err_m"),
            "obs_lateral_err_m": obs.get("lateral_err_m"),
            "edge_ready": obs.get("edge_ready"),
            "depth_valid": obs.get("depth_valid"),
            "roi_source": obs.get("roi_source"),
        })
    return row


def basic_jitter_prediction(rows: Sequence[Mapping[str, Any]], *, source: str = "") -> Dict[str, Any]:
    """Heuristic risk estimator for visible vehicle/camera shake.

    This is not a dynamics model. It is a fast triage score that combines
    measurement jitter, command derivative, sign flips, and dropout rate.
    """
    if not rows:
        return {"source": source, "n": 0, "risk_score": 0, "risk_level": "NO_DATA", "warnings": ["no rows"]}

    times = normalized_time(rows)
    yaw = [get_path(r, "filtered_yaw_err_rad", "yaw_err_rad", "obs_yaw_err_rad") for r in rows]
    dist = [get_path(r, "filtered_dist_err_m", "dist_err_m", "obs_dist_err_m") for r in rows]
    vx = [get_path(r, "vx_norm", "vx") for r in rows]
    wz = [get_path(r, "wz_norm", "wz") for r in rows]
    edge_found = [to_bool(get_path(r, "edge_found", default=True), True) for r in rows]
    valid = [to_bool(get_path(r, "valid", default=True), True) for r in rows]

    duration = max(1e-6, times[-1] - times[0]) if len(times) > 1 else max(1e-6, len(rows) * 0.2)
    yaw_std = safe_stdev([x for x in yaw if to_float(x, None) is not None]) or 0.0
    yaw_mad = robust_mad([x for x in yaw if to_float(x, None) is not None]) or 0.0
    dist_std = safe_stdev([x for x in dist if to_float(x, None) is not None]) or 0.0
    dist_mad = robust_mad([x for x in dist if to_float(x, None) is not None]) or 0.0
    wz_rate = diff_rate(times, wz)
    vx_rate = diff_rate(times, vx)
    wz_flip_hz = sign_changes(wz, deadband=0.015) / duration
    vx_flip_hz = sign_changes(vx, deadband=0.015) / duration
    dropout_rate = 1.0 - (sum(1 for x in edge_found if x) / max(1, len(edge_found)))
    invalid_rate = 1.0 - (sum(1 for x in valid if x) / max(1, len(valid)))
    wz_rate_std = safe_stdev(wz_rate) or 0.0
    vx_rate_std = safe_stdev(vx_rate) or 0.0

    # Thresholds are deliberately interpretable for this table-edge task.
    terms = {
        "yaw_jitter": min(1.0, max(yaw_std, yaw_mad) / 0.035),
        "dist_jitter": min(1.0, max(dist_std, dist_mad) / 0.025),
        "wz_flip": min(1.0, wz_flip_hz / 0.80),
        "vx_flip": min(1.0, vx_flip_hz / 0.50),
        "wz_rate": min(1.0, wz_rate_std / 0.45),
        "vx_rate": min(1.0, vx_rate_std / 0.30),
        "dropout": min(1.0, max(dropout_rate, invalid_rate) / 0.20),
    }
    weights = {
        "yaw_jitter": 0.22,
        "dist_jitter": 0.16,
        "wz_flip": 0.17,
        "vx_flip": 0.10,
        "wz_rate": 0.17,
        "vx_rate": 0.08,
        "dropout": 0.10,
    }
    score = round(100.0 * sum(terms[k] * weights[k] for k in weights), 1)
    if score < 25:
        level = "LOW"
    elif score < 50:
        level = "MEDIUM"
    elif score < 75:
        level = "HIGH"
    else:
        level = "VERY_HIGH"

    warnings: List[str] = []
    recommendations: List[str] = []
    if terms["yaw_jitter"] > 0.65:
        warnings.append("yaw observation jitter is high")
        recommendations.append("先检查相机刚性固定；再降低 ORCH_DOCKING_FILTER_ALPHA 或增大 filter_window")
    if terms["dist_jitter"] > 0.65:
        warnings.append("distance observation jitter is high")
        recommendations.append("检查 depth ROI 是否跳动；优先 replay 同一段日志比较 filter_alpha")
    if terms["wz_flip"] > 0.45:
        warnings.append("wz command sign flips frequently")
        recommendations.append("增大 COARSE_ENTER/EXIT 滞回间隔，或降低 yaw PID kd/min_abs_output")
    if terms["wz_rate"] > 0.60 or terms["vx_rate"] > 0.60:
        warnings.append("command derivative is large; visible jerk is likely")
        recommendations.append("降低速度上限或降低 ORCH_DOCKING_VX_SLEW / ORCH_DOCKING_WZ_SLEW")
    if terms["dropout"] > 0.45:
        warnings.append("observation dropout/invalid rate is high")
        recommendations.append("先修视觉稳定性/ROI；不要直接加大 PID")

    return {
        "source": source,
        "n": len(rows),
        "duration_s": duration,
        "risk_score": score,
        "risk_level": level,
        "terms": terms,
        "metrics": {
            "yaw_std_rad": yaw_std,
            "yaw_mad_std_rad": yaw_mad,
            "dist_std_m": dist_std,
            "dist_mad_std_m": dist_mad,
            "wz_flip_hz": wz_flip_hz,
            "vx_flip_hz": vx_flip_hz,
            "wz_rate_std_per_s": wz_rate_std,
            "vx_rate_std_per_s": vx_rate_std,
            "dropout_rate": dropout_rate,
            "invalid_rate": invalid_rate,
        },
        "warnings": warnings,
        "recommendations": list(dict.fromkeys(recommendations)),
    }
