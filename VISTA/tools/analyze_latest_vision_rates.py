#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, List, Optional


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _pick(row: dict, *paths: str) -> Optional[float]:
    for path in paths:
        cur: Any = row
        for key in path.split("."):
            if not isinstance(cur, dict) or key not in cur:
                cur = None
                break
            cur = cur[key]
        if isinstance(cur, (int, float)) and math.isfinite(float(cur)):
            return float(cur)
    return None


def _values(rows: Iterable[dict], *paths: str) -> List[float]:
    out: List[float] = []
    for row in rows:
        val = _pick(row, *paths)
        if val is not None:
            out.append(val)
    return out


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    arr = sorted(values)
    pos = (len(arr) - 1) * pct
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return arr[lo]
    return arr[lo] * (hi - pos) + arr[hi] * (pos - lo)


def _hz_from_rows(rows: List[dict], ts_path: str = "ts") -> float:
    ts = _values(rows, ts_path)
    if len(ts) < 2:
        return 0.0
    duration = max(ts) - min(ts)
    return float(len(ts) / duration) if duration > 0 else 0.0


def _summary(name: str, values: List[float]) -> str:
    if not values:
        return f"{name}: n=0"
    return (
        f"{name}: n={len(values)} mean={statistics.mean(values):.1f} "
        f"p50={_percentile(values, 0.50):.1f} "
        f"p90={_percentile(values, 0.90):.1f} "
        f"p95={_percentile(values, 0.95):.1f} "
        f"max={max(values):.1f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize latest vision/orchestrator rate logs.")
    parser.add_argument("run_dir", nargs="?", default="logs/runs/latest")
    args = parser.parse_args()

    root = Path(args.run_dir)
    perf = _load_jsonl(root / "vision" / "perf_timing.jsonl")
    orch_edge = _load_jsonl(root / "orchestrator" / "table_edge_obs.jsonl")
    orch_vision = _load_jsonl(root / "orchestrator" / "vision_obs.jsonl")

    active_rows = orch_edge or perf
    processed = _values(active_rows, "processed_frame_count")
    dropped = _values(active_rows, "dropped_frame_count")
    duration_ts = _values(active_rows, "ts")
    duration = (max(duration_ts) - min(duration_ts)) if len(duration_ts) >= 2 else 0.0
    processed_hz = ((processed[-1] - processed[0]) / duration) if processed and duration > 0 else _hz_from_rows(perf)
    dropped_hz = ((dropped[-1] - dropped[0]) / duration) if dropped and duration > 0 else 0.0

    print(f"run_dir: {root}")
    print(f"internal processed Hz: {processed_hz:.2f}")
    print(f"internal dropped Hz: {dropped_hz:.2f}")
    print(f"obs sent Hz: {_hz_from_rows(orch_vision):.2f}")
    print(f"orchestrator received Hz: {_hz_from_rows(orch_edge):.2f}")
    print(_summary("vision_process_ms", _values(orch_edge, "vision_process_ms", "table_edge.process_ms")))
    print(_summary("plane_fit_ms", _values(perf, "table_edge.profile_ms.plane_fit_ms")))
    print(_summary("fast_front_edge_ms", _values(perf, "table_edge.profile_ms.fast_front_edge_ms")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
