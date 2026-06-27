#!/usr/bin/env python3
"""Audit effective table-docking control configuration."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, Iterable, Tuple


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from common.config import load_global_config  # noqa: E402


def _get(obj: Any, name: str, fallback: Any) -> Tuple[Any, bool]:
    if hasattr(obj, name):
        value = getattr(obj, name)
        if value is not None:
            return value, False
    return fallback, True


def _record(out: Dict[str, Dict[str, Any]], group: str, key: str, obj: Any, attr: str, fallback: Any) -> None:
    value, used_fallback = _get(obj, attr, fallback)
    out.setdefault(group, {})[key] = {"value": value, "fallback": used_fallback}


def collect() -> tuple[Dict[str, Dict[str, Any]], list[str]]:
    cfg = load_global_config()
    orch = cfg.orchestrator
    control = orch.control
    car = orch.car
    values: Dict[str, Dict[str, Any]] = {}

    _record(values, "speed", "search_wz_radps", car, "search_table_wz_radps", 0.10)
    _record(values, "speed", "bbox_reacquire_wz_max_radps", car, "yolo_table_max_wz_radps", 0.06)
    _record(values, "speed", "bbox_track_forward_vx_mps", control, "bbox_track_forward_vx_mps", 0.012)
    _record(values, "speed", "bbox_track_forward_max_vx_mps", control, "bbox_track_forward_max_vx_mps", 0.015)
    _record(values, "speed", "edge_approach_vx_mps", car, "table_approach_safe_vx_mps", 0.020)
    _record(values, "speed", "near_slow_max_vx_mps", control, "near_slow_max_vx_mps", 0.020)
    _record(values, "speed", "near_slow_max_wz_radps", control, "near_slow_max_wz_radps", 0.04)

    for key, fallback in (
        ("final_yaw_deadband_rad", 0.12),
        ("final_lock_yaw_rad", 0.12),
        ("final_yaw_realign_rad", 0.18),
        ("final_yaw_stable_frames", 6),
        ("final_yaw_align_min_duration_ms", 1000),
        ("final_yaw_last_good_hold_s", 1.2),
    ):
        _record(values, "final", key, control, key, fallback)

    for key, fallback in (
        ("edge_readiness_enabled", True),
        ("edge_readiness_enter_score", 0.65),
        ("edge_readiness_exit_score", 0.35),
        ("edge_handoff_min_hold_ms", 800),
    ):
        _record(values, "handoff", key, control, key, fallback)

    for key, fallback in (
        ("progress_window_ms", 15000.0),
        ("min_progress_m", 0.010),
        ("multi_table_enabled", False),
    ):
        _record(values, "progress", key, control, key, fallback)

    _record(values, "lateral", "allow_vy", car, "table_approach_allow_vy", False)
    _record(values, "lateral", "near_slow_max_vy_mps", control, "near_slow_max_vy_mps", 0.0)
    _record(values, "lateral", "lateral_enabled", control, "lateral_enabled", False)
    _record(values, "lateral", "lateral_owner_default", control, "lateral_owner_default", "none")

    warnings: list[str] = []
    lateral_enabled = bool(values["lateral"]["lateral_enabled"]["value"])
    near_vy = float(values["lateral"]["near_slow_max_vy_mps"]["value"] or 0.0)
    allow_vy = bool(values["lateral"]["allow_vy"]["value"])
    if near_vy > 0.0 and not lateral_enabled:
        warnings.append("near_slow_max_vy_mps > 0 but lateral_enabled is false")
    if allow_vy and not lateral_enabled:
        warnings.append("allow_vy is true but lateral_enabled is false")
    return values, warnings


def _print_group(name: str, rows: Dict[str, Any]) -> None:
    print(f"\n{name}:")
    for key, item in rows.items():
        suffix = " [fallback]" if item["fallback"] else ""
        print(f"  {key}: {item['value']}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="also print a JSON dump after the human-readable audit")
    args = parser.parse_args()

    values, warnings = collect()
    print("Docking effective config audit")
    for group in ("speed", "final", "handoff", "progress", "lateral"):
        _print_group(group, values.get(group, {}))
    if warnings:
        print("\nwarnings:")
        for warning in warnings:
            print(f"  WARNING: {warning}")
    else:
        print("\nwarnings: none")
    if args.json:
        print("\njson:")
        print(json.dumps({"values": values, "warnings": warnings}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
