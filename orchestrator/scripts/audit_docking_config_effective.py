#!/usr/bin/env python3
"""Audit effective table-docking control configuration."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Tuple


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

    _record(values, "speed", "search_wz_radps", car, "search_table_wz_radps", 0.20)
    _record(values, "speed", "min_forward_vx_mps", control, "min_forward_vx_mps", 0.04)
    _record(values, "speed", "bbox_track_forward_vx_mps", control, "bbox_track_forward_vx_mps", 0.10)
    _record(values, "speed", "bbox_track_forward_max_vx_mps", control, "bbox_track_forward_max_vx_mps", 0.20)
    _record(values, "speed", "far_bbox_track_vx_mps", control, "far_bbox_track_vx_mps", 0.20)
    _record(values, "speed", "bbox_track_forward_max_wz_radps", control, "bbox_track_forward_max_wz_radps", 0.20)
    _record(values, "speed", "edge_handoff_forward_vx_mps", control, "edge_handoff_forward_vx_mps", 0.08)
    _record(values, "speed", "near_slow_max_vx_mps", control, "near_slow_max_vx_mps", 0.03)
    _record(values, "speed", "near_slow_max_wz_radps", control, "near_slow_max_wz_radps", 0.04)
    _record(values, "speed", "final_servo_enter_p10_m", control, "final_servo_enter_p10_m", 0.45)
    _record(values, "speed", "edge_final_enter_margin_m", control, "edge_final_enter_margin_m", 0.05)
    _record(values, "speed", "edge_final_stop_margin_m", control, "edge_final_stop_margin_m", 0.02)
    _record(values, "speed", "close_range_enter_p10_m", control, "close_range_enter_p10_m", 0.55)
    _record(values, "speed", "close_range_probe_vx_mps", control, "close_range_probe_vx_mps", 0.004)
    _record(values, "speed", "close_range_missing_probe_vx_mps", control, "close_range_missing_probe_vx_mps", 0.002)
    _record(values, "speed", "roi_final_stop_p10_m", control, "roi_final_stop_p10_m", 0.42)
    _record(values, "speed", "roi_final_slow_p10_m", control, "roi_final_slow_p10_m", 0.52)
    _record(values, "speed", "roi_final_probe_vx_mps", control, "roi_final_probe_vx_mps", 0.004)
    _record(values, "speed", "roi_final_missing_probe_vx_mps", control, "roi_final_missing_probe_vx_mps", 0.002)
    _record(values, "speed", "roi_final_missing_hold_s", control, "roi_final_missing_hold_s", 0.8)
    _record(values, "speed", "depth_envelope_stop_p10_m", control, "depth_envelope_stop_p10_m", 0.35)
    _record(values, "speed", "depth_envelope_slow_p10_m", control, "depth_envelope_slow_p10_m", 0.50)
    _record(values, "speed", "depth_envelope_mid_p10_m", control, "depth_envelope_mid_p10_m", 0.70)
    _record(values, "speed", "depth_envelope_slow_vx_mps", control, "depth_envelope_slow_vx_mps", 0.006)
    _record(values, "speed", "depth_envelope_mid_vx_mps", control, "depth_envelope_mid_vx_mps", 0.015)
    _record(values, "speed", "global_max_vx_mps", car, "max_vx_mps", 1.0)
    _record(values, "speed", "global_max_vy_mps", car, "max_vy_mps", 1.0)
    _record(values, "speed", "global_max_wz_radps", car, "max_wz_radps", 1.0)

    for key, fallback in (
        ("table_target_dist_m", 0.30),
        ("final_dist_deadband_m", 0.04),
        ("final_dist_kp", 0.08),
        ("final_forward_vx_max_mps", 0.006),
        ("final_reverse_vx_max_mps", 0.004),
        ("final_reverse_confirm_frames", 3),
        ("final_yaw_deadband_rad", 0.12),
        ("final_lock_yaw_rad", 0.12),
        ("final_yaw_realign_rad", 0.18),
        ("final_yaw_stable_frames", 6),
        ("final_yaw_align_min_duration_ms", 1000),
        ("final_yaw_last_good_hold_s", 1.2),
    ):
        _record(values, "final", key, control, key, fallback)

    _record(values, "lateral", "distance_scaled_lateral_enabled", control, "distance_scaled_lateral_enabled", True)
    _record(values, "lateral", "near_slow_max_vy_mps", control, "near_slow_max_vy_mps", 0.030)
    _record(values, "lateral", "lateral_enabled", control, "lateral_enabled", False)
    _record(values, "lateral", "lateral_vy_max_mps", control, "lateral_vy_max_mps", 0.15)
    _record(values, "lateral", "lateral_kp", control, "lateral_kp", 0.10)
    _record(values, "lateral", "lateral_deadband_norm", control, "lateral_deadband_norm", 0.025)
    _record(values, "lateral", "lateral_distance_ref_m", control, "lateral_distance_ref_m", 0.80)
    _record(values, "lateral", "lateral_distance_scale_min", control, "lateral_distance_scale_min", 1.0)
    _record(values, "lateral", "lateral_distance_scale_max", control, "lateral_distance_scale_max", 4.0)
    _record(values, "lateral", "far_lateral_vy_max_mps", control, "far_lateral_vy_max_mps", 0.15)
    _record(values, "lateral", "mid_lateral_vy_max_mps", control, "mid_lateral_vy_max_mps", 0.08)
    _record(values, "lateral", "near_lateral_vy_max_mps", control, "near_lateral_vy_max_mps", 0.030)
    _record(values, "lateral", "lateral_priority_mid_error_norm", control, "lateral_priority_mid_error_norm", 0.10)
    _record(values, "lateral", "lateral_priority_large_error_norm", control, "lateral_priority_large_error_norm", 0.18)
    _record(values, "lateral", "lateral_priority_mid_vx_cap_mps", control, "lateral_priority_mid_vx_cap_mps", 0.080)
    _record(values, "lateral", "lateral_priority_vx_cap_mps", control, "lateral_priority_vx_cap_mps", 0.040)

    for key, fallback in (
        ("yaw_flip_hold_window_s", 0.8),
        ("yaw_flip_count_limit", 2),
        ("yaw_ambiguous_wz_cap", 0.0),
        ("yaw_ambiguous_vy_boost", 1.5),
    ):
        _record(values, "yaw_anti_oscillation", key, control, key, fallback)

    warnings: list[str] = []
    lateral_enabled = bool(values["lateral"]["lateral_enabled"]["value"])
    near_vy = float(values["lateral"]["near_slow_max_vy_mps"]["value"] or 0.0)
    if near_vy > 0.0 and not lateral_enabled:
        warnings.append("near_slow_max_vy_mps > 0 but lateral_enabled is false")
    if float(values["speed"]["global_max_vx_mps"]["value"] or 0.0) < float(values["speed"]["bbox_track_forward_max_vx_mps"]["value"] or 0.0):
        warnings.append("global max_vx_mps is below bbox_track_forward_max_vx_mps")
    if float(values["speed"]["global_max_vy_mps"]["value"] or 0.0) < float(values["lateral"]["far_lateral_vy_max_mps"]["value"] or 0.0):
        warnings.append("global max_vy_mps is below far_lateral_vy_max_mps")
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
    for group in ("speed", "final", "lateral", "yaw_anti_oscillation"):
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
