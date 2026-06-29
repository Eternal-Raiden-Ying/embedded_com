#!/usr/bin/env python3
"""Audit effective table-docking control configuration."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from common.config import load_global_config  # noqa: E402
from common.config.loader import load_yaml_file  # noqa: E402
from common.config.schema import SystemGlobalConfig  # noqa: E402


def _get(obj: Any, name: str, fallback: Any) -> Tuple[Any, bool]:
    if hasattr(obj, name):
        value = getattr(obj, name)
        if value is not None:
            return value, False
    return fallback, True


def _dict_has_path(data: Dict[str, Any], path: tuple[str, ...]) -> bool:
    cur: Any = data
    for item in path:
        if not isinstance(cur, dict) or item not in cur:
            return False
        cur = cur[item]
    return True


def _schema_default(path: tuple[str, ...], fallback: Any) -> Any:
    cur: Any = SystemGlobalConfig()
    for item in path:
        if not hasattr(cur, item):
            return fallback
        cur = getattr(cur, item)
    return cur


def _source_for(system_yaml: Dict[str, Any], path: tuple[str, ...], fallback: Any) -> str:
    if _dict_has_path(system_yaml, path):
        return "configs/system_config.yaml"
    _schema_default(path, fallback)
    return "common/config/schema.py"


def _record(
    out: Dict[str, Dict[str, Any]],
    group: str,
    key: str,
    obj: Any,
    attr: str,
    fallback: Any,
    *,
    source: str = "",
) -> None:
    value, used_fallback = _get(obj, attr, fallback)
    out.setdefault(group, {})[key] = {"value": value, "fallback": used_fallback, "source": source or "unknown"}


def _record_control(
    out: Dict[str, Dict[str, Any]],
    system_yaml: Dict[str, Any],
    group: str,
    key: str,
    control: Any,
    fallback: Any,
) -> None:
    path = ("orchestrator", "control", key)
    _record(out, group, key, control, key, fallback, source=_source_for(system_yaml, path, fallback))


def _record_car(
    out: Dict[str, Dict[str, Any]],
    system_yaml: Dict[str, Any],
    group: str,
    key: str,
    car: Any,
    attr: str,
    fallback: Any,
) -> None:
    path = ("orchestrator", "car", attr)
    _record(out, group, key, car, attr, fallback, source=_source_for(system_yaml, path, fallback))


def collect() -> tuple[Dict[str, Dict[str, Any]], list[str]]:
    cfg = load_global_config()
    orch = cfg.orchestrator
    control = orch.control
    car = orch.car
    system_path = Path(os.getenv("SYSTEM_CONFIG_FILE") or os.path.join(ROOT, "configs", "system_config.yaml"))
    system_yaml = load_yaml_file(system_path)
    values: Dict[str, Dict[str, Any]] = {}
    values["config_chain"] = {
        "system_config_file": {"value": str(system_path), "fallback": False, "source": "SYSTEM_CONFIG_FILE" if os.getenv("SYSTEM_CONFIG_FILE") else "default"},
        "vision_params_file": {"value": getattr(cfg.vision.runtime, "vision_params_file", ""), "fallback": False, "source": "vision.runtime.vision_params_file"},
        "loaded_config_files": {"value": list(getattr(orch.runtime, "loaded_config_files", [])), "fallback": False, "source": "loader"},
    }

    _record_car(values, system_yaml, "speed", "search_wz_radps", car, "search_table_wz_radps", 0.20)
    for key, fallback in (
        ("min_forward_vx_mps", 0.04),
        ("bbox_track_forward_vx_mps", 0.10),
        ("bbox_track_forward_max_vx_mps", 0.20),
        ("far_bbox_track_vx_mps", 0.20),
        ("bbox_track_forward_max_wz_radps", 0.20),
        ("edge_handoff_forward_vx_mps", 0.08),
        ("near_slow_max_vx_mps", 0.03),
        ("near_slow_max_wz_radps", 0.04),
        ("final_servo_enter_p10_m", 0.45),
        ("edge_final_enter_margin_m", 0.06),
        ("edge_final_stop_margin_m", 0.02),
        ("close_range_enter_p10_m", 0.55),
        ("final_probe_vx_mps", 0.008),
        ("final_missing_probe_vx_mps", 0.004),
        ("close_range_probe_vx_mps", 0.008),
        ("close_range_missing_probe_vx_mps", 0.004),
        ("roi_final_stop_p10_m", 0.42),
        ("roi_final_slow_p10_m", 0.52),
        ("roi_final_probe_vx_mps", 0.008),
        ("roi_final_missing_probe_vx_mps", 0.004),
        ("roi_final_missing_hold_s", 0.8),
        ("depth_envelope_stop_p10_m", 0.35),
        ("depth_envelope_slow_p10_m", 0.50),
        ("depth_envelope_mid_p10_m", 0.70),
        ("depth_envelope_slow_vx_mps", 0.006),
        ("depth_envelope_mid_vx_mps", 0.015),
    ):
        _record_control(values, system_yaml, "speed", key, control, fallback)
    _record_car(values, system_yaml, "speed", "global_max_vx_mps", car, "max_vx_mps", 1.0)
    _record_car(values, system_yaml, "speed", "global_max_vy_mps", car, "max_vy_mps", 1.0)
    _record_car(values, system_yaml, "speed", "global_max_wz_radps", car, "max_wz_radps", 1.0)

    for key, fallback in (
        ("table_target_dist_m", 0.30),
        ("final_dist_deadband_m", 0.03),
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
        _record_control(values, system_yaml, "final", key, control, fallback)

    for key, fallback in (
        ("distance_scaled_lateral_enabled", True),
        ("near_slow_max_vy_mps", 0.040),
        ("lateral_enabled", True),
        ("lateral_vy_max_mps", 0.18),
        ("lateral_kp", 0.30),
        ("lateral_deadband_norm", 0.020),
        ("lateral_distance_ref_m", 0.50),
        ("lateral_distance_scale_min", 0.80),
        ("lateral_distance_scale_max", 2.0),
        ("far_lateral_vy_max_mps", 0.18),
        ("mid_lateral_vy_max_mps", 0.14),
        ("near_lateral_vy_max_mps", 0.060),
        ("lateral_priority_mid_error_norm", 0.99),
        ("lateral_priority_large_error_norm", 0.99),
        ("lateral_priority_mid_vx_cap_mps", 0.080),
        ("lateral_priority_vx_cap_mps", 0.040),
        ("edge_yaw_align_allow_lateral", True),
        ("edge_yaw_align_lateral_vy_max_mps", 0.080),
    ):
        _record_control(values, system_yaml, "lateral", key, control, fallback)

    for key, fallback in (
        ("yaw_flip_hold_window_s", 0.8),
        ("yaw_flip_count_limit", 2),
        ("yaw_ambiguous_wz_cap", 0.0),
        ("yaw_ambiguous_vy_boost", 1.5),
    ):
        _record_control(values, system_yaml, "yaw_anti_oscillation", key, control, fallback)

    for key, fallback in (
        ("edge_yaw_control_enter_rad", 0.30),
        ("edge_yaw_control_exit_rad", 0.12),
        ("edge_yaw_reject_rad", 1.40),
        ("edge_yaw_kp", 0.22),
        ("edge_yaw_min_wz_radps", 0.08),
        ("edge_yaw_max_wz_radps", 0.18),
    ):
        _record_control(values, system_yaml, "edge_yaw_control", key, control, fallback)
    _record_car(values, system_yaml, "edge_yaw_control", "edge_hard_rotate_only_yaw_rad", car, "table_edge_hard_rotate_only_yaw_rad", 1.40)

    # Bilateral distance config check
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    vision_params_path = os.path.join(repo_root, "VISTA", "configs", "vision_params.yaml")
    vista_dist = None
    if os.path.exists(vision_params_path):
        try:
            import yaml
            with open(vision_params_path, "r", encoding="utf-8") as f:
                vista_data = yaml.safe_load(f) or {}
            te = vista_data.get("table_edge", {})
            vista_dist = te.get("table_target_dist_m") or te.get("target_dist_m")
        except Exception:
            pass

    orch_dist = getattr(control, "table_target_dist_m", 0.30)
    values.setdefault("bilateral_distance", {})["vista_target_dist_m"] = {
        "value": vista_dist if vista_dist is not None else 0.30,
        "fallback": vista_dist is None,
        "source": "VISTA/configs/vision_params.yaml" if vista_dist is not None else "common/config/schema.py",
    }
    values.setdefault("bilateral_distance", {})["orch_table_target_dist_m"] = {
        "value": orch_dist,
        "fallback": False,
        "source": values.get("final", {}).get("table_target_dist_m", {}).get("source", "unknown"),
    }
    edge_override = getattr(getattr(cfg.online_edge, "detector", None), "target_dist_m_override", None)
    values.setdefault("bilateral_distance", {})["online_edge_target_dist_m_override"] = {
        "value": edge_override,
        "fallback": False,
        "source": "configs/system_config.yaml" if _dict_has_path(system_yaml, ("online_edge", "detector", "target_dist_m_override")) else "common/config/schema.py",
    }

    warnings: list[str] = []
    if vista_dist is not None and abs(float(vista_dist) - float(orch_dist)) > 1e-4:
        warnings.append(f"bilateral target_dist mismatch: vista_target_dist_m={vista_dist} vs orch_table_target_dist_m={orch_dist}")

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
        source = item.get("source", "unknown")
        print(f"  {key}: {item['value']}{suffix}  source={source}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="also print a JSON dump after the human-readable audit")
    args = parser.parse_args()

    values, warnings = collect()
    print("Docking effective config audit")
    for group in ("config_chain", "speed", "final", "lateral", "yaw_anti_oscillation", "edge_yaw_control", "bilateral_distance"):
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
