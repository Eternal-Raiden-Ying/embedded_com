#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Human-readable effective configuration dump."""

import sys
from typing import Any, Iterable, List, Optional


def _is_root_config(config: Any) -> bool:
    return hasattr(config, "orchestrator") and hasattr(config, "vision")


def _orchestrator_config(config: Any) -> Any:
    return config.orchestrator if _is_root_config(config) else config


def _vision_config(config: Any) -> Optional[Any]:
    return config.vision if _is_root_config(config) else None


def _loaded_files(config: Any, orch: Any) -> List[str]:
    loaded = []
    if _is_root_config(config):
        loaded.extend(getattr(config.vision.runtime, "loaded_config_files", []) or [])
    loaded.extend(getattr(orch.runtime, "loaded_config_files", []) or [])
    deduped = []
    for item in loaded:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _fmt_loaded(items: Iterable[str]) -> str:
    values = list(items)
    return ", ".join(values) if values else "<schema defaults only>"


def format_effective_config(config: Any, effective_dry_run: Optional[bool] = None) -> str:
    """Return a stable effective configuration dump string."""
    orch = _orchestrator_config(config)
    vision = _vision_config(config)
    runtime = orch.runtime
    serial = orch.serial
    control = orch.control
    car = orch.car

    profile = (
        getattr(config, "profile", "")
        or getattr(runtime, "config_profile", "")
        or "<unset>"
    )
    dry_run = bool(serial.dry_run if effective_dry_run is None else effective_dry_run)
    uart_port = "fake" if dry_run else str(serial.port)
    vision_control_interval = "<unavailable>"
    if vision is not None:
        send_hz = _safe_float(getattr(vision.runtime, "track_local_send_hz", 0.0))
        if send_hz > 0.0:
            vision_control_interval = f"{1.0 / send_hz:.3f}s"

    lines = [
        "=" * 80,
        "  EFFECTIVE RUNTIME CONFIGURATION",
        "=" * 80,
        f"  config profile       : {profile}",
        f"  loaded config files  : {_fmt_loaded(_loaded_files(config, orch))}",
        f"  UART port / dry_run  : {uart_port} / {int(dry_run)}",
        f"  tick_hz              : {_safe_float(runtime.tick_hz):.2f}",
        f"  near_stop_depth_m    : {_safe_float(getattr(control, 'near_stop_depth_m', 0.0)):.3f}",
        f"  edge_slide_vy_mps    : {_safe_float(car.edge_slide_vy_mps):.3f}",
        f"  edge_slide_max_vx_mps: {_safe_float(car.edge_slide_max_vx_mps):.3f}",
        (
            "  final_lock yaw/dist/lateral: "
            f"{_safe_float(control.final_lock_yaw_tol_rad):.4f} / "
            f"{_safe_float(control.final_lock_dist_tol_m):.4f} / "
            f"{_safe_float(control.final_lock_lateral_tol_m):.4f}"
        ),
        (
            "  STOP/SSTOP policy   : "
            f"{getattr(car, 'stop_policy', 'STOP=emergency,SSTOP=soft')}"
        ),
        f"  vision control_obs interval: {vision_control_interval}",
        "=" * 80,
    ]
    return "\n".join(lines)


def print_effective_config(config: Any, effective_dry_run: Optional[bool] = None, stream=None) -> None:
    """Print effective configuration and flush the target stream."""
    target = stream or sys.stdout
    print("\n" + format_effective_config(config, effective_dry_run=effective_dry_run) + "\n", file=target)
    try:
        target.flush()
    except Exception:
        pass
