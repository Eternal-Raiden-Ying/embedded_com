#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation rules for effective runtime configuration."""

import os
import platform
import sys
from typing import Any, Iterable, Tuple

from .schema import SystemGlobalConfig


def _is_test_process() -> bool:
    return (
        "pytest" in sys.modules
        or "unittest" in sys.modules
        or "PYTEST_CURRENT_TEST" in os.environ
        or any("pytest" in arg or "unittest" in arg for arg in sys.argv)
    )


def _is_main_app() -> bool:
    if not sys.argv:
        return False
    return (
        sys.argv[0].endswith("app/main.py")
        or sys.argv[0].endswith("app\\main.py")
        or "app.main" in sys.argv
        or "orchestrator_service.app.main" in sys.argv
    )


def _endpoint_items(config: SystemGlobalConfig) -> Iterable[Tuple[str, Any]]:
    orch = config.orchestrator
    for name in ("task_cmd_in", "task_ack_out", "vision_obs_in", "vision_req_out", "tts_event_out"):
        yield f"orchestrator.{name}", getattr(orch, name)

    vision = config.vision
    for name in ("req_in", "obs_out"):
        yield f"vision.{name}", getattr(vision, name)

    gateway = config.gateway
    for name in ("command_in", "status_out", "orchestrator_task_cmd_out", "orchestrator_task_ack_in"):
        yield f"gateway.{name}", getattr(gateway, name)

    yield "online_edge.output", config.online_edge.output


def _print_warnings(warnings) -> None:
    if not warnings:
        return
    print("\n" + "!" * 80, file=sys.stderr)
    print("  CONFIG WARNINGS DETECTED", file=sys.stderr)
    print("!" * 80, file=sys.stderr)
    for warning in warnings:
        print(f"  {warning}", file=sys.stderr)
    print("!" * 80 + "\n", file=sys.stderr)


def validate_config(config: SystemGlobalConfig, force_production: bool = False) -> None:
    """Validate the final effective config before runtime use."""
    warnings = []
    car = config.orchestrator.car
    serial = config.orchestrator.serial
    dry_run = bool(serial.dry_run)
    in_test = _is_test_process()
    production_path = force_production or (_is_main_app() and not in_test)

    if float(car.edge_slide_vy_mps) == 0.14:
        msg = (
            "edge_slide_vy_mps is still 0.14 m/s. This is a schema fallback, "
            "not an accepted runtime value; set the canonical orchestrator.car.edge_slide_vy_mps."
        )
        if production_path or not dry_run:
            raise ValueError(f"CRITICAL CONFIG ERROR: {msg}")
        warnings.append(msg)

    emergency_cmd = str(getattr(car, "emergency_stop_command", "STOP") or "").strip().upper()
    soft_cmd = str(getattr(car, "soft_stop_command", "SSTOP") or "").strip().upper()
    if emergency_cmd != "STOP":
        raise ValueError(f"Invalid STOP policy: emergency_stop_command must be STOP, got {emergency_cmd!r}")
    if soft_cmd != "SSTOP":
        raise ValueError(f"Invalid STOP policy: soft_stop_command must be SSTOP, got {soft_cmd!r}")
    if emergency_cmd == soft_cmd:
        raise ValueError("Invalid STOP policy: STOP and SSTOP commands must remain distinct")

    if platform.system().lower().startswith("win"):
        uds_endpoints = [
            name for name, endpoint in _endpoint_items(config)
            if str(getattr(endpoint, "transport", "") or "").lower() == "uds"
        ]
        if uds_endpoints:
            msg = "Windows profile cannot use UDS transport: " + ", ".join(uds_endpoints)
            if production_path or not dry_run:
                raise ValueError(msg)
            warnings.append(msg)

    uart_msg = f"UART serial mode: dry_run={int(dry_run)} port={serial.port}"
    if dry_run:
        warnings.append(uart_msg)
    elif not str(serial.port or "").strip():
        raise ValueError("Serial port must be explicit when dry_run is false")

    _print_warnings(warnings)
