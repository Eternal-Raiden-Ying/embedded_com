#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unified chassis stop semantics for control-layer callers."""

from typing import Any, Dict, Optional

from ...bridge.simple_car_protocol import encode_emergency_stop, encode_soft_stop, encode_vel


def emergency_stop_line() -> str:
    return encode_emergency_stop() + "\r\n"


def soft_stop_line() -> str:
    return encode_soft_stop() + "\r\n"


def zero_velocity_line() -> str:
    return encode_vel(0.0, 0.0, 0.0) + "\r\n"


def emergency_stop(uart: Any, tx_meta: Optional[Dict[str, Any]] = None) -> bool:
    """Hard STOP: clear queues and synchronously write STOP through the bridge."""
    return bool(uart.send_emergency_stop(tx_meta=tx_meta))


def soft_stop(uart: Any, tx_meta: Optional[Dict[str, Any]] = None) -> bool:
    """Soft STOP: queued SSTOP, used only for normal completion."""
    return bool(uart.send_soft_stop(tx_meta=tx_meta))


def zero_velocity(uart: Any, tx_meta: Optional[Dict[str, Any]] = None) -> bool:
    """Zero-speed hold command for control loops that should remain active."""
    return bool(uart.send_velocity(0.0, 0.0, 0.0, tx_meta=tx_meta))
