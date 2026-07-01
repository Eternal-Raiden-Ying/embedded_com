#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Conversion from grasp server result to arm POSE parameters."""

import math
from typing import Any

from ..ipc.protocol import ArmCommand


def normalize_roll(roll_deg: float) -> int:
    """Normalize roll angle to [-90, 90] degrees for a two-finger gripper."""
    roll = round(float(roll_deg)) % 180
    if roll > 90:
        roll = roll - 180
    return roll


def width_to_claw_angle(width_cm: float) -> int:
    """Convert gripper width (cm) to claw angle (degrees).

    Placeholder implementation — maps width to angle linearly.
    Replace with real STM32 lookup table when available.
    """
    return max(0, min(90, int(round(float(width_cm) * 10.0))))


def _required_float(grasp: dict, key: str) -> float:
    if key not in grasp or grasp.get(key) is None:
        raise ValueError(f"missing grasp pose field: {key}")
    try:
        value = float(grasp.get(key))
    except Exception as exc:
        raise ValueError(f"invalid grasp pose field: {key}") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite grasp pose field: {key}")
    return value


def _clamp_time_ms(value: Any) -> int:
    try:
        time_ms = int(round(float(value)))
    except Exception:
        time_ms = 800
    return max(100, min(5000, time_ms))


def grasp_to_pose_params(grasp: dict, time_ms: int = 800) -> ArmCommand:
    """Convert a grasp result dict from vision_obs to an ArmCommand.

    Args:
        grasp: Dict with x_cm, y_cm, z_cm, pitch_deg, roll_deg,
               gripper_width_cm, approach_depth_cm fields.
        time_ms: Arm movement time in milliseconds (default 500).

    Returns:
        ArmCommand ready for POSE encoding.
    """
    if not isinstance(grasp, dict):
        raise ValueError("grasp pose must be a dict")
    x_cm = _required_float(grasp, "x_cm")
    y_cm = _required_float(grasp, "y_cm")
    z_cm = _required_float(grasp, "z_cm")
    pitch_deg = _required_float(grasp, "pitch_deg")
    roll_deg = _required_float(grasp, "roll_deg")
    width_cm = _required_float(grasp, "gripper_width_cm")
    claw = width_to_claw_angle(width_cm)
    return ArmCommand(
        x_cm=x_cm,
        y_cm=y_cm,
        z_cm=z_cm,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        claw_deg=claw,
        time_ms=_clamp_time_ms(time_ms),
    )
