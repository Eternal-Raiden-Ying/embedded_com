#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Conversion from grasp server result to STM32 POSE parameters."""

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
    return max(0, int(float(width_cm) * 10))


def grasp_to_pose_params(grasp: dict, time_ms: int = 500) -> ArmCommand:
    """Convert a grasp result dict from vision_obs to an ArmCommand.

    Args:
        grasp: Dict with x_cm, y_cm, z_cm, pitch_deg, roll_deg,
               gripper_width_cm, approach_depth_cm fields.
        time_ms: Arm movement time in milliseconds (default 500).

    Returns:
        ArmCommand ready for POSE encoding.
    """
    return ArmCommand(
        x_cm=float(grasp.get("x_cm", 0.0)),
        y_cm=float(grasp.get("y_cm", 0.0)),
        z_cm=float(grasp.get("z_cm", 0.0)),
        pitch_deg=float(grasp.get("pitch_deg", 0.0)),
        roll_deg=float(normalize_roll(grasp.get("roll_deg", 0.0))),
        claw_deg=width_to_claw_angle(float(grasp.get("gripper_width_cm", 0.0))),
        time_ms=int(time_ms),
    )
