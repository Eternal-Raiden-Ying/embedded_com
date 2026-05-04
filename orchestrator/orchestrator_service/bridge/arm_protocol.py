#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional

from ..ipc.protocol import ArmResponse, now_ts


def encode_pose(
    x: float, y: float, z: float,
    pitch: float, roll: float,
    claw: float, time_ms: int,
) -> str:
    """Encode an arm POSE command for the STM32 arm MCU."""
    return (
        f"POSE {round(x)} {round(y)} {round(z)} "
        f"{round(pitch)} {round(roll)} {round(claw)} {int(time_ms)}\n"
    )


def encode_reset() -> str:
    """Encode a RESET command for the arm MCU."""
    return "RESET\n"


def parse_arm_response(line: str) -> Optional[ArmResponse]:
    """Parse a UART line from the arm MCU.

    Returns an ArmResponse for recognised arm messages (OK POSE / ERR IK).
    Returns None if the line does not match the arm protocol.
    """
    stripped = line.strip()
    if not stripped:
        return None

    upper = stripped.upper()

    if upper.startswith("OK POSE"):
        return ArmResponse(
            ok=True,
            message=stripped,
            raw_line=line,
            ts=now_ts(),
        )

    if upper.startswith("ERR IK"):
        return ArmResponse(
            ok=False,
            message=stripped,
            raw_line=line,
            ts=now_ts(),
        )

    return None
