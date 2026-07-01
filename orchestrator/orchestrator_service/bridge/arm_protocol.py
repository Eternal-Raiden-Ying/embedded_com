#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from typing import Any, Dict, Optional

from ..ipc.protocol import ArmResponse, now_ts


_POSE_KEY_ALIASES = {
    "x": "x",
    "y": "y",
    "z": "z",
    "pitch": "pitch",
    "roll": "roll",
    "claw": "claw",
    "t": "time_ms",
    "time": "time_ms",
    "time_ms": "time_ms",
}
_POSE_COMPARE_KEYS = ("x", "y", "z", "pitch", "roll", "claw", "time_ms")
_KV_RE = re.compile(r"\b([A-Za-z_]+)\s*=\s*(-?\d+(?:\.\d+)?)")


_NOISE_PATTERNS = (
    "UART1",
    "ALIVE",
    "PC ECHO MODE",
    "ECHO MODE",
    "PC MODE",
    "DIRECT ALIVE",
    "PRINTF ALIVE",
    "READY",
)


def encode_pose(
    x: float, y: float, z: float,
    pitch: float, roll: float,
    claw: float, time_ms: int,
) -> str:
    """Encode an arm POSE command for the STM32 arm MCU.

    The arm firmware PC mode expects one text line with seven integer
    parameters. The caller is responsible for validating the input range.
    """
    return (
        f"POSE {round(x)} {round(y)} {round(z)} "
        f"{round(pitch)} {round(roll)} {round(claw)} {int(time_ms)}"
    )


def encode_reset() -> str:
    """Encode a RESET command for the arm MCU."""
    return "RESET\n"


def pose_dict_from_command(line: str) -> Dict[str, int]:
    """Parse a POSE command line into the integer pose sent to firmware."""
    parts = str(line or "").strip().split()
    if len(parts) < 8 or parts[0].upper() != "POSE":
        return {}
    try:
        values = [int(round(float(value))) for value in parts[1:8]]
    except Exception:
        return {}
    return {
        "x": values[0],
        "y": values[1],
        "z": values[2],
        "pitch": values[3],
        "roll": values[4],
        "claw": values[5],
        "time_ms": values[6],
    }


def parse_pose_fields(line: str) -> Dict[str, int]:
    """Parse x=... y=... style pose fields from an arm response line."""
    pose: Dict[str, int] = {}
    for key, value in _KV_RE.findall(str(line or "")):
        canonical = _POSE_KEY_ALIASES.get(key.strip().lower())
        if canonical is None:
            continue
        try:
            pose[canonical] = int(round(float(value)))
        except Exception:
            continue
    return pose


def _is_noise_line(raw: str) -> bool:
    upper = str(raw or "").strip().upper()
    if not upper:
        return False
    # Firmware boot/status messages seen in logs include:
    #   UART1 direct alive
    #   UART1 printf alive
    #   PC echo mode
    # These lines are neither success nor failure of the current command.
    if upper.startswith("OK ") or upper.startswith("ERR "):
        return False
    return any(pattern in upper for pattern in _NOISE_PATTERNS)


def parse_arm_response_detail(line: str) -> Dict[str, Any]:
    """Parse one raw line from the arm serial port.

    Returns a dict with status in:
      OK_POSE, ERR_IK, ERR_CMD, NOISE, UNKNOWN.
    """
    raw = str(line or "").strip()
    if not raw:
        return {"status": "UNKNOWN", "raw": raw}

    upper = raw.upper()

    if _is_noise_line(raw):
        return {"status": "NOISE", "raw": raw}

    if upper.startswith("OK POSE"):
        return {"status": "OK_POSE", "raw": raw, "pose": parse_pose_fields(raw)}

    if upper.startswith("ERR IK"):
        detail = {"status": "ERR_IK", "raw": raw}
        pose = parse_pose_fields(raw)
        if pose:
            detail["pose"] = pose
        return detail

    if upper.startswith("ERR CMD"):
        return {"status": "ERR_CMD", "raw": raw}

    return {"status": "UNKNOWN", "raw": raw}


def pose_matches(sent_pose: Dict[str, Any], response_pose: Dict[str, Any]) -> bool:
    """Return True only when response OK_POSE echoes the current sent pose."""
    if not isinstance(sent_pose, dict) or not isinstance(response_pose, dict):
        return False
    for key in _POSE_COMPARE_KEYS:
        if key not in sent_pose or key not in response_pose:
            return False
        try:
            if int(round(float(sent_pose[key]))) != int(round(float(response_pose[key]))):
                return False
        except Exception:
            return False
    return True


def parse_arm_response(line: str) -> Optional[ArmResponse]:
    """Parse a UART line from the arm MCU.

    Returns an ArmResponse for recognised command results.
    Returns None for noise/unknown lines.
    """
    detail = parse_arm_response_detail(line)
    status = str(detail.get("status") or "")
    stripped = str(line or "").strip()
    if not stripped:
        return None

    if status == "OK_POSE":
        return ArmResponse(
            ok=True,
            message="OK_POSE",
            raw_line=line,
            ts=now_ts(),
            parsed_status="OK_POSE",
        )

    if status == "ERR_IK":
        return ArmResponse(
            ok=False,
            message="ERR_IK",
            raw_line=line,
            ts=now_ts(),
            parsed_status="ERR_IK",
        )

    if status == "ERR_CMD":
        return ArmResponse(
            ok=False,
            message="ERR_CMD",
            raw_line=line,
            ts=now_ts(),
            parsed_status="ERR_CMD",
        )

    return None