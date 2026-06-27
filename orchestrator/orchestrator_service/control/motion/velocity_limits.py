#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
import math
from typing import Any, Optional

from ...bridge.simple_car_protocol import (
    WIRE_MOTION_MODES,
    encode_emergency_stop,
    encode_mode,
    encode_soft_stop,
    encode_vel,
    normalize_wire_mode,
)
from ...config.schema import CarMotionConfig
from ...ipc.protocol import CmdVel


@dataclass
class SimpleCarCommand:
    raw_line: str
    kind: str
    mode: str
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    wz_radps: float = 0.0
    hold_ms: int = 0
    brake: bool = False


class VelocityLimiter:
    @staticmethod
    def clamp_abs(value: float, limit: float) -> float:
        limit = abs(float(limit or 0.0))
        if limit <= 0.0:
            return 0.0
        return max(-limit, min(limit, float(value)))


def clamp_int(value: Any, lo: int, hi: int) -> int:
    try:
        number = int(round(float(value)))
    except Exception:
        number = 0
    return max(lo, min(hi, number))


def coerce_micro_speed(value: Any, default: float) -> float:
    try:
        number = abs(float(value))
    except Exception:
        return float(default)
    if number <= 0.0:
        return float(default)
    if number > 1.0:
        number = number / 1000.0
    return number


def coerce_axis_limit(value: Any, default: float) -> float:
    try:
        number = abs(float(value))
    except Exception:
        return float(default)
    if not math.isfinite(number) or number <= 0.0:
        return float(default)
    return number


def clamp_float(value: Any, lo: float, hi: float) -> float:
    try:
        number = float(value)
    except Exception:
        number = 0.0
    if not math.isfinite(number):
        number = 0.0
    return max(float(lo), min(float(hi), number))


class SimpleCarMapper:
    def __init__(self, cfg: CarMotionConfig):
        self.cfg = cfg
        self._last_mode_sent: Optional[str] = None

    def _mode_prefix(self, mode: str) -> str:
        mode = normalize_wire_mode(mode)
        if mode == "STOP":
            return ""
        send_mode = bool(self.cfg.mode_line_every_cmd) or (
            bool(self.cfg.mode_line_on_change) and mode != self._last_mode_sent
        )
        if not send_mode:
            return ""
        self._last_mode_sent = mode
        return encode_mode(mode) + "\r\n"

    def from_cmd_vel(
        self,
        cmd: CmdVel,
        cx_norm_abs: Optional[float] = None,
        distance_ratio: Optional[float] = None,
    ) -> SimpleCarCommand:
        del cx_norm_abs, distance_ratio
        orig_mode = str(cmd.mode or "IDLE").strip().upper()
        mode = normalize_wire_mode(cmd.mode or "IDLE")
        vx = VelocityLimiter.clamp_abs(float(cmd.vx_mps), getattr(self.cfg, "max_vx_mps", 1.0))
        vy = VelocityLimiter.clamp_abs(float(cmd.vy_mps), getattr(self.cfg, "max_vy_mps", 1.0))
        wz = VelocityLimiter.clamp_abs(float(cmd.wz_radps), getattr(self.cfg, "max_wz_radps", 1.0))
        hold_ms = int(max(0, int(getattr(cmd, "hold_ms", self.cfg.cmd_hold_ms))))
        brake = bool(getattr(cmd, "brake", False))
        prefix = self._mode_prefix(mode)
        digits = max(0, int(self.cfg.serial_float_digits))

        if brake:
            return SimpleCarCommand(
                raw_line=encode_emergency_stop() + "\r\n",
                kind="stop",
                mode=mode,
                hold_ms=hold_ms,
                brake=True,
            )

        if mode == "STOP":
            line = encode_soft_stop() if orig_mode in {"IDLE", "DONE", "AT_TABLE_EDGE"} else encode_emergency_stop()
            return SimpleCarCommand(raw_line=line + "\r\n", kind="stop", mode=mode, hold_ms=hold_ms)

        if mode not in WIRE_MOTION_MODES:
            return SimpleCarCommand(raw_line=prefix, kind="mode", mode=mode, hold_ms=hold_ms)

        return SimpleCarCommand(
            raw_line=prefix + encode_vel(vx, vy, wz, digits=digits) + "\r\n",
            kind="cmd_vel",
            mode=mode,
            vx_mps=vx,
            vy_mps=vy,
            wz_radps=wz,
            hold_ms=hold_ms,
        )
