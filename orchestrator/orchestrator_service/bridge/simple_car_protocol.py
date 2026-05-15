#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一串口文本协议:

- 主控 -> STM32
    新版 SC171/STM32:
    VEL <s006> <s007> <s008> <s009> <seq>\n
    STOP <seq>\n
    JOG <s006> <s007> <s008> <s009> <duration_ms> <seq>\n
    STATUS\n

    旧版调试协议:
    MODE <STATE_NAME>\n
    VEL <vx> <vy> <wz> <hold_ms>\n
    STOP\n
    BRAKE\n

- STM32 -> 主控
    ACK_START seq=12\n
    ACK_DONE seq=12\n
    BUSY seq=12\n
    STATUS target=... applied=... jog=...\n
    [CAR][JOG_START] seq=12\n
    [CAR][JOG_DONE] seq=12\n
    [CAR][JOG_BUSY] seq=12\n
    [CAR][TIMEOUT] auto stop\n

    旧版调试协议:
    STATE <status> <vx> <vy> <wz> <fault_code>\n
    ESTOP <0|1>\n

同时兼容旧版 key=value / JSON 风格回传。
"""

import json
import re
from dataclasses import dataclass
from typing import Optional

from ..config.schema import CarMotionConfig
from ..ipc.protocol import CarState, CmdVel, now_ts


def _clamp_int(v, lo: int, hi: int) -> int:
    try:
        value = int(round(float(v)))
    except Exception:
        value = 0
    return max(lo, min(hi, value))


def _format_seq(seq) -> str:
    try:
        return str(int(seq))
    except Exception:
        return "0"


def _format_speed(v) -> str:
    return str(_clamp_int(v, -100, 100))


def encode_vel(s006, s007, s008, s009, seq) -> str:
    return f"VEL {_format_speed(s006)} {_format_speed(s007)} {_format_speed(s008)} {_format_speed(s009)} {_format_seq(seq)}"


def encode_stm32_vel(s006, s007, s008, s009, seq) -> str:
    return encode_vel(s006, s007, s008, s009, seq)


def encode_stop(seq) -> str:
    return f"STOP {_format_seq(seq)}"


def encode_stm32_stop(seq) -> str:
    return encode_stop(seq)


def encode_jog(s006, s007, s008, s009, duration_ms, seq) -> str:
    duration = _clamp_int(duration_ms, 20, 1000)
    return (
        f"JOG {_format_speed(s006)} {_format_speed(s007)} "
        f"{_format_speed(s008)} {_format_speed(s009)} {duration} {_format_seq(seq)}"
    )


def encode_stm32_jog(s006, s007, s008, s009, duration_ms, seq) -> str:
    return encode_jog(s006, s007, s008, s009, duration_ms, seq)


def encode_status() -> str:
    return "STATUS"


def encode_stm32_status() -> str:
    return encode_status()


@dataclass
class SimpleCarCommand:
    raw_line: str
    kind: str
    mode: str
    vx_norm: float = 0.0
    vy_norm: float = 0.0
    wz_norm: float = 0.0
    hold_ms: int = 0
    brake: bool = False


class SimpleCarMapper:
    def __init__(self, cfg: CarMotionConfig):
        self.cfg = cfg
        self._last_mode_sent: Optional[str] = None

    def _fmt(self, v: float) -> str:
        digits = max(0, int(self.cfg.serial_float_digits))
        return f"{float(v):.{digits}f}"

    @staticmethod
    def _clamp_abs(v: float, limit: float) -> float:
        limit = abs(float(limit or 0.0))
        if limit <= 0.0:
            return 0.0
        return max(-limit, min(limit, float(v)))

    def _mode_prefix(self, mode: str) -> str:
        mode = mode.upper().strip() or "IDLE"
        send_mode = bool(self.cfg.mode_line_every_cmd) or (
            bool(self.cfg.mode_line_on_change) and mode != self._last_mode_sent
        )
        if not send_mode:
            return ""
        self._last_mode_sent = mode
        return f"MODE {mode}\n"

    def from_cmd_vel(self, cmd: CmdVel, cx_norm_abs: Optional[float] = None, distance_ratio: Optional[float] = None) -> SimpleCarCommand:
        del cx_norm_abs, distance_ratio
        mode = (cmd.mode or "IDLE").upper().strip() or "IDLE"
        vx = self._clamp_abs(float(cmd.vx_norm), getattr(self.cfg, "max_vx_norm", 1.0))
        vy = self._clamp_abs(float(cmd.vy_norm), getattr(self.cfg, "max_vy_norm", 1.0))
        wz = self._clamp_abs(float(cmd.wz_norm), getattr(self.cfg, "max_wz_norm", 1.0))
        hold_ms = int(max(0, int(getattr(cmd, "hold_ms", self.cfg.cmd_hold_ms))))
        brake = bool(getattr(cmd, "brake", False))
        prefix = self._mode_prefix(mode)

        if brake:
            return SimpleCarCommand(
                raw_line=prefix + "BRAKE\n",
                kind="brake",
                mode=mode,
                hold_ms=hold_ms,
                brake=True,
            )

        if mode in {"STOP", "IDLE", "DONE"} and abs(vx) < 1e-6 and abs(vy) < 1e-6 and abs(wz) < 1e-6:
            line = prefix + "STOP\n"
            if not line.strip():
                line = "STOP\n"
            return SimpleCarCommand(raw_line=line, kind="stop", mode=mode, hold_ms=hold_ms)

        return SimpleCarCommand(
            raw_line=prefix + f"VEL {self._fmt(vx)} {self._fmt(vy)} {self._fmt(wz)} {hold_ms}\n",
            kind="cmd_vel",
            mode=mode,
            vx_norm=vx,
            vy_norm=vy,
            wz_norm=wz,
            hold_ms=hold_ms,
        )


def parse_car_state_line(line: str) -> Optional[CarState]:
    raw = str(line or "").strip()
    if not raw:
        return None

    if raw.startswith("{") and raw.endswith("}"):
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                payload.setdefault("raw", raw)
                payload.setdefault("source", "uart")
                payload.setdefault("type", "car_state")
                return CarState.from_dict(payload)
        except Exception:
            pass

    upper = raw.upper()

    stm32 = _parse_stm32_feedback(raw, upper)
    if stm32 is not None:
        return stm32

    if upper.startswith("MODE "):
        parts = upper.split(None, 1)
        mode = parts[1].strip() if len(parts) > 1 else None
        return CarState(ts=now_ts(), state="INFO", mode=mode, raw=raw, source="uart")

    estop_match = re.match(r"^ESTOP\s+([01])$", upper)
    if estop_match:
        estop = estop_match.group(1) == "1"
        return CarState(
            ts=now_ts(),
            state="ESTOP" if estop else "OK",
            ok=not estop,
            estop=estop,
            raw=raw,
            source="uart",
        )

    state = "UNKNOWN"
    ok = False
    timeout = False
    estop = False
    fault = False
    mode = None
    message = None
    vx = None
    vy = None
    wz = None
    fault_code = None

    state_match = re.match(
        r"^STATE\s+([A-Z_]+)"
        r"(?:\s+([-+]?\d*\.?\d+))?"
        r"(?:\s+([-+]?\d*\.?\d+))?"
        r"(?:\s+([-+]?\d*\.?\d+))?"
        r"(?:\s+([A-Z0-9_\-]+))?",
        upper,
    )
    if state_match:
        token = state_match.group(1)
        if token in {"OK", "BUSY", "DONE", "TIMEOUT", "ESTOP", "FAULT", "ERROR"}:
            state = "FAULT" if token == "ERROR" else token
            ok = state in {"OK", "BUSY", "DONE"}
            timeout = state == "TIMEOUT"
            estop = state == "ESTOP"
            fault = state == "FAULT"
            try:
                vx = float(state_match.group(2)) if state_match.group(2) is not None else None
                vy = float(state_match.group(3)) if state_match.group(3) is not None else None
                wz = float(state_match.group(4)) if state_match.group(4) is not None else None
            except Exception:
                vx = vy = wz = None
            fault_code = state_match.group(5)
            if any(state_match.group(i) is not None for i in (2, 3, 4, 5)):
                message = f"vx={state_match.group(2)} vy={state_match.group(3)} wz={state_match.group(4)} fault={state_match.group(5)}"

    if state == "UNKNOWN":
        if "ESTOP" in upper or "E_STOP" in upper or "EMERGENCY" in upper:
            state = "ESTOP"
            estop = True
        elif "TIMEOUT" in upper:
            state = "TIMEOUT"
            timeout = True
        elif "FAULT" in upper or "ERROR" in upper:
            state = "FAULT"
            fault = True
        elif re.search(r"(^|\b)(OK|ACK|DONE|BUSY)(\b|$)", upper):
            state = "OK"
            ok = True

    match = re.search(r"(?:MODE|STATE_MODE|RUN_MODE)\s*[:= ]\s*([A-Z_]+)", upper)
    if match:
        mode = match.group(1)

    if state == "UNKNOWN" and mode is None:
        return None

    return CarState(
        ts=now_ts(),
        state=state,
        ok=ok,
        timeout=timeout,
        estop=estop,
        fault=fault,
        mode=mode,
        message=message,
        raw=raw,
        vx=vx,
        vy=vy,
        wz=wz,
        fault_code=fault_code,
        source="uart",
    )


def _parse_stm32_feedback(raw: str, upper: str) -> Optional[CarState]:
    state = None
    ok = False
    timeout = False
    fault = False
    message = None

    ack_match = re.match(r"^(ACK_START|ACK_DONE)\b(?:\s+(.*))?$", raw, re.IGNORECASE)
    if ack_match:
        token = ack_match.group(1).upper()
        state = "ACK_START" if token == "ACK_START" else "DONE"
        ok = True
        message = (ack_match.group(2) or "").strip() or None

    busy_match = re.match(r"^BUSY\b(?:\s+(.*))?$", raw, re.IGNORECASE)
    if state is None and busy_match:
        state = "BUSY"
        ok = True
        message = (busy_match.group(1) or "").strip() or None

    status_match = re.match(r"^STATUS\b(?:\s+(.*))?$", raw, re.IGNORECASE)
    if state is None and status_match:
        state = "STATUS"
        ok = True
        message = (status_match.group(1) or "").strip() or None

    jog_match = re.match(r"^\[CAR\]\[(JOG_START|JOG_DONE|JOG_BUSY|TIMEOUT)\](?:\s+(.*))?$", raw, re.IGNORECASE)
    if state is None and jog_match:
        token = jog_match.group(1).upper()
        message = (jog_match.group(2) or "").strip() or None
        if token == "JOG_START":
            state = "ACK_START"
            ok = True
        elif token == "JOG_DONE":
            state = "DONE"
            ok = True
        elif token == "JOG_BUSY":
            state = "BUSY"
            ok = True
        elif token == "TIMEOUT":
            state = "TIMEOUT"
            timeout = True

    if state is None:
        return None

    if state in {"FAULT", "ERROR"}:
        fault = True

    return CarState(
        ts=now_ts(),
        state=state,
        ok=ok,
        timeout=timeout,
        fault=fault,
        message=message,
        raw=raw,
        source="uart",
    )
