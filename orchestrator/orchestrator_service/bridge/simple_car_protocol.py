#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一串口文本协议（分行协议，归一化速度）：
- 主控 -> STM32
    MODE <STOP|AUTOSEARCH|AUTOEXPLORE|SEARCH|RETURN>\n
    V <vx_norm> <wz_norm>\n          # 仅 SEARCH / RETURN 发送，取值范围建议 [-1, 1]
    STOP\n
          # STOP 高优先级显式急停
- STM32 -> 主控
    STATE <ok|timeout|estop|fault> <vL> <vR> <yaw>\n
也兼容 key=value / JSON 风格状态回传。
"""

import json
import re
from dataclasses import dataclass
from typing import Optional

from ..config.schema import CarMotionConfig
from ..ipc.protocol import CarState, CmdVel, now_ts


@dataclass
class SimpleCarCommand:
    raw_line: str
    kind: str
    mode: str
    vx_norm: float = 0.0
    wz_norm: float = 0.0


class SimpleCarMapper:
    NO_PARAM_MODES = {"STOP", "AUTOSEARCH", "AUTOEXPLORE"}
    PARAM_MODES = {"SEARCH", "RETURN"}

    def __init__(self, cfg: CarMotionConfig):
        self.cfg = cfg
        self._last_mode_sent: Optional[str] = None

    def _fmt(self, v: float) -> str:
        digits = max(0, int(self.cfg.serial_float_digits))
        return f"{float(v):.{digits}f}"

    def _mode_prefix(self, mode: str) -> str:
        mode = mode.upper().strip() or "STOP"
        send_mode = bool(self.cfg.mode_line_every_cmd) or (
            bool(self.cfg.mode_line_on_change) and mode != self._last_mode_sent
        )
        if not send_mode:
            return ""
        self._last_mode_sent = mode
        return f"MODE {mode}\n"

    def from_cmd_vel(self, cmd: CmdVel, cx_norm_abs: Optional[float] = None, distance_ratio: Optional[float] = None) -> SimpleCarCommand:
        del cx_norm_abs, distance_ratio
        mode = (cmd.mode or "STOP").upper().strip() or "STOP"
        vx = float(cmd.vx_norm)
        wz = float(cmd.wz_norm)
        prefix = self._mode_prefix(mode)

        if mode == "STOP":
            line = prefix + "STOP\n"
            if not line.strip():
                line = "STOP\n"
            return SimpleCarCommand(raw_line=line, kind="stop", mode="STOP", vx_norm=0.0, wz_norm=0.0)

        if mode in self.NO_PARAM_MODES:
            return SimpleCarCommand(raw_line=prefix, kind="mode_only", mode=mode, vx_norm=0.0, wz_norm=0.0)

        if mode not in self.PARAM_MODES:
            mode = "STOP"
            return SimpleCarCommand(raw_line="STOP\n", kind="stop", mode="STOP", vx_norm=0.0, wz_norm=0.0)

        return SimpleCarCommand(
            raw_line=prefix + f"V {self._fmt(vx)} {self._fmt(wz)}\n",
            kind="cmd_vel",
            mode=mode,
            vx_norm=vx,
            wz_norm=wz,
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

    if upper.startswith("MODE "):
        parts = upper.split(None, 1)
        mode = parts[1].strip() if len(parts) > 1 else None
        return CarState(ts=now_ts(), state="INFO", mode=mode, raw=raw, source="uart")

    state = "UNKNOWN"
    ok = False
    timeout = False
    estop = False
    fault = False
    mode = None
    message = None

    state_match = re.match(r"^STATE\s+([A-Z_]+)(?:\s+([-+]?\d*\.?\d+))?(?:\s+([-+]?\d*\.?\d+))?(?:\s+([-+]?\d*\.?\d+))?", upper)
    if state_match:
        token = state_match.group(1)
        if token in {"OK", "TIMEOUT", "ESTOP", "FAULT", "ERROR"}:
            state = "FAULT" if token == "ERROR" else token
            ok = state == "OK"
            timeout = state == "TIMEOUT"
            estop = state == "ESTOP"
            fault = state == "FAULT"
            if any(state_match.group(i) is not None for i in (2, 3, 4)):
                message = f"vL={state_match.group(2)} vR={state_match.group(3)} yaw={state_match.group(4)}"

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
        elif re.search(r"(^|\b)(OK|ACK)(\b|$)", upper):
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
        source="uart",
    )
