#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from typing import Any, Callable, Dict, Optional, Tuple


SAFE_DEFAULT_VX_MPS_PER_NORM = 0.30
SAFE_DEFAULT_VY_MPS_PER_NORM = 0.30
SAFE_DEFAULT_WZ_RADPS_PER_NORM = 1.00


class Stm32MotionAdapter:
    """SC171 -> STM32 text protocol adapter.

    The current STM32 firmware owns mecanum wheel solving.  SC171 sends only
    high-level mode and body velocity:
    MODE SEARCH/RETURN, V vx_mps vy_mps wz_radps, STOP.
    """

    def __init__(
        self,
        uart: Any,
        logger: Optional[Callable[[str], None]] = None,
        tx_meta_factory: Optional[Callable[[str, int, str], Dict[str, Any]]] = None,
        wheel_speed_limit: int = 100,
        vx_scale: float = SAFE_DEFAULT_VX_MPS_PER_NORM,
        vy_scale: float = SAFE_DEFAULT_VY_MPS_PER_NORM,
        wz_scale: float = SAFE_DEFAULT_WZ_RADPS_PER_NORM,
        jog_forward_speed: float = 0.02,
        jog_turn_speed: float = 0.05,
        jog_duration_ms: int = 100,
    ):
        del wheel_speed_limit
        self.uart = uart
        self.logger = logger
        self.tx_meta_factory = tx_meta_factory
        self.vx_scale = self._coerce_axis_scale(vx_scale, SAFE_DEFAULT_VX_MPS_PER_NORM)
        self.vy_scale = self._coerce_axis_scale(vy_scale, SAFE_DEFAULT_VY_MPS_PER_NORM)
        self.wz_scale = self._coerce_axis_scale(wz_scale, SAFE_DEFAULT_WZ_RADPS_PER_NORM)
        self.jog_forward_speed = self._coerce_micro_speed(jog_forward_speed, 0.02)
        self.jog_turn_speed = self._coerce_micro_speed(jog_turn_speed, 0.05)
        self.jog_duration_ms = self._clamp_int(jog_duration_ms, 60, 500)
        self._seq = 0
        self._last_wire_mode = ""

    @staticmethod
    def _clamp_int(value: Any, lo: int, hi: int) -> int:
        try:
            number = int(round(float(value)))
        except Exception:
            number = 0
        return max(lo, min(hi, number))

    @staticmethod
    def _coerce_micro_speed(value: Any, default: float) -> float:
        try:
            number = abs(float(value))
        except Exception:
            return float(default)
        if number <= 0.0:
            return float(default)
        # Backward compatibility for previous wheel-unit defaults like 25.
        if number > 1.0:
            number = number / 1000.0
        return number

    @staticmethod
    def _coerce_axis_scale(value: Any, default: float) -> float:
        try:
            number = abs(float(value))
        except Exception:
            return float(default)
        if not math.isfinite(number) or number <= 0.0:
            return float(default)
        return number

    @staticmethod
    def _clamp_float(value: Any, lo: float, hi: float) -> float:
        try:
            number = float(value)
        except Exception:
            number = 0.0
        if not math.isfinite(number):
            number = 0.0
        return max(float(lo), min(float(hi), number))

    def _next_seq(self) -> int:
        self._seq = (int(self._seq) % 999999) + 1
        return self._seq

    def _meta(self, kind: str, seq: int, reason: str, **extra: Any) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "kind": f"stm32_{kind}",
            "motion_protocol": "mode_v",
            "seq": int(seq),
            "reason": str(reason or ""),
        }
        meta.update({k: v for k, v in extra.items() if v is not None})
        if self.tx_meta_factory is not None:
            try:
                supplied = self.tx_meta_factory(kind, seq, reason)
                if isinstance(supplied, dict):
                    meta.update({k: v for k, v in supplied.items() if v is not None})
            except Exception:
                pass
        return meta

    def _log(self, line: str) -> None:
        if self.logger is not None:
            try:
                self.logger(line)
                return
            except Exception:
                pass
        print(line, flush=True)

    @staticmethod
    def wire_mode_for_mode(mode: str) -> str:
        token = str(mode or "").strip().upper()
        if token in {"RETURN"} or "RETURN" in token or "HOME" in token:
            return "RETURN"
        if token in {"SEARCH", "AUTOSEARCH", "AUTOEXPLORE"}:
            return "SEARCH"
        if token in {"STOP", "IDLE", "DONE", "ERROR", "ERROR_RECOVERY"}:
            return "STOP"
        return "SEARCH"

    @staticmethod
    def _cmd_is_stop(cmd: Any) -> bool:
        mode = str(getattr(cmd, "mode", "") or "").strip().upper()
        brake = bool(getattr(cmd, "brake", False))
        vx = abs(float(getattr(cmd, "vx_norm", 0.0) or 0.0))
        vy = abs(float(getattr(cmd, "vy_norm", 0.0) or 0.0))
        wz = abs(float(getattr(cmd, "wz_norm", 0.0) or 0.0))
        return brake or (mode in {"STOP", "IDLE", "DONE", "ERROR_RECOVERY"} and vx < 1e-6 and vy < 1e-6 and wz < 1e-6)

    def cmd_vel_to_velocity(self, cmd: Any) -> Tuple[float, float, float]:
        vx_norm = self._clamp_float(getattr(cmd, "vx_norm", 0.0), -1.0, 1.0)
        vy_norm = self._clamp_float(getattr(cmd, "vy_norm", 0.0), -1.0, 1.0)
        wz_norm = self._clamp_float(getattr(cmd, "wz_norm", 0.0), -1.0, 1.0)
        vx = self._clamp_float(vx_norm * self.vx_scale, -self.vx_scale, self.vx_scale)
        vy = self._clamp_float(vy_norm * self.vy_scale, -self.vy_scale, self.vy_scale)
        wz = self._clamp_float(wz_norm * self.wz_scale, -self.wz_scale, self.wz_scale)
        return vx, vy, wz

    def _mode_prefix(self, wire_mode: str) -> str:
        wire_mode = self.wire_mode_for_mode(wire_mode)
        if wire_mode == "STOP":
            return ""
        if wire_mode == self._last_wire_mode:
            return ""
        self._last_wire_mode = wire_mode
        return f"MODE {wire_mode}\r\n"

    def set_velocity(self, vx_mps: Any, vy_mps: Any, wz_radps: Any, mode: str = "SEARCH", reason: str = "") -> int:
        seq = self._next_seq()
        wire_mode = self.wire_mode_for_mode(mode)
        vx = self._clamp_float(vx_mps, -self.vx_scale, self.vx_scale)
        vy = self._clamp_float(vy_mps, -self.vy_scale, self.vy_scale)
        wz = self._clamp_float(wz_radps, -self.wz_scale, self.wz_scale)
        if wire_mode not in {"SEARCH", "RETURN"}:
            self._log(f"[MOTION][MODE] seq={seq} mode={wire_mode} reason={reason}")
            self.uart.send_mode(wire_mode, tx_meta=self._meta("mode", seq, reason, wire_mode=wire_mode))
            return seq
        self._log(
            f"[MOTION][V] seq={seq} mode={wire_mode} "
            f"vx_mps={vx:.3f} vy_mps={vy:.3f} wz_radps={wz:.3f} reason={reason}"
        )
        line = self._mode_prefix(wire_mode) + f"V {vx:.3f} {vy:.3f} {wz:.3f}\r\n"
        self.uart.send_motion_line(
            line,
            tx_meta=self._meta("vel", seq, reason, wire_mode=wire_mode, vx_mps=vx, vy_mps=vy, wz_radps=wz),
        )
        return seq

    def send_cmd_vel(self, cmd: Any, reason: str = "") -> int:
        if self._cmd_is_stop(cmd):
            return self.stop(reason=reason)
        vx, vy, wz = self.cmd_vel_to_velocity(cmd)
        return self.set_velocity(vx, vy, wz, mode=str(getattr(cmd, "mode", "SEARCH") or "SEARCH"), reason=reason)

    def stop(self, reason: str = "") -> int:
        seq = self._next_seq()
        self._last_wire_mode = "STOP"
        self._log(f"[MOTION][STOP] seq={seq} reason={reason}")
        self.uart.send_stm32_stop(seq, tx_meta=self._meta("stop", seq, reason, wire_mode="STOP"))
        return seq

    def jog_velocity(self, vx_mps: float = 0.0, vy_mps: float = 0.0, wz_radps: float = 0.0, reason: str = "") -> int:
        seq = self._next_seq()
        duration = self.jog_duration_ms
        self._last_wire_mode = "SEARCH"
        self._log(
            f"[MOTION][PULSE] seq={seq} mode=SEARCH vx_mps={vx_mps:.3f} "
            f"vy_mps={vy_mps:.3f} wz_radps={wz_radps:.3f} duration_ms={duration} reason={reason}"
        )
        meta = self._meta(
            "pulse",
            seq,
            reason,
            wire_mode="SEARCH",
            vx_mps=float(vx_mps),
            vy_mps=float(vy_mps),
            wz_radps=float(wz_radps),
            duration_ms=int(duration),
        )
        self.uart.send_motion_line(
            f"MODE SEARCH\r\nV {float(vx_mps):.3f} {float(vy_mps):.3f} {float(wz_radps):.3f}\r\n",
            tx_meta=meta,
            latest_override=False,
        )
        time.sleep(float(duration) / 1000.0)
        self.uart.send_motion_line("STOP\r\n", tx_meta=dict(meta, kind="stm32_stop", pulse_stop=True), latest_override=False)
        return seq

    def jog_forward_small(self, reason: str = "") -> int:
        return self.jog_velocity(vx_mps=float(self.jog_forward_speed), reason=reason)

    def jog_backward_small(self, reason: str = "") -> int:
        return self.jog_velocity(vx_mps=-float(self.jog_forward_speed), reason=reason)

    def jog_turn_left_small(self, reason: str = "") -> int:
        return self.jog_velocity(wz_radps=float(self.jog_turn_speed), reason=reason)

    def jog_turn_right_small(self, reason: str = "") -> int:
        return self.jog_velocity(wz_radps=-float(self.jog_turn_speed), reason=reason)

    def query_status(self) -> None:
        self._log("[MOTION][STATUS] skipped: current STM32 protocol uses FB echoes only")
