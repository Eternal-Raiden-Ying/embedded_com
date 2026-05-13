#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Callable, Dict, Optional


class Stm32MotionAdapter:
    def __init__(
        self,
        uart: Any,
        logger: Optional[Callable[[str], None]] = None,
        tx_meta_factory: Optional[Callable[[str, int, str], Dict[str, Any]]] = None,
        wheel_speed_limit: int = 100,
        vx_scale: float = 100.0,
        vy_scale: float = 100.0,
        wz_scale: float = 100.0,
        jog_forward_speed: int = 25,
        jog_turn_speed: int = 25,
        jog_duration_ms: int = 100,
    ):
        self.uart = uart
        self.logger = logger
        self.tx_meta_factory = tx_meta_factory
        self.wheel_speed_limit = min(100, max(0, abs(int(wheel_speed_limit or 0))))
        self.vx_scale = float(vx_scale)
        self.vy_scale = float(vy_scale)
        self.wz_scale = float(wz_scale)
        self.jog_forward_speed = self._clamp_int(jog_forward_speed, 20, 40)
        self.jog_turn_speed = self._clamp_int(jog_turn_speed, 20, 40)
        self.jog_duration_ms = self._clamp_int(jog_duration_ms, 60, 150)
        self._seq = 0

    @staticmethod
    def _clamp_int(value: Any, lo: int, hi: int) -> int:
        try:
            number = int(round(float(value)))
        except Exception:
            number = 0
        return max(lo, min(hi, number))

    def _next_seq(self) -> int:
        self._seq = (int(self._seq) % 999999) + 1
        return self._seq

    def _meta(self, kind: str, seq: int, reason: str) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "kind": f"stm32_{kind}",
            "motion_protocol": "stm32",
            "seq": int(seq),
            "reason": str(reason or ""),
        }
        if self.tx_meta_factory is not None:
            try:
                extra = self.tx_meta_factory(kind, seq, reason)
                if isinstance(extra, dict):
                    meta.update({k: v for k, v in extra.items() if v is not None})
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

    def _wheels(self, s006: Any, s007: Any, s008: Any, s009: Any) -> tuple:
        limit = self.wheel_speed_limit
        return (
            self._clamp_int(s006, -limit, limit),
            self._clamp_int(s007, -limit, limit),
            self._clamp_int(s008, -limit, limit),
            self._clamp_int(s009, -limit, limit),
        )

    def cmd_vel_to_wheels(self, cmd: Any) -> tuple:
        vx = float(getattr(cmd, "vx_norm", 0.0) or 0.0) * self.vx_scale
        vy = float(getattr(cmd, "vy_norm", 0.0) or 0.0) * self.vy_scale
        wz = float(getattr(cmd, "wz_norm", 0.0) or 0.0) * self.wz_scale
        return self._wheels(
            +vx - vy - wz,
            -vx - vy + wz,
            +vx + vy - wz,
            -vx + vy + wz,
        )

    @staticmethod
    def _cmd_is_stop(cmd: Any) -> bool:
        mode = str(getattr(cmd, "mode", "") or "").strip().upper()
        brake = bool(getattr(cmd, "brake", False))
        vx = abs(float(getattr(cmd, "vx_norm", 0.0) or 0.0))
        vy = abs(float(getattr(cmd, "vy_norm", 0.0) or 0.0))
        wz = abs(float(getattr(cmd, "wz_norm", 0.0) or 0.0))
        return brake or (mode in {"STOP", "IDLE", "DONE"} and vx < 1e-6 and vy < 1e-6 and wz < 1e-6)

    def set_velocity_wheels(self, s006, s007, s008, s009, reason: str = "") -> int:
        seq = self._next_seq()
        wheels = self._wheels(s006, s007, s008, s009)
        self._log(f"[MOTION][VEL] seq={seq} wheels={wheels} reason={reason}")
        self.uart.send_stm32_vel(*wheels, seq, tx_meta=self._meta("vel", seq, reason))
        return seq

    def send_cmd_vel(self, cmd: Any, reason: str = "") -> int:
        if self._cmd_is_stop(cmd):
            return self.stop(reason=reason)
        wheels = self.cmd_vel_to_wheels(cmd)
        return self.set_velocity_wheels(*wheels, reason=reason)

    def stop(self, reason: str = "") -> int:
        seq = self._next_seq()
        self._log(f"[MOTION][STOP] seq={seq} reason={reason}")
        self.uart.send_stm32_stop(seq, tx_meta=self._meta("stop", seq, reason))
        return seq

    def jog_wheels(self, s006, s007, s008, s009, duration_ms, reason: str = "") -> int:
        seq = self._next_seq()
        wheels = self._wheels(s006, s007, s008, s009)
        duration = self._clamp_int(duration_ms, 20, 1000)
        self._log(f"[MOTION][JOG] seq={seq} wheels={wheels} duration_ms={duration} reason={reason}")
        self.uart.send_stm32_jog(*wheels, duration, seq, tx_meta=self._meta("jog", seq, reason))
        return seq

    def _jog_from_axes(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0, reason: str = "") -> int:
        return self.jog_wheels(
            +vx - vy - wz,
            -vx - vy + wz,
            +vx + vy - wz,
            -vx + vy + wz,
            self.jog_duration_ms,
            reason=reason,
        )

    def jog_forward_small(self, reason: str = "") -> int:
        return self._jog_from_axes(vx=float(self.jog_forward_speed), reason=reason)

    def jog_backward_small(self, reason: str = "") -> int:
        return self._jog_from_axes(vx=-float(self.jog_forward_speed), reason=reason)

    def jog_turn_left_small(self, reason: str = "") -> int:
        return self._jog_from_axes(wz=float(self.jog_turn_speed), reason=reason)

    def jog_turn_right_small(self, reason: str = "") -> int:
        return self._jog_from_axes(wz=-float(self.jog_turn_speed), reason=reason)

    def query_status(self) -> None:
        self._log("[MOTION][STATUS]")
        self.uart.send_stm32_status(tx_meta={
            "kind": "stm32_status",
            "motion_protocol": "stm32",
        })
