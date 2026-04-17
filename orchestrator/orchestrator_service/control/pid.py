#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .types import PIDAxisConfig


class PIDController:
    def __init__(self, cfg: PIDAxisConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._derivative = 0.0
        self._first = True

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(v)))

    def update(self, error: float, dt: float, freeze_integrator: bool = False) -> float:
        dt = max(float(dt), 1e-6)
        err = float(error)

        if abs(err) <= float(self.cfg.deadband):
            err = 0.0

        if self._first:
            raw_derivative = 0.0
            self._first = False
        else:
            raw_derivative = (err - self._prev_error) / dt

        alpha = self._clamp(float(self.cfg.derivative_alpha), 0.0, 1.0)
        self._derivative = (1.0 - alpha) * self._derivative + alpha * raw_derivative

        if not freeze_integrator and float(self.cfg.ki) != 0.0:
            self._integral += err * dt
            lim = abs(float(self.cfg.integral_limit))
            self._integral = self._clamp(self._integral, -lim, lim)

        out = (
            float(self.cfg.kp) * err
            + float(self.cfg.ki) * self._integral
            + float(self.cfg.kd) * self._derivative
        )

        lim = abs(float(self.cfg.output_limit))
        out = self._clamp(out, -lim, lim)

        min_abs = abs(float(self.cfg.min_abs_output))
        if err == 0.0:
            if abs(out) < min_abs * 1.5:
                out = 0.0
        elif min_abs > 0.0 and out != 0.0 and abs(out) < min_abs:
            out = min_abs if out > 0.0 else -min_abs

        self._prev_error = err
        return out
