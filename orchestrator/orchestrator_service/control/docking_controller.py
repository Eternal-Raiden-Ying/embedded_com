#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from typing import Optional

from .pid import PIDController
from .types import DockingCommand, DockingControlConfig, EdgeControlObservation


class DockingController:
    def __init__(self, cfg: Optional[DockingControlConfig] = None):
        self.cfg = cfg or DockingControlConfig()
        self.yaw_pid = PIDController(self.cfg.yaw_pid)
        self.dist_pid = PIDController(self.cfg.dist_pid)
        self.lateral_pid = PIDController(self.cfg.lateral_pid)
        self._last_ts = 0.0
        self._last_mode = "HOLD"
        self._last_vx = 0.0
        self._last_vy = 0.0
        self._last_wz = 0.0
        self._pose_locked_since = 0.0

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(v)))

    def reset(self):
        self.yaw_pid.reset()
        self.dist_pid.reset()
        self.lateral_pid.reset()
        self._last_ts = 0.0
        self._last_vx = 0.0
        self._last_vy = 0.0
        self._last_wz = 0.0
        self._pose_locked_since = 0.0

    def _dt(self, now_s: float) -> float:
        if self._last_ts <= 0.0:
            self._last_ts = now_s
            return max(float(self.cfg.dt_min_s), 0.05)
        dt = max(float(now_s - self._last_ts), float(self.cfg.dt_min_s))
        self._last_ts = now_s
        return dt

    def _slew(self, prev: float, target: float, rate: float, dt: float) -> float:
        max_delta = max(0.0, float(rate)) * max(dt, 1e-6)
        delta = float(target) - float(prev)
        if delta > max_delta:
            return float(prev) + max_delta
        if delta < -max_delta:
            return float(prev) - max_delta
        return float(target)

    def _limit_cmd(
        self,
        vx: float,
        vy: float,
        wz: float,
        *,
        max_vx: float,
        max_vy: float,
        max_wz: float,
        dt: float,
    ) -> DockingCommand:
        tvx = self._clamp(vx, -abs(max_vx), abs(max_vx))
        tvy = self._clamp(vy, -abs(max_vy), abs(max_vy))
        twz = self._clamp(wz, -abs(max_wz), abs(max_wz))

        vx2 = self._slew(self._last_vx, tvx, self.cfg.vx_slew_per_s, dt)
        vy2 = self._slew(self._last_vy, tvy, self.cfg.vy_slew_per_s, dt)
        wz2 = self._slew(self._last_wz, twz, self.cfg.wz_slew_per_s, dt)

        self._last_vx, self._last_vy, self._last_wz = vx2, vy2, wz2
        return DockingCommand(vx=vx2, vy=vy2, wz=wz2, valid=True)

    def _hold(self, reason: str = "", valid: bool = False) -> DockingCommand:
        self._last_vx = 0.0
        self._last_vy = 0.0
        self._last_wz = 0.0
        return DockingCommand(vx=0.0, vy=0.0, wz=0.0, valid=valid, mode="HOLD", reason=reason)

    def _obs_ok(self, obs: Optional[EdgeControlObservation], now_s: float) -> bool:
        if obs is None or not obs.valid:
            return False
        if float(obs.confidence) < float(self.cfg.min_confidence):
            return False
        if now_s - float(obs.ts) > float(self.cfg.obs_timeout_s):
            return False
        return True

    def _update_pose_locked(self, obs: EdgeControlObservation, now_s: float) -> bool:
        yaw_ok = abs(float(obs.yaw_err_rad or 0.0)) <= float(self.cfg.precise_yaw_tol_rad)
        dist_ok = abs(float(obs.dist_err_m or 0.0)) <= float(self.cfg.precise_dist_tol_m)
        lat_ok = True
        if self.cfg.enable_lateral_control and obs.lateral_err_m is not None:
            lat_ok = abs(float(obs.lateral_err_m)) <= float(self.cfg.precise_lateral_tol_m)

        if yaw_ok and dist_ok and lat_ok:
            if self._pose_locked_since <= 0.0:
                self._pose_locked_since = now_s
            return (now_s - self._pose_locked_since) >= float(self.cfg.precise_stable_s)

        self._pose_locked_since = 0.0
        return False

    def update(self, mode: str, obs: Optional[EdgeControlObservation], now_s: Optional[float] = None) -> DockingCommand:
        now_s = float(now_s if now_s is not None else time.time())
        dt = self._dt(now_s)
        mode = str(mode or "HOLD").upper().strip() or "HOLD"

        if self.cfg.reset_on_mode_change and mode != self._last_mode:
            self.yaw_pid.reset()
            self.dist_pid.reset()
            self.lateral_pid.reset()
            self._pose_locked_since = 0.0
        self._last_mode = mode

        if mode == "HOLD":
            return self._hold(reason="hold", valid=True)

        if not self._obs_ok(obs, now_s):
            return self._hold(reason="edge observation invalid", valid=False)

        assert obs is not None
        yaw_err = float(obs.yaw_err_rad or 0.0)
        dist_err = float(obs.dist_err_m or 0.0)
        lat_err = float(obs.lateral_err_m or 0.0) if obs.lateral_err_m is not None else 0.0

        pose_locked = self._update_pose_locked(obs, now_s)

        if mode == "COARSE_ALIGN":
            wz = self.yaw_pid.update(yaw_err, dt)
            cmd = self._limit_cmd(0.0, 0.0, wz, max_vx=0.0, max_vy=0.0, max_wz=self.cfg.coarse_max_wz_norm, dt=dt)
            cmd.mode = mode
            cmd.pose_locked = pose_locked
            return cmd

        if mode == "CONTROLLED_APPROACH":
            wz = self.yaw_pid.update(yaw_err, dt)
            if abs(yaw_err) >= float(self.cfg.spin_only_yaw_rad):
                vx = 0.0
                freeze_int = True
            else:
                vx = self.dist_pid.update(dist_err, dt)
                freeze_int = False
            vy = 0.0
            if self.cfg.enable_lateral_control and obs.lateral_err_m is not None:
                vy = self.lateral_pid.update(lat_err, dt, freeze_integrator=freeze_int)
            cmd = self._limit_cmd(
                vx,
                vy,
                wz,
                max_vx=self.cfg.approach_max_vx_norm,
                max_vy=self.cfg.approach_max_vy_norm,
                max_wz=self.cfg.approach_max_wz_norm,
                dt=dt,
            )
            cmd.mode = mode
            cmd.pose_locked = pose_locked
            return cmd

        if mode == "FINAL_LOCK":
            wz = self.yaw_pid.update(yaw_err, dt)
            vx = self.dist_pid.update(dist_err, dt)
            vy = 0.0
            if self.cfg.enable_lateral_control and obs.lateral_err_m is not None:
                vy = self.lateral_pid.update(lat_err, dt)
            cmd = self._limit_cmd(
                vx,
                vy,
                wz,
                max_vx=self.cfg.final_max_vx_norm,
                max_vy=self.cfg.final_max_vy_norm,
                max_wz=self.cfg.final_max_wz_norm,
                dt=dt,
            )
            cmd.mode = mode
            cmd.pose_locked = pose_locked
            return cmd

        return self._hold(reason=f"unsupported mode={mode}", valid=False)
