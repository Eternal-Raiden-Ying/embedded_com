#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional, Tuple

from ..config.schema import CarMotionConfig, ControlThresholds
from ..ipc.protocol import CmdVel, HomeTagObs, TargetObs, now_ts


@dataclass
class MotionDecision:
    cmd: CmdVel
    cx_norm_abs: float = 0.0
    distance_ratio: float = 0.0


class MotionController:
    def __init__(self, cfg: ControlThresholds, car_cfg: CarMotionConfig):
        self.cfg = cfg
        self.car_cfg = car_cfg
        self._last_search_vx = 0.0
        self._last_search_wz = 0.0

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(v)))

    @staticmethod
    def _blend(prev: float, cur: float, alpha: float) -> float:
        alpha = max(0.0, min(1.0, float(alpha)))
        return (1.0 - alpha) * float(prev) + alpha * float(cur)

    def _scaled_turn(self, x_abs: float, mode: str) -> float:
        x_abs = self._clamp(x_abs, 0.0, 1.0)
        if mode == "RETURN":
            lo, hi = self.car_cfg.return_turn_norm_min, self.car_cfg.return_turn_norm_max
        else:
            lo, hi = self.car_cfg.search_turn_norm_min, self.car_cfg.search_turn_norm_max
        return lo + (hi - lo) * x_abs

    def _scaled_forward(self, distance_ratio: float, mode: str) -> float:
        ratio = self._clamp(distance_ratio, 0.0, 1.0)
        if mode == "RETURN":
            lo, hi = self.car_cfg.return_vx_norm_min, self.car_cfg.return_vx_norm_max
        else:
            lo, hi = self.car_cfg.search_vx_norm_min, self.car_cfg.search_vx_norm_max
        return lo + (hi - lo) * ratio

    def _extract_norm_pair(self, vx_val: Optional[float], wz_val: Optional[float]) -> Optional[Tuple[float, float]]:
        if vx_val is None and wz_val is None:
            return None
        vx = self._clamp(vx_val or 0.0, -1.0, 1.0)
        wz = self._clamp(wz_val or 0.0, -1.0, 1.0)
        return vx, wz

    def _reset_search_memory(self):
        self._last_search_vx = 0.0
        self._last_search_wz = 0.0

    def auto_search_cmd(self) -> MotionDecision:
        self._reset_search_memory()
        return MotionDecision(cmd=CmdVel(ts=now_ts(), mode="AUTOSEARCH", vx_norm=0.0, wz_norm=0.0))

    def auto_explore_cmd(self) -> MotionDecision:
        self._reset_search_memory()
        return MotionDecision(cmd=CmdVel(ts=now_ts(), mode="AUTOEXPLORE", vx_norm=0.0, wz_norm=0.0))

    def stop_cmd(self, mode: str = "STOP") -> MotionDecision:
        self._reset_search_memory()
        return MotionDecision(cmd=CmdVel(ts=now_ts(), mode=mode, vx_norm=0.0, wz_norm=0.0))

    def search_hold_cmd(self) -> MotionDecision:
        self._last_search_vx = self._blend(self._last_search_vx, 0.0, 0.7)
        self._last_search_wz = self._blend(self._last_search_wz, 0.0, 0.7)
        if abs(self._last_search_vx) < 0.02:
            self._last_search_vx = 0.0
        if abs(self._last_search_wz) < 0.02:
            self._last_search_wz = 0.0
        return MotionDecision(cmd=CmdVel(ts=now_ts(), mode="SEARCH", vx_norm=self._last_search_vx, wz_norm=self._last_search_wz))

    def search_cmd(self, obs: TargetObs) -> MotionDecision:
        explicit = self._extract_norm_pair(obs.vx_norm, obs.wz_norm)
        if explicit is not None:
            vx, wz = explicit
            self._last_search_vx, self._last_search_wz = vx, wz
            return MotionDecision(
                cmd=CmdVel(ts=now_ts(), mode="SEARCH", vx_norm=vx, wz_norm=wz),
                cx_norm_abs=abs(float(obs.cx_norm)),
                distance_ratio=max(0.0, min(1.0, 1.0 - float(obs.size_norm))),
            )

        x = self._clamp(float(obs.cx_norm), -1.0, 1.0)
        size = self._clamp(float(obs.size_norm), 0.0, 1.0)
        x_abs = abs(x)
        distance_ratio = max(0.0, min(1.0, 1.0 - size))

        # 角速度：始终可连续输出，偏差越大转得越快。
        if x_abs <= self.cfg.dead_zone_x:
            target_wz = 0.0
        else:
            turn_ratio = (x_abs - self.cfg.dead_zone_x) / max(1e-6, 1.0 - self.cfg.dead_zone_x)
            wz_mag = self._scaled_turn(turn_ratio, mode="SEARCH")
            target_wz = wz_mag if x > 0 else -wz_mag

        # 线速度：偏差越大，前进越慢；目标接近时也放慢。
        if size >= self.cfg.stop_size_norm:
            target_vx = 0.0
        elif x_abs >= float(self.car_cfg.search_spin_only_x_th):
            target_vx = 0.0
        else:
            align_factor = max(0.0, 1.0 - (x_abs / max(1e-6, float(self.car_cfg.search_spin_only_x_th))))
            align_factor = align_factor ** max(1.0, float(self.car_cfg.search_forward_align_exp))
            forward_ratio = self._clamp(distance_ratio * align_factor, 0.0, 1.0)
            target_vx = 0.0 if forward_ratio < 0.05 else self._scaled_forward(forward_ratio, mode="SEARCH")

        # 输出轻微平滑，避免低帧率下明显抖动。
        vx = self._blend(self._last_search_vx, target_vx, 0.45)
        wz = self._blend(self._last_search_wz, target_wz, 0.55)
        if abs(vx) < 0.02:
            vx = 0.0
        if abs(wz) < 0.02:
            wz = 0.0
        self._last_search_vx, self._last_search_wz = vx, wz

        return MotionDecision(
            cmd=CmdVel(ts=now_ts(), mode="SEARCH", vx_norm=vx, wz_norm=wz),
            cx_norm_abs=x_abs,
            distance_ratio=distance_ratio,
        )

    def return_hold_cmd(self) -> MotionDecision:
        return MotionDecision(cmd=CmdVel(ts=now_ts(), mode="RETURN", vx_norm=0.0, wz_norm=0.0))

    def return_cmd(self, obs: HomeTagObs) -> MotionDecision:
        explicit = self._extract_norm_pair(obs.vx_norm, obs.wz_norm)
        if explicit is not None:
            vx, wz = explicit
            return MotionDecision(
                cmd=CmdVel(ts=now_ts(), mode="RETURN", vx_norm=vx, wz_norm=wz),
                cx_norm_abs=abs(float(obs.yaw_err_rad)),
                distance_ratio=float(obs.distance_m or 0.0),
            )
        yaw = float(obs.yaw_err_rad)
        distance = float(obs.distance_m or 0.0)
        ratio = min(1.0, max(0.0, distance / 1.2 if distance > 0 else 0.3))
        if abs(yaw) > self.cfg.align_turn_threshold:
            wz = self._scaled_turn(abs(yaw), mode="RETURN")
            return MotionDecision(
                cmd=CmdVel(ts=now_ts(), mode="RETURN", vx_norm=0.0, wz_norm=wz if yaw > 0 else -wz),
                cx_norm_abs=abs(yaw),
                distance_ratio=ratio,
            )
        vx = self._scaled_forward(ratio, mode="RETURN")
        return MotionDecision(
            cmd=CmdVel(ts=now_ts(), mode="RETURN", vx_norm=vx, wz_norm=0.0),
            cx_norm_abs=abs(yaw),
            distance_ratio=ratio,
        )
