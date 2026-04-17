#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional, Tuple

from ..config.schema import CarMotionConfig, ControlThresholds
from ..control.docking_controller import DockingController
from ..control.types import DockingControlConfig, EdgeControlObservation
from ..ipc.protocol import CmdVel, HomeTagObs, TableEdgeObs, TargetObs, now_ts


@dataclass
class MotionDecision:
    cmd: CmdVel
    cx_norm_abs: float = 0.0
    distance_ratio: float = 0.0


class MotionController:
    def __init__(
        self,
        cfg: ControlThresholds,
        car_cfg: CarMotionConfig,
        docking_cfg: Optional[DockingControlConfig] = None,
    ):
        self.cfg = cfg
        self.car_cfg = car_cfg
        self.docking = DockingController(docking_cfg or DockingControlConfig())
        self._last_fallback_vx = 0.0
        self._last_fallback_wz = 0.0

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(v)))

    @staticmethod
    def _blend(prev: float, cur: float, alpha: float) -> float:
        alpha = max(0.0, min(1.0, float(alpha)))
        return (1.0 - alpha) * float(prev) + alpha * float(cur)

    def _cmd(self, mode: str, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0, brake: bool = False) -> CmdVel:
        return CmdVel(
            ts=now_ts(),
            mode=mode,
            vx_norm=float(vx),
            vy_norm=float(vy),
            wz_norm=float(wz),
            hold_ms=int(self.car_cfg.cmd_hold_ms),
            brake=bool(brake),
        )

    def _scaled_turn(self, x_abs: float) -> float:
        x_abs = self._clamp(x_abs, 0.0, 1.0)
        lo = float(self.car_cfg.fallback_align_turn_norm_min)
        hi = float(self.car_cfg.fallback_align_turn_norm_max)
        return lo + (hi - lo) * x_abs

    def _scaled_forward(self, distance_ratio: float) -> float:
        ratio = self._clamp(distance_ratio, 0.0, 1.0)
        lo = float(self.car_cfg.fallback_forward_vx_norm_min)
        hi = float(self.car_cfg.fallback_forward_vx_norm_max)
        return lo + (hi - lo) * ratio

    def _extract_norm_triplet(
        self,
        vx_val: Optional[float],
        vy_val: Optional[float],
        wz_val: Optional[float],
    ) -> Optional[Tuple[float, float, float]]:
        if vx_val is None and vy_val is None and wz_val is None:
            return None
        vx = self._clamp(vx_val or 0.0, -1.0, 1.0)
        vy = self._clamp(vy_val or 0.0, -1.0, 1.0)
        wz = self._clamp(wz_val or 0.0, -1.0, 1.0)
        return vx, vy, wz

    def _reset_fallback_memory(self):
        self._last_fallback_vx = 0.0
        self._last_fallback_wz = 0.0

    def stop_cmd(self, mode: str = "STOP", brake: bool = False) -> MotionDecision:
        self._reset_fallback_memory()
        self.docking.reset()
        return MotionDecision(cmd=self._cmd(mode, brake=brake))

    def search_table_cmd(self, turn_sign: int = 1) -> MotionDecision:
        self._reset_fallback_memory()
        wz = float(self.car_cfg.search_table_wz_norm) * (1.0 if int(turn_sign) >= 0 else -1.0)
        return MotionDecision(cmd=self._cmd("SEARCH_TABLE", wz=wz), cx_norm_abs=abs(wz), distance_ratio=1.0)

    def next_table_cmd(self, turn_sign: int = 1) -> MotionDecision:
        return self.search_table_cmd(turn_sign=turn_sign)

    def leave_edge_cmd(self) -> MotionDecision:
        self._reset_fallback_memory()
        return MotionDecision(cmd=self._cmd("LEAVE_EDGE", vx=float(self.car_cfg.leave_edge_vx_norm)))

    def relocate_cmd(self, turn_sign: int = 1) -> MotionDecision:
        self._reset_fallback_memory()
        wz = float(self.car_cfg.relocate_turn_wz_norm) * (1.0 if int(turn_sign) >= 0 else -1.0)
        return MotionDecision(cmd=self._cmd("RELOCATE_TO_EDGE", wz=wz), cx_norm_abs=abs(wz))

    def edge_slide_search_cmd(self, elapsed_s: float, direction_sign: int = 1) -> MotionDecision:
        if float(elapsed_s) < float(self.cfg.edge_slide_pause_s):
            return MotionDecision(cmd=self._cmd("EDGE_SLIDE_SEARCH"))
        vy = float(self.car_cfg.edge_slide_vy_norm) * (1.0 if int(direction_sign) >= 0 else -1.0)
        return MotionDecision(cmd=self._cmd("EDGE_SLIDE_SEARCH", vy=vy), cx_norm_abs=abs(vy), distance_ratio=1.0)

    def avoid_cmd(self, turn_dir: Optional[str]) -> MotionDecision:
        turn_dir = str(turn_dir or "").strip().lower()
        vx = 0.0
        vy = 0.0
        wz = 0.0
        if turn_dir in {"left", "l"}:
            wz = abs(float(self.car_cfg.avoid_turn_norm))
        elif turn_dir in {"right", "r"}:
            wz = -abs(float(self.car_cfg.avoid_turn_norm))
        elif turn_dir in {"back", "backward", "reverse"}:
            vx = -abs(float(self.car_cfg.avoid_reverse_vx_norm))
        else:
            wz = abs(float(self.car_cfg.avoid_turn_norm))
        return MotionDecision(cmd=self._cmd("AVOID_OBSTACLE", vx=vx, vy=vy, wz=wz), cx_norm_abs=abs(wz), distance_ratio=max(0.0, 1.0 - abs(vx)))

    def _fallback_table_cmd(self, obs: Optional[TableEdgeObs], mode: str, allow_forward: bool) -> MotionDecision:
        if obs is None:
            return MotionDecision(cmd=self._cmd(mode))
        cx = self._clamp(float(obs.table_cx_norm or 0.0), -1.0, 1.0)
        size = self._clamp(float(obs.table_size_norm or 0.0), 0.0, 1.0)
        x_abs = abs(cx)
        distance_ratio = max(0.0, min(1.0, 1.0 - size))

        if x_abs <= float(self.car_cfg.fallback_dead_zone_x):
            target_wz = 0.0
        else:
            turn_ratio = (x_abs - float(self.car_cfg.fallback_dead_zone_x)) / max(1e-6, 1.0 - float(self.car_cfg.fallback_dead_zone_x))
            wz_mag = self._scaled_turn(turn_ratio)
            target_wz = -wz_mag if cx > 0 else wz_mag

        if not allow_forward:
            target_vx = 0.0
        elif x_abs >= float(self.car_cfg.fallback_spin_only_x_th):
            target_vx = 0.0
        else:
            align_factor = max(0.0, 1.0 - (x_abs / max(1e-6, float(self.car_cfg.fallback_spin_only_x_th))))
            align_factor = align_factor ** max(1.0, float(self.car_cfg.fallback_forward_align_exp))
            forward_ratio = self._clamp(distance_ratio * align_factor, 0.0, 1.0)
            target_vx = 0.0 if forward_ratio < 0.05 else self._scaled_forward(forward_ratio)

        vx = self._blend(self._last_fallback_vx, target_vx, 0.45)
        wz = self._blend(self._last_fallback_wz, target_wz, 0.55)
        if abs(vx) < 0.02:
            vx = 0.0
        if abs(wz) < 0.02:
            wz = 0.0
        self._last_fallback_vx, self._last_fallback_wz = vx, wz
        return MotionDecision(cmd=self._cmd(mode, vx=vx, wz=wz), cx_norm_abs=x_abs, distance_ratio=distance_ratio)

    def _edge_obs_from_table(self, obs: Optional[TableEdgeObs]) -> Optional[EdgeControlObservation]:
        if obs is None:
            return None
        if obs.yaw_err_rad is None or obs.dist_err_m is None:
            return None
        return EdgeControlObservation(
            ts=float(obs.ts),
            edge_found=bool(obs.edge_found),
            confidence=float(obs.confidence or 0.0),
            yaw_err_rad=float(obs.yaw_err_rad),
            dist_err_m=float(obs.dist_err_m),
            lateral_err_m=(float(obs.lateral_err_m) if obs.lateral_err_m is not None else None),
            edge_ready=bool(obs.edge_ready),
            source=str(obs.source or "vision"),
        )

    def _from_docking_cmd(self, mode: str, obs: Optional[TableEdgeObs], fallback_forward: bool) -> MotionDecision:
        edge_obs = self._edge_obs_from_table(obs)
        if edge_obs is None:
            return self._fallback_table_cmd(obs, mode=mode, allow_forward=fallback_forward)
        out = self.docking.update(mode, edge_obs)
        if not out.valid:
            return self._fallback_table_cmd(obs, mode=mode, allow_forward=fallback_forward)
        return MotionDecision(
            cmd=self._cmd(mode, vx=out.vx, vy=out.vy, wz=out.wz),
            cx_norm_abs=abs(float(edge_obs.yaw_err_rad or 0.0)),
            distance_ratio=abs(float(edge_obs.dist_err_m or 0.0)),
        )

    def coarse_align_cmd(self, obs: Optional[TableEdgeObs]) -> MotionDecision:
        return self._from_docking_cmd("COARSE_ALIGN", obs, fallback_forward=False)

    def controlled_approach_cmd(self, obs: Optional[TableEdgeObs]) -> MotionDecision:
        return self._from_docking_cmd("CONTROLLED_APPROACH", obs, fallback_forward=True)

    def final_lock_cmd(self, obs: Optional[TableEdgeObs]) -> MotionDecision:
        return self._from_docking_cmd("FINAL_LOCK", obs, fallback_forward=True)

    def target_track_cmd(self, obs: Optional[TargetObs]) -> MotionDecision:
        if obs is None:
            return MotionDecision(cmd=self._cmd("EDGE_SLIDE_SEARCH"))
        explicit = self._extract_norm_triplet(obs.vx_norm, obs.vy_norm, obs.wz_norm)
        if explicit is None:
            return MotionDecision(cmd=self._cmd("EDGE_SLIDE_SEARCH"))
        vx, vy, wz = explicit
        return MotionDecision(
            cmd=self._cmd("EDGE_SLIDE_SEARCH", vx=vx, vy=vy, wz=wz),
            cx_norm_abs=abs(float(obs.cx_norm)),
            distance_ratio=max(0.0, min(1.0, 1.0 - float(obs.size_norm))),
        )

    def return_hold_cmd(self) -> MotionDecision:
        return MotionDecision(cmd=self._cmd("RETURN_HOME"))

    def return_cmd(self, obs: HomeTagObs) -> MotionDecision:
        explicit = self._extract_norm_triplet(obs.vx_norm, obs.vy_norm, obs.wz_norm)
        if explicit is not None:
            vx, vy, wz = explicit
            return MotionDecision(
                cmd=self._cmd("RETURN_HOME", vx=vx, vy=vy, wz=wz),
                cx_norm_abs=abs(float(obs.yaw_err_rad)),
                distance_ratio=float(obs.distance_m or 0.0),
            )
        yaw = float(obs.yaw_err_rad)
        distance = float(obs.distance_m or 0.0)
        ratio = min(1.0, max(0.0, distance / 1.2 if distance > 0 else 0.3))
        if abs(yaw) > 0.18:
            wz_mag = float(self.car_cfg.return_turn_norm_min) + (
                float(self.car_cfg.return_turn_norm_max) - float(self.car_cfg.return_turn_norm_min)
            ) * min(1.0, abs(yaw))
            return MotionDecision(
                cmd=self._cmd("RETURN_HOME", wz=wz_mag if yaw > 0 else -wz_mag),
                cx_norm_abs=abs(yaw),
                distance_ratio=ratio,
            )
        vx = float(self.car_cfg.return_vx_norm_min) + (
            float(self.car_cfg.return_vx_norm_max) - float(self.car_cfg.return_vx_norm_min)
        ) * ratio
        return MotionDecision(
            cmd=self._cmd("RETURN_HOME", vx=vx),
            cx_norm_abs=abs(yaw),
            distance_ratio=ratio,
        )
