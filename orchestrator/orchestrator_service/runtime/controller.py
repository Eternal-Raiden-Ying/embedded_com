#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..config.schema import CarMotionConfig, ControlThresholds
from ..control.docking_controller import DockingController
from ..control.types import DockingControlConfig, EdgeControlObservation
from ..ipc.protocol import ArmCommand, CmdVel, HomeTagObs, TableEdgeObs, TargetObs, now_ts


@dataclass
class MotionDecision:
    cmd: CmdVel
    cx_norm_abs: float = 0.0
    distance_ratio: float = 0.0
    control_summary: Optional[Dict[str, Any]] = None
    arm_cmd: Optional[ArmCommand] = None
    jog_action: str = ""
    jog_reason: str = ""


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

    def _summary(
        self,
        mode: str,
        cmd: CmdVel,
        obs: Optional[TableEdgeObs] = None,
        *,
        reason: str = "",
        lock_ready: Optional[bool] = None,
        lock_reason: str = "",
        edge_found: Optional[bool] = None,
    ) -> Dict[str, Any]:
        measured_distance = None
        target_distance = None
        if obs is not None:
            target_distance = obs.target_dist_m
            if obs.dist_err_m is not None and target_distance is not None:
                measured_distance = float(target_distance) + float(obs.dist_err_m)
        return {
            "state": mode,
            "edge_found": bool(edge_found if edge_found is not None else (obs.edge_found if obs is not None else False)),
            "edge_valid": bool(getattr(obs, "edge_valid", obs.edge_found) if obs is not None else False),
            "confidence": (float(obs.confidence) if obs is not None and obs.confidence is not None else None),
            "edge_conf": (float(getattr(obs, "edge_conf", obs.confidence)) if obs is not None and getattr(obs, "edge_conf", obs.confidence) is not None else None),
            "yaw_err_rad": (float(obs.yaw_err_rad) if obs is not None and obs.yaw_err_rad is not None else None),
            "dist_err_m": (float(obs.dist_err_m) if obs is not None and obs.dist_err_m is not None else None),
            "target_dist_m": target_distance,
            "measured_distance_m": measured_distance,
            "lock_ready": lock_ready,
            "lock_reason": lock_reason or reason,
            "reason": reason or lock_reason,
            "cmd": {
                "vx": float(cmd.vx_norm),
                "vy": float(cmd.vy_norm),
                "wz": float(cmd.wz_norm),
                "hold_ms": int(cmd.hold_ms),
            },
        }

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
        cmd = self._cmd(mode, brake=brake)
        return MotionDecision(cmd=cmd, control_summary=self._summary(mode, cmd, reason="stop"))

    def search_table_cmd(self, turn_sign: int = 1) -> MotionDecision:
        self._reset_fallback_memory()
        wz = float(self.car_cfg.search_table_wz_norm) * (1.0 if int(turn_sign) >= 0 else -1.0)
        cmd = self._cmd("SEARCH_TABLE", wz=wz)
        return MotionDecision(cmd=cmd, cx_norm_abs=abs(wz), distance_ratio=1.0, control_summary=self._summary("SEARCH_TABLE", cmd, reason="search_table"))

    def next_table_cmd(self, turn_sign: int = 1) -> MotionDecision:
        return self.search_table_cmd(turn_sign=turn_sign)

    def leave_edge_cmd(self) -> MotionDecision:
        self._reset_fallback_memory()
        cmd = self._cmd("LEAVE_EDGE", vx=float(self.car_cfg.leave_edge_vx_norm))
        return MotionDecision(cmd=cmd, control_summary=self._summary("LEAVE_EDGE", cmd, reason="leave_edge"))

    def relocate_cmd(self, turn_sign: int = 1) -> MotionDecision:
        self._reset_fallback_memory()
        wz = float(self.car_cfg.relocate_turn_wz_norm) * (1.0 if int(turn_sign) >= 0 else -1.0)
        cmd = self._cmd("RELOCATE_TO_EDGE", wz=wz)
        return MotionDecision(cmd=cmd, cx_norm_abs=abs(wz), control_summary=self._summary("RELOCATE_TO_EDGE", cmd, reason="relocate"))

    def edge_slide_search_cmd(
        self,
        elapsed_s: float,
        direction_sign: int = 1,
        edge_obs: Optional[TableEdgeObs] = None,
        vy_norm: Optional[float] = None,
        reason: str = "edge_slide",
    ) -> MotionDecision:
        if float(elapsed_s) < float(self.cfg.edge_slide_pause_s):
            cmd = self._cmd("EDGE_SLIDE_SEARCH")
            summary = self._summary("EDGE_SLIDE_SEARCH", cmd, edge_obs, reason="waiting_first_target_obs")
            summary.update(
                {
                    "slide_vy_norm": float(self.car_cfg.edge_slide_vy_norm),
                    "weak_slide_vy_norm": float(self.car_cfg.edge_slide_weak_vy_norm),
                    "vx_from_dist": 0.0,
                    "wz_from_yaw": 0.0,
                    "final_vx": 0.0,
                    "final_vy": 0.0,
                    "final_wz": 0.0,
                    "stop_reason": "initial_pause",
                }
            )
            return MotionDecision(
                cmd=cmd,
                control_summary=summary,
            )
        base_vy = float(self.car_cfg.edge_slide_vy_norm if vy_norm is None else vy_norm)
        vy = base_vy * (1.0 if int(direction_sign) >= 0 else -1.0)
        vx = 0.0
        wz = 0.0
        vx_from_dist = 0.0
        wz_from_yaw = 0.0
        if edge_obs is not None:
            dist_err = float(edge_obs.dist_err_m or 0.0)
            yaw_err = float(edge_obs.yaw_err_rad or 0.0)
            vx_from_dist = self._clamp(
                dist_err * float(self.car_cfg.edge_slide_dist_kp_norm_per_m),
                -abs(float(self.car_cfg.edge_slide_max_vx_norm)),
                abs(float(self.car_cfg.edge_slide_max_vx_norm)),
            )
            wz_from_yaw = self._clamp(
                yaw_err * float(self.car_cfg.edge_slide_yaw_kp_norm_per_rad),
                -abs(float(self.car_cfg.edge_slide_max_wz_norm)),
                abs(float(self.car_cfg.edge_slide_max_wz_norm)),
            )
            vx = vx_from_dist
            wz = wz_from_yaw
            if abs(vx) < 0.01:
                vx = 0.0
            if abs(wz) < 0.01:
                wz = 0.0
        reason = "edge_slide_vy_zero" if abs(vy) <= 1e-9 else (str(reason or "edge_slide"))
        cmd = self._cmd("EDGE_SLIDE_SEARCH", vx=vx, vy=vy, wz=wz)
        summary = self._summary("EDGE_SLIDE_SEARCH", cmd, edge_obs, reason=reason)
        summary.update(
            {
                "slide_vy_norm": float(self.car_cfg.edge_slide_vy_norm),
                "weak_slide_vy_norm": float(self.car_cfg.edge_slide_weak_vy_norm),
                "vx_from_dist": float(vx_from_dist),
                "wz_from_yaw": float(wz_from_yaw),
                "final_vx": float(vx),
                "final_vy": float(vy),
                "final_wz": float(wz),
                "stop_reason": "" if abs(vy) > 1e-9 else "vy_zero",
            }
        )
        return MotionDecision(
            cmd=cmd,
            cx_norm_abs=abs(vy),
            distance_ratio=abs(float(edge_obs.dist_err_m or 0.0)) if edge_obs is not None else 1.0,
            control_summary=summary,
        )

    def edge_slide_hold_cmd(self, reason: str = "safety_hold_no_edge", edge_obs: Optional[TableEdgeObs] = None) -> MotionDecision:
        cmd = self._cmd("EDGE_SLIDE_SEARCH")
        summary = self._summary("EDGE_SLIDE_SEARCH", cmd, edge_obs, reason=reason, edge_found=False if edge_obs is None else None)
        summary.update(
            {
                "slide_vy_norm": float(self.car_cfg.edge_slide_vy_norm),
                "weak_slide_vy_norm": float(self.car_cfg.edge_slide_weak_vy_norm),
                "vx_from_dist": 0.0,
                "wz_from_yaw": 0.0,
                "final_vx": 0.0,
                "final_vy": 0.0,
                "final_wz": 0.0,
                "stop_reason": str(reason or "hold"),
            }
        )
        return MotionDecision(cmd=cmd, control_summary=summary)

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
        cmd = self._cmd("AVOID_OBSTACLE", vx=vx, vy=vy, wz=wz)
        return MotionDecision(cmd=cmd, cx_norm_abs=abs(wz), distance_ratio=max(0.0, 1.0 - abs(vx)), control_summary=self._summary("AVOID_OBSTACLE", cmd, reason="avoid_obstacle"))

    def _fallback_table_cmd(self, obs: Optional[TableEdgeObs], mode: str, allow_forward: bool) -> MotionDecision:
        if obs is None:
            cmd = self._cmd(mode)
            return MotionDecision(cmd=cmd, control_summary=self._summary(mode, cmd, reason="edge_missing"))
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
        cmd = self._cmd(mode, vx=vx, wz=wz)
        reason = "fallback_table"
        return MotionDecision(cmd=cmd, cx_norm_abs=x_abs, distance_ratio=distance_ratio, control_summary=self._summary(mode, cmd, obs, reason=reason))

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
        cmd = self._cmd(mode, vx=out.vx, vy=out.vy, wz=out.wz)
        return MotionDecision(
            cmd=cmd,
            cx_norm_abs=abs(float(edge_obs.yaw_err_rad or 0.0)),
            distance_ratio=abs(float(edge_obs.dist_err_m or 0.0)),
            control_summary=self._summary(mode, cmd, obs, reason="docking_control"),
        )

    def coarse_align_cmd(self, obs: Optional[TableEdgeObs]) -> MotionDecision:
        return self._from_docking_cmd("COARSE_ALIGN", obs, fallback_forward=False)

    def controlled_approach_cmd(self, obs: Optional[TableEdgeObs]) -> MotionDecision:
        return self._from_docking_cmd("CONTROLLED_APPROACH", obs, fallback_forward=True)

    def final_lock_cmd(self, obs: Optional[TableEdgeObs]) -> MotionDecision:
        return self._from_docking_cmd("FINAL_LOCK", obs, fallback_forward=True)

    def target_track_cmd(self, obs: Optional[TargetObs]) -> MotionDecision:
        if obs is None:
            cmd = self._cmd("EDGE_SLIDE_SEARCH")
            return MotionDecision(cmd=cmd, control_summary=self._summary("EDGE_SLIDE_SEARCH", cmd, reason="edge_missing"))
        explicit = self._extract_norm_triplet(obs.vx_norm, obs.vy_norm, obs.wz_norm)
        if explicit is None:
            cmd = self._cmd("EDGE_SLIDE_SEARCH")
            return MotionDecision(cmd=cmd, control_summary=self._summary("EDGE_SLIDE_SEARCH", cmd, reason="target_confirming"))
        vx, vy, wz = explicit
        cmd = self._cmd("EDGE_SLIDE_SEARCH", vx=vx, vy=vy, wz=wz)
        return MotionDecision(
            cmd=cmd,
            cx_norm_abs=abs(float(obs.cx_norm)),
            distance_ratio=max(0.0, min(1.0, 1.0 - float(obs.size_norm))),
            control_summary={
                **self._summary("EDGE_SLIDE_SEARCH", cmd, reason="target_track"),
                "confidence": obs.confidence,
            },
        )

    def return_hold_cmd(self) -> MotionDecision:
        cmd = self._cmd("RETURN_HOME")
        return MotionDecision(cmd=cmd, control_summary=self._summary("RETURN_HOME", cmd, reason="return_hold"))

    def return_cmd(self, obs: HomeTagObs) -> MotionDecision:
        explicit = self._extract_norm_triplet(obs.vx_norm, obs.vy_norm, obs.wz_norm)
        if explicit is not None:
            vx, vy, wz = explicit
            cmd = self._cmd("RETURN_HOME", vx=vx, vy=vy, wz=wz)
            return MotionDecision(
                cmd=cmd,
                cx_norm_abs=abs(float(obs.yaw_err_rad)),
                distance_ratio=float(obs.distance_m or 0.0),
                control_summary=self._summary("RETURN_HOME", cmd, reason="return_track"),
            )
        yaw = float(obs.yaw_err_rad)
        distance = float(obs.distance_m or 0.0)
        ratio = min(1.0, max(0.0, distance / 1.2 if distance > 0 else 0.3))
        if abs(yaw) > 0.18:
            wz_mag = float(self.car_cfg.return_turn_norm_min) + (
                float(self.car_cfg.return_turn_norm_max) - float(self.car_cfg.return_turn_norm_min)
            ) * min(1.0, abs(yaw))
            cmd = self._cmd("RETURN_HOME", wz=wz_mag if yaw > 0 else -wz_mag)
            return MotionDecision(
                cmd=cmd,
                cx_norm_abs=abs(yaw),
                distance_ratio=ratio,
                control_summary=self._summary("RETURN_HOME", cmd, reason="return_yaw_align"),
            )
        vx = float(self.car_cfg.return_vx_norm_min) + (
            float(self.car_cfg.return_vx_norm_max) - float(self.car_cfg.return_vx_norm_min)
        ) * ratio
        cmd = self._cmd("RETURN_HOME", vx=vx)
        return MotionDecision(
            cmd=cmd,
            cx_norm_abs=abs(yaw),
            distance_ratio=ratio,
            control_summary=self._summary("RETURN_HOME", cmd, reason="return_approach"),
        )
