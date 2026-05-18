#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
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
        self._last_table_vx = 0.0
        self._last_table_vy = 0.0
        self._last_table_wz = 0.0
        self._last_view_err_norm = 0.0
        self._last_view_ts = 0.0

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
            "control_level": (str(getattr(obs, "control_level", "none") or "none") if obs is not None else "none"),
            "usable_for_approach": bool(getattr(obs, "usable_for_approach", False)) if obs is not None else False,
            "usable_for_alignment": bool(getattr(obs, "usable_for_alignment", False)) if obs is not None else False,
            "usable_for_stop": bool(getattr(obs, "usable_for_stop", False)) if obs is not None else False,
            "table_approach_phase": (str(getattr(obs, "table_approach_phase", "") or "") if obs is not None else ""),
            "view_source": (str(getattr(obs, "view_source", "") or "") if obs is not None else ""),
            "view_err_norm": (float(getattr(obs, "view_err_norm")) if obs is not None and getattr(obs, "view_err_norm", None) is not None else None),
            "view_reliable": bool(getattr(obs, "view_reliable", False)) if obs is not None else False,
            "fov_guard_active": bool(getattr(obs, "fov_guard_active", False)) if obs is not None else False,
            "fov_guard_reason": str(getattr(obs, "fov_guard_reason", "") or "") if obs is not None else "",
            "table_confirmed_by_yolo": bool(getattr(obs, "table_confirmed_by_yolo", False)) if obs is not None else False,
            "yolo_reliable": bool(getattr(obs, "yolo_reliable", False)) if obs is not None else False,
            "plane_cx_norm": (float(getattr(obs, "plane_cx_norm")) if obs is not None and getattr(obs, "plane_cx_norm", None) is not None else None),
            "plane_width_norm": (float(getattr(obs, "plane_width_norm")) if obs is not None and getattr(obs, "plane_width_norm", None) is not None else None),
            "plane_touch_left": bool(getattr(obs, "plane_touch_left", False)) if obs is not None else False,
            "plane_touch_right": bool(getattr(obs, "plane_touch_right", False)) if obs is not None else False,
            "table_view_wz_sign": float(getattr(self.car_cfg, "table_view_wz_sign", -1.0)),
            "table_view_vy_sign": float(getattr(self.car_cfg, "table_view_vy_sign", -1.0)),
            "table_plane_yaw_sign": float(getattr(self.car_cfg, "table_plane_yaw_sign", 1.0)),
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
        self._last_table_vx = 0.0
        self._last_table_vy = 0.0
        self._last_table_wz = 0.0

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

    def _plane_stable(self, obs: Optional[TableEdgeObs]) -> bool:
        if obs is None or obs.yaw_err_rad is None or obs.dist_err_m is None:
            return False
        if bool(getattr(obs, "usable_for_approach", False)):
            return True
        if bool(getattr(obs, "usable_for_alignment", False)) or bool(getattr(obs, "usable_for_stop", False)):
            return True
        level = str(getattr(obs, "control_level", "") or "").strip().lower()
        if level in {"approach", "alignment", "stop"}:
            return True
        return bool(getattr(obs, "edge_found", False)) and bool(getattr(obs, "edge_valid", True))

    def _yolo_reliable(self, obs: Optional[TableEdgeObs]) -> bool:
        if obs is None:
            return False
        if hasattr(obs, "yolo_reliable"):
            return bool(getattr(obs, "yolo_reliable", False))
        return bool(getattr(obs, "table_confirmed_by_yolo", False)) and obs.table_cx_norm is not None

    def _table_approach_phase(self, obs: Optional[TableEdgeObs], requested: str = "") -> str:
        req = str(requested or "").strip().upper()
        if req in {"PLANE_ACQUIRE", "PLANE_APPROACH", "PLANE_FINAL_LOCK", "PLANE_STOP"}:
            return req
        level = str(getattr(obs, "control_level", "none") or "none").strip().lower() if obs is not None else "none"
        if level == "stop":
            return "PLANE_STOP"
        if level == "alignment":
            return "PLANE_FINAL_LOCK"
        if level == "approach":
            return "PLANE_APPROACH"
        plane = self._plane_stable(obs)
        if plane:
            return "PLANE_FINAL_LOCK"
        return "PLANE_ACQUIRE"

    def _get_view_error(self, obs: Optional[TableEdgeObs]) -> Tuple[float, str, bool]:
        if obs is None:
            return 0.0, "none", False
        if getattr(obs, "plane_cx_norm", None) is not None and (
            self._plane_stable(obs) or bool(getattr(obs, "usable_for_approach", False))
        ):
            err = self._clamp(float(getattr(obs, "plane_cx_norm")), -1.0, 1.0)
            self._last_view_err_norm = err
            self._last_view_ts = time.time()
            return err, "plane", True
        if self._yolo_reliable(obs) and obs.table_cx_norm is not None:
            err = self._clamp(float(obs.table_cx_norm), -1.0, 1.0)
            self._last_view_err_norm = err
            self._last_view_ts = time.time()
            return err, "yolo", True
        if getattr(obs, "view_reliable", False) and getattr(obs, "view_err_norm", None) is not None:
            err = self._clamp(float(getattr(obs, "view_err_norm")), -1.0, 1.0)
            source = str(getattr(obs, "view_source", "") or "vision").strip().lower() or "vision"
            self._last_view_err_norm = err
            self._last_view_ts = time.time()
            return err, source, True
        ttl = float(getattr(self.car_cfg, "table_view_memory_ttl_s", 0.8) or 0.8)
        if self._last_view_ts > 0.0 and time.time() - self._last_view_ts <= ttl:
            return self._clamp(self._last_view_err_norm, -1.0, 1.0), "memory", True
        return 0.0, "none", False

    def _slew_table_axis(self, prev: float, target: float, rate: float, dt: float) -> float:
        max_delta = max(0.0, float(rate)) * max(1e-3, float(dt))
        delta = float(target) - float(prev)
        if delta > max_delta:
            return float(prev) + max_delta
        if delta < -max_delta:
            return float(prev) - max_delta
        return float(target)

    def _limit_table_cmd(self, vx: float, vy: float, wz: float) -> Tuple[float, float, float]:
        now_s = time.time()
        last_ts = getattr(self.docking, "_last_ts", 0.0) or now_s
        dt = max(0.05, min(0.25, now_s - float(last_ts)))
        vx = self._clamp(vx, -abs(float(getattr(self.car_cfg, "table_stage_c_vx_max_norm", 0.05))), abs(float(getattr(self.car_cfg, "table_stage_c_vx_max_norm", 0.05))))
        vy = self._clamp(vy, -abs(float(getattr(self.car_cfg, "table_vy_max_norm", 0.02))), abs(float(getattr(self.car_cfg, "table_vy_max_norm", 0.02))))
        wz_max = max(
            abs(float(getattr(self.car_cfg, "table_wz_view_max_norm", 0.10))),
            abs(float(getattr(self.car_cfg, "table_wz_plane_max_norm", 0.14))),
            abs(float(getattr(self.car_cfg, "table_stage_a_wz_norm", 0.08))),
        )
        wz = self._clamp(wz, -wz_max, wz_max)
        vx2 = self._slew_table_axis(self._last_table_vx, vx, getattr(self.car_cfg, "table_vx_slew_per_s", 0.25), dt)
        vy2 = self._slew_table_axis(self._last_table_vy, vy, getattr(self.car_cfg, "table_vy_slew_per_s", 0.12), dt)
        wz2 = self._slew_table_axis(self._last_table_wz, wz, getattr(self.car_cfg, "table_wz_slew_per_s", 0.35), dt)
        self._last_table_vx, self._last_table_vy, self._last_table_wz = vx2, vy2, wz2
        return vx2, vy2, wz2

    def _compose_fov_table_cmd(
        self,
        obs: Optional[TableEdgeObs],
        mode: str,
        *,
        phase: str = "",
        base_vx: Optional[float] = None,
        base_vy: float = 0.0,
        base_wz: Optional[float] = None,
        reason: str = "fov_table_approach",
    ) -> MotionDecision:
        view_err, view_source, view_reliable = self._get_view_error(obs)
        obs_view_reliable = bool(getattr(obs, "view_reliable", False)) if obs is not None else False
        view_valid_for_forward = bool(view_reliable and obs_view_reliable)
        phase_name = self._table_approach_phase(obs, phase)
        soft_th = abs(float(getattr(self.car_cfg, "table_fov_soft_th", 0.25) or 0.25))
        hard_th = abs(float(getattr(self.car_cfg, "table_fov_hard_th", 0.40) or 0.40))
        plane_touch_left = bool(getattr(obs, "plane_touch_left", False)) if obs is not None else False
        plane_touch_right = bool(getattr(obs, "plane_touch_right", False)) if obs is not None else False
        obs_stale = bool(getattr(obs, "is_stale", False)) if obs is not None else False
        guard_view_err = view_err
        if obs is not None and getattr(obs, "view_err_norm", None) is not None:
            try:
                guard_view_err = self._clamp(float(getattr(obs, "view_err_norm")), -1.0, 1.0)
            except Exception:
                guard_view_err = view_err
        fov_guard_reason = ""
        if plane_touch_right:
            fov_guard_reason = "plane_touch_right"
        elif plane_touch_left:
            fov_guard_reason = "plane_touch_left"
        elif abs(float(guard_view_err)) > hard_th:
            fov_guard_reason = "view_err_hard"
        hard_guard = bool(fov_guard_reason)

        dist_err = float(obs.dist_err_m) if obs is not None and obs.dist_err_m is not None else 0.0
        yaw_err = float(obs.yaw_err_rad) if obs is not None and obs.yaw_err_rad is not None else 0.0
        vx_from_dist = 0.0
        if base_vx is not None:
            vx_from_dist = float(base_vx)
        elif phase_name not in {"PLANE_ACQUIRE", "PLANE_STOP"} and dist_err > float(self.cfg.final_lock_dist_tol_m):
            vx_from_dist = dist_err * float(getattr(self.car_cfg, "table_dist_kp_norm_per_m", 0.12))
            max_vx = float(
                getattr(
                    self.car_cfg,
                    "table_stage_b_vx_max_norm" if phase_name == "PLANE_APPROACH" else "table_stage_c_vx_max_norm",
                    0.05,
                )
            )
            min_vx = min(abs(max_vx), abs(float(getattr(self.car_cfg, "table_stage_c_vx_min_norm", 0.04) or 0.04)))
            vx_from_dist = self._clamp(vx_from_dist, min_vx, abs(max_vx))

        wz_from_plane = 0.0
        if base_wz is not None:
            wz_from_plane = float(base_wz)
        elif phase_name not in {"PLANE_ACQUIRE", "PLANE_STOP"} and obs is not None and obs.yaw_err_rad is not None:
            wz_from_plane = yaw_err * float(getattr(self.car_cfg, "table_plane_yaw_kp_norm_per_rad", 0.60))
            wz_from_plane *= float(getattr(self.car_cfg, "table_plane_yaw_sign", 1.0))
            wz_from_plane = self._clamp(
                wz_from_plane,
                -abs(float(getattr(self.car_cfg, "table_wz_plane_max_norm", 0.14))),
                abs(float(getattr(self.car_cfg, "table_wz_plane_max_norm", 0.14))),
            )

        wz_from_view = 0.0
        vy_from_view = 0.0
        if view_valid_for_forward and phase_name not in {"PLANE_ACQUIRE", "PLANE_STOP"}:
            wz_from_view = view_err * float(getattr(self.car_cfg, "table_view_wz_kp", 0.18))
            wz_from_view *= float(getattr(self.car_cfg, "table_view_wz_sign", -1.0))
            wz_from_view = self._clamp(
                wz_from_view,
                -abs(float(getattr(self.car_cfg, "table_wz_view_max_norm", 0.10))),
                abs(float(getattr(self.car_cfg, "table_wz_view_max_norm", 0.10))),
            )
            vy_from_view = view_err * float(getattr(self.car_cfg, "table_view_vy_kp", 0.04))
            vy_from_view *= float(getattr(self.car_cfg, "table_view_vy_sign", -1.0))
            vy_from_view = self._clamp(
                vy_from_view,
                -abs(float(getattr(self.car_cfg, "table_vy_max_norm", 0.02))),
                abs(float(getattr(self.car_cfg, "table_vy_max_norm", 0.02))),
            )

        yaw_gate = max(0.0, 1.0 - min(1.0, abs(yaw_err) / 0.45))
        if phase_name == "PLANE_APPROACH":
            yaw_gate = max(0.35, yaw_gate)
        fov_gate = 0.0 if hard_guard else 1.0
        if view_valid_for_forward and abs(view_err) > soft_th and hard_th > soft_th:
            fov_gate = min(fov_gate, max(0.0, 1.0 - ((abs(view_err) - soft_th) / (hard_th - soft_th))))
        if not view_valid_for_forward:
            fov_gate = 0.0
        near_gate = 1.0
        if dist_err <= float(self.cfg.final_lock_dist_tol_m):
            near_gate = 0.0
        elif dist_err < float(self.cfg.final_lock_dist_tol_m) * 3.0:
            near_gate = max(0.25, dist_err / max(1e-6, float(self.cfg.final_lock_dist_tol_m) * 3.0))

        vx = 0.0 if phase_name in {"PLANE_ACQUIRE", "PLANE_STOP"} else vx_from_dist * yaw_gate * fov_gate * near_gate
        vy = float(base_vy or 0.0) + vy_from_view
        wz = wz_from_plane + wz_from_view
        if phase_name in {"PLANE_ACQUIRE", "PLANE_STOP"}:
            vy = 0.0
            wz = 0.0
        if hard_guard:
            vx = 0.0
            recover_err = float(guard_view_err)
            if plane_touch_right and recover_err <= 0.0:
                recover_err = 1.0
            elif plane_touch_left and recover_err >= 0.0:
                recover_err = -1.0
            recover_vy = abs(float(getattr(self.car_cfg, "table_view_recover_vy_norm", 0.008) or 0.008))
            recover_wz = abs(float(getattr(self.car_cfg, "table_view_recover_wz_norm", 0.04) or 0.04))
            recover_vy = min(recover_vy, abs(float(getattr(self.car_cfg, "table_vy_max_norm", 0.02) or 0.02)))
            recover_wz = min(recover_wz, abs(float(getattr(self.car_cfg, "table_wz_view_max_norm", 0.08) or 0.08)))
            direction = 1.0 if recover_err >= 0.0 else -1.0
            vy = direction * recover_vy * float(getattr(self.car_cfg, "table_view_vy_sign", -1.0))
            wz = direction * recover_wz * float(getattr(self.car_cfg, "table_view_wz_sign", -1.0))
        if (not view_valid_for_forward and phase_name not in {"PLANE_ACQUIRE"}) or obs_stale:
            vx = 0.0
            if obs_stale:
                vy = 0.0
                wz = 0.0
        vx, vy, wz = self._limit_table_cmd(vx, vy, wz)
        if hard_guard or (not view_valid_for_forward and phase_name not in {"PLANE_ACQUIRE"}) or obs_stale:
            vx = 0.0
            self._last_table_vx = 0.0
            if obs_stale:
                vy = 0.0
                wz = 0.0
            elif hard_guard:
                vy = direction * recover_vy * float(getattr(self.car_cfg, "table_view_vy_sign", -1.0))
                wz = direction * recover_wz * float(getattr(self.car_cfg, "table_view_wz_sign", -1.0))
        if abs(vx) < 0.005:
            vx = 0.0
        if abs(vy) < 0.005:
            vy = 0.0
        if abs(wz) < 0.005:
            wz = 0.0

        cmd = self._cmd(mode, vx=vx, vy=vy, wz=wz)
        summary = self._summary(mode, cmd, obs, reason=reason)
        summary.update(
            {
                "table_approach_phase": phase_name,
                "table_approach_reason": "plane_confirmed_table_front" if self._plane_stable(obs) else "plane_waiting",
                "approach_source": "plane_only",
                "view_source": view_source,
                "view_err_norm": float(view_err),
                "view_reliable": bool(obs_view_reliable),
                "view_inferred_reliable": bool(view_reliable),
                "fov_guard_active": bool(hard_guard),
                "fov_guard_reason": fov_guard_reason,
                "fov_guard_view_err_norm": float(guard_view_err),
                "obs_stale": bool(obs_stale),
                "vx_from_dist": float(vx_from_dist),
                "vy_from_view": float(vy_from_view),
                "wz_from_plane": float(wz_from_plane),
                "wz_from_view": float(wz_from_view),
                "yaw_gate": float(yaw_gate),
                "fov_gate": float(fov_gate),
                "near_gate": float(near_gate),
                "final_vx": float(vx),
                "final_vy": float(vy),
                "final_wz": float(wz),
                "table_confirmed_by_yolo": bool(getattr(obs, "table_confirmed_by_yolo", False)) if obs is not None else False,
                "yolo_reliable": bool(getattr(obs, "yolo_reliable", False)) if obs is not None else False,
                "yolo_gate_open": bool(getattr(obs, "yolo_gate_open", False)) if obs is not None else False,
                "plane_cx_norm": getattr(obs, "plane_cx_norm", None) if obs is not None else None,
                "plane_width_norm": getattr(obs, "plane_width_norm", None) if obs is not None else None,
                "plane_touch_left": bool(getattr(obs, "plane_touch_left", False)) if obs is not None else False,
                "plane_touch_right": bool(getattr(obs, "plane_touch_right", False)) if obs is not None else False,
            }
        )
        return MotionDecision(
            cmd=cmd,
            cx_norm_abs=abs(float(view_err)),
            distance_ratio=max(0.0, min(1.0, abs(dist_err))),
            control_summary=summary,
        )

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
            if mode in {"CONTROLLED_APPROACH", "FINAL_LOCK"}:
                return self._compose_fov_table_cmd(obs, mode, phase="", reason="fov_fallback_table")
            return self._fallback_table_cmd(obs, mode=mode, allow_forward=fallback_forward)
        out = self.docking.update(mode, edge_obs)
        if not out.valid:
            if mode in {"CONTROLLED_APPROACH", "FINAL_LOCK"}:
                return self._compose_fov_table_cmd(obs, mode, phase="", reason="fov_docking_invalid")
            return self._fallback_table_cmd(obs, mode=mode, allow_forward=fallback_forward)
        if mode in {"CONTROLLED_APPROACH", "FINAL_LOCK"}:
            return self._compose_fov_table_cmd(
                obs,
                mode,
                base_vx=out.vx,
                base_vy=out.vy,
                base_wz=out.wz,
                reason="docking_control_fov",
            )
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

    def fov_table_approach_cmd(self, obs: Optional[TableEdgeObs], phase: str = "", mode: str = "CONTROLLED_APPROACH") -> MotionDecision:
        return self._compose_fov_table_cmd(obs, mode, phase=phase, reason="fov_table_approach")

    def plane_approach_cmd(self, obs: Optional[TableEdgeObs], mode: str = "CONTROLLED_APPROACH", reason: str = "plane_approach") -> MotionDecision:
        self.docking.reset()
        return self._compose_fov_table_cmd(obs, mode, phase="", reason=reason)

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
