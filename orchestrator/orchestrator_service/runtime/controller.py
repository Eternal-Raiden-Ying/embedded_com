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
        self._pose_missing_since_mono = 0.0

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

    def _obs_timing(self, obs: Optional[TableEdgeObs], control_ts: Optional[float] = None) -> Dict[str, Any]:
        if obs is None:
            return {
                "control_ts": control_ts,
                "obs_total_age_ms": None,
                "control_loop_age_ms": None,
                "vision_process_ms": None,
                "edge_update_interval_ms": None,
            }
        control_ts = float(control_ts if control_ts is not None else time.time())
        frame_capture_ts = getattr(obs, "frame_capture_ts", None)
        obs_total_age_ms = getattr(obs, "obs_total_age_ms", None)
        if frame_capture_ts is not None:
            try:
                obs_total_age_ms = max(0.0, (control_ts - float(frame_capture_ts)) * 1000.0)
            except Exception:
                pass
        elif obs_total_age_ms is None:
            base_ts = getattr(obs, "obs_ts", None) if getattr(obs, "obs_ts", None) is not None else getattr(obs, "ts", None)
            try:
                obs_total_age_ms = max(0.0, (control_ts - float(base_ts)) * 1000.0)
            except Exception:
                obs_total_age_ms = None
        control_loop_age_ms = None
        recv_ts = getattr(obs, "obs_recv_ts", None)
        if recv_ts is not None:
            try:
                control_loop_age_ms = max(0.0, (control_ts - float(recv_ts)) * 1000.0)
            except Exception:
                control_loop_age_ms = None
        return {
            "frame_capture_ts": getattr(obs, "frame_capture_ts", None),
            "vision_start_ts": getattr(obs, "vision_start_ts", None),
            "vision_done_ts": getattr(obs, "vision_done_ts", None),
            "obs_publish_ts": getattr(obs, "obs_publish_ts", None),
            "obs_recv_ts": getattr(obs, "obs_recv_ts", None),
            "control_ts": control_ts,
            "frame_age_ms": getattr(obs, "frame_age_ms", None),
            "vision_process_ms": getattr(obs, "vision_process_ms", getattr(obs, "edge_process_ms", None)),
            "publish_delay_ms": getattr(obs, "publish_delay_ms", None),
            "obs_total_age_ms": obs_total_age_ms,
            "control_loop_age_ms": control_loop_age_ms,
            "edge_update_interval_ms": getattr(obs, "edge_update_interval_ms", None),
        }

    def _stale_guard(self, obs: Optional[TableEdgeObs], control_ts: Optional[float] = None) -> Dict[str, Any]:
        timing = self._obs_timing(obs, control_ts)
        age_ms = timing.get("obs_total_age_ms")
        soft = float(getattr(self.cfg, "table_obs_stale_soft_ms", 300) or 300)
        stop = float(getattr(self.cfg, "table_obs_stale_stop_ms", 500) or 500)
        hard = float(getattr(self.cfg, "table_obs_stale_hard_ms", 800) or 800)
        if obs is None or age_ms is None or bool(getattr(obs, "is_stale", False)) or getattr(obs, "depth_valid", None) is False:
            level = "dead" if obs is None or age_ms is None else "hard_stale"
        elif float(age_ms) <= soft:
            level = "fresh"
        elif float(age_ms) <= stop:
            level = "soft_stale"
        elif float(age_ms) <= hard:
            level = "hard_stale"
        else:
            level = "dead"
        reason = "" if level == "fresh" else f"obs_total_age_ms={age_ms if age_ms is not None else 'unknown'}"
        return {
            **timing,
            "stale_level": level,
            "stale_guard_active": level != "fresh",
            "stale_guard_reason": reason,
        }

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
        timing = self._stale_guard(obs, cmd.ts)
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
            **timing,
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
        summary = self._summary(mode, cmd, reason="stop")
        summary.update({
            "speed_profile": "stop",
            "speed_limit_reason": "stop",
            "forward_block_reason": "stop",
            "vx_norm": 0.0,
            "vy_norm": 0.0,
            "wz_norm": 0.0,
            "vx_mps": 0.0,
            "vy_mps": 0.0,
            "wz_radps": 0.0,
            "vx_mps_per_norm": self._axis_scale("vx"),
            "vy_mps_per_norm": self._axis_scale("vy"),
            "wz_radps_per_norm": self._axis_scale("wz"),
        })
        return MotionDecision(cmd=cmd, control_summary=summary)

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
        vx, vy, wz, speed_summary = self._apply_table_speed_profile(mode, mode, vx, 0.0, wz)
        cmd = self._cmd(mode, vx=vx, wz=wz)
        reason = "fallback_table"
        summary = self._summary(mode, cmd, obs, reason=reason)
        summary.update(speed_summary)
        summary.update({
            "vx_norm": float(vx),
            "vy_norm": float(vy),
            "wz_norm": float(wz),
            "forward_block_reason": "" if allow_forward and abs(vx) > 0.0 else "fallback_align_or_spin_only",
        })
        return MotionDecision(cmd=cmd, cx_norm_abs=x_abs, distance_ratio=distance_ratio, control_summary=summary)

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

    def _axis_scale(self, axis: str) -> float:
        attr = {
            "vx": "vx_mps_per_norm",
            "vy": "vy_mps_per_norm",
            "wz": "wz_radps_per_norm",
        }.get(axis, "vx_mps_per_norm")
        return max(1e-6, abs(float(getattr(self.car_cfg, attr, 1.0) or 1.0)))

    def _norm_to_physical(self, axis: str, value: float) -> float:
        return float(value) * self._axis_scale(axis)

    def _physical_to_norm(self, axis: str, value: float) -> float:
        return float(value) / self._axis_scale(axis)

    def _table_speed_profile_name(self, mode: str, phase: str = "") -> str:
        mode_text = str(mode or "").strip().upper()
        phase_text = str(phase or "").strip().upper()
        if mode_text == "COARSE_ALIGN":
            return "coarse_align"
        if mode_text == "FINAL_LOCK" or phase_text == "PLANE_STOP":
            return "final_lock"
        if mode_text == "CONTROLLED_APPROACH":
            return "controlled_approach"
        return "search"

    def _table_profile_limits(self, profile: str) -> Dict[str, float]:
        prefix = {
            "coarse_align": "table_coarse_align",
            "controlled_approach": "table_controlled",
            "final_lock": "table_final_lock",
        }.get(profile)
        if prefix is None:
            return {}
        return {
            "vx_min": abs(float(getattr(self.car_cfg, f"{prefix}_vx_min_mps", 0.0) or 0.0)),
            "vx_max": abs(float(getattr(self.car_cfg, f"{prefix}_vx_max_mps", 0.0) or 0.0)),
            "vy_min": abs(float(getattr(self.car_cfg, f"{prefix}_vy_min_mps", 0.0) or 0.0)),
            "vy_max": abs(float(getattr(self.car_cfg, f"{prefix}_vy_max_mps", 0.0) or 0.0)),
            "wz_min": abs(float(getattr(self.car_cfg, f"{prefix}_wz_min_radps", 0.0) or 0.0)),
            "wz_max": abs(float(getattr(self.car_cfg, f"{prefix}_wz_max_radps", 0.0) or 0.0)),
            "vx_deadband": abs(float(getattr(self.car_cfg, "table_vx_deadband_mps", 0.0) or 0.0)),
            "vy_deadband": abs(float(getattr(self.car_cfg, "table_vy_deadband_mps", 0.0) or 0.0)),
            "wz_deadband": abs(float(getattr(self.car_cfg, "table_wz_deadband_radps", 0.0) or 0.0)),
        }

    @staticmethod
    def _limit_physical_axis(value: float, min_abs: float, max_abs: float, deadband: float) -> Tuple[float, str]:
        magnitude = abs(float(value))
        if max_abs <= 0.0 or magnitude <= max(0.0, float(deadband)):
            return 0.0, "deadband" if magnitude > 0.0 else "zero"
        sign = 1.0 if value >= 0.0 else -1.0
        limited = min(magnitude, max_abs)
        reason = "max" if magnitude > max_abs else ""
        if min_abs > 0.0 and limited < min_abs:
            limited = min(min_abs, max_abs)
            reason = "min"
        return sign * limited, reason or "pass"

    def _apply_table_speed_profile(
        self,
        mode: str,
        phase: str,
        vx: float,
        vy: float,
        wz: float,
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        profile = self._table_speed_profile_name(mode, phase)
        limits = self._table_profile_limits(profile)
        if not limits:
            return vx, vy, wz, {
                "speed_profile": profile,
                "speed_limit_reason": "no_table_profile",
            }
        vx_mps = self._norm_to_physical("vx", vx)
        vy_mps = self._norm_to_physical("vy", vy)
        wz_radps = self._norm_to_physical("wz", wz)
        vx_limited, vx_reason = self._limit_physical_axis(vx_mps, limits["vx_min"], limits["vx_max"], limits["vx_deadband"])
        vy_limited, vy_reason = self._limit_physical_axis(vy_mps, limits["vy_min"], limits["vy_max"], limits["vy_deadband"])
        wz_limited, wz_reason = self._limit_physical_axis(wz_radps, limits["wz_min"], limits["wz_max"], limits["wz_deadband"])
        if profile == "final_lock":
            vx_limited = self._clamp(vx_limited, -limits["vx_max"], limits["vx_max"])
        reason_parts = [
            f"vx:{vx_reason}",
            f"vy:{vy_reason}",
            f"wz:{wz_reason}",
        ]
        return (
            self._physical_to_norm("vx", vx_limited),
            self._physical_to_norm("vy", vy_limited),
            self._physical_to_norm("wz", wz_limited),
            {
                "speed_profile": profile,
                "speed_limit_reason": ",".join(reason_parts),
                "vx_mps": float(vx_limited),
                "vy_mps": float(vy_limited),
                "wz_radps": float(wz_limited),
                "vx_mps_raw": float(vx_mps),
                "vy_mps_raw": float(vy_mps),
                "wz_radps_raw": float(wz_radps),
                "vx_mps_min": float(limits["vx_min"]),
                "vx_mps_max": float(limits["vx_max"]),
                "vy_mps_min": float(limits["vy_min"]),
                "vy_mps_max": float(limits["vy_max"]),
                "wz_radps_min": float(limits["wz_min"]),
                "wz_radps_max": float(limits["wz_max"]),
                "vx_deadband_mps": float(limits["vx_deadband"]),
                "vy_deadband_mps": float(limits["vy_deadband"]),
                "wz_deadband_radps": float(limits["wz_deadband"]),
                "vx_mps_per_norm": self._axis_scale("vx"),
                "vy_mps_per_norm": self._axis_scale("vy"),
                "wz_radps_per_norm": self._axis_scale("wz"),
            },
        )

    def _limit_table_cmd(self, vx: float, vy: float, wz: float) -> Tuple[float, float, float]:
        now_s = time.time()
        last_ts = getattr(self.docking, "_last_ts", 0.0) or now_s
        dt = max(0.05, min(0.25, now_s - float(last_ts)))
        vx_max = abs(float(getattr(self.car_cfg, "table_vx_norm_max", getattr(self.car_cfg, "table_stage_c_vx_max_norm", 0.05)) or 0.05))
        vx = self._clamp(vx, -vx_max, vx_max)
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
        guard = self._stale_guard(obs)
        stale_level = str(guard.get("stale_level") or "fresh")
        view_err, view_source, view_reliable = self._get_view_error(obs)
        obs_view_reliable = bool(getattr(obs, "view_reliable", False)) if obs is not None else False
        view_valid_for_forward = bool(view_reliable and obs_view_reliable)
        phase_name = self._table_approach_phase(obs, phase)
        soft_th = abs(float(getattr(self.car_cfg, "table_fov_soft_th", 0.25) or 0.25))
        hard_th = abs(float(getattr(self.car_cfg, "table_fov_hard_th", 0.40) or 0.40))
        control_level = str(getattr(obs, "control_level", "none") or "none").strip().lower() if obs is not None else "none"
        pose_found = bool(getattr(obs, "pose_found", False)) if obs is not None else False
        plane_touch_left = bool(getattr(obs, "plane_touch_left", False)) if obs is not None else False
        plane_touch_right = bool(getattr(obs, "plane_touch_right", False)) if obs is not None else False
        obs_stale = stale_level != "fresh"
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
        min_forward_dist = max(0.0, float(getattr(self.car_cfg, "table_min_forward_dist_err_m", 0.07) or 0.07))
        vx_min = abs(float(getattr(self.car_cfg, "table_vx_norm_min", 0.018) or 0.018))
        vx_max = abs(float(getattr(self.car_cfg, "table_vx_norm_max", 0.045) or 0.045))
        vx_min = min(vx_min, vx_max)
        vx_kp = max(0.0, float(getattr(self.car_cfg, "table_vx_kp_norm_per_m", 0.10) or 0.10))
        near_dist_err_th = max(min_forward_dist, float(getattr(self.car_cfg, "table_near_dist_err_th_m", 0.10) or 0.10))
        forward_allowed = bool(
            pose_found
            and view_valid_for_forward
            and not hard_guard
            and not obs_stale
            and dist_err > min_forward_dist
            and control_level in {"approach", "alignment"}
            and phase_name not in {"PLANE_ACQUIRE", "PLANE_STOP"}
        )
        forward_block_reason = ""
        if not pose_found:
            forward_block_reason = "pose_missing"
        elif not view_valid_for_forward:
            forward_block_reason = "view_unreliable"
        elif hard_guard:
            forward_block_reason = fov_guard_reason or "fov_guard"
        elif obs_stale:
            forward_block_reason = stale_level
        elif dist_err <= min_forward_dist:
            forward_block_reason = "dist_below_min_forward"
        elif control_level not in {"approach", "alignment"}:
            forward_block_reason = f"control_level_{control_level or 'none'}"
        elif phase_name in {"PLANE_ACQUIRE", "PLANE_STOP"}:
            forward_block_reason = f"phase_{phase_name.lower()}"
        if forward_allowed:
            vx_from_dist = self._clamp(dist_err * vx_kp, vx_min, vx_max)

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

        yaw_abs = abs(yaw_err)
        yaw_slow_th = abs(float(getattr(self.car_cfg, "table_yaw_slow_th_rad", 0.12) or 0.12))
        yaw_stop_th = max(yaw_slow_th + 1e-6, abs(float(getattr(self.car_cfg, "table_yaw_stop_th_rad", 0.45) or 0.45)))
        yaw_gate = 1.0
        if yaw_abs > yaw_slow_th:
            yaw_gate = self._clamp(1.0 - ((yaw_abs - yaw_slow_th) / max(1e-6, yaw_stop_th - yaw_slow_th)), 0.0, 1.0)
        fov_gate = 0.0 if hard_guard else 1.0
        if view_valid_for_forward and abs(view_err) > soft_th and hard_th > soft_th:
            fov_gate = min(fov_gate, max(0.0, 1.0 - ((abs(view_err) - soft_th) / (hard_th - soft_th))))
        if not view_valid_for_forward:
            fov_gate = 0.0
        near_gate = 1.0
        final_lock_dist_tol = max(0.0, float(self.cfg.final_lock_dist_tol_m))
        if dist_err <= final_lock_dist_tol:
            near_gate = 0.0
        elif dist_err < near_dist_err_th:
            near_gate = self._clamp((dist_err - final_lock_dist_tol) / max(1e-6, near_dist_err_th - final_lock_dist_tol), 0.0, 1.0)

        vx = vx_from_dist * yaw_gate * fov_gate * near_gate if forward_allowed else 0.0
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
        if obs_stale:
            vx = 0.0
            vx_from_dist = 0.0
            wz_from_plane = 0.0
            if stale_level == "soft_stale":
                recover_vy = abs(float(getattr(self.car_cfg, "table_view_recover_vy_norm", 0.008) or 0.008))
                recover_wz = abs(float(getattr(self.car_cfg, "table_view_recover_wz_norm", 0.04) or 0.04))
                vy = self._clamp(vy_from_view, -recover_vy, recover_vy)
                wz = self._clamp(wz_from_view, -recover_wz, recover_wz)
            else:
                vy = 0.0
                wz = 0.0
        if (not view_valid_for_forward and phase_name not in {"PLANE_ACQUIRE"}) or obs_stale:
            vx = 0.0
            if obs_stale and stale_level != "soft_stale":
                vy = 0.0
                wz = 0.0
        vx, vy, wz = self._limit_table_cmd(vx, vy, wz)
        if hard_guard or (not view_valid_for_forward and phase_name not in {"PLANE_ACQUIRE"}) or obs_stale:
            vx = 0.0
            self._last_table_vx = 0.0
            if obs_stale and stale_level != "soft_stale":
                vy = 0.0
                wz = 0.0
                self._last_table_vy = 0.0
                self._last_table_wz = 0.0
            elif obs_stale:
                recover_vy = abs(float(getattr(self.car_cfg, "table_view_recover_vy_norm", 0.008) or 0.008))
                recover_wz = abs(float(getattr(self.car_cfg, "table_view_recover_wz_norm", 0.04) or 0.04))
                vy = self._clamp(vy, -recover_vy, recover_vy)
                wz = self._clamp(wz, -recover_wz, recover_wz)
            elif hard_guard:
                vy = direction * recover_vy * float(getattr(self.car_cfg, "table_view_vy_sign", -1.0))
                wz = direction * recover_wz * float(getattr(self.car_cfg, "table_view_wz_sign", -1.0))
        if obs_stale:
            vx = 0.0
            vy = 0.0
            wz = 0.0
            self._last_table_vx = 0.0
            self._last_table_vy = 0.0
            self._last_table_wz = 0.0

        now_mono = time.monotonic()
        pose_missing_duration_s = 0.0
        pose_missing_safe_vx_active = False
        if pose_found:
            self._pose_missing_since_mono = 0.0
        elif obs is not None:
            if self._pose_missing_since_mono <= 0.0:
                self._pose_missing_since_mono = now_mono
            pose_missing_duration_s = max(0.0, now_mono - self._pose_missing_since_mono)
        else:
            self._pose_missing_since_mono = 0.0

        final_lock_dist_tol = max(0.0, float(self.cfg.final_lock_dist_tol_m))
        pose_missing_hold_s = max(0.0, float(getattr(self.car_cfg, "table_pose_missing_max_hold_s", 3.0) or 3.0))
        pose_missing_safe_vx_mps = abs(float(getattr(self.car_cfg, "table_pose_missing_safe_vx_mps", 0.010) or 0.010))
        pose_missing_safe_vx_norm = self._physical_to_norm("vx", pose_missing_safe_vx_mps)
        valid_for_safe_forward = bool(
            obs is not None
            and not pose_found
            and bool(getattr(obs, "edge_found", False))
            and bool(getattr(obs, "valid_for_control", False) or getattr(obs, "usable_for_approach", False))
            and stale_level not in {"hard_stale", "dead"}
            and not hard_guard
            and yaw_abs <= yaw_stop_th
            and dist_err > final_lock_dist_tol
            and str(getattr(obs, "reject_reason", "") or "").strip().lower() in {"", "none"}
            and getattr(obs, "depth_valid", True) is not False
            and phase_name not in {"PLANE_ACQUIRE", "PLANE_STOP"}
        )
        if valid_for_safe_forward:
            if pose_missing_duration_s <= pose_missing_hold_s:
                vx = max(float(vx), pose_missing_safe_vx_norm)
                forward_block_reason = "pose_missing_safe_vx"
                pose_missing_safe_vx_active = True
            else:
                vx = 0.0
                forward_block_reason = "pose_missing_timeout"
        elif not pose_found and obs is not None:
            if stale_level in {"hard_stale", "dead"}:
                forward_block_reason = "vision_stale"
            elif hard_guard:
                forward_block_reason = fov_guard_reason or "edge_invalid"
            elif yaw_abs > yaw_stop_th:
                forward_block_reason = "yaw_out_of_range"
            elif not bool(getattr(obs, "edge_found", False) and (getattr(obs, "valid_for_control", False) or getattr(obs, "usable_for_approach", False))):
                forward_block_reason = "edge_invalid"
        vx, vy, wz, speed_summary = self._apply_table_speed_profile(mode, phase_name, vx, vy, wz)
        if abs(vx) < self._physical_to_norm("vx", abs(float(getattr(self.car_cfg, "table_vx_deadband_mps", 0.004) or 0.004))):
            vx = 0.0
        if abs(vy) < self._physical_to_norm("vy", abs(float(getattr(self.car_cfg, "table_vy_deadband_mps", 0.003) or 0.003))):
            vy = 0.0
        if abs(wz) < self._physical_to_norm("wz", abs(float(getattr(self.car_cfg, "table_wz_deadband_radps", 0.006) or 0.006))):
            wz = 0.0

        cmd = self._cmd(mode, vx=vx, vy=vy, wz=wz)
        summary = self._summary(mode, cmd, obs, reason=reason)
        if abs(float(vx)) <= 1e-9 and forward_allowed and not forward_block_reason:
            forward_block_reason = "speed_profile_deadband_or_limit"
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
                "stale_level": stale_level,
                "stale_guard_active": bool(obs_stale),
                "stale_guard_reason": str(guard.get("stale_guard_reason") or ""),
                "pose_found": bool(pose_found),
                "control_level": control_level,
                "normalized_control_level": control_level,
                "yaw_err": float(yaw_err),
                "dist_err_m": float(dist_err),
                "forward_allowed": bool(forward_allowed),
                "forward_block_reason": forward_block_reason or "none",
                "final_lock_enabled": bool(getattr(self.cfg, "enable_final_lock", False)),
                "micro_adjust_enabled": bool(getattr(self.cfg, "enable_micro_adjust", False)),
                "stop_ready_ignored_for_stage_transition": False,
                "micro_adjust_skipped": False,
                "pose_missing_duration_s": float(pose_missing_duration_s),
                "pose_missing_safe_vx_active": bool(pose_missing_safe_vx_active),
                "pose_missing_safe_vx_mps": float(pose_missing_safe_vx_mps),
                "pose_missing_max_hold_s": float(pose_missing_hold_s),
                "min_forward_dist_err_m": float(min_forward_dist),
                "vx_norm_min": float(vx_min),
                "vx_norm_max": float(vx_max),
                "vx_kp_norm_per_m": float(vx_kp),
                "near_dist_err_th_m": float(near_dist_err_th),
                "yaw_slow_th_rad": float(yaw_slow_th),
                "yaw_stop_th_rad": float(yaw_stop_th),
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
                "vx_norm": float(vx),
                "vy_norm": float(vy),
                "wz_norm": float(wz),
                "table_confirmed_by_yolo": bool(getattr(obs, "table_confirmed_by_yolo", False)) if obs is not None else False,
                "yolo_reliable": bool(getattr(obs, "yolo_reliable", False)) if obs is not None else False,
                "yolo_gate_open": bool(getattr(obs, "yolo_gate_open", False)) if obs is not None else False,
                "plane_cx_norm": getattr(obs, "plane_cx_norm", None) if obs is not None else None,
                "plane_width_norm": getattr(obs, "plane_width_norm", None) if obs is not None else None,
                "plane_touch_left": bool(getattr(obs, "plane_touch_left", False)) if obs is not None else False,
                "plane_touch_right": bool(getattr(obs, "plane_touch_right", False)) if obs is not None else False,
            }
        )
        summary.update(speed_summary)
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
        vx, vy, wz, speed_summary = self._apply_table_speed_profile(mode, mode, out.vx, out.vy, out.wz)
        cmd = self._cmd(mode, vx=vx, vy=vy, wz=wz)
        summary = self._summary(mode, cmd, obs, reason="docking_control")
        summary.update(speed_summary)
        summary.update({
            "vx_norm": float(vx),
            "vy_norm": float(vy),
            "wz_norm": float(wz),
            "forward_block_reason": "coarse_align_no_forward" if mode == "COARSE_ALIGN" else "",
        })
        return MotionDecision(
            cmd=cmd,
            cx_norm_abs=abs(float(edge_obs.yaw_err_rad or 0.0)),
            distance_ratio=abs(float(edge_obs.dist_err_m or 0.0)),
            control_summary=summary,
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
