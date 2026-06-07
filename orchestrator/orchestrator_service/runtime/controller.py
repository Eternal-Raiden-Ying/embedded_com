#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..config.schema import CarMotionConfig, ControlThresholds
from ..control.docking_controller import DockingController
from ..control.types import DockingControlConfig, EdgeControlObservation
from ..ipc.protocol import ArmCommand, CmdVel, HomeTagObs, TableEdgeObs, TargetObs, now_ts
from .perception_semantics import build_table_perception_semantics, table_bbox_found as semantic_table_bbox_found


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
        stale_level = str(timing.get("stale_level") or "")
        yolo_area_fields = self._yolo_area_gate_fields(obs) if obs is not None else {
            "yolo_table_bbox_area_ratio": 0.0,
            "docking_allowed_by_yolo_area": False,
            "docking_blocked_by_yolo_area": False,
            "docking_enabled_by_yolo": False,
            "edge_control_allowed": False,
            "edge_control_block_reason": "table_bbox_unavailable",
        }
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
            "stale_source": (
                "edge"
                if obs is not None and stale_level and stale_level != "fresh" and self._yolo_reliable(obs)
                else ("table" if obs is not None and stale_level and stale_level != "fresh" else "")
            ),
            **timing,
            "table_bbox_found": bool(getattr(obs, "table_bbox_found", False)) if obs is not None else False,
            "table_bbox_xyxy": getattr(obs, "table_bbox_xyxy", None) if obs is not None else None,
            "table_bbox_area_ratio": getattr(obs, "table_bbox_area_ratio", getattr(obs, "yolo_bbox_area_norm", None)) if obs is not None else None,
            "table_bbox_conf_raw": getattr(obs, "table_bbox_conf_raw", getattr(obs, "yolo_table_conf", None)) if obs is not None else None,
            "table_bbox_conf_used_for_gate": bool(getattr(obs, "table_bbox_conf_used_for_gate", False)) if obs is not None else False,
            "table_confirmed_by_yolo": bool(getattr(obs, "table_confirmed_by_yolo", False)) if obs is not None else False,
            "yolo_reliable": bool(getattr(obs, "yolo_reliable", False)) if obs is not None else False,
            "yolo_table_control_valid": bool(getattr(obs, "yolo_table_control_valid", getattr(obs, "yolo_reliable", False))) if obs is not None else False,
            "yolo_valid_reason": str(getattr(obs, "yolo_valid_reason", "") or "") if obs is not None else "",
            "yolo_invalid_reason": str(getattr(obs, "yolo_invalid_reason", "") or "") if obs is not None else "",
            **yolo_area_fields,
            "yolo_table_roi_valid": bool(getattr(obs, "yolo_table_roi_valid", False)) if obs is not None else False,
            "yolo_table_conf": (float(getattr(obs, "yolo_table_conf")) if obs is not None and getattr(obs, "yolo_table_conf", None) is not None else None),
            "table_bbox_touch_left": bool(getattr(obs, "table_bbox_touch_left", getattr(obs, "yolo_bbox_touch_left", False))) if obs is not None else False,
            "table_bbox_touch_right": bool(getattr(obs, "table_bbox_touch_right", getattr(obs, "yolo_bbox_touch_right", False))) if obs is not None else False,
            "table_bbox_touch_bottom": bool(getattr(obs, "table_bbox_touch_bottom", getattr(obs, "yolo_bbox_touch_bottom", False))) if obs is not None else False,
            "table_bbox_boundary_allowed": bool(getattr(obs, "table_bbox_boundary_allowed", False)) if obs is not None else False,
            "roi_source": str(getattr(obs, "roi_source", "") or "") if obs is not None else "",
            "roi_reason": str(getattr(obs, "roi_reason", "") or "") if obs is not None else "",
            "roi_phase": str(getattr(obs, "roi_phase", "") or "") if obs is not None else "",
            "yolo_table_edge_stable_count": getattr(obs, "yolo_table_edge_stable_count", None) if obs is not None else None,
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
        summary = self._summary("SEARCH_TABLE", cmd, reason="search_table")
        summary.update(
            {
                "control_source": "local_rotate_search",
                "allow_rotate": True,
                "rotate_block_reason": "",
                "edge_yaw_err_rad": 0.0,
                "yolo_view_err_norm": 0.0,
                "yolo_edge_yaw_conflict": False,
            }
        )
        return MotionDecision(cmd=cmd, cx_norm_abs=abs(wz), distance_ratio=1.0, control_summary=summary)

    def yolo_table_search_cmd(
        self,
        obs: Optional[TableEdgeObs],
        turn_sign: int = 1,
        *,
        mode: str = "SEARCH_TABLE",
        reason: str = "yolo_table_far_guide",
        control_source: str = "yolo_assist",
    ) -> MotionDecision:
        if not bool(getattr(self.cfg, "yolo_table_control_enable", True)):
            return self.search_table_cmd(turn_sign=turn_sign)
        cx = self._clamp(float(getattr(obs, "table_cx_norm", 0.0) or 0.0), -1.0, 1.0)
        gain = float(getattr(self.car_cfg, "yolo_table_yaw_gain", 0.20) or 0.20)
        max_wz = abs(float(getattr(self.car_cfg, "yolo_table_max_wz", 0.06) or 0.06))
        sign = float(getattr(self.car_cfg, "table_view_wz_sign", -1.0))
        wz_raw = cx * gain * sign
        wz = self._clamp(wz_raw, -max_wz, max_wz)
        if abs(wz) < 0.02:
            wz = 0.0
        mode_name = str(mode or "SEARCH_TABLE").upper().strip() or "SEARCH_TABLE"
        source_name = str(control_source or "yolo_assist")
        assist_vx = 0.0
        if source_name == "yolo_forward":
            assist_vx = max(0.0, float(getattr(self.car_cfg, "yolo_table_forward_vx", 0.015) or 0.015))
            wz = 0.0
        elif source_name == "yolo_assist":
            assist_vx = max(0.0, float(getattr(self.car_cfg, "yolo_table_assist_vx", 0.010) or 0.010))
            # YOLO assist is still forward-safe in this first control refactor.
            # Posture correction is reserved for edge_adjust / trusted edge.
            wz = 0.0
        cmd = self._cmd(mode_name, vx=assist_vx, wz=wz)
        summary = self._summary(mode_name, cmd, obs, reason=reason)
        summary.update(
            {
                "control_source": source_name,
                "approach_source": "yolo_table_bbox",
                "bbox_cx_norm": float((cx + 1.0) * 0.5),
                "target_offset": float(cx * 0.5),
                "table_cx_norm_signed": float(cx),
                "yolo_yaw_gain": float(gain),
                "yolo_max_wz": float(max_wz),
                "yolo_wz_raw": float(wz_raw),
                "yolo_yaw_cmd": float(wz),
                "edge_yaw_cmd": 0.0,
                "final_yaw_cmd": float(wz),
                "wz_sign_basis": "wz = table_cx_norm_signed * yolo_table_yaw_gain * table_view_wz_sign",
                "table_view_wz_sign": float(sign),
                "final_wz": float(wz),
                "vx_norm": float(assist_vx),
                "vy_norm": 0.0,
                "wz_norm": float(wz),
                "control_intent": "forward",
                "allow_forward": bool(assist_vx > 0.0),
                "allow_rotate": False,
                "forward_block_reason": "" if assist_vx > 0.0 else "yolo_vx_zero",
                "rotate_block_reason": "yolo_forward_owns_control",
                "yolo_view_err_norm": float(cx),
                "edge_yaw_err_rad": float(getattr(obs, "yaw_err_rad", 0.0) or 0.0) if obs is not None else 0.0,
                "yolo_edge_yaw_conflict": False,
            }
        )
        return MotionDecision(cmd=cmd, cx_norm_abs=abs(cx), distance_ratio=1.0, control_summary=summary)

    def yolo_table_assist_cmd(self, obs: Optional[TableEdgeObs], mode: str, reason: str = "edge_lost_yolo_assist") -> MotionDecision:
        decision = self.yolo_table_search_cmd(obs, mode=mode, reason=reason, control_source="yolo_assist")
        decision.control_summary.update(
            {
                "edge_lost_but_yolo_valid": True,
                "search_blocked_by_yolo_valid": True,
                "vx_norm": float(decision.cmd.vx_norm),
                "vy_norm": 0.0,
            }
        )
        return decision

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
        if not self._table_bbox_found(obs):
            return False
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

    def _table_semantics(self, obs: Optional[TableEdgeObs], *, stale_level: str = "", stale_source: str = ""):
        return build_table_perception_semantics(obs, self.cfg, stale_level=stale_level, stale_source=stale_source)

    def _table_bbox_found(self, obs: Optional[TableEdgeObs]) -> bool:
        return semantic_table_bbox_found(obs)

    def _edge_trusted(self, obs: Optional[TableEdgeObs]) -> bool:
        return bool(self._table_semantics(obs).edge_trusted)

    def _yolo_reliable(self, obs: Optional[TableEdgeObs]) -> bool:
        # Current strategy deliberately treats table bbox existence as the only
        # table control gate. Confidence is logged but not used as a control gate.
        return self._table_bbox_found(obs)

    def _yolo_view_err_norm(self, obs: Optional[TableEdgeObs]) -> float:
        if obs is None:
            return 0.0
        for name in ("view_err_norm", "table_cx_norm"):
            value = getattr(obs, name, None)
            if value is None:
                continue
            try:
                return self._clamp(float(value), -1.0, 1.0)
            except Exception:
                continue
        return 0.0

    def _yolo_table_bbox_area_ratio(self, obs: Optional[TableEdgeObs]) -> float:
        # Diagnostic only.  It must not participate in control authority.
        if obs is None:
            return 0.0
        for name in ("table_bbox_area_ratio", "yolo_bbox_area_norm", "table_size_norm"):
            value = getattr(obs, name, None)
            if value is None:
                continue
            try:
                return self._clamp(float(value), 0.0, 1.0)
            except Exception:
                continue
        return 0.0

    def _yolo_area_gate_fields(self, obs: Optional[TableEdgeObs]) -> Dict[str, Any]:
        # Deprecated name retained for log/backward compatibility only.
        # The old 0.40 bbox area gate is no longer used for control.
        sem = self._table_semantics(obs)
        return {
            "yolo_table_bbox_area_ratio": float(self._yolo_table_bbox_area_ratio(obs)),
            "docking_allowed_by_yolo_area": False,
            "docking_blocked_by_yolo_area": False,
            "docking_enabled_by_yolo": bool(sem.table_bbox_found),
            "edge_control_allowed": bool(sem.edge_trusted),
            "edge_control_block_reason": "" if sem.edge_trusted else sem.edge_reject_for_control_reason,
            "table_bbox_found": bool(sem.table_bbox_found),
            "table_bbox_xyxy": sem.table_bbox_xyxy,
            "table_bbox_conf_raw": sem.table_bbox_conf_raw,
            "table_bbox_conf_used_for_gate": False,
            "yolo_table_control_valid": bool(sem.table_bbox_found),
            "edge_detected": bool(sem.edge_detected),
            "edge_valid": bool(sem.edge_valid),
            "edge_stable": bool(sem.edge_stable),
            "edge_trusted": bool(sem.edge_trusted),
            "edge_trust_reason": sem.edge_trust_reason,
            "edge_reject_for_control_reason": sem.edge_reject_for_control_reason,
        }

    def _rotate_gate(self, obs: Optional[TableEdgeObs], edge_yaw_cmd: float = 0.0) -> Dict[str, Any]:
        sem = self._table_semantics(obs)
        yaw_err = float(getattr(obs, "yaw_err_rad", 0.0) or 0.0) if obs is not None else 0.0
        yaw_th = abs(float(getattr(self.cfg, "rotate_yaw_threshold_rad", 0.20) or 0.20))
        yaw_over = bool(abs(yaw_err) >= yaw_th)
        allow = bool(sem.edge_trusted and yaw_over)
        if not sem.table_bbox_found:
            reason = "table_bbox_unavailable"
        elif not sem.edge_trusted:
            reason = sem.edge_reject_for_control_reason or "edge_not_trusted"
        elif not yaw_over:
            reason = "yaw_below_threshold"
        else:
            reason = ""
        yolo_view_err = self._yolo_view_err_norm(obs)
        return {
            "docking_enabled_by_yolo": bool(sem.table_bbox_found),
            "edge_control_allowed": bool(sem.edge_trusted),
            "edge_control_block_reason": "" if sem.edge_trusted else (sem.edge_reject_for_control_reason or "edge_not_trusted"),
            "allow_rotate": bool(allow),
            "rotate_block_reason": reason,
            "rotate_require_edge_stable_frames": int(max(1, int(getattr(self.cfg, "rotate_require_edge_stable_frames", 5) or 5))),
            "rotate_yaw_threshold_rad": float(yaw_th),
            "edge_stable_for_rotate": bool(sem.edge_stable),
            "roi_trusted_for_rotate": bool(sem.edge_trusted),
            "yaw_over_rotate_threshold": bool(yaw_over),
            "yolo_view_err_norm": float(yolo_view_err),
            "edge_yaw_err_rad": float(yaw_err),
            "yolo_edge_yaw_conflict": False,
            "yolo_edge_conflict_block_rotate": bool(getattr(self.cfg, "yolo_edge_conflict_block_rotate", True)),
            "edge_detected": bool(sem.edge_detected),
            "edge_valid": bool(sem.edge_valid),
            "edge_stable": bool(sem.edge_stable),
            "edge_trusted": bool(sem.edge_trusted),
            "edge_trust_reason": sem.edge_trust_reason,
            "edge_reject_for_control_reason": sem.edge_reject_for_control_reason,
        }

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

    def _compose_coarse_align_cmd(
        self,
        obs: Optional[TableEdgeObs],
        *,
        phase: str = "",
        reason: str = "coarse_align_yaw_only",
    ) -> MotionDecision:
        if self._table_bbox_found(obs) and not self._edge_trusted(obs):
            return self.yolo_table_search_cmd(
                obs,
                mode="CONTROLLED_APPROACH",
                reason="coarse_align_blocked_edge_not_trusted",
                control_source="yolo_forward",
            )
        guard = self._stale_guard(obs)
        stale_level = str(guard.get("stale_level") or "fresh")
        yaw_err = float(obs.yaw_err_rad) if obs is not None and obs.yaw_err_rad is not None else 0.0
        yaw_abs = abs(yaw_err)
        deadband = abs(float(getattr(self.car_cfg, "table_approach_yaw_deadband_rad", 0.08) or 0.08))
        edge_valid = bool(
            obs is not None
            and self._table_bbox_found(obs)
            and bool(getattr(obs, "edge_found", False))
            and getattr(obs, "yaw_err_rad", None) is not None
            and bool(
                getattr(obs, "edge_valid", False)
                or getattr(obs, "valid_for_control", False)
                or getattr(obs, "usable_for_alignment", False)
                or getattr(obs, "usable_for_approach", False)
            )
            and str(getattr(obs, "reject_reason", "") or "").strip().lower() in {"", "none"}
            and getattr(obs, "depth_valid", True) is not False
        )
        forward_block_reason = "coarse_align_no_forward"
        coarse_align_reason = "turn_yaw"
        wz = 0.0
        control_source = "edge_only"
        yolo_blend_weight = 0.0
        if stale_level != "fresh":
            coarse_align_reason = stale_level
        elif not edge_valid:
            coarse_align_reason = "edge_invalid"
        elif yaw_abs <= deadband:
            coarse_align_reason = "yaw_deadband"
        else:
            sign = 1.0 if yaw_err * float(getattr(self.car_cfg, "table_plane_yaw_sign", 1.0)) >= 0.0 else -1.0
            wz_min = abs(float(getattr(self.car_cfg, "table_coarse_align_wz_min_radps", 0.08) or 0.08))
            wz_max = max(wz_min, abs(float(getattr(self.car_cfg, "table_coarse_align_wz_max_radps", 0.15) or 0.15)))
            wz_radps = self._clamp(max(yaw_abs, wz_min), wz_min, wz_max)
            wz = self._physical_to_norm("wz", sign * wz_radps)

        yolo_control_enabled = bool(getattr(self.cfg, "yolo_table_control_enable", True))
        stable_count = int(getattr(obs, "yolo_table_edge_stable_count", 0) or 0) if obs is not None else 0
        blend_start = max(1, int(getattr(self.cfg, "yolo_table_blend_start_stable_frames", 5) or 5))
        edge_wz_cmd = float(wz)
        yolo_wz = 0.0
        if yolo_control_enabled and self._yolo_reliable(obs) and getattr(obs, "table_cx_norm", None) is not None:
            yolo_wz = self._clamp(
                float(obs.table_cx_norm)
                * float(getattr(self.car_cfg, "yolo_table_yaw_gain", 0.20) or 0.20)
                * float(getattr(self.car_cfg, "table_view_wz_sign", -1.0)),
                -abs(float(getattr(self.car_cfg, "yolo_table_max_wz", 0.06) or 0.06)),
                abs(float(getattr(self.car_cfg, "yolo_table_max_wz", 0.06) or 0.06)),
            )
            if edge_valid and stable_count >= blend_start:
                yolo_blend_weight = self._clamp(float(getattr(self.cfg, "yolo_table_blend_yolo_weight", 0.5) or 0.5), 0.0, 1.0)
                wz = yolo_blend_weight * yolo_wz + (1.0 - yolo_blend_weight) * wz
                control_source = "yolo_edge_blend"
            elif not edge_valid:
                wz = yolo_wz
                control_source = "yolo_assist"
                coarse_align_reason = "yolo_table_edge_invalid"
        elif not edge_valid:
            control_source = "search_fallback"

        rotate_gate = self._rotate_gate(obs, edge_yaw_cmd=edge_wz_cmd)
        if self._yolo_reliable(obs) and abs(float(wz)) > 1e-9 and not bool(rotate_gate.get("allow_rotate", False)):
            wz = 0.0
            control_source = "yolo_assist"
            coarse_align_reason = str(rotate_gate.get("rotate_block_reason") or "rotate_blocked")

        vx, vy, wz, speed_summary = self._apply_table_speed_profile("COARSE_ALIGN", phase or "COARSE_ALIGN", 0.0, 0.0, wz)
        vx = 0.0
        vy = 0.0
        speed_summary["vx_mps"] = 0.0
        speed_summary["vy_mps"] = 0.0
        cmd = self._cmd("COARSE_ALIGN", vx=vx, vy=vy, wz=wz)
        summary = self._summary("COARSE_ALIGN", cmd, obs, reason=reason)
        summary.update(
            {
                "table_approach_phase": phase or "PLANE_FINAL_LOCK",
                "phase": phase or "PLANE_FINAL_LOCK",
                "approach_source": "plane_only" if control_source == "edge_only" else control_source,
                "control_source": control_source,
                "yolo_edge_blend_weight": float(yolo_blend_weight),
                "yolo_table_control_enable": bool(yolo_control_enabled),
                "yolo_table_edge_stable_count": int(stable_count),
                "yolo_table_blend_start_stable_frames": int(blend_start),
                "yaw_err": float(yaw_err),
                "yaw_err_rad": float(yaw_err),
                "yaw_abs": float(yaw_abs),
                "coarse_align_yaw_deadband_rad": float(deadband),
                "coarse_align_reason": coarse_align_reason,
                "yolo_yaw_cmd": float(yolo_wz),
                "edge_yaw_cmd": float(edge_wz_cmd),
                "final_yaw_cmd": float(wz),
                "coarse_align_only_yaw": True,
                "edge_found": bool(getattr(obs, "edge_found", False)) if obs is not None else False,
                "usable_for_approach": bool(getattr(obs, "usable_for_approach", False)) if obs is not None else False,
                "valid_for_control": bool(getattr(obs, "valid_for_control", False)) if obs is not None else False,
                "pose_found": bool(getattr(obs, "pose_found", False)) if obs is not None else False,
                "obs_stale": bool(stale_level != "fresh"),
                "stale_level": stale_level,
                "forward_allowed": False,
                "forward_block_reason": forward_block_reason,
                "approach_allow_wz": True,
                "approach_allow_vy": False,
                "approach_speed_mode": "coarse_align_yaw_only",
                "final_vx": 0.0,
                "final_vy": 0.0,
                "final_wz": float(wz),
                "vx_norm": 0.0,
                "vy_norm": 0.0,
                "wz_norm": float(wz),
            }
        )
        summary.update(rotate_gate)
        summary.update(speed_summary)
        return MotionDecision(
            cmd=cmd,
            cx_norm_abs=yaw_abs,
            distance_ratio=max(0.0, min(1.0, abs(float(getattr(obs, "dist_err_m", 0.0) or 0.0)))) if obs is not None else 0.0,
            control_summary=summary,
        )

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
        if str(mode or "").strip().upper() == "COARSE_ALIGN":
            return self._compose_coarse_align_cmd(obs, phase=phase, reason=reason)
        guard = self._stale_guard(obs)
        stale_level = str(guard.get("stale_level") or "fresh")
        view_err, view_source, view_reliable = self._get_view_error(obs)
        obs_view_reliable = bool(getattr(obs, "view_reliable", False)) if obs is not None else False
        view_valid_for_forward = bool(view_reliable and obs_view_reliable)
        phase_name = self._table_approach_phase(obs, phase)
        soft_th = abs(float(getattr(self.car_cfg, "table_fov_soft_th", 0.25) or 0.25))
        hard_th = abs(float(getattr(self.car_cfg, "table_fov_hard_th", 0.40) or 0.40))
        control_level = str(getattr(obs, "control_level", "none") or "none").strip().lower() if obs is not None else "none"
        stable_count = int(getattr(obs, "yolo_table_edge_stable_count", 0) or 0) if obs is not None else 0
        blend_start = max(1, int(getattr(self.cfg, "yolo_table_blend_start_stable_frames", 5) or 5))
        yolo_control_enabled = bool(getattr(self.cfg, "yolo_table_control_enable", True))
        near_measured = False
        if obs is not None and obs.dist_err_m is not None:
            try:
                near_measured = (float(obs.target_dist_m or self.cfg.table_target_dist_m) + float(obs.dist_err_m)) <= float(getattr(self.cfg, "yolo_table_near_dist_m", 0.45) or 0.45)
            except Exception:
                near_measured = False
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
            self._table_bbox_found(obs)
            and pose_found
            and view_valid_for_forward
            and not hard_guard
            and not obs_stale
            and dist_err > min_forward_dist
            and control_level in {"approach", "alignment"}
            and phase_name not in {"PLANE_ACQUIRE", "PLANE_STOP"}
        )
        forward_block_reason = ""
        if not self._table_bbox_found(obs):
            forward_block_reason = "table_bbox_unavailable"
        elif not pose_found:
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

        control_source = "edge_only"
        yolo_blend_weight = 0.0
        edge_yaw_cmd = float(wz_from_plane)
        yolo_yaw_cmd = 0.0
        if yolo_control_enabled and view_source == "yolo" and not self._plane_stable(obs):
            control_source = "yolo_assist"
            yolo_yaw_cmd = float(wz_from_view)
        elif yolo_control_enabled and view_source in {"yolo", "plane"} and stable_count >= blend_start and not near_measured:
            control_source = "yolo_edge_blend"
            yolo_blend_weight = self._clamp(float(getattr(self.cfg, "yolo_table_blend_yolo_weight", 0.5) or 0.5), 0.0, 1.0)
            if getattr(obs, "table_cx_norm", None) is not None:
                yolo_wz = self._clamp(
                    float(obs.table_cx_norm)
                    * float(getattr(self.car_cfg, "yolo_table_yaw_gain", 0.20) or 0.20)
                    * float(getattr(self.car_cfg, "table_view_wz_sign", -1.0)),
                    -abs(float(getattr(self.car_cfg, "yolo_table_max_wz", 0.06) or 0.06)),
                    abs(float(getattr(self.car_cfg, "yolo_table_max_wz", 0.06) or 0.06)),
                )
                yolo_yaw_cmd = float(yolo_wz)
                wz_from_view = yolo_blend_weight * yolo_wz + (1.0 - yolo_blend_weight) * wz_from_view
        elif not view_reliable and not self._plane_stable(obs):
            control_source = "search_fallback"
        if near_measured and self._plane_stable(obs):
            control_source = "edge_only"

        yaw_abs = abs(yaw_err)
        yaw_slow_th = abs(float(getattr(self.car_cfg, "table_yaw_slow_th_rad", 0.12) or 0.12))
        yaw_stop_th = max(yaw_slow_th + 1e-6, abs(float(getattr(self.car_cfg, "table_yaw_stop_th_rad", 0.45) or 0.45)))
        approach_yaw_deadband = abs(float(getattr(self.car_cfg, "table_approach_yaw_deadband_rad", 0.08) or 0.08))
        approach_yaw_realign = max(
            approach_yaw_deadband + 1e-6,
            abs(float(getattr(self.car_cfg, "table_approach_yaw_realign_rad", 0.12) or 0.12)),
        )
        approach_allow_wz = bool(getattr(self.car_cfg, "table_approach_allow_wz", False))
        approach_allow_vy = bool(getattr(self.car_cfg, "table_approach_allow_vy", False))
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
        approach_speed_mode = "profile"
        approach_safe_vx_mps = abs(float(getattr(self.car_cfg, "table_approach_safe_vx_mps", pose_missing_safe_vx_mps) or pose_missing_safe_vx_mps))
        approach_max_vx_mps = abs(float(getattr(self.car_cfg, "table_approach_max_vx_mps", 0.030) or 0.030))
        if approach_max_vx_mps > 0.0:
            approach_safe_vx_mps = min(approach_safe_vx_mps, approach_max_vx_mps)
        approach_vx_norm = self._physical_to_norm("vx", approach_safe_vx_mps)
        approach_edge_valid = bool(
            obs is not None
            and self._table_bbox_found(obs)
            and bool(getattr(obs, "edge_found", False))
            and bool(getattr(obs, "valid_for_control", False) or getattr(obs, "usable_for_approach", False))
            and str(getattr(obs, "reject_reason", "") or "").strip().lower() in {"", "none"}
            and getattr(obs, "depth_valid", True) is not False
        )
        approach_stale = bool(stale_level != "fresh")
        approach_dist_needed = bool(dist_err > final_lock_dist_tol)
        approach_mode_active = bool(mode == "CONTROLLED_APPROACH" and phase_name not in {"PLANE_ACQUIRE", "PLANE_STOP"})
        valid_for_safe_forward = bool(
            obs is not None
            and self._table_bbox_found(obs)
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

        if approach_mode_active:
            if approach_stale:
                vx = 0.0
                vy = 0.0
                wz = 0.0
                forward_block_reason = "hard_stale" if stale_level in {"hard_stale", "dead"} else stale_level
                approach_speed_mode = f"blocked_{forward_block_reason}"
            elif hard_guard or not approach_edge_valid:
                vx = 0.0
                vy = 0.0
                wz = 0.0
                forward_block_reason = "edge_invalid"
                approach_speed_mode = "blocked_edge_invalid"
            elif yaw_abs > approach_yaw_realign:
                vx = 0.0
                vy = 0.0
                wz = 0.0
                forward_block_reason = "yaw_need_realign"
                approach_speed_mode = "blocked_yaw_need_realign"
            elif not approach_dist_needed:
                vx = 0.0
                vy = 0.0
                wz = 0.0
                forward_block_reason = "arrived_or_near"
                approach_speed_mode = "blocked_arrived_or_near"
            else:
                vx = approach_vx_norm
                if not approach_allow_vy:
                    vy = 0.0
                if not approach_allow_wz:
                    wz = 0.0
                if yaw_abs <= approach_yaw_deadband:
                    approach_speed_mode = "safe_straight"
                else:
                    approach_speed_mode = "safe_straight_yaw_watch"
                    if not approach_allow_wz:
                        wz = 0.0
                forward_block_reason = "none"
                pose_missing_safe_vx_active = bool(not pose_found)

        vx, vy, wz, speed_summary = self._apply_table_speed_profile(mode, phase_name, vx, vy, wz)
        rotate_gate = self._rotate_gate(obs, edge_yaw_cmd=edge_yaw_cmd)
        if self._yolo_reliable(obs) and abs(float(wz)) > 1e-9 and not bool(rotate_gate.get("allow_rotate", False)):
            wz = 0.0
            speed_summary["wz_radps"] = 0.0
            if control_source == "edge_only":
                control_source = "yolo_assist"
        if approach_mode_active:
            if not approach_allow_vy:
                vy = 0.0
                speed_summary["vy_mps"] = 0.0
            if not approach_allow_wz:
                wz = 0.0
                speed_summary["wz_radps"] = 0.0
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
                "phase": phase_name,
                "table_approach_reason": "plane_confirmed_table_front" if self._plane_stable(obs) else "plane_waiting",
                "approach_source": "plane_only" if control_source == "edge_only" else control_source,
                "control_source": control_source,
                "yolo_edge_blend_weight": float(yolo_blend_weight),
                "yolo_table_control_enable": bool(yolo_control_enabled),
                "yolo_table_edge_stable_count": int(stable_count),
                "yolo_table_blend_start_stable_frames": int(blend_start),
                "yolo_table_near_dist_m": float(getattr(self.cfg, "yolo_table_near_dist_m", 0.45) or 0.45),
                "yolo_near_measured": bool(near_measured),
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
                "edge_found": bool(getattr(obs, "edge_found", False)) if obs is not None else False,
                "usable_for_approach": bool(getattr(obs, "usable_for_approach", False)) if obs is not None else False,
                "valid_for_control": bool(getattr(obs, "valid_for_control", False)) if obs is not None else False,
                "approach_allow_wz": bool(approach_allow_wz),
                "approach_allow_vy": bool(approach_allow_vy),
                "approach_speed_mode": str(approach_speed_mode),
                "approach_safe_vx_mps": float(approach_safe_vx_mps),
                "approach_max_vx_mps": float(approach_max_vx_mps),
                "approach_yaw_deadband_rad": float(approach_yaw_deadband),
                "approach_yaw_realign_rad": float(approach_yaw_realign),
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
                "yolo_yaw_cmd": float(yolo_yaw_cmd),
                "edge_yaw_cmd": float(edge_yaw_cmd),
                "final_yaw_cmd": float(wz),
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
                "yolo_table_control_valid": bool(getattr(obs, "yolo_table_control_valid", getattr(obs, "yolo_reliable", False))) if obs is not None else False,
                "yolo_table_roi_valid": bool(getattr(obs, "yolo_table_roi_valid", False)) if obs is not None else False,
                "yolo_gate_open": bool(getattr(obs, "yolo_gate_open", False)) if obs is not None else False,
                "table_bbox_touch_left": bool(getattr(obs, "table_bbox_touch_left", getattr(obs, "yolo_bbox_touch_left", False))) if obs is not None else False,
                "table_bbox_touch_right": bool(getattr(obs, "table_bbox_touch_right", getattr(obs, "yolo_bbox_touch_right", False))) if obs is not None else False,
                "table_bbox_touch_bottom": bool(getattr(obs, "table_bbox_touch_bottom", getattr(obs, "yolo_bbox_touch_bottom", False))) if obs is not None else False,
                "table_bbox_boundary_allowed": bool(getattr(obs, "table_bbox_boundary_allowed", False)) if obs is not None else False,
                "plane_cx_norm": getattr(obs, "plane_cx_norm", None) if obs is not None else None,
                "plane_width_norm": getattr(obs, "plane_width_norm", None) if obs is not None else None,
                "plane_touch_left": bool(getattr(obs, "plane_touch_left", False)) if obs is not None else False,
                "plane_touch_right": bool(getattr(obs, "plane_touch_right", False)) if obs is not None else False,
            }
        )
        summary.update(rotate_gate)
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
        if self._table_bbox_found(obs) and not self._edge_trusted(obs):
            return self.yolo_table_search_cmd(
                obs,
                mode="CONTROLLED_APPROACH",
                reason="table_bbox_found_edge_not_trusted_default_forward",
                control_source="yolo_forward",
            )
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
