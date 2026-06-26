#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ...config.schema import CarMotionConfig, ControlThresholds
from ...control.types import DockingControlConfig
from ...ipc.protocol import (
    ArmCommand,
    ArmResponse,
    CarState,
    HomeTagObs,
    TableEdgeObs,
    TargetObs,
    TaskCmd,
    make_grasp_req,
    make_tts_event,
    make_vision_idle,
    make_vision_req,
    compute_bbox_control_geometry,
)
from ...bridge.arm_protocol import parse_arm_response
from ...utils.grasp_utils import grasp_to_pose_params
from ...utils.target_utils import target_to_class_id
from ..common import monotonic_ts
from ..context import RuntimeContext, State
from ..controller import MotionController, MotionDecision
from ..control_authority import decide_table_control_authority, ControlAuthority
from ..perception_semantics import build_table_perception_semantics
from ..core_types import (
    KNOWN_VISION_STATUS,
    MOVING_STATES,
    TABLE_APPROACH_STATES,
    TABLE_VISION_STATES,
    TARGET_SEARCH_STATES,
    TARGET_VISION_STATES,
    ObstacleSignal,
    VisionStageBinding,
    _GRASP_ARM_TIMEOUT_S,
    _GRASP_REPOSITION_TIMEOUT_S,
    _GRASP_RESPOND_TIMEOUT_S,
    _GRASP_RESULT_TIMEOUT_S,
    _GRASP_RETRY_LIMIT,
)


class TableDockingMixin:
    def _bbox_control_geometry(self, obs: Optional[TableEdgeObs]) -> Dict[str, object]:
        return compute_bbox_control_geometry(obs)

    def _bbox_yaw_hold_valid(self, obs: Optional[TableEdgeObs]) -> bool:
        """A soft-stale/held bbox center may still own rotate-only control."""
        geom = self._bbox_control_geometry(obs)
        if not bool(geom["bbox_center_valid"]):
            return False
        stale_level = str(self.controller._stale_guard(obs).get("stale_level") or "fresh")
        return stale_level not in {"hard_stale", "dead"}

    def _bbox_lost_hold_or_search(self, obs: Optional[TableEdgeObs], mode: str) -> MotionDecision:
        """Hold table docking state before clearing handoff/searching for bbox loss."""
        self._start_loss_timer("bbox_lost_since_mono")
        self.ctx.bbox_lost_hold_active = True
        hold_age_s = self._loss_elapsed(self.ctx.bbox_lost_since_mono)
        hold_limit_s = max(0.0, float(getattr(self.cfg, "table_loss_hold_s", 1.2) or 1.2))
        stale_level = str(self.controller._stale_guard(obs).get("stale_level") or "fresh")
        hard_lost = stale_level in {"hard_stale", "dead"}
        if not hard_lost and hold_age_s < hold_limit_s:
            last_yaw = getattr(self.ctx, "last_bbox_yaw_cmd", 0.0)
            if abs(last_yaw) > 1e-6:
                wz = last_yaw
            else:
                search_sign = int(self.ctx.search_wz_sign_latched or self.ctx.relocate_turn_sign or 1)
                wz = abs(float(self.car_cfg.search_table_wz_radps)) * search_sign
            cmd = self.controller._cmd(mode, vx=0.0, wz=wz)
            decision = MotionDecision(
                cmd=cmd,
                control_summary=self.controller._summary(mode, cmd, obs, reason="bbox_lost_hold_rotate")
            )
            summary = decision.control_summary
            summary.update(
                {
                    "bbox_lost_hold_active": True,
                    "bbox_lost_hold_age_ms": hold_age_s * 1000.0,
                    "bbox_lost_hold_limit_ms": hold_limit_s * 1000.0,
                    "bbox_lost_hold_reason": "bbox_lost_or_soft_stale",
                    "control_phase": self.ctx.control_phase,
                    "edge_handoff_complete": bool(self.ctx.edge_handoff_complete),
                    "bbox_valid_streak": int(self.ctx.bbox_valid_streak),
                    "bbox_centered_streak": int(self.ctx.bbox_centered_streak),
                    "control_source": "bbox_lost_hold",
                    "yaw_source": "last_bbox_yaw" if abs(last_yaw) > 1e-6 else "search",
                    "allow_forward": False,
                    "allow_rotate": True,
                    "vx_mps": 0.0,
                    "vy_mps": 0.0,
                    "wz_radps": float(wz),
                }
            )
            return decision

        self.ctx.bbox_lost_hold_active = False
        self.ctx.bbox_lost_since_mono = 0.0
        self.ctx.edge_handoff_complete = False
        self.ctx.edge_handoff_started_mono = 0.0
        self.ctx.edge_handoff_timeout = False
        self.ctx.bbox_valid_streak = 0
        self.ctx.bbox_centered_streak = 0
        self.ctx.edge_trusted_streak = 0
        self._transition(State.SEARCH_TABLE, f"{mode} bbox lost hold expired")
        return self.controller.search_table_cmd(*self._get_memory_search_params())

    def _control_phase_status(self, obs: Optional[TableEdgeObs], depth_stop_ready: bool) -> Dict[str, object]:
        """State-independent, hard ownership handoff between search/bbox/edge."""
        now = monotonic_ts()
        current_bbox = self._table_yolo_reliable(obs) or self._bbox_yaw_hold_valid(obs)
        hard_limit = abs(float(getattr(self.car_cfg, "yolo_forward_center_hard_limit", 0.25) or 0.25))
        geom = self._bbox_control_geometry(obs)
        center_error = geom["bbox_center_error_control"]
        touch = bool(obs is not None and (getattr(obs, "table_bbox_touch_left", False) or getattr(obs, "table_bbox_touch_right", False)))
        edge_trusted = bool(current_bbox and self.controller._edge_trusted(obs))
        edge_usable = bool(obs is not None and (getattr(obs, "edge_found", False) or getattr(obs, "usable_for_approach", False)))
        hard_yaw = abs(float(getattr(self.car_cfg, "table_edge_hard_rotate_only_yaw_rad", 0.45) or 0.45))
        edge_yaw = abs(float(getattr(obs, "yaw_err_rad", 0.0) or 0.0)) if obs is not None else 0.0
        edge_score_delta = 0.20 if edge_trusted else (0.05 if edge_usable and edge_yaw <= hard_yaw else -0.15)
        self.ctx.edge_conf_score = max(0.0, min(1.0, float(self.ctx.edge_conf_score) + edge_score_delta))
        if edge_trusted:
            self.ctx.last_edge_good_mono = now
        if not current_bbox:
            if bool(self.ctx.bbox_lost_hold_active):
                phase = self.ctx.control_phase if self.ctx.control_phase else "BBOX_ACQUIRE"
                reason = "bbox_lost_hold"
            else:
                self.ctx.bbox_valid_streak = self.ctx.bbox_centered_streak = self.ctx.edge_trusted_streak = 0
                self.ctx.edge_handoff_complete = False
                self.ctx.approach_commit_active = False
                self.ctx.edge_handoff_started_mono = 0.0
                self.ctx.edge_yaw_ema = None
                phase, reason = "SEARCH_SCAN", "no_current_fresh_bbox"
        else:
            self.ctx.bbox_valid_streak += 1
            centered = bool(center_error is not None and abs(float(center_error)) <= hard_limit and not (touch and abs(float(center_error)) > 0.12))
            self.ctx.bbox_centered_streak = self.ctx.bbox_centered_streak + 1 if centered else 0
            self.ctx.bbox_fov_violation_streak = 0 if centered else self.ctx.bbox_fov_violation_streak + 1
            if not edge_trusted:
                self.ctx.edge_trusted_streak = 0
                if not bool(self.ctx.approach_commit_active):
                    self.ctx.edge_yaw_ema = None
            else:
                self.ctx.edge_trusted_streak += 1
                yaw = float(getattr(obs, "yaw_err_rad", 0.0) or 0.0)
                self.ctx.edge_yaw_ema = yaw if self.ctx.edge_yaw_ema is None else (0.7 * self.ctx.edge_yaw_ema + 0.3 * yaw)
            stable_bbox = self.ctx.bbox_centered_streak >= 2
            bbox_guard_frames = max(1, int(getattr(self.car_cfg, "table_bbox_fov_guard_frames", 3) or 3))
            commit_bbox_safe = bool(
                center_error is not None
                and abs(float(center_error)) <= 0.35
                and int(self.ctx.bbox_fov_violation_streak) < bbox_guard_frames
            )
            recent_edge_good = bool(now - float(self.ctx.last_edge_good_mono or 0.0) <= 0.8)
            commit_coast_ready = bool(
                self.ctx.approach_commit_active
                and commit_bbox_safe
                and not depth_stop_ready
                and float(self.ctx.edge_conf_score) >= 0.25
                and (edge_usable or recent_edge_good)
            )
            dwell_fallback = False
            if not stable_bbox and current_bbox and self.ctx.bbox_valid_streak >= 2:
                if self.ctx.control_phase == "BBOX_ACQUIRE":
                    dwell_ms = (now - float(self.ctx.control_phase_since_mono or now)) * 1000.0
                    if dwell_ms >= 1800.0:
                        edge_available = bool(
                            self.controller._edge_trusted(obs)
                            or (obs is not None and getattr(obs, "edge_found", False) and getattr(obs, "yaw_err_rad", None) is not None and abs(float(obs.yaw_err_rad)) <= 0.45)
                        )
                        not_severe = bool(center_error is not None and abs(float(center_error)) <= 0.35)
                        no_safety = not depth_stop_ready
                        if edge_available and not_severe and no_safety:
                            dwell_fallback = True

            if depth_stop_ready:
                phase, reason = "DEPTH_FINAL_STOP", "stable_dynamic_roi_depth"
                self.ctx.approach_commit_active = False
            elif commit_coast_ready and not edge_trusted:
                phase, reason = "EDGE_GUIDED_APPROACH", "forward_coast_edge_unstable"
            elif not stable_bbox and not dwell_fallback:
                self.ctx.edge_handoff_complete = False
                self.ctx.edge_handoff_started_mono = 0.0
                phase, reason = "BBOX_ACQUIRE", "bbox_not_centered_or_not_stable"
            elif self.ctx.edge_handoff_complete and edge_trusted:
                phase, reason = "EDGE_GUIDED_APPROACH", "bbox_safe_edge_handoff_complete"
            else:
                if self.ctx.edge_handoff_started_mono <= 0.0:
                    self.ctx.edge_handoff_started_mono = now
                timeout_s = 2.0
                self.ctx.edge_handoff_timeout = bool(now - self.ctx.edge_handoff_started_mono >= timeout_s)
                if self.ctx.edge_handoff_timeout:
                    if self.ctx.approach_commit_active and commit_coast_ready:
                        phase, reason = "EDGE_GUIDED_APPROACH", "forward_coast_edge_unstable"
                    elif center_error is not None and abs(float(center_error)) <= 0.10:
                        phase, reason = "EDGE_HANDOFF_CONFIRM", "edge_handoff_timeout"
                    else:
                        phase, reason = "BBOX_ACQUIRE", "edge_handoff_timeout"
                elif self.ctx.edge_trusted_streak >= max(2, int(getattr(self.cfg, "edge_trusted_stable_frames", 3) or 3)):
                    self.ctx.edge_handoff_complete = True
                    phase, reason = "EDGE_GUIDED_APPROACH", "edge_handoff_confirmed"
                else:
                    phase, reason = "EDGE_HANDOFF_CONFIRM", "waiting_edge_trusted_stability"
        if phase != self.ctx.control_phase:
            if phase == "SEARCH_SCAN" or self.ctx.control_phase == "SEARCH_SCAN":
                self.ctx.search_wz_sign_latched = 0
                self.ctx.search_wz_latch_until_mono = 0.0
            self.ctx.control_phase = phase
            self.ctx.control_phase_since_mono = now
        dwell_ms = max(0.0, (now - float(self.ctx.control_phase_since_mono or now)) * 1000.0)
        return {"control_phase": phase, "phase_reason": reason, "bbox_center_error": center_error, **geom,
                "bbox_touch_left": bool(obs is not None and getattr(obs, "table_bbox_touch_left", False)),
                "bbox_touch_right": bool(obs is not None and getattr(obs, "table_bbox_touch_right", False)),
                "phase_dwell_ms": dwell_ms, "edge_handoff_complete": self.ctx.edge_handoff_complete,
                "handoff_timeout": self.ctx.edge_handoff_timeout,
                "edge_conf_score": float(self.ctx.edge_conf_score),
                "approach_commit_active": bool(self.ctx.approach_commit_active)}

    def _get_control_authority(self, obs: Optional[TableEdgeObs], depth_roi_stop_active: bool = False, explicit_stop_active: bool = False) -> ControlAuthority:
        sem = build_table_perception_semantics(obs, self.cfg)
        geom = self._bbox_control_geometry(obs)
        center_error = geom["bbox_center_error_control"]
        phase = self._control_phase_status(obs, depth_roi_stop_active)
        return decide_table_control_authority(
            state=self.ctx.state.value if hasattr(self.ctx.state, "value") else str(self.ctx.state),
            sem=sem,
            cfg=self.car_cfg,
            depth_roi_stop_active=depth_roi_stop_active,
            explicit_stop_active=explicit_stop_active,
            bbox_center_error=center_error,
            control_phase=str(phase["control_phase"]),
            phase_reason=str(phase["phase_reason"]),
            edge_handoff_complete=bool(phase["edge_handoff_complete"]),
            handoff_timeout=bool(phase["handoff_timeout"]),
            phase_dwell_ms=float(phase["phase_dwell_ms"]),
        )

    def _apply_control_authority(self, decision: MotionDecision, obs: Optional[TableEdgeObs], *, explicit_stop_active: bool = False) -> MotionDecision:
        """Apply table-docking authority after raw motion/safety generation.

        This is intentionally a one-way gate: it may zero an axis, never revive
        an axis already stopped by stale, obstacle, depth, emergency, or physical
        safety logic in the controller/service layers.
        """
        summary = decision.control_summary if decision.control_summary is not None else {}
        decision.control_summary = summary
        if bool(summary.get("bbox_lost_hold_active", False)):
            wz = float(summary.get("wz_radps", 0.0))
            decision.cmd.vx_mps = 0.0
            decision.cmd.vy_mps = 0.0
            decision.cmd.wz_radps = wz
            summary.update(
                {
                    "control_source": "bbox_lost_hold",
                    "yaw_source": str(summary.get("yaw_source", "none")),
                    "forward_source": "none",
                    "allow_forward": False,
                    "allow_rotate": True,
                    "forward_block_reason": "bbox_lost_hold",
                    "rotate_block_reason": "",
                    "authority_applied": True,
                    "vx_mps": 0.0,
                    "vy_mps": 0.0,
                    "wz_radps": wz,
                }
            )
            return decision
        raw_forward_reason = str(summary.get("forward_block_reason") or "")
        raw_rotate_reason = str(summary.get("rotate_block_reason") or "")
        depth_status = self._depth_roi_stop_status(obs)
        auth = self._get_control_authority(
            obs,
            depth_roi_stop_active=bool(depth_status.get("depth_roi_stop_ready")),
            explicit_stop_active=explicit_stop_active,
        )
        summary.update(auth.to_dict())
        summary.update(depth_status)
        phase = auth.control_phase
        # Pick one yaw owner after raw controller safety/limit generation; never blend.
        search_sign = int(self.ctx.search_wz_sign_latched or self.ctx.relocate_turn_sign or 1)
        if phase == "SEARCH_SCAN" and not self.ctx.search_wz_sign_latched:
            search_sign, _, _ = self._get_memory_search_params()
            self.ctx.search_wz_sign_latched = 1 if search_sign >= 0 else -1
            self.ctx.search_wz_latch_until_mono = monotonic_ts() + 0.8
        geom = self._bbox_control_geometry(obs)
        center_error = geom["bbox_center_error_control"]
        bbox_wz = 0.0
        deadband = 0.01
        if center_error is not None and abs(float(center_error)) > deadband:
            # Positive wz is right/clockwise. The control error is cx - 0.5,
            # so a right-side bbox must remain a positive yaw command.
            bbox_wz = float(center_error) * 2.0 * float(getattr(self.car_cfg, "yolo_table_yaw_gain", 0.20) or 0.20)
            max_wz = abs(float(getattr(self.car_cfg, "yolo_table_max_wz_radps", 0.06) or 0.06))
            bbox_wz = max(-max_wz, min(max_wz, bbox_wz))
            self.ctx.last_bbox_yaw_cmd = bbox_wz
        edge_wz = float(summary.get("edge_yaw_cmd", summary.get("wz_from_plane", 0.0)) or 0.0)
        sign = lambda value: 0 if abs(value) < deadband else (1 if value > 0.0 else -1)
        bbox_sign, edge_sign = sign(bbox_wz), sign(edge_wz)
        yaw_conflict = bool(bbox_sign and edge_sign and bbox_sign != edge_sign)
        safety_active = bool(summary.get("stale_guard_active") or summary.get("fov_guard_active") or decision.cmd.brake)
        stale_level = str(summary.get("stale_level") or "fresh").lower()
        bbox_owner_active = bool(
            phase == "BBOX_ACQUIRE"
            and bool(geom["bbox_center_valid"])
            and center_error is not None
            and abs(float(center_error)) > deadband
            and stale_level not in {"hard_stale", "dead"}
            and not explicit_stop_active
            and not bool(decision.cmd.brake)
        )
        if bbox_owner_active:
            decision.cmd.vx_mps = 0.0
            decision.cmd.vy_mps = 0.0
            decision.cmd.wz_radps = bbox_wz
        elif not safety_active and auth.allow_rotate:
            if phase == "SEARCH_SCAN":
                decision.cmd.wz_radps = abs(float(self.car_cfg.search_table_wz_radps)) * self.ctx.search_wz_sign_latched
            elif phase in {"BBOX_ACQUIRE", "EDGE_HANDOFF_CONFIRM"}:
                decision.cmd.wz_radps = bbox_wz
            elif phase == "EDGE_GUIDED_APPROACH":
                decision.cmd.wz_radps = edge_wz

        forward_commit_vx = 0.020
        edge_guided_candidate = bool(
            phase == "EDGE_GUIDED_APPROACH"
            and bool(self.ctx.edge_handoff_complete)
            and bool(auth.allow_forward)
            and bool(getattr(obs, "edge_trusted", False))
        )
        forward_coast_candidate = bool(
            phase == "EDGE_GUIDED_APPROACH"
            and bool(self.ctx.approach_commit_active)
            and bool(auth.allow_forward)
        )
        edge_commit_block_reason = ""
        stale_reason = str(summary.get("stale_guard_reason") or "").lower()
        if edge_guided_candidate or forward_coast_candidate:
            if explicit_stop_active or bool(decision.cmd.brake) or any(bool(summary.get(key, False)) for key in ("emergency_stop_active", "car_estop", "estop_active")):
                edge_commit_block_reason = "explicit_stop"
            elif stale_level in {"hard_stale", "dead"} or "perception_dead" in stale_reason or "vision_dead" in stale_reason:
                edge_commit_block_reason = "hard_stale"
            elif bool(depth_status.get("depth_roi_stop_ready")) or auth.stop_source == "roi_depth":
                edge_commit_block_reason = "depth_final_stop"
            elif any(bool(summary.get(key, False)) for key in ("obstacle_active", "obstacle_stop_active", "base_depth_hard_safety", "base_depth_stop_active", "base_depth_emergency_active", "depth_hard_stop_active", "safety_stop_active")):
                edge_commit_block_reason = "base_safety"
            else:
                bbox_guard_frames = max(1, int(getattr(self.car_cfg, "table_bbox_fov_guard_frames", 3) or 3))
                if bool(summary.get("bbox_fov_guard_active", False)) or int(self.ctx.bbox_fov_violation_streak) >= bbox_guard_frames:
                    edge_commit_block_reason = "bbox_fov_guard"
                else:
                    hard_yaw = abs(float(summary.get("hard_rotate_only_yaw_rad", getattr(self.car_cfg, "table_edge_hard_rotate_only_yaw_rad", 0.45)) or 0.45))
                    edge_yaw_abs = abs(float(summary.get("edge_yaw", getattr(obs, "yaw_err_rad", 0.0) if obs is not None else 0.0) or 0.0))
                    if edge_yaw_abs > hard_yaw or bool(getattr(obs, "hard_yaw_rotate_only_active", False)):
                        edge_commit_block_reason = "edge_yaw_too_large"

        edge_guided_commit = bool(edge_guided_candidate and not edge_commit_block_reason)
        forward_coast_active = False
        coast_reason = ""
        if edge_guided_commit:
            decision.cmd.vx_mps = forward_commit_vx
            decision.cmd.vy_mps = 0.0
            decision.cmd.wz_radps = edge_wz
            summary["forward_block_reason"] = ""
            summary["forward_allowed"] = True
            summary["pose_found"] = True
            summary["pose_missing_duration_s"] = 0.0
            summary["pose_missing_safe_vx_active"] = False
        elif forward_coast_candidate and not edge_commit_block_reason:
            edge_usable = bool(obs is not None and (getattr(obs, "edge_found", False) or getattr(obs, "usable_for_approach", False)))
            edge_recent = bool(monotonic_ts() - float(self.ctx.last_edge_good_mono or 0.0) <= 0.8)
            coast_ready = bool(
                geom["bbox_center_valid"]
                and float(self.ctx.edge_conf_score) >= 0.25
                and (edge_usable or edge_recent)
            )
            if coast_ready:
                coast_wz = edge_wz if bool(getattr(obs, "edge_trusted", False)) else float(self.ctx.last_edge_yaw_cmd or self.ctx.edge_yaw_ema or 0.0)
                decision.cmd.vx_mps = forward_commit_vx
                decision.cmd.vy_mps = 0.0
                decision.cmd.wz_radps = coast_wz
                forward_coast_active = True
                coast_reason = "forward_coast_edge_unstable"
                summary["forward_block_reason"] = ""
                summary["forward_allowed"] = True
                summary["pose_found"] = False
                summary["pose_missing_safe_vx_active"] = False
            else:
                edge_commit_block_reason = "edge_confidence_low"
                decision.cmd.vx_mps = 0.0
                summary["forward_block_reason"] = edge_commit_block_reason
                summary["forward_allowed"] = False
        elif edge_guided_candidate:
            decision.cmd.vx_mps = 0.0
            summary["forward_block_reason"] = edge_commit_block_reason
            summary["forward_allowed"] = False
            summary["pose_found"] = False
            summary["pose_missing_safe_vx_active"] = False

        if edge_commit_block_reason in {"explicit_stop", "hard_stale", "depth_final_stop", "base_safety", "bbox_fov_guard", "edge_yaw_too_large"}:
            self.ctx.approach_commit_active = False
        elif phase == "EDGE_GUIDED_APPROACH":
            summary["pose_found"] = False
            summary["pose_missing_safe_vx_active"] = False
            summary["forward_allowed"] = False

        summary.update({
            **geom, "bbox_yaw_cmd": float(bbox_wz),
            "control_phase": phase, "phase_reason": auth.phase_reason,
            "bbox_valid_streak": int(self.ctx.bbox_valid_streak), "bbox_centered_streak": int(self.ctx.bbox_centered_streak),
            "edge_trusted_streak": int(self.ctx.edge_trusted_streak), "edge_yaw": float(getattr(obs, "yaw_err_rad", 0.0) or 0.0) if obs else 0.0,
            "edge_yaw_ema": self.ctx.edge_yaw_ema, "bbox_wz_sign": bbox_sign, "edge_wz_sign": edge_sign,
            "yaw_conflict": yaw_conflict, "search_wz_sign_latched": int(self.ctx.search_wz_sign_latched),
            "bbox_yaw_owner_enforced": bool(bbox_owner_active),
            "forward_commit_vx": float(forward_commit_vx),
            "pose_gate_ignored_for_phase": bool(edge_guided_commit),
            "vx_override_reason": "edge_guided_commit" if edge_guided_commit else "",
            "edge_guided_commit_active": bool(edge_guided_commit),
            "edge_guided_commit_block_reason": edge_commit_block_reason,
            "approach_commit_active": bool(self.ctx.approach_commit_active),
            "forward_coast_active": bool(forward_coast_active),
            "edge_conf_score": float(self.ctx.edge_conf_score),
            "last_edge_yaw_cmd": float(self.ctx.last_edge_yaw_cmd),
            "coast_reason": coast_reason,
            "zero_cmd_age_ms": 0.0,
            "zero_escape_reason": "",
            "search_latch_age_ms": float(max(0.0, (monotonic_ts() - (float(self.ctx.search_wz_latch_until_mono or monotonic_ts()) - 0.8)) * 1000.0)),
            "search_latch_reason": str(self.ctx.current_search_direction_reason or "latched_search_direction"),
            "wz_sign_final": sign(float(decision.cmd.wz_radps)),
            "edge_handoff_complete": bool(self.ctx.edge_handoff_complete), "handoff_timeout": bool(self.ctx.edge_handoff_timeout),
            "phase_dwell_ms": float(max(0.0, (monotonic_ts() - float(self.ctx.control_phase_since_mono or monotonic_ts())) * 1000.0)),
        })
        summary["authority_applied"] = True
        summary["mode"] = str(decision.cmd.mode)
        if not auth.allow_forward and abs(float(decision.cmd.vx_mps)) > 1e-9:
            decision.cmd.vx_mps = 0.0
        if not auth.allow_rotate and abs(float(decision.cmd.wz_radps)) > 1e-9:
            decision.cmd.wz_radps = 0.0
        safety_tokens = ("stale", "emergency", "explicit", "obstacle", "hard_guard", "depth_p10", "safety")
        if not edge_guided_commit and auth.allow_forward and raw_forward_reason and any(token in raw_forward_reason.lower() for token in safety_tokens):
            summary["forward_block_reason"] = raw_forward_reason
            summary["block_reason"] = raw_forward_reason
        if auth.allow_rotate and raw_rotate_reason and any(token in raw_rotate_reason.lower() for token in safety_tokens):
            summary["rotate_block_reason"] = raw_rotate_reason
            summary["block_reason"] = raw_rotate_reason

        # Force bbox acquire yaw owner enforcement
        if phase == "BBOX_ACQUIRE" and center_error is not None and abs(float(center_error)) > deadband:
            safety_stop = (
                explicit_stop_active
                or bool(decision.cmd.brake)
                or any(bool(summary.get(key, False)) for key in ("emergency_stop_active", "car_estop", "estop_active"))
                or any(bool(summary.get(key, False)) for key in ("obstacle_active", "obstacle_stop_active", "base_depth_hard_safety", "base_depth_stop_active", "base_depth_emergency_active", "depth_hard_stop_active", "safety_stop_active"))
            )
            if not safety_stop:
                decision.cmd.vx_mps = 0.0
                decision.cmd.vy_mps = 0.0
                decision.cmd.wz_radps = bbox_wz
                bbox_owner_active = True
                summary["bbox_yaw_owner_enforced"] = True

        safety_stop_active = bool(
            explicit_stop_active
            or bool(decision.cmd.brake)
            or bool(depth_status.get("depth_roi_stop_ready"))
            or edge_commit_block_reason in {"explicit_stop", "hard_stale", "depth_final_stop", "base_safety", "bbox_fov_guard", "edge_yaw_too_large"}
        )
        active_docking = str(getattr(self.ctx.state, "value", self.ctx.state) or "") in {"YOLO_ACQUIRE_ALIGN", "YOLO_APPROACH", "EDGE_ADJUST"}
        zero_eps = 1e-6
        if active_docking and not safety_stop_active and abs(float(decision.cmd.vx_mps)) < zero_eps and abs(float(decision.cmd.wz_radps)) < zero_eps:
            now_mono = monotonic_ts()
            if float(self.ctx.zero_cmd_started_mono or 0.0) <= 0.0:
                self.ctx.zero_cmd_started_mono = now_mono
            zero_age_s = max(0.0, now_mono - float(self.ctx.zero_cmd_started_mono))
            summary["zero_cmd_age_ms"] = zero_age_s * 1000.0
            if zero_age_s >= 0.8:
                coast_safe = bool(
                    self.ctx.approach_commit_active
                    and bool(geom["bbox_center_valid"])
                    and float(self.ctx.edge_conf_score) >= 0.25
                )
                if coast_safe:
                    decision.cmd.vx_mps = forward_commit_vx
                    decision.cmd.vy_mps = 0.0
                    decision.cmd.wz_radps = float(self.ctx.last_edge_yaw_cmd or self.ctx.edge_yaw_ema or 0.0)
                    summary["forward_coast_active"] = True
                    summary["forward_source"] = "approach_commit"
                    summary["coast_reason"] = "zero_watchdog_forward_coast"
                    summary["zero_escape_reason"] = "forward_coast"
                elif geom["bbox_center_valid"] and center_error is not None and abs(float(center_error)) > deadband:
                    decision.cmd.wz_radps = bbox_wz
                    summary["zero_escape_reason"] = "bbox_reacquire_rotate"
                elif not bool(geom["bbox_center_valid"]):
                    decision.cmd.wz_radps = abs(float(self.car_cfg.search_table_wz_radps)) * search_sign
                    summary["zero_escape_reason"] = "search_rotate"
        else:
            self.ctx.zero_cmd_started_mono = 0.0

        # Check approach progress if phase is EDGE_GUIDED_APPROACH and vx > 0
        if phase == "EDGE_GUIDED_APPROACH" and decision.cmd.vx_mps > 0.0:
            if self._check_approach_progress(obs):
                decision = self._enter_no_progress_recovery_or_next("接近无进展超时")
                summary.update(decision.control_summary)
                summary["final_vx"] = float(decision.cmd.vx_mps)
                summary["final_vy"] = float(decision.cmd.vy_mps)
                summary["final_wz"] = float(decision.cmd.wz_radps)
                summary["vx_mps"] = float(decision.cmd.vx_mps)
                summary["vy_mps"] = float(decision.cmd.vy_mps)
                summary["wz_radps"] = float(decision.cmd.wz_radps)
                return decision
        else:
            self.ctx.min_dist_seen = 999.0
            self.ctx.dist_progress_last_refreshed_mono = 0.0
            self.ctx.dist_missing_started_mono = 0.0

        if phase == "EDGE_GUIDED_APPROACH" and decision.cmd.vx_mps > 0.0 and not safety_stop_active:
            self.ctx.approach_commit_active = True
            self.ctx.last_forward_cmd_mono = monotonic_ts()
            self.ctx.last_edge_yaw_cmd = float(decision.cmd.wz_radps)
        summary["approach_commit_active"] = bool(self.ctx.approach_commit_active)
        summary["last_edge_yaw_cmd"] = float(self.ctx.last_edge_yaw_cmd)

        # Always expose the final axis values, including a pre-existing safety stop.
        summary["final_vx"] = float(decision.cmd.vx_mps)
        summary["final_vy"] = float(decision.cmd.vy_mps)
        summary["final_wz"] = float(decision.cmd.wz_radps)
        summary["vx_mps"] = float(decision.cmd.vx_mps)
        summary["vy_mps"] = float(decision.cmd.vy_mps)
        summary["wz_radps"] = float(decision.cmd.wz_radps)
        return decision

    def _tick_search_table(self) -> MotionDecision:
        decision = self._tick_search_table_impl()
        obs = self._fresh_table_obs()
        explicit = "warmup" in str(decision.control_summary.get("control_source", "")) or "recovery" in str(decision.control_summary.get("control_source", "")) or "failed" in str(decision.control_summary.get("control_source", ""))
        decision = self._apply_control_authority(decision, obs, explicit_stop_active=explicit)
        self._ensure_speed_profile(decision)
        return decision

    def _tick_yolo_acquire_align(self) -> MotionDecision:
        decision = self._tick_yolo_acquire_align_impl()
        obs = self._fresh_table_obs()
        decision = self._apply_control_authority(decision, obs)
        self._ensure_speed_profile(decision)
        return decision

    def _tick_yolo_approach(self) -> MotionDecision:
        decision = self._tick_yolo_approach_impl()
        obs = self._fresh_table_obs()
        decision = self._apply_control_authority(decision, obs)
        self._ensure_speed_profile(decision)
        return decision

    def _tick_edge_adjust(self) -> MotionDecision:
        decision = self._tick_edge_adjust_impl()
        obs = self._fresh_table_obs()
        decision = self._apply_control_authority(decision, obs)
        self._ensure_speed_profile(decision)
        return decision

    def _tick_final_slow_stop(self) -> MotionDecision:
        decision = self._tick_final_slow_stop_impl()
        obs = self._fresh_table_obs()
        decision = self._apply_control_authority(decision, obs)
        self._ensure_speed_profile(decision)
        return decision

    def _tick_at_table_edge(self) -> MotionDecision:
        decision = self._tick_at_table_edge_impl()
        obs = self._fresh_table_obs()
        decision = self._apply_control_authority(decision, obs)
        self._ensure_speed_profile(decision)
        return decision

    def _ensure_speed_profile(self, decision: MotionDecision) -> None:
        """Populate speed_profile in control_summary if not already set."""
        if decision.control_summary is not None and "speed_profile" not in decision.control_summary:
            mode = decision.control_summary.get("state", decision.cmd.mode)
            decision.control_summary["speed_profile"] = self.controller._table_speed_profile_name(mode, "")

    def _get_memory_search_params(self) -> Tuple[int, str, str]:
        turn_sign = self.ctx.relocate_turn_sign
        search_src = "default"
        search_dir = "no_memory"
        
        mem_age = monotonic_ts() - self.ctx.last_table_seen_ts
        timeout = float(getattr(self.cfg, "table_memory_timeout_sec", 3.0) or 3.0)
        
        if mem_age > timeout:
            self.ctx.last_table_side = "unknown"
            
        side = self.ctx.last_table_side
        if side == "left":
            turn_sign = 1
            search_src = "memory"
            search_dir = "memory_left"
        elif side == "right":
            turn_sign = -1
            search_src = "memory"
            search_dir = "memory_right"
        elif side == "center":
            center_hold = float(getattr(self.cfg, "table_center_loss_hold_sec", 1.0) or 1.0)
            if mem_age <= center_hold:
                search_src = "memory"
                search_dir = "memory_center_hold"
            else:
                search_dir = "no_memory"
                
        self.ctx.current_search_direction_source = search_src
        self.ctx.current_search_direction_reason = search_dir
        return turn_sign, search_src, search_dir

    def _check_approach_progress(self, obs: Optional[TableEdgeObs]) -> bool:
        progress_window_ms = float(getattr(self.cfg, "progress_window_ms", 5000.0) or 5000.0)
        curr_dist = None
        if obs is not None:
            if obs.dist_err_m is not None:
                curr_dist = float(obs.target_dist_m or self.cfg.table_target_dist_m) + float(obs.dist_err_m)
            elif bool(getattr(obs, "table_roi_depth_valid", False)) and getattr(obs, "table_roi_depth_median", None) is not None:
                curr_dist = float(obs.table_roi_depth_median)

        now = monotonic_ts()

        if curr_dist is None:
            if self.ctx.dist_missing_started_mono <= 0.0:
                self.ctx.dist_missing_started_mono = now
                return False
            elapsed_ms = (now - self.ctx.dist_missing_started_mono) * 1000.0
            if elapsed_ms >= progress_window_ms:
                self._log("warn", f"Approach progress timeout (distance missing): elapsed {elapsed_ms:.1f}ms >= window {progress_window_ms}ms")
                return True
            return False

        if curr_dist < self.ctx.min_dist_seen - 0.002:
            self.ctx.min_dist_seen = curr_dist
            self.ctx.dist_progress_last_refreshed_mono = now
            self.ctx.dist_missing_started_mono = 0.0
            self._log("info", f"Approach progress refreshed: min_dist_seen={self.ctx.min_dist_seen:.4f}m")
        else:
            if self.ctx.dist_progress_last_refreshed_mono <= 0.0:
                self.ctx.dist_progress_last_refreshed_mono = now
                self.ctx.dist_missing_started_mono = 0.0
                return False
            elapsed_ms = (now - self.ctx.dist_progress_last_refreshed_mono) * 1000.0
            if elapsed_ms >= progress_window_ms:
                self._log("warn", f"Approach progress timeout: distance did not decrease by 2mm for {elapsed_ms:.1f}ms. Min dist seen: {self.ctx.min_dist_seen:.4f}m, current: {curr_dist:.4f}m")
                return True
        return False

    def _table_motion_signal_available(self, obs: Optional[TableEdgeObs]) -> bool:
        """Only a current, fresh YOLO bbox may leave local rotate search."""
        return self._table_yolo_reliable(obs)

    def _annotate_hard_yaw_gate(self, obs: Optional[TableEdgeObs]) -> None:
        if obs is None:
            return
        yaw_abs = abs(float(getattr(obs, "yaw_err_rad", 0.0) or 0.0))
        hard_yaw = max(
            abs(float(getattr(self.car_cfg, "table_edge_hard_rotate_only_yaw_rad", 0.45) or 0.45)),
            abs(float(getattr(self.car_cfg, "table_approach_yaw_realign_rad", 0.16) or 0.16)),
        )
        now = monotonic_ts()
        if yaw_abs > hard_yaw:
            if self.ctx.edge_hard_yaw_since_mono <= 0.0:
                self.ctx.edge_hard_yaw_since_mono = now
                self.ctx.edge_hard_yaw_frames = 0
            self.ctx.edge_hard_yaw_frames += 1
        else:
            self.ctx.edge_hard_yaw_since_mono = 0.0
            self.ctx.edge_hard_yaw_frames = 0
        min_frames = max(1, int(getattr(self.car_cfg, "table_edge_hard_yaw_rotate_only_frames", 3) or 3))
        min_ms = max(0, int(getattr(self.car_cfg, "table_edge_hard_yaw_rotate_only_ms", 350) or 350))
        elapsed_ms = max(0.0, (now - float(self.ctx.edge_hard_yaw_since_mono or now)) * 1000.0)
        active = bool(yaw_abs > hard_yaw and self.ctx.edge_hard_yaw_frames >= min_frames and elapsed_ms >= float(min_ms))
        setattr(obs, "hard_yaw_rotate_only_active", active)
        setattr(obs, "hard_yaw_frames", int(self.ctx.edge_hard_yaw_frames))
        setattr(obs, "hard_yaw_elapsed_ms", float(elapsed_ms))
        setattr(obs, "hard_rotate_only_yaw_rad", float(hard_yaw))

    def _approach_from_table_signal(self, obs: TableEdgeObs, *, reason: str) -> MotionDecision:
        if not self._table_yolo_reliable(obs):
            # Edge/plane/depth facts are useful diagnostics, never an approach gate.
            return self.controller.search_table_cmd(*self._get_memory_search_params())
        self._annotate_hard_yaw_gate(obs)
        yaw_abs = abs(float(getattr(obs, "yaw_err_rad", 0.0) or 0.0))
        hard_yaw = float(getattr(obs, "hard_rotate_only_yaw_rad", getattr(self.car_cfg, "table_edge_hard_rotate_only_yaw_rad", 0.45)) or 0.45)
        edge_guided = bool((getattr(obs, "edge_valid", False) or getattr(obs, "edge_trusted", False)) and yaw_abs <= hard_yaw)
        if edge_guided:
            # Edge guidance is a per-frame control source, not a task state.
            decision = self.controller.fov_table_approach_cmd(obs, phase="PLANE_APPROACH", mode=self.ctx.state.value)
            if decision.control_summary is not None:
                decision.control_summary.update(
                    {
                        "control_source": "edge_guided_forward",
                        "transition_reason": reason,
                        "hard_yaw_frames": int(getattr(obs, "hard_yaw_frames", 0) or 0),
                        "hard_yaw_elapsed_ms": float(getattr(obs, "hard_yaw_elapsed_ms", 0.0) or 0.0),
                    }
                )
            return decision
        if self.ctx.state != State.YOLO_APPROACH:
            self._transition(State.YOLO_APPROACH, reason)
        decision = self.controller.yolo_table_search_cmd(
            obs,
            turn_sign=self.ctx.relocate_turn_sign,
            mode="YOLO_APPROACH",
            reason=reason,
            control_source="yolo_track_forward",
        )
        return decision

    def _tick_yolo_acquire_align_impl(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        obs = self._fresh_table_obs()
        if not self._table_yolo_reliable(obs) and not self._bbox_yaw_hold_valid(obs):
            return self._bbox_lost_hold_or_search(obs, "YOLO_ACQUIRE_ALIGN")
        self.ctx.bbox_lost_hold_active = False
        self.ctx.bbox_lost_since_mono = 0.0
        self._reset_table_loss()
        geom = self._bbox_control_geometry(obs)
        cx_norm = geom["bbox_cx_norm_control"]
        center_error = geom["bbox_center_error_control"]
        if center_error is None:
            return self._bbox_lost_hold_or_search(obs, "YOLO_ACQUIRE_ALIGN")
        hard_limit = abs(float(getattr(self.car_cfg, "yolo_forward_center_hard_limit", 0.25) or 0.25))
        self._annotate_hard_yaw_gate(obs)
        yaw_err = abs(float(getattr(obs, "yaw_err_rad", 0.0) or 0.0))
        hard_yaw = max(
            abs(float(getattr(self.car_cfg, "table_approach_yaw_realign_rad", 0.16) or 0.16)),
            abs(float(getattr(self.car_cfg, "table_edge_hard_rotate_only_yaw_rad", 0.45) or 0.45)),
        )
        touch_side = bool(
            getattr(obs, "table_bbox_touch_left", getattr(obs, "yolo_bbox_touch_left", False))
            or getattr(obs, "table_bbox_touch_right", getattr(obs, "yolo_bbox_touch_right", False))
        )
        if (getattr(obs, "edge_valid", False) or getattr(obs, "edge_trusted", False)) and yaw_err <= hard_yaw:
            return self._approach_from_table_signal(obs, reason=f"YOLO_ACQUIRE_ALIGN edge forward handoff yaw={yaw_err:.3f}")
        if abs(center_error) <= hard_limit:
            if self.ctx.state != State.YOLO_APPROACH:
                self._transition(State.YOLO_APPROACH, f"YOLO_ACQUIRE_ALIGN bbox trackable (cx_norm={cx_norm:.3f}), transition to YOLO_APPROACH")
            decision = self.controller.yolo_table_search_cmd(
                obs,
                turn_sign=self.ctx.relocate_turn_sign,
                mode="YOLO_APPROACH",
                reason="yolo_acquire_track_forward",
                control_source="yolo_track_forward",
            )
            decision.control_summary.update(
                {
                    "yolo_acquire_align_active": True,
                    "bbox_cx_norm": float(cx_norm),
                    "center_error": float(center_error),
                    "transition_reason": self.ctx.last_enter_reason,
                }
            )
            return decision
        
        # Rotate in place towards center
        cx = (cx_norm * 2.0) - 1.0
        gain = float(getattr(self.car_cfg, "yolo_table_yaw_gain", 0.20) or 0.20)
        max_wz = abs(float(getattr(self.car_cfg, "yolo_table_max_wz_radps", 0.06) or 0.06))
        sign = float(getattr(self.car_cfg, "table_view_wz_sign", -1.0))
        wz_raw = cx * gain * sign
        wz = max(-max_wz, min(max_wz, wz_raw))
        if abs(wz) < 0.02:
            wz = 0.0
            
        cmd = self.controller._cmd("YOLO_ACQUIRE_ALIGN", vx=0.0, wz=wz)
        decision = MotionDecision(
            cmd=cmd,
            control_summary=self.controller._summary("YOLO_ACQUIRE_ALIGN", cmd, obs, reason="yolo_acquire_align"),
        )
        decision.control_summary.update({
            "control_source": "yolo_align",
            "yolo_acquire_align_active": True,
            "bbox_cx_norm": float(cx_norm),
            "vx_mps": 0.0,
            "vy_mps": 0.0,
            "wz_radps": float(wz),
            "allow_forward": False,
            "allow_rotate": True,
            "forward_block_reason": "yolo_bbox_touch_side_rotate_only" if touch_side else "yolo_center_error_too_large_rotate_only",
            "rotate_block_reason": "",
            "speed_profile": "search",
            "speed_limit_reason": "yolo_align",
        })
        return decision

    def _tick_yolo_approach_impl(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        obs = self._fresh_table_obs()
        if not self._table_yolo_reliable(obs) and not self._bbox_yaw_hold_valid(obs):
            return self._bbox_lost_hold_or_search(obs, "YOLO_APPROACH")
        self.ctx.bbox_lost_hold_active = False
        self.ctx.bbox_lost_since_mono = 0.0
        self._reset_table_loss()
        if self.controller._edge_trusted(obs):
            decision = self.controller.fov_table_approach_cmd(
                obs,
                phase="PLANE_APPROACH",
                mode="YOLO_APPROACH",
            )
            if decision.control_summary is not None:
                decision.control_summary.update(
                    {
                        "control_source": "edge_guided_forward",
                        "yaw_source": "edge",
                        "forward_source": "yolo_or_edge",
                        "transition_reason": "YOLO_APPROACH table edge trusted, stay forward-controlled",
                        "edge_trusted_yolo_approach_handoff": True,
                    }
                )
            return decision
            
        decision = self.controller.yolo_table_search_cmd(
            obs,
            turn_sign=self.ctx.relocate_turn_sign,
            mode="YOLO_APPROACH",
            reason="yolo_approach_forward",
            control_source="yolo_forward",
        )
        return decision

    def _tick_search_table_impl(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        obs = self._fresh_table_obs()
        yolo_status = self._yolo_table_status(obs)
        warmup_s = max(0.0, float(getattr(self.car_cfg, "table_perception_warmup_s", 1.0) or 1.0))
        task_elapsed_s = max(0.0, time.time() - float(getattr(self.ctx, "task_start_wall_ts", 0.0) or time.time()))
        if task_elapsed_s <= warmup_s:
            if obs is not None and self._table_motion_signal_available(obs):
                return self._approach_from_table_signal(obs, reason=f"perception_warmup_signal_seen elapsed_s={task_elapsed_s:.2f}")
            decision = self.controller.stop_cmd("SEARCH_TABLE")
            decision.control_summary.update(
                {
                    "control_source": "perception_warmup_hold",
                    "allow_forward": False,
                    "allow_rotate": False,
                    "forward_block_reason": "perception_warmup_waiting",
                    "rotate_block_reason": "perception_warmup_waiting",
                    "perception_warmup_active": True,
                    "perception_warmup_elapsed_s": float(task_elapsed_s),
                    "perception_warmup_s": float(warmup_s),
                    "transition_reason": "perception_warmup_waiting",
                }
            )
            return decision
        local_search_active = bool(
            self.ctx.prev_state in TABLE_APPROACH_STATES
            and ("丢失" in str(self.ctx.last_enter_reason or "") or "lost" in str(self.ctx.last_enter_reason or "").lower())
        )
        local_search_elapsed_s = self._state_elapsed() if local_search_active else 0.0
        local_search_timeout_s = max(
            0.0,
            float(
                getattr(
                    self.cfg,
                    "no_table_bbox_timeout_s",
                    getattr(self.cfg, "rotate_search_timeout_s", 10.0),
                )
                or getattr(self.cfg, "rotate_search_timeout_s", 10.0)
            ),
        )
        if local_search_active and local_search_elapsed_s >= local_search_timeout_s:
            if yolo_status["fresh"]:
                self._log(
                    "warn",
                    (
                        "[TABLE_TIMEOUT] suppressed table_lost_search_timeout "
                        f"current_state={self.ctx.state.value} selected_timeout_reason=bbox_visible_but_edge_invalid "
                        f"fallback_action=yolo_assist "
                        f"yolo_table_visible={int(yolo_status['visible'])} "
                        f"yolo_table_fresh={int(yolo_status['fresh'])} "
                        f"yolo_table_age_ms={yolo_status['age_ms']} "
                        f"edge_valid={int(bool(getattr(obs, 'edge_valid', False))) if obs is not None else 0} "
                        f"edge_trusted={int(bool(getattr(obs, 'edge_trusted', False))) if obs is not None else 0} "
                        f"edge_age_ms={self._table_obs_age_ms(obs)}"
                    ),
                )
            else:
                self.ctx.last_fail_reason = "table_lost_search_timeout:no_table_bbox_timeout"
                self._log(
                    "warn",
                    (
                        "[TABLE_TIMEOUT] triggering table_lost_search_timeout "
                        f"current_state={self.ctx.state.value} selected_timeout_reason=no_table_bbox_timeout "
                        f"fallback_action=error_recovery "
                        f"yolo_table_visible={int(yolo_status['visible'])} "
                        f"yolo_table_fresh={int(yolo_status['fresh'])} "
                        f"yolo_table_age_ms={yolo_status['age_ms']} "
                        f"edge_valid={int(bool(getattr(obs, 'edge_valid', False))) if obs is not None else 0} "
                        f"edge_trusted={int(bool(getattr(obs, 'edge_trusted', False))) if obs is not None else 0} "
                        f"edge_age_ms={self._table_obs_age_ms(obs)}"
                    ),
                )
                self._enter_error_recovery(self.ctx.last_fail_reason, tts_text="桌边丢失搜索超时，已停车", interrupt_tts=True)
                decision = self.controller.stop_cmd("ERROR_RECOVERY", brake=True)
                decision.control_summary.update(
                    {
                        "control_source": "search_failed_stop",
                        "table_lost_search_active": True,
                        "table_lost_search_elapsed_s": float(local_search_elapsed_s),
                        "table_lost_search_timeout": True,
                        "no_table_bbox_timeout": True,
                        "selected_timeout_reason": "no_table_bbox_timeout",
                        "fallback_action": "error_recovery",
                        "yolo_table_visible": bool(yolo_status["visible"]),
                        "yolo_table_fresh": bool(yolo_status["fresh"]),
                        "yolo_table_age_ms": yolo_status["age_ms"],
                        "edge_valid": bool(getattr(obs, "edge_valid", False)) if obs is not None else False,
                        "edge_trusted": bool(getattr(obs, "edge_trusted", False)) if obs is not None else False,
                        "edge_age_ms": self._table_obs_age_ms(obs),
                        "stop_source_state": "SEARCH_TABLE",
                        "stop_reason": self.ctx.last_fail_reason,
                    }
                )
                return decision
        level = self._control_level(obs)
        if obs is not None and self._table_motion_signal_available(obs):
            geom = self._bbox_control_geometry(obs)
            cx_norm = geom["bbox_cx_norm_control"]
            center_error = geom["bbox_center_error_control"]
            if center_error is not None:
                hard_limit = abs(float(getattr(self.car_cfg, "yolo_forward_center_hard_limit", 0.25) or 0.25))
                self._annotate_hard_yaw_gate(obs)
                yaw_err = abs(float(getattr(obs, "yaw_err_rad", 0.0) or 0.0))
                hard_yaw = max(
                    abs(float(getattr(self.car_cfg, "table_approach_yaw_realign_rad", 0.16) or 0.16)),
                    abs(float(getattr(self.car_cfg, "table_edge_hard_rotate_only_yaw_rad", 0.45) or 0.45)),
                )
                touch_side = bool(
                    getattr(obs, "table_bbox_touch_left", getattr(obs, "yolo_bbox_touch_left", False))
                    or getattr(obs, "table_bbox_touch_right", getattr(obs, "yolo_bbox_touch_right", False))
                )
                next_state = State.YOLO_APPROACH if abs(center_error) <= hard_limit else State.YOLO_ACQUIRE_ALIGN
                self._transition(next_state, f"table signal found cx_norm={cx_norm:.3f} center_error={center_error:.3f}")
                decision = self.controller.yolo_table_search_cmd(
                    obs,
                    turn_sign=self.ctx.relocate_turn_sign,
                    mode=next_state.value,
                    reason="search_table_yolo_track_forward" if next_state == State.YOLO_APPROACH else "search_table_yolo_rotate_only",
                    control_source="yolo_track_forward" if next_state == State.YOLO_APPROACH else "local_rotate_search",
                )
                return decision
        # Without table bbox, docking/edge must not drive state transitions.
        # It can still be logged by vision, but control falls back to local rotate search.
        self.ctx.table_found_frames = 0
        if self._state_elapsed() >= float(self.cfg.search_table_timeout_s):
            self.ctx.last_fail_reason = "搜索桌边超时"
            self._transition(State.NEXT_TABLE, self.ctx.last_fail_reason)
            return self.controller.next_table_cmd(turn_sign=self.ctx.relocate_turn_sign)
        decision = self.controller.search_table_cmd(*self._get_memory_search_params())
        decision.control_summary.update(
            {
                "control_source": "local_rotate_search",
                "table_lost_search_active": bool(local_search_active),
                "table_lost_search_elapsed_s": float(local_search_elapsed_s),
                "table_lost_search_timeout": False,
                "no_table_bbox_timeout": False,
                "selected_timeout_reason": "bbox_visible_but_edge_invalid" if yolo_status["visible"] else "searching_no_table_bbox",
                "fallback_action": "yolo_assist" if yolo_status["fresh"] else "local_rotate_search",
                "yolo_table_visible": bool(yolo_status["visible"]),
                "yolo_table_fresh": bool(yolo_status["fresh"]),
                "yolo_table_age_ms": yolo_status["age_ms"],
                "edge_valid": bool(getattr(obs, "edge_valid", False)) if obs is not None else False,
                "edge_trusted": bool(getattr(obs, "edge_trusted", False)) if obs is not None else False,
                "edge_age_ms": self._table_obs_age_ms(obs),
                "search_table_stale_gate_bypass": True,
            }
        )
        return decision

    def _tick_edge_adjust_impl(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        obs = self._fresh_table_obs()
        if not self._table_yolo_reliable(obs) and not self._bbox_yaw_hold_valid(obs):
            return self._bbox_lost_hold_or_search(obs, "EDGE_ADJUST")
        self.ctx.bbox_lost_hold_active = False
        self.ctx.bbox_lost_since_mono = 0.0
        self._reset_table_loss()

        level = self._control_level(obs)
        if level == "none":
            if self._table_yolo_reliable(obs):
                decision = self.controller.yolo_table_search_cmd(
                    obs,
                    mode="EDGE_ADJUST",
                    reason="edge_adjust_level_none_yolo_forward",
                    control_source="yolo_forward",
                )
                return decision
            decision = self.controller.stop_cmd("EDGE_ADJUST")
            return decision
        if level == "stop" or self._edge_ready(obs):
            decision = self._enter_final_slow_stop_or_keep_approach(obs, "满足最终停车条件，进入慢速停车")
            return decision

        if not self.ctx.table_dock_phase:
            self.ctx.table_dock_phase = "aligning"
            self.ctx.table_dock_phase_since_mono = monotonic_ts()

        phase_since = self.ctx.table_dock_phase_since_mono
        if phase_since <= 0.0:
            phase_since = self.ctx.state_enter_mono
        phase_elapsed = max(0.0, monotonic_ts() - phase_since) if phase_since > 0.0 else 0.0

        yaw_ready = self._yaw_ready_for_controlled_approach(obs)
        yaw_needs_realign = self._yaw_needs_realign_from_approach(obs)

        if self.ctx.table_dock_phase == "aligning":
            if self._count_table_motion_hysteresis_obs(
                obs,
                ok=yaw_ready,
                last_key_attr="align_hysteresis_last_obs_key",
                count_attr="approach_aligned_frames",
            ) >= self._align_to_approach_stable_obs() and phase_elapsed >= self._coarse_align_min_dwell_s():
                self.ctx.table_dock_phase = "approaching"
                self.ctx.table_dock_phase_since_mono = monotonic_ts()
                self.ctx.approach_realign_frames = 0
                self.ctx.table_motion_pending_transition_reason = ""
            else:
                self.ctx.table_motion_pending_transition_reason = "align_to_approach_pending"
        elif self.ctx.table_dock_phase == "approaching":
            if self._count_table_motion_hysteresis_obs(
                obs,
                ok=yaw_needs_realign,
                last_key_attr="approach_hysteresis_last_obs_key",
                count_attr="approach_realign_frames",
            ) >= self._approach_to_align_stable_obs() and phase_elapsed >= self._controlled_approach_min_dwell_s():
                self.ctx.table_dock_phase = "aligning"
                self.ctx.table_dock_phase_since_mono = monotonic_ts()
                self.ctx.approach_aligned_frames = 0
                self.ctx.table_motion_pending_transition_reason = ""
            else:
                self.ctx.table_motion_pending_transition_reason = "approach_to_align_pending"


        pending_reason = self.ctx.table_motion_pending_transition_reason

        if self.ctx.table_dock_phase == "aligning":
            decision = self.controller.fov_table_approach_cmd(obs, phase="PLANE_FINAL_LOCK", mode="COARSE_ALIGN")
            decision.control_summary["control_intent"] = "coarse_edge_adjust"
        else:
            decision = self._table_approach_decision(obs, phase="PLANE_APPROACH")
            decision.control_summary["control_intent"] = "edge_parallel"
        decision = self._annotate_table_motion_hysteresis(decision, pending_reason=pending_reason)
        return decision

    def _tick_final_slow_stop_impl(self) -> MotionDecision:
        elapsed_ms = self._state_elapsed() * 1000.0
        is_holding = elapsed_ms < self.cfg.final_lock_min_hold_ms

        if not self._table_final_lock_enabled():
            if is_holding:
                obs = self._fresh_table_obs()
                status = self._final_lock_status(obs, stable_count=self.ctx.table_lock_frames)
                return self._annotate_final_lock_decision(self.controller.stop_cmd("FINAL_SLOW_STOP"), status)
            obs = self._fresh_table_obs()
            if self._table_visible(obs):
                self._transition(State.EDGE_ADJUST, "final_lock disabled，回到微调状态")
                return self._table_approach_decision(obs, phase="PLANE_APPROACH", stop_ready_ignored=True)
            return self._handle_table_loss("final_lock disabled 且桌边丢失，回到搜索", State.SEARCH_TABLE, "FINAL_LOCK_DISABLED_HOLD")

        obs = self._fresh_table_obs()
        if not self._table_visible(obs):
            stale_obs = self.ctx.last_table_obs if obs is None else obs
            status = self._update_final_lock_count(stale_obs if stale_obs is not None else obs)
            
            # Start loss timer if not started
            self._start_loss_timer("table_loss_since_mono")
            loss_ms = self._loss_elapsed(self.ctx.table_loss_since_mono) * 1000.0
            
            if is_holding or loss_ms < self.cfg.final_lock_lost_timeout_ms:
                return self._annotate_final_lock_decision(self.controller.stop_cmd("FINAL_SLOW_STOP"), status)
                
            self.ctx.table_lost_frames += 1
            reason = str(status.get("reason") or "")
            self._log_final_lock_summary(stale_obs if stale_obs is not None else obs, lock_ready=False, reason=reason, stable_count=self.ctx.table_lock_frames, status=status)
            if str(status.get("lock_count_hold_reason") or ""):
                return self._annotate_final_lock_decision(self.controller.stop_cmd("FINAL_SLOW_STOP"), status)
            return self._handle_table_loss("最终停车时桌边丢失，回到搜索", State.SEARCH_TABLE, "FINAL_LOCK_HOLD")

        self._reset_table_loss()

        phase = str(self.ctx.table_dock_phase or "APPROACH").upper()
        if phase == "APPROACH":
            status = self._update_final_lock_count(obs)
            self._log_final_lock_summary(
                obs,
                lock_ready=bool(status["lock_ready"]),
                reason=str(status["reason"]),
                stable_count=self.ctx.table_lock_frames,
                phase=phase,
                status=status,
            )
            level = str(status.get("normalized_control_level") or self._control_level(obs))
            if not self._table_micro_adjust_enabled() and str(status.get("reason") or "") == "distance_too_far":
                if not is_holding:
                    self._transition(State.EDGE_ADJUST, "final_lock_distance_too_far_return_adjust")
                    return self._table_approach_decision(obs, phase="PLANE_APPROACH")
            if bool(status.get("final_lock_window_ready")):
                return self._final_lock_arrived_decision(obs, status)
            if bool(status["lock_ready"]) or level == "stop" or bool(getattr(obs, "usable_for_stop", False)):
                self.ctx.table_stop_sent = True
                self._enter_table_dock_phase("STOP_AND_SETTLE", "[TABLE_DOCK][STOP] final lock/stop condition reached")
                self._log("info", "[TABLE_DOCK][SETTLE] begin after STOP")
                return self._annotate_final_lock_decision(self.controller.stop_cmd("FINAL_SLOW_STOP"), status)
            self._maybe_resend_req(self._active_req_payload())
            if level == "none":
                return self._annotate_final_lock_decision(self.controller.stop_cmd("FINAL_SLOW_STOP"), status)
            if level == "approach":
                return self.controller.plane_approach_cmd(obs, mode="FINAL_SLOW_STOP", reason="plane_final_approach")
            if level == "stop":
                return self.controller.fov_table_approach_cmd(obs, phase="PLANE_STOP", mode="FINAL_SLOW_STOP")
            return self.controller.fov_table_approach_cmd(obs, phase="PLANE_FINAL_LOCK", mode="FINAL_SLOW_STOP")

        if phase == "STOP_AND_SETTLE":
            settle_s = max(0.0, float(getattr(self.cfg, "table_settle_s", 0.30)))
            if monotonic_ts() - float(self.ctx.table_dock_phase_since_mono or 0.0) < settle_s:
                return self._annotate_final_lock_decision(
                    self.controller.stop_cmd("FINAL_SLOW_STOP"),
                    self._final_lock_status(obs, stable_count=self.ctx.table_lock_frames),
                )
            lock_status = self._update_final_lock_count(obs)
            self._log_final_lock_summary(
                obs,
                lock_ready=bool(lock_status["lock_ready"]),
                reason=str(lock_status["reason"]),
                stable_count=self.ctx.table_lock_frames,
                phase=phase,
                status=lock_status,
            )
            if bool(lock_status.get("final_lock_window_ready")):
                return self._final_lock_arrived_decision(obs, lock_status)
            if bool(lock_status["lock_ready"]):
                self._log(
                    "info",
                    "[TABLE_DOCK][STABLE] "
                    f"frames={self.ctx.table_lock_frames}/{self._required_lock_count()} "
                    f"dist_err={obs.dist_err_m} yaw_err={obs.yaw_err_rad}",
                )
                if self.ctx.table_lock_frames >= self._required_lock_count():
                    self.ctx.no_progress_recovery_count = 0
                    self._capture_locked_edge(obs)
                    self._log("info", "[TABLE_DOCK][DONE] stable final lock confirmed")
                    if self._table_edge_only_test_enabled():
                        self._log("info", "[TABLE_EDGE_ONLY][DONE] table edge reached; stopping before target search")
                        self._transition(State.DONE, "table_edge_only_done")
                        self._queue_tts("桌边停靠测试完成")
                        return self._annotate_final_lock_decision(self.controller.stop_cmd("DONE"), lock_status)
                    self._transition(State.AT_TABLE_EDGE, "lock_ready")
                    self._queue_tts("已完成桌边停靠")
                    return self._annotate_final_lock_decision(self.controller.stop_cmd("AT_TABLE_EDGE"), lock_status)
                return self._annotate_final_lock_decision(self.controller.stop_cmd("FINAL_SLOW_STOP"), lock_status)

            hold_reason = str(lock_status.get("lock_count_hold_reason") or "")
            if hold_reason:
                return self._annotate_final_lock_decision(self.controller.stop_cmd("FINAL_SLOW_STOP"), lock_status)

            if not self._table_micro_adjust_enabled() and str(lock_status.get("reason") or "") == "distance_too_far":
                if not is_holding:
                    self._transition(State.EDGE_ADJUST, "final_lock_distance_too_far_return_adjust")
                    return self._table_approach_decision(obs, phase="PLANE_APPROACH")

            self._enter_table_dock_phase("MICRO_ADJUST", f"[TABLE_DOCK][SETTLE] done reason={lock_status['reason']}")

        if str(self.ctx.table_dock_phase or "").upper() == "MICRO_ADJUST":
            decision = self._table_dock_micro_adjust(obs)
            if decision is not None:
                return decision

        self._maybe_resend_req(self._active_req_payload())
        return self._annotate_final_lock_decision(
            self.controller.stop_cmd("FINAL_SLOW_STOP"),
            self._final_lock_status(obs, stable_count=self.ctx.table_lock_frames),
        )

    def _final_lock_arrived_decision(self, obs: TableEdgeObs, status: Dict[str, object]) -> MotionDecision:
        status = dict(status or {})
        status.update(
            {
                "final_lock_transition_reason": "final_lock_window_ready",
                "final_lock_transition_block_reason": "",
                "ready_obs_count": int(status.get("lock_ready_obs_count", 0) or 0),
                "required_ready_obs": int(self._final_lock_required_ready_obs()),
                "window_ms": int(self._final_lock_window_ms()),
                "latest_yaw_err": getattr(obs, "yaw_err_rad", None),
                "latest_dist_err": getattr(obs, "dist_err_m", None),
                "latest_obs_age_ms": self._table_obs_age_ms(obs),
            }
        )
        self.ctx.no_progress_recovery_count = 0
        self._capture_locked_edge(obs)
        self.ctx.final_lock_last_transition_reason = "final_lock_window_ready"
        self._log(
            "info",
            "[TABLE_DOCK][DONE] final_lock_window_ready "
            f"ready_obs_count={status['ready_obs_count']}/{status['required_ready_obs']} "
            f"window_ms={status['window_ms']} "
            f"latest_yaw_err={status.get('latest_yaw_err')} "
            f"latest_dist_err={status.get('latest_dist_err')} "
            f"latest_obs_age_ms={status.get('latest_obs_age_ms')}",
        )
        if self._table_edge_only_test_enabled():
            self._log("info", "[TABLE_EDGE_ONLY][DONE] table edge reached; stopping before target search")
            self._transition(State.DONE, "final_lock_window_ready")
            self._queue_tts("桌边停靠测试完成")
            return self._annotate_final_lock_decision(self.controller.stop_cmd("DONE"), status)
        self._transition(State.AT_TABLE_EDGE, "final_lock_window_ready")
        self._queue_tts("已完成桌边停靠")
        return self._annotate_final_lock_decision(self.controller.stop_cmd("AT_TABLE_EDGE"), status)

    def _tick_at_table_edge_impl(self) -> MotionDecision:
        if self._table_edge_only_test_enabled():
            if self._state_elapsed() < float(self.cfg.edge_settle_s):
                return self.controller.stop_cmd("AT_TABLE_EDGE")
            self._log("info", "[TABLE_EDGE_ONLY][DONE] table edge reached; stopping before target search")
            self._transition(State.DONE, "table_edge_only_done")
            self._queue_tts("桌边停靠测试完成")
            return self.controller.stop_cmd("DONE")
        if self._state_elapsed() < float(self.cfg.edge_settle_s):
            return self.controller.stop_cmd("AT_TABLE_EDGE")
        self._transition(State.SEARCH_TARGET_INIT, "桌边姿态稳定，初始化沿边搜索")
        return self.controller.stop_cmd("AT_TABLE_EDGE")

    def _table_edge_only_test_enabled(self) -> bool:
        return bool(getattr(self.cfg, "table_edge_only_test", False))

    def _fresh_table_obs(self) -> Optional[TableEdgeObs]:
        obs = self.ctx.last_table_obs
        if obs is None or time.time() - obs.ts > self.cfg.table_obs_max_age_s:
            return None
        if self.ctx.task_start_wall_ts > 0 and obs.ts < self.ctx.task_start_wall_ts:
            return None
        if obs.session_id and self.ctx.active_session_id and obs.session_id != self.ctx.active_session_id:
            return None
        return obs

    @staticmethod
    def _table_obs_key(obs: Optional[TableEdgeObs]) -> str:
        if obs is None:
            return ""
        seq = getattr(obs, "seq", None)
        frame_id = getattr(obs, "frame_id", None)
        obs_ts = getattr(obs, "obs_ts", None) or getattr(obs, "ts", None)
        return f"{seq}:{frame_id}:{obs_ts}"

    @staticmethod
    def _median(values: List[float]) -> Optional[float]:
        vals = sorted(float(v) for v in values if v is not None)
        if not vals:
            return None
        mid = len(vals) // 2
        if len(vals) % 2:
            return vals[mid]
        return (vals[mid - 1] + vals[mid]) * 0.5

    def _reset_slide_ref_handoff(self) -> None:
        self.ctx.slide_ref_ready = False
        self.ctx.slide_ref_yaw_err = None
        self.ctx.slide_ref_dist_err = None
        self.ctx.slide_ref_edge_conf = None
        self.ctx.slide_ref_roi = None
        self.ctx.slide_ref_seq = None
        self.ctx.slide_ref_samples.clear()
        self.ctx.slide_ref_last_sample_key = ""
        self.ctx.handoff_state = "collecting"
        self.ctx.last_edge_quality.clear()

    def _slide_ref_sample_key(self, obs: TableEdgeObs) -> str:
        return f"{obs.source_mode or ''}:{obs.frame_id if obs.frame_id is not None else ''}:{obs.seq if obs.seq is not None else ''}:{obs.obs_ts or obs.ts:.6f}"

    def _slide_ref_obs_usable(self, obs: Optional[TableEdgeObs]) -> bool:
        if obs is None or not self._table_visible(obs):
            return False
        if str(obs.source_mode or "").strip().upper() != "FIND_OBJECT":
            return False
        if self._edge_obs_is_stale(obs):
            return False
        if not self._edge_valid_for_follow(obs):
            return False
        if obs.yaw_err_rad is None or obs.dist_err_m is None:
            return False
        conf = float(obs.confidence if obs.confidence is not None else (obs.edge_conf or 0.0))
        min_conf = float(getattr(self.cfg, "edge_follow_min_edge_conf_track_local", 0.20) or 0.20)
        return conf >= min_conf

    def _maybe_add_slide_ref_sample(self, obs: Optional[TableEdgeObs]) -> None:
        if not self._slide_ref_obs_usable(obs):
            return
        assert obs is not None
        key = self._slide_ref_sample_key(obs)
        if key == self.ctx.slide_ref_last_sample_key:
            return
        self.ctx.slide_ref_last_sample_key = key
        roi = obs.depth_edge_roi or obs.table_edge_roi or obs.edge_roi
        self.ctx.slide_ref_samples.append(
            {
                "yaw_err": float(obs.yaw_err_rad),
                "dist_err": float(obs.dist_err_m),
                "edge_conf": float(obs.confidence if obs.confidence is not None else (obs.edge_conf or 0.0)),
                "roi": list(roi) if isinstance(roi, list) else None,
                "seq": int(obs.seq) if obs.seq is not None else None,
                "frame_id": int(obs.frame_id) if obs.frame_id is not None else None,
            }
        )
        max_samples = max(5, int(getattr(self.cfg, "edge_handoff_samples", 3) or 3))
        if len(self.ctx.slide_ref_samples) > max_samples:
            self.ctx.slide_ref_samples[:] = self.ctx.slide_ref_samples[-max_samples:]
        self.ctx.handoff_state = "collecting"

    def _finalize_slide_ref(self) -> None:
        samples = list(self.ctx.slide_ref_samples)
        yaw = self._median([float(s["yaw_err"]) for s in samples if s.get("yaw_err") is not None])
        dist = self._median([float(s["dist_err"]) for s in samples if s.get("dist_err") is not None])
        conf = self._median([float(s["edge_conf"]) for s in samples if s.get("edge_conf") is not None])
        if yaw is None or dist is None or conf is None:
            return
        last = samples[-1] if samples else {}
        self.ctx.slide_ref_ready = True
        self.ctx.slide_ref_yaw_err = float(yaw)
        self.ctx.slide_ref_dist_err = float(dist)
        self.ctx.slide_ref_edge_conf = float(conf)
        self.ctx.slide_ref_roi = list(last.get("roi")) if isinstance(last.get("roi"), list) else None
        self.ctx.slide_ref_seq = int(last.get("seq")) if last.get("seq") is not None else None
        self.ctx.handoff_state = "ready"

    def _full_vs_light_yaw_offset(self) -> Optional[float]:
        if self.ctx.slide_ref_yaw_err is None or self.ctx.locked_yaw_err is None:
            return None
        return float(self.ctx.slide_ref_yaw_err) - float(self.ctx.locked_yaw_err)

    def _full_vs_light_dist_offset(self) -> Optional[float]:
        if self.ctx.slide_ref_dist_err is None or self.ctx.locked_dist_err is None:
            return None
        return float(self.ctx.slide_ref_dist_err) - float(self.ctx.locked_dist_err)

    def _handoff_trace_fields(self) -> Dict[str, Any]:
        return {
            "handoff_state": self.ctx.handoff_state,
            "handoff_samples_count": len(self.ctx.slide_ref_samples),
            "handoff_valid_samples_count": len(self.ctx.slide_ref_samples),
            "slide_ref_ready": bool(self.ctx.slide_ref_ready),
            "slide_ref_yaw_err": self.ctx.slide_ref_yaw_err,
            "slide_ref_dist_err": self.ctx.slide_ref_dist_err,
            "slide_ref_edge_conf": self.ctx.slide_ref_edge_conf,
            "slide_ref_roi": self.ctx.slide_ref_roi,
            "slide_ref_seq": self.ctx.slide_ref_seq,
            "full_locked_yaw_err": self.ctx.locked_yaw_err,
            "full_locked_dist_err": self.ctx.locked_dist_err,
            "full_vs_light_yaw_offset": self._full_vs_light_yaw_offset(),
            "full_vs_light_dist_offset": self._full_vs_light_dist_offset(),
        }

    def _capture_locked_edge(self, obs: Optional[TableEdgeObs]) -> None:
        if obs is None:
            return
        self.ctx.locked_edge_id = str(self.ctx.current_edge_id or "")
        line = {}
        if obs.edge_k is not None:
            line["edge_k"] = float(obs.edge_k)
        if obs.edge_b is not None:
            line["edge_b"] = float(obs.edge_b)
        self.ctx.locked_edge_line = line or None
        roi = obs.depth_edge_roi or obs.table_edge_roi or obs.edge_roi
        self.ctx.locked_roi = list(roi) if isinstance(roi, list) else None
        self.ctx.locked_yaw_err = float(obs.yaw_err_rad) if obs.yaw_err_rad is not None else None
        self.ctx.locked_dist_err = float(obs.dist_err_m) if obs.dist_err_m is not None else None
        self.ctx.locked_edge_conf = float(obs.confidence or 0.0)
        self.ctx.locked_obs_seq = int(obs.seq) if obs.seq is not None else None

    def _edge_valid_for_follow(self, obs: Optional[TableEdgeObs]) -> bool:
        if obs is None:
            return False
        level = self._control_level(obs)
        if level in {"alignment", "stop"}:
            return True
        if level == "approach":
            return False
        edge_valid = getattr(obs, "edge_valid", None)
        if edge_valid is not None:
            return bool(edge_valid)
        return bool(obs.edge_found)

    def _edge_follow_quality(self, obs: TableEdgeObs) -> Dict[str, Any]:
        source_mode = str(obs.source_mode or self.ctx.confirmed_vision_mode or "").strip().upper()
        is_track_local = source_mode == "FIND_OBJECT" or str(self.ctx.confirmed_vision_mode or "").strip().upper() == "FIND_OBJECT"
        min_conf = float(
            getattr(
                self.cfg,
                "edge_follow_min_edge_conf_track_local" if is_track_local else "edge_follow_min_edge_conf_table_edge_perception",
                getattr(self.cfg, "edge_follow_min_edge_conf", 0.60),
            )
            or 0.0
        )
        if is_track_local:
            weak_conf = float(getattr(self.cfg, "edge_follow_weak_edge_conf_track_local", min_conf) or min_conf)
            strong_conf = float(getattr(self.cfg, "edge_follow_strong_edge_conf_track_local", min_conf) or min_conf)
        else:
            weak_conf = min_conf
            strong_conf = min_conf
        conf = float(obs.confidence or 0.0)
        yaw = float(obs.yaw_err_rad) if obs.yaw_err_rad is not None else None
        dist = float(obs.dist_err_m) if obs.dist_err_m is not None else None
        locked_yaw = self.ctx.locked_yaw_err
        locked_dist = self.ctx.locked_dist_err
        slide_ref_yaw = self.ctx.slide_ref_yaw_err
        slide_ref_dist = self.ctx.slide_ref_dist_err
        yaw_delta = None if yaw is None or locked_yaw is None else float(yaw - float(locked_yaw))
        dist_delta = None if dist is None or locked_dist is None else float(dist - float(locked_dist))
        yaw_delta_from_slide_ref = None if yaw is None or slide_ref_yaw is None else float(yaw - float(slide_ref_yaw))
        dist_delta_from_slide_ref = None if dist is None or slide_ref_dist is None else float(dist - float(slide_ref_dist))
        identity_basis = "slide_ref" if is_track_local and self.ctx.slide_ref_ready else "full_locked_edge"
        basis_yaw_delta = yaw_delta_from_slide_ref if identity_basis == "slide_ref" else yaw_delta
        basis_dist_delta = dist_delta_from_slide_ref if identity_basis == "slide_ref" else dist_delta
        yaw_mismatch = basis_yaw_delta is not None and abs(basis_yaw_delta) > float(getattr(self.cfg, "edge_identity_yaw_mismatch_rad", 0.15))
        dist_mismatch = basis_dist_delta is not None and abs(basis_dist_delta) > float(getattr(self.cfg, "edge_identity_dist_mismatch_m", 0.04))
        identity_ok = not (yaw_mismatch or dist_mismatch)
        if not identity_ok:
            mode = "identity_mismatch"
            reason = "edge_identity_mismatch"
        elif conf >= strong_conf:
            mode = "strong"
            reason = "edge_slide"
        elif conf >= weak_conf:
            mode = "weak"
            reason = "weak_edge_slide"
        else:
            mode = "pause"
            reason = "edge_conf_low"
        return {
            "mode": mode,
            "reason": reason,
            "edge_conf_threshold_used": min_conf,
            "weak_conf": weak_conf,
            "strong_conf": strong_conf,
            "locked_edge_conf": self.ctx.locked_edge_conf,
            "locked_yaw_err": locked_yaw,
            "locked_dist_err": locked_dist,
            "yaw_delta_from_locked": yaw_delta,
            "dist_delta_from_locked": dist_delta,
            "slide_ref_ready": bool(self.ctx.slide_ref_ready),
            "slide_ref_yaw_err": slide_ref_yaw,
            "slide_ref_dist_err": slide_ref_dist,
            "slide_ref_edge_conf": self.ctx.slide_ref_edge_conf,
            "yaw_delta_from_slide_ref": yaw_delta_from_slide_ref,
            "dist_delta_from_slide_ref": dist_delta_from_slide_ref,
            "edge_identity_basis": identity_basis,
            "full_locked_yaw_err": locked_yaw,
            "full_locked_dist_err": locked_dist,
            "full_vs_light_yaw_offset": self._full_vs_light_yaw_offset(),
            "full_vs_light_dist_offset": self._full_vs_light_dist_offset(),
            "handoff_state": self.ctx.handoff_state,
            "handoff_samples_count": len(self.ctx.slide_ref_samples),
            "handoff_valid_samples_count": len(self.ctx.slide_ref_samples),
            "edge_identity_ok": identity_ok,
            "slide_vy_mps": float(getattr(self.controller.car_cfg, "edge_slide_vy_mps", 0.14) or 0.14),
            "weak_slide_vy": float(getattr(self.controller.car_cfg, "edge_slide_weak_vy_mps", 0.05) or 0.05),
            "weak_slide_vy_mps": float(getattr(self.controller.car_cfg, "edge_slide_weak_vy_mps", 0.05) or 0.05),
            "fallback_candidate_state": self._edge_slide_fallback_state().value,
            "fallback_suppressed_reason": "fresh_geometry_stable" if mode in {"weak", "strong"} else "",
        }

    @staticmethod
    def _raw_control_level(obs: Optional[TableEdgeObs]) -> str:
        if obs is None:
            return "none"
        level = getattr(obs, "control_level", None)
        if level:
            level_str = str(level).strip().lower()
            if "stop" in level_str:
                return "stop"
            if "approach" in level_str:
                return "approach"
            if "align" in level_str or "rotate" in level_str:
                return "alignment"
            return "none"
        if getattr(obs, "usable_for_stop", False):
            return "stop"
        if getattr(obs, "usable_for_approach", False):
            return "approach"
        if getattr(obs, "usable_for_alignment", False):
            return "alignment"
        return "none"

    @classmethod
    def _control_level(cls, obs: Optional[TableEdgeObs]) -> str:
        level = cls._raw_control_level(obs)
        aliases = {
            "approach_slow": "approach",
            "approach": "approach",
            "rotate_only": "alignment",
            "align": "alignment",
            "alignment": "alignment",
            "stop_ready": "stop",
            "stop": "stop",
            "none": "none",
            "": "none",
        }
        return aliases.get(level, "none")

    def _table_yolo_reliable(self, obs: Optional[TableEdgeObs]) -> bool:
        return bool(self._yolo_table_status(obs)["fresh"])

    def _yolo_table_status(self, obs: Optional[TableEdgeObs]) -> Dict[str, object]:
        if obs is None:
            return {"visible": False, "fresh": False, "age_ms": None}
        geom = self._bbox_control_geometry(obs)
        has_bbox = geom["bbox_xyxy_for_control"] is not None
        has_center = geom["bbox_center_valid"]
        yolo_control = bool(getattr(obs, "yolo_table_control_valid", False) or getattr(obs, "table_bbox_control_valid", False))
        # Legacy table_bbox_found/control_valid may be held/fallback state and
        # must not grant a new forward approach without a current detection.
        visible = bool(
            getattr(obs, "table_bbox_current_found", False)
            or (bool(getattr(obs, "yolo_table_visible", False)) and yolo_control and (has_bbox or has_center))
        )
        age_ms = getattr(obs, "yolo_table_age_ms", None)
        if age_ms is None:
            age_ms = self._table_obs_age_ms(obs)
        explicit_fresh = getattr(obs, "yolo_table_fresh", None)
        max_age_ms = max(
            float(getattr(self.cfg, "table_obs_stale_stop_ms", 500) or 500),
            float(getattr(self.cfg, "no_table_bbox_timeout_s", 1.0) or 1.0) * 1000.0,
        )
        try:
            age_ok = age_ms is not None and float(age_ms) <= max_age_ms
        except Exception:
            age_ok = False
        if explicit_fresh is None:
            fresh = bool(visible and age_ok and not bool(getattr(obs, "is_stale", False)))
        else:
            fresh = bool(visible and bool(explicit_fresh) and age_ok)
        return {"visible": bool(visible), "fresh": bool(fresh), "age_ms": age_ms}

    def _table_plane_stable(self, obs: Optional[TableEdgeObs]) -> bool:
        # Geometry can be stable from the detector perspective, but it is only
        # allowed to affect table approach state transitions when it is trusted
        # under the new perception semantics.
        return bool(self.controller._edge_trusted(obs))

    def _table_approach_phase(self, obs: Optional[TableEdgeObs]) -> str:
        level = self._control_level(obs)
        if level == "stop":
            return "PLANE_STOP"
        if level == "alignment":
            return "PLANE_FINAL_LOCK"
        if level == "approach":
            return "PLANE_APPROACH"
        if self._table_plane_stable(obs):
            return "PLANE_FINAL_LOCK"
        return "PLANE_ACQUIRE"

    def _align_to_approach_yaw_rad(self) -> float:
        return max(0.0, float(getattr(self.cfg, "align_to_approach_yaw_rad", 0.08) or 0.08))

    def _approach_to_align_yaw_rad(self) -> float:
        align_yaw = self._align_to_approach_yaw_rad()
        return max(align_yaw + 1e-6, float(getattr(self.cfg, "approach_to_align_yaw_rad", 0.16) or 0.16))

    def _align_to_approach_stable_obs(self) -> int:
        return max(1, int(getattr(self.cfg, "align_to_approach_stable_obs", 2) or 2))

    def _approach_to_align_stable_obs(self) -> int:
        return max(1, int(getattr(self.cfg, "approach_to_align_stable_obs", 2) or 2))

    def _coarse_align_min_dwell_s(self) -> float:
        return max(0.0, float(getattr(self.cfg, "coarse_align_min_dwell_s", 0.8) or 0.8))

    def _controlled_approach_min_dwell_s(self) -> float:
        return max(0.0, float(getattr(self.cfg, "controlled_approach_min_dwell_s", 0.8) or 0.8))

    def _yaw_abs(self, obs: Optional[TableEdgeObs]) -> Optional[float]:
        if obs is None or obs.yaw_err_rad is None:
            return None
        try:
            return abs(float(obs.yaw_err_rad))
        except Exception:
            return None

    def _yaw_ready_for_controlled_approach(self, obs: Optional[TableEdgeObs]) -> bool:
        yaw_abs = self._yaw_abs(obs)
        return bool(yaw_abs is not None and yaw_abs <= self._align_to_approach_yaw_rad())

    def _yaw_needs_realign_from_approach(self, obs: Optional[TableEdgeObs]) -> bool:
        yaw_abs = self._yaw_abs(obs)
        return bool(yaw_abs is not None and yaw_abs >= self._approach_to_align_yaw_rad())

    def _count_table_motion_hysteresis_obs(self, obs: Optional[TableEdgeObs], *, ok: bool, last_key_attr: str, count_attr: str) -> int:
        obs_key = self._final_lock_obs_key(obs)
        if not ok:
            setattr(self.ctx, count_attr, 0)
            if obs_key:
                setattr(self.ctx, last_key_attr, obs_key)
            return 0
        if obs_key and obs_key != str(getattr(self.ctx, last_key_attr, "") or ""):
            setattr(self.ctx, last_key_attr, obs_key)
            setattr(self.ctx, count_attr, int(getattr(self.ctx, count_attr, 0) or 0) + 1)
        return int(getattr(self.ctx, count_attr, 0) or 0)

    def _annotate_table_motion_hysteresis(self, decision: MotionDecision, *, pending_reason: str = "") -> MotionDecision:
        if decision.control_summary is None:
            decision.control_summary = {}
        summary = decision.control_summary
        summary.update(
            {
                "align_to_approach_yaw_rad": float(self._align_to_approach_yaw_rad()),
                "approach_to_align_yaw_rad": float(self._approach_to_align_yaw_rad()),
                "align_to_approach_stable_obs": int(self._align_to_approach_stable_obs()),
                "approach_to_align_stable_obs": int(self._approach_to_align_stable_obs()),
                "coarse_align_min_dwell_s": float(self._coarse_align_min_dwell_s()),
                "controlled_approach_min_dwell_s": float(self._controlled_approach_min_dwell_s()),
                "state_dwell_s": float(self._state_elapsed()),
                "approach_aligned_frames": int(self.ctx.approach_aligned_frames),
                "approach_realign_frames": int(self.ctx.approach_realign_frames),
                "table_motion_pending_transition_reason": str(pending_reason or self.ctx.table_motion_pending_transition_reason or ""),
            }
        )
        return decision

    def _coarse_aligned(self, obs: Optional[TableEdgeObs]) -> bool:
        if obs is None:
            return False
        if self._edge_obs_is_stale(obs):
            return False
        if self._control_level(obs) == "stop":
            return True
        if obs.yaw_err_rad is not None:
            return abs(float(obs.yaw_err_rad)) <= self._align_to_approach_yaw_rad()
        geom = self._bbox_control_geometry(obs)
        if geom["bbox_center_valid"]:
            return abs(float(geom["bbox_center_error_control"])) <= 0.06
        return False

    def _edge_ready(self, obs: Optional[TableEdgeObs]) -> bool:
        if obs is None:
            return False
        if self._edge_obs_is_stale(obs):
            return False
        if self._control_level(obs) == "stop" or bool(getattr(obs, "usable_for_stop", False)):
            return True
        if obs.edge_ready is not None:
            return bool(obs.edge_ready)
        if obs.dist_err_m is not None:
            return abs(float(obs.dist_err_m)) <= float(self.cfg.final_lock_dist_tol_m) * 2.0
        return bool(obs.edge_found)

    def _depth_roi_stop_status(self, obs: Optional[TableEdgeObs]) -> Dict[str, object]:
        status = {
            "depth_roi_stop_ready": False,
            "reason": "",
            "roi_depth_stat": None,
            "roi_depth_valid_ratio": None,
            "roi_depth_sample_count": None,
            "roi_depth_threshold": None,
            "roi_depth_threshold_semantics": "table_roi_depth_p10_not_geometric_edge_distance",
            "target_dist": None,
            "margin": None,
        }
        if obs is None:
            status["reason"] = "no_recent_obs"
            return status
        if not self._table_yolo_reliable(obs):
            status["reason"] = "no_current_table_bbox"
            return status
        
        # Check table visibility
        if not self._table_visible(obs):
            status["reason"] = "table_not_visible"
            return status
            
        depth_valid = bool(getattr(obs, "table_roi_depth_valid", False))
        depth_p10 = getattr(obs, "table_roi_depth_p10", None)
        ratio = getattr(obs, "table_roi_depth_valid_ratio", None)
        samples = getattr(obs, "table_roi_depth_sample_count", None)
        if not depth_valid or depth_p10 is None:
            status["reason"] = "depth_roi_invalid"
            return status
            
        target_dist = self._table_target_dist_m(obs)
        margin = max(0.0, float(getattr(self.cfg, "table_stop_margin_m", 0.05)))
        status["roi_depth_stat"] = float(depth_p10)
        status["roi_depth_valid_ratio"] = ratio
        status["roi_depth_sample_count"] = samples
        status["target_dist"] = float(target_dist)
        status["margin"] = float(margin)
        status["roi_depth_threshold"] = float(target_dist + margin)
        
        if float(depth_p10) > target_dist + margin:
            status["reason"] = "distance_too_far"
            return status
            
        status["depth_roi_stop_ready"] = True
        status["reason"] = "allowed"
        return status

    def _roi_depth_stop_ready(self, obs: Optional[TableEdgeObs]) -> bool:
        return bool(self._depth_roi_stop_status(obs)["depth_roi_stop_ready"])

    def _final_lock_enter_status(self, obs: Optional[TableEdgeObs]) -> Dict[str, object]:
        level = self._control_level(obs)
        yaw_th = abs(float(getattr(self.cfg, "final_lock_enter_yaw_th_rad", 0.10) or 0.10))
        dist_th = abs(float(getattr(self.cfg, "final_lock_enter_dist_th_m", 0.08) or 0.08))
        min_conf = float(getattr(self.controller.docking.cfg, "min_confidence", 0.0) or 0.0)
        status: Dict[str, object] = {
            "final_lock_enabled": self._table_final_lock_enabled(),
            "micro_adjust_enabled": self._table_micro_adjust_enabled(),
            "normalized_control_level": level,
            "yaw_err": getattr(obs, "yaw_err_rad", None) if obs is not None else None,
            "dist_err_m": getattr(obs, "dist_err_m", None) if obs is not None else None,
            "final_lock_enter_allowed": False,
            "final_lock_enter_block_reason": "",
            "final_lock_enter_dist_th_m": float(dist_th),
            "final_lock_enter_yaw_th_rad": float(yaw_th),
        }
        if not self._table_final_lock_enabled():
            status["final_lock_enter_block_reason"] = "final_lock_disabled"
            return status
        if obs is None:
            status["final_lock_enter_block_reason"] = "no_recent_obs"
            return status
        stale_level = self._table_obs_stale_level(obs)
        if stale_level in {"hard_stale", "dead"}:
            status["final_lock_enter_block_reason"] = "vision_stale"
            return status
        if bool(getattr(obs, "is_stale", False)):
            status["final_lock_enter_block_reason"] = "obs_invalid"
            return status
            
        depth_status = self._depth_roi_stop_status(obs)
        depth_stop_ready = bool(depth_status.get("depth_roi_stop_ready"))
        lock_status = self._update_final_lock_count(obs) if depth_stop_ready else {}
        depth_window_ready = bool(lock_status.get("final_lock_window_ready", False))
        status.update(depth_status)
        status.update({
            "roi_depth_consecutive_frames": lock_status.get("lock_ready_obs_count", 0),
            "roi_depth_window_ready": depth_window_ready,
        })
        
        # If depth stop is ready and table is visible, we bypass edge trust checks
        if depth_stop_ready and depth_window_ready:
            status["final_lock_enter_allowed"] = True
            status["final_lock_enter_block_reason"] = "allowed"
            return status
        if depth_stop_ready:
            status["final_lock_enter_block_reason"] = "roi_depth_stability_window_pending"
            return status
            
        if getattr(obs, "depth_valid", True) is False:
            status["final_lock_enter_block_reason"] = "obs_invalid"
            return status
        if not self._table_visible(obs) or not bool(getattr(obs, "edge_found", False)):
            status["final_lock_enter_block_reason"] = "edge_invalid"
            return status
        if not bool(getattr(obs, "edge_trusted", False) or getattr(obs, "usable_for_stop", False)):
            status["final_lock_enter_block_reason"] = "not_valid_for_control"
            return status
        if level != "stop" and not bool(getattr(obs, "usable_for_stop", False)):
            status["final_lock_enter_block_reason"] = "not_stop_ready"
            return status
        confidence = float(getattr(obs, "confidence", 0.0) or 0.0)
        if confidence < min_conf:
            status["final_lock_enter_block_reason"] = "low_confidence"
            return status
        if obs.yaw_err_rad is None:
            status["final_lock_enter_block_reason"] = "yaw_missing"
            return status
        if abs(float(obs.yaw_err_rad)) > yaw_th:
            status["final_lock_enter_block_reason"] = "yaw_out_of_range"
            return status
        if obs.dist_err_m is None:
            status["final_lock_enter_block_reason"] = "dist_missing"
            return status
        if abs(float(obs.dist_err_m)) > dist_th:
            status["final_lock_enter_block_reason"] = "distance_too_far" if float(obs.dist_err_m) > 0.0 else "distance_too_close"
            return status
        status["final_lock_enter_allowed"] = True
        status["final_lock_enter_block_reason"] = "allowed"
        return status

    def _table_target_dist_m(self, obs: Optional[TableEdgeObs] = None) -> float:
        target = getattr(self.cfg, "table_target_dist_m", 0.015)
        if obs is not None and obs.target_dist_m is not None:
            target = obs.target_dist_m
        try:
            return max(0.0, float(target))
        except Exception:
            return 0.015

    def _table_measured_dist_m(self, obs: Optional[TableEdgeObs]) -> Optional[float]:
        if obs is None or obs.dist_err_m is None:
            return None
        return self._table_target_dist_m(obs) + float(obs.dist_err_m)

    def _table_dock_should_stop(self, obs: Optional[TableEdgeObs]) -> bool:
        if obs is None or self._edge_obs_is_stale(obs):
            return False
        if self._control_level(obs) != "stop" and not bool(getattr(obs, "usable_for_stop", False)):
            return False
        measured = self._table_measured_dist_m(obs)
        if measured is None:
            return False
        target = self._table_target_dist_m(obs)
        margin = max(0.0, float(getattr(self.cfg, "table_stop_margin_m", 0.05)))
        return float(measured) <= target + margin

    def _table_final_lock_enabled(self) -> bool:
        return bool(getattr(self.cfg, "enable_final_lock", False))

    def _table_micro_adjust_enabled(self) -> bool:
        return bool(getattr(self.cfg, "enable_micro_adjust", False))

    def _table_approach_decision(
        self,
        obs: Optional[TableEdgeObs],
        *,
        phase: str = "PLANE_APPROACH",
        mode: str = "EDGE_ADJUST",
        stop_ready_ignored: bool = False,
    ) -> MotionDecision:
        decision = self.controller.fov_table_approach_cmd(obs, phase=phase, mode=mode)
        if decision.control_summary is not None:
            enter_status = self._final_lock_enter_status(obs)
            decision.control_summary.update(
                {
                    **enter_status,
                    "stop_ready_ignored_for_stage_transition": bool(stop_ready_ignored),
                }
            )
        return decision

    def _enter_final_slow_stop_or_keep_approach(self, obs: Optional[TableEdgeObs], reason: str) -> MotionDecision:
        enter_status = self._final_lock_enter_status(obs)
        if bool(enter_status.get("final_lock_enter_allowed")):
            self._transition(State.FINAL_SLOW_STOP, reason)
            decision = self.controller.fov_table_approach_cmd(obs, phase="PLANE_FINAL_LOCK", mode="FINAL_SLOW_STOP")
            if decision.control_summary is not None:
                decision.control_summary.update(enter_status)
            return decision
        block_reason = str(enter_status.get("final_lock_enter_block_reason") or "blocked")
        decision = self._table_approach_decision(obs, phase="PLANE_APPROACH", stop_ready_ignored=True)
        if self.ctx.state != State.EDGE_ADJUST:
            self._transition(State.EDGE_ADJUST, f"final_slow_stop_enter_blocked:{block_reason}")
        return decision

    def _annotate_final_lock_decision(self, decision: MotionDecision, status: Dict[str, object]) -> MotionDecision:
        if decision.control_summary is not None:
            decision.control_summary.update(
                {
                    "final_lock_enabled": self._table_final_lock_enabled(),
                    "micro_adjust_enabled": self._table_micro_adjust_enabled(),
                    "normalized_control_level": status.get("normalized_control_level"),
                    "yaw_err": status.get("yaw_err"),
                    "dist_err_m": status.get("dist_err"),
                    "stable_lock_count": status.get("stable_lock_count"),
                    "required_lock_count": status.get("required_lock_count"),
                    "lock_ready_obs_count": status.get("lock_ready_obs_count"),
                    "window_ready_count": status.get("window_ready_count"),
                    "required_ready_obs": status.get("required_ready_obs"),
                    "final_lock_window_ms": status.get("final_lock_window_ms"),
                    "same_obs_reuse_count": status.get("same_obs_reuse_count"),
                    "consecutive_lost_count": status.get("consecutive_lost_count"),
                    "lock_count_inc_reason": status.get("lock_count_inc_reason"),
                    "lock_count_hold_reason": status.get("lock_count_hold_reason"),
                    "lock_count_reset_reason": status.get("lock_count_reset_reason"),
                    "final_lock_transition_block_reason": status.get("final_lock_transition_block_reason"),
                    "final_lock_transition_reason": status.get("final_lock_transition_reason"),
                }
            )
        return decision

    def _enter_table_dock_phase(self, phase: str, log_line: str = "") -> None:
        phase = str(phase or "").upper()
        if self.ctx.table_dock_phase != phase:
            self.ctx.table_dock_phase = phase
            self.ctx.table_dock_phase_since_mono = monotonic_ts()
        if log_line:
            self._log("info", log_line)

    def _table_dock_micro_adjust(self, obs: Optional[TableEdgeObs]) -> Optional[MotionDecision]:
        if not self._table_micro_adjust_enabled():
            self._log("info", "[TABLE_DOCK][MICRO_ADJUST] skipped enable_micro_adjust=false")
            decision = self._table_approach_decision(obs, phase="PLANE_APPROACH", stop_ready_ignored=True)
            if decision.control_summary is not None:
                decision.control_summary.update({"micro_adjust_skipped": True, "table_dock_phase": "MICRO_ADJUST_DISABLED"})
            return decision
        max_adjust = max(0, int(getattr(self.cfg, "table_max_micro_adjust", 4)))
        if self.ctx.table_micro_adjust_count >= max_adjust:
            reason = f"[TABLE_DOCK][FAIL] max_micro_adjust={max_adjust}"
            self.ctx.last_fail_reason = reason
            self._log("error", reason)
            self._enter_error_recovery("最终锁边微调次数超限", tts_text="桌边停靠失败，请检查", interrupt_tts=True)
            return self.controller.stop_cmd("ERROR_RECOVERY", brake=True)

        status = self._final_lock_status(obs, stable_count=self.ctx.table_lock_frames)
        action = ""
        tag = ""
        dist_delta = None
        if obs is not None and obs.dist_err_m is not None:
            measured = self._table_measured_dist_m(obs)
            target = self._table_target_dist_m(obs)
            dist_delta = (float(measured) - target) if measured is not None else float(obs.dist_err_m)
        dist_tol = float(getattr(self.cfg, "table_dist_tol_m", self.cfg.final_lock_dist_tol_m))
        yaw_tol = float(getattr(self.cfg, "table_yaw_tol_rad", self.cfg.final_lock_yaw_tol_rad))

        if dist_delta is not None and dist_delta > dist_tol:
            action = "forward"
            tag = "[TABLE_DOCK][JOG_FORWARD]"
        elif dist_delta is not None and dist_delta < -dist_tol:
            action = "backward"
            tag = "[TABLE_DOCK][JOG_BACKWARD]"
        elif obs is not None and obs.yaw_err_rad is not None and abs(float(obs.yaw_err_rad)) > yaw_tol:
            action = "turn_left" if float(obs.yaw_err_rad) > 0.0 else "turn_right"
            tag = "[TABLE_DOCK][JOG_TURN]"
        else:
            action = "forward" if str(status["reason"]) == "distance_too_far" else ""
            tag = "[TABLE_DOCK][JOG_FORWARD]" if action else ""

        if not action:
            self._log("info", f"[TABLE_DOCK][SETTLE] no jog action reason={status['reason']}")
            self._enter_table_dock_phase("STOP_AND_SETTLE")
            return self.controller.stop_cmd("FINAL_SLOW_STOP")

        self.ctx.table_micro_adjust_count += 1
        reason = (
            f"{tag} action={action} count={self.ctx.table_micro_adjust_count}/{max_adjust} "
            f"reason={status['reason']} dist_delta={dist_delta} yaw_err={obs.yaw_err_rad if obs is not None else None}"
        )
        self._log("info", reason)
        self.ctx.table_lock_frames = 0
        self._enter_table_dock_phase("STOP_AND_SETTLE")
        self._log("info", "[TABLE_DOCK][SETTLE] begin after JOG")
        decision = self.controller.stop_cmd("FINAL_SLOW_STOP")
        decision.jog_action = action
        decision.jog_reason = reason
        if decision.control_summary is not None:
            decision.control_summary.update({"table_dock_phase": "MICRO_ADJUST", "jog_action": action, "reason": reason})
        return decision

    def _final_lock_ready(self, obs: Optional[TableEdgeObs]) -> bool:
        return bool(self._final_lock_status(obs, stable_count=self.ctx.table_lock_frames)["lock_ready"])

    def _required_lock_count(self) -> int:
        return max(1, int(getattr(self.cfg, "table_stable_frames", self.cfg.final_lock_frames_to_arrive)))

    def _final_lock_required_ready_obs(self) -> int:
        return max(1, int(getattr(self.cfg, "final_lock_required_ready_obs", 3) or 3))

    def _final_lock_window_ms(self) -> int:
        return max(100, int(getattr(self.cfg, "final_lock_window_ms", 1000) or 1000))

    def _final_lock_max_consecutive_lost(self) -> int:
        return max(0, int(getattr(self.cfg, "final_lock_max_consecutive_lost", 2) or 2))

    def _final_lock_obs_key(self, obs: Optional[TableEdgeObs]) -> str:
        if obs is None:
            return ""
        obs_seq = getattr(obs, "obs_seq", None)
        if obs_seq is not None:
            return f"obs_seq:{obs_seq}"
        return self._table_obs_key(obs)

    def _prune_final_lock_window(self, now_mono: Optional[float] = None) -> None:
        now = monotonic_ts() if now_mono is None else float(now_mono)
        window_s = float(self._final_lock_window_ms()) / 1000.0
        self.ctx.final_lock_ready_window = [
            item
            for item in list(self.ctx.final_lock_ready_window or [])
            if now - float(item.get("mono_ts", now) or now) <= window_s
        ]
        self.ctx.table_lock_frames = len(self.ctx.final_lock_ready_window)

    def _reset_final_lock_window(self, reason: str = "") -> None:
        self.ctx.final_lock_ready_window.clear()
        self.ctx.table_lock_frames = 0
        self.ctx.final_lock_consecutive_lost_count = 0
        if reason:
            self.ctx.final_lock_last_transition_reason = str(reason)

    def _final_lock_ready_jump_reason(self, obs: Optional[TableEdgeObs]) -> str:
        if obs is None or not self.ctx.final_lock_ready_window:
            return ""
        last = self.ctx.final_lock_ready_window[-1]
        yaw = getattr(obs, "yaw_err_rad", None)
        dist = getattr(obs, "dist_err_m", None)
        last_yaw = last.get("yaw_err")
        last_dist = last.get("dist_err")
        try:
            yaw_jump_th = max(0.35, float(getattr(self.cfg, "edge_identity_yaw_mismatch_rad", 0.15) or 0.15) * 2.0)
            if yaw is not None and last_yaw is not None and abs(float(yaw) - float(last_yaw)) > yaw_jump_th:
                return "yaw_jump"
        except Exception:
            pass
        try:
            dist_jump_th = max(0.20, float(getattr(self.cfg, "edge_identity_dist_mismatch_m", 0.04) or 0.04) * 3.0)
            if dist is not None and last_dist is not None and abs(float(dist) - float(last_dist)) > dist_jump_th:
                return "dist_jump"
        except Exception:
            pass
        return ""

    def _update_final_lock_count(self, obs: Optional[TableEdgeObs]) -> Dict[str, object]:
        status = self._final_lock_status(obs, stable_count=self.ctx.table_lock_frames)
        now = monotonic_ts()
        self._prune_final_lock_window(now)
        obs_key = self._final_lock_obs_key(obs)
        new_obs = bool(obs_key and obs_key != self.ctx.final_lock_last_obs_key)
        if obs_key:
            if new_obs:
                self.ctx.final_lock_same_obs_reuse_count = 0
                self.ctx.final_lock_last_obs_key = obs_key
            else:
                self.ctx.final_lock_same_obs_reuse_count += 1

        reason = str(status.get("reason") or "")
        stale_reason = str(status.get("vision_stale_reason") or "")
        reset_reason = str(status.get("lock_count_reset_reason") or stale_reason or reason or "lock_ready_false")
        lost_like = reason in {"table_lost", "no_edge", "edge_invalid"} or reset_reason in {"table_lost", "no_edge", "edge_invalid"}
        hard_reset = reset_reason in {"hard_stale", "obs_invalid", "temporal_jump", "no_recent_obs", "yaw_jump", "dist_jump", "age_over_limit"}
        hold_reason = str(status.get("lock_count_hold_reason") or "")
        if hold_reason == "soft_stale" and not bool(getattr(self.cfg, "final_lock_soft_stale_hold", True)):
            hold_reason = ""

        status["obs_seq"] = getattr(obs, "obs_seq", None) if obs is not None else None
        status["same_obs_reuse_count"] = int(self.ctx.final_lock_same_obs_reuse_count)
        status["required_ready_obs"] = int(self._final_lock_required_ready_obs())
        status["final_lock_window_ms"] = int(self._final_lock_window_ms())
        status["required_lock_count"] = int(self._final_lock_required_ready_obs())
        status["legacy_required_lock_count"] = int(self._required_lock_count())

        if bool(status["lock_ready"]):
            jump_reason = self._final_lock_ready_jump_reason(obs)
            if jump_reason:
                self._reset_final_lock_window(jump_reason)
                status["lock_ready"] = False
                status["lock_count_inc_reason"] = ""
                status["lock_count_hold_reason"] = ""
                status["lock_count_reset_reason"] = jump_reason
                status["lock_reset_reason"] = jump_reason
            elif new_obs:
                self.ctx.final_lock_consecutive_lost_count = 0
                self.ctx.final_lock_ready_window.append(
                    {
                        "key": obs_key,
                        "mono_ts": now,
                        "obs_seq": getattr(obs, "obs_seq", None) if obs is not None else None,
                        "yaw_err": getattr(obs, "yaw_err_rad", None) if obs is not None else None,
                        "dist_err": getattr(obs, "dist_err_m", None) if obs is not None else None,
                        "obs_age_ms": status.get("obs_age_ms"),
                    }
                )
                self._prune_final_lock_window(now)
                status["lock_count_inc_reason"] = "fresh_lock_ready"
                status["lock_count_hold_reason"] = ""
                status["lock_count_reset_reason"] = ""
                status["lock_reset_reason"] = ""
            else:
                status["lock_count_inc_reason"] = ""
                status["lock_count_hold_reason"] = "same_obs_reuse"
                status["lock_count_reset_reason"] = ""
                status["lock_reset_reason"] = ""
        elif hold_reason:
            status["lock_count_inc_reason"] = ""
            status["lock_count_hold_reason"] = hold_reason
            status["lock_count_reset_reason"] = ""
            status["lock_reset_reason"] = ""
        elif lost_like and not hard_reset:
            if new_obs:
                self.ctx.final_lock_consecutive_lost_count += 1
            max_lost = self._final_lock_max_consecutive_lost()
            if self.ctx.final_lock_consecutive_lost_count <= max_lost:
                status["lock_count_inc_reason"] = ""
                status["lock_count_hold_reason"] = f"{reason or reset_reason}_lost_hold"
                status["lock_count_reset_reason"] = ""
                status["lock_reset_reason"] = ""
            else:
                reset_reason = f"{reason or reset_reason}_lost_exceeded"
                self._reset_final_lock_window(reset_reason)
                status["lock_count_inc_reason"] = ""
                status["lock_count_hold_reason"] = ""
                status["lock_count_reset_reason"] = reset_reason
                status["lock_reset_reason"] = reset_reason
        else:
            self._reset_final_lock_window(reset_reason)
            status["lock_count_inc_reason"] = ""
            status["lock_count_hold_reason"] = ""
            status["lock_count_reset_reason"] = reset_reason
            status["lock_reset_reason"] = reset_reason

        ready_count = len(self.ctx.final_lock_ready_window)
        window_ready = ready_count >= self._final_lock_required_ready_obs()
        status["lock_ready_obs_count"] = int(ready_count)
        status["window_ready_count"] = int(ready_count)
        status["stable_count"] = int(ready_count)
        status["stable_lock_count"] = int(ready_count)
        status["consecutive_lost_count"] = int(self.ctx.final_lock_consecutive_lost_count)
        status["final_lock_window_ready"] = bool(window_ready)
        status["final_lock_transition_reason"] = "final_lock_window_ready" if window_ready else ""
        status["final_lock_transition_block_reason"] = "" if window_ready else str(
            status.get("lock_count_reset_reason")
            or status.get("lock_count_hold_reason")
            or reason
            or "ready_obs_count_not_enough"
        )
        return status

    def _final_lock_status(self, obs: Optional[TableEdgeObs], stable_count: int = 0) -> Dict[str, object]:
        required_count = self._final_lock_required_ready_obs()

        def _status(
            *,
            lock_ready: bool,
            reason: str,
            yaw_locked: bool = False,
            dist_locked: bool = False,
            lat_locked: bool = False,
            confidence_ok: bool = False,
            lock_count_inc_reason: str = "",
            lock_count_hold_reason: str = "",
            lock_count_reset_reason: str = "",
        ) -> Dict[str, object]:
            obs_age_ms = self._table_obs_age_ms(obs)
            stale_reason = self._vision_stale_reason(obs)
            valid_for_control = bool(getattr(obs, "edge_trusted", False)) if obs is not None else False
            usable_for_approach = bool(getattr(obs, "usable_for_approach", False)) if obs is not None else False
            usable_for_stop = bool(getattr(obs, "usable_for_stop", False)) if obs is not None else False
            raw_control_level = self._raw_control_level(obs)
            normalized_control_level = self._control_level(obs)
            out_reason = str(reason or "")
            hold_text = str(lock_count_hold_reason or "")
            reset_text = str(lock_count_reset_reason or ("" if (bool(lock_ready) or hold_text) else out_reason))
            return {
                "lock_ready": bool(lock_ready),
                "reason": out_reason,
                "yaw_locked": bool(yaw_locked),
                "dist_locked": bool(dist_locked),
                "lat_locked": bool(lat_locked),
                "stable_count": int(stable_count),
                "yaw_err": getattr(obs, "yaw_err_rad", None) if obs is not None else None,
                "dist_err": getattr(obs, "dist_err_m", None) if obs is not None else None,
                "obs_age_ms": obs_age_ms,
                "valid_for_control": bool(valid_for_control),
                "usable_for_approach": bool(usable_for_approach),
                "usable_for_stop": bool(usable_for_stop),
                "raw_control_level": raw_control_level,
                "normalized_control_level": normalized_control_level,
                "control_level": normalized_control_level,
                "confidence": float(getattr(obs, "confidence", 0.0) or 0.0) if obs is not None else 0.0,
                "yaw_ok": bool(yaw_locked),
                "dist_ok": bool(dist_locked),
                "age_ok": bool(obs is not None and self._table_obs_stale_level(obs) == "fresh"),
                "confidence_ok": bool(confidence_ok),
                "stable_lock_count": int(stable_count),
                "required_lock_count": int(required_count),
                "required_ready_obs": int(required_count),
                "final_lock_window_ms": int(self._final_lock_window_ms()),
                "lock_ready_obs_count": int(stable_count),
                "window_ready_count": int(stable_count),
                "same_obs_reuse_count": int(self.ctx.final_lock_same_obs_reuse_count),
                "consecutive_lost_count": int(self.ctx.final_lock_consecutive_lost_count),
                "final_lock_transition_block_reason": "",
                "final_lock_transition_reason": "",
                "lock_count_inc_reason": str(lock_count_inc_reason or ("lock_ready" if bool(lock_ready) else "")),
                "lock_count_hold_reason": hold_text,
                "lock_count_reset_reason": reset_text,
                "lock_reset_reason": reset_text,
                "vision_stale_reason": stale_reason,
            }

        depth_stop_ready = self._roi_depth_stop_ready(obs)

        if obs is None:
            reason = "vision_stale" if self.ctx.last_table_obs is not None else "table_lost"
            return _status(lock_ready=False, reason=reason, lock_count_reset_reason="no_recent_obs")
        if not bool(getattr(obs, "table_found", False)) and not depth_stop_ready:
            return _status(lock_ready=False, reason="table_lost", lock_count_reset_reason="table_lost")
        if not bool(obs.edge_found) and not depth_stop_ready:
            return _status(lock_ready=False, reason="no_edge", lock_count_reset_reason="no_edge")
        if not self._edge_valid_for_follow(obs) and not depth_stop_ready:
            return _status(lock_ready=False, reason="edge_invalid", lock_count_reset_reason="edge_invalid")
        stale_reason = self._vision_stale_reason(obs)
        if stale_reason in {"hard_stale", "obs_invalid", "temporal_jump", "no_recent_obs", "yaw_jump", "dist_jump"} and not depth_stop_ready:
            return _status(lock_ready=False, reason="vision_stale", lock_count_reset_reason=stale_reason)
        min_confidence = float(getattr(self.controller.docking.cfg, "min_confidence", 0.0))
        confidence_ok = float(obs.confidence or 0.0) >= min_confidence
        if not confidence_ok and not depth_stop_ready:
            return _status(lock_ready=False, reason="low_confidence", confidence_ok=False, lock_count_reset_reason="low_confidence")
        if obs.depth_valid is False and not depth_stop_ready:
            return _status(lock_ready=False, reason="vision_stale", confidence_ok=confidence_ok, lock_count_reset_reason="obs_invalid")
        yaw_tol = float(getattr(self.cfg, "table_yaw_tol_rad", self.cfg.final_lock_yaw_tol_rad))
        dist_tol = float(getattr(self.cfg, "table_dist_tol_m", self.cfg.final_lock_dist_tol_m))
        
        if obs.yaw_err_rad is None:
            yaw_ok = True if depth_stop_ready else False
        else:
            yaw_ok = abs(float(obs.yaw_err_rad)) <= yaw_tol
            
        if depth_stop_ready:
            dist_ok = True
            dist_delta = 0.0
        else:
            measured_dist = self._table_measured_dist_m(obs)
            target_dist = self._table_target_dist_m(obs)
            dist_delta = (float(measured_dist) - target_dist) if measured_dist is not None else None
            dist_ok = dist_delta is not None and abs(float(dist_delta)) <= dist_tol
            
        if obs.lateral_err_m is None:
            lat_ok = True if depth_stop_ready else False
        else:
            lat_ok = abs(float(obs.lateral_err_m)) <= float(self.cfg.final_lock_lateral_tol_m)
            
        reason = "stable_count_not_enough"
        if not yaw_ok:
            reason = "yaw_not_aligned"
        elif dist_delta is None and not depth_stop_ready:
            reason = "vision_stale"
        elif not dist_ok:
            reason = "distance_too_far" if dist_delta is not None and float(dist_delta) > 0 else "distance_too_close"
        elif not lat_ok:
            reason = "yaw_not_aligned"
            
        lock_ready = bool((yaw_ok and dist_ok and lat_ok and stale_reason != "soft_stale") or depth_stop_ready)
        if depth_stop_ready:
            reason = "allowed"
            
        hold_reason = "soft_stale" if (stale_reason == "soft_stale" and not depth_stop_ready) else ""
        reset_reason = "" if (lock_ready or hold_reason) else reason
        return _status(
            lock_ready=lock_ready,
            reason="soft_stale" if hold_reason else reason,
            yaw_locked=bool(yaw_ok),
            dist_locked=bool(dist_ok),
            lat_locked=bool(lat_ok),
            confidence_ok=confidence_ok,
            lock_count_inc_reason="fresh_lock_ready" if lock_ready else "",
            lock_count_hold_reason=hold_reason,
            lock_count_reset_reason=reset_reason,
        )

    def _log_final_lock_summary(
        self,
        obs: Optional[TableEdgeObs],
        *,
        lock_ready: bool,
        reason: str,
        stable_count: int,
        phase: str = "",
        status: Optional[Dict[str, object]] = None,
    ) -> None:
        status = dict(status or self._final_lock_status(obs, stable_count=stable_count))
        measured_distance = None
        target_distance = None
        if obs is not None:
            target_distance = obs.target_dist_m
            if obs.dist_err_m is not None and target_distance is not None:
                measured_distance = float(target_distance) + float(obs.dist_err_m)
        reset_reason = str(status.get("lock_count_reset_reason") or status.get("lock_reset_reason") or "")
        stale_reason = str(status.get("vision_stale_reason") or "")
        reason_text = str(reason or status.get("reason") or "")
        should_emit = bool(
            reset_reason
            or status.get("lock_count_hold_reason")
            or status.get("lock_count_inc_reason")
            or status.get("final_lock_transition_block_reason")
            or status.get("final_lock_transition_reason")
            or "stale" in stale_reason
            or "stale" in reason_text
            or "lost" in reason_text
            or "reset" in reason_text
        )
        if not should_emit:
            return
        lines = [
            "FINAL_LOCK summary:",
            f"phase={phase or str(self.ctx.table_dock_phase or '').upper() or self.ctx.state.value}",
            f"state={self.ctx.state.value}",
            f"obs_seq={status.get('obs_seq')}",
            f"table_found={bool(obs.table_found) if obs is not None else False}",
            f"conf={float(obs.confidence or 0.0):.3f}" if obs is not None else "conf=0.000",
            f"yaw_err={obs.yaw_err_rad}" if obs is not None else "yaw_err=None",
            f"obs_age_ms={status.get('obs_age_ms')}",
            f"measured_distance={measured_distance}",
            f"target_distance={target_distance}",
            f"dist_err={obs.dist_err_m if obs is not None else None}",
            f"yaw_ok={bool(status.get('yaw_ok'))}",
            f"dist_ok={bool(status.get('dist_ok'))}",
            f"yaw_locked={bool(status['yaw_locked'])}",
            f"dist_locked={bool(status['dist_locked'])}",
            f"age_ok={bool(status.get('age_ok'))}",
            f"confidence_ok={bool(status.get('confidence_ok'))}",
            f"valid_for_control={bool(status.get('valid_for_control'))}",
            f"usable_for_approach={bool(status.get('usable_for_approach'))}",
            f"usable_for_stop={bool(status.get('usable_for_stop'))}",
            f"raw_control_level={status.get('raw_control_level')}",
            f"normalized_control_level={status.get('normalized_control_level')}",
            f"control_level={status.get('control_level')}",
            f"stable_lock_count={int(stable_count)}",
            f"stable_count={int(stable_count)}",
            f"required_lock_count={int(status.get('required_lock_count', 0) or 0)}",
            f"lock_ready_obs_count={int(status.get('lock_ready_obs_count', 0) or 0)}",
            f"window_ready_count={int(status.get('window_ready_count', 0) or 0)}",
            f"required_ready_obs={int(status.get('required_ready_obs', 0) or 0)}",
            f"final_lock_window_ms={int(status.get('final_lock_window_ms', 0) or 0)}",
            f"same_obs_reuse_count={int(status.get('same_obs_reuse_count', 0) or 0)}",
            f"consecutive_lost_count={int(status.get('consecutive_lost_count', 0) or 0)}",
            f"lock_count_inc_reason={status.get('lock_count_inc_reason')}",
            f"lock_count_hold_reason={status.get('lock_count_hold_reason')}",
            f"lock_count_reset_reason={status.get('lock_count_reset_reason')}",
            f"final_lock_transition_block_reason={status.get('final_lock_transition_block_reason')}",
            f"final_lock_transition_reason={status.get('final_lock_transition_reason')}",
            f"vision_stale_reason={status.get('vision_stale_reason')}",
            f"stop_source={status.get('stop_source')}",
            f"roi_depth_stat={status.get('roi_depth_stat')}",
            f"roi_depth_valid_ratio={status.get('roi_depth_valid_ratio')}",
            f"roi_depth_sample_count={status.get('roi_depth_sample_count')}",
            f"roi_depth_threshold={status.get('roi_depth_threshold')}",
            f"roi_depth_consecutive_frames={status.get('roi_depth_consecutive_frames')}",
            f"roi_depth_window_ready={status.get('roi_depth_window_ready')}",
            f"lock_reset_reason={status.get('lock_reset_reason')}",
            f"lock_ready={bool(lock_ready)}",
            f"reason={reason or status['reason']}",
        ]
        self._log("info", "\n".join(lines))
