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
)
from ...bridge.arm_protocol import parse_arm_response
from ...utils.grasp_utils import grasp_to_pose_params
from ...utils.target_utils import target_to_class_id
from ..common import monotonic_ts
from ..context import RuntimeContext, State
from ..controller import MotionController, MotionDecision
from ..control_authority import decide_table_control_authority
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


class TargetSearchMixin:
    def _tick_search_target_init(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        target_obs = self._fresh_target_obs()
        candidate_ok, candidate_reason = self._target_candidate_status(
            target_obs,
            self.cfg.target_confirm_conf_th,
            min_area=self.cfg.target_confirm_min_bbox_area,
        )
        if candidate_ok and target_obs is not None:
            self._transition(State.EDGE_SLIDE_SEARCH, self._format_target_transition_reason("target_found_start_lateral_align", target_obs))
            return self._annotate_target_lateral_decision(
                self.controller.stop_cmd("EDGE_SLIDE_SEARCH"),
                target_obs,
                active=False,
                reason="target_found_start_lateral_align",
                vy_cmd=0.0,
            )
        hold_s = max(0.0, float(getattr(self.cfg, "search_target_init_hold_s", 0.25) or 0.25))
        if self._state_elapsed() < hold_s:
            decision = self.controller.stop_cmd("SEARCH_TARGET_INIT")
            if decision.control_summary is not None:
                decision.control_summary.update(
                    {
                        "target_found": bool(target_obs is not None and getattr(target_obs, "found", False)),
                        "target_lateral_align_active": False,
                        "target_lateral_align_reason": candidate_reason,
                        "vx_mps": 0.0,
                        "vy_mps": 0.0,
                        "wz_radps": 0.0,
                    }
                )
            return decision
        self._transition(State.EDGE_SLIDE_SEARCH, f"target_search_init_hold_done reason={candidate_reason}")
        return self.controller.stop_cmd("EDGE_SLIDE_SEARCH")

    def _tick_edge_slide_search(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        target_obs = self._fresh_target_obs()
        candidate_ok, candidate_reason = self._target_candidate_status(
            target_obs,
            self.cfg.target_confirm_conf_th,
            min_area=self.cfg.target_confirm_min_bbox_area,
        )
        target_window = self._record_target_window_sample(target_obs, candidate_reason)
        if candidate_ok and target_obs is not None:
            self.ctx.target_found_frames += 1
            self.ctx.target_lost_frames = 0
            self._update_target_stability(target_obs)
            align_decision = self._target_lateral_align_decision(target_obs, state="EDGE_SLIDE_SEARCH")
            if align_decision is not None:
                return align_decision
            found_ratio_ok = (
                float(target_window.get("found_ratio", 0.0) or 0.0)
                >= float(getattr(self.cfg, "target_confirm_found_ratio_th", 0.5) or 0.5)
                and int(target_window.get("samples", 0) or 0) >= int(self.cfg.target_found_frames_to_confirm)
            )
            consecutive_ok = self.ctx.target_found_frames >= int(self.cfg.target_found_frames_to_confirm)
            centered_ok = self._target_lateral_centered(target_obs)
            stable_ok = int(self.ctx.target_lateral_stable_count) >= self._target_lateral_stable_frames()
            if (found_ratio_ok or consecutive_ok) and centered_ok and stable_ok:
                self.ctx.target_last_transition_reason = (
                    f"confirm_enter found_ratio={float(target_window.get('found_ratio', 0.0) or 0.0):.2f} "
                    f"consecutive_frames={int(self.ctx.target_found_frames)} bbox_valid={int(self._target_bbox_valid(target_obs))} "
                    f"target_lateral_stable_count={int(self.ctx.target_lateral_stable_count)}"
                )
                self._transition(
                    State.TARGET_CONFIRM,
                    self._format_target_transition_reason("target_found", target_obs),
                )
                return self._annotate_target_lateral_decision(
                    self.controller.stop_cmd("TARGET_CONFIRM"),
                    target_obs,
                    active=False,
                    reason="target_lateral_centered_confirm",
                    vy_cmd=0.0,
                )
            if centered_ok:
                return self._annotate_target_lateral_decision(
                    self.controller.stop_cmd("EDGE_SLIDE_SEARCH"),
                    target_obs,
                    active=False,
                    reason="target_centered_wait_stable",
                    vy_cmd=0.0,
                )
        else:
            had_recent_target = bool(
                self.ctx.target_found_frames > 0
                or self.ctx.target_lateral_stable_count > 0
                or self.ctx.target_stable_since_mono > 0.0
            )
            self.ctx.target_found_frames = 0
            self.ctx.target_lateral_stable_count = 0
            self.ctx.target_lateral_vy_cmd = 0.0
            self.ctx.target_last_lost_reason = candidate_reason
            if had_recent_target:
                self._start_loss_timer("target_loss_since_mono")
                lost_s = self._loss_elapsed(self.ctx.target_loss_since_mono)
                lost_hold_s = float(getattr(self.cfg, "target_lateral_align_lost_hold_s", 0.8) or 0.8)
                if lost_s < lost_hold_s:
                    return self._annotate_target_lateral_decision(
                        self.controller.stop_cmd("EDGE_SLIDE_SEARCH"),
                        target_obs,
                        active=False,
                        reason=f"target_lost_hold lost_s={lost_s:.2f}",
                        vy_cmd=0.0,
                    )
            else:
                self.ctx.target_loss_since_mono = 0.0
            return self._annotate_target_lateral_decision(
                self.controller.stop_cmd("EDGE_SLIDE_SEARCH"),
                target_obs,
                active=False,
                reason=candidate_reason or "target_not_found_hold",
                vy_cmd=0.0,
            )
        if self._state_elapsed() >= float(self.cfg.target_search_timeout_s):
            self.ctx.last_fail_reason = "当前桌边未找到目标"
            if self._can_relocate_edge():
                self.ctx.advance_edge()
                self._transition(State.LEAVE_EDGE, f"{self.ctx.last_fail_reason}，切换到边 {self.ctx.current_edge_id}")
                self._queue_tts("当前边未找到目标，准备换边")
                return self.controller.leave_edge_cmd()
            self._transition(State.NEXT_TABLE, f"{self.ctx.last_fail_reason}，准备切换下一张桌")
            self._queue_tts("当前桌位未找到目标，尝试下一张桌")
            return self.controller.next_table_cmd(turn_sign=self.ctx.relocate_turn_sign)
        edge_obs = self._fresh_table_obs()
        if edge_obs is None or not self._table_visible(edge_obs):
            return self._annotate_target_lateral_decision(
                self.controller.stop_cmd("EDGE_SLIDE_SEARCH"),
                target_obs,
                active=False,
                reason="target_search_hold_no_edge_motion",
                vy_cmd=0.0,
            )
        if self._edge_obs_is_stale(edge_obs):
            age_ms = self._table_obs_age_ms(edge_obs)
            age_text = "unknown" if age_ms is None else f"{age_ms:.0f}"
            self.ctx.last_fail_reason = f"edge_follow_stale age_ms={age_text}"
            self.ctx.last_edge_quality = {
                "mode": "stale",
                "reason": "edge_follow_stale",
                "fallback_candidate_state": self._edge_slide_stale_fallback_state().value,
                "fallback_decision": "stale_hold",
            }
            return self._handle_edge_slide_edge_loss(
                "edge_follow_stale",
                fallback_state=self._edge_slide_stale_fallback_state(),
                use_last_obs_for_fallback=False,
            )
        if not self._edge_valid_for_follow(edge_obs):
            reason = str(edge_obs.reason or "no_valid_edge").strip() or "no_valid_edge"
            self.ctx.last_fail_reason = reason
            quality = {
                "mode": "pause",
                "reason": reason,
                "fallback_candidate_state": self._edge_slide_fallback_state().value,
            }
            self.ctx.last_edge_quality = dict(quality)
            return self._edge_slide_pause_or_recover(edge_obs, quality)
        quality = self._edge_follow_quality(edge_obs)
        self.ctx.last_edge_quality = dict(quality)
        if str(quality.get("mode")) in {"identity_mismatch", "pause", "recover"}:
            return self._edge_slide_pause_or_recover(edge_obs, quality)
        if edge_obs.dist_err_m is None:
            return self._handle_edge_slide_edge_loss("dist_err_missing")
        if abs(float(edge_obs.dist_err_m)) > float(self.cfg.edge_slide_dist_tolerance_m):
            self._start_loss_timer("table_loss_since_mono")
            lost_s = self._loss_elapsed(self.ctx.table_loss_since_mono)
            dist = float(edge_obs.dist_err_m)
            tol = float(self.cfg.edge_slide_dist_tolerance_m)
            hold_s = float(getattr(self.cfg, "edge_slide_dist_out_of_range_hold_s", self.cfg.edge_slide_pause_hold_s) or self.cfg.edge_slide_pause_hold_s)
            warning = f"edge_distance_out_of_tolerance dist={dist:+.3f} tol={tol:.3f}"
            if warning not in self.ctx.task_warning_history:
                self.ctx.task_warning_history.append(warning)
            if lost_s >= hold_s:
                self.ctx.edge_slide_relock_attempts += 1
                max_attempts = max(0, int(getattr(self.cfg, "edge_slide_max_relock_attempts", 3) or 3))
                fatal = bool(getattr(self.cfg, "edge_slide_relock_failure_is_fatal", True))
                if fatal and self.ctx.edge_slide_relock_attempts > max_attempts:
                    reason = "edge_distance_out_of_tolerance_after_retries"
                    self.ctx.last_fail_reason = reason
                    self._enter_error_recovery(reason, tts_text="任务失败，无法稳定锁定桌边。", interrupt_tts=True)
                    return self.controller.stop_cmd("ERROR_RECOVERY", brake=True)
                fallback_state = self._edge_slide_fallback_state()
                self._transition(
                    fallback_state,
                    (
                        "edge_distance_out_of_tolerance "
                        f"dist={dist:+.3f} "
                        f"tol={tol:.3f} "
                        f"hold_s={lost_s:.2f} "
                        f"attempt={int(self.ctx.edge_slide_relock_attempts)}/{max_attempts} "
                        "recoverable=true severity=warning"
                    ),
                )
                return self._edge_slide_fallback_cmd(fallback_state, edge_obs)
            quality = dict(quality)
            pause_s = float(getattr(self.cfg, "edge_slide_pause_hold_s", 0.8) or 0.8)
            quality["mode"] = "pause" if lost_s < pause_s else "recover"
            quality["reason"] = "edge_distance_out_of_tolerance"
            quality["severity"] = "warning"
            quality["recoverable"] = True
            quality["dist_err_m"] = dist
            quality["dist_tolerance_m"] = tol
            quality["relock_attempts"] = int(self.ctx.edge_slide_relock_attempts)
            quality["max_relock_attempts"] = int(getattr(self.cfg, "edge_slide_max_relock_attempts", 3) or 3)
            self.ctx.last_edge_quality = dict(quality)
            decision = self.controller.edge_slide_hold_cmd(
                f"edge_distance_out_of_tolerance_{quality['mode']} elapsed_s={lost_s:.2f}",
                edge_obs=edge_obs,
            )
            return self._annotate_edge_slide_decision(
                decision,
                quality,
                stop_reason="edge_distance_out_of_tolerance",
                fallback_decision="relock_hold",
                pause_elapsed_s=min(lost_s, pause_s),
                recover_elapsed_s=max(0.0, lost_s - pause_s),
            )
        else:
            self._reset_table_loss()
        if self._obs_has_motion(target_obs) and self._target_quality_ok(target_obs, self.cfg.target_confirm_conf_th):
            return self.controller.target_track_cmd(target_obs)
        direction = self._edge_slide_direction()
        quality_mode = str(quality.get("mode") or "strong")
        vy_mps = None
        reason = "edge_slide"
        if quality_mode == "weak":
            vy_mps = float(getattr(self.controller.car_cfg, "edge_slide_weak_vy_mps", 0.05) or 0.05)
            reason = "weak_edge_slide"
        else:
            self._reset_table_loss()
        decision = self.controller.edge_slide_search_cmd(
            self._segment_elapsed(self.cfg.edge_slide_segment_s),
            direction_sign=direction,
            edge_obs=edge_obs,
            vy_mps=vy_mps,
            reason=reason,
        )
        return self._annotate_edge_slide_decision(decision, quality, fallback_decision="slide")

    def _edge_slide_fallback_state(self) -> State:
        if not self._table_final_lock_enabled():
            return State.EDGE_ADJUST
        raw = str(getattr(self.cfg, "edge_slide_fallback_state", "") or "").strip().upper()
        direct = bool(getattr(self.cfg, "edge_slide_direct_fallback_to_controlled_approach", False))
        if direct and raw in {"CONTROLLED_APPROACH", "EDGE_ADJUST"}:
            return State.EDGE_ADJUST
        return State.FINAL_SLOW_STOP

    def _edge_slide_stale_fallback_state(self) -> State:
        if not self._table_final_lock_enabled():
            return State.EDGE_ADJUST
        raw = str(getattr(self.cfg, "edge_follow_stale_fallback_state", "") or "").strip().upper()
        direct = bool(getattr(self.cfg, "edge_slide_direct_fallback_to_controlled_approach", False))
        if direct and raw in {"CONTROLLED_APPROACH", "EDGE_ADJUST"}:
            return State.EDGE_ADJUST
        return State.FINAL_SLOW_STOP

    def _edge_slide_fallback_cmd(self, state: State, edge_obs: Optional[TableEdgeObs]) -> MotionDecision:
        if state == State.FINAL_SLOW_STOP:
            return self.controller.fov_table_approach_cmd(edge_obs, phase="PLANE_FINAL_LOCK", mode="FINAL_SLOW_STOP")
        return self.controller.fov_table_approach_cmd(edge_obs, phase="PLANE_APPROACH", mode="EDGE_ADJUST")

    def _annotate_edge_slide_decision(
        self,
        decision: MotionDecision,
        quality: Dict[str, Any],
        *,
        stop_reason: str = "",
        fallback_decision: str = "",
        pause_elapsed_s: Optional[float] = None,
        recover_elapsed_s: Optional[float] = None,
    ) -> MotionDecision:
        summary = dict(decision.control_summary or {})
        cmd = decision.cmd
        summary.update(
            {
                "edge_quality_mode": quality.get("mode"),
                "stop_reason": stop_reason or summary.get("stop_reason") or "",
                "slide_vy_mps": float(getattr(self.controller.car_cfg, "edge_slide_vy_mps", 0.0) or 0.0),
                "weak_slide_vy_mps": float(getattr(self.controller.car_cfg, "edge_slide_weak_vy_mps", 0.0) or 0.0),
                "final_vx": float(cmd.vx_mps),
                "final_vy": float(cmd.vy_mps),
                "final_wz": float(cmd.wz_radps),
                "pause_elapsed_ms": int(round(max(0.0, float(pause_elapsed_s or 0.0)) * 1000.0)),
                "recover_elapsed_ms": int(round(max(0.0, float(recover_elapsed_s or 0.0)) * 1000.0)),
                "fallback_candidate_state": quality.get("fallback_candidate_state", self._edge_slide_fallback_state().value),
                "fallback_decision": fallback_decision or ("none" if abs(float(cmd.vy_mps or 0.0)) > 0.0 else "hold"),
                "severity": quality.get("severity"),
                "recoverable": quality.get("recoverable"),
                "dist_tolerance_m": quality.get("dist_tolerance_m"),
                "relock_attempts": quality.get("relock_attempts"),
                "max_relock_attempts": quality.get("max_relock_attempts"),
            }
        )
        if "vx_from_dist" not in summary:
            summary["vx_from_dist"] = float(cmd.vx_mps)
        if "wz_from_yaw" not in summary:
            summary["wz_from_yaw"] = float(cmd.wz_radps)
        decision.control_summary = summary
        return decision

    def _edge_slide_pause_or_recover(self, edge_obs: TableEdgeObs, quality: Dict[str, Any]) -> MotionDecision:
        reason = str(quality.get("reason") or quality.get("mode") or "edge_uncertain")
        if reason in {"edge_identity_mismatch", "edge_follow_stale", "edge_conf_low", "conf_low", "target_lost"}:
            warning = f"{reason} recoverable=true severity=warning"
            if warning not in self.ctx.task_warning_history:
                self.ctx.task_warning_history.append(warning)
        else:
            self.ctx.last_fail_reason = reason
        self._start_loss_timer("table_loss_since_mono")
        elapsed_s = self._loss_elapsed(self.ctx.table_loss_since_mono)
        pause_s = float(getattr(self.cfg, "edge_slide_pause_hold_s", 0.8) or 0.8)
        recover_timeout_s = float(getattr(self.cfg, "edge_slide_recover_timeout_s", self.cfg.table_loss_hold_s) or self.cfg.table_loss_hold_s)
        quality = dict(quality)
        if elapsed_s >= recover_timeout_s:
            fallback_state = self._edge_slide_fallback_state()
            quality["mode"] = "recover"
            quality["fallback_decision"] = f"fallback_to_{fallback_state.value}"
            self.ctx.last_edge_quality = dict(quality)
            self._transition(fallback_state, f"{reason} recover_timeout_s={elapsed_s:.2f}")
            return self._edge_slide_fallback_cmd(fallback_state, edge_obs)
        if elapsed_s >= pause_s:
            quality["mode"] = "recover"
            control_reason = f"edge_recover stop_reason={reason} elapsed_s={elapsed_s:.2f}"
            fallback_decision = "recover_hold"
        else:
            quality["raw_mode"] = quality.get("mode")
            quality["mode"] = "pause"
            control_reason = f"edge_pause stop_reason={reason} elapsed_s={elapsed_s:.2f}"
            fallback_decision = "pause_hold"
        quality["pause_elapsed_ms"] = int(round(min(elapsed_s, pause_s) * 1000.0))
        quality["recover_elapsed_ms"] = int(round(max(0.0, elapsed_s - pause_s) * 1000.0))
        quality["fallback_decision"] = fallback_decision
        self.ctx.last_edge_quality = dict(quality)
        decision = self.controller.edge_slide_hold_cmd(control_reason, edge_obs=edge_obs)
        return self._annotate_edge_slide_decision(
            decision,
            quality,
            stop_reason=reason,
            fallback_decision=fallback_decision,
            pause_elapsed_s=min(elapsed_s, pause_s),
            recover_elapsed_s=max(0.0, elapsed_s - pause_s),
        )

    def _handle_edge_slide_edge_loss(
        self,
        reason: str,
        fallback_state: Optional[State] = None,
        use_last_obs_for_fallback: bool = True,
    ) -> MotionDecision:
        self._start_loss_timer("table_loss_since_mono")
        lost_s = self._loss_elapsed(self.ctx.table_loss_since_mono)
        hold_s = self._edge_slide_loss_hold_s(reason)
        if lost_s < hold_s:
            return self.controller.edge_slide_hold_cmd(f"{reason}_hold lost_s={lost_s:.2f}")
        fallback_state = fallback_state or self._edge_slide_fallback_state()
        self._transition(fallback_state, f"{reason} lost_s={lost_s:.2f}")
        fallback_obs = self.ctx.last_table_obs if use_last_obs_for_fallback else None
        return self._edge_slide_fallback_cmd(fallback_state, fallback_obs)

    def _edge_slide_loss_hold_s(self, reason: str) -> float:
        raw = str(reason or "").strip()
        if raw.startswith("edge_follow_stale") or self._edge_obs_is_stale(self.ctx.last_table_obs):
            return float(getattr(self.cfg, "edge_follow_stale_hold_s", self.cfg.table_loss_hold_s) or self.cfg.table_loss_hold_s)
        return float(self.cfg.table_loss_hold_s)

    def _tick_target_confirm(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        obs = self._fresh_target_obs()
        visible_ok, visible_reason = self._target_candidate_status(
            obs,
            self.cfg.target_confirm_conf_th,
            min_area=self.cfg.target_confirm_min_bbox_area,
        )
        target_window = self._record_target_window_sample(obs, visible_reason)
        lock_ok, lock_reason = self._target_candidate_status(obs, self.cfg.target_lock_conf_th, min_area=0.0)
        if visible_ok and obs is not None:
            self.ctx.target_found_frames += 1
            self.ctx.target_lost_frames = 0
            center_jitter = self._update_target_stability(obs)
            align_decision = self._target_lateral_align_decision(obs, state="TARGET_CONFIRM")
            if align_decision is not None:
                return align_decision
            window_jitter = self._float_or_none(target_window.get("center_jitter"))
            if window_jitter is not None:
                center_jitter = window_jitter
                self.ctx.target_last_center_jitter = float(window_jitter)
            confirm_elapsed_s = self._state_elapsed()
            confirm_min_ok = confirm_elapsed_s >= float(self.cfg.target_confirm_min_s)
            found_ratio = float(target_window.get("found_ratio", 0.0) or 0.0)
            conf_median = self._float_or_none(target_window.get("conf_median"))
            stable_ok = self._target_stable_ms() >= int(round(float(self.cfg.target_lock_stable_s) * 1000.0))
            jitter_ok = center_jitter <= float(self.cfg.target_lock_center_jitter_th)
            ratio_ok = found_ratio >= float(getattr(self.cfg, "target_lock_found_ratio_th", 0.6) or 0.6)
            conf_ok = conf_median is not None and conf_median >= float(self.cfg.target_lock_conf_th or 0.0)
            centered_ok = self._target_lateral_centered(obs)
            lateral_stable_ok = int(self.ctx.target_lateral_stable_count) >= self._target_lateral_stable_frames()
            if lock_ok and confirm_min_ok and stable_ok and jitter_ok and ratio_ok and conf_ok and centered_ok and lateral_stable_ok:
                self.ctx.target_last_transition_reason = (
                    f"lock_ok found_ratio={found_ratio:.2f} conf_median={float(conf_median):.3f} "
                    f"center_jitter={float(center_jitter):.3f} stable_ms={self._target_stable_ms()} "
                    f"target_lateral_stable_count={int(self.ctx.target_lateral_stable_count)}"
                )
                self.ctx.target_locked = True
                self._transition(
                    State.TARGET_LOCKED,
                    self._format_target_transition_reason("target_confirmed", obs),
                )
                return self._annotate_target_lateral_decision(
                    self.controller.stop_cmd("TARGET_LOCKED"),
                    obs,
                    active=False,
                    reason="target_lateral_centered_locked",
                    vy_cmd=0.0,
                )
            if centered_ok and not lateral_stable_ok:
                return self._annotate_target_lateral_decision(
                    self.controller.stop_cmd("TARGET_CONFIRM"),
                    obs,
                    active=False,
                    reason="target_centered_wait_stable",
                    vy_cmd=0.0,
                )
            if self._state_elapsed() >= float(self.cfg.target_confirm_timeout_s):
                reasons = []
                if not lock_ok:
                    reasons.append(lock_reason)
                if not confirm_min_ok:
                    reasons.append("confirm_min_not_reached")
                if not stable_ok:
                    reasons.append("target_stable_not_enough")
                if not jitter_ok:
                    reasons.append("center_jitter_high")
                if not ratio_ok:
                    reasons.append(f"found_ratio_low ratio={found_ratio:.2f}")
                if not conf_ok:
                    reasons.append("conf_median_low")
                if not centered_ok:
                    reasons.append("target_not_centered")
                if not lateral_stable_ok:
                    reasons.append("target_lateral_stable_not_enough")
                self.ctx.target_last_lost_reason = ",".join(reasons) or "lock_condition_timeout"
                self._transition(
                    State.EDGE_SLIDE_SEARCH,
                    self._format_target_transition_reason("confirm_timeout", obs),
                )
            return self._annotate_target_lateral_decision(
                self.controller.stop_cmd("TARGET_CONFIRM"),
                obs,
                active=False,
                reason="target_confirm_hold",
                vy_cmd=0.0,
            )
        self.ctx.target_found_frames = 0
        self.ctx.target_lateral_stable_count = 0
        self.ctx.target_lateral_vy_cmd = 0.0
        self.ctx.target_lost_frames += 1
        self._start_loss_timer("target_loss_since_mono")
        lost_s = self._loss_elapsed(self.ctx.target_loss_since_mono)
        self.ctx.target_last_lost_reason = f"{visible_reason} lost_hold_ms={int(round(lost_s * 1000.0))}"
        if self._state_elapsed() < float(self.cfg.target_confirm_min_s):
            return self._annotate_target_lateral_decision(
                self.controller.stop_cmd("TARGET_CONFIRM"),
                obs,
                active=False,
                reason="target_confirm_lost_hold",
                vy_cmd=0.0,
            )
        if lost_s >= float(self.cfg.target_confirm_lost_hold_s):
            self._reset_target_stability(visible_reason)
            self._transition(
                State.EDGE_SLIDE_SEARCH,
                self._format_target_transition_reason("confirm_lost_hold_exceeded", obs),
            )
        return self.controller.stop_cmd("TARGET_CONFIRM")

    def _tick_target_locked(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        obs = self._fresh_target_obs()
        lock_ok, lock_reason = self._target_candidate_status(obs, self.cfg.target_lock_conf_th, min_area=0.0)
        target_window = self._record_target_window_sample(obs, lock_reason)
        if not lock_ok or obs is None:
            self.ctx.target_lost_frames += 1
            self.ctx.target_locked = False
            self._start_loss_timer("target_loss_since_mono")
            lost_s = self._loss_elapsed(self.ctx.target_loss_since_mono)
            self.ctx.target_last_lost_reason = f"{lock_reason} lost_hold_ms={int(round(lost_s * 1000.0))}"
            if lost_s >= float(self.cfg.target_lock_lost_hold_s):
                self._reset_target_stability(lock_reason)
                self._transition(
                    State.EDGE_SLIDE_SEARCH,
                    self._format_target_transition_reason("locked_lost_hold_exceeded", obs),
                )
                return self._annotate_target_lateral_decision(
                    self.controller.stop_cmd("EDGE_SLIDE_SEARCH"),
                    obs,
                    active=False,
                    reason="target_locked_lost_return_search",
                    vy_cmd=0.0,
                )
            return self._annotate_target_lateral_decision(
                self.controller.stop_cmd("TARGET_LOCKED"),
                obs,
                active=False,
                reason="target_locked_lost_hold",
                vy_cmd=0.0,
            )
        self.ctx.target_lost_frames = 0
        self.ctx.target_lock_frames += 1
        self.ctx.target_locked = True
        self.ctx.target_lateral_vy_cmd = 0.0
        center_jitter = self._update_target_stability(obs)
        window_jitter = self._float_or_none(target_window.get("center_jitter"))
        if window_jitter is not None:
            center_jitter = window_jitter
            self.ctx.target_last_center_jitter = float(window_jitter)
        found_ratio = float(target_window.get("found_ratio", 0.0) or 0.0)
        conf_median = self._float_or_none(target_window.get("conf_median"))
        stable_ok = self._target_stable_ms() >= int(round(float(self.cfg.target_lock_stable_s) * 1000.0))
        jitter_ok = center_jitter <= float(self.cfg.target_lock_center_jitter_th)
        conf_stable = conf_median is not None and conf_median >= float(self.cfg.target_lock_conf_th or 0.0)
        ratio_ok = found_ratio >= float(getattr(self.cfg, "target_lock_found_ratio_th", 0.6) or 0.6)
        centered_ok = self._target_lateral_centered(obs)
        lateral_stable_ok = int(self.ctx.target_lateral_stable_count) >= self._target_lateral_stable_frames()
        ready_for_grasp = self._check_target_ready_for_grasp(obs)
        if (
            self._state_elapsed() >= float(self.cfg.target_locked_freeze_after_s)
            and stable_ok
            and jitter_ok
            and conf_stable
            and ratio_ok
            and centered_ok
            and lateral_stable_ok
            and self.ctx.target_loss_since_mono <= 0.0
            and ready_for_grasp
        ):
            self.ctx.target_last_transition_reason = (
                f"freeze_ok found_ratio={found_ratio:.2f} conf_median={float(conf_median):.3f} "
                f"center_jitter={float(center_jitter):.3f} stable_ms={self._target_stable_ms()} "
                f"target_lateral_stable_count={int(self.ctx.target_lateral_stable_count)}"
            )
            self._transition(
                State.FREEZE_BASE,
                self._format_target_transition_reason("locked_stable_freeze", obs),
            )
            return self._annotate_target_lateral_decision(
                self.controller.stop_cmd("FREEZE_BASE"),
                obs,
                active=False,
                reason="target_locked_freeze_base",
                vy_cmd=0.0,
            )
        return self._annotate_target_lateral_decision(
            self.controller.stop_cmd("TARGET_LOCKED"),
            obs,
            active=False,
            reason="target_locked_hold",
            vy_cmd=0.0,
        )

    def _tick_freeze_base(self) -> MotionDecision:
        if self._state_elapsed() < float(self.cfg.freeze_settle_s):
            return self.controller.stop_cmd("FREEZE_BASE")
        if self.ctx.task_intent == "FIND" and self.ctx.active_target:
            self._transition(State.GRASP, f"已锁定 {self.ctx.active_target}，开始抓取")
            self._queue_tts(f"已锁定 {self.ctx.active_target}，开始抓取")
            return self.controller.stop_cmd("GRASP")
        self._transition(State.DONE, f"已在桌边锁定 {self.ctx.active_target}")
        self._queue_tts(f"已在桌边锁定 {self.ctx.active_target}")
        return self.controller.stop_cmd("DONE")

    def _target_matches_active(self, obs: TargetObs) -> bool:
        if not self.ctx.active_target or not obs.target:
            return True
        return str(obs.target).strip() == str(self.ctx.active_target).strip()

    def _target_cls_matches_active(self, obs: TargetObs) -> bool:
        active = str(self.ctx.active_target or "").strip()
        if not active:
            return True
        candidate_cls = str(obs.matched_cls or obs.target or "").strip()
        if not candidate_cls:
            return True
        return candidate_cls == active

    def _target_conf_value(self, obs: TargetObs) -> Optional[float]:
        value = obs.matched_conf if obs.matched_conf is not None else obs.confidence
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _target_bbox_valid(self, obs: Optional[TargetObs]) -> bool:
        if obs is None:
            return False
        if getattr(obs, "bbox_valid", None) is False:
            return False
        return True

    def _trim_target_obs_window(self, now_m: Optional[float] = None) -> None:
        now_m = monotonic_ts() if now_m is None else float(now_m)
        window_s = max(0.2, float(getattr(self.cfg, "target_confirm_window_s", 1.5) or 1.5))
        cutoff = now_m - window_s
        self.ctx.target_obs_window = [
            dict(item) for item in self.ctx.target_obs_window if float(item.get("t", 0.0) or 0.0) >= cutoff
        ][-80:]

    def _record_target_window_sample(self, obs: Optional[TargetObs], reason: str = "") -> Dict[str, Any]:
        now_m = monotonic_ts()
        basic_found = False
        conf = None
        center = None
        bbox_valid = False
        matched_cls = None
        if obs is not None:
            conf = self._target_conf_value(obs)
            center = self._target_center_pair(obs)
            bbox_valid = self._target_bbox_valid(obs)
            matched_cls = obs.matched_cls or obs.target
            basic_found = bool(obs.found) and self._target_cls_matches_active(obs) and self._target_matches_active(obs) and bbox_valid
        sample = {
            "t": now_m,
            "found": bool(basic_found),
            "conf": conf,
            "center": center,
            "bbox_valid": bool(bbox_valid),
            "matched_cls": matched_cls,
            "reason": str(reason or ""),
        }
        self.ctx.target_obs_window.append(sample)
        self._trim_target_obs_window(now_m)
        return self._target_window_stats(now_m=now_m)

    def _target_window_stats(self, now_m: Optional[float] = None) -> Dict[str, Any]:
        self._trim_target_obs_window(now_m)
        samples = list(self.ctx.target_obs_window)
        total = len(samples)
        found_samples = [item for item in samples if bool(item.get("found", False))]
        confs = sorted(
            float(item.get("conf"))
            for item in found_samples
            if item.get("conf") is not None
        )
        if confs:
            mid = len(confs) // 2
            conf_median = confs[mid] if len(confs) % 2 else (confs[mid - 1] + confs[mid]) / 2.0
            conf_max = max(confs)
        else:
            conf_median = None
            conf_max = None
        centers = [
            tuple(item.get("center"))
            for item in found_samples
            if isinstance(item.get("center"), (tuple, list)) and len(item.get("center")) >= 2
        ]
        center_jitter = 0.0
        if len(centers) >= 2:
            mean_cx = sum(float(item[0]) for item in centers) / float(len(centers))
            mean_cy = sum(float(item[1]) for item in centers) / float(len(centers))
            center_jitter = max(math.hypot(float(item[0]) - mean_cx, float(item[1]) - mean_cy) for item in centers)
        valid_bbox_count = sum(1 for item in samples if bool(item.get("bbox_valid", False)))
        latest_found = found_samples[-1] if found_samples else {}
        found_ratio = float(len(found_samples)) / float(total) if total else 0.0
        bbox_valid_ratio = float(valid_bbox_count) / float(total) if total else 0.0
        return {
            "samples": total,
            "found_samples": len(found_samples),
            "found_ratio": found_ratio,
            "conf_median": conf_median,
            "conf_max": conf_max,
            "center_jitter": float(center_jitter),
            "bbox_valid_ratio": bbox_valid_ratio,
            "latest_matched_cls": latest_found.get("matched_cls"),
            "latest_matched_conf": latest_found.get("conf"),
        }

    def _target_found_reason(self, obs: TargetObs) -> str:
        matched_cls = str(obs.matched_cls or obs.target or "").strip() or "n/a"
        conf = self._target_conf_value(obs)
        if conf is None:
            return f"target_found matched_cls={matched_cls} matched_conf=n/a"
        return f"target_found matched_cls={matched_cls} matched_conf={float(conf):.3f}"

    def _target_quality_ok(self, obs: TargetObs, conf_th: float) -> bool:
        return self._target_candidate_status(obs, conf_th, min_area=0.0)[0]

    def _target_stable_ms(self) -> int:
        if self.ctx.target_stable_since_mono <= 0.0:
            return 0
        return int(round(max(0.0, monotonic_ts() - self.ctx.target_stable_since_mono) * 1000.0))

    def _target_center_pair(self, obs: Optional[TargetObs]) -> Optional[Tuple[float, float]]:
        if obs is None:
            return None
        full = self._target_center_full_norm(obs)
        if full is None:
            full = self._target_center(obs)
        if full is None:
            return None
        cx = self._float_or_none(full.get("cx"))
        cy = self._float_or_none(full.get("cy"))
        if cx is None and cy is None:
            return None
        return (float(cx if cx is not None else 0.0), float(cy if cy is not None else 0.0))

    def _target_bbox_area(self, obs: TargetObs) -> Optional[float]:
        for value in (obs.matched_area, obs.size_norm, obs.mask_area_ratio):
            numeric = self._float_or_none(value)
            if numeric is not None and numeric > 0.0:
                return numeric
        bbox = obs.matched_bbox or obs.bbox or obs.mask_bbox
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            values = [self._float_or_none(item) for item in bbox[:4]]
            if all(item is not None for item in values):
                x1, y1, x2, y2 = [float(item) for item in values]  # type: ignore[arg-type]
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                if width > 1.0 or height > 1.0:
                    return None
                return width * height
        return None

    def _target_lateral_stable_frames(self) -> int:
        return max(1, int(getattr(self.cfg, "target_lateral_align_stable_frames", 3) or 3))

    def _target_lateral_center_x(self, obs: Optional[TargetObs]) -> Optional[float]:
        center = self._target_center_pair(obs)
        if center is None:
            return None
        return max(0.0, min(1.0, float(center[0])))

    def _target_lateral_error_x(self, obs: Optional[TargetObs]) -> Optional[float]:
        cx = self._target_lateral_center_x(obs)
        if cx is None:
            return None
        target = max(0.0, min(1.0, float(getattr(self.cfg, "target_lateral_align_center_x_target", 0.5) or 0.5)))
        return float(cx - target)

    def _target_lateral_centered(self, obs: Optional[TargetObs]) -> bool:
        err = self._target_lateral_error_x(obs)
        if err is None:
            return False
        tol = abs(float(getattr(self.cfg, "target_lateral_align_center_x_tol", 0.06) or 0.06))
        return abs(float(err)) <= tol

    def _annotate_target_lateral_decision(
        self,
        decision: MotionDecision,
        obs: Optional[TargetObs],
        *,
        active: bool,
        reason: str,
        vy_cmd: float,
    ) -> MotionDecision:
        summary = dict(decision.control_summary or {})
        conf = self._target_conf_value(obs) if obs is not None else None
        cx = self._target_lateral_center_x(obs)
        err = self._target_lateral_error_x(obs)
        summary.update(
            {
                "target_found": bool(obs is not None and getattr(obs, "found", False)),
                "target_cls": str(getattr(obs, "matched_cls", None) or getattr(obs, "target", "") or "") if obs is not None else "",
                "target_conf": conf,
                "target_center_x_norm": cx,
                "target_err_x": err,
                "target_lateral_align_active": bool(active),
                "target_lateral_align_reason": str(reason or ""),
                "target_lateral_vy_cmd": float(vy_cmd),
                "target_lateral_stable_count": int(self.ctx.target_lateral_stable_count),
                "target_lateral_stable_frames": int(self._target_lateral_stable_frames()),
                "target_locked": bool(self.ctx.target_locked),
                "grasp_request_sent": False,
                "grasp_dry_run": False,
            }
        )
        summary["vx_mps"] = float(decision.cmd.vx_mps)
        summary["vy_mps"] = float(decision.cmd.vy_mps)
        summary["wz_radps"] = float(decision.cmd.wz_radps)
        summary["final_vx"] = float(decision.cmd.vx_mps)
        summary["final_vy"] = float(decision.cmd.vy_mps)
        summary["final_wz"] = float(decision.cmd.wz_radps)
        decision.control_summary = summary
        self.ctx.target_lateral_align_reason = str(reason or "")
        self.ctx.target_lateral_vy_cmd = float(vy_cmd)
        return decision

    def _target_lateral_align_decision(self, obs: Optional[TargetObs], *, state: str) -> Optional[MotionDecision]:
        if not bool(getattr(self.cfg, "target_lateral_align_enable", True)):
            return None
        err = self._target_lateral_error_x(obs)
        if err is None:
            self.ctx.target_lateral_stable_count = 0
            self.ctx.target_lateral_vy_cmd = 0.0
            return self._annotate_target_lateral_decision(
                self.controller.stop_cmd(state),
                obs,
                active=False,
                reason="target_center_missing",
                vy_cmd=0.0,
            )
        deadband = abs(float(getattr(self.cfg, "target_lateral_align_center_x_deadband", 0.03) or 0.03))
        tol = abs(float(getattr(self.cfg, "target_lateral_align_center_x_tol", 0.06) or 0.06))
        if abs(float(err)) <= tol:
            self.ctx.target_lateral_stable_count += 1
        else:
            self.ctx.target_lateral_stable_count = 0
        if abs(float(err)) <= deadband:
            self.ctx.target_lateral_align_reason = "target_center_deadband"
            self.ctx.target_lateral_vy_cmd = 0.0
            return None
        kp = abs(float(getattr(self.cfg, "target_lateral_align_kp_vy", 0.08) or 0.08))
        vy_min = abs(float(getattr(self.cfg, "target_lateral_align_vy_min_mps", 0.015) or 0.015))
        vy_max = abs(float(getattr(self.cfg, "target_lateral_align_vy_max_mps", 0.060) or 0.060))
        vy_raw = -float(err) * kp
        vy_cmd = max(-vy_max, min(vy_max, vy_raw))
        if abs(vy_cmd) < vy_min:
            vy_cmd = vy_min if vy_raw >= 0.0 else -vy_min
        cmd = self.controller._cmd(state, vx=0.0, vy=vy_cmd, wz=0.0)
        decision = MotionDecision(
            cmd=cmd,
            control_summary=self.controller._summary(state, cmd, reason="target_lateral_align"),
        )
        return self._annotate_target_lateral_decision(
            decision,
            obs,
            active=True,
            reason="target_lateral_align",
            vy_cmd=vy_cmd,
        )

    def _target_candidate_status(
        self,
        obs: Optional[TargetObs],
        conf_th: float,
        *,
        min_area: float = 0.0,
    ) -> Tuple[bool, str]:
        if obs is None:
            return False, "vision_stale"
        if not bool(obs.found):
            return False, "target_lost"
        if not self._target_cls_matches_active(obs):
            return False, "class_mismatch"
        if not self._target_matches_active(obs):
            return False, "target_mismatch"
        if not self._target_bbox_valid(obs):
            reason = str(getattr(obs, "bbox_invalid_reason", "") or "bbox_invalid")
            return False, reason
        conf = self._target_conf_value(obs)
        if conf is None:
            if float(conf_th or 0.0) > 0.0:
                return False, "conf_missing"
        elif conf < float(conf_th or 0.0):
            return False, f"conf_low conf={conf:.3f} th={float(conf_th or 0.0):.3f}"
        area_th = float(min_area or 0.0)
        area = self._target_bbox_area(obs)
        if area_th > 0.0 and area is not None and area < area_th:
            return False, f"bbox_area_low area={area:.4f} th={area_th:.4f}"
        return True, "target_visible"

    def _update_target_stability(self, obs: TargetObs) -> float:
        now_m = monotonic_ts()
        if self.ctx.target_stable_since_mono <= 0.0:
            self.ctx.target_stable_since_mono = now_m
        self.ctx.target_loss_since_mono = 0.0
        self.ctx.target_last_lost_reason = ""
        center = self._target_center_pair(obs)
        if center is not None:
            self.ctx.target_center_history.append({"t": now_m, "cx": center[0], "cy": center[1]})
        window_s = max(float(self.cfg.target_lock_stable_s), float(self.cfg.target_confirm_min_s), 0.5)
        cutoff = now_m - window_s
        self.ctx.target_center_history = [
            item for item in self.ctx.target_center_history if float(item.get("t", 0.0) or 0.0) >= cutoff
        ][-30:]
        self.ctx.target_last_center_jitter = self._target_center_jitter()
        return self.ctx.target_last_center_jitter

    def _target_center_jitter(self) -> float:
        points = self.ctx.target_center_history
        if len(points) < 2:
            return 0.0
        mean_cx = sum(float(item.get("cx", 0.0) or 0.0) for item in points) / float(len(points))
        mean_cy = sum(float(item.get("cy", 0.0) or 0.0) for item in points) / float(len(points))
        return max(
            math.hypot(float(item.get("cx", 0.0) or 0.0) - mean_cx, float(item.get("cy", 0.0) or 0.0) - mean_cy)
            for item in points
        )

    def _reset_target_stability(self, reason: str) -> None:
        self.ctx.target_stable_since_mono = 0.0
        self.ctx.target_center_history.clear()
        self.ctx.target_last_center_jitter = 0.0
        self.ctx.target_last_lost_reason = str(reason or "")

    def _format_target_transition_reason(self, reason: str, obs: Optional[TargetObs] = None) -> str:
        obs = obs or self.ctx.last_target_obs
        matched_cls = str(obs.matched_cls or obs.target or "").strip() if obs is not None else ""
        conf = self._target_conf_value(obs) if obs is not None else None
        center = self._target_center(obs)
        window = self._target_window_stats()
        parts = [
            str(reason or "target_transition"),
            f"found_frames={int(self.ctx.target_found_frames)}",
            f"lost_frames={int(self.ctx.target_lost_frames)}",
            f"target_stable_ms={self._target_stable_ms()}",
            f"found_ratio={float(window.get('found_ratio', 0.0) or 0.0):.2f}",
        ]
        if matched_cls:
            parts.append(f"matched_cls={matched_cls}")
        if conf is not None:
            parts.append(f"matched_conf={float(conf):.3f}")
        if window.get("conf_median") is not None:
            parts.append(f"conf_median={float(window.get('conf_median')):.3f}")
        if window.get("conf_max") is not None:
            parts.append(f"conf_max={float(window.get('conf_max')):.3f}")
        if center is not None:
            parts.append(f"matched_center_full_norm={center}")
        offset = self._target_center_offset_norm(obs)
        if offset is not None:
            parts.append(f"matched_center_offset_norm={offset}")
        parts.append(f"center_jitter={float(self.ctx.target_last_center_jitter):.3f}")
        parts.append(f"bbox_valid_ratio={float(window.get('bbox_valid_ratio', 0.0) or 0.0):.2f}")
        if self.ctx.target_last_lost_reason:
            parts.append(f"lost_reason={self.ctx.target_last_lost_reason}")
        if self.ctx.target_last_transition_reason:
            parts.append(f"decision={self.ctx.target_last_transition_reason}")
        return " ".join(parts)

    def _obs_has_motion(self, obs: Optional[TargetObs]) -> bool:
        if obs is None:
            return False
        return obs.vx_mps is not None or obs.vy_mps is not None or obs.wz_radps is not None

    def _edge_slide_direction(self) -> int:
        segment_s = max(0.2, float(self.cfg.edge_slide_segment_s))
        segment_index = int(self._state_elapsed() / segment_s)
        return self.ctx.slide_direction_sign if segment_index % 2 == 0 else -self.ctx.slide_direction_sign
    def _segment_elapsed(self, segment_s: float) -> float:
        segment_s = max(0.1, float(segment_s))
        return self._state_elapsed() % segment_s

    def _can_relocate_edge(self) -> bool:
        if not bool(self.cfg.edge_relocate_enabled):
            return False
        if self.ctx.edge_transition_count >= int(self.cfg.max_edge_transitions_per_task):
            return False
        return self.ctx.edge_visit_index + 1 < len(self.ctx.edge_visit_order)

    def _check_target_ready_for_grasp(self, obs: Optional[TargetObs]) -> bool:
        if obs is None:
            return False
        if not getattr(obs, "found", False):
            return False
        
        active_target = str(self.ctx.active_target or "").strip()
        matched_cls = str(obs.matched_cls or obs.target or "").strip()
        if active_target and matched_cls and matched_cls != active_target:
            return False
            
        conf = obs.matched_conf if obs.matched_conf is not None else 0.0
        if conf < 0.45:
            return False
            
        mb = obs.matched_bbox
        if not mb or not isinstance(mb, list) or len(mb) < 4:
            return False
                    
        if int(getattr(self.ctx, "target_found_frames", 0) or 0) < 5:
            return False
            
        return True
