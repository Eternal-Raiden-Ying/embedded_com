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
from ...utils.target_utils import resolve_target, target_to_class_id
from ..target_policy import route_locked_target
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
    def _target_control_observation_update(self, obs: Optional[TargetObs], candidate_ok: bool) -> Tuple[int, bool]:
        key = None
        if obs is not None:
            key = (
                getattr(obs, "obs_seq", None),
                getattr(obs, "frame_id", None),
                getattr(obs, "capture_mono_ns", None),
            )
        if (
            getattr(self.ctx, "target_control_last_obs_key", None) is None
            and getattr(self.ctx, "target_prewarm_last_obs_key", None) is not None
        ):
            self.ctx.target_control_stable_count = int(getattr(self.ctx, "target_prewarm_stable_count", 0) or 0)
            self.ctx.target_control_last_obs_key = getattr(self.ctx, "target_prewarm_last_obs_key", None)
        is_new = key is not None and key != getattr(self.ctx, "target_control_last_obs_key", None)
        if is_new:
            self.ctx.target_control_last_obs_key = key
            if candidate_ok:
                self.ctx.target_control_stable_count = int(self.ctx.target_control_stable_count or 0) + 1
            else:
                self.ctx.target_control_stable_count = 0
        return int(self.ctx.target_control_stable_count or 0), bool(is_new)

    def _tick_search_target_init(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        if getattr(self.ctx, "final_to_lateral_fast_start_ready", None) is False:
            if not self._final_forward_speed_near_zero():
                return self.controller.stop_cmd("SEARCH_TARGET_INIT")
            self.ctx.final_to_lateral_fast_start_ready = True
        target_obs, candidate_reason = self._select_active_target_obs(self._fresh_target_obs())
        candidate_ok, candidate_reason = self._target_candidate_status(
            target_obs,
            self.cfg.target_confirm_conf_th,
            min_area=self.cfg.target_confirm_min_bbox_area,
        )
        stable_count, _is_new_obs = self._target_control_observation_update(target_obs, candidate_ok)
        stable_required = max(1, int(getattr(self.cfg, "target_prewarm_stable_obs", 2) or 2))
        if candidate_ok and target_obs is not None and stable_count >= stable_required:
            self._log("info", "target_search_start_slide")
            conf = self._target_conf_value(target_obs)
            target_name = str(getattr(target_obs, "matched_cls", None) or getattr(target_obs, "target", "") or self.ctx.active_target or "")
            self._log("info", f"[TARGET_SEARCH][FAST_START] target={target_name} conf={conf if conf is not None else 'n/a'} reason=fresh_target_obs")
            self._log("info", "[TARGET_SEARCH][SKIP_ZERO] state=SEARCH_TARGET_INIT")
            self._transition(State.EDGE_SLIDE_SEARCH, self._format_target_transition_reason("target_found_start_lateral_align", target_obs))
            self._log("info", "[TARGET_SEARCH][FAST_START_TO_SLICE] fresh_target=1")
            align_decision = self._target_lateral_align_decision(target_obs, state="EDGE_SLIDE_SEARCH")
            if align_decision is not None:
                return align_decision
        self._log("info", "target_search_start_slide")
        self._transition(State.EDGE_SLIDE_SEARCH, f"target_search_init_internal_skip_zero reason={candidate_reason}")
        self._log("info", f"[TARGET_SEARCH][SKIP_ZERO] state=SEARCH_TARGET_INIT reason={candidate_reason}")
        return self._tick_edge_slide_search()

    def _tick_edge_slide_search(self) -> MotionDecision:
        self._maybe_resend_req(self._active_req_payload())
        target_obs, latch_reason = self._target_control_observation(self._fresh_target_obs())
        target_obs, select_reason = self._select_active_target_obs(target_obs)
        candidate_ok, candidate_reason = self._target_candidate_status(
            target_obs,
            self.cfg.target_confirm_conf_th,
            min_area=self.cfg.target_confirm_min_bbox_area,
        )
        if select_reason and select_reason != "single_candidate":
            candidate_reason = select_reason if not candidate_ok else candidate_reason
        if latch_reason and candidate_ok:
            candidate_reason = latch_reason
        stable_count, is_new_obs = self._target_control_observation_update(target_obs, candidate_ok)
        stable_required = max(1, int(getattr(self.cfg, "target_prewarm_stable_obs", 2) or 2))
        target_window = (
            self._record_target_window_sample(target_obs, candidate_reason)
            if is_new_obs
            else self._target_window_stats()
        )
        timeout_reason = ""
        if candidate_ok and target_obs is not None:
            if is_new_obs:
                self.ctx.target_found_frames += 1
            self.ctx.target_lost_frames = 0
            self._remember_good_target(target_obs, self.ctx.target_lateral_vy_cmd)
            if is_new_obs:
                self._update_target_stability(target_obs)
            if is_new_obs:
                self._record_target_lateral_good(target_obs, None)
            timeout_reason = self._target_lateral_timeout_reason(candidate_ok=True)
            if timeout_reason:
                return self._handle_edge_slide_target_timeout(
                    target_obs, target_window, candidate_reason, timeout_reason=timeout_reason
                )
            if stable_count < stable_required:
                align_decision = self._target_lateral_align_decision(target_obs, state="EDGE_SLIDE_SEARCH")
                if align_decision is not None:
                    return align_decision
                return self._annotate_target_lateral_decision(
                    self.controller.stop_cmd("EDGE_SLIDE_SEARCH"),
                    target_obs,
                    active=False,
                    reason="target_prewarm_confirming",
                    vy_cmd=0.0,
                )
            found_ratio_ok = (
                float(target_window.get("found_ratio", 0.0) or 0.0)
                >= float(getattr(self.cfg, "target_confirm_found_ratio_th", 0.5) or 0.5)
                and int(target_window.get("samples", 0) or 0) >= int(self.cfg.target_found_frames_to_confirm)
            )
            consecutive_ok = self.ctx.target_found_frames >= int(self.cfg.target_found_frames_to_confirm)
            centered_ok = self._target_lateral_centered(target_obs)
            stable_ok = int(self.ctx.target_lateral_stable_count) >= self._target_lateral_stable_frames()
            if (found_ratio_ok or consecutive_ok) and centered_ok and stable_ok:
                confirm_allowed, confirm_block_reason, progress = self._edge_slide_confirm_guard()
                if confirm_allowed:
                    self._log(
                        "info",
                        "[SLICE][CONFIRM_ALLOWED] "
                        f"elapsed={float(progress.get('edge_slide_elapsed_s', 0.0) or 0.0):.2f} "
                        f"lateral_dist={float(progress.get('edge_slide_lateral_distance_m', 0.0) or 0.0):.3f} "
                        f"frames={int(progress.get('edge_slide_frames', 0) or 0)}",
                    )
                    self.ctx.target_last_transition_reason = (
                        f"confirm_enter found_ratio={float(target_window.get('found_ratio', 0.0) or 0.0):.2f} "
                        f"consecutive_frames={int(self.ctx.target_found_frames)} bbox_valid={int(self._target_bbox_valid(target_obs))} "
                        f"target_lateral_stable_count={int(self.ctx.target_lateral_stable_count)} "
                        f"edge_slide_elapsed_s={float(progress.get('edge_slide_elapsed_s', 0.0) or 0.0):.2f} "
                        f"edge_slide_lateral_distance_m={float(progress.get('edge_slide_lateral_distance_m', 0.0) or 0.0):.3f} "
                        f"edge_slide_frames={int(progress.get('edge_slide_frames', 0) or 0)}"
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
                self.ctx.edge_slide_confirm_block_reason = confirm_block_reason
                self._log(
                    "info",
                    "[SLICE][MIN_GUARD] "
                    f"elapsed={float(progress.get('edge_slide_elapsed_s', 0.0) or 0.0):.2f} "
                    f"lateral_dist={float(progress.get('edge_slide_lateral_distance_m', 0.0) or 0.0):.3f} "
                    f"frames={int(progress.get('edge_slide_frames', 0) or 0)} "
                    "confirm_allowed=false",
                )
            align_decision = self._target_lateral_align_decision(target_obs, state="EDGE_SLIDE_SEARCH")
            if align_decision is not None:
                return align_decision
            if centered_ok:
                return self._annotate_target_lateral_decision(
                    self.controller.stop_cmd("EDGE_SLIDE_SEARCH"),
                    target_obs,
                    active=False,
                    reason="target_centered_wait_stable",
                    vy_cmd=0.0,
                )
        else:
            timeout_reason = self._target_lateral_timeout_reason(candidate_ok=False)
            if timeout_reason:
                return self._handle_edge_slide_target_timeout(
                    target_obs, target_window, candidate_reason, timeout_reason=timeout_reason
                )
            had_recent_target = bool(
                self.ctx.target_found_frames > 0
                or self.ctx.target_lateral_stable_count > 0
                or self.ctx.target_stable_since_mono > 0.0
                or self.ctx.last_good_target_mono > 0.0
            )
            self.ctx.target_found_frames = 0
            self.ctx.target_lateral_stable_count = 0
            self.ctx.target_last_lost_reason = candidate_reason
            hold_decision = self._target_lateral_hold_decision(state="EDGE_SLIDE_SEARCH", reason=candidate_reason)
            if hold_decision is not None:
                return hold_decision
            if had_recent_target:
                self._start_loss_timer("target_loss_since_mono")
                lost_s = self._loss_elapsed(self.ctx.target_loss_since_mono)
                lost_stop_s = float(getattr(self.cfg, "target_lateral_lost_stop_s", 1.2) or 1.2)
                if lost_s >= lost_stop_s:
                    self._log("warn", f"[SLICE][TARGET_LOST_STOP] lost_s={lost_s:.2f} reason=target_lost_timeout")
            else:
                self.ctx.target_loss_since_mono = 0.0
            self.ctx.target_lateral_vy_cmd = 0.0
            return self._annotate_target_lateral_decision(
                self.controller.stop_cmd("EDGE_SLIDE_SEARCH"),
                target_obs,
                active=False,
                reason="target_lost_timeout" if had_recent_target else (candidate_reason or "target_never_found"),
                vy_cmd=0.0,
            )
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

    def _handle_edge_slide_target_timeout(
        self,
        target_obs: Optional[TargetObs],
        target_window: Dict[str, Any],
        candidate_reason: str,
        *,
        timeout_reason: str = "",
    ) -> MotionDecision:
        reject_reason = str(timeout_reason or self._target_lateral_timeout_reason(candidate_ok=target_obs is not None))
        if not reject_reason:
            reject_reason = self._edge_slide_timeout_reason(target_obs, target_window, candidate_reason)
        now_mono = monotonic_ts()
        last_progress_mono = float(getattr(self.ctx, "target_lateral_last_progress_mono", 0.0) or 0.0)
        progress_age_s = max(0.0, now_mono - last_progress_mono) if last_progress_mono > 0.0 else None
        no_progress_timeout_s = max(
            0.1,
            float(
                getattr(
                    self.cfg,
                    "target_lateral_no_progress_timeout_s",
                    getattr(self.cfg, "target_search_timeout_s", 10.0),
                )
                or 10.0
            ),
        )
        progress_timeout_violation = bool(
            target_obs is not None
            and getattr(target_obs, "found", False)
            and progress_age_s is not None
            and progress_age_s < no_progress_timeout_s
            and reject_reason != "target_absolute_timeout"
        )
        if progress_timeout_violation:
            self._log(
                "error",
                "[TARGET_CONTROL_INVARIANT] TARGET_PROGRESS_BUT_TIMEOUT_TRIGGERED "
                f"reason={reject_reason} progress_age_s={progress_age_s:.3f}",
            )
        self.ctx.target_last_lost_reason = reject_reason
        self.ctx.target_lateral_timeout_type = reject_reason
        self.ctx.last_fail_reason = reject_reason
        min_abs_err = getattr(self.ctx, "target_lateral_min_abs_err_x", None)
        last_err = getattr(self.ctx, "target_lateral_last_err_x", None)
        timeout_s = float(self.cfg.target_search_timeout_s)
        self._log(
            "warn",
            "[SLICE][ALIGN_TIMEOUT] "
            f"min_abs_err={min_abs_err if min_abs_err is not None else 'n/a'} "
            f"last_err={last_err if last_err is not None else 'n/a'} "
            f"timeout_s={timeout_s:.1f} reason={reject_reason}",
        )
        if self._can_relocate_edge():
            self.ctx.advance_edge()
            self._transition(State.LEAVE_EDGE, f"{self.ctx.last_fail_reason}，切换到边 {self.ctx.current_edge_id}")
            self._queue_tts("当前边未找到目标，准备换边")
            decision = self.controller.leave_edge_cmd()
        elif bool(getattr(self.cfg, "multi_table_enabled", False)):
            self._transition(State.NEXT_TABLE, f"{self.ctx.last_fail_reason}，准备切换下一张桌")
            self._queue_tts("当前桌位未找到目标，尝试下一张桌")
            decision = self.controller.next_table_cmd(turn_sign=self.ctx.relocate_turn_sign)
        else:
            self._enter_error_recovery(reject_reason)
            decision = self.controller.stop_cmd("ERROR_RECOVERY", brake=True)
            decision.control_summary.update(
                {"control_source": "search_failed_stop", "multi_table_enabled": False}
            )
        if decision.control_summary is not None:
            decision.control_summary.update(
                {
                    "target_search_reject_reason": reject_reason,
                    "target_search_timeout": True,
                    "slice_timeout_reason": reject_reason,
                    "target_lateral_min_abs_err": min_abs_err,
                    "target_lateral_last_err": last_err,
                    "found_ratio": float(target_window.get("found_ratio", 0.0) or 0.0),
                    "stable_count": int(self.ctx.target_found_frames),
                    "lateral_stable_count": int(self.ctx.target_lateral_stable_count),
                    "center_jitter": float(target_window.get("center_jitter", 0.0) or 0.0),
                    "target_progress_age_s": progress_age_s,
                    "control_invariant_violation": (
                        "TARGET_PROGRESS_BUT_TIMEOUT_TRIGGERED" if progress_timeout_violation else ""
                    ),
                }
            )
        return decision

    def _target_lateral_timeout_reason(self, *, candidate_ok: bool) -> str:
        """Use progress/loss/UART clocks instead of the short state wall clock."""
        now = monotonic_ts()
        elapsed = max(0.0, self._state_elapsed())
        normal_timeout = max(0.1, float(getattr(self.cfg, "target_search_timeout_s", 10.0) or 10.0))
        absolute_timeout = max(
            normal_timeout,
            float(getattr(self.cfg, "target_search_absolute_timeout_s", 60.0) or 60.0),
        )
        if elapsed >= absolute_timeout:
            return "target_absolute_timeout"

        last_good = float(getattr(self.ctx, "target_lateral_last_good_obs_mono", 0.0) or 0.0)
        had_target = bool(last_good > 0.0 or getattr(self.ctx, "target_lateral_min_abs_err_x", None) is not None)
        if not had_target:
            return "target_never_found_timeout" if elapsed >= normal_timeout else ""
        if not candidate_ok:
            lost_timeout = max(0.0, float(getattr(self.cfg, "target_lateral_lost_stop_s", 1.2) or 1.2))
            if last_good > 0.0 and now - last_good >= lost_timeout:
                return "target_lost_timeout"
            return ""

        last_progress = float(getattr(self.ctx, "target_lateral_last_progress_mono", 0.0) or 0.0)
        if last_progress <= 0.0:
            self.ctx.target_lateral_last_progress_mono = now
            last_progress = now
        no_progress_timeout = max(
            0.1,
            float(getattr(self.cfg, "target_lateral_no_progress_timeout_s", normal_timeout) or normal_timeout),
        )
        if now - last_progress >= no_progress_timeout:
            return "target_lateral_no_progress_timeout"

        last_cmd = float(getattr(self.ctx, "target_lateral_last_cmd_mono", 0.0) or 0.0)
        last_accept = float(getattr(self.ctx, "target_lateral_last_uart_accept_mono", 0.0) or 0.0)
        uart_timeout = max(0.1, float(getattr(self.cfg, "target_lateral_uart_no_progress_timeout_s", 3.0) or 3.0))
        if last_cmd > 0.0 and now - last_cmd >= uart_timeout and last_accept < last_cmd:
            return "target_uart_no_progress_timeout"
        return ""

    def _edge_slide_timeout_reason(
        self,
        target_obs: Optional[TargetObs],
        target_window: Dict[str, Any],
        candidate_reason: str,
    ) -> str:
        del target_obs
        samples = int(target_window.get("samples", 0) or 0)
        found_ratio = float(target_window.get("found_ratio", 0.0) or 0.0)
        had_target = bool(
            samples > 0
            and (
                found_ratio > 0.0
                or getattr(self.ctx, "target_lateral_last_good_obs_mono", 0.0) > 0.0
                or getattr(self.ctx, "target_lateral_min_abs_err_x", None) is not None
            )
        )
        if not had_target:
            return "target_never_found_timeout"
        lost_s = self._loss_elapsed(self.ctx.target_loss_since_mono) if self.ctx.target_loss_since_mono > 0.0 else 0.0
        lost_stop_s = max(0.0, float(getattr(self.cfg, "target_lateral_lost_stop_s", 1.2) or 1.2))
        if lost_s >= lost_stop_s and str(candidate_reason or "").startswith("target"):
            return "target_lost_timeout"
        if int(self.ctx.target_lateral_stable_count) < self._target_lateral_stable_frames():
            return "target_lateral_align_timeout"
        return "target_confirm_timeout"

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
        obs, visible_reason = self._select_active_target_obs(self._fresh_target_obs())
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
            self._remember_good_target(obs, self.ctx.target_lateral_vy_cmd)
            center_jitter = self._update_target_stability(obs)
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

            # TARGET_CONFIRM uses the wider center_x_tol as the semantic "centered enough"
            # condition.  Do not keep issuing tiny lateral-align commands merely because the
            # target is outside the narrower deadband; otherwise the early align return can
            # starve both lock evaluation and confirm-timeout handling.
            if centered_ok:
                self.ctx.target_lateral_stable_count += 1
                self.ctx.target_lateral_vy_cmd = 0.0
                self.ctx.target_lateral_align_reason = "target_center_tol"
            else:
                self.ctx.target_lateral_stable_count = 0
            lateral_stable_ok = int(self.ctx.target_lateral_stable_count) >= self._target_lateral_stable_frames()

            err_x = self._target_lateral_error_x(obs)
            deadband = self._target_deadband_x()
            tol = abs(float(getattr(self.cfg, "target_lateral_align_center_x_tol", 0.06) or 0.06))
            blockers = []
            if not lock_ok:
                blockers.append(lock_reason)
            if not confirm_min_ok:
                blockers.append("confirm_min_not_reached")
            if not stable_ok:
                blockers.append("target_stable_not_enough")
            if not jitter_ok:
                blockers.append("center_jitter_high")
            if not ratio_ok:
                blockers.append(f"found_ratio_low ratio={found_ratio:.2f}")
            if not conf_ok:
                blockers.append("conf_median_low")
            if not centered_ok:
                blockers.append("target_not_centered")
            if not lateral_stable_ok:
                blockers.append("target_lateral_stable_not_enough")

            debug_fields = {
                "target_confirm_elapsed_s": float(confirm_elapsed_s),
                "target_confirm_err_x": err_x,
                "target_confirm_deadband": float(deadband),
                "target_confirm_center_tol": float(tol),
                "target_confirm_centered_ok": bool(centered_ok),
                "target_confirm_lock_ok": bool(lock_ok),
                "target_confirm_lock_reason": str(lock_reason or ""),
                "target_confirm_min_ok": bool(confirm_min_ok),
                "target_confirm_stable_ok": bool(stable_ok),
                "target_confirm_jitter_ok": bool(jitter_ok),
                "target_confirm_ratio_ok": bool(ratio_ok),
                "target_confirm_conf_ok": bool(conf_ok),
                "target_confirm_lateral_stable_ok": bool(lateral_stable_ok),
                "target_confirm_blockers": list(blockers),
                "target_lateral_stable_count": int(self.ctx.target_lateral_stable_count),
                "target_lateral_stable_frames": int(self._target_lateral_stable_frames()),
                "target_confirm_found_ratio": float(found_ratio),
                "target_confirm_conf_median": conf_median,
                "target_confirm_center_jitter": float(center_jitter),
            }

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
                decision = self._annotate_target_lateral_decision(
                    self.controller.stop_cmd("TARGET_LOCKED"),
                    obs,
                    active=False,
                    reason="target_lateral_centered_locked",
                    vy_cmd=0.0,
                )
                if decision.control_summary is not None:
                    decision.control_summary.update(debug_fields)
                return decision

            # Confirm timeout must be checked before any lateral-align early return.  If the
            # target cannot satisfy the lock conditions in time, return to slide search rather
            # than staying forever in TARGET_CONFIRM.
            if confirm_elapsed_s >= float(self.cfg.target_confirm_timeout_s):
                self.ctx.target_last_lost_reason = ",".join(blockers) or "lock_condition_timeout"
                self._transition(
                    State.EDGE_SLIDE_SEARCH,
                    self._format_target_transition_reason("confirm_timeout", obs),
                )
                decision = self._annotate_target_lateral_decision(
                    self.controller.stop_cmd("EDGE_SLIDE_SEARCH"),
                    obs,
                    active=False,
                    reason=f"target_confirm_timeout blockers={self.ctx.target_last_lost_reason}",
                    vy_cmd=0.0,
                )
                if decision.control_summary is not None:
                    decision.control_summary.update(debug_fields)
                return decision

            # Only request lateral motion in TARGET_CONFIRM when the target is outside the
            # wider tolerance band.  Inside tolerance, allow the lock/stability conditions to
            # settle instead of starving them with micro-align commands.
            if not centered_ok:
                align_decision = self._target_lateral_align_decision(obs, state="TARGET_CONFIRM")
                if align_decision is not None:
                    if align_decision.control_summary is not None:
                        align_decision.control_summary.update(debug_fields)
                        align_decision.control_summary["target_confirm_align_allowed"] = True
                    return align_decision

            if centered_ok and not lateral_stable_ok:
                decision = self._annotate_target_lateral_decision(
                    self.controller.stop_cmd("TARGET_CONFIRM"),
                    obs,
                    active=False,
                    reason="target_centered_wait_stable",
                    vy_cmd=0.0,
                )
                if decision.control_summary is not None:
                    decision.control_summary.update(debug_fields)
                return decision

            decision = self._annotate_target_lateral_decision(
                self.controller.stop_cmd("TARGET_CONFIRM"),
                obs,
                active=False,
                reason="target_confirm_hold",
                vy_cmd=0.0,
            )
            if decision.control_summary is not None:
                decision.control_summary.update(debug_fields)
            return decision
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
        elapsed_s = self._state_elapsed()
        settle_s = float(getattr(self.cfg, "target_lock_settle_s", 0.5) or 0.5)
        stuck_guard_s = 3.0
        if elapsed_s >= settle_s:
            reason = "target_locked_stuck_guard" if elapsed_s >= stuck_guard_s else "target_locked_settle_done"
            if elapsed_s >= stuck_guard_s:
                self._log("warn", f"target_locked_stuck_guard elapsed_s={elapsed_s:.2f} forcing FREEZE_BASE")
            else:
                self._log("info", f"target_locked_settle_done elapsed_s={elapsed_s:.2f} settle_s={settle_s:.2f}")
            self._transition(State.FREEZE_BASE, reason)
            self._log("info", "target_locked_to_freeze_base")
            decision = self.controller.stop_cmd("FREEZE_BASE")
            if decision.control_summary is not None:
                decision.control_summary.update(
                    {
                        "target_locked_freeze_elapsed_s": float(elapsed_s),
                        "target_locked_freeze_after_s": float(settle_s),
                        "target_ready_for_grasp": True,
                        "target_ready_for_grasp_reason": reason,
                        "target_lock_freeze_blockers": [],
                    }
                )
            return decision
        obs = self._fresh_target_obs()
        lock_ok, lock_reason = self._target_candidate_status(obs, self.cfg.target_lock_conf_th, min_area=0.0)
        target_window = self._record_target_window_sample(obs, lock_reason)
        if not lock_ok or obs is None:
            self.ctx.target_lost_frames += 1
            self.ctx.target_locked = False
            self._start_loss_timer("target_loss_since_mono")
            lost_s = self._loss_elapsed(self.ctx.target_loss_since_mono)
            self.ctx.target_last_lost_reason = f"{lock_reason} lost_hold_ms={int(round(lost_s * 1000.0))}"
            freeze_fields = {
                "target_locked_freeze_elapsed_s": float(self._state_elapsed()),
                "target_locked_freeze_after_s": float(self.cfg.target_locked_freeze_after_s),
                "target_lock_ok": bool(lock_ok),
                "target_lock_reason": str(lock_reason or "target_obs_missing"),
                "target_centered_ok": False,
                "target_lateral_stable_ok": False,
                "target_lateral_stable_count": int(self.ctx.target_lateral_stable_count),
                "target_lateral_stable_frames": int(self._target_lateral_stable_frames()),
                "target_ready_for_grasp": False,
                "target_ready_for_grasp_reason": "target_lock_not_ok",
                "target_lock_stable_ok": False,
                "target_lock_jitter_ok": False,
                "target_lock_conf_stable": False,
                "target_lock_ratio_ok": False,
                "target_lock_freeze_blockers": ["target_lock_not_ok"],
            }
            if lost_s >= float(self.cfg.target_lock_lost_hold_s):
                self._reset_target_stability(lock_reason)
                self._transition(
                    State.EDGE_SLIDE_SEARCH,
                    self._format_target_transition_reason("locked_lost_hold_exceeded", obs),
                )
                decision = self._annotate_target_lateral_decision(
                    self.controller.stop_cmd("EDGE_SLIDE_SEARCH"),
                    obs,
                    active=False,
                    reason="target_locked_lost_return_search",
                    vy_cmd=0.0,
                )
                if decision.control_summary is not None:
                    decision.control_summary.update(freeze_fields)
                return decision
            decision = self._annotate_target_lateral_decision(
                self.controller.stop_cmd("TARGET_LOCKED"),
                obs,
                active=False,
                reason="target_locked_lost_hold",
                vy_cmd=0.0,
            )
            if decision.control_summary is not None:
                decision.control_summary.update(freeze_fields)
            return decision
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
        if centered_ok:
            self.ctx.target_lateral_stable_count += 1
        else:
            self.ctx.target_lateral_stable_count = 0
        lateral_stable_ok = int(self.ctx.target_lateral_stable_count) >= self._target_lateral_stable_frames()
        ready_for_grasp, ready_reason = self._target_ready_for_grasp_status(obs)
        bbox_valid = self._target_bbox_valid(obs)
        cls_ok = self._target_cls_matches_active(obs)
        found_ok = bool(getattr(obs, "found", False))
        freeze_after_s = float(getattr(self.cfg, "target_locked_freeze_after_s", settle_s) or settle_s)
        freeze_blockers = []
        if elapsed_s < freeze_after_s:
            freeze_blockers.append("freeze_settle_not_elapsed")
        if not lock_ok:
            freeze_blockers.append("target_lock_not_ok")
        if not centered_ok:
            freeze_blockers.append("target_not_centered")
        if not bbox_valid:
            freeze_blockers.append("target_bbox_invalid")
        if not found_ok:
            freeze_blockers.append("target_not_found")
        if not cls_ok:
            freeze_blockers.append("target_cls_mismatch")
        if not lateral_stable_ok:
            freeze_blockers.append("target_lateral_stable_not_enough")
        if not ready_for_grasp:
            freeze_blockers.append(f"ready_for_grasp:{ready_reason}")

        freeze_fields = {
            "target_locked_freeze_elapsed_s": float(elapsed_s),
            "target_locked_freeze_after_s": float(freeze_after_s),
            "target_lock_ok": bool(lock_ok),
            "target_lock_reason": str(lock_reason or ""),
            "target_centered_ok": bool(centered_ok),
            "target_lateral_stable_ok": bool(lateral_stable_ok),
            "target_lateral_stable_count": int(self.ctx.target_lateral_stable_count),
            "target_lateral_stable_frames": int(self._target_lateral_stable_frames()),
            "target_ready_for_grasp": bool(ready_for_grasp),
            "target_ready_for_grasp_reason": str(ready_reason or ""),
            "target_lock_stable_ok": bool(stable_ok),
            "target_lock_jitter_ok": bool(jitter_ok),
            "target_lock_conf_stable": bool(conf_stable),
            "target_lock_ratio_ok": bool(ratio_ok),
            "target_lock_freeze_blockers": list(freeze_blockers),
        }
        if not freeze_blockers:
            self.ctx.target_last_transition_reason = (
                f"freeze_ok found_ratio={found_ratio:.2f} conf_median={float(conf_median):.3f} "
                f"center_jitter={float(center_jitter):.3f} stable_ms={self._target_stable_ms()} "
                f"target_lateral_stable_count={int(self.ctx.target_lateral_stable_count)}"
            )
            self._transition(
                State.FREEZE_BASE,
                self._format_target_transition_reason("locked_stable_freeze", obs),
            )
            decision = self._annotate_target_lateral_decision(
                self.controller.stop_cmd("FREEZE_BASE"),
                obs,
                active=False,
                reason="target_locked_freeze_base",
                vy_cmd=0.0,
            )
            if decision.control_summary is not None:
                decision.control_summary.update(freeze_fields)
            return decision
        decision = self._annotate_target_lateral_decision(
            self.controller.stop_cmd("TARGET_LOCKED"),
            obs,
            active=False,
            reason="target_locked_hold",
            vy_cmd=0.0,
        )
        if decision.control_summary is not None:
            decision.control_summary.update(freeze_fields)
        return decision

    def _tick_freeze_base(self) -> MotionDecision:
        if self._state_elapsed() < float(self.cfg.freeze_settle_s):
            return self.controller.stop_cmd("FREEZE_BASE")
        spec = resolve_target(self.ctx.canonical_target or self.ctx.active_target or "")
        if spec is None:
            self._transition(State.DONE, "target_catalog_missing_after_lock")
            return self.controller.stop_cmd("DONE")
        route = route_locked_target(spec)
        if route.tts_event:
            self._emit_tts_event(route.tts_event, state=State.FREEZE_BASE.value)
        next_state = State(route.next_state)
        self._transition(next_state, f"target_policy={spec.action_policy} target={spec.canonical_name}")
        return self.controller.stop_cmd(next_state.value)

    def _target_matches_active(self, obs: TargetObs) -> bool:
        if not self.ctx.active_target:
            return True
        candidate = str(obs.canonical_target or obs.target or "").strip()
        if not candidate:
            return True
        spec = resolve_target(candidate)
        candidate_canonical = spec.canonical_target if spec is not None else candidate
        return candidate_canonical == str(self.ctx.canonical_target or self.ctx.active_target).strip()

    def _target_cls_matches_active(self, obs: TargetObs) -> bool:
        active = str(self.ctx.active_target or "").strip()
        if not active:
            return True
        expected_id = self.ctx.class_id
        matched_id = getattr(obs, "matched_class_id", None)
        if expected_id is not None and matched_id is not None:
            try:
                return int(expected_id) == int(matched_id)
            except Exception:
                pass
        expected_name = str(self.ctx.class_name or "").strip()
        candidate_cls = str(obs.matched_cls or obs.target or "").strip()
        if not candidate_cls:
            return True
        spec = resolve_target(candidate_cls)
        if spec is not None:
            return spec.canonical_target == str(self.ctx.canonical_target or active).strip()
        return bool(expected_name and candidate_cls == expected_name) or candidate_cls == active

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

    def _target_deadband_x(self) -> float:
        tol = abs(float(getattr(self.cfg, "target_lateral_align_center_x_tol", 0.06) or 0.06))
        return max(0.0, 0.5 * tol)

    def _target_candidate_list(self, obs: Optional[TargetObs]) -> List[Dict[str, Any]]:
        if obs is None:
            return []
        raw = getattr(obs, "target_candidates", None)
        if not isinstance(raw, list) or not raw:
            return []
        return [dict(item) for item in raw if isinstance(item, dict)]

    def _candidate_class_matches_active(self, cand: Dict[str, Any]) -> bool:
        active = str(self.ctx.active_target or "").strip()
        if not active:
            return True
        expected_id = self.ctx.class_id
        for key in ("matched_class_id", "class_id", "target_class_id"):
            if expected_id is not None and cand.get(key) is not None:
                try:
                    if int(expected_id) == int(cand.get(key)):
                        return True
                except Exception:
                    pass
        expected = str(self.ctx.canonical_target or active).strip()
        for key in ("canonical_target", "matched_cls", "class_name", "target", "label", "cls"):
            text = str(cand.get(key) or "").strip()
            if not text:
                continue
            spec = resolve_target(text)
            canonical = spec.canonical_target if spec is not None else text
            if canonical == expected or text == str(self.ctx.class_name or "").strip() or text == active:
                return True
        return False

    def _candidate_bbox(self, cand: Dict[str, Any]) -> Optional[List[float]]:
        bbox = cand.get("matched_bbox") or cand.get("bbox") or cand.get("box")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None
        vals = [self._float_or_none(item) for item in bbox[:4]]
        if not all(item is not None for item in vals):
            return None
        return [float(item) for item in vals]  # type: ignore[arg-type]

    def _candidate_center_x(self, cand: Dict[str, Any]) -> Optional[float]:
        center = cand.get("matched_center_full_norm") or cand.get("matched_center") or cand.get("center")
        if isinstance(center, dict):
            cx = self._float_or_none(center.get("cx", center.get("x_norm")))
            if cx is not None:
                return max(0.0, min(1.0, float(cx)))
        for key in ("cx_norm", "x_norm", "cx"):
            cx = self._float_or_none(cand.get(key))
            if cx is not None:
                return max(0.0, min(1.0, float(cx)))
        bbox = self._candidate_bbox(cand)
        if bbox is None:
            return None
        x0, _, x1, _ = bbox
        if max(abs(x0), abs(x1)) > 1.5:
            return None
        return max(0.0, min(1.0, 0.5 * (x0 + x1)))

    def _candidate_area(self, cand: Dict[str, Any]) -> Optional[float]:
        for key in ("matched_area", "area_norm", "size_norm", "mask_area_ratio"):
            value = self._float_or_none(cand.get(key))
            if value is not None and value > 0.0:
                return float(value)
        bbox = self._candidate_bbox(cand)
        if bbox is None:
            return None
        x0, y0, x1, y1 = bbox
        width = abs(x1 - x0)
        height = abs(y1 - y0)
        if width > 1.0 or height > 1.0:
            return None
        return max(0.0, width * height)

    def _candidate_conf(self, cand: Dict[str, Any]) -> Optional[float]:
        for key in ("matched_conf", "confidence", "conf", "score"):
            value = self._float_or_none(cand.get(key))
            if value is not None:
                return max(0.0, min(1.0, float(value)))
        return None

    def _candidate_execute_score(self, cand: Dict[str, Any]) -> Tuple[float, str]:
        cx = self._candidate_center_x(cand)
        conf = self._candidate_conf(cand)
        area = self._candidate_area(cand)
        target_x = max(0.0, min(1.0, float(getattr(self.cfg, "target_lateral_align_center_x_target", 0.40) or 0.40)))
        center_score = 0.0 if cx is None else max(0.0, 1.0 - abs(float(cx) - target_x))
        if area is None:
            area_score = 0.4
            edge_penalty = 0.2
        else:
            area_score = max(0.0, min(1.0, area / 0.20))
            edge_penalty = 0.0 if 0.002 <= area <= 0.60 else 0.35
        bbox = self._candidate_bbox(cand)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            if min(x0, y0) <= 0.01 or max(x1, y1) >= 0.99:
                edge_penalty += 0.20
        track_score = 0.0
        last = self.ctx.last_good_target_obs
        last_cx = self._target_lateral_center_x(last) if last is not None else None
        if cx is not None and last_cx is not None:
            track_score = max(0.0, 1.0 - abs(float(cx) - float(last_cx)) * 2.0)
        score = 0.45 * float(conf or 0.0) + 0.15 * area_score + 0.25 * center_score + 0.15 * track_score - edge_penalty
        return float(score), "score"

    def _apply_candidate_to_obs(self, obs: TargetObs, cand: Dict[str, Any], idx: int, score: float, reason: str) -> TargetObs:
        obs.found = True
        obs.target_found = True
        obs.matched_cls = str(cand.get("matched_cls") or cand.get("class_name") or cand.get("target") or cand.get("label") or obs.matched_cls or obs.target or "")
        obs.target = str(cand.get("target") or obs.target or obs.matched_cls or "")
        spec = resolve_target(obs.matched_cls or obs.target or "")
        if spec is not None:
            obs.canonical_target = spec.canonical_target
        obs.matched_class_id = self._float_or_none(cand.get("matched_class_id") or cand.get("class_id") or cand.get("target_class_id"))
        if obs.matched_class_id is not None:
            try:
                obs.matched_class_id = int(obs.matched_class_id)
            except Exception:
                pass
        obs.matched_conf = self._candidate_conf(cand)
        bbox = self._candidate_bbox(cand)
        if bbox is not None:
            obs.matched_bbox = list(bbox)
            obs.bbox = list(bbox)
        cx = self._candidate_center_x(cand)
        if cx is not None:
            center = dict(obs.matched_center_full_norm or {})
            center["cx"] = float(cx)
            if cand.get("cy_norm") is not None:
                center["cy"] = self._float_or_none(cand.get("cy_norm"))
            obs.matched_center_full_norm = center
            obs.cx_norm = float(cx)
        area = self._candidate_area(cand)
        if area is not None:
            obs.matched_area = float(area)
            obs.size_norm = float(area)
        obs.matched_rank_in_all_boxes = idx
        self.ctx.selected_candidate_idx = int(idx)
        self.ctx.selected_candidate_score = float(score)
        self.ctx.selected_candidate_reason = str(reason or "score")
        return obs

    def _select_active_target_obs(self, obs: Optional[TargetObs]) -> Tuple[Optional[TargetObs], str]:
        candidates = self._target_candidate_list(obs)
        if obs is None:
            self.ctx.selected_candidate_idx = None
            self.ctx.selected_candidate_score = None
            self.ctx.selected_candidate_reason = "target_missing"
            return None, "target_missing"
        if not candidates:
            self.ctx.selected_candidate_idx = 0 if bool(getattr(obs, "found", False)) else None
            self.ctx.selected_candidate_score = None
            self.ctx.selected_candidate_reason = "single_candidate"
            return obs, "single_candidate"
        scored: List[Tuple[float, int, Dict[str, Any], str]] = []
        for idx, cand in enumerate(candidates):
            if not self._candidate_class_matches_active(cand):
                self._log("info", f"[SLICE][CANDIDATE_REJECT] reason=class_mismatch idx={idx}")
                continue
            cx = self._candidate_center_x(cand)
            if cx is None:
                self._log("info", f"[SLICE][CANDIDATE_REJECT] reason=bad_bbox idx={idx}")
                continue
            score, reason = self._candidate_execute_score(cand)
            scored.append((score, idx, cand, reason))
        if not scored:
            self.ctx.selected_candidate_idx = None
            self.ctx.selected_candidate_score = None
            self.ctx.selected_candidate_reason = "no_matching_candidate"
            return None, "class_mismatch"
        score, idx, cand, reason = max(scored, key=lambda item: item[0])
        selected = self._apply_candidate_to_obs(obs, cand, idx, score, reason)
        conf = self._candidate_conf(cand)
        cx = self._candidate_center_x(cand)
        self._log(
            "info",
            "[SLICE][CANDIDATE_SELECT] "
            f"count={len(candidates)} selected_idx={idx} selected_score={score:.3f} "
            f"selected_conf={conf if conf is not None else 'n/a'} selected_cx={cx if cx is not None else 'n/a'}",
        )
        return selected, "candidate_selected"

    def _remember_good_target(self, obs: Optional[TargetObs], vy_cmd: float) -> None:
        if obs is None or not bool(getattr(obs, "found", False)):
            return
        now_m = monotonic_ts()
        obs_mono = (
            float(self.ctx.last_valid_target_obs_mono)
            if obs is getattr(self.ctx, "last_valid_target_obs", None)
            and float(getattr(self.ctx, "last_valid_target_obs_mono", 0.0) or 0.0) > 0.0
            else now_m
        )
        self.ctx.last_good_target_obs = obs
        self.ctx.last_good_target_mono = obs_mono
        self.ctx.last_good_vy_mps = float(vy_cmd)
        self.ctx.target_control_latch_active = True
        self.ctx.target_lateral_hold_active = False
        self.ctx.target_lateral_hold_source = "current"

    def _target_control_observation(
        self, obs: Optional[TargetObs]
    ) -> Tuple[Optional[TargetObs], str]:
        if (
            obs is not None
            and bool(getattr(obs, "found", False))
            and getattr(obs, "inference_completed", None) is not False
        ):
            return obs, "current_completed_inference"
        if not bool(getattr(self.ctx, "target_control_latch_active", False)):
            return obs, ""
        held = getattr(self.ctx, "last_valid_target_obs", None)
        held_mono = float(getattr(self.ctx, "last_valid_target_obs_mono", 0.0) or 0.0)
        ttl_s = max(0.0, float(getattr(self.cfg, "target_lateral_lost_stop_s", 1.2) or 1.2))
        negative_limit = max(1, int(getattr(self.cfg, "target_confirm_lost_frames", 2) or 2))
        age_s = max(0.0, monotonic_ts() - held_mono) if held_mono > 0.0 else float("inf")
        if held is None or age_s > ttl_s or int(self.ctx.target_explicit_negative_count) >= negative_limit:
            self.ctx.target_control_latch_active = False
            return obs, ""
        return held, "latched_completed_inference"

    def _target_lateral_hold_decision(self, *, state: str, reason: str) -> Optional[MotionDecision]:
        if not bool(getattr(self.cfg, "target_lateral_hold_enable", True)):
            return None
        if self.ctx.last_good_target_mono <= 0.0:
            return None
        age_s = max(0.0, monotonic_ts() - float(self.ctx.last_good_target_mono))
        hold_s = max(0.0, float(getattr(self.cfg, "target_lateral_hold_s", 0.8) or 0.8))
        lost_stop_s = max(hold_s, float(getattr(self.cfg, "target_lateral_lost_stop_s", 1.2) or 1.2))
        if age_s > lost_stop_s:
            self.ctx.target_lateral_hold_active = False
            self.ctx.target_lateral_hold_source = "stop"
            return None
        if age_s > hold_s:
            return None
        vy = float(self.ctx.last_good_vy_mps or 0.0)
        min_vy = abs(float(getattr(self.cfg, "target_lateral_min_vy_mps", 0.016) or 0.016))
        if abs(vy) < min_vy:
            last_obs = self.ctx.last_good_target_obs
            err = self._target_lateral_error_x(last_obs)
            if err is None or abs(float(err)) <= self._target_deadband_x():
                vy = 0.0
            else:
                vy = min_vy if -float(err) >= 0.0 else -min_vy
        self.ctx.target_lateral_hold_active = True
        self.ctx.target_lateral_hold_source = "hold"
        self.ctx.target_lateral_vy_cmd = float(vy)
        cmd = self.controller._cmd(state, vx=0.0, vy=vy, wz=0.0)
        decision = MotionDecision(
            cmd=cmd,
            control_summary=self.controller._summary(state, cmd, reason="target_lateral_hold"),
        )
        self._log(
            "info",
            f"[SLICE][TARGET_HOLD] age_s={age_s:.2f} vy={vy:.3f} reason={reason or 'target_missing'}",
        )
        return self._annotate_target_lateral_decision(
            decision,
            self.ctx.last_good_target_obs,
            active=bool(abs(vy) > 1e-9),
            reason=f"target_lateral_hold lost_age_s={age_s:.2f}",
            vy_cmd=vy,
        )

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
        target = max(0.0, min(1.0, float(getattr(self.cfg, "target_lateral_align_center_x_target", 0.40) or 0.40)))
        # target_err_x = target_center_x_norm - target_lateral_align_center_x_target
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
        lateral_cmd_source: str = "current",
        hold_active: bool = False,
        hold_age_s: Optional[float] = None,
        slice_timeout_reason: str = "",
        target_found_override: Optional[bool] = None,
    ) -> MotionDecision:
        summary = dict(decision.control_summary or {})
        conf = self._target_conf_value(obs) if obs is not None else None
        cx = self._target_lateral_center_x(obs)
        err = self._target_lateral_error_x(obs)
        target_x = max(0.0, min(1.0, float(getattr(self.cfg, "target_lateral_align_center_x_target", 0.40) or 0.40)))
        centered_ok = bool(self._target_lateral_centered(obs))
        last_good_age_s = self._target_lateral_last_good_age_s()
        edge_slide_progress = self._edge_slide_progress_snapshot()
        target_found = bool(obs is not None and getattr(obs, "found", False)) if target_found_override is None else bool(target_found_override)
        summary.update(
            {
                "target_found": bool(target_found),
                "raw_target": str(getattr(obs, "raw_target", "") or self.ctx.raw_target or "") if obs is not None else self.ctx.raw_target,
                "canonical_target": str(getattr(obs, "canonical_target", "") or self.ctx.canonical_target or "") if obs is not None else self.ctx.canonical_target,
                "expected_class_name": str(getattr(obs, "expected_class_name", "") or self.ctx.class_name or "") if obs is not None else self.ctx.class_name,
                "expected_class_id": getattr(obs, "expected_class_id", None) if obs is not None else self.ctx.class_id,
                "target_cls": str(getattr(obs, "matched_cls", None) or getattr(obs, "target", "") or "") if obs is not None else "",
                "matched_cls": str(getattr(obs, "matched_cls", None) or "") if obs is not None else "",
                "matched_class_id": getattr(obs, "matched_class_id", None) if obs is not None else None,
                "best_cls": str(getattr(obs, "best_cls", "") or "") if obs is not None else "",
                "best_conf": getattr(obs, "best_conf", None) if obs is not None else None,
                "target_conf": conf,
                "target_center_x_norm": cx,
                "target_lateral_align_center_x_target": float(target_x),
                "target_err_x": err,
                "target_lateral_centered_ok": bool(centered_ok),
                "target_lateral_align_active": bool(active),
                "target_lateral_align_reason": str(reason or ""),
                "target_lateral_hold_active": bool(hold_active),
                "target_lateral_hold_age_s": hold_age_s,
                "last_good_target_age_s": last_good_age_s,
                "last_good_vy_mps": float(getattr(self.ctx, "target_lateral_last_good_vy_mps", 0.0) or 0.0),
                "lateral_cmd_source": str(lateral_cmd_source or "current"),
                "slice_timeout_reason": str(slice_timeout_reason or ""),
                "edge_slide_elapsed_s": float(edge_slide_progress.get("edge_slide_elapsed_s", 0.0) or 0.0),
                "edge_slide_lateral_distance_m": float(edge_slide_progress.get("edge_slide_lateral_distance_m", 0.0) or 0.0),
                "edge_slide_frames": int(edge_slide_progress.get("edge_slide_frames", 0) or 0),
                "confirm_block_reason": str(getattr(self.ctx, "edge_slide_confirm_block_reason", "") or ""),
                "target_search_reject_reason": self._target_search_reject_reason(
                    obs,
                    self._target_window_stats(),
                    str(reason or ""),
                ),
                "centered_ok": bool(centered_ok),
                "bbox_valid": bool(self._target_bbox_valid(obs)),
                "stable_count": int(self.ctx.target_found_frames),
                "found_ratio": float(self._target_window_stats().get("found_ratio", 0.0) or 0.0),
                "center_jitter": float(self.ctx.target_last_center_jitter),
                "target_lateral_vy_cmd": float(vy_cmd),
                "lateral_owner": "target_lateral" if bool(getattr(self.ctx, "target_control_latch_active", False)) else "none",
                "yaw_owner": "none",
                "forward_owner": "none",
                "docking_action": "TARGET_LATERAL_ALIGN" if abs(float(vy_cmd)) > 1e-9 else "TARGET_CONFIRM_HOLD",
                "target_lateral_min_abs_err": getattr(self.ctx, "target_lateral_min_abs_err_x", None),
                "target_lateral_last_err": getattr(self.ctx, "target_lateral_last_err_x", None),
                "target_lateral_last_progress_age_s": (
                    max(0.0, monotonic_ts() - float(self.ctx.target_lateral_last_progress_mono))
                    if float(getattr(self.ctx, "target_lateral_last_progress_mono", 0.0) or 0.0) > 0.0 else None
                ),
                "target_lateral_last_uart_accept_age_s": (
                    max(0.0, monotonic_ts() - float(self.ctx.target_lateral_last_uart_accept_mono))
                    if float(getattr(self.ctx, "target_lateral_last_uart_accept_mono", 0.0) or 0.0) > 0.0 else None
                ),
                "target_timeout_type": str(getattr(self.ctx, "target_lateral_timeout_type", "") or ""),
                "target_lateral_stable_count": int(self.ctx.target_lateral_stable_count),
                "target_lateral_stable_frames": int(self._target_lateral_stable_frames()),
                "target_lateral_hold_active": bool(getattr(self.ctx, "target_lateral_hold_active", False)),
                "target_lateral_hold_age_s": (
                    max(0.0, monotonic_ts() - float(self.ctx.last_good_target_mono))
                    if float(getattr(self.ctx, "last_good_target_mono", 0.0) or 0.0) > 0.0 else None
                ),
                "last_good_target_age_s": (
                    max(0.0, monotonic_ts() - float(self.ctx.last_good_target_mono))
                    if float(getattr(self.ctx, "last_good_target_mono", 0.0) or 0.0) > 0.0 else None
                ),
                "last_good_vy_mps": float(getattr(self.ctx, "last_good_vy_mps", 0.0) or 0.0),
                "lateral_cmd_source": str(getattr(self.ctx, "target_lateral_hold_source", "") or ("current" if active else "stop")),
                "slice_timeout_reason": str(self.ctx.target_last_lost_reason or ""),
                "candidate_count": int(getattr(obs, "num_target_candidates", 0) or len(self._target_candidate_list(obs))),
                "selected_candidate_idx": getattr(self.ctx, "selected_candidate_idx", None),
                "selected_candidate_score": getattr(self.ctx, "selected_candidate_score", None),
                "selected_candidate_conf": conf,
                "selected_candidate_cx": cx,
                "selected_candidate_reason": str(getattr(self.ctx, "selected_candidate_reason", "") or ""),
                "target_locked": bool(self.ctx.target_locked),
                "grasp_request_sent": False,
                "grasp_dry_run": False,
                "target_control_latch_active": bool(getattr(self.ctx, "target_control_latch_active", False)),
                "last_completed_inference_id": str(getattr(self.ctx, "last_completed_target_inference_id", "") or ""),
                "explicit_negative_count": int(getattr(self.ctx, "target_explicit_negative_count", 0) or 0),
                "has_new_inference": bool(getattr(obs, "has_new_inference", False)) if obs is not None else False,
                "explicit_negative": bool(getattr(obs, "explicit_negative_detection", False)) if obs is not None else False,
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
        if str(getattr(decision.cmd, "mode", "") or "").upper() == "EDGE_SLIDE_SEARCH":
            now_m = monotonic_ts()
            self.ctx.last_valid_target_lateral_cmd = {
                "vx_mps": float(decision.cmd.vx_mps),
                "vy_mps": float(decision.cmd.vy_mps),
                "wz_radps": float(decision.cmd.wz_radps),
            }
            self.ctx.last_valid_target_lateral_cmd_mono = now_m
            self.ctx.last_valid_target_cmd_owner = "target_lateral"
            self.ctx.last_valid_target_cmd_action = "TARGET_LATERAL_ALIGN"
        if obs is not None and bool(getattr(obs, "found", False)) and lateral_cmd_source == "current":
            self._record_target_lateral_good(obs, vy_cmd if active else None)
        if str(getattr(decision.cmd, "mode", "") or "") == "EDGE_SLIDE_SEARCH":
            self._log(
                "info",
                "[SLICE][LATERAL_CMD] "
                f"found={str(bool(target_found)).lower()} "
                f"cx={cx if cx is not None else 'n/a'} "
                f"err={err if err is not None else 'n/a'} "
                f"vy={float(vy_cmd):.3f} "
                f"source={str(lateral_cmd_source or 'current')}"
            )
        if obs is not None and (active or reason in {"target_locked_hold", "target_locked_freeze_base", "target_lateral_centered_locked", "target_lateral_centered_confirm"}):
            self._log(
                "info",
                "[TARGET][LATERAL_ALIGN] "
                f"target={self.ctx.active_target or self.ctx.canonical_target or ''} "
                f"cx={cx if cx is not None else 'n/a'} "
                f"target_lateral_align_center_x_target={target_x:.2f} "
                f"err={err if err is not None else 'n/a'} "
                f"centered={str(centered_ok).lower()} "
                f"target_lateral_stable_count={int(self.ctx.target_lateral_stable_count)}"
            )
        return decision

    def _edge_slide_progress_snapshot(self) -> Dict[str, Any]:
        now_m = monotonic_ts()
        enter_m = float(getattr(self.ctx, "edge_slide_enter_mono", 0.0) or 0.0)
        if enter_m <= 0.0:
            enter_m = float(getattr(self.ctx, "state_enter_mono", now_m) or now_m)
        return {
            "edge_slide_elapsed_s": max(0.0, now_m - enter_m),
            "edge_slide_lateral_distance_m": max(0.0, float(getattr(self.ctx, "edge_slide_lateral_distance_m", 0.0) or 0.0)),
            "edge_slide_frames": max(0, int(getattr(self.ctx, "edge_slide_frames", 0) or 0)),
        }

    def _update_edge_slide_progress(self) -> Dict[str, Any]:
        now_m = monotonic_ts()
        enter_m = float(getattr(self.ctx, "edge_slide_enter_mono", 0.0) or 0.0)
        if enter_m <= 0.0:
            enter_m = now_m
            self.ctx.edge_slide_enter_mono = now_m
        last_m = float(getattr(self.ctx, "edge_slide_last_progress_mono", 0.0) or 0.0)
        if last_m <= 0.0:
            last_m = now_m
        dt_s = max(0.0, now_m - last_m)
        last_vy = abs(float(getattr(self.ctx, "target_lateral_vy_cmd", 0.0) or 0.0))
        self.ctx.edge_slide_lateral_distance_m = max(
            0.0,
            float(getattr(self.ctx, "edge_slide_lateral_distance_m", 0.0) or 0.0) + last_vy * dt_s,
        )
        self.ctx.edge_slide_last_progress_mono = now_m
        self.ctx.edge_slide_frames = max(0, int(getattr(self.ctx, "edge_slide_frames", 0) or 0)) + 1
        return self._edge_slide_progress_snapshot()

    def _edge_slide_confirm_guard(self) -> Tuple[bool, str, Dict[str, Any]]:
        progress = self._edge_slide_progress_snapshot()
        if bool(getattr(self.cfg, "target_fast_start_confirm_enable", False)):
            self.ctx.edge_slide_confirm_block_reason = ""
            return True, "", progress

        elapsed_s = float(progress.get("edge_slide_elapsed_s", 0.0) or 0.0)
        lateral_dist_m = float(progress.get("edge_slide_lateral_distance_m", 0.0) or 0.0)
        frames = int(progress.get("edge_slide_frames", 0) or 0)
        min_duration_s = max(0.0, float(getattr(self.cfg, "edge_slide_min_duration_s", 1.5) or 1.5))
        min_distance_m = max(0.0, float(getattr(self.cfg, "edge_slide_min_lateral_distance_m", 0.08) or 0.08))
        min_frames = max(0, int(getattr(self.cfg, "edge_slide_min_frames_before_confirm", 10) or 10))

        frames_ok = frames >= min_frames
        duration_ok = elapsed_s >= min_duration_s
        distance_ok = lateral_dist_m >= min_distance_m
        confirm_allowed = bool(frames_ok and (duration_ok or distance_ok))
        if confirm_allowed:
            self.ctx.edge_slide_confirm_block_reason = ""
            return True, "", progress

        missing = []
        if not frames_ok:
            missing.append(f"frames<{min_frames}")
        if not (duration_ok or distance_ok):
            missing.append(f"duration<{min_duration_s:.2f}_and_lateral_dist<{min_distance_m:.3f}")
        reason = "slice_min_guard:" + ",".join(missing or ["not_ready"])
        self.ctx.edge_slide_confirm_block_reason = reason
        return False, reason, progress

    def _target_lateral_last_good_age_s(self) -> Optional[float]:
        ts = float(getattr(self.ctx, "target_lateral_last_good_obs_mono", 0.0) or 0.0)
        if ts <= 0.0:
            return None
        return max(0.0, monotonic_ts() - ts)

    def _record_target_lateral_good(self, obs: TargetObs, vy_cmd: Optional[float]) -> None:
        err = self._target_lateral_error_x(obs)
        now_m = monotonic_ts()
        obs_mono = (
            float(self.ctx.last_valid_target_obs_mono)
            if obs is getattr(self.ctx, "last_valid_target_obs", None)
            and float(getattr(self.ctx, "last_valid_target_obs_mono", 0.0) or 0.0) > 0.0
            else now_m
        )
        self.ctx.target_lateral_last_good_obs = obs
        self.ctx.target_lateral_last_good_obs_mono = obs_mono
        if vy_cmd is not None:
            self.ctx.target_lateral_last_good_vy_mps = float(vy_cmd)
        if err is not None:
            self.ctx.target_lateral_last_err_x = float(err)
            abs_err = abs(float(err))
            previous = getattr(self.ctx, "target_lateral_min_abs_err_x", None)
            progress_delta = abs(float(getattr(self.cfg, "target_lateral_progress_min_delta", 0.01) or 0.01))
            if previous is None or abs_err <= float(previous) - progress_delta:
                self.ctx.target_lateral_min_abs_err_x = abs_err
                self.ctx.target_lateral_last_progress_mono = now_m

    def _target_lateral_hold_decision(
        self,
        candidate_reason: Optional[str] = None,
        *,
        lost_s: Optional[float] = None,
        state: str = "EDGE_SLIDE_SEARCH",
        reason: Optional[str] = None,
    ) -> Optional[MotionDecision]:
        if not bool(getattr(self.cfg, "target_lateral_hold_enable", True)):
            return None
        candidate_reason = str(candidate_reason or reason or "target_missing")
        last_obs = getattr(self.ctx, "target_lateral_last_good_obs", None)
        age_s = self._target_lateral_last_good_age_s()
        last_vy = float(getattr(self.ctx, "target_lateral_last_good_vy_mps", 0.0) or 0.0)
        lost_stop_s = max(0.0, float(getattr(self.cfg, "target_lateral_lost_stop_s", 1.2) or 1.2))
        if lost_s is None:
            lost_s = age_s if age_s is not None else 0.0
        if last_obs is None or age_s is None or abs(last_vy) <= 1e-9:
            return None
        if lost_s >= lost_stop_s:
            self._log("warn", f"[SLICE][TARGET_LOST_STOP] lost_s={lost_s:.2f}")
            return None
        hold_s = max(0.0, float(getattr(self.cfg, "target_lateral_hold_s", 0.8) or 0.8))
        coast_s = max(0.0, float(getattr(self.cfg, "target_lateral_coast_s", 0.5) or 0.5))
        max_motion_s = max(hold_s, lost_stop_s)
        if age_s > max_motion_s:
            return None
        if age_s <= coast_s or max_motion_s <= coast_s:
            scale = 1.0
        else:
            scale = max(0.0, 1.0 - ((age_s - coast_s) / max(1e-6, max_motion_s - coast_s)))
        min_vy = abs(float(getattr(self.cfg, "target_lateral_min_vy_mps", getattr(self.cfg, "target_lateral_align_vy_min_mps", 0.016)) or 0.016))
        vy_max = abs(float(getattr(self.cfg, "target_lateral_align_vy_max_mps", 0.050) or 0.050))
        vy_cmd = max(-vy_max, min(vy_max, last_vy * scale))
        if 0.0 < abs(vy_cmd) < min_vy:
            vy_cmd = min_vy if last_vy >= 0.0 else -min_vy
        if abs(vy_cmd) <= 1e-9:
            return None
        self.ctx.target_lateral_vy_cmd = float(vy_cmd)
        self.ctx.target_lateral_last_cmd_mono = monotonic_ts()
        self._log(
            "info",
            f"[SLICE][TARGET_HOLD] state={state or 'EDGE_SLIDE_SEARCH'} age={age_s:.2f} last_vy={last_vy:.3f} reason={candidate_reason}",
        )
        state_name = str(state or "EDGE_SLIDE_SEARCH")
        cmd = self.controller._cmd(state_name, vx=0.0, vy=vy_cmd, wz=0.0)
        decision = MotionDecision(
            cmd=cmd,
            control_summary=self.controller._summary(state_name, cmd, reason="target_lateral_hold"),
        )
        return self._annotate_target_lateral_decision(
            decision,
            last_obs,
            active=True,
            reason=f"target_lateral_hold lost_s={lost_s:.2f}",
            vy_cmd=vy_cmd,
            lateral_cmd_source="hold",
            hold_active=True,
            hold_age_s=age_s,
            target_found_override=False,
        )

    def _target_search_reject_reason(
        self,
        obs: Optional[TargetObs],
        target_window: Dict[str, Any],
        reason: str,
        *,
        timeout: bool = False,
    ) -> str:
        if timeout:
            return "timeout"
        raw = str(reason or "").strip()
        if obs is None:
            return "target_missing"
        if raw in {"class_mismatch", "target_cls_mismatch", "target_mismatch"}:
            return "class_mismatch"
        if raw.startswith("bbox_") or raw == "target_bbox_invalid":
            return "bbox_invalid"
        if raw.startswith("target_lost") or raw in {"target_missing", "vision_stale"}:
            return "target_missing"
        if not self._target_lateral_centered(obs):
            return "not_centered"
        if not self._target_bbox_valid(obs):
            return "bbox_invalid"
        found_ratio = float(target_window.get("found_ratio", 0.0) or 0.0)
        if found_ratio < float(getattr(self.cfg, "target_confirm_found_ratio_th", 0.5) or 0.5):
            return "found_ratio_low"
        if int(self.ctx.target_found_frames) < int(self.cfg.target_found_frames_to_confirm):
            return "stable_count_not_enough"
        if float(target_window.get("center_jitter", 0.0) or 0.0) > float(self.cfg.target_lock_center_jitter_th):
            return "jitter_too_large"
        if int(self.ctx.target_lateral_stable_count) < self._target_lateral_stable_frames():
            return "stable_count_not_enough"
        return raw or "stable_count_not_enough"

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
        deadband = self._target_deadband_x()
        tol = abs(float(getattr(self.cfg, "target_lateral_align_center_x_tol", 0.06) or 0.06))
        if abs(float(err)) <= tol:
            self.ctx.target_lateral_stable_count += 1
        else:
            self.ctx.target_lateral_stable_count = 0
        if abs(float(err)) <= deadband:
            self.ctx.target_lateral_align_reason = "target_center_deadband"
            self.ctx.target_lateral_vy_cmd = 0.0
            self._remember_good_target(obs, 0.0)
            return None
        kp = abs(float(getattr(self.cfg, "target_lateral_align_kp_vy", 0.08) or 0.08))
        vy_min = abs(float(getattr(self.cfg, "target_lateral_align_vy_min_mps", 0.015) or 0.015))
        vy_max = abs(float(getattr(self.cfg, "target_lateral_align_vy_max_mps", 0.060) or 0.060))
        vy_raw = -float(err) * kp
        vy_cmd = max(-vy_max, min(vy_max, vy_raw))
        if abs(vy_cmd) < vy_min:
            vy_cmd = vy_min if vy_raw >= 0.0 else -vy_min
        cmd = self.controller._cmd(state, vx=0.0, vy=vy_cmd, wz=0.0)
        self.ctx.target_lateral_last_cmd_mono = monotonic_ts()
        decision = MotionDecision(
            cmd=cmd,
            control_summary=self.controller._summary(state, cmd, reason="target_lateral_align"),
        )
        self._remember_good_target(obs, vy_cmd)
        self._log(
            "info",
            f"[SLICE][LATERAL_CMD] source=current target_lateral_align_center_x_target={float(getattr(self.cfg, 'target_lateral_align_center_x_target', 0.40) or 0.40):.2f} "
            f"err={err:.3f} vy={vy_cmd:.3f}",
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
            return False, "target_missing"
        if not bool(obs.found):
            return False, "target_missing"
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
        return False

    def _target_ready_for_grasp_status(self, obs: Optional[TargetObs]) -> Tuple[bool, str]:
        if obs is None:
            return False, "obs_missing"
        if not getattr(obs, "found", False):
            return False, "target_not_found"
        
        active_target = str(self.ctx.active_target or "").strip()
        matched_cls = str(obs.matched_cls or obs.target or "").strip()
        if active_target and matched_cls and not self._target_cls_matches_active(obs):
            return False, "target_cls_mismatch"
            
        conf = obs.matched_conf if obs.matched_conf is not None else 0.0
        if conf < 0.45:
            return False, "target_conf_low"
            
        mb = obs.matched_bbox
        if not mb or not isinstance(mb, list) or len(mb) < 4:
            return False, "target_bbox_invalid"
                    
        if int(getattr(self.ctx, "target_found_frames", 0) or 0) < 5:
            return False, "target_found_frames_low"
            
        return True, "ready"

    def _check_target_ready_for_grasp(self, obs: Optional[TargetObs]) -> bool:
        ready, _ = self._target_ready_for_grasp_status(obs)
        return bool(ready)
