#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config.schema import CarMotionConfig, ControlThresholds
from ..control.types import DockingControlConfig
from ..ipc.protocol import (
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
from ..bridge.arm_protocol import parse_arm_response
from ..utils.grasp_utils import grasp_to_pose_params
from ..utils.target_utils import resolve_target, supported_targets, target_to_class_id
from .common import monotonic_ts
from .context import RuntimeContext, State
from .controller import MotionController, MotionDecision
from .control_authority import decide_table_control_authority
from .core_types import (
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


class TaskRuntimeMixin:
    def handle_task_cmd(self, cmd: TaskCmd) -> Tuple[bool, str]:
        self.ctx.last_task_cmd = cmd
        if cmd.intent == "STOP":
            self._last_stop_mono = monotonic_ts()
            self.ctx.active_session_id = cmd.session_id
            self.ctx.active_epoch = cmd.epoch
            self._interrupt_to_idle("收到 STOP 命令", tts_text="已停止", interrupt_tts=True, send_vision_idle=True)
            return True, "STOP accepted"
        if self._last_stop_mono > 0 and (monotonic_ts() - self._last_stop_mono) < float(self.cfg.post_stop_ignore_s):
            self._log("info", f"忽略 STOP 短窗内的后续命令: {cmd.intent}")
            return False, "ignored in post-stop guard"
        if cmd.confidence < self.cfg.cmd_confidence_th:
            self._log("warn", f"忽略低置信度 task_cmd: {cmd.confidence}")
            self._queue_tts("命令置信度过低")
            return False, "low confidence"
        if cmd.intent == "FIND":
            spec = resolve_target(cmd.target or "")
            if spec is None:
                setattr(self.ctx, "last_task_ack_extra", {
                    "raw_target": str(cmd.target or ""),
                    "supported_targets": supported_targets(),
                })
                return False, "unsupported_target"
            self._start_find_task(cmd)
            return True, "accepted"
        if cmd.intent == "RETURN":
            self._start_return_task(cmd)
            return True, "RETURN accepted"
        return False, "unsupported intent"

    def handle_table_obs(self, obs: TableEdgeObs):
        status = str(getattr(obs, "vision_status", "") or "").strip().upper()
        if status and status not in KNOWN_VISION_STATUS:
            self._enter_error_recovery(f"unknown vision status: {status}")
            return
        self.ctx.last_table_obs = obs
        if getattr(obs, "source_mode", None):
            self.confirm_vision_state("SEARCH", obs.source_mode, source="vision_obs")
        if self.controller._table_bbox_found(obs):
            self.ctx.last_table_seen_ts = monotonic_ts()
            self.ctx.last_table_seen_frame = getattr(obs, "frame_id", None)
            xyxy = getattr(obs, "table_bbox_xyxy", getattr(obs, "yolo_bbox_xyxy", None))
            if xyxy:
                self.ctx.last_table_bbox_xyxy = list(xyxy)
            
            cx_norm = None
            cy_norm = None
            if getattr(obs, "table_cx_norm", None) is not None:
                cx_norm = (float(obs.table_cx_norm) + 1.0) * 0.5
            if getattr(obs, "table_cy_norm", None) is not None:
                cy_norm = (float(obs.table_cy_norm) + 1.0) * 0.5
                
            if cx_norm is not None:
                self.ctx.last_table_center_x_norm = cx_norm
            if cy_norm is not None:
                self.ctx.last_table_center_y_norm = cy_norm
                
            touch_left = bool(getattr(obs, "table_bbox_touch_left", getattr(obs, "yolo_bbox_touch_left", False)))
            touch_right = bool(getattr(obs, "table_bbox_touch_right", getattr(obs, "yolo_bbox_touch_right", False)))
            touch_bottom = bool(getattr(obs, "table_bbox_touch_bottom", getattr(obs, "yolo_bbox_touch_bottom", False)))
            
            self.ctx.last_table_touch_left = touch_left
            self.ctx.last_table_touch_right = touch_right
            self.ctx.last_table_touch_bottom = touch_bottom
            
            if touch_left:
                self.ctx.last_table_side = "left"
            elif touch_right:
                self.ctx.last_table_side = "right"
            elif cx_norm is not None and cx_norm < 0.35:
                self.ctx.last_table_side = "left"
            elif cx_norm is not None and cx_norm > 0.65:
                self.ctx.last_table_side = "right"
            else:
                self.ctx.last_table_side = "center"

    def handle_target_obs(self, obs: TargetObs):
        status = str(getattr(obs, "vision_status", "") or "").strip().upper()
        if status and status not in KNOWN_VISION_STATUS:
            self._enter_error_recovery(f"unknown vision status: {status}")
            return
        self.ctx.last_target_obs = obs
        if self.ctx.desired_vision_mode == "FIND_OBJECT":
            self.confirm_vision_state("SEARCH", "FIND_OBJECT", source="vision_obs")

    def handle_home_obs(self, obs: HomeTagObs):
        self.ctx.last_home_obs = obs

    def handle_grasp_obs(self, obs: Dict[str, Any]):
        status = str(obs.get("status") or "").strip().upper()
        if status and status not in KNOWN_VISION_STATUS:
            self._enter_error_recovery(f"unknown vision status: {status}")
            return
        result = obs.get("result") if isinstance(obs.get("result"), dict) else {}
        self.ctx.grasp_status = status
        grasp = obs.get("grasp") if isinstance(obs.get("grasp"), dict) else None
        if grasp is None and isinstance(result.get("grasp"), dict):
            grasp = result.get("grasp")
        if grasp is None and isinstance(obs.get("canonical_grasp"), dict):
            grasp = obs.get("canonical_grasp")
        if grasp is None and isinstance(result.get("canonical_grasp"), dict):
            grasp = result.get("canonical_grasp")
        if status == "RESULT_READY" and grasp is None:
            self.ctx.grasp_status = "FAILED"
            self.ctx.grasp_result = None
            self.ctx.grasp_reason = "grasp_result_missing"
            self._log("warn", "[GRASP][RESULT] RESULT_READY missing grasp payload; reason=grasp_result_missing")
            return
        self.ctx.grasp_result = grasp
        reason = str(obs.get("reason") or result.get("reason") or "")
        remote_error = str(
            obs.get("remote_error")
            or result.get("remote_error")
            or obs.get("error")
            or result.get("error")
            or ""
        )
        status_code = obs.get("status_code", result.get("status_code"))
        missing_fields = obs.get("missing_fields", result.get("missing_fields"))
        if status == "FAILED" and reason == "remote_predict_failed" and remote_error:
            reason = f"remote_predict_failed:{remote_error}"
        elif status == "FAILED" and reason == "remote_init_failed" and remote_error:
            reason = f"remote_init_failed:{remote_error}"
        elif status == "FAILED" and not reason and remote_error:
            reason = remote_error
        elif status == "FAILED" and not reason and status_code is not None:
            reason = f"remote_status_{status_code}"
        if status == "FAILED" and reason == "grasp_pose_schema_invalid" and missing_fields:
            reason = f"grasp_pose_schema_invalid:{missing_fields}"
        self.ctx.grasp_reason = reason
        if status == "FAILED":
            if str(reason).startswith("remote_init_failed"):
                self._log(
                    "error",
                    "[GRASP_REMOTE][INIT_FAILED] "
                    f"reason={reason} status_code={status_code} "
                    f"base_url={obs.get('base_url') or result.get('base_url')} "
                    f"endpoint={obs.get('endpoint') or result.get('endpoint')} "
                    f"elapsed_ms={obs.get('elapsed_ms') or result.get('elapsed_ms')}",
                )
            elif str(reason).startswith("remote_") or remote_error:
                self._log(
                    "error",
                    "[GRASP_REMOTE][FAILED] "
                    f"reason={reason} status_code={status_code} remote_error={remote_error}",
                )
                if str(reason).startswith("remote_predict_failed"):
                    self._log(
                        "error",
                        "[GRASP_REMOTE][PREDICT_FAILED] "
                        f"reason={reason} status_code={status_code} remote_error={remote_error} "
                        f"remote_no_detection_but_local_target_present={bool(result.get('remote_no_detection_but_local_target_present'))} "
                        f"local_target_bbox_xyxy={result.get('local_target_bbox_xyxy')} "
                        f"local_target_conf={result.get('local_target_conf')}",
                    )
        elif status == "RESULT_READY" and (
            bool(result.get("remote_grasp"))
            or str(result.get("source") or obs.get("source") or "").startswith("remote")
            or isinstance(grasp, dict)
        ):
            setattr(self.ctx, "remote_init_last_success_mono", monotonic_ts())
            self._log("info", "[GRASP_REMOTE][INIT_READY] source=grasp_result")
        if status == "RESULT_READY" and isinstance(grasp, dict):
            self._log(
                "info",
                f"[GRASP][RESULT] ctx.grasp_result updated keys={sorted(str(key) for key in grasp.keys())}",
            )
        proposal = obs.get("reposition_proposal")
        if not isinstance(proposal, dict):
            proposal = result.get("reposition_proposal")
        self.ctx.grasp_reposition_proposal = proposal if isinstance(proposal, dict) else None

    def handle_arm_response(self, resp: ArmResponse):
        self.ctx.arm_response = resp

    def handle_car_state(self, state: CarState):
        self.ctx.last_car_state = state
        self.ctx.last_car_state_mono = monotonic_ts()
        if state.estop and self.cfg.car_estop_to_stop:
            reason = f"底盘急停: {state.message or state.state}"
            self.ctx.last_safety_reason = reason
            self._interrupt_to_idle(reason, tts_text="底盘急停，已停止", interrupt_tts=True)
        elif state.fault and self.cfg.car_fault_to_fail:
            reason = f"底盘故障: {state.message or state.state}"
            self.ctx.last_fail_reason = reason
            self._enter_error_recovery(reason, tts_text="底盘故障，请检查小车", interrupt_tts=True)
        elif state.timeout and self.cfg.car_timeout_to_stop and self.ctx.state != State.IDLE:
            reason = f"底盘超时: {state.message or state.state}"
            self.ctx.last_fail_reason = reason
            self._enter_error_recovery(reason, tts_text="底盘通信超时，已停止", interrupt_tts=True)

    def drain_vision_msgs(self) -> List[Dict]:
        out = list(self.ctx.pending_vision_msgs)
        self.ctx.pending_vision_msgs.clear()
        return out

    def drain_tts_msgs(self) -> List[Dict]:
        out = list(self.ctx.pending_tts_msgs)
        self.ctx.pending_tts_msgs.clear()
        return out

    def _target_debounce_snapshot(self) -> Dict[str, Any]:
        return {
            "target_found_frames": int(self.ctx.target_found_frames),
            "target_lost_frames": int(self.ctx.target_lost_frames),
            "target_lock_frames": int(self.ctx.target_lock_frames),
            "target_loss_since_mono": float(self.ctx.target_loss_since_mono),
            "target_stable_since_mono": float(self.ctx.target_stable_since_mono),
            "target_center_history": [dict(item) for item in self.ctx.target_center_history],
            "target_obs_window": [dict(item) for item in self.ctx.target_obs_window],
            "target_last_center_jitter": float(self.ctx.target_last_center_jitter),
            "target_last_lost_reason": str(self.ctx.target_last_lost_reason or ""),
            "target_last_transition_reason": str(self.ctx.target_last_transition_reason or ""),
            "target_lateral_stable_count": int(getattr(self.ctx, "target_lateral_stable_count", 0) or 0),
            "target_lateral_align_reason": str(getattr(self.ctx, "target_lateral_align_reason", "") or ""),
            "target_lateral_vy_cmd": float(getattr(self.ctx, "target_lateral_vy_cmd", 0.0) or 0.0),
        }

    def _restore_target_debounce_snapshot(self, snapshot: Dict[str, Any]) -> None:
        if not snapshot:
            return
        self.ctx.target_found_frames = int(snapshot.get("target_found_frames", 0) or 0)
        self.ctx.target_lost_frames = int(snapshot.get("target_lost_frames", 0) or 0)
        self.ctx.target_lock_frames = int(snapshot.get("target_lock_frames", 0) or 0)
        self.ctx.target_loss_since_mono = float(snapshot.get("target_loss_since_mono", 0.0) or 0.0)
        self.ctx.target_stable_since_mono = float(snapshot.get("target_stable_since_mono", 0.0) or 0.0)
        self.ctx.target_center_history = [dict(item) for item in snapshot.get("target_center_history", [])]
        self.ctx.target_last_center_jitter = float(snapshot.get("target_last_center_jitter", 0.0) or 0.0)
        self.ctx.target_obs_window = [dict(item) for item in snapshot.get("target_obs_window", [])]
        self.ctx.target_last_lost_reason = str(snapshot.get("target_last_lost_reason", "") or "")
        self.ctx.target_last_transition_reason = str(snapshot.get("target_last_transition_reason", "") or "")
        self.ctx.target_lateral_stable_count = int(snapshot.get("target_lateral_stable_count", 0) or 0)
        self.ctx.target_lateral_align_reason = str(snapshot.get("target_lateral_align_reason", "") or "")
        self.ctx.target_lateral_vy_cmd = float(snapshot.get("target_lateral_vy_cmd", 0.0) or 0.0)

    def _emit_reset_trace(self, reset_state: str, reason: str, cleared_fields: List[str]) -> None:
        self._pending_reset_traces.append(
            {
                "event": "reset_state",
                "reset_state": reset_state,
                "reset_reason": reason,
                "cleared_fields": list(cleared_fields),
                "session_id": self.ctx.active_session_id,
                "target": self.ctx.active_target,
                "state": self.ctx.state.value,
            }
        )

    def reset_edge_tracking(self, reason: str) -> None:
        cleared = [
            "last_table_obs",
            "locked_edge_id",
            "locked_edge_line",
            "locked_roi",
            "locked_yaw_err",
            "locked_dist_err",
            "locked_edge_conf",
            "locked_obs_seq",
            "slide_ref_ready",
            "slide_ref_yaw_err",
            "slide_ref_dist_err",
            "slide_ref_edge_conf",
            "slide_ref_roi",
            "slide_ref_seq",
            "slide_ref_samples",
            "slide_ref_last_sample_key",
            "handoff_state",
            "last_edge_quality",
            "table_found_frames",
            "table_lost_frames",
            "table_lock_frames",
            "edge_identity_state",
            "edge_slope_final_ready_latched",
        ]
        self.ctx.reset_edge_slope_final_ready(reason)
        self.ctx.last_table_obs = None
        self.ctx.locked_edge_id = ""
        self.ctx.locked_edge_line = None
        self.ctx.locked_roi = None
        self.ctx.locked_yaw_err = None
        self.ctx.locked_dist_err = None
        self.ctx.locked_edge_conf = None
        self.ctx.locked_obs_seq = None
        self.ctx.slide_ref_ready = False
        self.ctx.slide_ref_yaw_err = None
        self.ctx.slide_ref_dist_err = None
        self.ctx.slide_ref_edge_conf = None
        self.ctx.slide_ref_roi = None
        self.ctx.slide_ref_seq = None
        self.ctx.slide_ref_samples.clear()
        self.ctx.slide_ref_last_sample_key = ""
        self.ctx.handoff_state = ""
        self.ctx.last_edge_quality.clear()
        self.ctx.table_found_frames = 0
        self.ctx.table_lost_frames = 0
        self.ctx.table_lock_frames = 0
        self._emit_reset_trace("edge", reason, cleared)

    def reset_target_tracking(self, reason: str) -> None:
        cleared = [
            "last_target_obs",
            "target_found_frames",
            "target_lost_frames",
            "target_lock_frames",
            "target_loss_since_mono",
            "target_stable_since_mono",
            "target_center_history",
            "target_last_center_jitter",
            "target_obs_window",
            "target_last_lost_reason",
            "target_last_transition_reason",
        ]
        self.ctx.last_target_obs = None
        self.ctx.target_found_frames = 0
        self.ctx.target_lost_frames = 0
        self.ctx.target_lock_frames = 0
        self.ctx.target_loss_since_mono = 0.0
        self.ctx.target_stable_since_mono = 0.0
        self.ctx.target_center_history.clear()
        self.ctx.target_obs_window.clear()
        self.ctx.target_last_center_jitter = 0.0
        self.ctx.target_last_lost_reason = ""
        self.ctx.target_last_transition_reason = ""
        self._emit_reset_trace("target", reason, cleared)

    def reset_slide_reference(self, reason: str) -> None:
        cleared = [
            "slide_ref_ready",
            "slide_ref_yaw_err",
            "slide_ref_dist_err",
            "slide_ref_edge_conf",
            "slide_ref_roi",
            "slide_ref_seq",
            "slide_ref_samples",
            "slide_ref_last_sample_key",
            "handoff_state",
        ]
        self.ctx.slide_ref_ready = False
        self.ctx.slide_ref_yaw_err = None
        self.ctx.slide_ref_dist_err = None
        self.ctx.slide_ref_edge_conf = None
        self.ctx.slide_ref_roi = None
        self.ctx.slide_ref_seq = None
        self.ctx.slide_ref_samples.clear()
        self.ctx.slide_ref_last_sample_key = ""
        self.ctx.handoff_state = ""
        self._emit_reset_trace("slide_ref", reason, cleared)

    def reset_task_runtime(self, reason: str, keep_session: bool = False) -> None:
        cleared = [
            "task_intent",
            "active_target",
            "active_session_id",
            "active_epoch",
            "active_req_id",
            "desired_vision_stage",
            "desired_vision_mode",
            "confirmed_vision_stage",
            "confirmed_vision_mode",
            "current_edge_id",
            "edge_visit_index",
            "edge_transition_count",
            "table_cycle_count",
            "locked_edge_id",
            "locked_edge_line",
            "locked_roi",
            "locked_yaw_err",
            "locked_dist_err",
            "locked_edge_conf",
            "locked_obs_seq",
            "slide_ref_ready",
            "slide_ref_yaw_err",
            "slide_ref_dist_err",
            "slide_ref_edge_conf",
            "slide_ref_roi",
            "slide_ref_seq",
            "slide_ref_samples",
            "slide_ref_last_sample_key",
            "handoff_state",
            "last_edge_quality",
            "last_fail_reason",
            "last_enter_reason",
            "last_safety_reason",
            "vision_req_fail_streak",
            "task_slide_entries_count",
            "task_target_confirm_count",
            "task_target_locked_count",
            "edge_slide_relock_attempts",
            "task_warning_history",
            "task_done_summary_emitted",
        ]
        session_id = self.ctx.active_session_id
        self._emit_reset_trace("task", reason, cleared)
        self.ctx.clear_task_context()
        self.ctx.last_fail_reason = ""
        self.ctx.last_enter_reason = ""
        self.ctx.last_safety_reason = ""
        self.ctx.vision_req_fail_streak = 0
        self.ctx.task_slide_entries_count = 0
        self.ctx.task_target_confirm_count = 0
        self.ctx.task_target_locked_count = 0
        self.ctx.edge_slide_relock_attempts = 0
        self.ctx.task_warning_history.clear()
        self.ctx.task_done_summary_emitted = False
        self._reset_vision_request_dedupe()
        if keep_session:
            self.ctx.active_session_id = session_id

    def _interrupt_to_idle(self, reason: str, tts_text: Optional[str] = None, interrupt_tts: bool = False, send_vision_idle: bool = False):
        if send_vision_idle:
            self._queue_vision_req(make_vision_idle(session_id=self.ctx.active_session_id, epoch=self.ctx.active_epoch), force=True)
        self._transition(State.IDLE, reason)
        if tts_text:
            self._queue_tts(tts_text, interrupt=interrupt_tts)

    def _enter_error_recovery(self, reason: str, tts_text: Optional[str] = None, interrupt_tts: bool = False):
        self.ctx.resume_state = None
        self._transition(State.ERROR_RECOVERY, reason)
        if tts_text:
            self._queue_tts(tts_text, interrupt=interrupt_tts)

    def _queue_tts(self, text: str, interrupt: bool = False):
        try:
            self.ctx.pending_tts_msgs.append(make_tts_event(text, interrupt=interrupt))
        except Exception:
            pass

    def _start_find_task(self, cmd: TaskCmd):
        raw_target = str(cmd.target or "").strip()
        spec = resolve_target(raw_target)
        if not raw_target or spec is None:
            self._log("warn", "FIND target 为空，忽略")
            return
        self.ctx.clear_task_context()
        self._reset_vision_request_dedupe()
        self.ctx.task_intent = "FIND"
        self.ctx.raw_target = raw_target
        self.ctx.canonical_target = spec.canonical_target
        self.ctx.class_name = spec.class_name
        self.ctx.class_id = int(spec.class_id)
        self.ctx.active_task_id = f"task_{int(time.time() * 1000)}"
        self.ctx.active_target = spec.canonical_target
        self.ctx.active_session_id = cmd.session_id
        self.ctx.active_epoch = cmd.epoch
        self.ctx.task_start_wall_ts = time.time()
        setattr(self.ctx, "last_task_ack_extra", {
            "raw_target": raw_target,
            "canonical_target": spec.canonical_target,
            "class_name": spec.class_name,
            "class_id": int(spec.class_id),
            "task_id": self.ctx.active_task_id,
        })
        self._queue_remote_init_warmup(target=spec.class_name)
        self._transition(State.SEARCH_TABLE, f"开始桌边任务，进入桌边搜索，目标 {spec.canonical_target}")
        self._queue_tts(f"开始寻找 {spec.class_name}")

    def _queue_remote_init_warmup(self, *, target: str) -> None:
        now = monotonic_ts()
        min_interval = max(0.0, float(getattr(self.cfg, "remote_init_min_interval_s", 30.0) or 30.0))
        last_success = float(getattr(self.ctx, "remote_init_last_success_mono", 0.0) or 0.0)
        if last_success > 0.0 and (now - last_success) < min_interval:
            self._log("info", f"[GRASP_REMOTE][INIT_SKIP] already_ready age_s={now - last_success:.1f} min_interval_s={min_interval:.1f}")
            return
        last_attempt = float(getattr(self.ctx, "remote_init_last_attempt_mono", 0.0) or 0.0)
        if last_attempt > 0.0 and (now - last_attempt) < min_interval:
            self._log("info", f"[GRASP_REMOTE][INIT_SKIP] recent_attempt age_s={now - last_attempt:.1f} min_interval_s={min_interval:.1f}")
            return
        setattr(self.ctx, "remote_init_last_attempt_mono", now)
        self._log("info", "[GRASP_REMOTE][INIT_WARMUP] start base_url=from_vista_profile")
        self._queue_vision_req(
            make_vision_req(
                target=target,
                session_id=self.ctx.active_session_id,
                epoch=self.ctx.active_epoch,
                op="UPDATE",
                stage="GRASP",
                mode_hint="GRASP_REMOTE_INIT",
                req_type="target_update",
                payload={
                    "remote_init_warmup": True,
                    "remote_init_min_interval_s": min_interval,
                    "request_reason": "task_start_remote_init_warmup",
                    "orchestrator_state": self.ctx.state.value,
                    "task_id": self.ctx.active_task_id,
                    "raw_target": self.ctx.raw_target,
                    "canonical_target": self.ctx.canonical_target or target,
                    "class_name": self.ctx.class_name,
                    "class_id": self.ctx.class_id,
                },
            ),
            force=True,
        )

    def _start_return_task(self, cmd: TaskCmd):
        self.ctx.clear_task_context()
        self._reset_vision_request_dedupe()
        self.ctx.task_intent = "RETURN"
        self.ctx.active_session_id = cmd.session_id
        self.ctx.active_epoch = cmd.epoch
        self.ctx.task_start_wall_ts = time.time()
        self._transition(State.RETURN_HOME, "开始返航")
        self._queue_tts("开始返航")

    def _tick_idle(self) -> MotionDecision:
        return self.controller.stop_cmd("IDLE")

    def _fresh_target_obs(self) -> Optional[TargetObs]:
        obs = self.ctx.last_target_obs
        if obs is None or time.time() - obs.ts > self.cfg.target_obs_max_age_s:
            return None
        if self.ctx.task_start_wall_ts > 0 and obs.ts < self.ctx.task_start_wall_ts:
            return None
        if obs.session_id and self.ctx.active_session_id and obs.session_id != self.ctx.active_session_id:
            return None
        return obs

    def _fresh_home_obs(self) -> Optional[HomeTagObs]:
        obs = self.ctx.last_home_obs
        if obs is None or time.time() - obs.ts > self.cfg.home_obs_max_age_s:
            return None
        if self.ctx.task_start_wall_ts > 0 and obs.ts < self.ctx.task_start_wall_ts:
            return None
        if obs.session_id and self.ctx.active_session_id and obs.session_id != self.ctx.active_session_id:
            return None
        return obs

    def _state_elapsed(self) -> float:
        return monotonic_ts() - self.ctx.state_enter_mono

    def _start_loss_timer(self, attr_name: str):
        if getattr(self.ctx, attr_name, 0.0) <= 0.0:
            setattr(self.ctx, attr_name, monotonic_ts())

    def _loss_elapsed(self, started_mono: float) -> float:
        if started_mono <= 0.0:
            return 0.0
        return max(0.0, monotonic_ts() - started_mono)

    def _reset_table_loss(self):
        self.ctx.table_lost_frames = 0
        self.ctx.table_loss_since_mono = 0.0
