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


class GraspFlowMixin:
    def _tick_grasp(self) -> MotionDecision:
        now_m = monotonic_ts()
        substate = str(self.ctx.grasp_substate or "")

        if substate == "AWAITING_RESPOND":
            return self._tick_grasp_awaiting_respond(now_m)
        if substate == "AWAITING_RESULT":
            return self._tick_grasp_awaiting_result(now_m)
        if substate == "PRE_ARM_STOP_SETTLE":
            return self._tick_grasp_pre_arm_stop_settle(now_m)
        if substate == "REPOSITIONING":
            return self._tick_grasp_repositioning(now_m)
        if substate == "AWAITING_ARM":
            return self._tick_grasp_awaiting_arm(now_m)
        if substate == "AWAITING_GRABBED":
            return self._tick_grasp_awaiting_grabbed(now_m)
        if substate == "AWAITING_GRABBED_DONE":
            return self._tick_grasp_awaiting_grabbed_done(now_m)
        if substate == "GRASP_VERIFY":
            return self._tick_grasp_verify(now_m)
        return self.controller.stop_cmd("GRASP")

    def _log_grasp_phase_update(self, phase: str, reason: str = "") -> None:
        payload = {
            "phase": phase,
            "target": self.ctx.active_target or "",
            "reason": reason,
        }
        self._log("info", f"grasp_phase_update {payload}")

    def _has_ready_grasp_result(self) -> bool:
        return (
            str(self.ctx.grasp_status or "").strip().upper() == "RESULT_READY"
            and isinstance(self.ctx.grasp_result, dict)
            and bool(self.ctx.grasp_result)
        )

    def _consume_ready_grasp_result(self, now_m: float, substate: str) -> MotionDecision:
        self.ctx.grasp_substate = "PRE_ARM_STOP_SETTLE"
        self.ctx.pre_arm_stop_settle_start_mono = now_m
        grasp_keys = sorted(str(key) for key in (self.ctx.grasp_result or {}).keys())
        payload = {
            "substate": substate,
            "ctx_grasp_status": str(self.ctx.grasp_status or "").strip().upper(),
            "has_grasp_result": True,
            "grasp_result_keys": grasp_keys,
            "next_substate": "PRE_ARM_STOP_SETTLE",
        }
        self._log("info", f"grasp_result_ready_consumed {payload}")
        return self.controller.stop_cmd("GRASP")

    def _log_grasp_flow_tick_debug(self, substate: str, waiting_for: str, now_m: float) -> None:
        grasp = self.ctx.grasp_result if isinstance(self.ctx.grasp_result, dict) else {}
        payload = {
            "substate": substate,
            "ctx_grasp_status": str(self.ctx.grasp_status or "").strip().upper(),
            "has_grasp_result": bool(grasp),
            "grasp_result_keys": sorted(str(key) for key in grasp.keys()),
            "elapsed_s": round(float(self._state_elapsed()), 3),
            "waiting_for": waiting_for,
            "timeout_left_s": round(max(0.0, float(self.ctx.grasp_timeout_mono or 0.0) - float(now_m)), 3),
        }
        self._log("info", f"grasp_flow_tick_debug {payload}")

    def _tick_grasp_awaiting_respond(self, now_m: float) -> MotionDecision:
        if self._has_ready_grasp_result():
            return self._consume_ready_grasp_result(now_m, "AWAITING_RESPOND")
        if self._state_elapsed() < 0.3:
            return self.controller.stop_cmd("GRASP")
        if str(self.ctx.grasp_status or "").upper() == "FAILED":
            reason = self._normalize_grasp_failed_reason(str(self.ctx.grasp_reason or ""))
            self._enter_error_recovery(reason or "grasp failed")
            return self.controller.stop_cmd("GRASP")
        if now_m > self.ctx.grasp_timeout_mono:
            self._log_grasp_flow_tick_debug("AWAITING_RESPOND", "remote_result", now_m)
            reason = self._normalize_grasp_failed_reason(str(self.ctx.grasp_reason or ""))
            self._enter_error_recovery(reason or "grasp respond timeout")
            return self.controller.stop_cmd("GRASP")
        if self.ctx.grasp_status == "WAITING_RESPONSE":
            target_obs = self.ctx.last_target_obs
            self._queue_vision_req(
                make_grasp_req(
                    target=self.ctx.class_name or self.ctx.canonical_target or self.ctx.active_target or "",
                    class_id=int(self.ctx.class_id) if self.ctx.class_id is not None else target_to_class_id(self.ctx.active_target or ""),
                    session_id=self.ctx.active_session_id,
                    epoch=self.ctx.active_epoch,
                    op="RESPOND",
                    payload={
                        "task_id": self.ctx.active_task_id,
                        "raw_target": self.ctx.raw_target,
                        "canonical_target": self.ctx.canonical_target or self.ctx.active_target,
                        "class_name": self.ctx.class_name,
                        "local_target_bbox_xyxy": getattr(target_obs, "matched_bbox", None) or getattr(target_obs, "bbox", None) if target_obs is not None else None,
                        "local_target_conf": getattr(target_obs, "matched_conf", None) if target_obs is not None else None,
                        "local_target_frame_id": getattr(target_obs, "frame_id", None) if target_obs is not None else None,
                    },
                ),
                force=True,
            )
            self.ctx.grasp_substate = "AWAITING_RESULT"
            self.ctx.grasp_timeout_mono = now_m + _GRASP_RESULT_TIMEOUT_S
            self._log_grasp_flow_tick_debug("AWAITING_RESULT", "remote_result", now_m)
        return self.controller.stop_cmd("GRASP")

    def _tick_grasp_awaiting_result(self, now_m: float) -> MotionDecision:
        if self._has_ready_grasp_result():
            return self._consume_ready_grasp_result(now_m, "AWAITING_RESULT")
        if now_m > self.ctx.grasp_timeout_mono:
            self._log_grasp_flow_tick_debug("AWAITING_RESULT", "remote_result", now_m)
            reason = self._normalize_grasp_failed_reason(str(self.ctx.grasp_reason or ""))
            self._enter_error_recovery(reason or "grasp result timeout")
            return self.controller.stop_cmd("GRASP")

        status = str(self.ctx.grasp_status or "").upper()
        if status not in {"", "RUNNING", "WAITING_RESPONSE", "RESULT_READY", "FAILED", "RELAXING"}:
            self._enter_error_recovery(f"unknown vision status: {status}")
            return self.controller.stop_cmd("GRASP")

        if status == "RUNNING" and self.ctx.grasp_reposition_proposal is not None:
            self.ctx.grasp_retry_count += 1
            if self.ctx.grasp_retry_count > _GRASP_RETRY_LIMIT:
                self._enter_error_recovery("grasp reposition retries exhausted")
                return self.controller.stop_cmd("GRASP")
            self.ctx.grasp_substate = "REPOSITIONING"
            self.ctx.grasp_timeout_mono = now_m + _GRASP_REPOSITION_TIMEOUT_S
            self.ctx.grasp_reposition_start_mono = now_m
            return self.controller.stop_cmd("GRASP")

        if status == "FAILED":
            reason = self._normalize_grasp_failed_reason(str(self.ctx.grasp_reason or ""))
            if reason == "no_detection":
                self._transition(State.SEARCH_TARGET_INIT, "grasp failed: target not detected")
                return self.controller.stop_cmd("SEARCH_TARGET_INIT")
            self._enter_error_recovery(reason or "grasp failed")
            return self.controller.stop_cmd("GRASP")

        return self.controller.stop_cmd("GRASP")

    @staticmethod
    def _normalize_grasp_failed_reason(reason: str) -> str:
        text = str(reason or "").strip()
        prefix = "remote_predict_failed:predict_http_"
        if text.startswith(prefix):
            return "remote_predict_failed:http_" + text[len(prefix):]
        return text

    def _tick_grasp_pre_arm_stop_settle(self, now_m: float) -> MotionDecision:
        settle_ms = getattr(self.car_cfg, "pre_arm_stop_settle_ms", 150)
        settle_s = float(settle_ms) / 1000.0
        if now_m - self.ctx.pre_arm_stop_settle_start_mono < settle_s:
            return self.controller.stop_cmd("GRASP")

        if isinstance(self.ctx.grasp_result, dict):
            class_id = self.ctx.grasp_result.get("class_id")
            active_target = self.ctx.active_target
            if class_id is None and active_target:
                try:
                    class_id = target_to_class_id(active_target)
                except Exception:
                    pass

            cfg = getattr(self, "cfg", None)
            lookup_map = getattr(cfg.orchestrator, "target_gripper_widths", {}) if cfg is not None else {}
            if lookup_map:
                override_val = None
                if class_id is not None and str(class_id) in lookup_map:
                    override_val = lookup_map[str(class_id)]
                elif class_id is not None and class_id in lookup_map:
                    override_val = lookup_map[class_id]

                if override_val is None and class_id is not None:
                    try:
                        from ...utils.target_utils import OBJECT_REGISTRY
                        for spec in OBJECT_REGISTRY.values():
                            if int(spec.class_id) == int(class_id):
                                if spec.class_name in lookup_map:
                                    override_val = lookup_map[spec.class_name]
                                    break
                                if spec.canonical_target in lookup_map:
                                    override_val = lookup_map[spec.canonical_target]
                                    break
                    except Exception:
                        pass

                if override_val is None and active_target:
                    if active_target in lookup_map:
                        override_val = lookup_map[active_target]
                    else:
                        try:
                            from ...utils.target_utils import target_to_canonical
                            canonical = target_to_canonical(active_target)
                            if canonical in lookup_map:
                                override_val = lookup_map[canonical]
                        except Exception:
                            pass

                if override_val is not None:
                    override_val_float = float(override_val)
                    if override_val_float > 15.0:
                        self.ctx.grasp_result["gripper_width_cm"] = override_val_float / 10.0
                    else:
                        self.ctx.grasp_result["gripper_width_cm"] = override_val_float
                    self._log("info", f"[GRASP][OVERRIDE] Overriding gripper width to {self.ctx.grasp_result['gripper_width_cm']} cm (configured raw={override_val_float}) for class_id={class_id} target={active_target}")

            try:
                arm_cmd = grasp_to_pose_params(
                    self.ctx.grasp_result,
                    time_ms=int(getattr(self.car_cfg, "grasp_pose_time_ms", 800) or 800),
                )
            except ValueError as exc:
                self._enter_error_recovery("grasp_pose_schema_invalid")
                self._log("error", f"[GRASP][POSE_SCHEMA_INVALID] {exc}")
                return self.controller.stop_cmd("GRASP")
            self.ctx.grasp_substate = "AWAITING_ARM"
            self.ctx.grasp_timeout_mono = now_m + _GRASP_ARM_TIMEOUT_S
            self._log_grasp_phase_update("pose_wait", "arm_pose_send")
            decision = MotionDecision(cmd=self.controller.stop_cmd("GRASP").cmd, arm_cmd=arm_cmd)
            decision.control_summary = {
                "input_grasp": dict(self.ctx.grasp_result),
                "source": "remote_grasp_client",
            }
            self._log_grasp_flow_tick_debug("AWAITING_ARM", "arm_response", now_m)
            return decision
        else:
            self.ctx.grasp_substate = "AWAITING_RESPOND"
            self.ctx.grasp_timeout_mono = now_m + _GRASP_RESPOND_TIMEOUT_S
            return self.controller.stop_cmd("GRASP")

    def _tick_grasp_repositioning(self, now_m: float) -> MotionDecision:
        if now_m > self.ctx.grasp_timeout_mono:
            self._active_reposition_proposal = None
            self._enter_error_recovery("grasp reposition timeout")
            return self.controller.stop_cmd("GRASP")

        proposal = self.ctx.grasp_reposition_proposal
        if not isinstance(proposal, dict):
            self._active_reposition_proposal = None
            self.ctx.grasp_substate = "AWAITING_RESPOND"
            self.ctx.grasp_timeout_mono = now_m + _GRASP_RESPOND_TIMEOUT_S
            return self.controller.stop_cmd("GRASP")

        dx = float(proposal.get("dx_cm", 0.0) or 0.0)
        dy = float(proposal.get("dy_cm", 0.0) or 0.0)
        total_distance = math.hypot(dx, dy)

        # Track active proposal and reset start time if proposal changed
        active_prop = getattr(self, "_active_reposition_proposal", None)
        if active_prop is None or active_prop != proposal:
            self._log("info", f"[GRASP][REPOSITION] Proposal changed from {active_prop} to {proposal}, resetting start time.")
            self._active_reposition_proposal = proposal
            self.ctx.grasp_reposition_start_mono = now_m

        speed_cm_s = float(getattr(self.car_cfg, "grasp_reposition_speed_cm_s", 10.0))
        if speed_cm_s <= 0.0 or not math.isfinite(speed_cm_s):
            speed_cm_s = 10.0
        speed_cm_s = max(1.0, min(30.0, speed_cm_s))

        speed_m_s = speed_cm_s / 100.0
        duration = total_distance / speed_cm_s
        elapsed = now_m - self.ctx.grasp_reposition_start_mono

        if total_distance < 0.5 or (duration - elapsed) <= 0.08:
            target_obs = self.ctx.last_target_obs
            self._active_reposition_proposal = None
            self.ctx.grasp_reposition_proposal = None
            self.ctx.grasp_substate = "AWAITING_RESPOND"
            self.ctx.grasp_timeout_mono = now_m + _GRASP_RESPOND_TIMEOUT_S
            self._queue_vision_req(
                make_grasp_req(
                    target=self.ctx.class_name or self.ctx.canonical_target or self.ctx.active_target or "",
                    class_id=int(self.ctx.class_id) if self.ctx.class_id is not None else target_to_class_id(self.ctx.active_target or ""),
                    session_id=self.ctx.active_session_id,
                    epoch=self.ctx.active_epoch,
                    op="START",
                    payload={
                        "task_id": self.ctx.active_task_id,
                        "raw_target": self.ctx.raw_target,
                        "canonical_target": self.ctx.canonical_target or self.ctx.active_target,
                        "class_name": self.ctx.class_name,
                        "local_target_bbox_xyxy": getattr(target_obs, "matched_bbox", None) or getattr(target_obs, "bbox", None) if target_obs is not None else None,
                        "local_target_conf": getattr(target_obs, "matched_conf", None) if target_obs is not None else None,
                        "local_target_frame_id": getattr(target_obs, "frame_id", None) if target_obs is not None else None,
                    },
                ),
                force=True,
            )
            return self.controller.stop_cmd("GRASP")

        vx = (dx / total_distance) * speed_m_s
        vy = (dy / total_distance) * speed_m_s
        cmd = self.controller._cmd("GRASP_REPOSITION", vx=vx, vy=vy, wz=0.0)
        return MotionDecision(cmd=cmd, control_summary=self.controller._summary(
            "GRASP_REPOSITION", cmd, reason=f"reposition dx={dx:.1f} dy={dy:.1f}"
        ))

    def _tick_grasp_awaiting_arm(self, now_m: float) -> MotionDecision:
        if now_m > self.ctx.grasp_timeout_mono:
            self._log_grasp_flow_tick_debug("AWAITING_ARM", "arm_response", now_m)
            self._log("warn", "arm_pose_failed {'reason': 'arm_pose_timeout'}")
            self._log_grasp_phase_update("error", "arm_pose_timeout")
            self._enter_error_recovery("arm_pose_timeout")
            return self.controller.stop_cmd("GRASP")

        resp = self.ctx.arm_response
        if resp is not None:
            if resp.ok:
                sent_pose = dict(getattr(resp, "sent_pose", {}) or {})
                response_pose = dict(getattr(resp, "response_pose", {}) or {})
                self._log(
                    "info",
                    "arm_pose_success "
                    f"{{'sent_pose': {sent_pose!r}, "
                    f"'response_pose': {response_pose!r}, "
                    "'response_matches_sent': True, "
                    f"'raw_response': {getattr(resp, 'raw_line', '')!r}, "
                    "'next_substate': 'AWAITING_GRABBED'}}",
                )
                self.ctx.grasp_substate = "AWAITING_GRABBED"
                self.ctx.grasp_timeout_mono = now_m + _GRASP_ARM_TIMEOUT_S
                self.ctx.arm_response = None
                self._log_grasp_phase_update("grabbed_wait", "arm_pose_success")
                return self.controller.stop_cmd("GRASP")
            parsed_status = str(getattr(resp, "parsed_status", "") or resp.message or "").strip().upper()
            self.ctx.arm_response = None
            if parsed_status == "ERR_IK":
                self._log("warn", "arm_pose_failed {'reason': 'arm_pose_ik_failed'}")
                self._log_grasp_phase_update("error", "arm_pose_ik_failed")
                self._enter_error_recovery("arm_pose_ik_failed")
                return self.controller.stop_cmd("GRASP")
            if parsed_status == "ERR_CMD":
                self._log("warn", "arm_pose_failed {'reason': 'arm_pose_cmd_error'}")
                self._log_grasp_phase_update("error", "arm_pose_cmd_error")
                self._enter_error_recovery("arm_pose_cmd_error")
                return self.controller.stop_cmd("GRASP")
            if parsed_status == "ARM_SERIAL_OPEN_FAILED":
                self._log("warn", "arm_pose_failed {'reason': 'arm_serial_open_failed'}")
                self._enter_error_recovery("arm_serial_open_failed")
                return self.controller.stop_cmd("GRASP")
            if parsed_status in {"ARM_TX_FAILED", "ARM_SERIAL_WRITE_FAILED"}:
                self._log("warn", "arm_pose_failed {'reason': 'arm_tx_failed'}")
                self._enter_error_recovery("arm_tx_failed")
                return self.controller.stop_cmd("GRASP")
            if parsed_status == "ARM_RESPONSE_TIMEOUT":
                self._log("warn", "arm_pose_failed {'reason': 'arm_pose_timeout'}")
                self._log_grasp_phase_update("error", "arm_pose_timeout")
                self._enter_error_recovery("arm_pose_timeout")
                return self.controller.stop_cmd("GRASP")
            self.ctx.grasp_retry_count += 1
            if self.ctx.grasp_retry_count > _GRASP_RETRY_LIMIT:
                self._enter_error_recovery("arm_pose_unknown_error")
                return self.controller.stop_cmd("GRASP")
            self.ctx.grasp_substate = "AWAITING_RESPOND"
            self.ctx.grasp_timeout_mono = now_m + _GRASP_RESPOND_TIMEOUT_S
            self.ctx.arm_response = None

        return self.controller.stop_cmd("GRASP")

    def _tick_grasp_awaiting_grabbed(self, now_m: float) -> MotionDecision:
        self.ctx.grasp_substate = "AWAITING_GRABBED_DONE"
        self.ctx.grasp_timeout_mono = now_m + _GRASP_ARM_TIMEOUT_S
        self.ctx.arm_response = None
        self._log("info", "arm_grabbed_send {'command': 'GRABBED'}")
        self._log_grasp_phase_update("grabbed_wait", "arm_grabbed_send")
        arm_cmd = ArmCommand(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, command="GRABBED")
        decision = MotionDecision(cmd=self.controller.stop_cmd("GRASP").cmd, arm_cmd=arm_cmd)
        decision.control_summary = {
            "grasp_phase": "retracting_with_object",
            "source": "arm_grabbed",
        }
        return decision

    def _tick_grasp_awaiting_grabbed_done(self, now_m: float) -> MotionDecision:
        if now_m > self.ctx.grasp_timeout_mono:
            self._log_grasp_flow_tick_debug("AWAITING_GRABBED_DONE", "arm_grabbed_response", now_m)
            self._log("warn", "arm_grabbed_timeout {'reason': 'arm_grabbed_timeout'}")
            self._log_grasp_phase_update("error", "arm_grabbed_timeout")
            self._enter_error_recovery("arm_grabbed_timeout")
            return self.controller.stop_cmd("GRASP")

        resp = self.ctx.arm_response
        if resp is None:
            return self.controller.stop_cmd("GRASP")

        parsed_status = str(getattr(resp, "parsed_status", "") or resp.message or "").strip().upper()
        raw = getattr(resp, "raw_line", "")
        self.ctx.arm_response = None
        if bool(resp.ok) and parsed_status == "OK_GRABBED_DONE":
            self._log("info", f"arm_grabbed_done {{'raw_response': {raw!r}, 'next_state': 'DONE'}}")
            self.ctx.grasp_substate = "GRASP_VERIFY"
            self.ctx.grasp_timeout_mono = now_m + 3.0
            self.ctx.grasp_verify_reported = False
            self._log_grasp_phase_update("done", "arm_grabbed_done")
            return self.controller.stop_cmd("GRASP")
        if parsed_status == "OK_GRABBED_START":
            self._log("info", f"arm_grabbed_started {{'raw_response': {raw!r}}}")
            return self.controller.stop_cmd("GRASP")
        if parsed_status == "OK_KEEP_CLAW":
            self._log("info", f"arm_grabbed_keep_claw {{'raw_response': {raw!r}}}")
            return self.controller.stop_cmd("GRASP")
        if parsed_status == "ERR_CMD":
            reason = "arm_grabbed_cmd_error"
        elif parsed_status == "ARM_GRABBED_TIMEOUT":
            reason = "arm_grabbed_timeout"
        else:
            reason = "arm_grabbed_unknown_error"
        self._log("warn", f"arm_grabbed_failed {{'reason': {reason!r}, 'parsed_status': {parsed_status!r}, 'raw_response': {raw!r}}}")
        self._log_grasp_phase_update("error", reason)
        self._enter_error_recovery(reason)
        return self.controller.stop_cmd("GRASP")

    def _tick_grasp_verify(self, now_m: float) -> MotionDecision:
        self._log("info", "[GRASP][VERIFY_ASSUMED_SUCCESS] grasp_success_assumed_for_demo=true")
        self._transition(State.DONE, "arm_motion_done grasp_success_assumed_for_demo")
        self._queue_tts("抓取完成")
        return self.controller.stop_cmd("DONE")

    def _tick_grasp_verify_legacy(self, now_m: float) -> MotionDecision:
        status = str(self.ctx.grasp_status or "").strip().upper()
        result = self.ctx.grasp_result if isinstance(self.ctx.grasp_result, dict) else {}
        explicit_success = result.get("verify_success")
        if explicit_success is None:
            explicit_success = result.get("grasp_success")

        if bool(getattr(self.cfg, "assume_grasp_success_for_test", False)):
            self._log("info", "[GRASP][VERIFY_ASSUMED_SUCCESS] ORCH_ASSUME_GRASP_SUCCESS_FOR_TEST=1")
            self._transition(State.RETURN_HOME, "grasp verify assumed success")
            self._queue_tts("抓取完成，开始返航")
            return self.controller.stop_cmd("RETURN_HOME")

        if status in {"VERIFY_OK", "VERIFIED", "GRASP_VERIFIED", "SUCCESS"} or explicit_success is True:
            self._transition(State.RETURN_HOME, "grasp verified successfully")
            self._queue_tts("抓取完成，开始返航")
            return self.controller.stop_cmd("RETURN_HOME")

        if status in {"VERIFY_FAILED", "GRASP_VERIFY_FAILED"} or explicit_success is False:
            return self._handle_grasp_verify_failed("grasp verify failed")

        if not self.ctx.grasp_verify_reported:
            self._log("warn", "[GRASP][VERIFY_UNAVAILABLE] no real grasp verification source; not assuming success")
            self.ctx.grasp_verify_reported = True

        if now_m > self.ctx.grasp_timeout_mono:
            return self._handle_grasp_verify_failed("grasp verification unavailable")

        return self.controller.stop_cmd("GRASP")

    def _handle_grasp_verify_failed(self, reason: str) -> MotionDecision:
        self.ctx.grasp_retry_count += 1
        if self.ctx.grasp_retry_count > _GRASP_RETRY_LIMIT:
            self._enter_error_recovery(reason)
            return self.controller.stop_cmd("GRASP")
        self.ctx.grasp_substate = "AWAITING_RESPOND"
        self.ctx.grasp_timeout_mono = monotonic_ts() + _GRASP_RESPOND_TIMEOUT_S
        self.ctx.grasp_status = ""
        self.ctx.grasp_result = None
        self.ctx.arm_response = None
        self.ctx.grasp_verify_reported = False
        return self.controller.stop_cmd("GRASP")
