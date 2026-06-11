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
from ..utils.target_utils import target_to_class_id
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


class VisionSyncMixin:
    def _reset_vision_request_dedupe(self) -> None:
        self._last_req_mono = 0.0
        self._last_mode_request_key = ""
        self._last_target_update_key = ""
        self._last_target_update_mono = 0.0

    def confirm_vision_state(self, stage: str, mode: str, source: str = "vision_obs") -> bool:
        stage = str(stage or "").strip().upper()
        mode = str(mode or "").strip().upper()
        if not stage and not mode:
            return False

        desired_stage = str(self.ctx.desired_vision_stage or "").strip().upper()
        desired_mode = str(self.ctx.desired_vision_mode or "").strip().upper()
        confirmed_stage = str(self.ctx.confirmed_vision_stage or "").strip().upper()
        confirmed_mode = str(self.ctx.confirmed_vision_mode or "").strip().upper()

        matches_desired = (
            (not desired_stage or stage == desired_stage) and
            (not desired_mode or mode == desired_mode)
        )
        matches_current = (
            (not confirmed_stage or stage == confirmed_stage) and
            (not confirmed_mode or mode == confirmed_mode)
        )

        if matches_desired:
            self.ctx.confirmed_vision_stage = stage or desired_stage
            self.ctx.confirmed_vision_mode = mode or desired_mode
            self.ctx.vision_confirm_source = source
            self._log("info", f"[VISION_CONFIRM] Confirmed vision state via {source}: stage={self.ctx.confirmed_vision_stage} mode={self.ctx.confirmed_vision_mode} desired={desired_stage}/{desired_mode}")
            return True
        elif matches_current:
            self.ctx.vision_confirm_source = source
            return False
        else:
            self._log("warn", f"[VISION_CONFIRM] Ignored stale/unrelated vision state from {source}: incoming stage={stage} mode={mode} | desired={desired_stage}/{desired_mode} | confirmed={confirmed_stage}/{confirmed_mode}")
            return False

    def handle_vision_req_send_result(self, sent: bool, payload: Dict, error: str = ""):
        req_type = str(payload.get("req_type") or "").strip().lower()
        if req_type == "mode_request":
            if sent:
                self.ctx.vision_req_fail_streak = 0
                self.ctx.active_req_id = str(payload.get("req_id", "") or self.ctx.active_req_id)
                stage = str(payload.get("stage") or "").strip().upper()
                mode = str(payload.get("mode_hint") or "").strip().upper()
                self.confirm_vision_state(stage, mode, source="send_result")
                self._last_mode_request_key = self._vision_request_key(payload, req_type="mode_request")
                return
            else:
                self.ctx.vision_req_fail_streak += 1
                self.ctx.desired_vision_stage = self.ctx.confirmed_vision_stage
                self.ctx.desired_vision_mode = self.ctx.confirmed_vision_mode
                self._log("warn", f"[VISION_SEND] mode_request send failed (streak={self.ctx.vision_req_fail_streak}). Rolled back desired to confirmed: {self.ctx.confirmed_vision_stage}/{self.ctx.confirmed_vision_mode}")
        else:
            if sent:
                self.ctx.vision_req_fail_streak = 0
                self.ctx.active_req_id = str(payload.get("req_id", "") or self.ctx.active_req_id)
                return
            else:
                self.ctx.vision_req_fail_streak += 1
                self._log("warn", f"[VISION_SEND] target_update send failed (streak={self.ctx.vision_req_fail_streak})")

        if not self.cfg.vision_req_fail_to_stop:
            return
        if self.ctx.state in {State.IDLE, State.ERROR_RECOVERY}:
            return
        if self.ctx.vision_req_fail_streak < int(self.cfg.vision_req_fail_threshold):
            return
        reason = f"vision_req_out 发送失败 {self.ctx.vision_req_fail_streak} 次"
        if error:
            reason += f": {error}"
        self.ctx.last_fail_reason = reason
        self._enter_error_recovery(reason, tts_text="视觉链路异常，已停车", interrupt_tts=True)

    def _maybe_resend_req(self, req: Optional[Dict]):
        self._queue_vision_req(req, force=False)

    def _queue_vision_req(self, payload: Dict, force: bool = False):
        if not isinstance(payload, dict) or not payload:
            return
        now_m = monotonic_ts()
        req_type = str(payload.get("req_type") or (payload.get("payload") or {}).get("req_type") or "").strip().lower()
        if not req_type:
            req_type = "mode_request" if str(payload.get("op") or "").strip().upper() in {"START", "STOP"} else "target_update"
            payload["req_type"] = req_type
        req_payload = dict(payload.get("payload") or {})
        req_payload["req_type"] = req_type
        payload["payload"] = req_payload

        request_key = self._vision_request_key(payload, req_type=req_type)
        if req_type == "mode_request":
            if request_key and request_key == self._last_mode_request_key:
                return
        elif req_type == "target_update":
            target_period_s = max(1.0, float(self.cfg.req_resend_period_s or 0.0))
            if request_key and request_key == self._last_target_update_key and (now_m - self._last_target_update_mono) < target_period_s:
                return
        elif not force and now_m - self._last_req_mono < self.cfg.req_resend_period_s:
            return

        if self.ctx.active_session_id and not payload.get("session_id"):
            payload["session_id"] = self.ctx.active_session_id
        if self.ctx.active_epoch and payload.get("epoch") in (None, 0):
            payload["epoch"] = self.ctx.active_epoch
        self.ctx.active_req_id = str(payload.get("req_id", self.ctx.active_req_id) or self.ctx.active_req_id)
        self.ctx.pending_vision_msgs.append(payload)
        self._last_req_mono = now_m
        if req_type == "target_update":
            self._last_target_update_key = request_key
            self._last_target_update_mono = now_m

    def _active_req_payload(self) -> Optional[Dict]:
        binding = self._vision_binding_for_state(self.ctx.state)
        if binding is None:
            return None
        prev_stage = str(self.ctx.confirmed_vision_stage or "").strip().upper()
        prev_mode = str(self.ctx.confirmed_vision_mode or "").strip().upper()
        next_stage = str(binding.stage or "").strip().upper()
        next_mode = str(binding.mode_hint or "").strip().upper()
        changed_mode_level = (not prev_stage) or prev_stage != next_stage or prev_mode != next_mode
        req_type = "mode_request" if changed_mode_level else "target_update"
        op = "START" if req_type == "mode_request" else "UPDATE"
        self.ctx.desired_vision_stage = binding.stage
        self.ctx.desired_vision_mode = binding.mode_hint
        payload = dict(binding.payload or {})
        payload["req_type"] = req_type
        payload["request_reason"] = "vision_mode_changed" if req_type == "mode_request" else "target_or_stage_update"
        
        self._log("info", f"[VISION_PAYLOAD] state={self.ctx.state.value} type={req_type} op={op} desired={self.ctx.desired_vision_stage}/{self.ctx.desired_vision_mode} confirmed={prev_stage}/{prev_mode}")
        
        return make_vision_req(
            target=binding.target,
            session_id=self.ctx.active_session_id,
            epoch=self.ctx.active_epoch,
            op=op,
            stage=binding.stage,
            mode_hint=binding.mode_hint,
            req_type=req_type,
            payload=payload,
        )

    def _vision_request_key(self, payload: Dict, *, req_type: str) -> str:
        req_payload = dict(payload.get("payload") or {})
        stage = str(payload.get("stage") or "").strip().upper()
        mode = str(payload.get("mode_hint") or "").strip().upper()
        session_id = str(payload.get("session_id") or self.ctx.active_session_id or "").strip()
        target = str(payload.get("target") or self.ctx.active_target or "").strip()
        if req_type == "mode_request":
            return "|".join([req_type, session_id, target, stage, mode])
        roi = req_payload.get("locked_roi") or req_payload.get("roi") or req_payload.get("target_roi") or []
        return "|".join(
            [
                req_type,
                session_id,
                target,
                stage,
                mode,
                str(req_payload.get("search_kind") or "").strip().upper(),
                str(req_payload.get("current_edge_id") or "").strip(),
                str(req_payload.get("locked_edge_id") or "").strip(),
                repr(roi),
            ]
        )

    def _vision_binding_for_state(self, state: State) -> Optional[VisionStageBinding]:
        if state in TABLE_VISION_STATES:
            return VisionStageBinding(
                stage="SEARCH",
                mode_hint="FIND_EDGE",
                target=None,
                payload={
                    "search_kind": "TABLE_EDGE",
                    "need_depth": True,
                    "current_edge_id": self.ctx.current_edge_id,
                    "orchestrator_state": state.value,
                    "table_cycle_count": int(self.ctx.table_cycle_count),
                    "edge_visit_index": int(self.ctx.edge_visit_index),
                },
            )
        if state in TARGET_VISION_STATES or state == State.AT_TABLE_EDGE:
            return VisionStageBinding(
                stage="SEARCH",
                mode_hint="FIND_OBJECT",
                target=self.ctx.active_target,
                payload={
                    "search_kind": "TARGET",
                    "need_depth": True,
                    "edge_follow": True,
                    "track_local_edge_update_hz": float(getattr(self.cfg, "edge_follow_track_local_edge_update_hz", 5.0) or 5.0),
                    "current_edge_id": self.ctx.current_edge_id,
                    "locked_edge_id": self.ctx.locked_edge_id,
                    "locked_edge_line": dict(self.ctx.locked_edge_line or {}),
                    "locked_roi": list(self.ctx.locked_roi or []),
                    "locked_yaw_err": self.ctx.locked_yaw_err,
                    "locked_dist_err": self.ctx.locked_dist_err,
                    "locked_edge_conf": self.ctx.locked_edge_conf,
                    "locked_obs_seq": self.ctx.locked_obs_seq,
                    "orchestrator_state": state.value,
                    "edge_visit_index": int(self.ctx.edge_visit_index),
                },
            )
        if state == State.GRASP:
            class_id = target_to_class_id(self.ctx.active_target or "")
            return VisionStageBinding(
                stage="GRASP",
                mode_hint="GRASP_REMOTE",
                target=self.ctx.active_target,
                payload={
                    "class_id": class_id,
                    "remote_grasp": True,
                    "need_depth": True,
                    "orchestrator_state": state.value,
                },
            )
        if state == State.RETURN_HOME:
            return VisionStageBinding(
                stage="RETURN",
                mode_hint="FIND_OBJECT",
                target=None,
                payload={
                    "search_kind": "HOME_TAG",
                    "orchestrator_state": state.value,
                },
            )
        return None

