#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
import time
from typing import Dict, Optional, Tuple

from ...ipc.protocol import VisionReq
from ...utils.detect import compute_target_obs
from .base import BaseStagePlan, StageContext, StageOutput, StageTickInput, next_interaction_id, normalize_upper


def _coerce_optional_int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _default_target_obs(target: Optional[str]) -> Dict[str, object]:
    return {
        "found": True,
        "target": target,
        "confidence": 0.88,
        "cx_norm": 0.12,
        "size_norm": 0.18,
        "bbox": [180, 140, 360, 360],
    }


def _default_result(target: Optional[str]) -> Dict[str, object]:
    return {
        "target": target,
        "grasp_pose": {
            "x_m": 0.41,
            "y_m": -0.06,
            "z_m": 0.18,
            "yaw_rad": 1.57,
        },
        "confidence": 0.87,
        "source": "mock_remote_grasp_client",
    }


def _next_remote_request_id() -> str:
    return f"rr_{int(time.time() * 1000)}"


def _required_remote_cameras(need_depth: bool) -> list:
    cameras = ["rgb"]
    if bool(need_depth):
        cameras.append("depth")
    return cameras


def _grasp_state_from_req(req: VisionReq, target: Optional[str]) -> Dict[str, object]:
    payload = req.payload if isinstance(req.payload, dict) else {}
    need_depth = bool(payload.get("need_depth", True))
    return {
        "target_obs": dict(payload.get("target_obs") or payload.get("mock_target_obs") or _default_target_obs(target)),
        "result": dict(payload.get("result") or _default_result(target)),
        "remote_grasp": bool(payload.get("remote_grasp", True)),
        "need_depth": need_depth,
        "adjust_round": 0,
        "last_response": None,
        "last_feedback": None,
        "remote_request_id": None,
        "remote_predict_sent": False,
        "remote_result_sent": False,
        "remote_ready_frame_seq": 0,
        "remote_init_retry_count": 0,
        "remote_init_retry_limit": 3,
        "remote_init_retry_inflight": False,
        "remote_init_retry_target_attempt": 0,
        "remote_robot_id": str(payload.get("robot_id") or "arm_001"),
        "remote_timeout_s": float(payload.get("remote_timeout_s", 10.0) or 10.0),
        "remote_class_id": _coerce_optional_int(payload.get("class_id")),
        "remote_metadata": dict(payload.get("remote_metadata") or {}) if isinstance(payload.get("remote_metadata"), dict) else {},
        "remote_required_cameras": _required_remote_cameras(need_depth),
    }


def _target_obs_from_results(results: Dict[str, object], target: Optional[str]) -> Optional[Dict[str, object]]:
    local = dict((results or {}).get("local_perception") or {})
    contract_ok = bool(local.get("contract_ok", True))
    contract_error = str(local.get("contract_error") or "")
    contract_warnings = list(local.get("contract_warnings") or [])
    target_obs = local.get("target_obs")
    if isinstance(target_obs, dict):
        merged = {"found": bool(target_obs.get("found", True)), "target": target}
        merged.update(target_obs)
        merged.setdefault("target", target)
        if contract_error:
            merged.setdefault("contract_error", contract_error)
        if contract_warnings:
            merged.setdefault("contract_warnings", contract_warnings)
        return merged
    boxes = local.get("infer_boxes")
    class_names = local.get("class_names")
    rgb_shape = local.get("rgb_shape")
    weak_payload = {"found": False, "target": target}
    if contract_error:
        weak_payload["contract_error"] = contract_error
    if contract_warnings:
        weak_payload["contract_warnings"] = contract_warnings
    if not isinstance(boxes, list) or not rgb_shape:
        return weak_payload if (not contract_ok or contract_error or contract_warnings) else None
    try:
        obs = compute_target_obs(tuple(rgb_shape), target, boxes, class_names=class_names)
    except Exception as exc:
        weak_payload["contract_error"] = weak_payload.get("contract_error") or f"invalid_local_perception:{exc}"
        return weak_payload
    if obs is None:
        return weak_payload if (not contract_ok or contract_error or contract_warnings) else None
    payload = {"found": True, "target": target}
    payload.update(obs)
    payload.setdefault("target", target)
    if contract_error:
        payload["contract_error"] = contract_error
    if contract_warnings:
        payload["contract_warnings"] = contract_warnings
    return payload


# ── GraspStagePlan ─────────────────────────────────────────────────────────

class GraspStagePlan(BaseStagePlan):
    """Grasp stage state machine.

    Mode transitions
    ================
    on_enter:  mode_hint or ``SILENT`` (default).
    on_respond:
      GRASP_REMOTE_INIT  → return initializing (no mode switch)
      GRASP_REMOTE + ACCEPT    → *MICRO_ADJUST*
      GRASP_REMOTE + other     → *SILENT*
      MICRO_ADJUST + ACCEPT    → *GRASP_REMOTE*
      MICRO_ADJUST + REJECT    → *SILENT*
    tick:
      GRASP_REMOTE_INIT        → ready → *GRASP_REMOTE*   (auto)

    Mode behaviours
    ===============
    SILENT              RELAXING  — no capability, waiting for instruction
    GRASP_REMOTE_INIT   RUNNING   — task init, auto-switch to GRASP_REMOTE
    GRASP_REMOTE        RUNNING   — task predict, outputs result/reposition/fail
    MICRO_ADJUST        WAITING_RESPONSE — waiting for orchestrator to respond
    """

    stage_name = "GRASP"
    default_mode = "SILENT"
    common_routes = ("frame_meta", "runtime_status")
    optional_routes = {
        "SILENT": (),
        "GRASP_REMOTE_INIT": ("remote_init_status",),
        "GRASP_REMOTE": ("remote_result",),
        "MICRO_ADJUST": (),
    }

    # ── on_* handlers (request-driven) ─────────────────────────────────

    def on_enter(self, req: VisionReq, ctx: StageContext) -> Optional[StageOutput]:
        super().on_enter(req, ctx)
        ctx.target_name = req.target or ctx.target_name
        requested = normalize_upper(req.mode_hint, self.default_mode)
        ctx.current_mode = requested
        ctx.interaction_id = None
        ctx.stage_state.clear()
        ctx.stage_state.update(_grasp_state_from_req(req, ctx.target_name))
        if requested == "GRASP_REMOTE" and ctx.server_status != "ready":
            ctx.current_mode = "GRASP_REMOTE_INIT"
            ctx.stage_state["mode_override_reason"] = (
                f"requested={requested} actual=GRASP_REMOTE_INIT"
                f" server_status={ctx.server_status}"
            )

    def on_update(self, req: VisionReq, ctx: StageContext) -> Optional[StageOutput]:
        if req.target:
            ctx.target_name = req.target
        if req.mode_hint:
            ctx.current_mode = normalize_upper(req.mode_hint, self.default_mode)
        if isinstance(req.payload, dict):
            refreshed = _grasp_state_from_req(req, ctx.target_name)
            for key, value in refreshed.items():
                if key in {
                    "adjust_round",
                    "last_response",
                    "last_feedback",
                    "remote_request_id",
                    "remote_predict_sent",
                    "remote_result_sent",
                    "remote_ready_frame_seq",
                    "remote_init_retry_count",
                    "remote_init_retry_limit",
                    "remote_init_retry_inflight",
                    "remote_init_retry_target_attempt",
                }:
                    continue
                ctx.stage_state[key] = value
        return StageOutput()

    def on_respond(self, req: VisionReq, ctx: StageContext) -> Optional[StageOutput]:
        current_mode = normalize_upper(ctx.current_mode, self.default_mode)
        decision = normalize_upper((req.response or {}).get("decision"), "REJECT")
        ctx.stage_state["last_response"] = dict(req.response or {})
        ctx.stage_state["last_feedback"] = dict(req.payload or {}) if isinstance(req.payload, dict) else {}

        # ── GRASP_REMOTE_INIT → return initializing (task still running) ──
        if current_mode == "GRASP_REMOTE_INIT":
            return StageOutput(signals={"response": "BUSY", "reason": "init_in_progress"})

        # ── GRASP_REMOTE → ACCEPT → MICRO_ADJUST (reposition flow) ──
        if current_mode == "GRASP_REMOTE":
            if decision == "ACCEPT":
                ctx.current_mode = "MICRO_ADJUST"
                ctx.interaction_id = None
                return StageOutput(signals={"response": "ACCEPT"})
            ctx.current_mode = "SILENT"
            ctx.interaction_id = None
            return StageOutput(signals={"response": decision or "REJECT"})

        # ── MICRO_ADJUST → ACCEPT → GRASP_REMOTE (retry predict) ──
        if current_mode == "MICRO_ADJUST":
            if decision == "ACCEPT":
                ctx.current_mode = "GRASP_REMOTE"
                ctx.interaction_id = None
                ctx.stage_state["remote_request_id"] = _next_remote_request_id()
                ctx.stage_state["remote_predict_sent"] = False
                ctx.stage_state["remote_result_sent"] = False
                ctx.stage_state["remote_ready_frame_seq"] = 0
                return StageOutput(signals={"response": "ACCEPT"})
            ctx.current_mode = "SILENT"
            ctx.interaction_id = None
            return StageOutput(signals={"response": decision or "REJECT"})

        # ── SILENT / unknown → stay silent ──
        return StageOutput(signals={"response": decision})

    def on_stop(self, req: VisionReq, ctx: StageContext) -> Optional[StageOutput]:
        _ = req
        ctx.interaction_id = None
        return None

    # ── tick ───────────────────────────────────────────────────────────

    def tick(self, tick_input: StageTickInput, ctx: StageContext) -> Optional[StageOutput]:
        results = dict(tick_input.results or {})
        stage_state = ctx.stage_state
        target_obs = _target_obs_from_results(results, ctx.target_name)
        if target_obs is None:
            target_obs = deepcopy(stage_state.get("target_obs") or _default_target_obs(ctx.target_name))
        else:
            stage_state["target_obs"] = dict(target_obs)
        target_obs = deepcopy(target_obs)
        current_mode = normalize_upper(ctx.current_mode, self.default_mode)
        snapshot = {"generation": int(tick_input.generation), "result_keys": sorted(results.keys())}

        # ── SILENT ─────────────────────────────────────────────────
        if current_mode == "SILENT":
            return StageOutput(
                vision_obs=self.build_obs(ctx, status="RELAXING"),
                snapshot=snapshot,
            )

        # ── GRASP_REMOTE_INIT ───────────────────────────────────────
        if current_mode == "GRASP_REMOTE_INIT":
            if ctx.server_status == "ready":
                ctx.current_mode = "GRASP_REMOTE"
                return StageOutput(
                    vision_obs=self.build_obs(ctx, status="RUNNING",
                                               perception={"target_obs": target_obs},
                                               result={"remote_state": "init_ok_switching_to_predict"}),
                    snapshot=snapshot,
                )
            if ctx.server_status == "error":
                return StageOutput(
                    vision_obs=self.build_obs(ctx, status="FAILED",
                                               perception={"target_obs": target_obs},
                                               result={"reason": "init_failed", "server_status": ctx.server_status}),
                    snapshot=snapshot,
                )
            return StageOutput(
                vision_obs=self.build_obs(ctx, status="RUNNING",
                                           perception={"target_obs": target_obs},
                                           result={"remote_state": "initializing"}),
                snapshot=snapshot,
            )

        # ── GRASP_REMOTE ────────────────────────────────────────────
        if current_mode == "GRASP_REMOTE":
            remote = dict(results.get("remote_result") or {})
            request_id = str(stage_state.get("remote_request_id") or "").strip()
            last_action = normalize_upper(remote.get("last_action"), "")
            remote_error = str(remote.get("last_error") or "")

            # Waiting for PREDICT task to complete
            if last_action != "predict":
                return StageOutput(
                    vision_obs=self.build_obs(ctx, status="RUNNING",
                                               perception={"target_obs": target_obs},
                                               result={"remote_state": "awaiting_predict", "request_id": request_id}),
                    snapshot=snapshot,
                )

            # PREDICT result arrived
            if stage_state.get("remote_result_sent", False):
                return None
            stage_state["remote_result_sent"] = True

            if not bool(remote.get("last_ok", False)):
                return StageOutput(
                    vision_obs=self.build_obs(
                        ctx, status="FAILED",
                        perception={"target_obs": target_obs},
                        result={"reason": "remote_predict_failed", "request_id": request_id,
                                "remote_error": remote_error or "predict_failed",
                                "status_code": remote.get("status_code")},
                    ),
                    snapshot=snapshot,
                )

            if not bool(remote.get("has_result", False)):
                return StageOutput(
                    vision_obs=self.build_obs(ctx, status="RUNNING",
                                               perception={"target_obs": target_obs},
                                               result={"remote_state": "awaiting_predict_result", "request_id": request_id}),
                    snapshot=snapshot,
                )

            server_response = remote.get("result") or {}
            server_status = str(server_response.get("status") or "").strip().lower()
            server_detection = server_response.get("detection") if isinstance(server_response.get("detection"), dict) else {}
            server_reason = str(server_response.get("reason") or "")

            if server_status == "success":
                targets = server_response.get("targets")
                first_target = dict(targets[0]) if isinstance(targets, (list, tuple)) and targets else {}
                result = {
                    "grasp": first_target,
                    "detection": dict(server_detection),
                    "source": "remote_grasp_client",
                    "request_id": request_id,
                }
                return StageOutput(
                    vision_obs=self.build_obs(ctx, status="RESULT_READY",
                                               perception={"target_obs": target_obs},
                                               result=result),
                    snapshot=snapshot,
                )

            if server_status == "reposition_required":
                result = {
                    "reposition_proposal": server_response.get("reposition_proposal") or {},
                    "reason": server_reason or "reposition_required",
                    "message": str(server_response.get("message") or ""),
                    "detection": dict(server_detection),
                    "source": "remote_grasp_client",
                    "request_id": request_id,
                }
                return StageOutput(
                    vision_obs=self.build_obs(ctx, status="RUNNING",
                                               perception={"target_obs": target_obs},
                                               result=result),
                    snapshot=snapshot,
                )

            # Unknown status → FAILED
            return StageOutput(
                vision_obs=self.build_obs(
                    ctx, status="FAILED",
                    perception={"target_obs": target_obs},
                    result={
                        "reason": server_reason or server_status or "grasp_failed",
                        "message": str(server_response.get("message") or ""),
                        "detection": dict(server_detection),
                        "source": "remote_grasp_client",
                        "request_id": request_id,
                    },
                ),
                snapshot=snapshot,
            )

        # ── MICRO_ADJUST ────────────────────────────────────────────
        if current_mode == "MICRO_ADJUST":
            if not ctx.interaction_id:
                ctx.interaction_id = next_interaction_id()
                stage_state["adjust_round"] = int(stage_state.get("adjust_round", 0) or 0) + 1
            return StageOutput(
                vision_obs=self.build_obs(
                    ctx,
                    status="WAITING_RESPONSE",
                    perception={"target_obs": target_obs},
                    interaction={
                        "required": True,
                        "interaction_id": ctx.interaction_id,
                        "kind": "MOVE_HINT",
                        "round": int(stage_state.get("adjust_round", 1) or 1),
                    },
                ),
                snapshot=snapshot,
            )

        # ── Unknown mode → SILENT ───────────────────────────────────
        ctx.current_mode = "SILENT"
        return StageOutput(
            vision_obs=self.build_obs(ctx, status="RELAXING"),
            snapshot=snapshot,
        )
