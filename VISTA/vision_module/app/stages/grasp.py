#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
import time
from typing import Dict, Optional, Tuple

from ...ipc.protocol import VisionReq
from ...utils.detect import compute_target_obs
from .base import BaseStagePlan, StageContext, StageOutput, StageTickInput, next_interaction_id, normalize_upper


def _merge(base: Dict[str, object], override: Optional[Dict[str, object]]) -> Dict[str, object]:
    merged = dict(base)
    if isinstance(override, dict):
        merged.update(override)
    return merged


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


def _default_proposal() -> Dict[str, object]:
    return {
        "motion_delta": {
            "dx_m": 0.03,
            "dy_m": -0.01,
            "dyaw_rad": 0.08,
        },
        "reason": "mock_micro_adjust_before_remote_grasp",
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
        "target_obs": _merge(_default_target_obs(target), payload.get("target_obs") or payload.get("mock_target_obs")),
        "proposal": _merge(_default_proposal(), payload.get("proposal")),
        "result_template": _merge(_default_result(target), payload.get("result")),
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


def _remote_effect(op: str, payload: Dict[str, object]) -> Dict[str, object]:
    return {
        "type": "PUBLISH_EVENT",
        "route": "remote_cmd",
        "payload": {
            "op": normalize_upper(op, "UNKNOWN"),
            **dict(payload or {}),
        },
    }


def _remote_command_payload(stage_state: Dict[str, object], target: Optional[str], request_id: str) -> Dict[str, object]:
    return {
        "request_id": request_id,
        "timeout_s": float(stage_state.get("remote_timeout_s", 10.0) or 10.0),
        "target": target,
        "robot_id": str(stage_state.get("remote_robot_id") or "arm_001"),
        "need_depth": bool(stage_state.get("need_depth", True)),
        "class_id": _coerce_optional_int(stage_state.get("remote_class_id")),
        "metadata": dict(stage_state.get("remote_metadata") or {}),
    }


def _remote_service_init_payload(stage_state: Dict[str, object]) -> Dict[str, object]:
    return {
        "timeout_s": float(stage_state.get("remote_timeout_s", 10.0) or 10.0),
    }


def _frame_ready_for_remote(results: Dict[str, object], required_cameras) -> Dict[str, object]:
    frame_meta = dict((results or {}).get("frame_meta") or {})
    available_cameras = sorted(str(name) for name in tuple(frame_meta.get("cameras") or ()))
    required = [str(name) for name in tuple(required_cameras or ("rgb",))]
    has_frames = bool(frame_meta.get("has_frames", False))
    frame_seq = int(frame_meta.get("frame_seq", 0) or 0)
    ready = bool(has_frames and frame_seq > 0 and set(required).issubset(set(available_cameras)))
    return {
        "ready": ready,
        "frame_seq": frame_seq,
        "available_cameras": available_cameras,
        "required_cameras": required,
    }


class GraspStagePlan(BaseStagePlan):
    """Stage plan for micro-adjustment and remote grasp cooperation."""

    stage_name = "GRASP"
    default_mode = "MICRO_ADJUST"
    common_routes = ("frame_meta", "runtime_status")
    optional_routes = {
        "MICRO_ADJUST": ("local_perception",),
        "GRASP_REMOTE": ("remote_result", "remote_ack"),
        "GRASP_REMOTE_INIT": ("remote_result", "remote_ack"),
    }

    def on_enter(self, req: VisionReq, ctx: StageContext) -> None:
        super().on_enter(req, ctx)
        ctx.target_name = req.target or ctx.target_name
        ctx.current_mode = normalize_upper(req.mode_hint, self.default_mode)
        ctx.interaction_id = None
        ctx.stage_state.clear()
        ctx.stage_state.update(_grasp_state_from_req(req, ctx.target_name))

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
        stage_state = ctx.stage_state
        target_obs = deepcopy(stage_state.get("target_obs") or _default_target_obs(ctx.target_name))
        if ctx.interaction_id and req.interaction_id and str(req.interaction_id) != str(ctx.interaction_id):
            return StageOutput(
                vision_obs=self.build_obs(
                    ctx,
                    status="FAILED",
                    perception={"target_obs": target_obs},
                    result={
                        "reason": "interaction_id_mismatch",
                        "expected": ctx.interaction_id,
                        "received": req.interaction_id,
                    },
                ),
                signals={"response": "ERROR", "reason": "interaction_id_mismatch"},
            )

        # BUSY: if already in GRASP_REMOTE_INIT and INIT task is still in-flight
        current_mode = normalize_upper(ctx.current_mode, self.default_mode)
        if current_mode == "GRASP_REMOTE_INIT":
            return StageOutput(
                vision_obs=self.build_obs(
                    ctx, status="FAILED",
                    perception={"target_obs": target_obs},
                    result={"reason": "busy", "current_mode": "GRASP_REMOTE_INIT"},
                ),
                signals={"response": "ERROR", "reason": "busy"},
            )

        decision = normalize_upper((req.response or {}).get("decision"), "REJECT")
        stage_state["last_response"] = dict(req.response or {})
        stage_state["last_feedback"] = dict(req.payload or {}) if isinstance(req.payload, dict) else {}

        if decision != "ACCEPT":
            ctx.current_mode = "MICRO_ADJUST"
            ctx.interaction_id = None
            stage_state["remote_request_id"] = None
            stage_state["remote_predict_sent"] = False
            stage_state["remote_result_sent"] = False
            stage_state["remote_ready_frame_seq"] = 0
            stage_state["remote_init_retry_count"] = 0
            stage_state["remote_init_retry_inflight"] = False
            stage_state["remote_init_retry_target_attempt"] = 0
            return StageOutput(signals={"response": decision or "REJECT"})

        if not bool(stage_state.get("remote_grasp", True)):
            result = deepcopy(stage_state.get("result_template") or _default_result(ctx.target_name))
            result["accepted"] = True
            result["response"] = dict(req.response or {})
            result["feedback"] = dict(req.payload or {}) if isinstance(req.payload, dict) else {}
            ctx.interaction_id = None
            return StageOutput(
                vision_obs=self.build_obs(
                    ctx,
                    status="RESULT_READY",
                    perception={"target_obs": target_obs},
                    result=result,
                ),
                signals={"response": "ACCEPT"},
            )

        class_id = _coerce_optional_int(stage_state.get("remote_class_id"))
        if class_id is None:
            ctx.current_mode = "MICRO_ADJUST"
            ctx.interaction_id = None
            return StageOutput(
                vision_obs=self.build_obs(
                    ctx,
                    status="FAILED",
                    perception={"target_obs": target_obs},
                    result={"reason": "missing_class_id"},
                ),
                signals={"response": "ERROR", "reason": "missing_class_id"},
            )

        request_id = _next_remote_request_id()
        stage_state["remote_class_id"] = class_id
        stage_state["remote_request_id"] = request_id
        stage_state["remote_predict_sent"] = False
        stage_state["remote_result_sent"] = False
        stage_state["remote_ready_frame_seq"] = 0
        stage_state["remote_init_retry_count"] = 0
        stage_state["remote_init_retry_inflight"] = False
        stage_state["remote_init_retry_target_attempt"] = 0
        # Route through GRASP_REMOTE_INIT if server not yet confirmed ready
        if ctx.server_status != "ready":
            ctx.current_mode = "GRASP_REMOTE_INIT"
        else:
            ctx.current_mode = "GRASP_REMOTE"
        ctx.interaction_id = None
        return StageOutput(signals={"response": "ACCEPT"})

    def on_stop(self, req: VisionReq, ctx: StageContext) -> Optional[StageOutput]:
        _ = req
        ctx.current_stage = "IDLE"
        ctx.current_mode = "IDLE"
        ctx.interaction_id = None
        return None

    def tick(self, tick_input: StageTickInput, ctx: StageContext) -> Optional[StageOutput]:
        results = dict(tick_input.results or {})
        stage_state = ctx.stage_state
        target_obs = _target_obs_from_results(results, ctx.target_name)
        if target_obs is None:
            target_obs = deepcopy(stage_state.get("target_obs") or _default_target_obs(ctx.target_name))
        else:
            stage_state["target_obs"] = dict(target_obs)
        target_obs = deepcopy(target_obs)
        remote_result = dict(results.get("remote_result") or {})
        output_snapshot = {
            "generation": int(tick_input.generation),
            "result_keys": sorted(results.keys()),
        }

        if normalize_upper(ctx.current_mode, self.default_mode) == "GRASP_REMOTE_INIT":
            init_state = str(remote_result.get("service_init_state") or "")
            if ctx.server_status == "ready":
                ctx.current_mode = "GRASP_REMOTE"
                return StageOutput(
                    vision_obs=self.build_obs(
                        ctx, status="RUNNING",
                        perception={"target_obs": target_obs},
                        result={"remote_state": "init_ok_switching_to_predict"},
                    ),
                    snapshot=output_snapshot,
                )
            if ctx.server_status == "error" or init_state == "init_exhausted":
                return StageOutput(
                    vision_obs=self.build_obs(
                        ctx, status="FAILED",
                        perception={"target_obs": target_obs},
                        result={"reason": "init_failed", "remote_state": init_state or "failed"},
                    ),
                    snapshot=output_snapshot,
                )
            return StageOutput(
                vision_obs=self.build_obs(
                    ctx, status="RUNNING",
                    perception={"target_obs": target_obs},
                    result={"remote_state": "initializing", "server_status": ctx.server_status},
                ),
                snapshot=output_snapshot,
            )

        if normalize_upper(ctx.current_mode, self.default_mode) == "GRASP_REMOTE":
            request_id = str(stage_state.get("remote_request_id") or "").strip()
            matched = bool(request_id) and str(remote_result.get("request_id") or "").strip() == request_id
            last_action = normalize_upper(remote_result.get("last_action"), "")
            remote_error = str(remote_result.get("last_error") or "")
            service_init_confirmed = bool(
                remote_result.get("service_init_confirmed", remote_result.get("init_confirmed", False))
            )
            service_init_state = str(remote_result.get("service_init_state") or "")
            service_init_attempts = int(remote_result.get("service_init_attempts", 0) or 0)
            service_init_error = str(remote_result.get("service_init_last_error") or remote_error or "")
            retry_count = int(stage_state.get("remote_init_retry_count", 0) or 0)
            retry_limit = int(stage_state.get("remote_init_retry_limit", 3) or 3)
            retry_inflight = bool(stage_state.get("remote_init_retry_inflight", False))
            retry_target_attempt = int(stage_state.get("remote_init_retry_target_attempt", 0) or 0)

            if retry_inflight and service_init_attempts >= retry_target_attempt and last_action == "INIT":
                stage_state["remote_init_retry_inflight"] = False
                retry_inflight = False

            frame_gate = _frame_ready_for_remote(results, stage_state.get("remote_required_cameras"))

            if not service_init_confirmed:
                if not retry_inflight and retry_count < retry_limit:
                    next_attempt = int(service_init_attempts) + 1
                    stage_state["remote_init_retry_count"] = retry_count + 1
                    stage_state["remote_init_retry_inflight"] = True
                    stage_state["remote_init_retry_target_attempt"] = next_attempt
                    return StageOutput(
                        vision_obs=self.build_obs(
                            ctx,
                            status="RUNNING",
                            perception={"target_obs": target_obs},
                            result={
                                "remote_state": "retrying_init",
                                "request_id": request_id,
                                "init_confirmed": False,
                                "service_init_state": service_init_state or "uninitialized",
                                "service_init_attempts": service_init_attempts,
                                "init_retry_count": int(stage_state.get("remote_init_retry_count", 0) or 0),
                                "init_retry_limit": retry_limit,
                                "remote_error": service_init_error,
                            },
                        ),
                        effects=[_remote_effect("INIT", _remote_service_init_payload(stage_state))],
                        snapshot=output_snapshot,
                    )
                if not retry_inflight and retry_count >= retry_limit:
                    if stage_state.get("remote_result_sent", False):
                        return None
                    stage_state["remote_result_sent"] = True
                    return StageOutput(
                        vision_obs=self.build_obs(
                            ctx,
                            status="FAILED",
                            perception={"target_obs": target_obs},
                            result={
                                "reason": "remote_init_failed",
                                "request_id": request_id,
                                "remote_error": service_init_error or "init_failed",
                                "init_attempts": retry_limit,
                                "init_confirmed": False,
                                "service_init_state": service_init_state or "failed",
                                "service_init_attempts": service_init_attempts,
                                "status_code": remote_result.get("status_code"),
                            },
                        ),
                        snapshot=output_snapshot,
                    )

            if service_init_confirmed and not bool(stage_state.get("remote_predict_sent", False)):
                if frame_gate["ready"]:
                    stage_state["remote_predict_sent"] = True
                    stage_state["remote_ready_frame_seq"] = int(frame_gate["frame_seq"])
                    return StageOutput(
                        vision_obs=self.build_obs(
                            ctx,
                            status="RUNNING",
                            perception={"target_obs": target_obs},
                            result={
                                "remote_state": "predict_requested",
                                "request_id": request_id,
                                "init_confirmed": True,
                                "service_init_state": service_init_state or "ready",
                                "service_init_attempts": service_init_attempts,
                                "frame_ready": True,
                                "frame_seq": int(frame_gate["frame_seq"]),
                                "required_cameras": list(frame_gate["required_cameras"]),
                                "available_cameras": list(frame_gate["available_cameras"]),
                            },
                        ),
                        effects=[
                            _remote_effect("PREDICT", _remote_command_payload(stage_state, ctx.target_name, request_id)),
                        ],
                        snapshot=output_snapshot,
                    )

            if matched and last_action == "PREDICT" and bool(remote_result.get("last_ok", False)) and bool(remote_result.get("has_result", False)):
                if stage_state.get("remote_result_sent", False):
                    return None
                stage_state["remote_result_sent"] = True
                server_response = remote_result.get("result") or {}
                server_status = str(server_response.get("status") or "").strip().lower()
                server_detection = server_response.get("detection") if isinstance(server_response.get("detection"), dict) else {}
                server_reason = str(server_response.get("reason") or "")
                server_message = str(server_response.get("message") or "")

                if server_status == "success":
                    targets = server_response.get("targets") if isinstance(server_response.get("targets"), list) else []
                    grasp = dict(targets[0]) if targets else {}
                    return StageOutput(
                        vision_obs=self.build_obs(
                            ctx,
                            status="RESULT_READY",
                            perception={"target_obs": target_obs},
                            result={
                                "grasp": grasp,
                                "detection": server_detection,
                                "source": "remote_grasp_client",
                                "request_id": request_id,
                            },
                        ),
                        snapshot=output_snapshot,
                    )

                if server_status == "reposition_required":
                    reposition_proposal = server_response.get("reposition_proposal")
                    return StageOutput(
                        vision_obs=self.build_obs(
                            ctx,
                            status="RUNNING",
                            perception={"target_obs": target_obs},
                            result={
                                "reposition_proposal": reposition_proposal if isinstance(reposition_proposal, dict) else None,
                                "reason": server_reason or "reposition_required",
                                "message": server_message,
                                "detection": server_detection,
                                "source": "remote_grasp_client",
                                "request_id": request_id,
                            },
                        ),
                        snapshot=output_snapshot,
                    )

                return StageOutput(
                    vision_obs=self.build_obs(
                        ctx,
                        status="FAILED",
                        perception={"target_obs": target_obs},
                        result={
                            "reason": server_reason or server_status or "grasp_failed",
                            "message": server_message,
                            "detection": server_detection,
                            "source": "remote_grasp_client",
                            "request_id": request_id,
                        },
                    ),
                    snapshot=output_snapshot,
                )

            if matched and last_action == "PREDICT" and not bool(remote_result.get("last_ok", False)):
                if stage_state.get("remote_result_sent", False):
                    return None
                stage_state["remote_result_sent"] = True
                return StageOutput(
                    vision_obs=self.build_obs(
                        ctx,
                        status="FAILED",
                        perception={"target_obs": target_obs},
                        result={
                            "reason": "remote_predict_failed",
                            "request_id": request_id,
                            "remote_error": remote_error or "predict_failed",
                            "status_code": remote_result.get("status_code"),
                        },
                    ),
                    snapshot=output_snapshot,
                )

            remote_state = "awaiting_remote"
            if not service_init_confirmed:
                remote_state = "awaiting_init_retry" if retry_inflight else "awaiting_init"
            elif not bool(stage_state.get("remote_predict_sent", False)):
                remote_state = "awaiting_fresh_frames"
            else:
                remote_state = "awaiting_predict_result"

            return StageOutput(
                vision_obs=self.build_obs(
                    ctx,
                    status="RUNNING",
                    perception={"target_obs": target_obs},
                    result={
                        "remote_state": remote_state,
                        "request_id": request_id,
                        "last_response": deepcopy(stage_state.get("last_response")),
                        "remote_enabled": bool(remote_result.get("enabled")),
                        "remote_error": remote_error,
                        "remote_sequence": int(remote_result.get("sequence", 0) or 0),
                        "remote_last_action": str(remote_result.get("last_action") or ""),
                        "remote_last_ok": bool(remote_result.get("last_ok", False)),
                        "init_confirmed": service_init_confirmed,
                        "service_init_state": service_init_state or ("ready" if service_init_confirmed else "failed"),
                        "service_init_attempts": service_init_attempts,
                        "init_retry_count": int(stage_state.get("remote_init_retry_count", 0) or 0),
                        "init_retry_limit": retry_limit,
                        "init_retry_inflight": bool(stage_state.get("remote_init_retry_inflight", False)),
                        "predict_sent": bool(stage_state.get("remote_predict_sent", False)),
                        "frame_ready": bool(frame_gate["ready"]),
                        "frame_seq": int(frame_gate["frame_seq"]),
                        "required_cameras": list(frame_gate["required_cameras"]),
                        "available_cameras": list(frame_gate["available_cameras"]),
                    },
                ),
                snapshot=output_snapshot,
            )

        if not ctx.interaction_id:
            ctx.interaction_id = next_interaction_id()
            stage_state["adjust_round"] = int(stage_state.get("adjust_round", 0) or 0) + 1
        if normalize_upper(ctx.current_mode, "") not in ("MICRO_ADJUST", "GRASP_REMOTE_INIT"):
            ctx.current_mode = "MICRO_ADJUST"
        return StageOutput(
            vision_obs=self.build_obs(
                ctx,
                status="WAITING_RESPONSE",
                perception={"target_obs": target_obs},
                proposal=deepcopy(stage_state.get("proposal") or _default_proposal()),
                interaction={
                    "required": True,
                    "interaction_id": ctx.interaction_id,
                    "kind": "MOVE_HINT",
                    "round": int(stage_state.get("adjust_round", 1) or 1),
                },
            ),
            snapshot=output_snapshot,
        )
