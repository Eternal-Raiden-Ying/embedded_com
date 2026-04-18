#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
import time
from typing import Dict, Optional

from ...ipc.protocol import VisionReq
from ...utils.detect import compute_target_obs
from .base import BaseStagePlan, StageContext, StageOutput, StageTickInput, next_interaction_id, normalize_upper


def _merge(base: Dict[str, object], override: Optional[Dict[str, object]]) -> Dict[str, object]:
    merged = dict(base)
    if isinstance(override, dict):
        merged.update(override)
    return merged


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


def _grasp_state_from_req(req: VisionReq, target: Optional[str]) -> Dict[str, object]:
    payload = req.payload if isinstance(req.payload, dict) else {}
    return {
        "target_obs": _merge(_default_target_obs(target), payload.get("target_obs") or payload.get("mock_target_obs")),
        "proposal": _merge(_default_proposal(), payload.get("proposal")),
        "result_template": _merge(_default_result(target), payload.get("result")),
        "remote_grasp": bool(payload.get("remote_grasp", True)),
        "need_depth": bool(payload.get("need_depth", True)),
        "adjust_round": 0,
        "last_response": None,
        "last_feedback": None,
        "remote_request_id": None,
        "remote_result_sent": False,
        "remote_release_sent": False,
        "remote_robot_id": str(payload.get("robot_id") or "arm_001"),
        "remote_timeout_s": float(payload.get("remote_timeout_s", 10.0) or 10.0),
        "remote_class_id": payload.get("class_id"),
        "remote_base_url": str(payload.get("remote_base_url") or "").strip() or None,
        "remote_metadata": dict(payload.get("remote_metadata") or {}) if isinstance(payload.get("remote_metadata"), dict) else {},
    }


def _target_obs_from_results(results: Dict[str, object], target: Optional[str]) -> Optional[Dict[str, object]]:
    local = dict((results or {}).get("local_perception") or {})
    target_obs = local.get("target_obs")
    if isinstance(target_obs, dict):
        merged = {"found": bool(target_obs.get("found", True)), "target": target}
        merged.update(target_obs)
        merged.setdefault("target", target)
        return merged
    boxes = local.get("infer_boxes")
    rgb_shape = local.get("rgb_shape")
    if not isinstance(boxes, list) or not rgb_shape:
        return None
    try:
        obs = compute_target_obs(tuple(rgb_shape), target, boxes)
    except Exception:
        return None
    if obs is None:
        return None
    payload = {"found": True, "target": target}
    payload.update(obs)
    payload.setdefault("target", target)
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


class GraspStagePlan(BaseStagePlan):
    """Stage plan for micro-adjustment and remote grasp cooperation."""

    stage_name = "GRASP"
    default_mode = "MICRO_ADJUST"

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
                if key in {"adjust_round", "last_response", "last_feedback", "remote_request_id", "remote_result_sent", "remote_release_sent"}:
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

        decision = normalize_upper((req.response or {}).get("decision"), "REJECT")
        stage_state["last_response"] = dict(req.response or {})
        stage_state["last_feedback"] = dict(req.payload or {}) if isinstance(req.payload, dict) else {}

        if decision != "ACCEPT":
            ctx.current_mode = "MICRO_ADJUST"
            ctx.interaction_id = None
            stage_state["remote_request_id"] = None
            stage_state["remote_result_sent"] = False
            stage_state["remote_release_sent"] = False
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

        request_id = _next_remote_request_id()
        timeout_s = float(stage_state.get("remote_timeout_s", 10.0) or 10.0)
        base_payload = {
            "request_id": request_id,
            "timeout_s": timeout_s,
            "target": ctx.target_name,
            "robot_id": str(stage_state.get("remote_robot_id") or "arm_001"),
            "need_depth": bool(stage_state.get("need_depth", True)),
            "class_id": stage_state.get("remote_class_id"),
            "base_url": stage_state.get("remote_base_url"),
            "metadata": dict(stage_state.get("remote_metadata") or {}),
        }
        stage_state["remote_request_id"] = request_id
        stage_state["remote_result_sent"] = False
        stage_state["remote_release_sent"] = False
        ctx.current_mode = "GRASP_REMOTE"
        ctx.interaction_id = None
        return StageOutput(
            signals={"response": "ACCEPT"},
            effects=[
                _remote_effect("INIT", base_payload),
                _remote_effect("PREDICT", base_payload),
            ],
        )

    def on_stop(self, req: VisionReq, ctx: StageContext) -> Optional[StageOutput]:
        _ = req
        request_id = str(ctx.stage_state.get("remote_request_id") or "").strip()
        should_release = bool(request_id) and not bool(ctx.stage_state.get("remote_release_sent", False))
        ctx.current_stage = "IDLE"
        ctx.current_mode = "IDLE"
        ctx.interaction_id = None
        if not should_release:
            return None
        ctx.stage_state["remote_release_sent"] = True
        return StageOutput(
            effects=[
                _remote_effect(
                    "RELEASE",
                    {
                        "request_id": request_id,
                        "timeout_s": float(ctx.stage_state.get("remote_timeout_s", 5.0) or 5.0),
                        "base_url": ctx.stage_state.get("remote_base_url"),
                    },
                )
            ]
        )

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

        if normalize_upper(ctx.current_mode, self.default_mode) == "GRASP_REMOTE":
            request_id = str(stage_state.get("remote_request_id") or "").strip()
            matched = bool(request_id) and str(remote_result.get("request_id") or "").strip() == request_id
            last_action = normalize_upper(remote_result.get("last_action"), "")
            remote_error = str(remote_result.get("last_error") or "")
            if matched and last_action == "PREDICT" and bool(remote_result.get("last_ok", False)) and bool(remote_result.get("has_result", False)):
                if stage_state.get("remote_result_sent", False):
                    return None
                stage_state["remote_result_sent"] = True
                effects = []
                if not stage_state.get("remote_release_sent", False):
                    stage_state["remote_release_sent"] = True
                    effects.append(
                        _remote_effect(
                            "RELEASE",
                            {
                                "request_id": request_id,
                                "timeout_s": float(stage_state.get("remote_timeout_s", 5.0) or 5.0),
                                "base_url": stage_state.get("remote_base_url"),
                            },
                        )
                    )
                result = deepcopy(remote_result.get("result") or {})
                result.setdefault("target", ctx.target_name)
                result["source"] = "remote_grasp_client"
                result["request_id"] = request_id
                return StageOutput(
                    vision_obs=self.build_obs(
                        ctx,
                        status="RESULT_READY",
                        perception={"target_obs": target_obs},
                        result=result,
                    ),
                    effects=effects,
                    snapshot=output_snapshot,
                )

            if matched and last_action == "PREDICT" and not bool(remote_result.get("last_ok", False)):
                if stage_state.get("remote_result_sent", False):
                    return None
                stage_state["remote_result_sent"] = True
                effects = []
                if not stage_state.get("remote_release_sent", False):
                    stage_state["remote_release_sent"] = True
                    effects.append(
                        _remote_effect(
                            "RELEASE",
                            {
                                "request_id": request_id,
                                "timeout_s": float(stage_state.get("remote_timeout_s", 5.0) or 5.0),
                                "base_url": stage_state.get("remote_base_url"),
                            },
                        )
                    )
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
                    effects=effects,
                    snapshot=output_snapshot,
                )

            return StageOutput(
                vision_obs=self.build_obs(
                    ctx,
                    status="RUNNING",
                    perception={"target_obs": target_obs},
                    result={
                        "remote_state": str(remote_result.get("state") or "awaiting_remote"),
                        "request_id": request_id,
                        "last_response": deepcopy(stage_state.get("last_response")),
                        "remote_enabled": bool(remote_result.get("enabled")),
                        "remote_error": remote_error,
                        "remote_sequence": int(remote_result.get("sequence", 0) or 0),
                        "remote_last_action": str(remote_result.get("last_action") or ""),
                        "remote_last_ok": bool(remote_result.get("last_ok", False)),
                    },
                ),
                snapshot=output_snapshot,
            )

        if not ctx.interaction_id:
            ctx.interaction_id = next_interaction_id()
            stage_state["adjust_round"] = int(stage_state.get("adjust_round", 0) or 0) + 1
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
