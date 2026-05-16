#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Optional, Tuple

from ...ipc.protocol import VisionReq
from ...utils.detect import compute_target_obs
from .base import BaseStagePlan, StageContext, StageOutput, StageTickInput, normalize_upper, resolve_stage_summary


def _home_tag_obs_from_payload(payload: Optional[Dict[str, object]], target: Optional[str]) -> Dict[str, object]:
    base: Dict[str, object] = {"found": False, "target": target, "tag_id": None}
    source = None
    if isinstance(payload, dict):
        source = payload.get("home_tag_obs") or payload.get("mock_home_tag_obs")
    if isinstance(source, dict):
        base.update(source)
    base.setdefault("target", target)
    base["found"] = bool(base.get("found", False))
    return base


def _home_tag_obs_from_results(results: Dict[str, object], target: Optional[str]) -> Optional[Dict[str, object]]:
    local = dict((results or {}).get("local_perception") or {})
    contract_ok = bool(local.get("contract_ok", True))
    contract_error = str(local.get("contract_error") or "")
    contract_warnings = list(local.get("contract_warnings") or [])
    home_tag_obs = local.get("home_tag_obs")
    if isinstance(home_tag_obs, dict):
        payload = {"found": bool(home_tag_obs.get("found", False)), "target": target, "tag_id": None}
        payload.update(home_tag_obs)
        payload.setdefault("target", target)
        payload.setdefault("tag_id", None)
        if contract_error:
            payload.setdefault("contract_error", contract_error)
        if contract_warnings:
            payload.setdefault("contract_warnings", contract_warnings)
        return payload

    weak_payload = {
        "found": False,
        "source": "detect",
        "target": target,
        "tag_id": None,
    }
    if contract_error:
        weak_payload["contract_error"] = contract_error
    if contract_warnings:
        weak_payload["contract_warnings"] = contract_warnings
    if not str(target or "").strip():
        weak_payload["reason"] = "missing_return_target"
        return weak_payload

    boxes = local.get("infer_boxes")
    class_names = local.get("class_names")
    rgb_shape = local.get("rgb_shape")
    if not isinstance(boxes, list) or not rgb_shape:
        return weak_payload if (not contract_ok or contract_error or contract_warnings) else None
    try:
        obs = compute_target_obs(tuple(rgb_shape), target, boxes, class_names=class_names)
    except Exception as exc:
        weak_payload["contract_error"] = weak_payload.get("contract_error") or f"invalid_local_perception:{exc}"
        return weak_payload
    if obs is None:
        return weak_payload if (not contract_ok or contract_error or contract_warnings) else None
    payload = {
        "found": True,
        "source": "detect",
        "target": target,
        "tag_id": None,
        "confidence": obs.get("confidence"),
        "cx_norm": obs.get("cx_norm"),
        "area_norm": obs.get("area_norm"),
        "bbox": obs.get("bbox"),
    }
    if contract_error:
        payload["contract_error"] = contract_error
    if contract_warnings:
        payload["contract_warnings"] = contract_warnings
    return payload


class ReturnStagePlan(BaseStagePlan):
    """Stage plan for home-tag or return-target perception."""

    stage_name = "RETURN"
    default_mode = "TRACK_LOCAL"
    common_routes = ("frame_meta", "runtime_status")
    optional_routes = {
        "TRACK_LOCAL": ("local_perception",),
    }

    def on_enter(self, req: VisionReq, ctx: StageContext) -> None:
        """Initialize return-home state and choose the initial return mode."""
        super().on_enter(req, ctx)
        ctx.target_name = req.target or ctx.target_name
        ctx.current_mode = normalize_upper(req.mode_hint, self.default_mode)
        ctx.interaction_id = None
        ctx.stage_state["home_tag_obs"] = _home_tag_obs_from_payload(req.payload, ctx.target_name)

    def on_update(self, req: VisionReq, ctx: StageContext) -> Optional[StageOutput]:
        """Refresh return parameters while staying in RETURN."""
        if req.target:
            ctx.target_name = req.target
        if req.mode_hint:
            ctx.current_mode = normalize_upper(req.mode_hint, self.default_mode)
        if isinstance(req.payload, dict):
            ctx.stage_state["home_tag_obs"] = _home_tag_obs_from_payload(req.payload, ctx.target_name)
        return StageOutput()

    def tick(self, tick_input: StageTickInput, ctx: StageContext) -> Optional[StageOutput]:
        """Produce home-observation outputs used by the return workflow."""
        results = dict(tick_input.results or {})
        home_tag_obs, source = resolve_stage_summary(
            results=results,
            stage_state=ctx.stage_state,
            state_key="home_tag_obs",
            default_factory=lambda: _home_tag_obs_from_payload(None, ctx.target_name),
            result_factory=lambda payload: _home_tag_obs_from_results(payload, ctx.target_name),
        )
        return StageOutput(
            vision_obs=self.build_obs(
                ctx,
                status="RUNNING",
                perception={"home_tag_obs": home_tag_obs},
            ),
            snapshot={
                "generation": int(tick_input.generation),
                "result_keys": sorted(results.keys()),
                "source": source,
            },
        )
