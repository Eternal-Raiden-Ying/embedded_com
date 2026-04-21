#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Optional

from ...ipc.protocol import VisionReq
from ...utils.detect import compute_target_obs
from .base import BaseStagePlan, StageContext, StageOutput, StageTickInput, normalize_upper, resolve_stage_summary


def _target_obs_from_payload(payload: Optional[Dict[str, object]], target: Optional[str]) -> Dict[str, object]:
    base: Dict[str, object] = {"found": False, "target": target}
    source = None
    if isinstance(payload, dict):
        source = payload.get("target_obs") or payload.get("mock_target_obs")
    if isinstance(source, dict):
        base.update(source)
    base.setdefault("target", target)
    base["found"] = bool(base.get("found", False))
    return base


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


class SearchStagePlan(BaseStagePlan):
    """Stage plan for local target search and tracking."""

    stage_name = "SEARCH"
    default_mode = "TRACK_LOCAL"

    def on_enter(self, req: VisionReq, ctx: StageContext) -> None:
        """Prepare target metadata and choose the initial local tracking mode."""
        super().on_enter(req, ctx)
        ctx.target_name = req.target or ctx.target_name
        ctx.current_mode = normalize_upper(req.mode_hint, self.default_mode)
        ctx.interaction_id = None
        ctx.stage_state["target_obs"] = _target_obs_from_payload(req.payload, ctx.target_name)

    def on_update(self, req: VisionReq, ctx: StageContext) -> Optional[StageOutput]:
        """Refresh target or search parameters without leaving SEARCH."""
        if req.target:
            ctx.target_name = req.target
        if req.mode_hint:
            ctx.current_mode = normalize_upper(req.mode_hint, self.default_mode)
        if isinstance(req.payload, dict):
            ctx.stage_state["target_obs"] = _target_obs_from_payload(req.payload, ctx.target_name)
        return StageOutput()

    def tick(self, tick_input: StageTickInput, ctx: StageContext) -> Optional[StageOutput]:
        """Produce local search observations and SEARCH stage output envelopes."""
        results = dict(tick_input.results or {})
        target_obs, source = resolve_stage_summary(
            results=results,
            stage_state=ctx.stage_state,
            state_key="target_obs",
            default_factory=lambda: _target_obs_from_payload(None, ctx.target_name),
            result_factory=lambda payload: _target_obs_from_results(payload, ctx.target_name),
        )
        return StageOutput(
            vision_obs=self.build_obs(
                ctx,
                status="RUNNING",
                perception={"target_obs": target_obs},
            ),
            snapshot={
                "generation": int(tick_input.generation),
                "result_keys": sorted(results.keys()),
                "source": source,
            },
        )
