#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Optional

from ...ipc.protocol import VisionReq
from .base import BaseStagePlan, StageContext, StageOutput, StageTickInput, normalize_upper, resolve_stage_summary


def _home_tag_obs_from_payload(payload: Optional[Dict[str, object]]) -> Dict[str, object]:
    base: Dict[str, object] = {"found": False}
    source = None
    if isinstance(payload, dict):
        source = payload.get("home_tag_obs") or payload.get("mock_home_tag_obs")
    if isinstance(source, dict):
        base.update(source)
    base["found"] = bool(base.get("found", False))
    return base


def _home_tag_obs_from_results(results: Dict[str, object]) -> Optional[Dict[str, object]]:
    local = dict((results or {}).get("local_perception") or {})
    home_tag_obs = local.get("home_tag_obs")
    if not isinstance(home_tag_obs, dict):
        return None
    payload = {"found": bool(home_tag_obs.get("found", False))}
    payload.update(home_tag_obs)
    return payload


class ReturnStagePlan(BaseStagePlan):
    """Stage plan for home-tag or return-target perception."""

    stage_name = "RETURN"
    default_mode = "TRACK_LOCAL"

    def on_enter(self, req: VisionReq, ctx: StageContext) -> None:
        """Initialize return-home state and choose the initial return mode."""
        super().on_enter(req, ctx)
        ctx.current_mode = normalize_upper(req.mode_hint, self.default_mode)
        ctx.interaction_id = None
        ctx.stage_state["home_tag_obs"] = _home_tag_obs_from_payload(req.payload)

    def on_update(self, req: VisionReq, ctx: StageContext) -> Optional[StageOutput]:
        """Refresh return parameters while staying in RETURN."""
        if req.mode_hint:
            ctx.current_mode = normalize_upper(req.mode_hint, self.default_mode)
        if isinstance(req.payload, dict):
            ctx.stage_state["home_tag_obs"] = _home_tag_obs_from_payload(req.payload)
        return StageOutput()

    def tick(self, tick_input: StageTickInput, ctx: StageContext) -> Optional[StageOutput]:
        """Produce home-observation outputs used by the return workflow."""
        results = dict(tick_input.results or {})
        home_tag_obs, source = resolve_stage_summary(
            results=results,
            stage_state=ctx.stage_state,
            state_key="home_tag_obs",
            default_factory=lambda: _home_tag_obs_from_payload(None),
            result_factory=_home_tag_obs_from_results,
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
