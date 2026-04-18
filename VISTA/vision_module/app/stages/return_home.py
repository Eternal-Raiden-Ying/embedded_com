#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import Dict, Optional

from ...ipc.protocol import VisionReq
from .base import BaseStagePlan, StageContext, StageOutput, StageTickInput, normalize_upper


def _default_home_obs() -> Dict[str, object]:
    return {
        "found": False,
        "tag_id": None,
        "confidence": 0.0,
        "cx_norm": 0.0,
        "area_norm": 0.0,
        "bbox": None,
    }


class ReturnStagePlan(BaseStagePlan):
    stage_name = "RETURN"
    default_mode = "TRACK_LOCAL"

    def on_enter(self, req: VisionReq, ctx: StageContext) -> None:
        super().on_enter(req, ctx)
        ctx.current_mode = normalize_upper(req.mode_hint, self.default_mode)
        ctx.stage_state.clear()
        payload = req.payload if isinstance(req.payload, dict) else {}
        if isinstance(payload.get("home_tag_obs"), dict):
            ctx.stage_state["home_tag_obs"] = dict(payload["home_tag_obs"])

    def on_update(self, req: VisionReq, ctx: StageContext) -> Optional[StageOutput]:
        if req.mode_hint:
            ctx.current_mode = normalize_upper(req.mode_hint, self.default_mode)
        payload = req.payload if isinstance(req.payload, dict) else {}
        if isinstance(payload.get("home_tag_obs"), dict):
            ctx.stage_state["home_tag_obs"] = dict(payload["home_tag_obs"])
        return StageOutput()

    def tick(self, tick_input: StageTickInput, ctx: StageContext) -> Optional[StageOutput]:
        results = dict(tick_input.results or {})
        local = dict(results.get("local_perception") or {})
        home_obs = local.get("home_tag_obs")
        if not isinstance(home_obs, dict):
            home_obs = deepcopy(ctx.stage_state.get("home_tag_obs") or _default_home_obs())
        else:
            ctx.stage_state["home_tag_obs"] = dict(home_obs)
        return StageOutput(
            vision_obs=self.build_obs(
                ctx,
                status="RUNNING",
                perception={"home_tag_obs": home_obs},
            ),
            snapshot={
                "generation": int(tick_input.generation),
                "result_keys": sorted(results.keys()),
            },
        )
