#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional

from .base import BaseStagePlan, StageContext, StageOutput, StageTickInput, normalize_upper


class InitStagePlan(BaseStagePlan):
    """Service-initialisation stage — runs once at startup, then auto-transitions to IDLE.

    INIT mode triggers a single remote ``/init`` task.  The stage does **not**
    accept external requests.
    """

    stage_name = "INIT"
    default_mode = "SILENT"
    common_routes = ("frame_meta", "runtime_status")
    optional_routes = {
        "INIT": ("remote_init_status",),
    }

    def on_enter(self, req, ctx: StageContext) -> Optional[StageOutput]:
        ctx.current_stage = self.stage_name
        ctx.current_mode = normalize_upper(getattr(req, "mode_hint", None) or self.default_mode)

    def on_update(self, req, ctx: StageContext) -> Optional[StageOutput]:
        return StageOutput()

    def on_respond(self, req, ctx: StageContext) -> Optional[StageOutput]:
        return StageOutput()

    def on_stop(self, req, ctx: StageContext) -> Optional[StageOutput]:
        return None

    def tick(self, tick_input: StageTickInput, ctx: StageContext) -> Optional[StageOutput]:
        mode = normalize_upper(ctx.current_mode, self.default_mode)

        if mode == "INIT":
            plan = dict((getattr(tick_input, "snapshot", {}) or {}).get("plan") or {})
            remote = dict((plan.get("capabilities") or {}).get("remote") or {})
            if not bool(remote.get("enabled", False)):
                return StageOutput(
                    vision_obs=self.build_obs(
                        ctx,
                        status="RUNNING",
                        result={"message": "init skipped; remote disabled", "server_status": ctx.server_status},
                    ),
                    next_stage="IDLE",
                )
            if ctx.server_status == "ready":
                return StageOutput(
                    vision_obs=self.build_obs(ctx, status="RUNNING"),
                    next_stage="IDLE",
                )
            if ctx.server_status == "error":
                return StageOutput(
                    vision_obs=self.build_obs(
                        ctx, status="FAILED",
                        result={"reason": "init_failed", "server_status": ctx.server_status},
                    ),
                )
            return StageOutput(
                vision_obs=self.build_obs(
                    ctx, status="RUNNING",
                    result={"message": "Initializing... wait", "server_status": ctx.server_status},
                ),
            )

        # SILENT / unknown
        return StageOutput(vision_obs=self.build_obs(ctx, status="RELAXING"))
