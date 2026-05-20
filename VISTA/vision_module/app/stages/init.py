#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional

from .base import BaseStagePlan, StageContext, StageOutput, StageTickInput


class InitStagePlan(BaseStagePlan):
    """Service-initialisation stage — runs once at startup, then auto-transitions to IDLE.

    INIT mode triggers a single remote ``/init`` task.  The stage does **not**
    accept external requests (on_enter / on_update / on_respond / on_stop are
    no-ops).  Control returns to the main loop immediately; each tick polls
    ``ctx.server_status``, sends ``RUNNING`` while initialising, and sets
    ``StageOutput.next_stage = "IDLE"`` once the server is ready.
    """

    stage_name = "INIT"
    default_mode = "INIT"
    common_routes = ("frame_meta", "runtime_status")
    optional_routes = {
        "INIT": ("remote_result",),
    }

    def on_enter(self, req, ctx: StageContext) -> None:
        ctx.current_stage = self.stage_name
        ctx.current_mode = self.default_mode

    def on_update(self, req, ctx: StageContext) -> Optional[StageOutput]:
        return StageOutput()

    def on_respond(self, req, ctx: StageContext) -> Optional[StageOutput]:
        return StageOutput()

    def on_stop(self, req, ctx: StageContext) -> Optional[StageOutput]:
        return None

    def tick(self, tick_input: StageTickInput, ctx: StageContext) -> Optional[StageOutput]:
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
