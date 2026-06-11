#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional

from ....ipc.protocol import VisionReq
from ..base import BaseStagePlan, StageContext, StageOutput, StageTickInput, normalize_upper, resolve_stage_summary
from .request_mapping import invalid_search_kind_reason, is_valid_search_kind, mode_for_request
from .status_policy import FAILED, RELAXING, RUNNING, invalid_search_kind_result
from .table_edge_obs_builder import (
    annotate_table_edge_obs,
    payload_has_table_edge_obs,
    table_edge_obs_from_payload,
    table_edge_obs_from_results,
)
from .target_obs_builder import (
    payload_has_target_obs,
    target_obs_from_payload,
    target_obs_from_results,
)

def _sync_edge_follow_payload(req: VisionReq, ctx: StageContext) -> None:
    payload = req.payload if isinstance(req.payload, dict) else {}
    for key in (
        "locked_edge_id",
        "locked_edge_line",
        "locked_roi",
        "locked_yaw_err",
        "locked_dist_err",
        "locked_edge_conf",
        "locked_obs_seq",
        "current_edge_id",
    ):
        if key in payload:
            ctx.stage_state[key] = payload.get(key)


class SearchStagePlan(BaseStagePlan):
    """Stage plan for local target search and depth-based table-edge perception."""

    stage_name = "SEARCH"
    default_mode = "FIND_OBJECT"
    common_routes = ("frame_meta", "runtime_status")
    optional_routes = {
        "FIND_OBJECT": ("local_perception", "table_edge_obs"),
        "FIND_EDGE": ("local_perception", "table_edge_obs"),
        "FIND_TABLE": ("local_perception",),
    }

    @staticmethod
    def _mode_for_request(req: VisionReq, search_kind: str, default: str) -> str:
        return mode_for_request(req, search_kind, default)

    def on_enter(self, req: VisionReq, ctx: StageContext) -> Optional[StageOutput]:
        super().on_enter(req, ctx)
        ctx.target_name = req.target or ctx.target_name
        ctx.interaction_id = None
        payload = req.payload if isinstance(req.payload, dict) else {}
        search_kind = normalize_upper(payload.get("search_kind", ""), "")
        if not is_valid_search_kind(search_kind):
            reason = invalid_search_kind_reason(search_kind)
            return StageOutput(
                vision_obs=self.build_obs(
                    ctx,
                    status=FAILED,
                    perception={},
                    result=invalid_search_kind_result(reason),
                ),
                signals={"invalid_search_kind": True},
            )
        ctx.current_mode = self._mode_for_request(req, search_kind, self.default_mode)
        ctx.stage_state["search_kind"] = search_kind
        ctx.stage_state["target_obs"] = target_obs_from_payload(payload, ctx.target_name)
        ctx.stage_state["table_edge_obs"] = table_edge_obs_from_payload(payload)
        _sync_edge_follow_payload(req, ctx)
        return None

    def on_update(self, req: VisionReq, ctx: StageContext) -> Optional[StageOutput]:
        if req.target:
            ctx.target_name = req.target
        if isinstance(req.payload, dict):
            search_kind = normalize_upper(req.payload.get("search_kind", ""), "")
            if not is_valid_search_kind(search_kind):
                reason = invalid_search_kind_reason(search_kind)
                return StageOutput(
                    vision_obs=self.build_obs(
                        ctx,
                        status=FAILED,
                        perception={},
                        result=invalid_search_kind_result(reason),
                    ),
                    signals={"invalid_search_kind": True},
                )
            ctx.current_mode = self._mode_for_request(req, search_kind, self.default_mode)
            ctx.stage_state["search_kind"] = search_kind
            _sync_edge_follow_payload(req, ctx)
            if payload_has_target_obs(req.payload):
                ctx.stage_state["target_obs"] = target_obs_from_payload(req.payload, ctx.target_name)
            else:
                ctx.stage_state.setdefault("target_obs", target_obs_from_payload(None, ctx.target_name))
            if payload_has_table_edge_obs(req.payload):
                ctx.stage_state["table_edge_obs"] = table_edge_obs_from_payload(req.payload)
            else:
                ctx.stage_state.setdefault("table_edge_obs", table_edge_obs_from_payload(None))
        return StageOutput()

    def tick(self, tick_input: StageTickInput, ctx: StageContext) -> Optional[StageOutput]:
        results = dict(tick_input.results or {})
        mode = normalize_upper(ctx.current_mode, "")

        if mode == "FIND_EDGE":
            local_perception = results.get("local_perception")
            search_kind = normalize_upper(ctx.stage_state.get("search_kind", ""), "")
            target_obs = None
            target_source = ""
            if search_kind in {"EDGE_FOLLOW_TARGET", "TARGET_ON_EDGE"}:
                target_obs, target_source = resolve_stage_summary(
                    results=results,
                    stage_state=ctx.stage_state,
                    state_key="target_obs",
                    default_factory=lambda: target_obs_from_payload(None, ctx.target_name),
                    result_factory=lambda payload: target_obs_from_results(payload, ctx.target_name),
                )
                ctx.stage_state["target_obs"] = dict(target_obs)
            table_edge_obs, table_edge_source = resolve_stage_summary(
                results=results,
                stage_state=ctx.stage_state,
                state_key="table_edge_obs",
                default_factory=lambda: table_edge_obs_from_payload(None),
                result_factory=table_edge_obs_from_results,
                result_route="table_edge_obs",
            )
            table_edge_obs = annotate_table_edge_obs(
                table_edge_obs,
                tick_ts=tick_input.ts,
                source=table_edge_source,
                source_mode=ctx.current_mode,
            )
            ctx.stage_state["table_edge_obs"] = dict(table_edge_obs)
            perception = {"table_edge_obs": table_edge_obs}
            if target_obs is not None:
                perception["target_obs"] = target_obs
            elif search_kind != "TABLE_EDGE":
                perception["local_perception"] = local_perception
            source_payload = {"table_edge_obs": table_edge_source}
            if target_source:
                source_payload["target_obs"] = target_source
            return StageOutput(
                vision_obs=self.build_obs(
                    ctx,
                    status=RUNNING,
                    perception=perception,
                ),
                snapshot={
                    "generation": int(tick_input.generation),
                    "result_keys": sorted(results.keys()),
                    "source": source_payload,
                    "search_kind": ctx.stage_state.get("search_kind"),
                },
            )

        if mode == "FIND_TABLE":
            target_obs, target_source = resolve_stage_summary(
                results=results,
                stage_state=ctx.stage_state,
                state_key="target_obs",
                default_factory=lambda: target_obs_from_payload(None, ctx.target_name),
                result_factory=lambda payload: target_obs_from_results(payload, ctx.target_name),
            )
            return StageOutput(
                vision_obs=self.build_obs(
                    ctx,
                    status=RUNNING,
                    perception={"target_obs": target_obs},
                ),
                snapshot={
                    "generation": int(tick_input.generation),
                    "result_keys": sorted(results.keys()),
                    "source": {"target_obs": target_source},
                    "search_kind": ctx.stage_state.get("search_kind"),
                },
            )

        if mode == "FIND_OBJECT":
            target_obs, source = resolve_stage_summary(
                results=results,
                stage_state=ctx.stage_state,
                state_key="target_obs",
                default_factory=lambda: target_obs_from_payload(None, ctx.target_name),
                result_factory=lambda payload: target_obs_from_results(payload, ctx.target_name),
            )
            table_edge_obs, table_edge_source = resolve_stage_summary(
                results=results,
                stage_state=ctx.stage_state,
                state_key="table_edge_obs",
                default_factory=lambda: table_edge_obs_from_payload(None),
                result_factory=table_edge_obs_from_results,
                result_route="table_edge_obs",
            )
            table_edge_obs = annotate_table_edge_obs(
                table_edge_obs,
                tick_ts=tick_input.ts,
                source=table_edge_source,
                source_mode=ctx.current_mode,
            )
            ctx.stage_state["table_edge_obs"] = dict(table_edge_obs)
            return StageOutput(
                vision_obs=self.build_obs(
                    ctx,
                    status=RUNNING,
                    perception={
                        "target_obs": target_obs,
                        "table_edge_obs": table_edge_obs,
                    },
                ),
                snapshot={
                    "generation": int(tick_input.generation),
                    "result_keys": sorted(results.keys()),
                    "source": {"target_obs": source, "table_edge_obs": table_edge_source},
                    "search_kind": ctx.stage_state.get("search_kind"),
                },
            )

        return StageOutput(
            vision_obs=self.build_obs(ctx, status=RELAXING, perception={}),
            snapshot={
                "generation": int(tick_input.generation),
                "result_keys": sorted(results.keys()),
            },
        )
