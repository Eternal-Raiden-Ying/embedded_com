#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Optional

from ...ipc.protocol import VisionReq
from ...utils.detect import compute_target_obs
from .base import BaseStagePlan, StageContext, StageOutput, StageTickInput, normalize_upper, resolve_stage_summary


def _search_kind(req: VisionReq, default_mode: str) -> str:
    payload = req.payload if isinstance(req.payload, dict) else {}
    explicit = normalize_upper(payload.get("search_kind"), "")
    if explicit in {"TABLE_EDGE", "TARGET"}:
        return explicit
    hinted_mode = normalize_upper(req.mode_hint, default_mode)
    if hinted_mode == "DEPTH_PERCEPTION":
        return "TABLE_EDGE"
    return "TARGET"


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


def _default_table_edge_obs() -> Dict[str, object]:
    return {
        "table_found": False,
        "edge_found": False,
        "confidence": 0.0,
        "yaw_err_rad": None,
        "dist_err_m": None,
        "edge_k": None,
        "edge_b": None,
        "depth_valid": False,
        "point_count": 0,
        "table_point_count": 0,
        "source": "vision_table_edge_manager",
        "type": "table_edge_obs",
    }


def _table_edge_obs_from_payload(payload: Optional[Dict[str, object]]) -> Dict[str, object]:
    base = _default_table_edge_obs()
    source = None
    if isinstance(payload, dict):
        source = payload.get("table_edge_obs") or payload.get("mock_table_edge_obs")
    if isinstance(source, dict):
        base.update(source)
    return base


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


def _table_edge_obs_from_results(results: Dict[str, object]) -> Optional[Dict[str, object]]:
    table_edge = (results or {}).get("table_edge_obs")
    if not isinstance(table_edge, dict):
        return None
    merged = _default_table_edge_obs()
    merged.update(table_edge)
    merged["type"] = "table_edge_obs"
    return merged


class SearchStagePlan(BaseStagePlan):
    """Stage plan for local target search and depth-based table-edge perception."""

    stage_name = "SEARCH"
    default_mode = "TRACK_LOCAL"

    def _resolve_mode(self, req: VisionReq) -> str:
        if req.mode_hint:
            return normalize_upper(req.mode_hint, self.default_mode)
        return "DEPTH_PERCEPTION" if _search_kind(req, self.default_mode) == "TABLE_EDGE" else self.default_mode

    def on_enter(self, req: VisionReq, ctx: StageContext) -> None:
        """Prepare target metadata and choose the initial local tracking mode."""
        super().on_enter(req, ctx)
        ctx.target_name = req.target or ctx.target_name
        ctx.current_mode = self._resolve_mode(req)
        ctx.interaction_id = None
        ctx.stage_state["search_kind"] = _search_kind(req, self.default_mode)
        ctx.stage_state["target_obs"] = _target_obs_from_payload(req.payload, ctx.target_name)
        ctx.stage_state["table_edge_obs"] = _table_edge_obs_from_payload(req.payload)

    def on_update(self, req: VisionReq, ctx: StageContext) -> Optional[StageOutput]:
        """Refresh target or search parameters without leaving SEARCH."""
        if req.target:
            ctx.target_name = req.target
        ctx.current_mode = self._resolve_mode(req)
        if isinstance(req.payload, dict):
            ctx.stage_state["search_kind"] = _search_kind(req, self.default_mode)
            ctx.stage_state["target_obs"] = _target_obs_from_payload(req.payload, ctx.target_name)
            ctx.stage_state["table_edge_obs"] = _table_edge_obs_from_payload(req.payload)
        return StageOutput()

    def tick(self, tick_input: StageTickInput, ctx: StageContext) -> Optional[StageOutput]:
        """Produce SEARCH stage outputs for target or table-edge perception."""
        results = dict(tick_input.results or {})
        search_kind = normalize_upper(ctx.stage_state.get("search_kind"), "TARGET")

        if search_kind == "TABLE_EDGE":
            table_edge_obs, source = resolve_stage_summary(
                results=results,
                stage_state=ctx.stage_state,
                state_key="table_edge_obs",
                default_factory=lambda: _table_edge_obs_from_payload(None),
                result_factory=_table_edge_obs_from_results,
                result_route="table_edge_obs",
            )
            return StageOutput(
                vision_obs=self.build_obs(
                    ctx,
                    status="RUNNING",
                    perception={"table_edge_obs": table_edge_obs},
                ),
                snapshot={
                    "generation": int(tick_input.generation),
                    "result_keys": sorted(results.keys()),
                    "source": source,
                    "search_kind": search_kind,
                },
            )

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
                "search_kind": search_kind,
            },
        )
