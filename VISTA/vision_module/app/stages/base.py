#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ...ipc.protocol import VisionObs, VisionReq


@dataclass
class StageContext:
    """Mutable session-scoped state shared across stage plans."""

    current_stage: str = "IDLE"
    current_mode: str = "IDLE"
    session_id: Optional[str] = None
    req_id: Optional[str] = None
    epoch: int = 0
    target_name: Optional[str] = None
    interaction_id: Optional[str] = None
    server_status: str = "unknown"  # "unknown" | "ready" | "error" — remote grasp server health
    stage_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageTickInput:
    """Read-only inputs available to a stage on each control loop tick."""

    ts: float
    generation: int = 0
    results: Dict[str, Any] = field(default_factory=dict)
    signals: Dict[str, Any] = field(default_factory=dict)
    snapshot: Dict[str, Any] = field(default_factory=dict)

    def result(self, key: str, default=None):
        return (self.results or {}).get(key, default)

    def has_result(self, key: str) -> bool:
        return key in (self.results or {})

    def signal(self, key: str, default=None):
        if key in (self.signals or {}):
            return self.signals.get(key, default)
        return default


@dataclass
class StageOutput:
    """Stage-level decision envelope returned from a stage tick."""

    vision_obs: Optional[Dict[str, Any]] = None
    signals: Dict[str, Any] = field(default_factory=dict)
    effects: List[Dict[str, Any]] = field(default_factory=list)
    snapshot: Dict[str, Any] = field(default_factory=dict)
    next_stage: Optional[str] = None  # set to auto-transition after tick (e.g. INIT → IDLE)

    def has_outbound(self) -> bool:
        return self.vision_obs is not None

    def signal(self, key: str, default=None):
        if key in (self.signals or {}):
            return self.signals.get(key, default)
        return default


def normalize_upper(value: Any, default: str = "") -> str:
    text = str(value or default).strip().upper()
    return text or str(default).strip().upper()


def next_interaction_id() -> str:
    return f"ia_{int(time.time() * 1000)}"


def build_vision_obs(
    ctx: StageContext,
    status: str,
    perception: Optional[Dict[str, Any]] = None,
    proposal: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
    interaction: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return VisionObs(
        ts=time.time(),
        stage=ctx.current_stage,
        mode=ctx.current_mode,
        status=normalize_upper(status, "RUNNING"),
        session_id=ctx.session_id,
        req_id=ctx.req_id,
        epoch=int(ctx.epoch),
        interaction=interaction,
        perception=perception,
        proposal=proposal,
        result=result,
    ).to_dict()


def resolve_stage_summary(
    results: Dict[str, Any],
    stage_state: Dict[str, Any],
    state_key: str,
    default_factory,
    result_factory,
    result_route: str = "local_perception",
):
    summary = result_factory(dict(results or {}))
    source = "results" if summary is not None and result_route in (results or {}) else "stage_state"
    if summary is None:
        summary = dict(stage_state.get(state_key) or default_factory())
    else:
        summary = dict(summary)
    stage_state[state_key] = dict(summary)
    return summary, source


class BaseStagePlan(ABC):
    """Base interface for business-stage orchestration logic."""

    stage_name: str = "IDLE"
    default_mode: str = "IDLE"

    # Callback to check if a mode profile is registered (injected by StageController)
    mode_available: Optional[Callable[[str], bool]] = None

    # routes subscribed in ALL modes under this stage
    common_routes: Tuple[str, ...] = ("frame_meta", "runtime_status")

    # additional routes per specific mode; key=mode_name, value=tuple of route names
    optional_routes: Dict[str, Tuple[str, ...]] = {}

    def on_enter(self, req: VisionReq, ctx: StageContext) -> None:
        """Initialize stage-owned state when the stage becomes active."""
        _ = req
        ctx.current_stage = self.stage_name
        ctx.current_mode = self.default_mode

    def on_update(self, req: VisionReq, ctx: StageContext) -> Optional[StageOutput]:
        """Handle START or UPDATE messages that keep execution in this stage."""
        _ = (req, ctx)
        return None

    def on_respond(self, req: VisionReq, ctx: StageContext) -> Optional[StageOutput]:
        """Handle RESPOND messages for multi-round interaction stages."""
        _ = (req, ctx)
        return None

    def on_stop(self, req: VisionReq, ctx: StageContext) -> Optional[StageOutput]:
        """Handle STOP semantics before the controller transitions to idle."""
        _ = req
        ctx.current_stage = "IDLE"
        ctx.current_mode = "IDLE"
        return None

    def on_exit(self, ctx: StageContext) -> None:
        """Release stage-local state before another stage takes control."""
        ctx.stage_state.clear()

    def subscribed_routes(self, mode: str) -> Tuple[str, ...]:
        """Return the union of common and optional routes for the given mode."""
        optional = self.optional_routes.get(mode, ())
        return self.common_routes + optional

    def build_obs(
        self,
        ctx: StageContext,
        status: str,
        perception: Optional[Dict[str, Any]] = None,
        proposal: Optional[Dict[str, Any]] = None,
        result: Optional[Dict[str, Any]] = None,
        interaction: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build one `vision_obs` envelope using the shared stage context."""
        return build_vision_obs(
            ctx,
            status=status,
            perception=perception,
            proposal=proposal,
            result=result,
            interaction=interaction,
        )

    def tick(self, tick_input: StageTickInput, ctx: StageContext) -> Optional[StageOutput]:
        """Run one stage iteration and optionally emit a stage output."""
        _ = (tick_input, ctx)
        return None
