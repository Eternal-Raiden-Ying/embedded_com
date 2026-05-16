#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
from dataclasses import asdict
from typing import Any, Dict, Optional

from ..ipc.protocol import VisionReq
from .stages.base import BaseStagePlan, StageContext, StageOutput, StageTickInput, build_vision_obs, normalize_upper


class StageController:
    """Top-level business workflow coordinator for VISTA.

    This class will own stage registration, request routing, stage transitions,
    and handoff between stage logic and the mode/capability layer.
    """

    def __init__(self, logger=None, event_sink=None, mode_controller=None, runtime_service=None):
        self._plans: Dict[str, BaseStagePlan] = {}
        self._ctx = StageContext()
        self.logger = logger
        self._event_sink = event_sink
        self._mode_controller = mode_controller
        self._runtime_service = runtime_service
        self._last_applied_mode = "IDLE"
        self._last_interaction_state_key = None
        self._last_request_signature = None
        self.last_request_trace: Dict[str, Any] = {}
        if self._mode_controller is not None:
            try:
                self._last_applied_mode = normalize_upper(self._mode_controller.current_mode(), "IDLE")
            except Exception:
                self._last_applied_mode = "IDLE"

    def _log(self, message: str, **fields) -> None:
        if self.logger is not None:
            extra = fields or None
            self.logger.info(f"{message} | {extra}" if extra else message)

    def _sync_request_context(self, req: VisionReq) -> None:
        if req.session_id:
            self._ctx.session_id = req.session_id
        if req.req_id:
            self._ctx.req_id = req.req_id
        self._ctx.epoch = int(req.epoch)
        if req.target:
            self._ctx.target_name = req.target

    @staticmethod
    def _request_type(req: VisionReq) -> str:
        payload = req.payload if isinstance(req.payload, dict) else {}
        req_type = str(getattr(req, "req_type", "") or payload.get("req_type") or "").strip().lower()
        if req_type in {"mode_request", "target_update", "keepalive"}:
            return req_type
        op = normalize_upper(req.op, "START")
        if req.is_stop() or op in {"START", "STOP"}:
            return "mode_request"
        return "target_update"

    @staticmethod
    def _request_signature(req: VisionReq) -> tuple:
        payload = req.payload if isinstance(req.payload, dict) else {}
        roi = payload.get("locked_roi") or payload.get("roi") or payload.get("target_roi") or []
        return (
            StageController._request_type(req),
            str(req.session_id or "").strip(),
            str(req.target or "").strip(),
            normalize_upper(req.stage, "IDLE"),
            normalize_upper(req.mode_hint, ""),
            normalize_upper(payload.get("search_kind"), ""),
            str(payload.get("current_edge_id") or "").strip(),
            str(payload.get("locked_edge_id") or "").strip(),
            repr(roi),
        )

    def _store_request_trace(
        self,
        req: VisionReq,
        before: StageContext,
        *,
        req_type: str,
        idempotent: bool,
        reason: str,
    ) -> None:
        current_mode_before = normalize_upper(before.current_mode, "IDLE")
        current_mode_after = normalize_upper(self._ctx.current_mode, "IDLE")
        self.last_request_trace = {
            "req_id": req.req_id,
            "req_type": req_type,
            "session_id": req.session_id or self._ctx.session_id,
            "target": req.target or self._ctx.target_name,
            "requested_mode": normalize_upper(req.mode_hint, ""),
            "current_mode_before": current_mode_before,
            "current_mode_after": current_mode_after,
            "current_stage_before": normalize_upper(before.current_stage, "IDLE"),
            "current_stage_after": normalize_upper(self._ctx.current_stage, "IDLE"),
            "changed_mode": current_mode_before != current_mode_after,
            "idempotent": bool(idempotent),
            "reason": str(reason or ""),
        }

    def _finalize_request(
        self,
        plan: Optional[BaseStagePlan],
        output: Optional[StageOutput],
        req: VisionReq,
        before: StageContext,
        *,
        req_type: str,
        idempotent: bool,
        reason: str,
    ) -> Optional[StageOutput]:
        self._store_request_trace(req, before, req_type=req_type, idempotent=idempotent, reason=reason)
        self._last_request_signature = self._request_signature(req)
        return self._finalize_output(plan, output)

    def _event_context(self) -> Dict[str, Any]:
        return {
            "stage": self._ctx.current_stage,
            "mode": self._ctx.current_mode,
            "session_id": self._ctx.session_id,
            "req_id": self._ctx.req_id,
            "epoch": int(self._ctx.epoch),
            "interaction_id": self._ctx.interaction_id,
            "target": self._ctx.target_name,
        }

    def _emit_event(self, event: str, trigger: str = "stage_controller", data=None, **fields) -> None:
        if self._event_sink is None:
            return
        payload = self._event_context()
        payload.update(fields or {})
        payload["trigger"] = trigger
        payload["data"] = dict(data or {})
        try:
            self._event_sink(str(event or "EVENT").strip().upper(), payload)
        except Exception:
            pass

    def _clone_context(self) -> StageContext:
        return deepcopy(self._ctx)

    def _restore_context(self, snapshot: StageContext) -> None:
        self._ctx = snapshot
        self._last_applied_mode = normalize_upper(self._ctx.current_mode, "IDLE")
        self._last_interaction_state_key = None

    def _mode_apply_failed_output(
        self,
        request_op: str,
        request_stage: str,
        requested_mode: str,
        reason: str,
    ) -> StageOutput:
        failed_data = {
            "failure_type": "mode_apply_failed",
            "request_op": normalize_upper(request_op, "UNKNOWN"),
            "request_stage": normalize_upper(request_stage, self._ctx.current_stage),
            "requested_mode": normalize_upper(requested_mode, "IDLE"),
            "reason": str(reason or "mode_apply_failed"),
            "current_stage": normalize_upper(self._ctx.current_stage, "IDLE"),
            "current_mode": normalize_upper(self._ctx.current_mode, "IDLE"),
        }
        self._emit_event("MODE_APPLY_FAILED", data=failed_data)
        return StageOutput(
            vision_obs=build_vision_obs(
                self._ctx,
                status="FAILED",
                result=failed_data,
            ),
            signals={
                "mode_apply_failed": True,
                "reason": failed_data["reason"],
                "requested_mode": failed_data["requested_mode"],
            },
        )

    def _apply_context_mode(self, reason: str = "", force: bool = False) -> bool:
        target_mode = normalize_upper(self._ctx.current_mode, "IDLE")
        if not force and target_mode == normalize_upper(self._last_applied_mode, "IDLE"):
            return True
        if self._mode_controller is None:
            self._last_applied_mode = target_mode
            return True
        try:
            profile = self._mode_controller.switch_mode(
                target_mode,
                reason=reason,
                force=force,
                apply_mode_plan=(self._runtime_service.apply_mode_plan if self._runtime_service is not None else None),
            )
        except Exception:
            return False
        if profile is None:
            return False
        applied_mode = normalize_upper(getattr(profile, "name", target_mode), target_mode)
        self._ctx.current_mode = applied_mode
        self._last_applied_mode = applied_mode
        return True

    def set_runtime_mode(self, mode: str, reason: str = "", force: bool = False) -> bool:
        previous_mode = self._ctx.current_mode
        self._ctx.current_mode = normalize_upper(mode, "IDLE")
        ok = self._apply_context_mode(reason=reason or "runtime_mode", force=force)
        if not ok:
            self._ctx.current_mode = previous_mode
        else:
            self._publish_runtime_status()
        return ok

    def _emit_transition(
        self,
        action: str,
        request_op: str,
        from_stage: str,
        from_mode: str,
        request_stage: str,
    ) -> None:
        self._emit_event(
            "STAGE_TRANSITION",
            data={
                "action": str(action or "transition").strip().lower(),
                "request_op": str(request_op or "START").strip().upper(),
                "request_stage": str(request_stage or self._ctx.current_stage).strip().upper(),
                "from_stage": str(from_stage or "IDLE").strip().upper(),
                "to_stage": str(self._ctx.current_stage or "IDLE").strip().upper(),
                "from_mode": str(from_mode or "IDLE").strip().upper(),
                "to_mode": str(self._ctx.current_mode or "IDLE").strip().upper(),
            },
        )

    def _emit_output_events(self, output: Optional[StageOutput]) -> None:
        if output is None or output.vision_obs is None:
            return
        status = normalize_upper(output.vision_obs.get("status"), "RUNNING")
        if status not in {"WAITING_RESPONSE", "RESULT_READY", "FAILED"}:
            self._last_interaction_state_key = None
            return
        interaction = output.vision_obs.get("interaction") or {}
        interaction_id = interaction.get("interaction_id") or self._ctx.interaction_id
        state_key = (
            normalize_upper(self._ctx.current_stage, "IDLE"),
            status,
            str(interaction_id or ""),
            str(interaction.get("kind") or ""),
            int(interaction.get("round", 0) or 0),
        )
        if state_key == self._last_interaction_state_key:
            return
        self._last_interaction_state_key = state_key
        data = {
            "state": status,
            "obs_type": output.vision_obs.get("type"),
        }
        if interaction.get("kind"):
            data["kind"] = interaction.get("kind")
        if interaction.get("round") is not None:
            data["round"] = interaction.get("round")
        self._emit_event(
            "INTERACTION_STATE_CHANGED",
            interaction_id=interaction_id,
            data=data,
        )

    def _finalize_output(self, plan: Optional[BaseStagePlan], output: Optional[StageOutput]) -> Optional[StageOutput]:
        finalized = self._ensure_output(plan, output)
        if finalized is not None and finalized.signals and self._runtime_service is not None:
            try:
                self._runtime_service.push_stage_signals(dict(finalized.signals or {}))
            except Exception:
                pass
        if finalized is not None and finalized.effects:
            self._publish_effects(finalized.effects)
        self._publish_runtime_status()
        self._emit_output_events(finalized)
        return finalized

    def _publish_effects(self, effects) -> None:
        if self._runtime_service is None:
            return
        for effect in list(effects or ()):
            if not isinstance(effect, dict):
                continue
            effect_type = normalize_upper(effect.get("type"), "")
            route = str(effect.get("route") or "").strip()
            payload = dict(effect.get("payload") or {})
            if effect_type != "PUBLISH_EVENT" or not route:
                continue
            try:
                self._runtime_service.publish_event(route, payload)
            except Exception:
                pass

    def _runtime_status_payload(self) -> Dict[str, Any]:
        payload = {
            "stage": normalize_upper(self._ctx.current_stage, "IDLE"),
            "mode": normalize_upper(self._ctx.current_mode, "IDLE"),
            "session_id": self._ctx.session_id,
            "req_id": self._ctx.req_id,
            "epoch": int(self._ctx.epoch),
            "interaction_id": self._ctx.interaction_id,
            "target": self._ctx.target_name,
        }
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
            if key in self._ctx.stage_state:
                payload[key] = self._ctx.stage_state.get(key)
        return payload

    def _publish_runtime_status(self) -> None:
        if self._runtime_service is None:
            return
        try:
            self._runtime_service.publish_result("runtime_status", self._runtime_status_payload())
        except Exception:
            pass

    def _transition_to(self, stage: str, req: Optional[VisionReq] = None) -> Optional[BaseStagePlan]:
        target = normalize_upper(stage, "IDLE")
        current = self.current_plan()
        if current is not None and normalize_upper(current.stage_name) != target:
            current.on_exit(self._ctx)
        self._last_interaction_state_key = None
        if target == "IDLE":
            self._ctx = StageContext(
                current_stage="IDLE",
                current_mode="IDLE",
                session_id=self._ctx.session_id,
                req_id=self._ctx.req_id,
                epoch=int(self._ctx.epoch),
            )
            return None
        plan = self.plan_for(target)
        if plan is None:
            return None
        self._ctx.stage_state.clear()
        self._ctx.current_stage = plan.stage_name
        if req is not None:
            plan.on_enter(req, self._ctx)
        else:
            self._ctx.current_mode = plan.default_mode
        return plan

    def _mode_available(self, name: str) -> bool:
        """Check whether a mode profile is registered and available."""
        if self._mode_controller is None:
            return True
        resolver = getattr(self._mode_controller, "resolve_profile", None)
        if callable(resolver):
            return resolver(normalize_upper(name, "IDLE")) is not None
        return True

    def _ensure_output(self, plan: Optional[BaseStagePlan], output: Optional[StageOutput]) -> Optional[StageOutput]:
        if plan is None and output is None:
            return None
        if output is None:
            output = StageOutput()
        return output

    def register_plan(self, plan: BaseStagePlan) -> None:
        """Register one stage plan instance by its declared stage name."""
        plan.mode_available = self._mode_available
        self._plans[str(plan.stage_name).upper()] = plan

    def register_default_plans(self, plans: Dict[str, BaseStagePlan]) -> None:
        """Register a prebuilt stage plan mapping during app bootstrap."""
        for plan in plans.values():
            self.register_plan(plan)

    def plan_for(self, stage: str) -> Optional[BaseStagePlan]:
        """Return the stage plan for a stage name if it is registered."""
        return self._plans.get(str(stage or "").upper())

    def current_plan(self) -> Optional[BaseStagePlan]:
        """Return the currently active stage plan."""
        return self.plan_for(self._ctx.current_stage)

    def context(self) -> StageContext:
        """Expose the mutable runtime context owned by the controller."""
        return self._ctx

    def activate_stage(self, stage: str, req: Optional[VisionReq] = None) -> Optional[BaseStagePlan]:
        """Prepare a stage for execution and invoke its enter hook.

        The full transition policy is intentionally deferred to later work.
        """
        return self._transition_to(stage, req=req)

    def handle_request(self, req: VisionReq) -> Optional[StageOutput]:
        """Route a protocol request to the currently active or target stage.

        Later implementation should normalize START/UPDATE/RESPOND/STOP and
        handle cross-stage transitions here.
        """
        req_type = self._request_type(req)
        self._sync_request_context(req)
        stage = normalize_upper(req.stage, "IDLE")
        op = normalize_upper(req.op, "START")
        current = self.current_plan()
        prev_stage = normalize_upper(self._ctx.current_stage, "IDLE")
        prev_mode = normalize_upper(self._ctx.current_mode, "IDLE")
        ctx_before = self._clone_context()
        request_signature = self._request_signature(req)
        same_stage = normalize_upper(self._ctx.current_stage, "IDLE") == stage
        requested_mode = normalize_upper(req.mode_hint, "")
        same_mode = not requested_mode or normalize_upper(self._ctx.current_mode, "IDLE") == requested_mode
        idempotent = bool(self._last_request_signature == request_signature and same_stage and same_mode)

        if req_type == "keepalive":
            return self._finalize_request(
                current,
                StageOutput(signals={"request_op": op, "req_type": req_type}),
                req,
                ctx_before,
                req_type=req_type,
                idempotent=True,
                reason="keepalive",
            )

        if idempotent and current is not None and same_stage:
            output = current.on_update(req, self._ctx)
            requested_mode = normalize_upper(self._ctx.current_mode, "IDLE")
            if not self._apply_context_mode(reason="idempotent_update", force=False):
                self._restore_context(ctx_before)
                return self._finalize_request(
                    self.current_plan(),
                    self._mode_apply_failed_output(op, stage, requested_mode, reason="idempotent_mode_apply_failed"),
                    req,
                    ctx_before,
                    req_type=req_type,
                    idempotent=True,
                    reason="idempotent_mode_apply_failed",
                )
            self._log("stage idempotent_update", stage=stage, mode=self._ctx.current_mode, req_id=req.req_id)
            return self._finalize_request(
                current,
                output or StageOutput(signals={"request_op": op, "req_type": req_type, "idempotent": True}),
                req,
                ctx_before,
                req_type=req_type,
                idempotent=True,
                reason="idempotent_update",
            )

        if req.is_stop() or op == "STOP":
            stop_output = current.on_stop(req, self._ctx) if current is not None else None
            self._transition_to("IDLE")
            if not self._apply_context_mode(reason="stop", force=False):
                self._restore_context(ctx_before)
                return self._finalize_request(
                    self.current_plan(),
                    self._mode_apply_failed_output(op, stage, "IDLE", reason="stop_mode_apply_failed"),
                    req,
                    ctx_before,
                    req_type=req_type,
                    idempotent=False,
                    reason="stop_mode_apply_failed",
                )
            self._log("stage stop", stage=stage, req_id=req.req_id, session_id=req.session_id)
            self._emit_transition("stop", op, prev_stage, prev_mode, stage)
            return self._finalize_request(
                current,
                stop_output or StageOutput(),
                req,
                ctx_before,
                req_type=req_type,
                idempotent=False,
                reason="stop",
            )

        if op == "RESPOND":
            plan = current if current is not None and normalize_upper(current.stage_name) == stage else self.plan_for(stage)
            if plan is None:
                self._store_request_trace(req, ctx_before, req_type=req_type, idempotent=False, reason="respond_no_plan")
                return None
            output = plan.on_respond(req, self._ctx)
            requested_mode = normalize_upper(self._ctx.current_mode, "IDLE")
            if not self._apply_context_mode(reason="respond", force=False):
                self._restore_context(ctx_before)
                return self._finalize_request(
                    self.current_plan(),
                    self._mode_apply_failed_output(op, stage, requested_mode, reason="respond_mode_apply_failed"),
                    req,
                    ctx_before,
                    req_type=req_type,
                    idempotent=False,
                    reason="respond_mode_apply_failed",
                )
            self._log("stage respond", stage=self._ctx.current_stage, mode=self._ctx.current_mode, req_id=req.req_id)
            self._emit_event(
                "INTERACTION_RESPONSE_HANDLED",
                interaction_id=req.interaction_id or self._ctx.interaction_id,
                data={
                    "decision": normalize_upper((req.response or {}).get("decision"), "REJECT"),
                    "request_stage": stage,
                },
            )
            return self._finalize_request(
                plan,
                output,
                req,
                ctx_before,
                req_type=req_type,
                idempotent=False,
                reason="respond",
            )

        if current is None or normalize_upper(self._ctx.current_stage) != stage:
            plan = self._transition_to(stage, req=req)
            requested_mode = normalize_upper(self._ctx.current_mode, "IDLE")
            if not self._apply_context_mode(reason="enter", force=False):
                self._restore_context(ctx_before)
                return self._finalize_request(
                    self.current_plan(),
                    self._mode_apply_failed_output(op, stage, requested_mode, reason="enter_mode_apply_failed"),
                    req,
                    ctx_before,
                    req_type=req_type,
                    idempotent=False,
                    reason="enter_mode_apply_failed",
                )
            self._log("stage enter", stage=stage, mode=self._ctx.current_mode, req_id=req.req_id)
            self._emit_transition("enter", op, prev_stage, prev_mode, stage)
            return self._finalize_request(
                plan,
                StageOutput(signals={"request_op": op, "transition": "enter", "req_type": req_type}),
                req,
                ctx_before,
                req_type=req_type,
                idempotent=False,
                reason="enter",
            )

        if op == "START":
            plan = self._transition_to(stage, req=req)
            requested_mode = normalize_upper(self._ctx.current_mode, "IDLE")
            if not self._apply_context_mode(reason="restart", force=False):
                self._restore_context(ctx_before)
                return self._finalize_request(
                    self.current_plan(),
                    self._mode_apply_failed_output(op, stage, requested_mode, reason="restart_mode_apply_failed"),
                    req,
                    ctx_before,
                    req_type=req_type,
                    idempotent=False,
                    reason="restart_mode_apply_failed",
                )
            self._log("stage restart", stage=stage, mode=self._ctx.current_mode, req_id=req.req_id)
            self._emit_transition("restart", op, prev_stage, prev_mode, stage)
            return self._finalize_request(
                plan,
                StageOutput(signals={"request_op": op, "transition": "restart", "req_type": req_type}),
                req,
                ctx_before,
                req_type=req_type,
                idempotent=False,
                reason="restart",
            )

        output = current.on_update(req, self._ctx)
        requested_mode = normalize_upper(self._ctx.current_mode, "IDLE")
        if not self._apply_context_mode(reason="update", force=False):
            self._restore_context(ctx_before)
            return self._finalize_request(
                self.current_plan(),
                self._mode_apply_failed_output(op, stage, requested_mode, reason="update_mode_apply_failed"),
                req,
                ctx_before,
                req_type=req_type,
                idempotent=False,
                reason="update_mode_apply_failed",
            )
        self._log("stage update", stage=stage, mode=self._ctx.current_mode, req_id=req.req_id)
        return self._finalize_request(
            current,
            output,
            req,
            ctx_before,
            req_type=req_type,
            idempotent=False,
            reason="update",
        )

    def tick(self, tick_input: StageTickInput) -> Optional[StageOutput]:
        """Execute one stage tick and return the stage-level output envelope."""
        plan = self.current_plan()
        if plan is None:
            return None
        ctx_before = self._clone_context()
        output = plan.tick(tick_input, self._ctx)
        requested_mode = normalize_upper(self._ctx.current_mode, "IDLE")
        if not self._apply_context_mode(reason="tick", force=False):
            self._restore_context(ctx_before)
            return self._finalize_output(
                self.current_plan(),
                self._mode_apply_failed_output("TICK", self._ctx.current_stage, requested_mode, reason="tick_mode_apply_failed"),
            )
        return self._finalize_output(plan, output)

    def reset(self) -> None:
        """Reset stage runtime state to a clean idle context."""
        self._ctx = StageContext()
        self._last_applied_mode = "IDLE"
        self._last_interaction_state_key = None

    def snapshot(self) -> Dict[str, object]:
        """Return a lightweight diagnostic snapshot for logging and heartbeat."""
        data = asdict(self._ctx)
        data["registered_stages"] = sorted(self._plans.keys())
        data["last_applied_mode"] = self._last_applied_mode
        if self._mode_controller is not None:
            getter = getattr(self._mode_controller, "snapshot", None)
            if callable(getter):
                try:
                    data["mode_controller"] = dict(getter() or {})
                except Exception:
                    data["mode_controller"] = {"error": "snapshot_failed"}
        return data
