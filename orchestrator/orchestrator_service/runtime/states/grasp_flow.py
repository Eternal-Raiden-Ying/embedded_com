#!/usr/bin/env python3
"""Config-driven fixed grasp sequence state handlers."""
from __future__ import annotations

from ...ipc.protocol import ArmCommand
from ..common import monotonic_ts
from ..context import State
from ..controller import MotionDecision


class GraspFlowMixin:
    def _tick_grasp(self) -> MotionDecision:
        now_m = monotonic_ts()
        substate = str(self.ctx.grasp_substate or "")
        if substate == "RECIPE_WAIT_ARM_READY":
            return self._tick_recipe_wait_arm_ready(now_m)
        if substate == "RECIPE_SEND_STEP":
            return self._tick_recipe_send_step(now_m)
        if substate == "RECIPE_WAIT_STEP":
            return self._tick_recipe_wait_step(now_m)
        if substate == "RECIPE_SETTLE":
            return self._tick_recipe_settle(now_m)
        if substate == "RECIPE_DONE":
            return self._tick_recipe_done(now_m)
        self._enter_error_recovery(f"invalid_grasp_recipe_substate:{substate or 'empty'}")
        return self.controller.stop_cmd("GRASP")

    def _selected_grasp_recipe(self):
        recipe = getattr(self.ctx, "selected_grasp_recipe", None)
        return recipe if bool(getattr(self.ctx, "grasp_recipe_ready", False)) else None

    def _selected_grasp_step(self):
        recipe = self._selected_grasp_recipe()
        index = int(getattr(self.ctx, "grasp_recipe_step_index", 0) or 0)
        steps = tuple(getattr(recipe, "steps", ()) or ()) if recipe is not None else ()
        return recipe, index, (steps[index] if 0 <= index < len(steps) else None)

    def _tick_recipe_wait_arm_ready(self, now_m: float) -> MotionDecision:
        del now_m
        recipe = self._selected_grasp_recipe()
        if recipe is None:
            self._emit_tts_event("RECIPE_NOT_READY", state=State.GRASP.value)
            self._transition(State.DONE, "RECIPE_NOT_READY")
            return self.controller.stop_cmd("DONE")
        if not bool(getattr(self.ctx, "arm_serial_ready", False)):
            self._enter_error_recovery("arm_serial_not_ready")
            return self.controller.stop_cmd("GRASP")
        self.ctx.grasp_substate = "RECIPE_SEND_STEP"
        return self._tick_recipe_send_step(monotonic_ts())

    def _tick_recipe_send_step(self, now_m: float) -> MotionDecision:
        recipe, index, step = self._selected_grasp_step()
        if recipe is None:
            self._emit_tts_event("RECIPE_NOT_READY", state=State.GRASP.value)
            self._transition(State.DONE, "RECIPE_NOT_READY")
            return self.controller.stop_cmd("DONE")
        if step is None:
            self.ctx.grasp_substate = "RECIPE_DONE"
            return self._tick_recipe_done(now_m)
        self.ctx.arm_response = None
        self.ctx.grasp_recipe_step_send_mono = now_m
        self.ctx.grasp_timeout_mono = now_m + float(step.timeout_s)
        self.ctx.grasp_substate = "RECIPE_WAIT_STEP"
        self._log(
            "info",
            f"grasp_recipe_step_send recipe_name={recipe.name} step_index={index} "
            f"step_command={step.command} send_mono={now_m:.6f}",
        )
        arm_cmd = ArmCommand(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            command=step.command,
            recipe_name=recipe.name,
            recipe_step_index=index,
            expect_ack=bool(step.expect_ack),
            success_ack=step.success_ack,
            timeout_s=float(step.timeout_s),
        )
        decision = MotionDecision(cmd=self.controller.stop_cmd("GRASP").cmd, arm_cmd=arm_cmd)
        decision.control_summary = {
            "grasp_source": "recipe",
            "recipe_name": recipe.name,
            "step_index": index,
            "step_command": step.command,
            "arm_serial_ready": bool(self.ctx.arm_serial_ready),
        }
        return decision

    def _tick_recipe_wait_step(self, now_m: float) -> MotionDecision:
        recipe, index, step = self._selected_grasp_step()
        if recipe is None or step is None:
            self._enter_error_recovery("grasp_recipe_runtime_missing")
            return self.controller.stop_cmd("GRASP")
        resp = self.ctx.arm_response
        if resp is None:
            if now_m > float(self.ctx.grasp_timeout_mono or 0.0):
                self._enter_error_recovery(f"grasp_recipe_step_timeout:{recipe.name}:{index}")
            return self.controller.stop_cmd("GRASP")
        self.ctx.arm_response = None
        duration_ms = max(
            0.0,
            (now_m - float(self.ctx.grasp_recipe_step_send_mono or now_m)) * 1000.0,
        )
        if not bool(getattr(resp, "ok", False)):
            self._log(
                "error",
                f"grasp_recipe_step_result recipe_name={recipe.name} step_index={index} "
                f"step_duration_ms={duration_ms:.1f} step_result=failed "
                f"status={getattr(resp, 'parsed_status', '')}",
            )
            self._enter_error_recovery(f"grasp_recipe_step_failed:{recipe.name}:{index}")
            return self.controller.stop_cmd("GRASP")
        self._log(
            "info",
            f"grasp_recipe_step_result recipe_name={recipe.name} step_index={index} "
            f"ack_mono={now_m:.6f} step_duration_ms={duration_ms:.1f} step_result=success",
        )
        self.ctx.grasp_recipe_step_index = index + 1
        if float(step.settle_after_s) > 0.0:
            self.ctx.grasp_recipe_settle_until_mono = now_m + float(step.settle_after_s)
            self.ctx.grasp_substate = "RECIPE_SETTLE"
        else:
            self.ctx.grasp_substate = "RECIPE_SEND_STEP"
        return self.controller.stop_cmd("GRASP")

    def _tick_recipe_settle(self, now_m: float) -> MotionDecision:
        if now_m >= float(getattr(self.ctx, "grasp_recipe_settle_until_mono", 0.0) or 0.0):
            self.ctx.grasp_substate = "RECIPE_SEND_STEP"
        return self.controller.stop_cmd("GRASP")

    def _tick_recipe_done(self, now_m: float) -> MotionDecision:
        del now_m
        recipe = self._selected_grasp_recipe()
        self.ctx.grasp_source = "recipe"
        self.ctx.carrying_object = True
        self.ctx.carried_target = str(getattr(recipe, "name", "") or self.ctx.canonical_target or "")
        self.ctx.grasp_substate = "RECIPE_DONE"
        self._transition(State.DONE, f"arm_motion_done recipe={self.ctx.carried_target}")
        self._queue_tts("抓取完成")
        return self.controller.stop_cmd("DONE")
