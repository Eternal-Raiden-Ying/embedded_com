#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

from ...ipc.protocol import ArmCommand, TargetObs, make_vision_req
from ...utils.target_utils import resolve_target
from ..common import monotonic_ts
from ..context import State
from ..controller import MotionDecision


class ReturnPlaceMixin:
    def _return_place_summary(
        self,
        mode: str,
        cmd: Any,
        *,
        reason: str,
        obs: Optional[TargetObs] = None,
        allow_forward: bool = False,
        allow_rotate: bool = False,
    ) -> Dict[str, Any]:
        summary = self.controller._summary(mode, cmd, None, reason=reason)
        conf = None
        found = False
        if obs is not None:
            found, conf = self._basket_found(obs)
        summary.update(
            {
                "return_place_phase": mode,
                "return_place_target": "basket",
                "carried_target": getattr(self.ctx, "carried_target", None),
                "basket_found": bool(found),
                "basket_conf": conf,
                "allow_forward": bool(allow_forward),
                "allow_rotate": bool(allow_rotate),
                "allow_lateral": False,
                "forward_block_reason": "" if allow_forward and float(getattr(cmd, "vx_mps", 0.0) or 0.0) > 1e-9 else "",
                "rotate_block_reason": "" if allow_rotate and abs(float(getattr(cmd, "wz_radps", 0.0) or 0.0)) > 1e-9 else "",
                "lateral_block_reason": "return_place_vy_disabled",
                "return_place_ignores_table_stale": True,
                "stale_source": "target" if obs is not None else "",
                "vx_mps": float(getattr(cmd, "vx_mps", 0.0) or 0.0),
                "vy_mps": 0.0,
                "wz_radps": float(getattr(cmd, "wz_radps", 0.0) or 0.0),
            }
        )
        return summary

    def _tick_post_grasp_turn_180(self) -> MotionDecision:
        if False:
            self._transition(State.SEARCH_BASKET, "post_grasp_turn_disabled")
            return self.controller.stop_cmd("SEARCH_BASKET")
        duration = 3.5
        started_mono = float(getattr(self.ctx, "post_grasp_turn_started_mono", 0.0) or 0.0)
        elapsed = 0.0 if started_mono <= 0.0 else max(0.0, monotonic_ts() - started_mono)
        direction = "left"
        sign = -1.0 if direction == "right" else 1.0
        wz = sign * 0.45
        if elapsed >= duration:
            self._transition(State.SEARCH_BASKET, "post_grasp_turn_done turn_mode=open_loop_time")
            return self.controller.stop_cmd("SEARCH_BASKET")
        self._log(
            "info",
            f"[RETURN_PLACE][TURN_180] elapsed={elapsed:.2f} duration={duration:.2f} "
            f"wz={wz:.3f} direction={direction} turn_mode=open_loop_time",
        )
        cmd = self.controller._cmd("POST_GRASP_TURN_180", vx=0.0, vy=0.0, wz=wz)
        return MotionDecision(
            cmd=cmd,
            control_summary=self._return_place_summary(
                "POST_GRASP_TURN_180",
                cmd,
                reason="return_place_turn_180_open_loop_time",
                allow_rotate=True,
            ),
        )

    def _tick_search_basket(self) -> MotionDecision:
        basket_spec = resolve_target("basket")
        if basket_spec is None:
            self._log("error", "[RETURN_PLACE][SEARCH_BASKET] unsupported_basket_target reason=basket_not_supported")
            self._enter_error_recovery("basket_not_supported")
            return self.controller.stop_cmd("ERROR_RECOVERY", brake=True)
        first_tick = self.ctx.basket_search_start_ts <= 0.0
        if first_tick:
            self.ctx.basket_search_start_ts = monotonic_ts()
        self._queue_vision_req(self._basket_vision_req(), force=first_tick)
        obs = self._fresh_target_obs()
        found, conf = self._basket_found(obs)
        elapsed = monotonic_ts() - float(self.ctx.basket_search_start_ts or monotonic_ts())
        self._log(
            "info",
            f"[RETURN_PLACE][SEARCH_BASKET] found={str(found).lower()} conf={conf} elapsed={elapsed:.2f} "
            f"return_place_target=basket carried_target={getattr(self.ctx, 'carried_target', None)}",
        )
        if found:
            self.ctx.basket_approach_stable_count = 0
            self._transition(State.APPROACH_BASKET, "basket_found")
            return self.controller.stop_cmd("APPROACH_BASKET")
        timeout_s = 15.0
        if elapsed >= timeout_s:
            self._enter_error_recovery("basket_search_timeout")
            return self.controller.stop_cmd("ERROR_RECOVERY", brake=True)
        wz = 0.25
        cmd = self.controller._cmd("SEARCH_BASKET", vx=0.0, vy=0.0, wz=wz)
        return MotionDecision(
            cmd=cmd,
            control_summary=self._return_place_summary(
                "SEARCH_BASKET",
                cmd,
                reason="basket_search_rotate",
                obs=obs,
                allow_rotate=True,
            ),
        )

    def _tick_approach_basket(self) -> MotionDecision:
        self._queue_vision_req(self._basket_vision_req(), force=False)
        obs = self._fresh_target_obs()
        found, conf = self._basket_found(obs)
        elapsed = self._state_elapsed()
        cx, h, area = self._basket_bbox_metrics(obs) if obs is not None else (None, None, None)
        self._log(
            "info",
            f"[RETURN_PLACE][BASKET_OBS] found={str(found).lower()} cx={cx} area={area} h={h} conf={conf}",
        )
        if not found:
            self.ctx.basket_approach_stable_count = 0
            if elapsed >= 20.0:
                self._enter_error_recovery("basket_approach_timeout")
                return self.controller.stop_cmd("ERROR_RECOVERY", brake=True)
            return self.controller.stop_cmd("APPROACH_BASKET")

        target_x = 0.50
        tol = 0.08
        err = None if cx is None else float(cx) - target_x
        centered = bool(err is not None and abs(err) <= tol)
        area_stop = bool(area is not None and area >= 0.18)
        height_stop = bool(h is not None and h >= 0.38)
        if area_stop or height_stop:
            self._log(
                "info",
                f"[RETURN_PLACE][BASKET_STOP_REACHED] reason=area_or_height cx={cx} area={area} h={h}",
            )
            self._transition(State.PLACE_TO_BASKET, "basket_stop_reached_area_or_height")
            return self.controller.stop_cmd("PLACE_TO_BASKET")
        self.ctx.basket_approach_stable_count = 0

        if elapsed >= 20.0:
            self._enter_error_recovery("basket_approach_timeout")
            return self.controller.stop_cmd("ERROR_RECOVERY", brake=True)

        vx = 0.0
        wz = 0.0
        if centered:
            vx = 0.08
            if area is not None and area >= 0.5 * 0.18:
                vx = float(getattr(self.cfg, "basket_approach_vx_slow_mps", 0.035) or 0.035)
        elif err is not None:
            wz = max(-0.18, min(0.18, -float(err) * 0.45))
        self._log(
            "info",
            f"[RETURN_PLACE][BASKET_APPROACH] cx={cx} err={err} area={area} h={h} "
            f"stable={int(self.ctx.basket_approach_stable_count)} vx={vx:.3f} wz={wz:.3f} "
            f"basket_internal_align_center_x={target_x:.2f}",
        )
        cmd = self.controller._cmd("APPROACH_BASKET", vx=vx, vy=0.0, wz=wz)
        summary = self._return_place_summary(
            "APPROACH_BASKET",
            cmd,
            reason="basket_bbox_approach",
            obs=obs,
            allow_forward=bool(vx > 1e-9),
            allow_rotate=bool(abs(wz) > 1e-9),
        )
        summary.update(
            {
                "basket_found": bool(found),
                "basket_conf": conf,
                "basket_center_x_norm": cx,
                "basket_internal_align_center_x": target_x,
                "basket_err_x": err,
                "basket_centered_ok": bool(centered),
                "basket_bbox_area_norm": area,
                "basket_bbox_height_norm": h,
                "basket_approach_stable_count": int(self.ctx.basket_approach_stable_count),
            }
        )
        return MotionDecision(cmd=cmd, control_summary=summary)

    def _tick_place_to_basket(self) -> MotionDecision:
        if self.ctx.basket_place_substate != "AWAITING_ARM":
            self.ctx.basket_place_substate = "AWAITING_ARM"
            self.ctx.basket_place_timeout_mono = monotonic_ts() + 10.0
            arm_cmd = ArmCommand(
                12.0,
                0.0,
                8.0,
                float(getattr(self.cfg, "place_pose_pitch_deg", 0.0) or 0.0),
                float(getattr(self.cfg, "place_pose_roll_deg", 0.0) or 0.0),
                80.0,
                800,
                command="POSE",
            )
            self._log("info", "[RETURN_PLACE][PLACE_SEND] source=configured_place_pose unit=cm_deg_claw line=POSE")
            decision = MotionDecision(cmd=self.controller.stop_cmd("PLACE_TO_BASKET").cmd, arm_cmd=arm_cmd)
            decision.control_summary = {
                "return_place_phase": "place_send",
                "return_place_target": "basket",
                "carried_target": getattr(self.ctx, "carried_target", None),
                "source": "configured_place_pose",
                "vx_mps": 0.0,
                "vy_mps": 0.0,
                "wz_radps": 0.0,
            }
            return decision
        resp = self.ctx.arm_response
        if resp is not None:
            self.ctx.arm_response = None
            parsed_status = str(getattr(resp, "parsed_status", "") or getattr(resp, "message", "") or "").strip().upper()
            raw = str(getattr(resp, "raw_line", "") or "")
            if bool(getattr(resp, "ok", False)) and parsed_status == "OK_POSE":
                self._log("info", f"[RETURN_PLACE][PLACE_DONE] parsed_status={parsed_status} raw={raw!r}")
                self.ctx.clear_carrying_object()
                self._transition(State.DONE, "basket_place_done")
                self._queue_tts("已放入篮子")
                return self.controller.stop_cmd("DONE")
            self._log("warn", f"[RETURN_PLACE][PLACE_FAILED] parsed_status={parsed_status} raw={raw!r}")
            self._enter_error_recovery(f"place_to_basket_failed:{parsed_status or 'unknown'}")
            return self.controller.stop_cmd("ERROR_RECOVERY", brake=True)
        if monotonic_ts() > float(self.ctx.basket_place_timeout_mono or 0.0):
            self._log("warn", "[RETURN_PLACE][PLACE_FAILED] reason=arm_pose_timeout")
            self._enter_error_recovery("place_to_basket_arm_pose_timeout")
            return self.controller.stop_cmd("ERROR_RECOVERY", brake=True)
        return self.controller.stop_cmd("PLACE_TO_BASKET")

    def _basket_vision_req(self) -> Dict[str, Any]:
        spec = resolve_target("basket")
        class_id = spec.class_id if spec is not None else None
        return make_vision_req(
            target="basket",
            session_id=self.ctx.active_session_id,
            epoch=self.ctx.active_epoch,
            op="START",
            stage="SEARCH",
            mode_hint="FIND_OBJECT",
            req_type="mode_request",
            payload={
                "search_kind": "RETURN_PLACE_BASKET",
                "target": "basket",
                "canonical_target": "basket",
                "class_name": "basket",
                "class_id": class_id,
                "return_place_target": "basket",
                "carried_target": getattr(self.ctx, "carried_target", None),
                "need_depth": False,
                "orchestrator_state": self.ctx.state.value,
            },
        )

    def _basket_found(self, obs: Optional[TargetObs]) -> Tuple[bool, Optional[float]]:
        if obs is None or not bool(getattr(obs, "found", False)):
            return False, None
        name = str(getattr(obs, "canonical_target", "") or getattr(obs, "matched_cls", "") or getattr(obs, "target", "") or "").strip()
        spec = resolve_target(name)
        matched_id = getattr(obs, "matched_class_id", None)
        basket_id = resolve_target("basket").class_id if resolve_target("basket") is not None else None
        cls_ok = bool(spec is not None and spec.canonical_target == "basket")
        if not cls_ok and basket_id is not None and matched_id is not None:
            try:
                cls_ok = int(matched_id) == int(basket_id)
            except Exception:
                cls_ok = False
        conf = getattr(obs, "matched_conf", None)
        if conf is None:
            conf = getattr(obs, "confidence", None)
        try:
            conf = float(conf) if conf is not None else None
        except Exception:
            conf = None
        return bool(cls_ok), conf

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def _basket_center_x(self, obs: TargetObs) -> Optional[float]:
        center = getattr(obs, "matched_center", None)
        if isinstance(center, dict):
            for key in ("cx", "cx_norm", "x_norm"):
                value = self._as_float(center.get(key))
                if value is not None:
                    return value
        for key in ("cx_norm", "x_norm", "cx"):
            value = self._as_float(getattr(obs, key, None))
            if value is not None:
                return value
        return None

    def _bbox_xyxy(self, bbox: Any) -> Optional[Tuple[float, float, float, float]]:
        if isinstance(bbox, dict):
            if all(k in bbox for k in ("x1", "y1", "x2", "y2")):
                values = [bbox.get(k) for k in ("x1", "y1", "x2", "y2")]
            elif all(k in bbox for k in ("xmin", "ymin", "xmax", "ymax")):
                values = [bbox.get(k) for k in ("xmin", "ymin", "xmax", "ymax")]
            elif all(k in bbox for k in ("x", "y", "w", "h")):
                x = self._as_float(bbox.get("x"))
                y = self._as_float(bbox.get("y"))
                w = self._as_float(bbox.get("w"))
                h = self._as_float(bbox.get("h"))
                if x is None or y is None or w is None or h is None:
                    return None
                return x, y, x + w, y + h
            else:
                return None
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            values = bbox[:4]
        else:
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in values]
        except Exception:
            return None
        return x1, y1, x2, y2

    def _basket_bbox_metrics(self, obs: TargetObs) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        cx = self._basket_center_x(obs)
        area = self._as_float(getattr(obs, "matched_area", None))
        bbox = getattr(obs, "matched_bbox", None) or getattr(obs, "bbox", None) or getattr(obs, "mask_bbox", None)
        xyxy = self._bbox_xyxy(bbox)
        if xyxy is None:
            return cx, None, area
        x1, y1, x2, y2 = xyxy
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        if w > 1.0 or h > 1.0:
            return cx, None, area
        if cx is None:
            cx = (x1 + x2) * 0.5
        if area is None:
            area = w * h
        return cx, h, area
