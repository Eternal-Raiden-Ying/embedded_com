#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import deque
import logging
import threading
import time
from typing import Any, Deque, Dict, Optional

from ..app.stages.base import StageTickInput


class Scheduler:
    """Pure route bus for manager-owned worker loops."""

    def __init__(self):
        self.runtime_running = False
        self.active_generation = 0
        self.active_plan: Optional[Dict[str, Any]] = None
        self.routes: Dict[str, Any] = {}
        self.result_slots: Dict[str, Dict[str, Any]] = {}
        self.event_latches: Dict[str, Deque[Dict[str, Any]]] = {}
        self.pending_signals: Dict[str, Any] = {}
        self.last_snapshot: Dict[str, Any] = {}
        self._last_result_generation: int = 0
        self._lock = threading.RLock()
        self._publish_trace_count: Dict[str, int] = {}
        self._last_publish_trace_ts: Dict[str, float] = {}
        self._last_collect_trace_ts: float = 0.0
        self._logger = logging.getLogger("vision.scheduler")
        self._debug_trace_enable = False

    def _route_cfg(self, route_name: str) -> Dict[str, Any]:
        raw = (self.routes or {}).get(route_name)
        if raw is None:
            return {}
        if isinstance(raw, str):
            raw = {"source": raw}
        cfg = dict(raw or {})
        cfg.setdefault("policy", "slot")
        cfg.setdefault("scope", "stage")
        return cfg

    def _should_accept_generation(self, generation: Optional[int]) -> bool:
        if generation is None:
            return True
        return int(generation) == int(self.active_generation)

    def _slot_visible(self, slot: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(slot, dict):
            return False
        return self._should_accept_generation(slot.get("generation"))

    def start_runtime(self) -> None:
        with self._lock:
            self.runtime_running = True
            self.last_snapshot["runtime_started_ts"] = time.time()

    def stop_runtime(self) -> None:
        with self._lock:
            self.runtime_running = False
            self.active_plan = None
            self.routes.clear()
            self.result_slots.clear()
            self.event_latches.clear()
            self.pending_signals.clear()
            self.last_snapshot["runtime_stopped_ts"] = time.time()

    def configure(self, plan: Dict[str, Any], generation: int) -> None:
        with self._lock:
            self.active_plan = dict(plan or {})
            self.active_generation = int(generation)
            self._debug_trace_enable = bool((self.active_plan or {}).get("vision_debug_trace_enable", False))
            self.routes = dict((self.active_plan or {}).get("routes") or {})
            self.result_slots.clear()
            self.event_latches.clear()
            self.last_snapshot.update(
                {
                    "last_config_ts": time.time(),
                    "active_mode": self.active_plan.get("mode"),
                    "generation": int(self.active_generation),
                    "route_count": len(self.routes),
                    "slots_cleared_on_config": True,
                    "events_cleared_on_config": True,
                }
            )
            contract_results = dict((self.active_plan or {}).get("contract", {}).get("results") or {})
            self._logger.info(
                "[SCHEDULER_CONFIGURE] generation=%s mode=%s routes_keys=%s table_edge_obs_route_cfg=%s contract_results_stage=%s",
                int(self.active_generation),
                self.active_plan.get("mode"),
                sorted(self.routes.keys()),
                self._route_cfg("table_edge_obs"),
                list(contract_results.get("stage") or []),
            )

    def publish_result(self, route: str, payload: Any, generation: Optional[int] = None) -> bool:
        route_name = str(route or "").strip()
        if not route_name:
            return False
        with self._lock:
            now = time.time()
            result_generation = int(self.active_generation if generation is None else generation)
            routes_keys = sorted((self.routes or {}).keys())
            result_slots_keys_before = sorted((self.result_slots or {}).keys())
            cfg = self._route_cfg(route_name)
            if not cfg:
                reason = "route_missing"
                accepted = False
            elif str(cfg.get("policy", "slot")).strip().lower() != "slot":
                reason = "policy_not_slot"
                accepted = False
            elif not self._should_accept_generation(result_generation):
                reason = "generation_mismatch"
                accepted = False
            else:
                reason = "accepted"
                accepted = True

            self.last_snapshot["publish_reject_reason"] = reason
            self.last_snapshot["last_publish_result_route"] = route_name
            self.last_snapshot["last_publish_result_reason"] = reason
            self.last_snapshot["last_publish_result_accepted"] = bool(accepted)
            self.last_snapshot["last_publish_result_generation"] = int(result_generation)
            self.last_snapshot["last_publish_result_active_generation"] = int(self.active_generation)

            if not accepted:
                self.last_snapshot["rejected_result_route"] = route_name
                self.last_snapshot["rejected_result_ts"] = now
                if reason == "generation_mismatch":
                    self.last_snapshot["dropped_result_generation"] = result_generation
                    self.last_snapshot["dropped_result_ts"] = now
                self._log_publish_result_trace(
                    route_name=route_name,
                    accepted=False,
                    reason=reason,
                    publish_generation=result_generation,
                    routes_keys=routes_keys,
                    route_cfg=cfg,
                    result_slots_keys_before=result_slots_keys_before,
                    result_slots_keys_after=sorted((self.result_slots or {}).keys()),
                )
                self._logger.info(
                    "[PUBLISH_REJECT] route=%s reason=%s publish_generation=%s scheduler_active_generation=%s routes_keys=%s route_cfg=%s",
                    route_name,
                    reason,
                    result_generation,
                    int(self.active_generation),
                    routes_keys,
                    cfg,
                )
                return False
            self._last_result_generation = result_generation
            self.last_snapshot["last_result_generation"] = result_generation
            slot = self.result_slots.setdefault(route_name, {})
            next_seq = int(slot.get("seq", 0) or 0) + 1
            slot.update(
                {
                    "generation": result_generation,
                    "ts": now,
                    "seq": next_seq,
                    "payload": payload,
                }
            )
            self.last_snapshot["last_result_ts"] = now
            self._log_publish_result_trace(
                route_name=route_name,
                accepted=True,
                reason=reason,
                publish_generation=result_generation,
                routes_keys=routes_keys,
                route_cfg=cfg,
                result_slots_keys_before=result_slots_keys_before,
                result_slots_keys_after=sorted((self.result_slots or {}).keys()),
            )
        return True

    def _log_publish_result_trace(
        self,
        *,
        route_name: str,
        accepted: bool,
        reason: str,
        publish_generation: int,
        routes_keys,
        route_cfg,
        result_slots_keys_before,
        result_slots_keys_after,
    ) -> None:
        now = time.time()
        if not bool(getattr(self, "_debug_trace_enable", False)):
            return
        count = int(self._publish_trace_count.get(route_name, 0) or 0)
        last_ts = float(self._last_publish_trace_ts.get(route_name, 0.0) or 0.0)
        should_log = route_name == "table_edge_obs" and (count < 5 or not accepted or (now - last_ts) >= 0.5)
        if not should_log:
            return
        self._publish_trace_count[route_name] = count + 1
        self._last_publish_trace_ts[route_name] = now
        self._logger.debug(
            "[PUBLISH_RESULT_TRACE] route=%s accepted=%s reason=%s publish_generation=%s scheduler_active_generation=%s routes_keys=%s route_cfg=%s result_slots_keys_before=%s result_slots_keys_after=%s",
            route_name,
            str(bool(accepted)).lower(),
            reason,
            int(publish_generation),
            int(self.active_generation),
            list(routes_keys or []),
            dict(route_cfg or {}),
            list(result_slots_keys_before or []),
            list(result_slots_keys_after or []),
        )

    def publish_event(self, route: str, payload: Any, generation: Optional[int] = None) -> bool:
        route_name = str(route or "").strip()
        if not route_name:
            return False
        with self._lock:
            cfg = self._route_cfg(route_name)
            if not cfg or str(cfg.get("policy", "slot")).strip().lower() != "event":
                self.last_snapshot["rejected_event_route"] = route_name
                self.last_snapshot["rejected_event_ts"] = time.time()
                return False
            event_generation = int(self.active_generation if generation is None else generation)
            if not self._should_accept_generation(event_generation):
                self.last_snapshot["dropped_event_generation"] = event_generation
                self.last_snapshot["dropped_event_ts"] = time.time()
                return False
            latch = self.event_latches.setdefault(route_name, deque(maxlen=32))
            latch.append(
                {
                    "generation": event_generation,
                    "ts": time.time(),
                    "payload": payload,
                }
            )
            self.last_snapshot["last_event_ts"] = time.time()
        return True

    def publish_results(
        self,
        results: Dict[str, Any],
        snapshot: Optional[Dict[str, Any]] = None,
        generation: Optional[int] = None,
    ) -> None:
        for key, value in dict(results or {}).items():
            self.publish_result(str(key), value, generation=generation)
        if snapshot:
            with self._lock:
                self.last_snapshot.update(dict(snapshot or {}))

    def read_slot(self, route: str) -> Optional[Dict[str, Any]]:
        route_name = str(route or "").strip()
        if not route_name:
            return None
        with self._lock:
            cfg = self._route_cfg(route_name)
            if not cfg or str(cfg.get("policy", "slot")).strip().lower() != "slot":
                return None
            slot = self.result_slots.get(route_name)
            if not self._slot_visible(slot):
                if isinstance(slot, dict):
                    self.last_snapshot["dropped_stale_slot_route"] = route_name
                    self.last_snapshot["dropped_stale_slot_generation"] = int(slot.get("generation", 0) or 0)
                    self.last_snapshot["dropped_stale_slot_ts"] = time.time()
                return None
            return {
                "generation": int(slot.get("generation", 0) or 0),
                "ts": float(slot.get("ts", 0.0) or 0.0),
                "seq": int(slot.get("seq", 0) or 0),
                "payload": slot.get("payload"),
            }

    def read_result(self, route: str, default=None):
        slot = self.read_slot(route)
        if slot is None:
            return default
        return slot.get("payload", default)

    def consume_event(self, route: str):
        route_name = str(route or "").strip()
        if not route_name:
            return None
        with self._lock:
            cfg = self._route_cfg(route_name)
            if not cfg or str(cfg.get("policy", "slot")).strip().lower() != "event":
                return None
            latch = self.event_latches.get(route_name)
            if not latch:
                return None
            while latch:
                try:
                    item = latch.popleft()
                except Exception:
                    return None
                if self._should_accept_generation((item or {}).get("generation")):
                    return (item or {}).get("payload")
                self.last_snapshot["dropped_stale_event_route"] = route_name
                self.last_snapshot["dropped_stale_event_generation"] = int((item or {}).get("generation", 0) or 0)
                self.last_snapshot["dropped_stale_event_ts"] = time.time()
            return None

    def push_stage_signals(self, signals: Dict[str, Any]) -> None:
        if not signals:
            return
        with self._lock:
            self.pending_signals.update(dict(signals or {}))
            self.last_snapshot["last_stage_signal_ts"] = time.time()

    def collect_tick_input(self, ts: float, route_filter: set = None) -> StageTickInput:
        with self._lock:
            if bool(getattr(self, "_debug_trace_enable", False)):
                if "table_edge_obs" in self.result_slots:
                    slot = self.result_slots["table_edge_obs"]
                    self._logger.debug(
                        "[DIAG_SCHEDULER] table_edge_obs slot exists: generation=%s active_gen=%s visible=%s payload_frame=%s",
                        slot.get("generation"),
                        self.active_generation,
                        self._slot_visible(slot),
                        (slot.get("payload") or {}).get("frame_id") if isinstance(slot.get("payload"), dict) else None,
                    )
                else:
                    self._logger.debug("[DIAG_SCHEDULER] table_edge_obs slot DOES NOT exist in result_slots")

            signals = dict(self.pending_signals or {})
            self.pending_signals.clear()
            stage_results: Dict[str, Any] = {}
            skipped_routes = []
            for route_name in sorted((self.routes or {}).keys()):
                cfg = self._route_cfg(route_name)
                if str(cfg.get("scope", "stage")).strip().lower() != "stage":
                    skipped_routes.append((route_name, "scope_not_stage"))
                    continue
                if route_filter is not None and route_name not in route_filter:
                    skipped_routes.append((route_name, "route_filter_excluded"))
                    continue
                slot = self.result_slots.get(route_name)
                if slot is None:
                    skipped_routes.append((route_name, "slot_missing"))
                    continue
                if not self._slot_visible(slot):
                    if isinstance(slot, dict):
                        self.last_snapshot["skipped_stage_route"] = route_name
                        self.last_snapshot["skipped_stage_route_generation"] = int(slot.get("generation", 0) or 0)
                        self.last_snapshot["skipped_stage_route_ts"] = time.time()
                    skipped_routes.append((route_name, "slot_not_visible"))
                    continue
                payload = (slot or {}).get("payload")
                if payload is not None:
                    stage_results[route_name] = payload
                else:
                    skipped_routes.append((route_name, "payload_none"))
            visible_routes = sorted(stage_results.keys())
            now = time.time()
            if bool(getattr(self, "_debug_trace_enable", False)) and now - float(self._last_collect_trace_ts or 0.0) >= 2.0:
                self._last_collect_trace_ts = now
                for skipped_route, skip_reason in skipped_routes:
                    self.last_snapshot["collect_skipped_route"] = skipped_route
                    self.last_snapshot["collect_skip_reason"] = skip_reason
                    if skipped_route == "table_edge_obs":
                        break
                self._logger.debug(
                    "[SCHEDULER_COLLECT_TRACE] stage=%s mode=%s subscribed_routes=%s result_slots_keys=%s visible_routes=%s skipped_routes=%s",
                    (self.active_plan or {}).get("stage"),
                    (self.active_plan or {}).get("mode"),
                    sorted(route_filter) if route_filter is not None else None,
                    sorted((self.result_slots or {}).keys()),
                    visible_routes,
                    skipped_routes,
                )
            snapshot = {
                "runtime_running": bool(self.runtime_running),
                "generation": int(self.active_generation),
                "active_mode": (self.active_plan or {}).get("mode"),
                "plan": dict(self.active_plan or {}),
                "scheduler": self.snapshot(),
            }
        return StageTickInput(
            ts=float(ts),
            generation=int(self.active_generation),
            results=stage_results,
            signals=signals,
            snapshot=snapshot,
        )

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "runtime_running": bool(self.runtime_running),
                "active_generation": int(self.active_generation),
                "active_mode": (self.active_plan or {}).get("mode"),
                "route_keys": sorted(self.routes.keys()),
                "result_keys": sorted(self.result_slots.keys()),
                "event_keys": sorted(self.event_latches.keys()),
                "pending_signal_keys": sorted(self.pending_signals.keys()),
                "last_result_generation": int(self._last_result_generation),
                "last_snapshot": dict(self.last_snapshot or {}),
            }
