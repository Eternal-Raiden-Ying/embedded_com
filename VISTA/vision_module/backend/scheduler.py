#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import deque
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

    def _route_cfg(self, route_name: str) -> Dict[str, Any]:
        raw = (self.routes or {}).get(route_name, {})
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
            self.routes = dict((self.active_plan or {}).get("routes") or {})
            self.last_snapshot.update(
                {
                    "last_config_ts": time.time(),
                    "active_mode": self.active_plan.get("mode"),
                    "generation": int(self.active_generation),
                    "route_count": len(self.routes),
                }
            )

    def publish_result(self, route: str, payload: Any, generation: Optional[int] = None) -> bool:
        route_name = str(route or "").strip()
        if not route_name:
            return False
        with self._lock:
            result_generation = int(self.active_generation if generation is None else generation)
            self._last_result_generation = result_generation
            self.last_snapshot["last_result_generation"] = result_generation
            if not self._should_accept_generation(result_generation):
                self.last_snapshot["dropped_result_generation"] = result_generation
                self.last_snapshot["dropped_result_ts"] = time.time()
                return False
            slot = self.result_slots.setdefault(route_name, {})
            next_seq = int(slot.get("seq", 0) or 0) + 1
            slot.update(
                {
                    "generation": result_generation,
                    "ts": time.time(),
                    "seq": next_seq,
                    "payload": payload,
                }
            )
            self.last_snapshot["last_result_ts"] = time.time()
        return True

    def publish_event(self, route: str, payload: Any, generation: Optional[int] = None) -> bool:
        route_name = str(route or "").strip()
        if not route_name:
            return False
        with self._lock:
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
            slot = self.result_slots.get(route_name)
            if not isinstance(slot, dict):
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
            latch = self.event_latches.get(route_name)
            if not latch:
                return None
            try:
                item = latch.popleft()
            except Exception:
                return None
        return item.get("payload")

    def push_stage_signals(self, signals: Dict[str, Any]) -> None:
        if not signals:
            return
        with self._lock:
            self.pending_signals.update(dict(signals or {}))
            self.last_snapshot["last_stage_signal_ts"] = time.time()

    def collect_tick_input(self, ts: float) -> StageTickInput:
        with self._lock:
            signals = dict(self.pending_signals or {})
            self.pending_signals.clear()
            stage_results: Dict[str, Any] = {}
            for route_name in sorted((self.routes or {}).keys()):
                cfg = self._route_cfg(route_name)
                if str(cfg.get("scope", "stage")).strip().lower() != "stage":
                    continue
                payload = (self.result_slots.get(route_name) or {}).get("payload")
                if payload is not None:
                    stage_results[route_name] = payload
            # Keep backward compatibility for unregistered routes.
            for route_name in sorted((self.result_slots or {}).keys()):
                if route_name in stage_results:
                    continue
                if route_name in (self.routes or {}):
                    continue
                payload = (self.result_slots.get(route_name) or {}).get("payload")
                if payload is not None:
                    stage_results[route_name] = payload
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
