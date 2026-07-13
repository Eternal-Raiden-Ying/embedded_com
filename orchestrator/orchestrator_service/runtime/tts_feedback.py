#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structured Phone-TTS feedback catalog, policy and bounded deduplication."""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from ..ipc.protocol import make_tts_event
from ..utils.target_utils import resolve_target, target_display_name

# (template, priority, minimum verbosity).  Keep all business wording here.
EVENT_CATALOG = {
    "WAKE_PROMPT": ("\u6211\u5728\uff0c\u8bf7\u8bf4\u51fa\u8981\u62ff\u7684\u7269\u54c1\u3002", "P2", "basic"),
    "TASK_ACCEPTED": ("\u5df2\u6536\u5230\uff0c\u51c6\u5907\u5bfb\u627e{target_name}\u3002", "P2", "basic"),
    "TASK_SWITCHED": ("\u5df2\u53d6\u6d88{old_target_name}\u4efb\u52a1\uff0c\u6b63\u5728\u6539\u4e3a\u5bfb\u627e{target_name}\u3002", "P2", "basic"),
    "SEARCH_TABLE_STARTED": ("\u6b63\u5728\u5bfb\u627e\u684c\u8fb9\u3002", "P3", "detailed"),
    "TABLE_FOUND": ("\u5df2\u7ecf\u53d1\u73b0\u684c\u8fb9\u3002", "P3", "detailed"),
    "TABLE_APPROACH_STARTED": ("\u6b63\u5728\u9760\u8fd1\u684c\u5b50\u3002", "P3", "detailed"),
    "TABLE_EDGE_REACHED": ("\u5df2\u7ecf\u5230\u8fbe\u684c\u8fb9\u3002", "P3", "detailed"),
    "SEARCH_TARGET_STARTED": ("\u6b63\u5728\u5bfb\u627e{target_name}\u3002", "P3", "detailed"),
    "TARGET_LOCKED": ("\u5df2\u7ecf\u627e\u5230{target_name}\u3002", "P3", "detailed"),
    "PICK_PREPARING": ("\u6b63\u5728\u51c6\u5907\u6293\u53d6{target_name}\u3002", "P3", "detailed"),
    "PICK_EXECUTING": ("\u6b63\u5728\u6293\u53d6{target_name}\u3002", "P3", "detailed"),
    "PICK_SUCCEEDED": ("\u5df2\u7ecf\u62ff\u5230{target_name}\u3002", "P1", "basic"),
    "PICK_FAILED": ("\u6293\u53d6{target_name}\u6ca1\u6709\u6210\u529f\uff0c\u51c6\u5907\u91cd\u65b0\u5c1d\u8bd5\u3002", "P1", "basic"),
    "TASK_STOPPED": ("\u4efb\u52a1\u5df2\u505c\u6b62\u3002", "P0", "basic"),
    "TASK_TIMEOUT": ("\u6682\u65f6\u6ca1\u6709\u627e\u5230{target_name}\u3002", "P2", "basic"),
    "TASK_COMPLETED": ("{target_name}\u5df2\u7ecf\u53d6\u5230\u3002", "P1", "basic"),
    "SYSTEM_RECOVERED": ("\u7cfb\u7edf\u5df2\u7ecf\u6062\u590d\uff0c\u53ef\u4ee5\u7ee7\u7eed\u4f7f\u7528\u3002", "P2", "basic"),
}

_VERBOSITY = {"off": 0, "basic": 1, "detailed": 2}
_TTL_BY_PRIORITY = {"P0": 20.0, "P1": 15.0, "P2": 10.0, "P3": 5.0}


class TtsFeedbackEmitter:
    """Creates one structured event for each accepted business transition."""

    def __init__(self, config: Optional[Any] = None, now=time.time, monotonic=time.monotonic) -> None:
        self.config = config
        self._now = now
        self._monotonic = monotonic
        self._seen: "OrderedDict[tuple[str, int, str, str], float]" = OrderedDict()
        self._last_emit_mono = 0.0
        self._last_progress_mono = 0.0
        self._max_seen = 512

    def _get(self, name: str, default: Any) -> Any:
        return getattr(self.config, name, default) if self.config is not None else default

    def _allows(self, minimum: str, force: bool) -> bool:
        if force:
            return True
        if not bool(self._get("enabled", True)):
            return False
        verbosity = str(self._get("verbosity", "detailed")).lower()
        if verbosity == "off":
            return False
        if minimum == "detailed" and not bool(self._get("detailed_progress_enabled", True)):
            return False
        return _VERBOSITY.get(verbosity, 0) >= _VERBOSITY[minimum]

    def emit(
        self,
        pending: list[Dict[str, Any]],
        event_key: str,
        *,
        session_id: str,
        epoch: int,
        state: str = "",
        target: str = "",
        old_target: str = "",
        force: bool = False,
    ) -> Optional[Dict[str, Any]]:
        catalog = EVENT_CATALOG.get(event_key)
        if catalog is None:
            return None
        template, priority, minimum = catalog
        if not self._allows(minimum, force):
            return None

        target_spec = resolve_target(target) if target else None
        old_target_spec = resolve_target(old_target) if old_target else None
        canonical_target = target_spec.canonical_target if target_spec else ""
        canonical_old = old_target_spec.canonical_target if old_target_spec else ""
        target_name = target_display_name(target_spec) if target_spec else "\u7269\u54c1"
        old_target_name = target_display_name(old_target_spec) if old_target_spec else "\u5f53\u524d"
        key = (str(session_id or ""), int(epoch or 0), event_key, canonical_target)
        now_mono = self._monotonic()
        window = max(0.0, float(self._get("dedupe_window_s", 30.0)))
        while self._seen and now_mono - next(iter(self._seen.values())) > window:
            self._seen.popitem(last=False)
        if key in self._seen:
            return None
        min_interval = max(0.0, float(self._get("min_interval_s", 1.5)))
        # P3 is the only high-frequency class; terminal and safety feedback must not wait.
        if priority == "P3" and now_mono - self._last_progress_mono < min_interval:
            return None

        created_at = float(self._now())
        ttl_s = _TTL_BY_PRIORITY.get(priority, float(self._get("default_ttl_s", 8.0)))
        dedupe_key = ":".join((str(session_id or ""), str(int(epoch or 0)), event_key, canonical_target))
        event = make_tts_event(
            session_id=str(session_id or ""),
            text=template.format(target_name=target_name, old_target_name=old_target_name),
            priority=priority,
            interrupt=(priority == "P0"),
            dedup_key=dedupe_key,
            source="orchestrator",
            epoch=int(epoch or 0),
            event_key=event_key,
            state=state,
            target=canonical_target,
            created_at=created_at,
            expires_at=created_at + ttl_s,
            ttl_s=ttl_s,
        )
        self._seen[key] = now_mono
        self._seen.move_to_end(key)
        while len(self._seen) > self._max_seen:
            self._seen.popitem(last=False)
        self._last_emit_mono = now_mono
        if priority == "P3":
            self._last_progress_mono = now_mono
        pending.append(event)
        return event
