#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Control-authority arbitration for table approach.

This module owns the vocabulary of table-control sources.  It keeps the state
machine from reintroducing ambiguous legacy control-source names.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict

from .perception_semantics import TablePerceptionSemantics


VALID_CONTROL_SOURCES = {
    "yolo_forward",
    "yolo_track_forward",
    "edge_adjust",
    "edge_guided_forward",
    "local_rotate_search",
    "final_lock",
    "final_slow_stop",
    "search_failed_stop",
    "explicit_stop",
    "stop",
    "yolo_acquire_align",
    "depth_roi_stop",
}


@dataclass(frozen=True)
class ControlAuthority:
    control_source: str
    yaw_source: str
    forward_source: str
    stop_source: str
    allow_forward: bool
    allow_rotate: bool
    block_reason: str
    forward_block_reason: str = ""
    rotate_block_reason: str = ""

    @property
    def source(self) -> str:
        return self.control_source

    @property
    def intent(self) -> str:
        return self.control_source

    @property
    def reason(self) -> str:
        return self.block_reason

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update({
            "source": self.source,
            "intent": self.intent,
            "reason": self.reason,
            "forward_block_reason": self.forward_block_reason,
            "rotate_block_reason": self.rotate_block_reason,
        })
        return d


def normalize_control_source(source: str) -> str:
    source = str(source or "").strip().lower()
    legacy_map = {
        "edge_only": "edge_adjust",
        "search_fallback": "local_rotate_search",
    }
    source = legacy_map.get(source, source)
    return source if source in VALID_CONTROL_SOURCES else "stop"


def decide_table_control_authority(
    state: str,
    sem: TablePerceptionSemantics,
    cfg: Any = None,
    depth_roi_stop_active: bool = False,
    explicit_stop_active: bool = False,
    bbox_center_error: float | None = None,
) -> ControlAuthority:
    """Return the owner of the current table-approach motion command.

    Stateless arbitration mapping of current state and semantics.
    """
    state = str(state or "").upper().strip()

    def make(source: str, yaw: str, forward: str, stop: str, allow_forward: bool, allow_rotate: bool, reason: str = "") -> ControlAuthority:
        return ControlAuthority(
            control_source=source, yaw_source=yaw, forward_source=forward,
            stop_source=stop, allow_forward=allow_forward, allow_rotate=allow_rotate,
            block_reason=reason,
            forward_block_reason="" if allow_forward else reason,
            rotate_block_reason="" if allow_rotate else reason,
        )

    if explicit_stop_active:
        return make("explicit_stop", "none", "none", "explicit", False, False, "explicit_stop_active")

    if state in {"FINAL_SLOW_STOP", "AT_TABLE_EDGE"}:
        return make("final_slow_stop", "edge" if sem.edge_trusted else "last_stable", "none", "final_lock", False, False, "final_slow_stop_state")

    if depth_roi_stop_active:
        return make("depth_roi_stop", "edge" if sem.edge_trusted else "yolo", "none", "roi_depth", False, False, "depth_roi_stop_active")

    if not sem.table_bbox_control_valid:
        return make("local_rotate_search", "none", "none", "none", False, True, "table_bbox_unavailable")

    hard_limit = abs(float(getattr(getattr(cfg, "car", cfg), "yolo_forward_center_hard_limit", 0.25) or 0.25))
    if state == "YOLO_ACQUIRE_ALIGN" or (bbox_center_error is not None and abs(float(bbox_center_error)) > hard_limit):
        return make("yolo_acquire_align", "yolo", "none", "none", False, True, "yolo_center_error_too_large")

    if state == "EDGE_ADJUST":
        if sem.edge_trusted:
            return make("edge_guided_forward", "edge", "edge", "none", True, True)
        return make("yolo_track_forward", "yolo", "yolo", "none", True, True)

    if state == "YOLO_APPROACH":
        if sem.edge_trusted:
            return make("edge_guided_forward", "edge", "yolo_or_edge", "none", True, True)
        else:
            return make("yolo_track_forward", "yolo", "yolo", "none", True, True)

    return make("local_rotate_search", "none", "none", "none", False, True, f"unknown_state_{state}")
