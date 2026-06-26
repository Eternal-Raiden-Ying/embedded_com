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

    @property
    def source(self) -> str:
        return self.control_source

    @property
    def intent(self) -> str:
        return self.control_source

    @property
    def reason(self) -> str:
        return self.block_reason

    @property
    def forward_block_reason(self) -> str:
        return self.block_reason if not self.allow_forward else ""

    @property
    def rotate_block_reason(self) -> str:
        return self.block_reason if not self.allow_rotate else ""

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
) -> ControlAuthority:
    """Return the owner of the current table-approach motion command.

    Stateless arbitration mapping of current state and semantics.
    """
    state = str(state or "").upper().strip()

    if explicit_stop_active:
        return ControlAuthority(
            control_source="explicit_stop",
            yaw_source="none",
            forward_source="none",
            stop_source="explicit",
            allow_forward=False,
            allow_rotate=False,
            block_reason="explicit_stop_active",
        )

    if state in {"FINAL_SLOW_STOP", "AT_TABLE_EDGE"}:
        return ControlAuthority(
            control_source="final_slow_stop",
            yaw_source="edge" if sem.edge_trusted else "last_stable",
            forward_source="none",
            stop_source="final_lock",
            allow_forward=False,
            allow_rotate=False,
            block_reason="final_slow_stop_state",
        )

    if depth_roi_stop_active:
        return ControlAuthority(
            control_source="depth_roi_stop",
            yaw_source="edge" if sem.edge_trusted else "yolo",
            forward_source="none",
            stop_source="depth_roi",
            allow_forward=False,
            allow_rotate=False,
            block_reason="depth_roi_stop_active",
        )

    if not sem.table_bbox_control_valid:
        return ControlAuthority(
            control_source="local_rotate_search",
            yaw_source="none",
            forward_source="none",
            stop_source="none",
            allow_forward=False,
            allow_rotate=True,
            block_reason="table_bbox_unavailable",
        )

    if state == "YOLO_ACQUIRE_ALIGN":
        return ControlAuthority(
            control_source="yolo_acquire_align",
            yaw_source="yolo",
            forward_source="none",
            stop_source="none",
            allow_forward=False,
            allow_rotate=True,
            block_reason="yolo_center_error_too_large",
        )

    if state == "EDGE_ADJUST":
        return ControlAuthority(
            control_source="edge_guided_forward",
            yaw_source="edge",
            forward_source="edge",
            stop_source="none",
            allow_forward=True,
            allow_rotate=True,
            block_reason="",
        )

    if state == "YOLO_APPROACH":
        if sem.edge_trusted:
            return ControlAuthority(
                control_source="edge_guided_forward",
                yaw_source="edge",
                forward_source="yolo_or_edge",
                stop_source="none",
                allow_forward=True,
                allow_rotate=True,
                block_reason="",
            )
        else:
            return ControlAuthority(
                control_source="yolo_track_forward",
                yaw_source="yolo",
                forward_source="yolo",
                stop_source="none",
                allow_forward=True,
                allow_rotate=True,
                block_reason="",
            )

    return ControlAuthority(
        control_source="local_rotate_search",
        yaw_source="none",
        forward_source="none",
        stop_source="none",
        allow_forward=False,
        allow_rotate=True,
        block_reason=f"unknown_state_{state}",
    )
