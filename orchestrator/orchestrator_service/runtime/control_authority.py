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
    "edge_adjust",
    "local_rotate_search",
    "final_lock",
    "final_slow_stop",
    "search_failed_stop",
    "explicit_stop",
    "stop",
}


@dataclass(frozen=True)
class ControlAuthority:
    source: str
    intent: str
    allow_forward: bool
    allow_rotate: bool
    reason: str = ""
    forward_block_reason: str = ""
    rotate_block_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_control_source(source: str) -> str:
    source = str(source or "").strip().lower()
    legacy_map = {
        "edge_only": "edge_adjust",
        "search_fallback": "local_rotate_search",
    }
    source = legacy_map.get(source, source)
    return source if source in VALID_CONTROL_SOURCES else "stop"


def decide_table_control_authority(state: str, sem: TablePerceptionSemantics, cfg: Any = None) -> ControlAuthority:
    """Return the owner of the current table-approach motion command.

    Current policy:
    - final states own final-lock / stop decisions.
    - if table bbox is control-valid and edge is trusted, edge_adjust may own
      posture correction.
    - if table bbox is control-valid but edge is not trusted, yolo_forward owns
      straight forward motion.
    - without table bbox, edge/docking is not allowed to control; use local
      rotate search.
    """
    state = str(state or "").upper().strip()
    if state in {"FINAL_SLOW_STOP", "AT_TABLE_EDGE"}:
        return ControlAuthority("final_slow_stop", "final_slow_stop", False, False, reason="final_slow_stop_state")

    if sem.table_bbox_control_valid:
        if sem.edge_trusted:
            return ControlAuthority(
                "edge_adjust",
                "posture_adjust",
                allow_forward=True,
                allow_rotate=True,
                reason="edge_trusted",
            )
        return ControlAuthority(
            "yolo_forward",
            "forward",
            allow_forward=True,
            allow_rotate=False,
            reason="table_bbox_control_valid_edge_not_trusted",
            rotate_block_reason=sem.edge_reject_for_control_reason or "edge_not_trusted",
        )

    return ControlAuthority(
        "local_rotate_search",
        "search",
        allow_forward=False,
        allow_rotate=True,
        reason="table_bbox_unavailable",
        forward_block_reason="table_bbox_unavailable",
    )
