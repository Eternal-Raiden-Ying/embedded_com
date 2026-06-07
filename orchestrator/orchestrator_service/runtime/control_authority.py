#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Control authority arbitration primitives for table approach.

The state machine should decide coarse task phases; this module decides which
source is allowed to own the motion command for the current table approach tick.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict

from .perception_semantics import TablePerceptionSemantics


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


def decide_table_control_authority(state: str, sem: TablePerceptionSemantics, cfg: Any = None) -> ControlAuthority:
    state = str(state or "").upper().strip()
    if state in {"FINAL_LOCK", "AT_TABLE_EDGE"}:
        return ControlAuthority("final_lock", "final_lock", False, False, reason="final_lock_state")
    if sem.table_bbox_found:
        if sem.edge_trusted and state == "COARSE_ALIGN":
            return ControlAuthority("edge_adjust", "posture_adjust", False, True, reason="edge_trusted_for_rotation")
        # Current first-priority strategy: table bbox is a forward permit.
        # Edge can be logged while untrusted, but it cannot steal motion authority.
        return ControlAuthority("yolo_forward", "forward", True, False, reason="table_bbox_found_default_forward", rotate_block_reason="yolo_forward_owns_control")
    return ControlAuthority("local_rotate_search", "search", False, True, reason="table_bbox_missing_local_search", forward_block_reason="table_bbox_unavailable")
