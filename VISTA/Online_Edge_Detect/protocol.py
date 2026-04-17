#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


def now_ts() -> float:
    return time.time()


def _compact(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            compacted = _compact(item)
            if compacted is None:
                continue
            out[key] = compacted
        return out
    if isinstance(value, list):
        return [_compact(item) for item in value if item is not None]
    return value


@dataclass
class TableEdgeObsMsg:
    ts: float
    table_found: bool
    edge_found: bool
    confidence: float
    yaw_err_rad: Optional[float] = None
    dist_err_m: Optional[float] = None
    edge_k: Optional[float] = None
    edge_b: Optional[float] = None
    depth_valid: bool = True
    point_count: int = 0
    table_point_count: int = 0
    frame_id: int = 0
    source: str = "online_edge_detect"
    type: str = "table_edge_obs"

    def to_dict(self) -> Dict[str, Any]:
        return _compact(asdict(self))
