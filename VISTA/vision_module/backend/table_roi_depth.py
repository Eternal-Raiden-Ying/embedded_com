"""Pure depth statistics for a mapped dynamic table ROI.

The ROI is always expressed in depth-frame pixel coordinates.  Keeping this
helper free of camera/runtime dependencies makes its safety gates testable.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np


def table_roi_depth_statistics(
    depth_frame: np.ndarray,
    depth_scale: float,
    table_roi_xyxy: Optional[Sequence[float]],
    *,
    current_table_bbox_found: bool = True,
    min_valid_ratio: float = 0.20,
    min_sample_count: int = 64,
    min_depth_m: float = 0.05,
    max_depth_m: float = 5.0,
) -> Dict[str, Any]:
    """Return robust depth statistics for the lower, inner part of table ROI."""
    out: Dict[str, Any] = {
        "table_roi_depth_valid": False,
        "table_roi_depth_p10": None,
        "table_roi_depth_median": None,
        "table_roi_depth_mean": None,
        "table_roi_depth_valid_ratio": 0.0,
        "table_roi_depth_sample_count": 0,
        "table_roi_depth_bbox": None,
        "table_roi_depth_bbox_norm": None,
        "table_roi_depth_coord_space": "depth_frame_xyxy",
    }
    if not current_table_bbox_found or not isinstance(depth_frame, np.ndarray) or depth_frame.ndim != 2 or depth_frame.size == 0 or not table_roi_xyxy or len(table_roi_xyxy) < 4:
        return out
    h, w = depth_frame.shape
    try:
        x0, y0, x1, y1 = [float(v) for v in table_roi_xyxy[:4]]
    except (TypeError, ValueError):
        return out
    x0, x1 = sorted((max(0, min(w, int(round(x0)))), max(0, min(w, int(round(x1))))))
    y0, y1 = sorted((max(0, min(h, int(round(y0)))), max(0, min(h, int(round(y1))))))
    if x1 <= x0 or y1 <= y0:
        return out
    # Avoid bbox edges/background and focus on the table-facing lower band.
    bw, bh = x1 - x0, y1 - y0
    sx0, sx1 = x0 + int(bw * 0.15), x1 - int(bw * 0.15)
    sy0, sy1 = y0 + int(bh * 0.55), y0 + int(bh * 0.95)
    if sx1 <= sx0 or sy1 <= sy0:
        return out
    roi = depth_frame[sy0:sy1, sx0:sx1].astype(np.float32) * float(depth_scale)
    finite = np.isfinite(roi)
    valid = finite & (roi >= float(min_depth_m)) & (roi <= float(max_depth_m))
    total = int(roi.size)
    values = roi[valid]
    sample_count = int(values.size)
    ratio = float(sample_count / total) if total else 0.0
    bbox = [int(sx0), int(sy0), int(sx1), int(sy1)]
    out.update({
        "table_roi_depth_valid_ratio": ratio,
        "table_roi_depth_sample_count": sample_count,
        "table_roi_depth_bbox": bbox,
        "table_roi_depth_bbox_norm": [sx0 / w, sy0 / h, sx1 / w, sy1 / h],
    })
    if ratio < float(min_valid_ratio) or sample_count < int(min_sample_count):
        return out
    out.update({
        "table_roi_depth_valid": True,
        "table_roi_depth_p10": float(np.percentile(values, 10)),
        "table_roi_depth_median": float(np.median(values)),
        "table_roi_depth_mean": float(np.mean(values)),
    })
    return out
