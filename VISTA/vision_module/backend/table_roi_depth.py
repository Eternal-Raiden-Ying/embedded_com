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
    allow_without_table_bbox: bool = False,
    roi_is_latched: bool = False,
    min_valid_ratio: Optional[float] = None,
    min_sample_count: Optional[int] = None,
    min_depth_m: float = 0.05,
    max_depth_m: float = 5.0,
) -> Dict[str, Any]:
    """Return robust depth statistics for the lower, inner part of table ROI."""
    del current_table_bbox_found, allow_without_table_bbox
    threshold_ratio = float(
        min_valid_ratio
        if min_valid_ratio is not None
        else (0.03 if bool(roi_is_latched) else 0.08)
    )
    threshold_samples = int(
        min_sample_count
        if min_sample_count is not None
        else (32 if bool(roi_is_latched) else 64)
    )
    out: Dict[str, Any] = {
        "table_roi_depth_valid": False,
        "table_roi_depth_p10": None,
        "table_roi_depth_median": None,
        "table_roi_depth_mean": None,
        "table_roi_depth_valid_ratio": 0.0,
        "table_roi_depth_sample_count": 0,
        "table_roi_depth_total_count": 0,
        "table_roi_depth_bbox": None,
        "table_roi_depth_bbox_norm": None,
        "table_roi_depth_coord_space": "depth_frame_xyxy",
        "table_roi_depth_sampler_level": None,
        "table_roi_depth_sampler_name": "",
        "table_roi_depth_invalid_reason": "",
    }
    if not isinstance(depth_frame, np.ndarray) or depth_frame.ndim != 2 or depth_frame.size == 0:
        out["table_roi_depth_invalid_reason"] = "depth_frame_invalid"
        return out
    if not table_roi_xyxy or len(table_roi_xyxy) < 4:
        out["table_roi_depth_invalid_reason"] = "roi_xyxy_invalid"
        return out
    h, w = depth_frame.shape
    try:
        x0, y0, x1, y1 = [float(v) for v in table_roi_xyxy[:4]]
    except (TypeError, ValueError):
        out["table_roi_depth_invalid_reason"] = "roi_xyxy_invalid"
        return out
    x0, x1 = sorted((max(0, min(w, int(round(x0)))), max(0, min(w, int(round(x1))))))
    y0, y1 = sorted((max(0, min(h, int(round(y0)))), max(0, min(h, int(round(y1))))))
    if x1 <= x0 or y1 <= y0:
        out["table_roi_depth_invalid_reason"] = "roi_xyxy_invalid"
        return out
    bw, bh = x1 - x0, y1 - y0
    samplers = (
        (0, "primary", 0.15, 0.85, 0.55, 0.95),
        (1, "wider_lower", 0.05, 0.95, 0.50, 0.98),
        (2, "lower_full", 0.00, 1.00, 0.40, 0.98),
        (3, "full_roi_safe", 0.00, 1.00, 0.25, 0.98),
    )
    best = None
    for level, name, fx0, fx1, fy0, fy1 in samplers:
        sx0, sx1 = x0 + int(round(bw * fx0)), x0 + int(round(bw * fx1))
        sy0, sy1 = y0 + int(round(bh * fy0)), y0 + int(round(bh * fy1))
        sx0, sx1 = max(0, min(w, sx0)), max(0, min(w, sx1))
        sy0, sy1 = max(0, min(h, sy0)), max(0, min(h, sy1))
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        roi = depth_frame[sy0:sy1, sx0:sx1].astype(np.float32) * float(depth_scale)
        finite = np.isfinite(roi)
        valid = finite & (roi >= float(min_depth_m)) & (roi <= float(max_depth_m))
        total = int(roi.size)
        values = roi[valid]
        sample_count = int(values.size)
        ratio = float(sample_count / total) if total else 0.0
        bbox = [int(sx0), int(sy0), int(sx1), int(sy1)]
        stats = {
            "table_roi_depth_valid_ratio": ratio,
            "table_roi_depth_sample_count": sample_count,
            "table_roi_depth_total_count": total,
            "table_roi_depth_bbox": bbox,
            "table_roi_depth_bbox_norm": [sx0 / w, sy0 / h, sx1 / w, sy1 / h],
            "table_roi_depth_sampler_level": int(level),
            "table_roi_depth_sampler_name": str(name),
        }
        if best is None or sample_count > int(best.get("table_roi_depth_sample_count", 0) or 0):
            best = dict(stats)
        if sample_count <= 0:
            invalid_reason = "no_valid_depth_samples"
        elif sample_count < threshold_samples:
            invalid_reason = "sample_count_below_threshold"
        elif ratio < threshold_ratio:
            invalid_reason = "valid_ratio_below_threshold"
        else:
            out.update(stats)
            out.update({
                "table_roi_depth_valid": True,
                "table_roi_depth_p10": float(np.percentile(values, 10)),
                "table_roi_depth_median": float(np.median(values)),
                "table_roi_depth_mean": float(np.mean(values)),
                "table_roi_depth_invalid_reason": "",
            })
            return out
        stats["table_roi_depth_invalid_reason"] = invalid_reason
        if best is None or sample_count >= int(best.get("table_roi_depth_sample_count", 0) or 0):
            best = dict(stats)
    if best:
        out.update(best)
    else:
        out["table_roi_depth_invalid_reason"] = "roi_xyxy_invalid"
    return out
