#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from .board_config import CONFIG
    from .detector import OnlineTableEdgeDetector, load_calib
except ImportError:
    from board_config import CONFIG
    from detector import OnlineTableEdgeDetector, load_calib


def build_synthetic_depth(height: int = 480, width: int = 640) -> np.ndarray:
    depth = np.zeros((height, width), dtype=np.uint16)
    y0, y1 = CONFIG.detector.roi_y0, CONFIG.detector.roi_y1
    x0, x1 = CONFIG.detector.roi_x0, CONFIG.detector.roi_x1
    for y in range(y0, min(y1, height)):
        for x in range(x0, min(x1, width)):
            base_m = 0.55 + ((x - x0) / max(1, (x1 - x0))) * 0.08
            depth[y, x] = int(base_m / 0.001)
    return depth


def main():
    calib, target_dist = load_calib(Path(CONFIG.detector.calib_json))
    if float(CONFIG.detector.target_dist_m_override) > 0:
        target_dist = float(CONFIG.detector.target_dist_m_override)
    detector = OnlineTableEdgeDetector(calib, CONFIG.detector, target_dist)
    depth = build_synthetic_depth()
    result, _ = detector.process_depth(depth)
    print("synthetic result:")
    print({
        "edge_found": result.edge_found,
        "yaw_err_rad": result.yaw_err_rad,
        "dist_err_m": result.dist_err_m,
        "edge_confidence": result.edge_confidence,
        "point_count": result.point_count,
        "table_point_count": result.table_point_count,
    })
    if result.point_count <= 0:
        raise RuntimeError("synthetic detector produced no points")


if __name__ == "__main__":
    main()
