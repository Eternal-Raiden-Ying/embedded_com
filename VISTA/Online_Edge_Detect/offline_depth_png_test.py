#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用在线版轻量检测器离线验证一张 16-bit depth png")
    parser.add_argument("--depth-png", type=Path, required=True)
    parser.add_argument("--calib-json", type=Path, default=Path(CONFIG.detector.calib_json))
    parser.add_argument("--out-dir", type=Path, default=Path("VISTA/Online_Edge_Detect/offline_test_output"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    depth_png = args.depth_png.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    depth_raw = cv2.imread(str(depth_png), cv2.IMREAD_ANYDEPTH)
    if depth_raw is None:
        raise RuntimeError(f"failed to load depth png: {depth_png}")

    calib, target_dist = load_calib(args.calib_json.resolve())
    if float(CONFIG.detector.target_dist_m_override) > 0:
        target_dist = float(CONFIG.detector.target_dist_m_override)
    detector = OnlineTableEdgeDetector(calib, CONFIG.detector, target_dist)
    result, debug = detector.process_depth(depth_raw)

    depth_vis = cv2.convertScaleAbs(depth_raw, alpha=0.03)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    x0, y0, x1, y1 = debug["roi_box"]
    cv2.rectangle(depth_vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
    text1 = f"edge={int(result.edge_found)} conf={result.edge_confidence:.3f}"
    text2 = f"yaw={result.yaw_err_rad:.4f} dist={result.dist_err_m:.4f}"
    text3 = f"points={result.point_count} table_points={result.table_point_count}"
    for idx, text in enumerate((text1, text2, text3)):
        cv2.putText(depth_vis, text, (20, 35 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    preview_path = out_dir / f"offline_preview_{stamp}.png"
    result_path = out_dir / f"offline_result_{stamp}.json"
    cv2.imwrite(str(preview_path), depth_vis)

    payload = {
        "depth_png": str(depth_png),
        "preview_png": str(preview_path),
        "edge_found": bool(result.edge_found),
        "yaw_err_rad": float(result.yaw_err_rad),
        "dist_err_m": float(result.dist_err_m),
        "edge_confidence": float(result.edge_confidence),
        "line_k": result.line_k,
        "line_b": result.line_b,
        "point_count": int(result.point_count),
        "table_point_count": int(result.table_point_count),
        "roi_box": list(debug["roi_box"]),
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
