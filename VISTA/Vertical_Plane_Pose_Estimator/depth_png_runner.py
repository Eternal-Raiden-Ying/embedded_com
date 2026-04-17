#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import cv2

try:
    from .estimator import VerticalPlanePoseEstimator
    from .schema import CameraIntrinsics, VerticalPlaneEstimatorConfig
except ImportError:
    from estimator import VerticalPlanePoseEstimator
    from schema import CameraIntrinsics, VerticalPlaneEstimatorConfig


def _load_calib(json_path: Path) -> CameraIntrinsics:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return CameraIntrinsics(
        fx=float(data["fx"]),
        fy=float(data["fy"]),
        cx=float(data["cx"]),
        cy=float(data["cy"]),
        depth_scale=float(data.get("depth_scale", 0.001)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Use the new vertical-plane estimator on a depth png")
    parser.add_argument("--depth-png", type=Path, required=True)
    parser.add_argument("--calib-json", type=Path, required=True)
    parser.add_argument("--target-distance-m", type=float, default=0.50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    depth_png = args.depth_png.resolve()
    calib_json = args.calib_json.resolve()

    depth = cv2.imread(str(depth_png), cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise RuntimeError("failed to load depth png: %s" % depth_png)

    intrinsics = _load_calib(calib_json)
    cfg = VerticalPlaneEstimatorConfig(target_distance_m=float(args.target_distance_m))
    estimator = VerticalPlanePoseEstimator(cfg)
    estimate = estimator.estimate_from_depth(depth, intrinsics)

    print(json.dumps({
        "depth_png": str(depth_png),
        "candidate_count": estimate.debug.candidate_count,
        "inlier_count": estimate.debug.inlier_count,
        "inlier_ratio": estimate.debug.inlier_ratio,
        "confidence": estimate.confidence,
        "yaw_err_rad": estimate.yaw_err_rad,
        "dist_err_m": estimate.dist_err_m,
        "valid": estimate.valid,
        "held": estimate.held,
        "plane": estimate.plane,
        "roi_box": list(estimate.debug.roi_box),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
