#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Math and geometry utilities extracted for visual semantics and table docking."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np


def finite_percentiles(
    values: Any,
    qs: Sequence[Union[int, float]] = (10, 50, 90),
) -> Dict[str, Optional[float]]:
    """Calculate percentiles of finite values.

    This function filters out infinite and non-finite values (like NaN) before
    computing the percentiles.

    Args:
        values: Input array-like data.
        qs: A sequence of percentiles to calculate.

    Returns:
        A dictionary mapping e.g., 'p10', 'p50', 'p90' to their respective
        percentile values (float) or None if no finite elements exist.
    """
    try:
        arr = np.asarray(values, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
    except Exception:
        arr = np.asarray([], dtype=np.float32)
    if arr.size <= 0:
        return {f"p{int(q)}": None for q in qs}
    return {f"p{int(q)}": float(np.percentile(arr, q)) for q in qs}


def camera_points_to_robot(
    x_cam: Any,
    y_cam_down: Any,
    z_cam_forward: Any,
    *,
    pitch_deg: float,
    camera_height_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert RealSense camera coordinates to robot frame.

    Camera convention: X right, Y down, Z forward.
    Robot convention: X lateral, Y forward along ground, Z upward from ground.

    Positive pitch_deg means the optical axis is pitched downward.
    With pitch=0, Y_robot=Z_cam and Z_robot = camera_height - Y_cam.

    Args:
        x_cam: Camera X coordinates (right).
        y_cam_down: Camera Y coordinates (down).
        z_cam_forward: Camera Z coordinates (forward).
        pitch_deg: Pitch angle of the camera in degrees (pitched downward is positive).
        camera_height_m: Camera height from the ground in meters.

    Returns:
        A tuple (x_robot, y_robot, z_robot) of NumPy arrays in robot frame.
    """
    theta = math.radians(float(pitch_deg))
    c, s = math.cos(theta), math.sin(theta)
    x_r = np.asarray(x_cam, dtype=np.float32)
    y_down = np.asarray(y_cam_down, dtype=np.float32)
    z_fwd = np.asarray(z_cam_forward, dtype=np.float32)
    y_r = z_fwd * c - y_down * s
    z_r = float(camera_height_m) - (z_fwd * s + y_down * c)
    return x_r, y_r.astype(np.float32, copy=False), z_r.astype(np.float32, copy=False)


def weighted_line_fit(
    x: Any,
    y: Any,
    weights: Optional[Any] = None,
) -> Tuple[float, float]:
    """Perform a weighted first-order polynomial (line) fit.

    Args:
        x: Independent variables.
        y: Dependent variables.
        weights: Optional weights for each data point.

    Returns:
        A tuple of (slope, intercept) as floats.
    """
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    if weights is None:
        k, b = np.polyfit(x_arr, y_arr, 1)
        return float(k), float(b)
    w_arr = np.asarray(weights, dtype=np.float32)
    n = int(min(len(x_arr), len(y_arr), len(w_arr)))
    if n <= 0:
        k, b = np.polyfit(x_arr, y_arr, 1)
        return float(k), float(b)
    w_arr = np.clip(w_arr[:n], 0.2, 3.0)
    if not np.all(np.isfinite(w_arr)) or float(np.max(w_arr)) <= 0.0:
        k, b = np.polyfit(x_arr[:n], y_arr[:n], 1)
    else:
        k, b = np.polyfit(x_arr[:n], y_arr[:n], 1, w=w_arr)
    return float(k), float(b)


def ransac_line_fit(
    x: Any,
    y: Any,
    max_iterations: int = 50,
    inlier_threshold: float = 0.05,
) -> Tuple[Tuple[float, float], np.ndarray]:
    """Fit a line to 2D points using the RANSAC algorithm.

    Args:
        x: Input X coordinates.
        y: Input Y coordinates.
        max_iterations: Maximum number of RANSAC iterations.
        inlier_threshold: Distance threshold to consider a point as an inlier.

    Returns:
        A tuple containing:
            - A tuple (k, b) of the best line fit parameter (slope and intercept).
            - A boolean NumPy array representing the inlier mask.
    """
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    n = len(x_arr)
    if n < 2:
        return (0.0, 0.0), np.zeros(n, dtype=bool)

    best_k, best_b = 0.0, 0.0
    best_inliers = np.zeros(n, dtype=bool)
    max_inliers_count = -1

    rng = np.random.default_rng(seed=42)

    for _ in range(max_iterations):
        indices = rng.choice(n, size=2, replace=False)
        p1_x, p1_y = x_arr[indices[0]], y_arr[indices[0]]
        p2_x, p2_y = x_arr[indices[1]], y_arr[indices[1]]

        dx = p2_x - p1_x
        if abs(dx) < 1e-5:
            continue

        k = (p2_y - p1_y) / dx
        b = p1_y - k * p1_x

        # Calculate orthogonal distance of all points to the line:
        # d = |k*x - y + b| / sqrt(k^2 + 1)
        dist = np.abs(k * x_arr - y_arr + b) / math.sqrt(k * k + 1.0)
        inliers = dist <= inlier_threshold
        inlier_count = int(np.sum(inliers))

        if inlier_count > max_inliers_count:
            max_inliers_count = inlier_count
            best_inliers = inliers
            best_k = k
            best_b = b

    if max_inliers_count <= 0:
        p1_x, p1_y = x_arr[0], y_arr[0]
        p2_x, p2_y = x_arr[min(1, n - 1)], y_arr[min(1, n - 1)]
        dx = p2_x - p1_x
        best_k = (p2_y - p1_y) / dx if abs(dx) > 1e-5 else 0.0
        best_b = p1_y - best_k * p1_x
        best_inliers = np.ones(n, dtype=bool)

    # Refit using all inliers to improve precision
    if int(best_inliers.sum()) >= 2:
        try:
            k_refined, b_refined = np.polyfit(x_arr[best_inliers], y_arr[best_inliers], 1)
            best_k = float(k_refined)
            best_b = float(b_refined)
        except Exception:
            pass

    return (best_k, best_b), best_inliers

