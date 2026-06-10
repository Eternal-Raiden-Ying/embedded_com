#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PIDAxisConfig:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    integral_limit: float = 0.5
    output_limit: float = 1.0
    derivative_alpha: float = 0.30
    deadband: float = 0.0
    min_abs_output: float = 0.0


@dataclass
class DockingControlConfig:
    min_confidence: float = 0.55
    obs_timeout_s: float = 0.35
    dt_min_s: float = 0.02
    reset_on_mode_change: bool = True

    coarse_align_enter_rad: float = 0.18
    coarse_align_exit_rad: float = 0.08
    spin_only_yaw_rad: float = 0.18

    precise_yaw_tol_rad: float = 0.025
    precise_dist_tol_m: float = 0.015  # Distance error stopping tolerance/threshold for precise approach (mapped from controlled_approach.target_dist_m in stage_params.yaml)
    precise_lateral_tol_m: float = 0.015
    precise_stable_s: float = 0.50

    coarse_max_wz_radps: float = 0.45
    approach_max_vx_mps: float = 0.28
    approach_max_vy_mps: float = 0.18
    approach_max_wz_radps: float = 0.32
    final_max_vx_mps: float = 0.12
    final_max_vy_mps: float = 0.12
    final_max_wz_radps: float = 0.18

    vx_slew_per_s: float = 0.80
    vy_slew_per_s: float = 0.80
    wz_slew_per_s: float = 1.20

    enable_lateral_control: bool = True

    yaw_pid: PIDAxisConfig = field(default_factory=lambda: PIDAxisConfig(
        kp=1.8,
        ki=0.02,
        kd=0.10,
        integral_limit=0.40,
        output_limit=0.80,
        derivative_alpha=0.35,
        deadband=0.010,
        min_abs_output=0.06,
    ))
    dist_pid: PIDAxisConfig = field(default_factory=lambda: PIDAxisConfig(
        kp=1.4,
        ki=0.03,
        kd=0.08,
        integral_limit=0.35,
        output_limit=0.40,
        derivative_alpha=0.30,
        deadband=0.004,
        min_abs_output=0.04,
    ))
    lateral_pid: PIDAxisConfig = field(default_factory=lambda: PIDAxisConfig(
        kp=1.2,
        ki=0.02,
        kd=0.08,
        integral_limit=0.30,
        output_limit=0.30,
        derivative_alpha=0.30,
        deadband=0.004,
        min_abs_output=0.04,
    ))


@dataclass
class EdgeControlObservation:
    ts: float
    edge_found: bool
    confidence: float
    yaw_err_rad: Optional[float] = None
    dist_err_m: Optional[float] = None
    lateral_err_m: Optional[float] = None
    edge_ready: bool = False
    source: str = ""

    @property
    def valid(self) -> bool:
        return bool(self.edge_found) and self.yaw_err_rad is not None and self.dist_err_m is not None


@dataclass
class DockingCommand:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    valid: bool = False
    mode: str = "HOLD"
    pose_locked: bool = False
    reason: str = ""
