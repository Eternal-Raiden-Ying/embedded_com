#!/usr/bin/env python3
"""Pure-Python synthetic checks for table docking authority and ROI depth."""
from __future__ import annotations

from types import SimpleNamespace
import contextlib
import io
import os
import sys
import time
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "VISTA"))

import numpy as np

from orchestrator.orchestrator_service.runtime.control_authority import ControlAuthority, decide_table_control_authority
from orchestrator.orchestrator_service.runtime.motion_arbiter import MotionIntent, arbitrate_table_docking_motion
from orchestrator.orchestrator_service.runtime.perception_semantics import TablePerceptionSemantics
from orchestrator.orchestrator_service.runtime.states.table_docking import TableDockingMixin
from orchestrator.orchestrator_service.runtime.states.recovery import RecoveryMixin
from orchestrator.orchestrator_service.runtime.safety.stale_guard import StaleGuardMixin
from orchestrator.orchestrator_service.runtime.common import monotonic_ts
from orchestrator.orchestrator_service.runtime.context import RuntimeContext, State
from orchestrator.orchestrator_service.runtime.controller import MotionDecision
from orchestrator.orchestrator_service.runtime.service import OrchestratorService, sanitize_control_summary
from orchestrator.orchestrator_service.config.schema import CarMotionConfig, ControlThresholds
from orchestrator.orchestrator_service.control.motion_controller import MotionController
from orchestrator.orchestrator_service.ipc.protocol import CmdVel, TableEdgeObs, now_ts
from orchestrator.orchestrator_service.bridge.uart_bridge import UartBridge
from vision_module.backend.table_roi_depth import table_roi_depth_statistics
from vision_module.app.stages.search.table_edge_obs_builder import annotate_table_edge_obs, merge_table_bbox_from_local_perception
from orchestrator.orchestrator_service.runtime.docking_model import DockingAction, DockingStage
from common.config.loader import load_global_config


def auth(state: str, *, bbox: bool, edge: bool = False, error: float = 0.0, depth_stop: bool = False, phase: str = ""):
    return decide_table_control_authority(
        state,
        TablePerceptionSemantics(table_bbox_current_found=bbox, table_bbox_control_valid=bbox, edge_trusted=edge),
        SimpleNamespace(yolo_forward_center_hard_limit=0.25),
        depth_roi_stop_active=depth_stop,
        bbox_center_error=error,
        control_phase=phase,
    )


def main() -> None:
    effective_cfg = load_global_config(str(Path(ROOT) / "configs/system_config.yaml"))
    loaded_ctrl = effective_cfg.orchestrator.control
    loaded_car = effective_cfg.orchestrator.car
    assert not hasattr(effective_cfg.orchestrator.runtime, "stage_params_file")
    assert not hasattr(effective_cfg.orchestrator.runtime, "car_cmd_params_file")
    assert abs(loaded_car.search_table_wz_radps - 0.20) < 1e-9
    assert abs(loaded_ctrl.table_target_dist_m - 0.30) < 1e-9
    vista_cfg_text = (Path(ROOT) / "VISTA/configs/vision_params.yaml").read_text(encoding="utf-8")
    assert vista_cfg_text.count("target_dist_m: 0.30") >= 3
    assert abs(loaded_ctrl.min_forward_vx_mps - 0.04) < 1e-9
    assert abs(loaded_ctrl.bbox_track_forward_vx_mps - 0.10) < 1e-9
    assert abs(loaded_ctrl.bbox_track_forward_max_vx_mps - 0.20) < 1e-9
    assert abs(loaded_ctrl.bbox_track_forward_center_band - 0.45) < 1e-9
    assert abs(loaded_ctrl.final_servo_enter_p10_m - 0.45) < 1e-9
    assert abs(loaded_ctrl.edge_final_enter_margin_m - 0.06) < 1e-9
    assert abs(loaded_ctrl.edge_final_stop_margin_m - 0.02) < 1e-9
    assert abs(loaded_ctrl.close_range_enter_p10_m - 0.55) < 1e-9
    assert abs(loaded_ctrl.final_probe_vx_mps - 0.008) < 1e-9
    assert abs(loaded_ctrl.final_missing_probe_vx_mps - 0.004) < 1e-9
    assert abs(loaded_ctrl.close_range_probe_vx_mps - 0.008) < 1e-9
    assert abs(loaded_ctrl.close_range_missing_probe_vx_mps - 0.004) < 1e-9
    assert abs(loaded_ctrl.roi_final_stop_p10_m - 0.42) < 1e-9
    assert abs(loaded_ctrl.roi_final_slow_p10_m - 0.52) < 1e-9
    assert abs(loaded_ctrl.roi_final_probe_vx_mps - 0.008) < 1e-9
    assert abs(loaded_ctrl.roi_final_missing_probe_vx_mps - 0.004) < 1e-9
    assert abs(loaded_ctrl.roi_final_missing_hold_s - 0.8) < 1e-9
    assert abs(loaded_ctrl.depth_envelope_stop_p10_m - 0.35) < 1e-9
    assert abs(loaded_ctrl.depth_envelope_slow_p10_m - 0.50) < 1e-9
    assert abs(loaded_ctrl.depth_envelope_mid_p10_m - 0.70) < 1e-9
    assert abs(loaded_ctrl.depth_envelope_slow_vx_mps - 0.006) < 1e-9
    assert abs(loaded_ctrl.depth_envelope_mid_vx_mps - 0.015) < 1e-9
    assert abs(loaded_ctrl.far_bbox_track_vx_mps - 0.20) < 1e-9
    assert loaded_ctrl.bbox_track_forward_min_hold_ms == 800
    assert abs(loaded_ctrl.bbox_track_forward_max_wz_radps - 0.20) < 1e-9
    assert abs(loaded_ctrl.edge_readiness_yaw_max_rad - 0.35) < 1e-9
    assert abs(loaded_ctrl.edge_handoff_forward_vx_mps - 0.08) < 1e-9
    assert abs(loaded_ctrl.lateral_vy_max_mps - 0.18) < 1e-9
    assert abs(loaded_ctrl.lateral_kp - 0.30) < 1e-9
    assert abs(loaded_ctrl.lateral_deadband_norm - 0.020) < 1e-9
    assert loaded_ctrl.distance_scaled_lateral_enabled is True
    assert abs(loaded_ctrl.lateral_distance_ref_m - 0.50) < 1e-9
    assert abs(loaded_ctrl.far_lateral_vy_max_mps - 0.18) < 1e-9
    assert abs(loaded_ctrl.mid_lateral_vy_max_mps - 0.14) < 1e-9
    assert abs(loaded_ctrl.near_lateral_vy_max_mps - 0.060) < 1e-9
    assert abs(loaded_ctrl.lateral_priority_mid_error_norm - 0.99) < 1e-9
    assert abs(loaded_ctrl.lateral_priority_large_error_norm - 0.99) < 1e-9
    assert abs(loaded_ctrl.lateral_priority_mid_vx_cap_mps - 0.08) < 1e-9
    assert abs(loaded_ctrl.lateral_priority_vx_cap_mps - 0.04) < 1e-9
    assert loaded_ctrl.edge_yaw_align_allow_lateral is True
    assert abs(loaded_ctrl.edge_yaw_align_lateral_vy_max_mps - 0.08) < 1e-9
    assert abs(loaded_ctrl.yaw_flip_hold_window_s - 0.8) < 1e-9
    assert loaded_ctrl.yaw_flip_count_limit == 2
    assert abs(loaded_ctrl.yaw_ambiguous_wz_cap - 0.0) < 1e-9
    assert abs(loaded_ctrl.yaw_ambiguous_vy_boost - 1.5) < 1e-9
    assert abs(loaded_ctrl.final_dist_deadband_m - 0.03) < 1e-9
    assert abs(loaded_ctrl.final_dist_kp - 0.08) < 1e-9
    assert abs(loaded_ctrl.final_forward_vx_max_mps - 0.006) < 1e-9
    assert abs(loaded_ctrl.final_reverse_vx_max_mps - 0.004) < 1e-9
    assert loaded_ctrl.final_reverse_confirm_frames == 3
    assert abs(loaded_ctrl.forward_commit_min_s - 1.8) < 1e-9
    assert abs(loaded_ctrl.far_forward_commit_min_s - 2.0) < 1e-9
    assert loaded_ctrl.stop_after_table_docking is False
    assert abs(ControlThresholds().near_slow_max_vx_mps - 0.030) < 1e-9
    assert abs(ControlThresholds().bbox_track_forward_vx_mps - 0.100) < 1e-9
    assert abs(ControlThresholds().bbox_track_forward_max_vx_mps - 0.200) < 1e-9
    assert abs(ControlThresholds().bbox_track_forward_center_band - 0.45) < 1e-9
    assert abs(ControlThresholds().final_servo_enter_p10_m - 0.45) < 1e-9
    assert abs(ControlThresholds().final_probe_vx_mps - 0.008) < 1e-9
    assert abs(ControlThresholds().final_missing_probe_vx_mps - 0.004) < 1e-9
    assert abs(ControlThresholds().roi_final_probe_vx_mps - 0.008) < 1e-9
    assert abs(ControlThresholds().depth_envelope_slow_vx_mps - 0.006) < 1e-9
    for path in (
        Path(ROOT) / "orchestrator/orchestrator_service/config/schema.py",
        Path(ROOT) / "common/config/schema.py",
    ):
        text = path.read_text(encoding="utf-8")
        assert text.count("near_slow_max_vx_mps") == 1
        assert text.count("bbox_track_forward_center_band") == 1
        assert text.count("far_bbox_track_vx_mps") == 1
        assert text.count("far_forward_commit_min_s") == 1

    vista_obs = TableEdgeObs.from_dict({
        "ts": 1.0, "table_found": True, "edge_found": False,
        "yolo_table_visible": True, "yolo_table_fresh": True, "yolo_table_control_valid": True,
        "yolo_bbox_center_x_norm": 0.70, "table_bbox_touch_right": True,
    })
    assert vista_obs.table_bbox_current_found and vista_obs.table_bbox_control_valid
    assert vista_obs.yolo_bbox_center_x_norm == 0.70 and vista_obs.table_bbox_touch_right
    probe_geom = TableDockingMixin._bbox_control_geometry(SimpleNamespace(), vista_obs)
    assert probe_geom["bbox_center_valid"] and abs(probe_geom["bbox_center_error_control"] - 0.20) < 1e-6
    contract_obs = TableEdgeObs.from_dict({
        "ts": 1.0,
        "table_bbox_xyxy": [100, 20, 300, 200],
        "rgb_shape": [480, 640],
        "yolo_bbox_center_x_norm": 0.3125,
        "table_bbox_control_valid": True,
        "yolo_table_control_valid": True,
        "yolo_table_visible": True,
        "yolo_table_fresh": True,
        "table_bbox_touch_left": False,
        "table_bbox_touch_right": False,
        "table_bbox_touch_bottom": False,
        "edge_found": True,
        "edge_valid": True,
        "edge_trusted": True,
        "edge_confidence": 0.82,
        "yaw_err_rad": 0.10,
        "dist_err_m": 0.20,
        "table_roi_depth_valid": True,
        "table_roi_depth_p10": 0.45,
        "table_roi_depth_median": 0.50,
        "table_roi_depth_mean": 0.52,
        "table_roi_depth_sample_count": 96,
        "table_roi_depth_valid_ratio": 0.75,
    })
    assert contract_obs.table_bbox_current_found and contract_obs.table_bbox_control_valid
    assert contract_obs.yolo_table_control_valid and contract_obs.yolo_table_visible
    assert contract_obs.edge_confidence == 0.82
    assert contract_obs.table_roi_depth_mean == 0.52
    assert contract_obs.obs_parse_missing_fields == []
    confidence_regression = annotate_table_edge_obs(
        {"edge_found": True, "edge_valid": True, "confidence": 0.82, "edge_conf": 0.0, "edge_confidence": 0.0},
        tick_ts=1.0,
        source="results",
        source_mode="FIND_EDGE",
    )
    assert abs(confidence_regression["edge_conf"] - 0.82) < 1e-6
    assert abs(confidence_regression["edge_confidence"] - 0.82) < 1e-6
    parser_confidence_regression = TableEdgeObs.from_dict({"ts": 1.0, "confidence": 0.82, "edge_conf": 0.0, "edge_confidence": 0.0})
    assert abs((parser_confidence_regression.edge_conf or 0.0) - 0.82) < 1e-6
    assert abs((parser_confidence_regression.edge_confidence or 0.0) - 0.82) < 1e-6
    lateral_regression = annotate_table_edge_obs(
        {"edge_found": True, "edge_valid": True, "confidence": 0.82, "dist_err_m": 1.2},
        tick_ts=1.0,
        source="results",
        source_mode="FIND_EDGE",
    )
    assert lateral_regression["lateral_err_m"] is None
    assert lateral_regression["lateral"] is None
    parser_lateral_regression = TableEdgeObs.from_dict({"ts": 1.0, "dist_err_m": 1.2})
    assert parser_lateral_regression.lateral_err_m is None
    partial_contract_obs = TableEdgeObs.from_dict({"ts": 1.0, "table_found": True, "edge_found": False})
    assert "yolo_bbox_center_x_norm" in partial_contract_obs.obs_parse_missing_fields
    xyxy_obs = TableEdgeObs.from_dict({"ts": 1.0, "table_found": True, "edge_found": False,
        "table_bbox_xyxy": [320, 20, 640, 200], "rgb_shape": [480, 640]})
    xyxy_geom = TableDockingMixin._bbox_control_geometry(SimpleNamespace(), xyxy_obs)
    assert xyxy_geom["bbox_center_valid"] and abs(xyxy_geom["bbox_cx_norm_control"] - 0.75) < 1e-6
    assert xyxy_obs.table_bbox_control_valid and xyxy_obs.yolo_table_control_valid

    yolo_search_obs = TableEdgeObs.from_dict({
        "ts": 1.0,
        "table_found": True,
        "edge_found": False,
        "yolo_table_visible": True,
        "yolo_table_fresh": True,
        "yolo_table_control_valid": True,
        "table_bbox_xyxy": [327, 218, 633, 591],
        "rgb_shape": [640, 640, 3],
    })
    yolo_search_decision = MotionController(ControlThresholds(), CarMotionConfig()).yolo_table_search_cmd(yolo_search_obs)
    yolo_search_summary = yolo_search_decision.control_summary
    assert abs(yolo_search_summary["bbox_cx_norm"] - 0.75) < 1e-6
    assert abs(yolo_search_summary["center_error"] - 0.25) < 1e-6
    assert abs(yolo_search_summary["yolo_view_err_norm"] - 0.5) < 1e-6
    assert isinstance(yolo_search_decision, MotionDecision)

    # Test unavailable bbox center
    invalid_obs = TableEdgeObs.from_dict({"ts": 1.0, "table_found": True, "edge_found": False})
    invalid_geom = TableDockingMixin._bbox_control_geometry(SimpleNamespace(), invalid_obs)
    assert not invalid_geom["bbox_center_valid"]
    assert invalid_geom["bbox_cx_norm_control"] is None
    assert invalid_geom["bbox_center_error_control"] is None
    # Queue 2: final/depth stop no longer depends on a current bbox once the
    # depth/final path owns near docking.
    a = auth("YOLO_APPROACH", bbox=False, edge=True, depth_stop=True)
    assert (a.control_source, a.allow_forward, a.allow_rotate) == ("depth_roi_stop", False, False)
    a = auth("YOLO_APPROACH", bbox=True, error=0.3)
    assert (a.control_source, a.allow_forward, a.allow_rotate) == ("yolo_acquire_align", False, True)
    a = auth("YOLO_APPROACH", bbox=True, edge=False)
    assert a.control_source == "yolo_track_forward" and a.allow_forward and a.yaw_source == "yolo"
    a = auth("YOLO_APPROACH", bbox=True, edge=True)
    assert a.control_source == "edge_guided_forward" and a.yaw_source == "edge"
    a = auth("YOLO_APPROACH", bbox=True, depth_stop=True)
    assert a.control_source == "depth_roi_stop" and not a.allow_forward
    a = auth("YOLO_APPROACH", bbox=True, edge=True)
    assert a.stop_source == "none", "edge trust alone must not become a depth stop"
    assert auth("YOLO_APPROACH", bbox=True, edge=True, phase="BBOX_ACQUIRE").yaw_source == "bbox"
    assert auth("YOLO_APPROACH", bbox=True, edge=True, phase="EDGE_HANDOFF_CONFIRM").allow_forward is False
    assert auth("YOLO_APPROACH", bbox=True, edge=True, phase="EDGE_GUIDED_APPROACH").yaw_source == "edge"

    class DockingProbe(TableDockingMixin, StaleGuardMixin):
        def __init__(self):
            self.cfg = ControlThresholds()
            self.car_cfg = CarMotionConfig()
            self.ctx = RuntimeContext(state=State.YOLO_APPROACH)
            self.controller = MotionController(self.cfg, self.car_cfg)
            self.force_edge_guided = False
            self.force_depth_stop = False

        def _transition(self, state, _reason):
            self.ctx.prev_state = self.ctx.state
            self.ctx.state = state

        def _table_visible(self, _obs):
            return True

        def _log(self, *_args):
            pass

        def _enter_no_progress_recovery_or_next(self, reason: str):
            self.ctx.state = State.NO_PROGRESS_RECOVERY
            cmd = self.controller._cmd("NO_PROGRESS_RECOVERY", vx=0.0, wz=0.0)
            return MotionDecision(cmd=cmd, control_summary={"control_source": "no_progress_recovery", "state": "NO_PROGRESS_RECOVERY"})

        def _start_loss_timer(self, attr):
            if getattr(self.ctx, attr, 0.0) <= 0.0:
                setattr(self.ctx, attr, monotonic_ts())

        def _loss_elapsed(self, started):
            return max(0.0, monotonic_ts() - float(started or 0.0)) if started else 0.0

        def _get_control_authority(self, obs, depth_roi_stop_active=False, explicit_stop_active=False):
            if self.force_edge_guided:
                return ControlAuthority(
                    control_phase="EDGE_GUIDED_APPROACH", phase_reason="static_edge_handoff",
                    control_source="edge_guided_forward", yaw_source="edge", forward_source="edge",
                    stop_source="none", allow_forward=True, allow_rotate=True, block_reason="edge_handoff_complete",
                )
            return super()._get_control_authority(obs, depth_roi_stop_active, explicit_stop_active)

        def _depth_roi_stop_status(self, obs):
            if self.force_depth_stop:
                return {"depth_roi_stop_ready": True, "reason": "static_depth_final_stop"}
            return super()._depth_roi_stop_status(obs)

    def bbox_obs(center: float, *, soft_stale: bool = False, found: bool = True) -> TableEdgeObs:
        ts = now_ts() - (0.35 if soft_stale else 0.01)
        return TableEdgeObs.from_dict({
            "ts": ts,
            "frame_capture_ts": ts,
            "table_found": found,
            "table_bbox_current_found": found,
            "yolo_table_visible": found,
            "yolo_table_fresh": True,
            "yolo_table_control_valid": found,
            "yolo_table_age_ms": 350.0 if soft_stale else 10.0,
            "yolo_bbox_center_x_norm": center,
            "table_bbox_xyxy": [100, 20, 300, 200] if found else None,
            "rgb_shape": [480, 640],
            "edge_found": False,
            "depth_valid": True,
            "depth_p10": 0.60,
            "table_roi_depth_valid": True,
            "table_roi_depth_p10": 0.60,
            "table_roi_depth_median": 0.60,
        })

    def near_depth_obs(center: float, depth_m: float, *, yaw: float = 0.10) -> TableEdgeObs:
        obs = bbox_obs(center)
        obs.edge_found = True
        obs.edge_valid = True
        obs.edge_trusted = True
        obs.usable_for_approach = True
        obs.yaw_err_rad = yaw
        obs.table_roi_depth_valid = True
        obs.table_roi_depth_p10 = depth_m
        obs.table_roi_depth_median = depth_m
        obs.table_roi_depth_valid_ratio = 0.80
        obs.table_roi_depth_sample_count = 120
        return obs

    def authority_decision(probe: DockingProbe, obs: TableEdgeObs, raw_wz: float) -> MotionDecision:
        cmd = probe.controller._cmd("YOLO_APPROACH", vx=0.02, wz=raw_wz)
        return probe._apply_control_authority(
            MotionDecision(cmd=cmd, control_summary=probe.controller._summary("YOLO_APPROACH", cmd, obs)),
            obs,
        )

    def edge_guided_decision(probe: DockingProbe, obs: TableEdgeObs, *, edge_wz: float = 0.03) -> MotionDecision:
        probe.force_edge_guided = True
        probe.ctx.edge_handoff_complete = True
        cmd = probe.controller._cmd("YOLO_APPROACH", vx=0.0, wz=0.0)
        summary = probe.controller._summary("YOLO_APPROACH", cmd, obs)
        summary.update(
            {
                "edge_yaw_cmd": edge_wz,
                "edge_yaw": float(getattr(obs, "yaw_err_rad", 0.0) or 0.0),
                "hard_rotate_only_yaw_rad": 0.45,
                "pose_found": False,
                "pose_missing_duration_s": 99.0,
                "pose_missing_safe_vx_active": False,
                "forward_allowed": False,
            }
        )
        return probe._apply_control_authority(MotionDecision(cmd=cmd, control_summary=summary), obs)

    far_touch_probe = DockingProbe()
    far_touch_obs = bbox_obs(0.50)
    far_touch_obs.edge_found = True
    far_touch_obs.edge_valid = True
    far_touch_obs.edge_trusted = True
    far_touch_obs.usable_for_approach = True
    far_touch_obs.table_bbox_touch_bottom = True
    far_touch_obs.table_roi_depth_valid = True
    far_touch_obs.table_roi_depth_p10 = 1.20
    far_touch_obs.table_roi_depth_median = 1.36
    far_touch_summary = {"control_phase": "BBOX_ACQUIRE"}
    far_touch_probe._refresh_near_final_latches(far_touch_obs, {"depth_roi_stop_ready": False}, far_touch_summary)
    assert far_touch_probe.ctx.near_table_latched is False
    assert far_touch_summary["near_bbox_touch_hint"] is False

    unsafe_edge_obs = bbox_obs(0.50)
    unsafe_edge_obs.edge_found = True
    unsafe_edge_obs.edge_valid = False
    unsafe_edge_obs.edge_trusted = False
    unsafe_edge = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        unsafe_edge_obs,
        MotionIntent("edge_guided_forward", desired_vx=0.02, desired_wz=0.02, yaw_owner="edge", forward_allowed_by_behavior=True),
        {
            "control_phase": "EDGE_GUIDED_APPROACH",
            "edge_handoff_complete": False,
            "edge_readiness_score": 0.0,
            "edge_readiness_enter_score": 0.65,
            "yolo_forward_allowed": True,
            "bbox_yaw_cmd": 0.02,
        },
    )
    assert unsafe_edge.summary["docking_action"] != "EDGE_APPROACH_FORWARD"

    final_consume_probe = DockingProbe()
    final_consume_probe.ctx.state = State.YOLO_APPROACH
    final_consume_probe.ctx.final_depth_latched = True
    final_consume_probe.ctx.final_depth_latched_mono = monotonic_ts() - 2.0
    final_consume_probe.ctx.final_depth_stable_frames = final_consume_probe._final_depth_latch_frames()
    final_consume_obs = bbox_obs(0.50)
    final_consume_obs.yaw_err_rad = None
    final_cmd = final_consume_probe.controller._cmd("YOLO_APPROACH", vx=0.0, wz=0.0)
    final_decision = final_consume_probe._apply_control_authority(
        MotionDecision(cmd=final_cmd, control_summary=final_consume_probe.controller._summary("YOLO_APPROACH", final_cmd, final_consume_obs)),
        final_consume_obs,
    )
    assert final_consume_probe.ctx.state in {State.FINAL_SLOW_STOP, State.AT_TABLE_EDGE}
    assert final_consume_probe.ctx.state != State.YOLO_APPROACH
    assert final_decision.control_summary["docking_action"] == "FINAL_SLOW_PROBE"
    assert final_decision.control_summary.get("final_distance_servo_active", False) or final_consume_probe.ctx.final_locked is True

    # BBOX_ACQUIRE owns final yaw. Right side is positive/right turn even when
    # the raw/search command asks for the opposite turn.
    probe = DockingProbe()
    right = authority_decision(probe, bbox_obs(0.85), raw_wz=-0.10)
    assert "control_phase" not in right.control_summary
    assert right.control_summary["bbox_yaw_cmd"] > 0.0
    assert right.cmd.wz_radps > 0.0 and right.cmd.vx_mps == 0.0
    assert right.control_summary["bbox_yaw_owner_enforced"]

    probe = DockingProbe()
    left = authority_decision(probe, bbox_obs(0.15), raw_wz=0.10)
    assert left.control_summary["bbox_yaw_cmd"] < 0.0
    assert left.cmd.wz_radps < 0.0 and left.cmd.vx_mps == 0.0

    probe = DockingProbe()
    soft = authority_decision(probe, bbox_obs(0.85, soft_stale=True), raw_wz=-0.10)
    assert soft.control_summary["stale_level"] == "soft_stale"
    assert "control_phase" not in soft.control_summary
    assert soft.cmd.wz_radps == soft.control_summary["bbox_yaw_cmd"] > 0.0
    assert soft.cmd.vx_mps == 0.0

    edge_obs = bbox_obs(0.50)
    edge_obs.edge_found = True
    edge_obs.edge_trusted = True
    edge_obs.yaw_err_rad = 0.10
    edge_obs.table_roi_depth_p10 = 0.90
    edge_obs.table_roi_depth_median = 0.90
    probe = DockingProbe()
    committed = edge_guided_decision(probe, edge_obs)
    assert committed.cmd.vx_mps >= 0.04
    assert committed.cmd.vx_mps != 0.020
    assert committed.control_summary["approach_commit_active"]
    assert committed.control_summary["pose_gate_ignored_for_phase"]
    assert committed.control_summary["vx_override_reason"] == "edge_guided_commit"
    assert committed.cmd.wz_radps == committed.control_summary["edge_yaw_cmd"]

    # A committed approach must coast through a transient untrusted edge rather
    # than returning to a centered BBOX_ACQUIRE stop.
    edge_obs.edge_trusted = False
    probe.ctx.edge_conf_score = 0.50
    probe.ctx.last_edge_good_mono = monotonic_ts()
    coast = edge_guided_decision(probe, edge_obs)
    assert coast.cmd.vx_mps >= 0.04
    assert coast.cmd.vx_mps != 0.020
    assert coast.control_summary["forward_coast_active"]
    assert "control_phase" not in coast.control_summary
    assert coast.control_summary["docking_action"] == "EDGE_APPROACH_FORWARD"

    # A short hard-stale observation dropout after an established commit holds
    # the safe approach command instead of immediately stopping.
    probe = DockingProbe()
    dropout_good = bbox_obs(0.50)
    dropout_good.edge_found = True
    dropout_good.edge_trusted = True
    dropout_good.yaw_err_rad = 0.10
    dropout_good.table_roi_depth_p10 = 0.90
    dropout_good.table_roi_depth_median = 0.90
    edge_guided_decision(probe, dropout_good)
    edge_guided_decision(probe, dropout_good)
    assert probe.ctx.approach_commit_active and probe.ctx.last_good_table_obs_mono > 0.0
    dropout_obs = bbox_obs(0.50)
    dropout_obs.edge_found = True
    dropout_obs.edge_trusted = True
    dropout_obs.yaw_err_rad = 0.10
    dropout_obs.table_roi_depth_p10 = 0.90
    dropout_obs.table_roi_depth_median = 0.90
    dropout_obs.ts = now_ts() - 0.60
    dropout_obs.frame_capture_ts = dropout_obs.ts
    dropout = edge_guided_decision(probe, dropout_obs)
    assert dropout.cmd.vx_mps >= 0.04
    assert dropout.cmd.vx_mps != 0.020
    assert dropout.control_summary["perception_dropout_hold_active"]
    assert dropout.control_summary["stale_hold_policy"] == "approach_commit_short_dropout"
    assert dropout.control_summary["motion_class"] in {"recovery", "normal"}
    assert dropout.control_summary["stop_class"] == "none"

    # Once the dropout hold expires, the commit is cleared and normal stale
    # recovery may stop or search instead of continuing blind forward motion.
    probe.ctx.last_good_table_obs_mono = monotonic_ts() - 3.1
    expired_dropout = edge_guided_decision(probe, dropout_obs)
    assert expired_dropout.cmd.vx_mps == 0.0
    assert not probe.ctx.approach_commit_active
    assert expired_dropout.control_summary["stale_hold_policy"] == "perception_dropout_hold_expired"

    # A handoff timeout with a centered valid bbox must retain the committed
    # edge approach instead of falling into a double-zero BBOX_ACQUIRE command.
    probe = DockingProbe()
    probe.ctx.approach_commit_active = True
    probe.ctx.edge_conf_score = 0.50
    probe.ctx.last_edge_good_mono = monotonic_ts()
    probe.ctx.edge_handoff_started_mono = monotonic_ts() - 2.1
    timeout_obs = bbox_obs(0.50)
    timeout_obs.edge_found = True
    timeout_obs.edge_trusted = False
    timeout_phase = probe._control_phase_status(timeout_obs, depth_stop_ready=False)
    assert timeout_phase["control_phase"] == "EDGE_GUIDED_APPROACH"
    assert timeout_phase["phase_reason"] == "forward_coast_edge_unstable"

    # A committed double-zero command does not fall back to old low-speed caps
    # and does not pretend edge approach is ready.
    probe = DockingProbe()
    probe.force_edge_guided = True
    probe.ctx.approach_commit_active = True
    probe.ctx.edge_conf_score = 0.50
    probe.ctx.zero_cmd_started_mono = monotonic_ts() - 0.9
    watchdog_obs = bbox_obs(0.50)
    watchdog = authority_decision(probe, watchdog_obs, raw_wz=0.0)
    assert watchdog.control_summary["docking_action"] != "EDGE_APPROACH_FORWARD"
    assert watchdog.control_summary["stop_class"] == "none"

    # An emergency remains a hard forward-coast blocker.
    probe = DockingProbe()
    probe.force_edge_guided = True
    probe.ctx.approach_commit_active = True
    probe.ctx.edge_conf_score = 0.50
    emergency_obs = bbox_obs(0.50)
    emergency_obs.edge_found = True
    emergency_obs.edge_trusted = False
    emergency_cmd = probe.controller._cmd("YOLO_APPROACH", vx=0.0, wz=0.0)
    emergency = probe._apply_control_authority(
        MotionDecision(cmd=emergency_cmd, control_summary={"emergency_stop_active": True}), emergency_obs,
    )
    assert emergency.cmd.vx_mps == 0.0
    assert emergency.control_summary["stop_class"] == "emergency"

    uart_probe = UartBridge(port="COM_DRY_RUN", baudrate=115200, timeout_s=0.1, dry_run=True)
    assert uart_probe._writer_discard_reason({"line": "V 0.020 0.000 0.010", "tx_meta": {}}) == ""
    assert uart_probe._writer_discard_reason({"line": "V 0.000 0.000 0.100", "tx_meta": {}}) == ""
    assert uart_probe._writer_discard_reason({"line": "MODE SEARCH 0", "tx_meta": {}}) == ""
    assert uart_probe._writer_discard_reason({"line": "V 0.000 0.000 0.100", "tx_meta": {}, "publish_mono": time.monotonic()}) == ""
    uart_probe._last_estop_mono = time.monotonic()
    assert uart_probe._writer_discard_reason({"line": "V 0.000 0.000 0.100", "tx_meta": {}, "publish_mono": time.monotonic()}) == "estop_cooldown"
    assert uart_probe._writer_discard_reason({
        "line": "V 0.000 0.000 0.100",
        "tx_meta": {"stop_class": "stale_recovery", "estop_cooldown_applied": False},
        "publish_mono": time.monotonic(),
    }) == ""
    uart_events = []
    uart_tx_probe = UartBridge(port="COM_DRY_RUN", baudrate=115200, timeout_s=0.1, dry_run=True, dry_run_echo_stdout=False, tx_callback=lambda line, dry, meta: uart_events.append((line, dry, meta)))
    assert uart_tx_probe.send_velocity(0.020, 0.0, 0.010, tx_meta={"kind": "vel"})
    queued = uart_tx_probe._pop_pending_tx()
    assert queued is not None and queued["tx_meta"]["uart_enqueue_ok"]
    uart_tx_probe._write_line(queued["line"], tx_meta=queued["tx_meta"], publish_mono=queued["publish_mono"])
    assert uart_events[-1][2]["writer_accept_cmd"]
    assert uart_events[-1][2]["serial_write_attempted"] and uart_events[-1][2]["serial_write_ok"]
    assert uart_events[-1][2]["uart_tx_ok"]

    search_result = arbitrate_table_docking_motion(
        RuntimeContext(state=State.SEARCH_TABLE),
        None,
        MotionIntent(
            intent_type="local_rotate_search",
            desired_vx=0.0,
            desired_wz=0.10,
            yaw_owner="search",
            forward_allowed_by_behavior=False,
            rotate_allowed_by_behavior=True,
            reason="normal_stale_search_rotate",
        ),
        {"stale_level": "hard_stale", "control_phase": "SEARCH_SCAN"},
    )
    assert search_result.final_vx == 0.0 and abs(search_result.final_wz - 0.10) < 1e-6
    assert search_result.stop_class == "none"

    hard_stop_result = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("edge_guided_forward", desired_vx=0.02, desired_wz=0.0, forward_allowed_by_behavior=True),
        {"emergency_stop_active": True},
    )
    assert hard_stop_result.final_vx == 0.0 and hard_stop_result.final_wz == 0.0
    assert hard_stop_result.stop_class == "emergency"

    probe = DockingProbe()
    edge_obs.edge_trusted = True
    edge_obs.yaw_err_rad = 0.60
    yaw_blocked = edge_guided_decision(probe, edge_obs)
    assert yaw_blocked.cmd.vx_mps == 0.0
    assert yaw_blocked.cmd.wz_radps != 0.0
    assert yaw_blocked.control_summary["effective_block_reason"] == ""

    # A large, bottom-touching bbox with a mild side offset is no longer a FOV
    # guard; only bbox center controls motion.
    probe = DockingProbe()
    soft_fov_obs = bbox_obs(0.36)
    soft_fov_obs.edge_found = True
    soft_fov_obs.edge_trusted = True
    soft_fov_obs.yaw_err_rad = 0.10
    soft_fov_obs.table_bbox_touch_left = True
    soft_fov_obs.table_bbox_touch_bottom = True
    soft_fov_obs.table_bbox_area_ratio = 0.38
    soft_fov_obs.table_roi_depth_p10 = 0.90
    soft_fov_obs.table_roi_depth_median = 0.90
    soft_fov = edge_guided_decision(probe, soft_fov_obs)
    assert soft_fov.control_summary["fov_guard_level"] == "none"
    assert soft_fov.cmd.vx_mps >= 0.04
    assert soft_fov.cmd.vx_mps != 0.020
    assert soft_fov.control_summary["approach_commit_active"]
    assert soft_fov.control_summary["motion_class"] == "normal"
    assert soft_fov.control_summary["stop_class"] == "none"

    # Near-table latch: depth ROI is near but not yet final, so edge/near owns
    # yaw and YOLO is downgraded from primary control.
    probe = DockingProbe()
    near_obs = near_depth_obs(0.50, 0.35, yaw=0.08)
    edge_guided_decision(probe, near_obs)
    near_latched = edge_guided_decision(probe, near_obs)
    assert near_latched.control_summary["near_table_latched"]
    assert not near_latched.control_summary["yolo_control_enabled"]
    assert near_latched.control_summary["near_stage_yaw_source"] in {"edge", "last_good_edge"}
    assert near_latched.control_summary["yaw_owner"] not in {"bbox", "yolo"}

    # Final depth latch + large yaw: final mode disables omega and only allows
    # the very small final distance/ROI motion.
    probe = DockingProbe()
    probe.ctx.near_table_latched = True
    probe.ctx.final_depth_latched = True
    probe.ctx.final_depth_latch_reason = "static_final_depth"
    final_yaw_deadband = abs(float(getattr(probe.cfg, "final_lock_yaw_tol_rad", getattr(probe.cfg, "table_yaw_tol_rad", 0.08)) or 0.08))
    large_yaw = max(final_yaw_deadband + 0.05, 0.30)
    final_yaw_obs = near_depth_obs(0.50, 0.50, yaw=large_yaw)
    final_yaw = edge_guided_decision(probe, final_yaw_obs)
    assert abs(final_yaw.cmd.vx_mps) <= probe.cfg.final_forward_vx_max_mps
    assert final_yaw.cmd.vy_mps == 0.0
    assert final_yaw.cmd.wz_radps == 0.0
    assert final_yaw.control_summary["final_depth_latched"]
    assert not final_yaw.control_summary["final_yaw_align_active"]
    assert not final_yaw.control_summary["final_locked"]
    assert final_yaw.control_summary["yaw_owner"] == "none"

    # Final yaw align must be applied to the actual CmdVel, not only summary.
    probe = DockingProbe()
    final_align_cmd = probe.controller._cmd("YOLO_APPROACH", vx=0.02, wz=0.0)
    final_align_decision = MotionDecision(
        cmd=final_align_cmd,
        control_summary={
            "control_phase": "DEPTH_FINAL_STOP",
            "control_source": "depth_roi_stop",
            "final_depth_latched": True,
            "final_yaw_align_active": True,
            "final_locked": False,
            "edge_yaw": 0.30,
            "final_yaw_deadband_rad": 0.08,
            "edge_yaw_cmd_for_final_align": -0.15,
            "allow_forward": False,
            "allow_rotate": False,
            "rotate_block_reason": "allow_rotate_false_intercept",
        },
    )
    final_align_applied = probe._arbitrate_table_motion_decision(final_align_decision, final_yaw_obs)
    assert abs(final_align_applied.cmd.vx_mps) <= probe.cfg.final_forward_vx_max_mps
    assert final_align_applied.cmd.vy_mps == 0.0
    assert final_align_applied.cmd.wz_radps == 0.0
    assert final_align_applied.control_summary["yaw_owner"] == "none"

    # Final depth latch + stable small yaw: become locked instead of returning
    # to SEARCH_TABLE.
    probe = DockingProbe()
    probe.ctx.near_table_latched = True
    probe.ctx.final_depth_latched = True
    probe.ctx.final_depth_latch_reason = "static_final_depth"
    probe.ctx.final_yaw_aligned_frames = 5
    small_yaw = min(final_yaw_deadband * 0.5, 0.10)
    final_lock_obs = near_depth_obs(0.50, 0.50, yaw=small_yaw)
    final_lock = edge_guided_decision(probe, final_lock_obs)
    assert final_lock.cmd.vx_mps == 0.0 and final_lock.cmd.wz_radps == 0.0
    assert final_lock.control_summary["final_depth_latched"]
    assert final_lock.control_summary["final_locked"]
    assert not final_lock.control_summary["final_yaw_align_active"]
    assert probe.ctx.state != State.SEARCH_TABLE

    # Near latch + bbox lost: do not transition back to SEARCH_TABLE and do not
    # allow SEARCH_SCAN to take ownership.
    probe = DockingProbe()
    probe.ctx.near_table_latched = True
    probe.ctx.near_table_latch_reason = "static_near"
    lost_near = bbox_obs(0.50, found=False)
    held_near = probe._bbox_lost_hold_or_search(lost_near, "YOLO_APPROACH")
    assert probe.ctx.state == State.YOLO_APPROACH
    assert held_near.control_summary["bbox_lost_ignored_due_to_near_latch"]
    assert held_near.control_summary["control_phase"] != "SEARCH_SCAN"

    # Final depth latch outranks an already-active forward coast.
    probe = DockingProbe()
    probe.ctx.near_table_latched = True
    probe.ctx.final_depth_latched = True
    probe.ctx.final_depth_latch_reason = "static_final_depth"
    probe.ctx.approach_commit_active = True
    probe.ctx.last_edge_yaw_cmd = 0.04
    coast_blocked = authority_decision(probe, final_yaw_obs, raw_wz=0.04)
    assert abs(coast_blocked.cmd.vx_mps) <= probe.cfg.final_forward_vx_max_mps
    assert coast_blocked.control_summary["final_depth_latched"]
    assert coast_blocked.cmd.vy_mps == 0.0
    assert coast_blocked.cmd.wz_radps == 0.0

    # Emergency/explicit hard safety clears near/final latches.
    probe = DockingProbe()
    probe.ctx.near_table_latched = True
    probe.ctx.final_depth_latched = True
    emergency_clear = probe._apply_control_authority(
        MotionDecision(cmd=probe.controller._cmd("YOLO_APPROACH", vx=0.0, wz=0.0), control_summary={"emergency_stop_active": True}),
        final_yaw_obs,
    )
    assert emergency_clear.cmd.vx_mps == 0.0
    assert emergency_clear.cmd.wz_radps == 0.0
    assert not probe.ctx.near_table_latched and not probe.ctx.final_depth_latched

    # Service must not overwrite the arbiter final yaw align CmdVel.
    service_probe = OrchestratorService.__new__(OrchestratorService)
    service_probe.cfg = SimpleNamespace(
        car=SimpleNamespace(
            motion_hold_ms=150,
            cmd_hold_ms=150,
            hard_stale_stop_ms=800,
            soft_stale_hold_enable=True,
        )
    )
    service_probe.core = SimpleNamespace(ctx=RuntimeContext(state=State.YOLO_APPROACH))
    service_probe._last_valid_motion_cmd = None
    service_probe._last_valid_motion_ts = 0.0
    arbiter_final_cmd = CmdVel(ts=now_ts(), mode="YOLO_APPROACH", vx_mps=0.0, vy_mps=0.0, wz_radps=-0.15, hold_ms=150, brake=False)
    effective_final_yaw, service_meta = service_probe._arbitrate_uart_motion_cmd(
        arbiter_final_cmd,
        {
            "arbiter_applied": True,
            "final_cmd_source": "arbiter_final_yaw_align",
            "final_yaw_align_active": True,
            "final_yaw_align_yaw_cmd": -0.15,
            "final_wz": -0.15,
            "motion_class": "normal",
            "stop_class": "none",
        },
    )
    assert effective_final_yaw.vx_mps == 0.0 and effective_final_yaw.vy_mps == 0.0
    assert abs(effective_final_yaw.wz_radps + 0.15) < 1e-6
    assert not service_meta["service_override"]
    assert service_meta["effective_cmd_after_service"]["wz_radps"] == -0.15

    # Extreme center error remains a hard guard.
    probe = DockingProbe()
    extreme_fov_obs = bbox_obs(0.10)
    extreme_fov_obs.edge_found = True
    extreme_fov_obs.edge_trusted = True
    extreme_fov = edge_guided_decision(probe, extreme_fov_obs)
    assert extreme_fov.control_summary["fov_guard_level"] == "hard"
    assert extreme_fov.cmd.vx_mps == 0.0
    assert extreme_fov.cmd.wz_radps != 0.0
    assert extreme_fov.control_summary["effective_block_reason"] == ""

    # Side/bottom touch never becomes motion-control FOV; only center extreme is hard.
    probe = DockingProbe()
    side_hard_obs = bbox_obs(0.70)
    side_hard_obs.edge_found = True
    side_hard_obs.edge_trusted = True
    side_hard_obs.table_bbox_touch_right = True
    side_hard_obs.table_bbox_touch_bottom = True
    side_hard_obs.table_roi_depth_valid = True
    side_hard_obs.table_roi_depth_p10 = 1.30
    side_hard_obs.table_roi_depth_median = 1.30
    probe.ctx.bbox_fov_violation_streak = 3
    fov_blocked = edge_guided_decision(probe, side_hard_obs)
    assert fov_blocked.control_summary["fov_guard_level"] == "none"
    assert fov_blocked.control_summary["docking_action"] in {"BBOX_TRACK_FORWARD", "EDGE_APPROACH_FORWARD"}
    assert fov_blocked.cmd.vx_mps > 0.0
    assert fov_blocked.cmd.vy_mps < 0.0

    probe = DockingProbe()
    probe.force_depth_stop = True
    depth_stopped = authority_decision(probe, bbox_obs(0.50), raw_wz=0.03)
    assert "control_phase" not in depth_stopped.control_summary
    assert depth_stopped.control_summary["docking_action"] == "FINAL_SLOW_PROBE"
    assert abs(depth_stopped.cmd.vx_mps) <= max(probe.cfg.final_forward_vx_max_mps, probe.cfg.roi_final_probe_vx_mps)
    assert depth_stopped.cmd.vy_mps == 0.0
    assert depth_stopped.cmd.wz_radps == 0.0

    probe = DockingProbe()
    probe.ctx.bbox_valid_streak = 3
    probe.ctx.edge_handoff_complete = True
    lost = bbox_obs(0.5, found=False)
    held = probe._bbox_lost_hold_or_search(lost, "YOLO_APPROACH")
    assert held.control_summary["bbox_lost_hold_active"]
    assert held.cmd.wz_radps == 0.0
    assert held.cmd.vx_mps == 0.0
    assert held.control_summary["bbox_lost_hold_reason"] != "bbox_lost_hold_rotate"

    safe_lost = bbox_obs(0.5, found=False)
    safe_lost.edge_found = True
    safe_lost.edge_valid = True
    safe_lost.usable_for_approach = True
    safe_lost.table_roi_depth_valid = True
    safe_lost.table_roi_depth_p10 = 0.90
    safe_lost.table_roi_depth_median = 0.95
    probe.ctx.bbox_lost_since_mono = 0.0
    safe_held = probe._bbox_lost_hold_or_search(safe_lost, "YOLO_APPROACH")
    assert safe_held.control_summary["bbox_lost_hold_active"]
    assert safe_held.cmd.vx_mps >= 0.04
    assert safe_held.control_summary["bbox_lost_hold_reason"] == "bbox_lost_edge_depth_safe_forward_hold"

    assert probe.ctx.state == State.YOLO_APPROACH
    assert probe.ctx.bbox_valid_streak == 3 and probe.ctx.edge_handoff_complete
    probe.ctx.bbox_lost_since_mono = monotonic_ts() - float(probe.cfg.table_loss_hold_s) - 0.1
    expired = probe._bbox_lost_hold_or_search(lost, "YOLO_APPROACH")
    assert probe.ctx.state == State.SEARCH_TABLE
    assert not probe.ctx.edge_handoff_complete and probe.ctx.bbox_valid_streak == 0
    assert expired.cmd.mode == "SEARCH_TABLE"

    depth = np.full((100, 100), 1000, dtype=np.uint16)
    stats = table_roi_depth_statistics(depth, 0.001, [10, 10, 90, 90])
    assert stats["table_roi_depth_valid"] and stats["table_roi_depth_p10"] == 1.0
    assert stats["table_roi_depth_mean"] == 1.0
    assert stats["table_roi_depth_bbox"] and stats["table_roi_depth_coord_space"] == "depth_frame_xyxy"
    depth[:] = 0
    invalid = table_roi_depth_statistics(depth, 0.001, [10, 10, 90, 90])
    assert not invalid["table_roi_depth_valid"]
    no_bbox = table_roi_depth_statistics(depth + 1000, 0.001, [10, 10, 90, 90], current_table_bbox_found=False)
    assert not no_bbox["table_roi_depth_valid"] and no_bbox["table_roi_depth_p10"] is None
    merged = merge_table_bbox_from_local_perception(
        {"table_roi_depth_valid": True, "table_roi_depth_p10": 0.2, "table_roi_depth_sample_count": 100},
        {"table_bbox_current_found": False, "table_found": False}, tick_ts=1.0,
    )
    assert not merged["table_roi_depth_valid"] and merged["table_roi_depth_p10"] is None
    merged_bbox = merge_table_bbox_from_local_perception(
        {"edge_found": True, "edge_valid": True, "edge_trusted": True, "edge_conf": 0.9},
        {"table_bbox_current_found": True, "table_found": True, "table_bbox_xyxy": [100, 20, 300, 200], "rgb_shape": [480, 640]},
        tick_ts=1.0,
    )
    assert merged_bbox["table_bbox_control_valid"] and merged_bbox["yolo_table_control_valid"]
    assert abs(float(merged_bbox["yolo_bbox_center_x_norm"]) - 0.3125) < 1e-6

    class ProgressProbe(TableDockingMixin):
        def __init__(self):
            self.cfg = SimpleNamespace(table_target_dist_m=0.5)  # deliberately lacks progress_window_ms
            self.ctx = SimpleNamespace(min_dist_seen=999.0, dist_progress_last_refreshed_mono=0.0, dist_missing_started_mono=0.0)
        def _log(self, *_args):
            pass

    probe = ProgressProbe()
    assert not probe._check_approach_progress(None)
    assert probe.ctx.dist_missing_started_mono > 0.0
    probe.ctx.dist_missing_started_mono = monotonic_ts() - 15.1
    assert probe._check_approach_progress(None)

    # 1. BBOX_ACQUIRE phase may move forward aggressively; ensure no-progress
    # is NOT triggered even after 5s.
    p_acquire = DockingProbe()
    p_acquire.ctx.control_phase = "BBOX_ACQUIRE"
    obs_acq = bbox_obs(0.70)
    obs_acq.dist_err_m = 0.5
    obs_acq.target_dist_m = 0.5
    # Call authority_decision multiple times, or manually simulate time elapsed
    dec = authority_decision(p_acquire, obs_acq, raw_wz=0.10)
    assert dec.control_summary["docking_action"] in {"BBOX_TRACK_FORWARD", "BBOX_REACQUIRE_ROTATE", "EDGE_APPROACH_FORWARD"}
    # Simulate 5s elapsed
    p_acquire.ctx.dist_missing_started_mono = monotonic_ts() - 6.0
    p_acquire.ctx.dist_progress_last_refreshed_mono = monotonic_ts() - 6.0
    # Call again
    dec = authority_decision(p_acquire, obs_acq, raw_wz=0.10)
    assert p_acquire.ctx.state != State.NO_PROGRESS_RECOVERY  # Should NOT trigger recovery

    # 2. EDGE_HANDOFF_CONFIRM phase, vx=0. Ensure no-progress is NOT triggered
    p_confirm = DockingProbe()
    p_confirm.ctx.control_phase = "EDGE_HANDOFF_CONFIRM"
    obs_conf = bbox_obs(0.50)
    obs_conf.dist_err_m = 0.5
    # Simulate 5s elapsed
    p_confirm.ctx.dist_missing_started_mono = monotonic_ts() - 6.0
    p_confirm.ctx.dist_progress_last_refreshed_mono = monotonic_ts() - 6.0
    dec = authority_decision(p_confirm, obs_conf, raw_wz=0.10)
    assert p_confirm.ctx.state != State.NO_PROGRESS_RECOVERY  # Should NOT trigger recovery

    # 3. Slow EDGE_GUIDED_APPROACH: five seconds of unchanged distance is a
    # warning only; recovery is not allowed until the 15-second window expires.
    p_approach = DockingProbe()
    p_approach.force_edge_guided = True
    p_approach.ctx.edge_handoff_complete = True
    p_approach.ctx.control_phase = "EDGE_GUIDED_APPROACH"
    obs_app = bbox_obs(0.50)
    obs_app.edge_found = True
    obs_app.edge_trusted = True
    obs_app.dist_err_m = 0.5
    obs_app.target_dist_m = 0.5
    obs_app.table_roi_depth_p10 = 0.90
    obs_app.table_roi_depth_median = 0.90
    
    # First call resets/initializes min_dist_seen
    dec = edge_guided_decision(p_approach, obs_app)
    assert dec.cmd.vx_mps >= 0.04
    assert dec.cmd.vx_mps != 0.02
    assert p_approach.ctx.state != State.NO_PROGRESS_RECOVERY
    
    # Five seconds without distance change remains tolerated for aggressive forward.
    p_approach.ctx.dist_progress_last_refreshed_mono = monotonic_ts() - 6.0
    dec = edge_guided_decision(p_approach, obs_app)
    assert p_approach.ctx.state != State.NO_PROGRESS_RECOVERY
    assert dec.cmd.vx_mps >= 0.04
    assert dec.cmd.vx_mps != 0.02
    assert dec.control_summary["no_progress_warning"]
    assert dec.control_summary["no_progress_policy"] == "slow_forward_tolerated"

    # Only the longer 15-second window may permit strong recovery.
    p_approach.ctx.dist_progress_last_refreshed_mono = monotonic_ts() - 15.1
    dec = edge_guided_decision(p_approach, obs_app)
    assert p_approach.ctx.state == State.NO_PROGRESS_RECOVERY
    assert dec.control_summary["progress_recovery_allowed"]

    # Close/final latch blocks no-progress recovery even when forward probe is
    # slow and distance does not decrease.
    p_final_progress = DockingProbe()
    p_final_progress.force_edge_guided = True
    p_final_progress.ctx.edge_handoff_complete = True
    p_final_progress.ctx.control_phase = "EDGE_GUIDED_APPROACH"
    p_final_progress.ctx.dist_progress_last_refreshed_mono = monotonic_ts() - 20.0
    obs_final_progress = bbox_obs(0.50)
    obs_final_progress.table_roi_depth_valid = True
    obs_final_progress.table_roi_depth_p10 = 0.54
    obs_final_progress.table_roi_depth_median = 0.54
    dec = edge_guided_decision(p_final_progress, obs_final_progress)
    assert p_final_progress.ctx.state != State.NO_PROGRESS_RECOVERY
    assert dec.control_summary["close_range_latched"]
    assert dec.control_summary["progress_recovery_allowed"] is False
    assert dec.control_summary["progress_recovery_block_reason"] in {"final_or_close_range_latched", "final_vx_zero_or_non_edge_phase"}

    class RecoveryProbe(RecoveryMixin, TableDockingMixin):
        def __init__(self, *, multi_table_enabled: bool):
            self.cfg = ControlThresholds(multi_table_enabled=multi_table_enabled)
            self.car_cfg = CarMotionConfig()
            self.ctx = RuntimeContext(state=State.YOLO_APPROACH)
            self.ctx.no_progress_recovery_count = self.cfg.dock_retry_limit
            self.controller = MotionController(self.cfg, self.car_cfg)

        def _transition(self, state, _reason):
            self.ctx.prev_state = self.ctx.state
            self.ctx.state = state

    # Exhausted no-progress retries remain on the current single-table task by
    # default; NEXT_TABLE is reserved for explicit multi-table mode.
    recovery_probe = RecoveryProbe(multi_table_enabled=False)
    recovery_probe._enter_no_progress_recovery_or_next("static no progress")
    assert recovery_probe.ctx.state == State.SEARCH_TABLE
    multi_table_probe = RecoveryProbe(multi_table_enabled=True)
    multi_table_probe._enter_no_progress_recovery_or_next("static no progress")
    assert multi_table_probe.ctx.state == State.NEXT_TABLE

    # Required docking motion invariants for the typed arbiter.
    inv1 = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_ACQUIRE_ALIGN),
        None,
        MotionIntent("yolo_acquire_align", desired_vx=0.0, desired_wz=0.06, yaw_owner="bbox", rotate_allowed_by_behavior=True),
        {
            "control_phase": "BBOX_ACQUIRE",
            "bbox_center_valid": True,
            "bbox_center_error_control": 0.45,
            "bbox_yaw_cmd": 0.06,
            "bbox_fov_guard_level": "hard",
            "bbox_fov_guard_reason": "bbox_center_extreme",
        },
    )
    assert inv1.summary["docking_action"] == "BBOX_REACQUIRE_ROTATE"
    assert inv1.final_vx == 0.0 and abs(inv1.final_wz) > 0.0
    assert inv1.stop_class == "none"

    inv2 = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("edge_guided_forward", desired_vx=0.02, desired_wz=0.0, yaw_owner="edge", forward_allowed_by_behavior=True),
        {
            "control_phase": "EDGE_GUIDED_APPROACH",
            "edge_guided_commit_block_reason": "bbox_fov_guard_hard",
            "bbox_yaw_cmd": -0.05,
            "bbox_fov_guard_level": "hard",
            "bbox_fov_guard_reason": "side_touch_center_error_streak",
        },
    )

    assert inv2.summary["docking_action"] == "BBOX_REACQUIRE_ROTATE"
    assert inv2.final_vx == 0.0 and abs(inv2.final_wz) > 0.0
    assert inv2.stop_class == "none"

    inv3 = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("final_align", desired_vx=0.02, desired_wz=0.0, yaw_owner="edge"),
        {
            "control_phase": "DEPTH_FINAL_STOP",
            "final_depth_latched": True,
            "final_yaw_align_active": True,
            "edge_yaw": 0.30,
            "final_yaw_deadband_rad": 0.08,
            "edge_yaw_cmd_for_final_align": -0.12,
            "edge_usable": True,
        },
    )

    assert inv3.summary["docking_action"] == "CLOSE_RANGE_PROBE"
    assert inv3.summary["docking_action"] != "FINAL_LOCKED_STOP"
    assert inv3.final_vx == 0.0 and inv3.final_vy == 0.0 and inv3.final_wz == 0.0
    assert inv3.summary["yaw_owner"] == "none"

    inv4 = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("final_hold", desired_vx=0.02, desired_wz=0.0),
        {
            "control_phase": "DEPTH_FINAL_STOP",
            "final_depth_latched": True,
            "final_locked": True,
            "edge_yaw": 0.01,
            "final_yaw_deadband_rad": 0.08,
        },
    )
    assert inv4.summary["docking_action"] == "FINAL_LOCKED_STOP"
    assert inv4.final_vx == 0.0 and inv4.final_vy == 0.0 and inv4.final_wz == 0.0

    inv5 = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("local_rotate_search", desired_vx=0.0, desired_wz=0.10, yaw_owner="search", rotate_allowed_by_behavior=True),
        {
            "control_phase": "SEARCH_SCAN",
            "near_table_latched": True,
            "last_good_edge_yaw_cmd": 0.04,
        },
    )
    assert inv5.summary["docking_stage"] != "SEARCH"
    assert inv5.summary["docking_action"] != "SEARCH_ROTATE"
    assert inv5.summary["docking_action"] == "NEAR_EDGE_FORWARD"
    assert inv5.reason == "near_hold"
    assert inv5.final_wz != 0.0
    assert inv5.summary["effective_block_reason"] == ""

    inv6 = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("local_rotate_search", desired_vx=0.0, desired_wz=0.10, yaw_owner="search", rotate_allowed_by_behavior=True),
        {
            "control_phase": "SEARCH_SCAN",
            "near_table_latched": True,
            "final_depth_latched": True,
            "final_yaw_align_active": True,
            "edge_yaw": 0.30,
            "final_yaw_deadband_rad": 0.08,
            "last_good_edge_yaw_cmd": 0.05,
            "last_good_edge_yaw_age_ms": 100.0,
            "edge_usable": True,
        },
    )
    assert inv6.summary["docking_stage"] != "SEARCH"
    assert inv6.summary["docking_action"] != "SEARCH_ROTATE"

    inv7_uart = UartBridge(port="COM_DRY_RUN", baudrate=115200, timeout_s=0.1, dry_run=True)
    assert inv7_uart._writer_discard_reason({
        "line": "V 0.000 0.000 -0.100",
        "tx_meta": {"stop_class": "none", "estop_cooldown_applied": False},
        "publish_mono": time.monotonic(),
    }) == ""

    inv8 = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("edge_guided_forward", desired_vx=0.0, desired_wz=0.0, yaw_owner="edge"),
        {
            "control_phase": "EDGE_GUIDED_APPROACH",
            "zero_cmd_age_ms": 1200.0,
            "bbox_center_valid": True,
            "bbox_yaw_cmd": 0.05,
        },
    )
    assert inv8.summary["docking_action"] in {"BBOX_REACQUIRE_ROTATE", "SEARCH_ROTATE", "NEAR_EDGE_FORWARD"}
    assert inv8.final_vx != 0.0 or inv8.final_wz != 0.0
    assert inv8.stop_class == "none"

    inv9 = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("approach_commit_dropout_hold", desired_vx=0.02, desired_wz=0.03, yaw_owner="edge_hold", forward_allowed_by_behavior=True),
        {
            "control_phase": "EDGE_GUIDED_APPROACH",
            "perception_dropout_hold_active": True,
            "stale_hold_policy": "approach_commit_short_dropout",
            "last_edge_yaw_cmd": 0.03,
            "last_good_edge_yaw_age_ms": 300.0,
        },
    )
    assert inv9.summary["docking_action"] == "PERCEPTION_DROPOUT_HOLD"
    assert inv9.summary["docking_stage"] == "PERCEPTION_DROPOUT_HOLD"
    assert inv9.stop_class == "none"

    inv9_bbox_track = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("yolo_track_forward", desired_vx=0.012, desired_wz=0.03, yaw_owner="bbox", forward_allowed_by_behavior=True),
        {
            "control_phase": "BBOX_ACQUIRE",
            "bbox_center_valid": True,
            "bbox_center_error": 0.02,
            "bbox_yaw_cmd": 0.03,
            "yolo_forward_allowed": True,
            "yolo_forward_center_good_limit": 0.15,
            "bbox_fov_guard_level": "none",
            "cmd": {"vx_mps": 0.012},
            "table_roi_depth_valid": True,
            "table_roi_depth_p10": 0.80,
        },
    )
    assert inv9_bbox_track.summary["docking_stage"] == "BBOX_ACQUIRE"
    assert inv9_bbox_track.summary["docking_action"] == "BBOX_TRACK_FORWARD"
    assert inv9_bbox_track.summary["docking_action"] != "EDGE_APPROACH_FORWARD"
    assert inv9_bbox_track.final_vx > 0.0 and inv9_bbox_track.final_vy == 0.0
    assert inv9_bbox_track.summary["yaw_owner"] == "bbox"
    assert inv9_bbox_track.summary["forward_owner"] == "bbox_track"
    assert inv9_bbox_track.summary["lateral_owner"] == "none"
    assert inv9_bbox_track.final_vx >= 0.04
    assert inv9_bbox_track.final_vx not in {0.008, 0.010, 0.012, 0.015, 0.020}
    assert abs(inv9_bbox_track.final_wz) <= 0.20

    slim_probe = DockingProbe()
    slim_obs = bbox_obs(0.50)
    slim_obs.table_roi_depth_p10 = 0.90
    slim_obs.table_roi_depth_median = 0.90
    slim_cmd = slim_probe.controller._cmd("YOLO_APPROACH", vx=0.012, wz=0.01)
    slim_summary = slim_probe.controller._summary("YOLO_APPROACH", slim_cmd, slim_obs)
    for key in (
        "camera_frame_interval_ms",
        "camera_frame_hz",
        "vision_process_interval_ms",
        "vision_publish_interval_ms",
        "obs_out_send_interval_ms",
        "obs_out_send_hz",
        "table_edge_obs_recv_interval_ms",
        "orchestrator_recv_interval_ms",
        "state_machine_tick_interval_ms",
        "state_machine_consume_interval_ms",
        "same_obs_reuse_count",
        "obs_seq_gap",
        "vision_publish_to_orch_recv_ms",
        "orch_recv_to_state_consume_ms",
        "table_edge_worker_interval_ms",
        "table_edge_no_new_frame_count",
        "scheduler_publish_ms",
        "obs_out_drop_or_skip_count",
        "obs_out_skip_reason",
        "send_hz_config",
        "track_local_send_hz_config",
        "dropped_frame_count",
        "processed_frame_count",
        "latest_frame_lag_ms",
        "control_loop_age_ms",
        "edge_update_interval_ms",
        "camera_frame_seq",
        "camera_frame_ts_ms",
        "vision_process_start_ts_ms",
        "vision_process_end_ts_ms",
        "vision_publish_ts_ms",
        "obs_out_send_ts_ms",
        "orchestrator_recv_ts_ms",
        "state_machine_consume_ts_ms",
        "cmd_publish_ts_ms",
        "forward_block_reason",
        "rotate_block_reason",
        "lateral_block_reason",
        "bbox_track_exit_reason",
        "edge_handoff_block_reason",
        "dropout_hold_block_reason",
        "near_latch_block_reason",
        "edge_trusted",
        "edge_control_allowed",
        "edge_readiness_score",
        "edge_readiness_level",
        "vy_cmd_raw",
        "vy_cmd_limited",
        "vy_enabled",
        "vy_block_reason",
        "advance_condition",
        "fallback_condition",
        "table_bbox_touch_left",
        "table_bbox_touch_right",
        "table_bbox_touch_bottom",
        "yolo_bbox_touch_left",
        "yolo_bbox_touch_right",
        "yolo_bbox_touch_bottom",
    ):
        slim_summary[key] = "debug"
    slim_decision = slim_probe._apply_control_authority(MotionDecision(cmd=slim_cmd, control_summary=slim_summary), slim_obs)
    for key in (
        "camera_frame_interval_ms",
        "camera_frame_hz",
        "vision_process_interval_ms",
        "vision_publish_interval_ms",
        "obs_out_send_interval_ms",
        "obs_out_send_hz",
        "table_edge_obs_recv_interval_ms",
        "orchestrator_recv_interval_ms",
        "state_machine_tick_interval_ms",
        "state_machine_consume_interval_ms",
        "same_obs_reuse_count",
        "obs_seq_gap",
        "vision_publish_to_orch_recv_ms",
        "orch_recv_to_state_consume_ms",
        "table_edge_worker_interval_ms",
        "table_edge_no_new_frame_count",
        "scheduler_publish_ms",
        "obs_out_drop_or_skip_count",
        "obs_out_skip_reason",
        "send_hz_config",
        "track_local_send_hz_config",
        "dropped_frame_count",
        "processed_frame_count",
        "latest_frame_lag_ms",
        "control_loop_age_ms",
        "edge_update_interval_ms",
        "camera_frame_seq",
        "camera_frame_ts_ms",
        "vision_process_start_ts_ms",
        "vision_process_end_ts_ms",
        "vision_publish_ts_ms",
        "obs_out_send_ts_ms",
        "orchestrator_recv_ts_ms",
        "state_machine_consume_ts_ms",
        "cmd_publish_ts_ms",
        "forward_block_reason",
        "rotate_block_reason",
        "lateral_block_reason",
        "bbox_track_exit_reason",
        "edge_handoff_block_reason",
        "dropout_hold_block_reason",
        "near_latch_block_reason",
        "edge_trusted",
        "edge_control_allowed",
        "edge_readiness_score",
        "edge_readiness_level",
        "vy_cmd_raw",
        "vy_cmd_limited",
        "vy_enabled",
        "vy_block_reason",
        "advance_condition",
        "fallback_condition",
        "table_bbox_touch_left",
        "table_bbox_touch_right",
        "table_bbox_touch_bottom",
        "yolo_bbox_touch_left",
        "yolo_bbox_touch_right",
        "yolo_bbox_touch_bottom",
    ):
        assert key not in slim_decision.control_summary
    assert slim_decision.control_summary["effective_block_reason"] == ""
    sanitized = sanitize_control_summary(dict(slim_decision.control_summary, table_bbox_touch_right=True, vy_cmd_raw=9.0))
    assert set(sanitized) == {
        "state", "docking_action", "docking_reason", "vx_mps", "vy_mps", "wz_radps",
        "yaw_owner", "forward_owner", "lateral_owner", "effective_block_reason",
        "table_bbox_control_valid", "bbox_center_error", "fov_guard_level",
        "table_roi_depth_valid", "table_roi_depth_p10", "table_roi_depth_median",
        "table_roi_source", "table_roi_latched", "table_roi_latch_age_s", "table_roi_xyxy",
        "edge_valid", "edge_ready_for_approach", "edge_ready_for_final",
        "edge_lost_age_s", "yaw_err_rad", "near_table_latched",
        "final_depth_latched", "final_distance_servo_reason", "final_locked", "final_lock_reason",
        "depth_speed_envelope_reason", "depth_speed_envelope_vx_cap",
    }

    inv9_handoff_bbox_track = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("yolo_track_forward", desired_vx=0.012, desired_wz=0.02, yaw_owner="bbox", forward_allowed_by_behavior=True),
        {
            "control_phase": "EDGE_HANDOFF_CONFIRM",
            "edge_readiness_score": 0.0,
            "edge_readiness_enter_score": 0.65,
            "bbox_center_valid": True,
            "bbox_center_error": 0.02,
            "bbox_yaw_cmd": 0.02,
            "yolo_forward_allowed": True,
            "bbox_fov_guard_level": "none",
            "table_roi_depth_valid": True,
            "table_roi_depth_p10": 0.80,
            "cmd": {"vx_mps": 0.012},
        },
    )
    assert inv9_handoff_bbox_track.summary["docking_action"] == "BBOX_TRACK_FORWARD"
    assert inv9_handoff_bbox_track.summary["docking_action"] != "BBOX_REACQUIRE_ROTATE"
    assert inv9_handoff_bbox_track.final_vx >= 0.04
    assert inv9_handoff_bbox_track.final_vx not in {0.008, 0.010, 0.012, 0.015, 0.020}

    commit_ctx = RuntimeContext(state=State.YOLO_APPROACH)
    commit_ctx.forward_commit_until_mono = time.monotonic() + 1.0
    commit_ctx.forward_commit_reason = "bbox_track_forward"
    inv9_commit_bbox_track = arbitrate_table_docking_motion(
        commit_ctx,
        None,
        MotionIntent("yolo_track_forward", desired_vx=0.012, desired_wz=0.02, yaw_owner="bbox", forward_allowed_by_behavior=True),
        {
            "control_phase": "EDGE_HANDOFF_CONFIRM",
            "edge_readiness_score": 0.0,
            "edge_readiness_enter_score": 0.65,
            "bbox_center_valid": True,
            "bbox_center_error": 0.18,
            "bbox_yaw_cmd": 0.02,
            "bbox_track_forward_center_band": 0.12,
            "yolo_forward_allowed": True,
            "bbox_fov_guard_level": "none",
            "table_roi_depth_valid": True,
            "table_roi_depth_p10": 0.90,
            "cmd": {"vx_mps": 0.012},
        },
    )
    assert inv9_commit_bbox_track.summary["docking_action"] == "BBOX_TRACK_FORWARD"
    assert inv9_commit_bbox_track.final_vx > 0.0

    inv9_guided_gate_bbox_track = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("edge_guided_forward", desired_vx=0.020, desired_wz=0.02, yaw_owner="edge", forward_allowed_by_behavior=True),
        {
            "control_phase": "EDGE_GUIDED_APPROACH",
            "edge_handoff_complete": False,
            "edge_readiness_score": 0.0,
            "edge_readiness_enter_score": 0.65,
            "edge_trusted": False,
            "bbox_center_valid": True,
            "bbox_center_error": 0.02,
            "bbox_yaw_cmd": 0.02,
            "yolo_forward_allowed": True,
            "bbox_fov_guard_level": "none",
            "table_roi_depth_valid": True,
            "table_roi_depth_p10": 0.80,
            "cmd": {"vx_mps": 0.012},
        },
    )
    assert inv9_guided_gate_bbox_track.summary["docking_action"] == "BBOX_TRACK_FORWARD"
    assert inv9_guided_gate_bbox_track.summary["docking_action"] != "EDGE_APPROACH_FORWARD"
    assert inv9_guided_gate_bbox_track.final_vx >= 0.04
    assert inv9_guided_gate_bbox_track.final_vx not in {0.008, 0.010, 0.012, 0.015, 0.020}

    inv9_bbox_large_err = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("yolo_acquire_align", desired_vx=0.012, desired_wz=0.05, yaw_owner="bbox", rotate_allowed_by_behavior=True),
        {
            "control_phase": "BBOX_ACQUIRE",
            "bbox_center_valid": True,
            "bbox_center_error": 0.31,
            "bbox_yaw_cmd": 0.05,
            "yolo_forward_allowed": True,
            "cmd": {"vx_mps": 0.012},
        },
    )
    assert inv9_bbox_large_err.summary["docking_action"] == "BBOX_REACQUIRE_ROTATE"
    assert inv9_bbox_large_err.final_vx == 0.0 and abs(inv9_bbox_large_err.final_wz) > 0.0
    assert inv9_bbox_large_err.summary["bbox_track_exit_reason"] == "bbox_center_error_large"

    inv9_bbox_fov_hard = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("yolo_track_forward", desired_vx=0.012, desired_wz=0.05, yaw_owner="bbox", forward_allowed_by_behavior=True),
        {
            "control_phase": "BBOX_ACQUIRE",
            "bbox_center_valid": True,
            "bbox_center_error": 0.02,
            "bbox_yaw_cmd": 0.05,
            "yolo_forward_allowed": True,
            "bbox_fov_guard_level": "hard",
            "cmd": {"vx_mps": 0.012},
        },
    )
    assert inv9_bbox_fov_hard.summary["docking_action"] in {"BBOX_REACQUIRE_ROTATE", "CONTROL_RECOVERY_ROTATE"}
    assert inv9_bbox_fov_hard.final_vx == 0.0
    assert inv9_bbox_fov_hard.summary["bbox_track_exit_reason"] == "bbox_fov_guard_hard"

    inv9_handoff_fov_hard = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("yolo_track_forward", desired_vx=0.012, desired_wz=0.05, yaw_owner="bbox", forward_allowed_by_behavior=True),
        {
            "control_phase": "EDGE_HANDOFF_CONFIRM",
            "bbox_center_valid": True,
            "bbox_center_error": 0.02,
            "bbox_yaw_cmd": 0.05,
            "yolo_forward_allowed": True,
            "bbox_fov_guard_level": "hard",
            "cmd": {"vx_mps": 0.012},
        },
    )
    assert inv9_handoff_fov_hard.summary["docking_action"] in {"BBOX_REACQUIRE_ROTATE", "CONTROL_RECOVERY_ROTATE"}
    assert inv9_handoff_fov_hard.final_vx == 0.0

    inv9_bbox_final = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("yolo_track_forward", desired_vx=0.012, desired_wz=0.05, yaw_owner="bbox", forward_allowed_by_behavior=True),
        {
            "control_phase": "BBOX_ACQUIRE",
            "bbox_center_valid": True,
            "bbox_center_error": 0.02,
            "bbox_yaw_cmd": 0.05,
            "yolo_forward_allowed": True,
            "final_depth_latched": True,
            "final_locked": True,
            "cmd": {"vx_mps": 0.012},
        },
    )
    assert inv9_bbox_final.summary["docking_action"] == "FINAL_LOCKED_STOP"
    assert inv9_bbox_final.final_vx == 0.0 and inv9_bbox_final.final_vy == 0.0

    for motion_result in (inv3, inv4, inv5, inv6, inv8, inv9, inv9_bbox_track, inv9_bbox_large_err, inv9_bbox_fov_hard, inv9_bbox_final):
        for owner_key in ("yaw_owner", "forward_owner", "lateral_owner"):
            assert owner_key in motion_result.summary
            assert str(motion_result.summary[owner_key]) != ""

    readiness_handoff_obs = bbox_obs(0.50)
    readiness_handoff_obs.table_roi_depth_valid = True
    readiness_handoff_obs.table_roi_depth_p10 = 0.90
    readiness_handoff = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        readiness_handoff_obs,
        MotionIntent("yolo_acquire_align", desired_vx=0.0, desired_wz=0.03, yaw_owner="edge_candidate", rotate_allowed_by_behavior=True),
        {
            "control_phase": "EDGE_HANDOFF_CONFIRM",
            "edge_readiness_score": 0.70,
            "edge_readiness_enter_score": 0.65,
            "edge_readiness_exit_score": 0.35,
            "edge_yaw_cmd": 0.03,
            "edge_found": True,
            "edge_valid": True,
            "edge_trusted": False,
            "table_roi_depth_valid": True,
            "table_roi_depth_p10": 0.90,
        },
    )
    assert readiness_handoff.summary["docking_action"] == "EDGE_READINESS_HANDOFF"
    assert readiness_handoff.summary["yaw_owner"] == "edge_candidate"
    assert readiness_handoff.final_vx >= 0.08
    assert readiness_handoff.final_vx not in {0.008, 0.010, 0.012, 0.015, 0.020}
    assert readiness_handoff.summary["forward_owner"] == "edge_handoff"

    readiness_approach = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("edge_guided_forward", desired_vx=0.02, desired_wz=0.03, yaw_owner="edge", forward_allowed_by_behavior=True),
        {
            "control_phase": "EDGE_GUIDED_APPROACH",
            "edge_readiness_score": 0.70,
            "edge_readiness_enter_score": 0.65,
            "edge_readiness_exit_score": 0.35,
            "edge_found": True,
            "edge_valid": True,
            "edge_trusted": False,
            "usable_for_approach": True,
            "table_roi_depth_valid": True,
            "forward_commit_vx": 0.020,
        },
    )
    assert readiness_approach.summary["docking_action"] == "EDGE_APPROACH_FORWARD"
    assert readiness_approach.final_vx > 0.0
    assert readiness_approach.summary["forward_owner"] == "edge_approach"

    readiness_final = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("edge_guided_forward", desired_vx=0.02, desired_wz=0.03, yaw_owner="edge", forward_allowed_by_behavior=True),
        {
            "control_phase": "EDGE_GUIDED_APPROACH",
            "edge_readiness_score": 1.0,
            "edge_readiness_enter_score": 0.65,
            "final_depth_latched": True,
            "final_locked": True,
            "edge_found": True,
            "edge_valid": True,
        },
    )
    assert readiness_final.summary["docking_action"] == "FINAL_LOCKED_STOP"
    assert readiness_final.final_vx == 0.0 and readiness_final.final_wz == 0.0

    near_speed_envelope = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("edge_guided_forward", desired_vx=0.020, desired_wz=0.02, yaw_owner="edge", forward_allowed_by_behavior=True),
        {
            "control_phase": "EDGE_GUIDED_APPROACH",
            "near_table_latched": True,
            "final_depth_latched": False,
            "edge_readiness_score": 1.0,
            "edge_readiness_enter_score": 0.65,
            "near_slow_max_vx_mps": 0.008,
            "edge_yaw_cmd": 0.02,
        },
    )
    assert near_speed_envelope.summary["docking_action"] == "NEAR_EDGE_FORWARD"
    assert near_speed_envelope.final_vx <= 0.008
    assert near_speed_envelope.final_vy == 0.0
    assert str(near_speed_envelope.blocked_by or "") == ""
    assert str(near_speed_envelope.reason) == "near_edge_forward"
    assert near_speed_envelope.summary["effective_block_reason"] == ""

    depth_slow_envelope = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("edge_guided_forward", desired_vx=0.020, desired_wz=0.02, yaw_owner="edge", forward_allowed_by_behavior=True),
        {
            "control_phase": "EDGE_GUIDED_APPROACH",
            "edge_readiness_score": 1.0,
            "edge_readiness_enter_score": 0.65,
            "edge_found": True,
            "edge_valid": True,
            "usable_for_approach": True,
            "table_roi_depth_valid": True,
            "table_roi_depth_p10": 0.60,
            "depth_envelope_mid_vx_mps": 0.015,
        },
    )
    assert depth_slow_envelope.summary["docking_action"] == "EDGE_APPROACH_FORWARD"
    assert depth_slow_envelope.final_vx <= 0.015
    assert depth_slow_envelope.summary["depth_speed_envelope_reason"] == "depth_p10_mid"
    assert depth_slow_envelope.final_vy == 0.0

    final_semantic_envelope = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("edge_guided_forward", desired_vx=0.020, desired_wz=0.0, yaw_owner="edge", forward_allowed_by_behavior=True),
        {
            "control_phase": "EDGE_GUIDED_APPROACH",
            "final_depth_latched": True,
            "edge_readiness_score": 1.0,
            "edge_readiness_enter_score": 0.65,
        },
    )
    assert final_semantic_envelope.summary["docking_action"] == "CLOSE_RANGE_PROBE"
    assert final_semantic_envelope.summary["docking_action"] != "FINAL_LOCKED_STOP"
    assert final_semantic_envelope.summary.get("final_locked") is False
    assert abs(final_semantic_envelope.final_vx) <= 0.03

    probe_readiness = DockingProbe()
    probe_readiness.cfg.edge_readiness_min_inliers = 0
    probe_readiness.cfg.edge_readiness_rise = 0.35
    probe_readiness.cfg.edge_readiness_decay = 0.05
    probe_readiness.ctx.bbox_centered_streak = 2
    obs_readiness = bbox_obs(0.50)
    obs_readiness.edge_found = True
    obs_readiness.edge_valid = True
    obs_readiness.edge_trusted = False
    obs_readiness.usable_for_approach = True
    obs_readiness.edge_confidence = 0.80
    obs_readiness.yaw_err_rad = 0.10
    obs_readiness.table_roi_depth_valid = True
    obs_readiness.table_roi_depth_p10 = 0.70
    obs_readiness.table_roi_depth_median = 0.70
    for _ in range(3):
        phase_readiness = probe_readiness._control_phase_status(obs_readiness, depth_stop_ready=False)
    assert phase_readiness["edge_readiness_score"] >= phase_readiness["edge_readiness_enter_score"]
    assert phase_readiness["control_phase"] in {"EDGE_HANDOFF_CONFIRM", "EDGE_GUIDED_APPROACH"}

    probe_sparse_readiness = DockingProbe()
    probe_sparse_readiness.cfg.edge_readiness_rise = 0.20
    probe_sparse_readiness.ctx.bbox_centered_streak = 2
    obs_sparse = bbox_obs(0.50)
    obs_sparse.edge_found = True
    obs_sparse.edge_valid = True
    obs_sparse.edge_trusted = False
    obs_sparse.usable_for_approach = False
    obs_sparse.edge_confidence = 0.80
    obs_sparse.edge_inlier_count = 3
    obs_sparse.valid_edge_points = 4
    obs_sparse.yaw_err_rad = 0.10
    obs_sparse.table_roi_depth_valid = True
    obs_sparse.table_roi_depth_p10 = 0.80
    obs_sparse.table_roi_depth_median = 0.80
    sparse_readiness = probe_sparse_readiness._refresh_edge_readiness(
        obs_sparse,
        {},
        current_bbox=True,
        stable_bbox=True,
        depth_stop_ready=False,
    )
    assert sparse_readiness["edge_readiness_score"] > 0.0
    assert sparse_readiness["edge_readiness_reason"] == "fast_sparse_edge_candidate_healthy"

    bbox_right_vy = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("yolo_track_forward", desired_vx=0.012, desired_wz=0.02, yaw_owner="bbox", forward_allowed_by_behavior=True),
        {
            "control_phase": "BBOX_ACQUIRE",
            "bbox_center_valid": True,
            "bbox_center_error": 0.15,
            "bbox_cx_norm_control": 0.65,
            "bbox_yaw_cmd": 0.02,
            "yolo_forward_allowed": True,
            "cmd": {"vx_mps": 0.012},
            "lateral_enabled": True,
        },
    )
    assert bbox_right_vy.summary["docking_action"] == "BBOX_TRACK_FORWARD"
    assert bbox_right_vy.final_vy < 0.0
    assert bbox_right_vy.summary["lateral_owner"] == "bbox"

    touch_forward_vy = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("yolo_track_forward", desired_vx=0.014, desired_wz=0.02, yaw_owner="bbox", forward_allowed_by_behavior=True),
        {
            "control_phase": "BBOX_ACQUIRE",
            "bbox_center_valid": True,
            "bbox_center_error": 0.20,
            "bbox_cx_norm_control": 0.70,
            "table_bbox_touch_right": True,
            "bbox_yaw_cmd": 0.02,
            "yolo_forward_allowed": True,
            "cmd": {"vx_mps": 0.014},
            "table_roi_depth_valid": True,
            "table_roi_depth_p10": 1.30,
            "lateral_enabled": True,
        },
    )
    assert touch_forward_vy.summary["fov_guard_level"] == "none"
    assert touch_forward_vy.summary["docking_action"] == "BBOX_TRACK_FORWARD"
    assert touch_forward_vy.final_vx > 0.0 and touch_forward_vy.final_vy < 0.0

    extreme_center_rotate = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("yolo_track_forward", desired_vx=0.014, desired_wz=0.04, yaw_owner="bbox", forward_allowed_by_behavior=True, rotate_allowed_by_behavior=True),
        {
            "control_phase": "BBOX_ACQUIRE",
            "bbox_center_valid": True,
            "bbox_center_error": 0.65,
            "bbox_cx_norm_control": 1.15,
            "bbox_yaw_cmd": 0.04,
            "yolo_forward_allowed": True,
            "cmd": {"vx_mps": 0.014},
            "lateral_enabled": True,
        },
    )
    assert extreme_center_rotate.summary["docking_action"] == "BBOX_REACQUIRE_ROTATE"
    assert extreme_center_rotate.final_vx == 0.0

    bbox_left_vy = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("yolo_track_forward", desired_vx=0.012, desired_wz=-0.02, yaw_owner="bbox", forward_allowed_by_behavior=True),
        {
            "control_phase": "BBOX_ACQUIRE",
            "bbox_center_valid": True,
            "bbox_center_error": -0.15,
            "bbox_cx_norm_control": 0.35,
            "bbox_yaw_cmd": -0.02,
            "yolo_forward_allowed": True,
            "cmd": {"vx_mps": 0.012},
            "lateral_enabled": True,
        },
    )
    assert bbox_left_vy.summary["docking_action"] == "BBOX_TRACK_FORWARD"
    assert bbox_left_vy.final_vy > 0.0

    bbox_deadband_vy = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("yolo_track_forward", desired_vx=0.012, desired_wz=0.0, yaw_owner="bbox", forward_allowed_by_behavior=True),
        {
            "control_phase": "BBOX_ACQUIRE",
            "bbox_center_valid": True,
            "bbox_center_error": 0.02,
            "bbox_cx_norm_control": 0.52,
            "bbox_yaw_cmd": 0.0,
            "yolo_forward_allowed": True,
            "cmd": {"vx_mps": 0.012},
            "lateral_enabled": True,
        },
    )
    assert bbox_deadband_vy.summary["docking_action"] == "BBOX_TRACK_FORWARD"
    assert bbox_deadband_vy.final_vy == 0.0

    edge_bbox_vy = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("edge_guided_forward", desired_vx=0.02, desired_wz=0.02, yaw_owner="edge", forward_allowed_by_behavior=True),
        {
            "control_phase": "EDGE_GUIDED_APPROACH",
            "edge_readiness_score": 0.80,
            "edge_readiness_enter_score": 0.65,
            "edge_found": True,
            "edge_valid": True,
            "usable_for_approach": True,
            "bbox_center_valid": True,
            "bbox_center_error": 0.15,
            "bbox_cx_norm_control": 0.65,
            "lateral_enabled": True,
        },
    )
    assert edge_bbox_vy.summary["docking_action"] == "EDGE_APPROACH_FORWARD"
    assert edge_bbox_vy.summary["yaw_owner"] == "edge"
    assert edge_bbox_vy.final_wz == 0.02
    assert edge_bbox_vy.final_vy < 0.0

    slow_depth_vy = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("yolo_track_forward", desired_vx=0.012, desired_wz=0.02, yaw_owner="bbox", forward_allowed_by_behavior=True),
        {
            "control_phase": "BBOX_ACQUIRE",
            "bbox_center_valid": True,
            "bbox_center_error": 0.20,
            "bbox_cx_norm_control": 0.70,
            "bbox_yaw_cmd": 0.02,
            "yolo_forward_allowed": True,
            "cmd": {"vx_mps": 0.012},
            "table_roi_depth_valid": True,
            "table_roi_depth_p10": 0.62,
            "near_slow_max_vy_mps": 0.0025,
            "lateral_enabled": True,
        },
    )
    assert 0.0 < abs(slow_depth_vy.final_vy) <= 0.0025 + 1e-9

    stop_depth_vy = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("yolo_track_forward", desired_vx=0.012, desired_wz=0.02, yaw_owner="bbox", forward_allowed_by_behavior=True),
        {
            "control_phase": "BBOX_ACQUIRE",
            "bbox_center_valid": True,
            "bbox_center_error": 0.20,
            "bbox_cx_norm_control": 0.70,
            "bbox_yaw_cmd": 0.02,
            "yolo_forward_allowed": True,
            "cmd": {"vx_mps": 0.012},
            "table_roi_depth_valid": True,
            "table_roi_depth_p10": 0.55,
            "lateral_enabled": True,
        },
    )
    assert stop_depth_vy.final_vy == 0.0

    far_vx = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("yolo_track_forward", desired_vx=0.014, desired_wz=0.0, yaw_owner="bbox", forward_allowed_by_behavior=True),
        {
            "control_phase": "BBOX_ACQUIRE",
            "bbox_center_valid": True,
            "bbox_center_error": 0.02,
            "bbox_cx_norm_control": 0.52,
            "bbox_yaw_cmd": 0.0,
            "yolo_forward_allowed": True,
            "cmd": {"vx_mps": 0.014},
            "table_roi_depth_valid": True,
            "table_roi_depth_p10": 1.30,
            "bbox_track_forward_vx_mps": 0.014,
            "bbox_track_forward_max_vx_mps": 0.018,
            "far_bbox_track_vx_mps": 0.016,
            "lateral_enabled": True,
        },
    )
    assert 0.015 <= far_vx.final_vx <= 0.018

    no_lateral_fallback = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("yolo_track_forward", desired_vx=0.012, desired_wz=0.02, yaw_owner="bbox", forward_allowed_by_behavior=True),
        {
            "control_phase": "BBOX_ACQUIRE",
            "bbox_center_valid": True,
            "bbox_center_error": 0.02,
            "bbox_yaw_cmd": 0.02,
            "yolo_forward_allowed": True,
            "cmd": {"vx_mps": 0.012},
            "dist_err_m": 0.30,
        },
    )
    assert no_lateral_fallback.summary["lateral_err_norm"] is None
    assert no_lateral_fallback.summary["lateral_source"] == ""
    assert no_lateral_fallback.final_vy == 0.0

    short_dropout_vy = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("edge_guided_forward", desired_vx=0.010, desired_wz=0.04, yaw_owner="edge", forward_allowed_by_behavior=True),
        {
            "stale_hold_policy": "approach_commit_short_dropout",
            "perception_dropout_hold_active": True,
            "last_good_edge_yaw_age_ms": 300.0,
            "bbox_center_valid": True,
            "bbox_center_error": 0.15,
            "bbox_cx_norm_control": 0.65,
            "table_roi_depth_valid": True,
            "table_roi_depth_p10": 0.90,
            "lateral_enabled": True,
        },
    )
    assert short_dropout_vy.summary["docking_action"] == "PERCEPTION_DROPOUT_HOLD"
    assert abs(short_dropout_vy.final_wz) <= 1e-9
    assert short_dropout_vy.final_vy < 0.0
    assert short_dropout_vy.summary["yaw_owner"] == "hold"

    long_dropout_bbox = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("edge_guided_forward", desired_vx=0.010, desired_wz=0.04, yaw_owner="edge", forward_allowed_by_behavior=True),
        {
            "stale_hold_policy": "approach_commit_short_dropout",
            "perception_dropout_hold_active": True,
            "last_good_edge_yaw_age_ms": 1500.0,
            "control_phase": "BBOX_ACQUIRE",
            "bbox_center_valid": True,
            "bbox_center_error": 0.02,
            "bbox_cx_norm_control": 0.52,
            "bbox_yaw_cmd": 0.0,
            "yolo_forward_allowed": True,
            "cmd": {"vx_mps": 0.012},
            "lateral_enabled": True,
        },
    )
    assert long_dropout_bbox.summary["docking_action"] in {"BBOX_TRACK_FORWARD", "BBOX_REACQUIRE_ROTATE"}

    final_lateral_block = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("final_hold", desired_vx=0.0, desired_wz=0.0),
        {
            "control_phase": "DEPTH_FINAL_STOP",
            "final_depth_latched": True,
            "final_locked": True,
            "desired_vy": 0.008,
            "lateral_enabled": True,
            "lateral_err_norm": 0.20,
        },
    )
    assert final_lateral_block.summary["docking_action"] == "FINAL_LOCKED_STOP"
    assert final_lateral_block.final_vy == 0.0

    inv10 = arbitrate_table_docking_motion(
        RuntimeContext(state=State.YOLO_APPROACH),
        None,
        MotionIntent("edge_guided_forward", desired_vx=0.02, desired_wz=0.0, forward_allowed_by_behavior=True),
        {"emergency_stop_active": True},
    )
    assert inv10.summary["docking_action"] == "EMERGENCY_STOP"
    assert inv10.stop_class == "emergency"

    # Dwell fallback tests
    def touch_obs(center: float, touch: bool = True) -> TableEdgeObs:
        obs = bbox_obs(center)
        obs.table_bbox_touch_right = touch
        obs.edge_found = True
        obs.edge_trusted = True
        obs.yaw_err_rad = 0.10
        obs.dist_err_m = 0.50
        return obs

    # 1. bbox center 0.75 + touch_right + acquire dwell 未到 -> BBOX_ACQUIRE
    p_fallback_1 = DockingProbe()
    p_fallback_1.ctx.control_phase = "BBOX_ACQUIRE"
    p_fallback_1.ctx.control_phase_since_mono = monotonic_ts() - 0.5
    p_fallback_1.ctx.bbox_valid_streak = 3
    obs_fallback_1 = touch_obs(0.75)
    dec_fallback_1 = authority_decision(p_fallback_1, obs_fallback_1, raw_wz=0.10)
    assert "control_phase" not in dec_fallback_1.control_summary
    assert dec_fallback_1.control_summary["docking_action"] in {"BBOX_REACQUIRE_ROTATE", "BBOX_TRACK_FORWARD"}

    # 2. bbox center 0.75 + touch_right + acquire dwell 到达 + edge 可用 -> EDGE_HANDOFF_CONFIRM
    p_fallback_2 = DockingProbe()
    p_fallback_2.ctx.control_phase = "BBOX_ACQUIRE"
    p_fallback_2.ctx.control_phase_since_mono = monotonic_ts() - 2.0
    p_fallback_2.ctx.bbox_valid_streak = 3
    obs_fallback_2 = touch_obs(0.75)
    dec_fallback_2 = authority_decision(p_fallback_2, obs_fallback_2, raw_wz=0.10)
    assert "control_phase" not in dec_fallback_2.control_summary
    assert dec_fallback_2.control_summary["docking_action"] in {"EDGE_READINESS_HANDOFF", "BBOX_TRACK_FORWARD", "BBOX_REACQUIRE_ROTATE"}

    # 3. severe touch -> 仍 BBOX_ACQUIRE
    p_fallback_3 = DockingProbe()
    p_fallback_3.ctx.control_phase = "BBOX_ACQUIRE"
    p_fallback_3.ctx.control_phase_since_mono = monotonic_ts() - 2.0
    p_fallback_3.ctx.bbox_valid_streak = 3
    obs_fallback_3 = touch_obs(0.90)  # error 0.40 > 0.35
    dec_fallback_3 = authority_decision(p_fallback_3, obs_fallback_3, raw_wz=0.10)
    assert "control_phase" not in dec_fallback_3.control_summary
    assert dec_fallback_3.control_summary["docking_action"] == "BBOX_REACQUIRE_ROTATE"

    # Queue 2: Final stop static tests with omega disabled
    # Case A: final_depth_latched=True + yaw_err=0.20 (which is > deadband 0.12)
    # Action must stay in final hold with wz=0; final no longer chases edge yaw.
    ctx_q2_a = RuntimeContext(state=State.FINAL_SLOW_STOP)
    ctx_q2_a.final_depth_latched = True
    ctx_q2_a.final_yaw_align_start_mono = 1.0
    inv_q2_a = arbitrate_table_docking_motion(
        ctx_q2_a,
        None,
        MotionIntent("final_align", desired_vx=0.0, desired_wz=0.0, yaw_owner="edge"),
        {
            "control_phase": "DEPTH_FINAL_STOP",
            "final_depth_latched": True,
            "final_yaw_align_active": True,
            "edge_yaw": 0.20,
            "yaw_err_rad": 0.20,
            "final_yaw_deadband_rad": 0.12,
            "final_lock_yaw_rad": 0.12,
            "edge_yaw_cmd_for_final_align": 0.10,
            "last_good_edge_yaw_age_ms": 10.0,
            "edge_usable": True,
        },
    )

    assert inv_q2_a.summary["docking_action"] == "CLOSE_RANGE_PROBE"
    assert inv_q2_a.summary["docking_action"] != "FINAL_LOCKED_STOP"
    assert inv_q2_a.summary["docking_stage"] == "FINAL_DISTANCE_HOLD"
    assert inv_q2_a.final_vx == 0.0 and inv_q2_a.final_vy == 0.0 and inv_q2_a.final_wz == 0.0
    assert inv_q2_a.summary["yaw_owner"] == "none"
    assert inv_q2_a.summary["final_locked"] is False

    # Case B: final_depth_latched=True + yaw_err=0.05 consecutively stable for 6 frames (final_yaw_stable_frames=6)
    probe_q2_b = DockingProbe()
    probe_q2_b.ctx.state = State.FINAL_SLOW_STOP
    probe_q2_b.ctx.final_depth_latched = True
    probe_q2_b.ctx.final_yaw_align_start_mono = 1.0
    # Simulate 6 ticks with small yaw error
    for _ in range(6):
        obs_b = TableEdgeObs.from_dict({
            "ts": time.time(), "table_found": True, "edge_found": True, "edge_valid": True, "edge_trusted": True,
            "yaw_err_rad": 0.05, "table_roi_depth_valid": True, "table_roi_depth_p10": 0.30,
        })
        summary_b = {
            "control_phase": "DEPTH_FINAL_STOP",
            "final_depth_latched": True,
            "yaw_err_rad": 0.05,
            "edge_yaw_cmd": 0.0,
            "last_good_edge_yaw_cmd": 0.0,
            "last_good_edge_yaw_age_ms": 10.0,
        }
        probe_q2_b._refresh_near_final_latches(obs_b, {"depth_roi_stop_ready": True}, summary_b)
    assert probe_q2_b.ctx.final_locked is True
    # Now run arbiter to assert
    inv_q2_b = arbitrate_table_docking_motion(
        probe_q2_b.ctx,
        None,
        MotionIntent("final_hold", desired_vx=0.0, desired_wz=0.0),
        {
            "control_phase": "DEPTH_FINAL_STOP",
            "final_depth_latched": True,
            "final_locked": True,
            "yaw_err_rad": 0.05,
            "final_yaw_deadband_rad": 0.12,
        },
    )
    assert inv_q2_b.summary["docking_action"] == "FINAL_LOCKED_STOP"
    assert inv_q2_b.summary["docking_stage"] == "FINAL_LOCKED"
    assert inv_q2_b.final_vx == 0.0 and inv_q2_b.final_wz == 0.0

    # Case C: Already FINAL_LOCKED, yaw err exceeds realignment threshold 0.18 for 3 frames -> realign
    for _ in range(3):
        obs_c = TableEdgeObs.from_dict({
            "ts": time.time(), "table_found": True, "edge_found": True, "edge_valid": True, "edge_trusted": True,
            "yaw_err_rad": 0.20, "table_roi_depth_valid": True, "table_roi_depth_p10": 0.30,
        })
        summary_c = {
            "control_phase": "DEPTH_FINAL_STOP",
            "final_depth_latched": True,
            "yaw_err_rad": 0.20,
            "edge_yaw_cmd": 0.12,
            "last_good_edge_yaw_cmd": 0.12,
            "last_good_edge_yaw_age_ms": 10.0,
        }
        probe_q2_b._refresh_near_final_latches(obs_c, {"depth_roi_stop_ready": True}, summary_c)
    assert probe_q2_b.ctx.final_locked is False
    assert probe_q2_b.ctx.final_yaw_align_active is True

    # Case D: final depth + edge yaw stale (age > hold_timeout) still holds
    # final mode with omega disabled.
    ctx_q2_d = RuntimeContext(state=State.FINAL_SLOW_STOP)
    ctx_q2_d.final_depth_latched = True
    inv_q2_d = arbitrate_table_docking_motion(
        ctx_q2_d,
        None,
        MotionIntent("final_align", desired_vx=0.0, desired_wz=0.0),
        {
            "control_phase": "DEPTH_FINAL_STOP",
            "final_depth_latched": True,
            "final_yaw_align_active": True,
            "yaw_err_rad": 0.25,
            "last_good_edge_yaw_age_ms": 5000.0, # stale!
        },
    )
    assert inv_q2_d.summary["docking_action"] == "CLOSE_RANGE_PROBE"
    assert inv_q2_d.summary["docking_action"] != "FINAL_LOCKED_STOP"
    assert inv_q2_d.summary["docking_stage"] == "FINAL_DISTANCE_HOLD"
    assert inv_q2_d.final_vx == 0.0 and inv_q2_d.final_wz == 0.0
    assert inv_q2_d.summary["yaw_owner"] == "none"

    # Queue 3: Full workflow smoke test (SEARCH -> BBOX -> EDGE -> ROI_STOP -> FINAL_LOCKED)
    probe_smoke = DockingProbe()
    probe_smoke.ctx.state = State.SEARCH_TABLE
    probe_smoke.ctx.table_lost_frames = 0

    # Step 1: SEARCH_TABLE with no observations -> Rotational search
    obs_s1 = bbox_obs(0.5, found=False)
    dec_s1 = edge_guided_decision(probe_smoke, obs_s1)
    assert probe_smoke.ctx.state == State.SEARCH_TABLE
    assert dec_s1.cmd.vx_mps == 0.0 and abs(dec_s1.cmd.wz_radps) > 0.0

    # Step 2: BBOX found, transition to YOLO_APPROACH / APPROACH
    probe_smoke.ctx.state = State.YOLO_APPROACH
    obs_s2 = TableEdgeObs.from_dict({
        "ts": now_ts(), "table_found": True, "edge_found": False, "edge_valid": False, "edge_trusted": False,
        "table_bbox_control_valid": True, "yolo_table_control_valid": True, "yolo_bbox_center_x_norm": 0.50,
        "table_bbox_area_ratio": 0.02, "table_roi_depth_valid": False,
    })
    dec_s2 = edge_guided_decision(probe_smoke, obs_s2)
    assert probe_smoke.ctx.state == State.YOLO_APPROACH

    # Step 3: Edge found at near distance -> near_table_latched becomes True
    probe_smoke.ctx.state = State.EDGE_ADJUST
    obs_s3 = TableEdgeObs.from_dict({
        "ts": now_ts(), "table_found": True, "edge_found": True, "edge_valid": True, "edge_trusted": True,
        "table_bbox_control_valid": True, "yaw_err_rad": 0.15, "dist_err_m": 0.40, "table_roi_depth_valid": True, "table_roi_depth_p10": 0.40,
    })
    dec_s3 = edge_guided_decision(probe_smoke, obs_s3)
    assert probe_smoke.ctx.near_table_latched is True

    # Step 4: Final ROI stop at close p10 -> final_locked becomes True with no omega.
    probe_smoke.ctx.state = State.FINAL_SLOW_STOP
    for _ in range(3):
        obs_s4 = TableEdgeObs.from_dict({
            "ts": now_ts(), "table_found": True, "edge_found": True, "edge_valid": True, "edge_trusted": True,
            "table_bbox_control_valid": True, "yaw_err_rad": 0.15, "target_dist_m": 0.30, "dist_err_m": 0.0, "table_roi_depth_valid": True, "table_roi_depth_p10": 0.20,
        })
        dec_s4 = edge_guided_decision(probe_smoke, obs_s4)
    assert probe_smoke.ctx.final_depth_latched is True
    assert probe_smoke.ctx.final_locked is True
    assert dec_s4.cmd.vx_mps == 0.0 and dec_s4.cmd.vy_mps == 0.0 and dec_s4.cmd.wz_radps == 0.0

    # Step 5: Stable small yaw remains locked; final does not chase yaw.
    probe_smoke.ctx.final_yaw_align_start_mono = monotonic_ts() - 2.0
    for _ in range(6):
        obs_s5 = TableEdgeObs.from_dict({
            "ts": now_ts(), "table_found": True, "edge_found": True, "edge_valid": True, "edge_trusted": True,
            "table_bbox_control_valid": True, "yaw_err_rad": 0.05, "target_dist_m": 0.30, "dist_err_m": 0.0, "table_roi_depth_valid": True, "table_roi_depth_p10": 0.20,
        })
        dec_s5 = edge_guided_decision(probe_smoke, obs_s5)
    assert probe_smoke.ctx.final_locked is True

    # Bugfix 1 synthetic test: missing near_slow_max_vx_mps configuration
    class MinimalOldConfig:
        near_slow_depth_m = 0.40
        near_slow_max_wz_radps = 0.04
        table_obs_max_age_s = 1.0
        # near_slow_max_vx_mps is missing!
        # near_slow_max_vy_mps is missing!

    from orchestrator.orchestrator_service.runtime.safety.base_motion_safety import apply_base_motion_safety

    mock_cfg = MinimalOldConfig()
    mock_cmd = CmdVel(ts=now_ts(), mode="YOLO_APPROACH", vx_mps=0.03, vy_mps=0.0, wz_radps=0.0)
    mock_decision = MotionDecision(cmd=mock_cmd, control_summary={"allow_forward": True, "allow_rotate": True, "allow_lateral": True})
    mock_obs = TableEdgeObs.from_dict({
        "ts": now_ts(), "table_roi_depth_valid": True, "depth_p10": 0.30, "table_roi_depth_p10": 0.30
    })
    mock_ctx = SimpleNamespace(
        state=State.YOLO_APPROACH,
        last_table_obs=mock_obs,
        task_start_wall_ts=0,
        active_session_id=""
    )

    # This should not raise AttributeError and should successfully enforce limits using fallbacks
    apply_base_motion_safety(mock_decision, ctx=mock_ctx, cfg=mock_cfg)
    assert mock_decision.cmd.vx_mps <= 0.020
    assert mock_decision.cmd.vy_mps == 0.0

    # Tests B & C: arbitrate_table_docking_motion checks
    intent = MotionIntent(
        intent_type="edge_guided_forward",
        desired_vx=0.02,
        desired_wz=0.05,
        forward_allowed_by_behavior=True,
        rotate_allowed_by_behavior=True,
    )

    # Test B: final yaw request is ignored in final mode; omega remains zero.
    obs_b = TableEdgeObs.from_dict({
        "ts": now_ts(), "table_found": True, "edge_found": True, "edge_valid": True, "edge_trusted": True,
        "yaw_err_rad": 0.15, "dist_err_m": 0.20, "table_roi_depth_valid": True, "table_roi_depth_p10": 0.20,
    })
    summary_b = {
        "state": "FINAL_SLOW_STOP",
        "control_phase": "DEPTH_FINAL_STOP",
        "final_depth_latched": True,
        "final_yaw_align_active": True,
        "edge_yaw_cmd": 0.05,
        "edge_yaw_cmd_for_final_align": 0.05,
        "near_stage_yaw_source": "edge",
        "stale_level": "fresh",
        "current_obs_healthy": True,
        "last_good_obs_healthy": True,
        "last_good_obs_age_ms": 0.0,
    }
    test_ctx = RuntimeContext(state=State.YOLO_APPROACH)
    test_ctx.cfg = probe_smoke.cfg

    res_b = arbitrate_table_docking_motion(test_ctx, obs_b, intent, summary_b)
    assert res_b.summary["docking_action"] == "FINAL_SLOW_PROBE" or res_b.summary["docking_action"] == DockingAction.FINAL_SLOW_PROBE
    assert res_b.summary["docking_stage"] in {"FINAL_LOCKED", "FINAL_DISTANCE_HOLD", DockingStage.FINAL_LOCKED, DockingStage.FINAL_DISTANCE_HOLD}
    assert res_b.final_vy == 0.0
    assert res_b.final_wz == 0.0

    # Test C: dead stale with close-range p10 stays in final hold instead of
    # recovery/search rotate.
    summary_c = {
        "state": "YOLO_APPROACH",
        "control_phase": "EDGE_GUIDED_APPROACH",
        "stale_level": "dead",
        "current_obs_healthy": False,
        "last_good_obs_healthy": False,
        "last_good_obs_age_ms": 999999.0,
    }
    res_c = arbitrate_table_docking_motion(test_ctx, obs_b, intent, summary_c)
    assert res_c.summary["docking_action"] == "FINAL_SLOW_PROBE"
    assert res_c.summary["close_range_latched"]
    assert res_c.final_vx == 0.0
    assert res_c.final_vy == 0.0
    assert res_c.final_wz == 0.0

    probe_done = DockingProbe()
    probe_done.cfg.stop_after_table_docking = True
    probe_done.ctx.state = State.AT_TABLE_EDGE
    probe_done.ctx.final_locked = True
    probe_done.ctx.final_lock_reason = "final_depth_only_lock"
    probe_done.ctx.last_table_obs = TableEdgeObs.from_dict({
        "ts": now_ts(),
        "table_found": True,
        "edge_found": True,
        "edge_valid": True,
        "yaw_err_rad": 0.04,
        "table_roi_depth_valid": True,
        "table_roi_depth_p10": 0.24,
    })
    done_stdout = io.StringIO()
    with contextlib.redirect_stdout(done_stdout):
        done_decision = probe_done._tick_at_table_edge_impl()
    assert "[DOCKING_DONE]" in done_stdout.getvalue()
    assert probe_done.ctx.state == State.AT_TABLE_EDGE
    assert probe_done.ctx.state != State.SEARCH_TARGET_INIT
    assert done_decision.cmd.vx_mps == 0.0 and done_decision.cmd.wz_radps == 0.0

    print("docking static verification: PASS")


if __name__ == "__main__":
    main()
