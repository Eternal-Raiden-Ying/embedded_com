#!/usr/bin/env python3
"""Pure-Python synthetic checks for table docking authority and ROI depth."""
from __future__ import annotations

from types import SimpleNamespace
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "VISTA"))

import numpy as np

from orchestrator.orchestrator_service.runtime.control_authority import ControlAuthority, decide_table_control_authority
from orchestrator.orchestrator_service.runtime.perception_semantics import TablePerceptionSemantics
from orchestrator.orchestrator_service.runtime.states.table_docking import TableDockingMixin
from orchestrator.orchestrator_service.runtime.common import monotonic_ts
from orchestrator.orchestrator_service.runtime.context import RuntimeContext, State
from orchestrator.orchestrator_service.runtime.controller import MotionDecision
from orchestrator.orchestrator_service.config.schema import CarMotionConfig, ControlThresholds
from orchestrator.orchestrator_service.control.motion_controller import MotionController
from orchestrator.orchestrator_service.ipc.protocol import TableEdgeObs, now_ts
from orchestrator.orchestrator_service.bridge.uart_bridge import UartBridge
from vision_module.backend.table_roi_depth import table_roi_depth_statistics
from vision_module.app.stages.search.table_edge_obs_builder import merge_table_bbox_from_local_perception


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
    vista_obs = TableEdgeObs.from_dict({
        "ts": 1.0, "table_found": True, "edge_found": False,
        "yolo_table_visible": True, "yolo_table_fresh": True, "yolo_table_control_valid": True,
        "yolo_bbox_center_x_norm": 0.70, "table_bbox_touch_right": True,
    })
    assert vista_obs.table_bbox_current_found and vista_obs.table_bbox_control_valid
    assert vista_obs.yolo_bbox_center_x_norm == 0.70 and vista_obs.table_bbox_touch_right
    probe_geom = TableDockingMixin._bbox_control_geometry(SimpleNamespace(), vista_obs)
    assert probe_geom["bbox_center_valid"] and abs(probe_geom["bbox_center_error_control"] - 0.20) < 1e-6
    xyxy_obs = TableEdgeObs.from_dict({"ts": 1.0, "table_found": True, "edge_found": False,
        "table_bbox_xyxy": [320, 20, 640, 200], "rgb_shape": [480, 640]})
    xyxy_geom = TableDockingMixin._bbox_control_geometry(SimpleNamespace(), xyxy_obs)
    assert xyxy_geom["bbox_center_valid"] and abs(xyxy_geom["bbox_cx_norm_control"] - 0.75) < 1e-6

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
    # Edge/depth facts without a current bbox must remain a rotate-only search.
    a = auth("YOLO_APPROACH", bbox=False, edge=True, depth_stop=True)
    assert (a.control_source, a.allow_forward, a.allow_rotate) == ("local_rotate_search", False, True)
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

    class DockingProbe(TableDockingMixin):
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
        })

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

    # BBOX_ACQUIRE owns final yaw. Right side is positive/right turn even when
    # the raw/search command asks for the opposite turn.
    probe = DockingProbe()
    right = authority_decision(probe, bbox_obs(0.70), raw_wz=-0.10)
    assert right.control_summary["control_phase"] == "BBOX_ACQUIRE"
    assert right.control_summary["bbox_yaw_cmd"] > 0.0
    assert right.cmd.wz_radps > 0.0 and right.cmd.vx_mps == 0.0
    assert right.control_summary["bbox_yaw_owner_enforced"]

    probe = DockingProbe()
    left = authority_decision(probe, bbox_obs(0.30), raw_wz=0.10)
    assert left.control_summary["bbox_yaw_cmd"] < 0.0
    assert left.cmd.wz_radps < 0.0 and left.cmd.vx_mps == 0.0

    probe = DockingProbe()
    soft = authority_decision(probe, bbox_obs(0.70, soft_stale=True), raw_wz=-0.10)
    assert soft.control_summary["stale_level"] == "soft_stale"
    assert soft.control_summary["control_phase"] == "BBOX_ACQUIRE"
    assert soft.cmd.wz_radps == soft.control_summary["bbox_yaw_cmd"] > 0.0
    assert soft.cmd.vx_mps == 0.0

    edge_obs = bbox_obs(0.50)
    edge_obs.edge_found = True
    edge_obs.edge_trusted = True
    edge_obs.yaw_err_rad = 0.10
    probe = DockingProbe()
    committed = edge_guided_decision(probe, edge_obs)
    assert committed.cmd.vx_mps == 0.020
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
    assert coast.cmd.vx_mps == 0.020
    assert coast.control_summary["forward_coast_active"]
    assert coast.control_summary["control_phase"] == "EDGE_GUIDED_APPROACH"

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

    # The watchdog revives a committed, safe double-zero command even when no
    # current edge is usable enough for the normal coast branch.
    probe = DockingProbe()
    probe.force_edge_guided = True
    probe.ctx.approach_commit_active = True
    probe.ctx.edge_conf_score = 0.50
    probe.ctx.zero_cmd_started_mono = monotonic_ts() - 0.9
    watchdog_obs = bbox_obs(0.50)
    watchdog = authority_decision(probe, watchdog_obs, raw_wz=0.0)
    assert watchdog.cmd.vx_mps == 0.020
    assert watchdog.control_summary["zero_escape_reason"] == "forward_coast"

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

    uart_probe = UartBridge(port="COM_DRY_RUN", baudrate=115200, timeout_s=0.1, dry_run=True)
    assert uart_probe._writer_discard_reason({"line": "V 0.020 0.000 0.010", "tx_meta": {}}) == ""
    assert uart_probe._writer_discard_reason({"line": "MODE SEARCH", "tx_meta": {}}) == "non_velocity_line"

    probe = DockingProbe()
    edge_obs.edge_trusted = True
    edge_obs.yaw_err_rad = 0.60
    yaw_blocked = edge_guided_decision(probe, edge_obs)
    assert yaw_blocked.cmd.vx_mps == 0.0
    assert yaw_blocked.control_summary["forward_block_reason"] == "edge_yaw_too_large"

    probe = DockingProbe()
    edge_obs.yaw_err_rad = 0.10
    probe.ctx.bbox_fov_violation_streak = 3
    fov_blocked = edge_guided_decision(probe, edge_obs)
    assert fov_blocked.cmd.vx_mps == 0.0
    assert fov_blocked.control_summary["forward_block_reason"] == "bbox_fov_guard"

    probe = DockingProbe()
    probe.force_depth_stop = True
    depth_stopped = authority_decision(probe, bbox_obs(0.50), raw_wz=0.03)
    assert depth_stopped.control_summary["control_phase"] == "DEPTH_FINAL_STOP"
    assert depth_stopped.cmd.vx_mps == 0.0

    probe = DockingProbe()
    probe.ctx.bbox_valid_streak = 3
    probe.ctx.edge_handoff_complete = True
    lost = bbox_obs(0.5, found=False)
    held = probe._bbox_lost_hold_or_search(lost, "YOLO_APPROACH")
    assert held.control_summary["bbox_lost_hold_active"]
    assert held.cmd.wz_radps != 0.0
    assert held.cmd.vx_mps == 0.0
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

    class ProgressProbe(TableDockingMixin):
        def __init__(self):
            self.cfg = SimpleNamespace(table_target_dist_m=0.5)  # deliberately lacks progress_window_ms
            self.ctx = SimpleNamespace(min_dist_seen=999.0, dist_progress_last_refreshed_mono=0.0, dist_missing_started_mono=0.0)
        def _log(self, *_args):
            pass

    probe = ProgressProbe()
    assert not probe._check_approach_progress(None)
    assert probe.ctx.dist_missing_started_mono > 0.0
    probe.ctx.dist_missing_started_mono = monotonic_ts() - 5.1
    assert probe._check_approach_progress(None)

    # 1. BBOX_ACQUIRE phase, vx=0, wz!=0. Ensure no-progress is NOT triggered even after 5s
    p_acquire = DockingProbe()
    p_acquire.ctx.control_phase = "BBOX_ACQUIRE"
    obs_acq = bbox_obs(0.70)
    obs_acq.dist_err_m = 0.5
    obs_acq.target_dist_m = 0.5
    # Call authority_decision multiple times, or manually simulate time elapsed
    dec = authority_decision(p_acquire, obs_acq, raw_wz=0.10)
    assert dec.cmd.vx_mps == 0.0
    assert dec.cmd.wz_radps > 0.0
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

    # 3. EDGE_GUIDED_APPROACH phase, vx=0.02. Ensure no-progress IS triggered when distance is constant
    p_approach = DockingProbe()
    p_approach.force_edge_guided = True
    p_approach.ctx.edge_handoff_complete = True
    p_approach.ctx.control_phase = "EDGE_GUIDED_APPROACH"
    obs_app = bbox_obs(0.50)
    obs_app.edge_found = True
    obs_app.edge_trusted = True
    obs_app.dist_err_m = 0.5
    obs_app.target_dist_m = 0.5
    
    # First call resets/initializes min_dist_seen
    dec = edge_guided_decision(p_approach, obs_app)
    assert dec.cmd.vx_mps == 0.02
    assert p_approach.ctx.state != State.NO_PROGRESS_RECOVERY
    
    # Set the last refreshed timestamp to 6 seconds ago to simulate no progress
    p_approach.ctx.dist_progress_last_refreshed_mono = monotonic_ts() - 6.0
    
    # Second call should trigger no-progress recovery
    dec = edge_guided_decision(p_approach, obs_app)
    assert p_approach.ctx.state == State.NO_PROGRESS_RECOVERY

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
    assert dec_fallback_1.control_summary["control_phase"] == "BBOX_ACQUIRE"

    # 2. bbox center 0.75 + touch_right + acquire dwell 到达 + edge 可用 -> EDGE_HANDOFF_CONFIRM
    p_fallback_2 = DockingProbe()
    p_fallback_2.ctx.control_phase = "BBOX_ACQUIRE"
    p_fallback_2.ctx.control_phase_since_mono = monotonic_ts() - 2.0
    p_fallback_2.ctx.bbox_valid_streak = 3
    obs_fallback_2 = touch_obs(0.75)
    dec_fallback_2 = authority_decision(p_fallback_2, obs_fallback_2, raw_wz=0.10)
    assert dec_fallback_2.control_summary["control_phase"] == "EDGE_HANDOFF_CONFIRM"

    # 3. severe touch -> 仍 BBOX_ACQUIRE
    p_fallback_3 = DockingProbe()
    p_fallback_3.ctx.control_phase = "BBOX_ACQUIRE"
    p_fallback_3.ctx.control_phase_since_mono = monotonic_ts() - 2.0
    p_fallback_3.ctx.bbox_valid_streak = 3
    obs_fallback_3 = touch_obs(0.90)  # error 0.40 > 0.35
    dec_fallback_3 = authority_decision(p_fallback_3, obs_fallback_3, raw_wz=0.10)
    assert dec_fallback_3.control_summary["control_phase"] == "BBOX_ACQUIRE"

    print("docking static verification: PASS")


if __name__ == "__main__":
    main()
