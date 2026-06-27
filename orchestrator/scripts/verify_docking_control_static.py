#!/usr/bin/env python3
"""Pure-Python synthetic checks for table docking authority and ROI depth."""
from __future__ import annotations

from types import SimpleNamespace
import os
import sys
import time

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
from orchestrator.orchestrator_service.runtime.service import OrchestratorService
from orchestrator.orchestrator_service.config.schema import CarMotionConfig, ControlThresholds
from orchestrator.orchestrator_service.control.motion_controller import MotionController
from orchestrator.orchestrator_service.ipc.protocol import CmdVel, TableEdgeObs, now_ts
from orchestrator.orchestrator_service.bridge.uart_bridge import UartBridge
from vision_module.backend.table_roi_depth import table_roi_depth_statistics
from vision_module.app.stages.search.table_edge_obs_builder import merge_table_bbox_from_local_perception
from orchestrator.orchestrator_service.runtime.docking_model import DockingAction, DockingStage


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

    # A short hard-stale observation dropout after an established commit holds
    # the safe approach command instead of immediately stopping.
    probe = DockingProbe()
    dropout_good = bbox_obs(0.50)
    dropout_good.edge_found = True
    dropout_good.edge_trusted = True
    dropout_good.yaw_err_rad = 0.10
    edge_guided_decision(probe, dropout_good)
    edge_guided_decision(probe, dropout_good)
    assert probe.ctx.approach_commit_active and probe.ctx.last_good_table_obs_mono > 0.0
    dropout_obs = bbox_obs(0.50)
    dropout_obs.edge_found = True
    dropout_obs.edge_trusted = True
    dropout_obs.yaw_err_rad = 0.10
    dropout_obs.ts = now_ts() - 0.60
    dropout_obs.frame_capture_ts = dropout_obs.ts
    dropout = edge_guided_decision(probe, dropout_obs)
    assert dropout.cmd.vx_mps == 0.020
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
    assert uart_probe._writer_discard_reason({"line": "MODE SEARCH", "tx_meta": {}}) == "non_velocity_line"
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
    assert yaw_blocked.control_summary["forward_block_reason"] == "edge_yaw_too_large"

    # A large, bottom-touching bbox with a mild side offset is soft framing:
    # it must permit the first EDGE_GUIDED_APPROACH forward commit.
    probe = DockingProbe()
    soft_fov_obs = bbox_obs(0.36)
    soft_fov_obs.edge_found = True
    soft_fov_obs.edge_trusted = True
    soft_fov_obs.yaw_err_rad = 0.10
    soft_fov_obs.table_bbox_touch_left = True
    soft_fov_obs.table_bbox_touch_bottom = True
    soft_fov_obs.table_bbox_area_ratio = 0.38
    soft_fov = edge_guided_decision(probe, soft_fov_obs)
    assert soft_fov.control_summary["bbox_fov_guard_level"] == "soft"
    assert soft_fov.cmd.vx_mps == 0.020
    assert soft_fov.control_summary["approach_commit_active"]
    assert soft_fov.control_summary["motion_class"] == "normal"
    assert soft_fov.control_summary["stop_class"] == "none"

    # Near-table latch: depth ROI is near but not yet final, so edge/near owns
    # yaw and YOLO is downgraded from primary control.
    probe = DockingProbe()
    near_obs = near_depth_obs(0.50, 0.58, yaw=0.08)
    edge_guided_decision(probe, near_obs)
    near_latched = edge_guided_decision(probe, near_obs)
    assert near_latched.control_summary["near_table_latched"]
    assert not near_latched.control_summary["yolo_control_enabled"]
    assert near_latched.control_summary["near_stage_yaw_source"] in {"edge", "last_good_edge"}
    assert near_latched.control_summary["yaw_owner"] not in {"bbox", "yolo"}

    # Final depth latch + large yaw: forward is permanently blocked but yaw
    # alignment may continue in place.
    probe = DockingProbe()
    probe.ctx.near_table_latched = True
    probe.ctx.final_depth_latched = True
    probe.ctx.final_depth_latch_reason = "static_final_depth"
    final_yaw_deadband = abs(float(getattr(probe.cfg, "final_lock_yaw_tol_rad", getattr(probe.cfg, "table_yaw_tol_rad", 0.08)) or 0.08))
    large_yaw = max(final_yaw_deadband + 0.05, 0.30)
    final_yaw_obs = near_depth_obs(0.50, 0.50, yaw=large_yaw)
    final_yaw = edge_guided_decision(probe, final_yaw_obs)
    assert final_yaw.cmd.vx_mps == 0.0
    assert abs(final_yaw.cmd.wz_radps) > 0.0
    assert final_yaw.control_summary["final_depth_latched"]
    assert final_yaw.control_summary["final_yaw_align_active"]
    assert not final_yaw.control_summary["final_locked"]
    assert final_yaw.control_summary["arbitration_reason"] == "final_yaw_align"
    assert abs(final_yaw.control_summary["final_yaw_align_yaw_cmd"]) > 0.0

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
    assert final_align_applied.cmd.vx_mps == 0.0
    assert final_align_applied.cmd.vy_mps == 0.0
    assert abs(final_align_applied.cmd.wz_radps + 0.15) < 1e-6
    assert final_align_applied.control_summary["final_cmd_source"] == "arbiter_final_yaw_align"
    assert final_align_applied.control_summary["rotate_allowed"]
    assert final_align_applied.control_summary["rotate_block_reason"] != "allow_rotate_false_intercept"

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
    assert coast_blocked.cmd.vx_mps == 0.0
    assert coast_blocked.control_summary["final_depth_latched"]
    assert coast_blocked.cmd.vy_mps == 0.0

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
    assert extreme_fov.control_summary["bbox_fov_guard_level"] == "hard"
    assert extreme_fov.cmd.vx_mps == 0.0
    assert extreme_fov.control_summary["forward_block_reason"] == "bbox_fov_guard_hard"

    # Side touch, large center error, and an established violation streak are
    # also hard; touch_bottom alone is never sufficient.
    probe = DockingProbe()
    side_hard_obs = bbox_obs(0.80)
    side_hard_obs.edge_found = True
    side_hard_obs.edge_trusted = True
    side_hard_obs.table_bbox_touch_right = True
    probe.ctx.bbox_fov_violation_streak = 3
    fov_blocked = edge_guided_decision(probe, side_hard_obs)
    assert fov_blocked.control_summary["bbox_fov_guard_level"] == "hard"
    assert fov_blocked.cmd.vx_mps == 0.0

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
    
    # First call resets/initializes min_dist_seen
    dec = edge_guided_decision(p_approach, obs_app)
    assert dec.cmd.vx_mps == 0.02
    assert p_approach.ctx.state != State.NO_PROGRESS_RECOVERY
    
    # Five seconds without distance change remains tolerated for vx=0.020.
    p_approach.ctx.dist_progress_last_refreshed_mono = monotonic_ts() - 6.0
    dec = edge_guided_decision(p_approach, obs_app)
    assert p_approach.ctx.state != State.NO_PROGRESS_RECOVERY
    assert dec.cmd.vx_mps == 0.02
    assert dec.control_summary["no_progress_warning"]
    assert dec.control_summary["no_progress_policy"] == "slow_forward_tolerated"

    # Only the longer 15-second window may permit strong recovery.
    p_approach.ctx.dist_progress_last_refreshed_mono = monotonic_ts() - 15.1
    dec = edge_guided_decision(p_approach, obs_app)
    assert p_approach.ctx.state == State.NO_PROGRESS_RECOVERY
    assert dec.control_summary["progress_recovery_allowed"]

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
    assert inv2.summary["docking_action"] == "CONTROL_RECOVERY_ROTATE"
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
    assert inv3.summary["docking_action"] == "FINAL_YAW_ALIGN"
    assert inv3.final_vx == 0.0 and inv3.final_vy == 0.0 and abs(inv3.final_wz) > 0.0

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
    assert inv5.reason == "near_latch_suppressed_far_fallback"

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

    for motion_result in (inv3, inv4, inv5, inv6, inv8, inv9, inv9_bbox_track):
        for owner_key in ("yaw_owner", "forward_owner", "lateral_owner"):
            assert owner_key in motion_result.summary
            assert str(motion_result.summary[owner_key]) != ""

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

    # Queue 2: Final yaw lock & realignment static tests
    # Case A: final_depth_latched=True + yaw_err=0.20 (which is > deadband 0.12)
    # Action must be FINAL_YAW_ALIGN, wz != 0, final_locked=False
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
    assert inv_q2_a.summary["docking_action"] == "FINAL_YAW_ALIGN"
    assert inv_q2_a.summary["docking_stage"] == "FINAL_YAW_ALIGN"
    assert inv_q2_a.final_vx == 0.0 and inv_q2_a.final_vy == 0.0 and abs(inv_q2_a.final_wz) > 0.0
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

    # Case D: final depth + edge yaw stale (age > hold_timeout) -> action=FINAL_LOCKED_STOP, wz=0, vx=0, reason=edge_yaw_stale
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
    assert inv_q2_d.summary["docking_action"] == "FINAL_LOCKED_STOP"
    assert inv_q2_d.summary["docking_stage"] == "FINAL_DISTANCE_HOLD"
    assert inv_q2_d.final_vx == 0.0 and inv_q2_d.final_wz == 0.0
    assert inv_q2_d.reason == "edge_yaw_stale"

    # Queue 3: Full workflow smoke test (SEARCH -> BBOX -> EDGE -> FINAL_YAW -> FINAL_LOCKED)
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

    # Step 4: Final depth latched at < 0.25m -> final_depth_latched becomes True, final_locked is False
    probe_smoke.ctx.state = State.FINAL_SLOW_STOP
    for _ in range(3):
        obs_s4 = TableEdgeObs.from_dict({
            "ts": now_ts(), "table_found": True, "edge_found": True, "edge_valid": True, "edge_trusted": True,
            "table_bbox_control_valid": True, "yaw_err_rad": 0.15, "dist_err_m": 0.20, "table_roi_depth_valid": True, "table_roi_depth_p10": 0.20,
        })
        dec_s4 = edge_guided_decision(probe_smoke, obs_s4)
    assert probe_smoke.ctx.final_depth_latched is True
    assert probe_smoke.ctx.final_locked is False

    # Step 5: Stable small yaw for 6 frames -> final_locked becomes True
    probe_smoke.ctx.final_yaw_align_start_mono = monotonic_ts() - 2.0
    for _ in range(6):
        obs_s5 = TableEdgeObs.from_dict({
            "ts": now_ts(), "table_found": True, "edge_found": True, "edge_valid": True, "edge_trusted": True,
            "table_bbox_control_valid": True, "yaw_err_rad": 0.05, "dist_err_m": 0.20, "table_roi_depth_valid": True, "table_roi_depth_p10": 0.20,
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

    # Test B: final yaw align cmd -> docking_action=FINAL_YAW_ALIGN, wz!=0
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
    assert res_b.summary["docking_action"] == "FINAL_YAW_ALIGN" or res_b.summary["docking_action"] == DockingAction.FINAL_YAW_ALIGN
    assert res_b.summary["docking_stage"] == "FINAL_YAW_ALIGN" or res_b.summary["docking_stage"] == DockingStage.FINAL_YAW_ALIGN
    assert abs(res_b.final_wz) > 1e-9

    # Test C: dead stale with no healthy history -> choose recovery/hold (vx=0)
    summary_c = {
        "state": "YOLO_APPROACH",
        "control_phase": "EDGE_GUIDED_APPROACH",
        "stale_level": "dead",
        "current_obs_healthy": False,
        "last_good_obs_healthy": False,
        "last_good_obs_age_ms": 999999.0,
    }
    res_c = arbitrate_table_docking_motion(test_ctx, obs_b, intent, summary_c)
    assert res_c.summary["docking_action"] in {"CONTROL_RECOVERY_ROTATE", "SEARCH_ROTATE", DockingAction.CONTROL_RECOVERY_ROTATE, DockingAction.SEARCH_ROTATE}
    assert res_c.final_vx == 0.0
    assert res_c.final_vy == 0.0

    print("docking static verification: PASS")


if __name__ == "__main__":
    main()
