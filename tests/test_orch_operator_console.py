#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
ORCH_ROOT = ROOT / "orchestrator"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from common.console_presenter import DemoConsolePresenter  # noqa: E402
from common.runtime_logging import OperatorConsole  # noqa: E402
from orchestrator_service.config.schema import OrchestratorConfig  # noqa: E402
from orchestrator_service.ipc.protocol import CmdVel, TableEdgeObs, now_ts  # noqa: E402
from orchestrator_service.runtime.controller import MotionDecision  # noqa: E402
from orchestrator_service.runtime.common import monotonic_ts  # noqa: E402
from orchestrator_service.runtime.context import State  # noqa: E402
from orchestrator_service.runtime.service import OrchestratorService  # noqa: E402
from orchestrator_service.runtime.state_machine import OrchestratorCore  # noqa: E402
from orchestrator_service.ipc.protocol import TargetObs  # noqa: E402
from orchestrator_service.mobile_gateway.runtime.service import MobileGatewayService  # noqa: E402


class _RunLogger:
    def __init__(self):
        self.records = []

    def write_jsonl(self, name, payload):
        self.records.append((name, dict(payload)))


def _service(lines, mode="operator"):
    svc = OrchestratorService.__new__(OrchestratorService)
    svc.cfg = OrchestratorConfig()
    svc.operator_console = OperatorConsole(mode=mode, default_interval_s=1.0, sink=lines.append)
    svc.demo_console = DemoConsolePresenter(svc.operator_console, dry_run=svc.cfg.serial.dry_run, emoji_enabled=False)
    svc.demo_console.level = "demo"
    svc._ipc_console_enabled = False
    svc._heartbeat_console_enabled = False
    svc._uart_console_mode = "operator"
    svc._mobile_status_console_mode = "change"
    svc._uart_console_key = ""
    svc._uart_console_last_emit_ts = 0.0
    svc._uart_console_repeat_count = 0
    svc._uart_console_last_payload = None
    svc._uart_lowfreq_key = ""
    svc._uart_lowfreq_last_emit_ts = 0.0
    svc._uart_lowfreq_repeat_count = 0
    svc._uart_lowfreq_last_payload = None
    svc._last_target_obs_console_payload = {}
    svc._last_target_search_req_console_key = ""
    svc._demo_start_pending_target = ""
    svc._demo_deferred_phases = []
    svc._pending_state_traces = []
    svc._last_uart_tx_ts = 0.0
    svc._edge_obs_rate_ts = []
    svc._target_obs_rate_ts = []
    svc.run_logger = _RunLogger()
    svc.core = SimpleNamespace(
        ctx=SimpleNamespace(
            state=SimpleNamespace(value="CONTROLLED_APPROACH"),
            current_edge_id=1,
            active_target="apple",
            active_vision_mode="TRACK_LOCAL",
            last_target_obs=None,
            last_table_obs=None,
            table_lock_frames=0,
            table_loss_since_mono=0.0,
            target_found_frames=0,
            target_lost_frames=0,
            last_enter_reason="distance_too_far",
        ),
        _loss_elapsed=lambda started_mono: 0.0,
        export_state_block=lambda: {
            "edge_found": True,
            "confidence": 0.86,
            "yaw_err_rad": 0.022,
            "dist_err_m": 0.057,
            "target_dist_m": None,
            "lock_ready": False,
            "lock_reason": "distance_too_far",
        },
    )
    return svc


def _set_slide_ref(core, yaw=0.01, dist=0.02, conf=0.9):
    core.ctx.slide_ref_ready = True
    core.ctx.slide_ref_yaw_err = yaw
    core.ctx.slide_ref_dist_err = dist
    core.ctx.slide_ref_edge_conf = conf
    core.ctx.handoff_state = "ready"


class OrchestratorOperatorConsoleTest(unittest.TestCase):
    def test_demo_presenter_start_phone_health_and_intent(self) -> None:
        lines = []
        console = OperatorConsole(mode="operator", default_interval_s=1.0, sink=lines.append)
        presenter = DemoConsolePresenter(console, dry_run=True, emoji_enabled=False, health_interval_s=3.0)
        presenter.task_start("apple")
        presenter.phone_command("apple")
        presenter.phone_accepted()
        presenter.health({
            "state": "EDGE_SLIDE_SEARCH",
            "target": "apple",
            "edge": "OK",
            "edge_hz": 6.1,
            "yolo_hz": 5.7,
            "preview": 5.8,
            "dry_run": True,
        })
        presenter.slide_intent(0.02, 0.14, -0.03)
        joined = "\n".join(lines)
        self.assertIn("[DEMO][START] target=apple mode=DRY_RUN", joined)
        self.assertIn("[DEMO][DRY_RUN] car commands are control intentions only; serial port is not opened.", joined)
        self.assertIn("[DEMO][PHONE] command received target=apple", joined)
        self.assertIn("[DEMO][PHONE] gateway accepted", joined)
        self.assertIn("[DEMO][HEALTH] state=EDGE_SLIDE_SEARCH target=apple edge=OK edge_hz=6.1 yolo_hz=5.7 preview=5.8FPS dry_run=1", joined)
        self.assertIn("[DEMO][SLIDE_INTENT] vx=+0.02 vy=+0.14 wz=-0.03 dry_run=1", joined)

    def test_demo_start_is_before_deferred_table_phase(self) -> None:
        lines = []
        svc = _service(lines, mode="operator")
        svc.cfg.serial.dry_run = True
        svc.demo_console.dry_run = True
        svc._demo_start_pending_target = "apple"
        svc._on_state_transition("IDLE", "SEARCH_TABLE", "开始桌边任务，目标 apple")
        self.assertNotIn("[DEMO][TABLE] searching edge", lines)
        svc.demo_console.task_start("apple")
        svc._flush_demo_deferred_phases()
        self.assertLess(lines.index("[DEMO][START] target=apple mode=DRY_RUN"), lines.index("[DEMO][TABLE] searching edge"))

    def test_idle_health_uses_off_not_stale(self) -> None:
        lines = []
        svc = _service(lines, mode="operator")
        svc.demo_console.level = "normal"
        svc.cfg.serial.dry_run = True
        svc.core.ctx.state = SimpleNamespace(value="IDLE")
        decision = MotionDecision(
            cmd=CmdVel(ts=now_ts(), mode="IDLE", vx_norm=0.0, vy_norm=0.0, wz_norm=0.0),
            control_summary={"state": "IDLE", "edge_obs_is_stale": True},
        )
        svc._emit_operator_control(decision)
        joined = "\n".join(lines)
        self.assertIn("[DEMO][IDLE] waiting for command", joined)
        self.assertIn("[DEMO][HEALTH] state=IDLE target=n/a edge=OFF yolo=OFF preview=n/a dry_run=1", joined)
        self.assertNotIn("edge=STALE", joined)

    def test_operator_mode_suppresses_ipc_success_and_heartbeat(self) -> None:
        lines = []
        svc = _service(lines, mode="operator")
        svc._operator_ipc_event("vision_req_out", "send_attempt", {})
        svc._operator_ipc_event("vision_req_out", "send_ok", {})
        svc._operator_ipc_event("vision_req_out", "async_enqueue", {})
        svc._operator_ipc_event("task_cmd_in", "received", {})
        self.assertEqual(lines, [])

    def test_operator_mode_reports_ipc_connectivity_errors(self) -> None:
        lines = []
        svc = _service(lines, mode="operator")
        svc._operator_ipc_event("vision_req_out", "connected", {})
        svc._operator_ipc_event("vision_req_out", "connect_failed", {"error": "Connection refused", "fail_count": 3})
        svc._operator_ipc_event("vision_obs_in", "invalid_json", {"peer": "p1", "error": "bad json"})
        joined = "\n".join(lines)
        self.assertIn("[ORCH] IPC vision_req_out connected", joined)
        self.assertIn("[ORCH] WARN vision_req_out connect_failed", joined)
        self.assertIn("[ORCH] ERROR vision_obs_in invalid_json", joined)

    def test_full_mode_allows_ipc_success_console(self) -> None:
        lines = []
        svc = _service(lines, mode="full")
        svc._operator_ipc_event("vision_req_out", "send_ok", {})
        self.assertTrue(any("send_ok" in line for line in lines))

    def test_repeated_control_summary_is_rate_limited(self) -> None:
        lines = []
        svc = _service(lines, mode="operator")
        decision = MotionDecision(
            cmd=CmdVel(ts=now_ts(), mode="CONTROLLED_APPROACH", vx_norm=0.04, vy_norm=0.0, wz_norm=-0.06),
            control_summary={"state": "CONTROLLED_APPROACH", "reason": "distance_too_far"},
        )
        with patch("common.runtime_logging.time.time", side_effect=[100.0, 100.2, 101.2]):
            svc._emit_operator_control(decision)
            svc._emit_operator_control(decision)
            svc._emit_operator_control(decision)
        self.assertEqual(len(lines), 2)
        self.assertIn("[ORCH] CTRL state=CONTROLLED_APPROACH", lines[0])

    def test_state_transition_emits_once_and_no_unchanged_tick(self) -> None:
        lines = []
        svc = _service(lines, mode="operator")
        svc._on_state_transition("SEARCH_TABLE", "COARSE_ALIGN", "table_edge_seen")
        svc._on_state_transition("SEARCH_TABLE", "COARSE_ALIGN", "table_edge_seen")
        self.assertIn("[ORCH] STATE SEARCH_TABLE -> COARSE_ALIGN reason=table_edge_seen", lines)
        self.assertIn("[DEMO][TABLE] edge found, aligning", lines)

    def test_fake_car_repeated_vel_mode_and_speed_policy(self) -> None:
        lines = []
        svc = _service(lines, mode="operator")
        base = {
            "ts": 100.0,
            "dry_run": True,
            "summary_key": "a",
            "mode": "CONTROLLED_APPROACH",
            "kind": "cmd_vel",
            "vx_norm": 0.04,
            "vy_norm": 0.0,
            "wz_norm": -0.06,
            "hold_ms": 150,
            "raw": "VEL 0.040 0.000 -0.060 150",
        }
        svc._update_uart_console(dict(base))
        svc._update_uart_console(dict(base, ts=100.2))
        svc._update_uart_console(dict(base, ts=100.3, summary_key="b", mode="FINAL_LOCK"))
        svc._update_uart_console(dict(base, ts=100.4, summary_key="c", mode="FINAL_LOCK", vx_norm=0.052))
        self.assertEqual(len(lines), 2)
        self.assertIn("[ORCH] CAR_VEL vx=+0.040", lines[0])
        self.assertIn("vx=+0.052", lines[1])

    def test_uart_console_splits_mode_and_vel(self) -> None:
        lines = []
        svc = _service(lines, mode="operator")
        svc._on_uart_tx(
            "MODE SEARCH_TABLE\nVEL 0.000 0.000 0.220 150\n",
            True,
            {"mode": "SEARCH_TABLE", "kind": "cmd_vel", "vx_norm": 0.0, "vy_norm": 0.0, "wz_norm": 0.22, "hold_ms": 150, "state": "SEARCH_TABLE"},
        )
        joined = "\n".join(lines)
        self.assertIn("[ORCH] CAR_MODE mode=SEARCH_TABLE", joined)
        self.assertIn("[ORCH] CAR_VEL vx=+0.000 vy=+0.000 wz=+0.220 hold=150ms", joined)
        self.assertNotIn("[ORCH] CAR mode=", joined)

    def test_ctrl_nonzero_without_vel_warns_no_vel_sent(self) -> None:
        lines = []
        svc = _service(lines, mode="operator")
        svc._on_uart_tx(
            "MODE EDGE_SLIDE_SEARCH\n",
            True,
            {"mode": "EDGE_SLIDE_SEARCH", "kind": "cmd_vel", "vx_norm": 0.0, "vy_norm": 0.04, "wz_norm": 0.0, "hold_ms": 150, "state": "EDGE_SLIDE_SEARCH", "reason": "edge_slide"},
        )
        joined = "\n".join(lines)
        self.assertIn("[ORCH] WARN no_vel_sent state=EDGE_SLIDE_SEARCH reason=edge_slide", joined)
        self.assertIn("[ORCH] CAR_MODE mode=EDGE_SLIDE_SEARCH", joined)

    def test_mobile_status_change_policy(self) -> None:
        lines = []
        svc = MobileGatewayService.__new__(MobileGatewayService)
        svc._mobile_status_console_mode = "change"
        svc._last_operator_status_key = ""
        svc.operator_console = OperatorConsole(mode="operator", default_interval_s=1.0, sink=lines.append)
        payload = {
            "state": "locking_table",
            "backend_state": "FINAL_LOCK",
            "target": "apple",
            "progress": 65,
            "lock_reason": "yaw_not_aligned",
        }
        svc._emit_mobile_status_console(dict(payload))
        svc._emit_mobile_status_console(dict(payload))
        svc._emit_mobile_status_console(dict(payload, lock_reason="distance_too_close"))
        self.assertEqual(len(lines), 2)
        self.assertIn("reason=yaw_not_aligned", lines[0])
        self.assertIn("reason=distance_too_close", lines[1])

    def test_target_obs_summary_in_search_states(self) -> None:
        lines = []
        svc = _service(lines, mode="operator")
        svc.core.ctx.state = SimpleNamespace(value="EDGE_SLIDE_SEARCH")
        payload = {"type": "target_obs", "target": "apple", "found": False, "boxes_count": 3, "vision_mode": "TRACK_LOCAL"}
        svc._emit_target_obs_console(payload)
        self.assertEqual(lines, ["[ORCH] OBS target=apple found=0 boxes=3 mode=TRACK_LOCAL"])
        svc.operator_console._last_ts_by_key.clear()
        svc._emit_target_obs_console({
            "type": "target_obs",
            "target": "apple",
            "target_found": True,
            "matched_cls": "apple",
            "matched_conf": 0.81,
            "best_cls": "keyboard",
            "best_conf": 0.92,
            "matched_center_full_norm": {"cx": 0.54, "cy": 0.47},
            "matched_center_offset_norm": {"dx": -0.08, "dy": 0.06},
            "cx_norm": 0.54,
            "cy_norm": 0.47,
        })
        self.assertIn("[ORCH] OBS target=apple found=1 matched_cls=apple matched_conf=0.81 best_cls=keyboard best_conf=0.92 cx=0.54 cy=0.47", lines[-1])

    def test_target_obs_keeps_full_center_separate_from_offset(self) -> None:
        obs = TargetObs.from_dict(
            {
                "type": "target_obs",
                "target": "apple",
                "target_found": True,
                "matched_center": {"cx": 0.92, "cy": 0.50},
                "matched_center_full_norm": {"cx": 0.92, "cy": 0.50},
                "matched_center_offset_norm": {"dx": -0.84, "dy": 0.0},
                "cx_norm": -0.84,
                "cy_norm": 0.50,
                "bbox_valid": False,
                "bbox_invalid_reason": "bbox_out_of_frame",
            }
        )
        payload = obs.to_dict()
        self.assertAlmostEqual(payload["matched_center_full_norm"]["cx"], 0.92)
        self.assertAlmostEqual(payload["matched_center_offset_norm"]["dx"], -0.84)
        self.assertAlmostEqual(payload["cx_norm"], -0.84)
        self.assertFalse(payload["bbox_valid"])
        self.assertEqual(payload["bbox_invalid_reason"], "bbox_out_of_frame")

    def test_state_trace_center_reconstructs_legacy_offset_center(self) -> None:
        cfg = OrchestratorConfig()
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        obs = TargetObs.from_dict(
            {
                "type": "target_obs",
                "target": "apple",
                "target_found": True,
                "matched_center": {"cx": -0.7890625, "cy": 0.50546875},
                "cx_norm": -0.7890625,
                "cy_norm": 0.50546875,
            }
        )
        center = core._target_center(obs)
        self.assertIsNotNone(center)
        self.assertGreaterEqual(center["cx"], 0.0)
        self.assertLessEqual(center["cx"], 1.0)
        self.assertAlmostEqual(center["cx"], 0.89453125)
        self.assertAlmostEqual(center["cy"], 0.50546875)

    def test_target_obs_keeps_detection_summary_fields(self) -> None:
        obs = TargetObs.from_dict(
            {
                "type": "target_obs",
                "target": "apple",
                "found": False,
                "boxes_count": 2,
                "matched_cls": "cup",
                "matched_conf": 0.55,
                "best_cls": "cup",
                "best_conf": 0.67,
                "cy_norm": 0.44,
                "reason": "no_boxes",
            }
        )
        payload = obs.to_dict()
        self.assertEqual(payload["boxes_count"], 2)
        self.assertEqual(payload["matched_cls"], "cup")
        self.assertAlmostEqual(payload["matched_conf"], 0.55)
        self.assertEqual(payload["best_cls"], "cup")
        self.assertAlmostEqual(payload["best_conf"], 0.67)
        self.assertAlmostEqual(payload["cy_norm"], 0.44)
        self.assertEqual(payload["reason"], "no_boxes")

    def test_target_confirm_holds_through_short_lost_obs(self) -> None:
        cfg = OrchestratorConfig()
        cfg.control.target_confirm_min_s = 0.8
        cfg.control.target_confirm_lost_hold_s = 1.2
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.TARGET_CONFIRM
        core.ctx.active_target = "apple"
        core.ctx.task_start_wall_ts = 1.0
        core.ctx.state_enter_mono = monotonic_ts()
        core.ctx.last_target_obs = TargetObs.from_dict(
            {
                "type": "target_obs",
                "target": "apple",
                "target_found": False,
                "matched_cls": "apple",
                "matched_conf": 0.0,
                "ts": now_ts(),
            }
        )
        core.tick()
        self.assertEqual(core.ctx.state, State.TARGET_CONFIRM)
        self.assertEqual(core.ctx.target_lost_frames, 1)

    def test_target_locked_holds_through_single_lost_obs(self) -> None:
        cfg = OrchestratorConfig()
        cfg.control.target_lock_lost_hold_s = 1.5
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.TARGET_LOCKED
        core.ctx.active_target = "apple"
        core.ctx.task_start_wall_ts = 1.0
        core.ctx.state_enter_mono = monotonic_ts()
        core.ctx.last_target_obs = TargetObs.from_dict(
            {
                "type": "target_obs",
                "target": "apple",
                "target_found": False,
                "matched_cls": "apple",
                "matched_conf": 0.0,
                "ts": now_ts(),
            }
        )
        core.tick()
        self.assertEqual(core.ctx.state, State.TARGET_LOCKED)
        self.assertEqual(core.ctx.target_lost_frames, 1)

    def test_target_locked_uses_window_stats_to_freeze(self) -> None:
        cfg = OrchestratorConfig()
        cfg.control.target_lock_conf_th = 0.4
        cfg.control.target_lock_found_ratio_th = 0.6
        cfg.control.target_lock_stable_s = 0.1
        cfg.control.target_locked_freeze_after_s = 0.1
        cfg.control.target_lock_center_jitter_th = 0.08
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.TARGET_LOCKED
        core.ctx.active_target = "apple"
        core.ctx.task_start_wall_ts = 1.0
        core.ctx.state_enter_mono = monotonic_ts() - 0.2
        core.ctx.target_stable_since_mono = monotonic_ts() - 0.2
        core.ctx.last_target_obs = TargetObs.from_dict(
            {
                "type": "target_obs",
                "target": "apple",
                "target_found": True,
                "matched_cls": "apple",
                "matched_conf": 0.82,
                "matched_center_full_norm": {"cx": 0.52, "cy": 0.48},
                "matched_center_offset_norm": {"dx": -0.04, "dy": 0.04},
                "cx_norm": -0.04,
                "cy_norm": 0.48,
                "bbox_valid": True,
                "ts": now_ts(),
            }
        )
        core.tick()
        self.assertEqual(core.ctx.state, State.FREEZE_BASE)

    def test_edge_slide_zero_cmd_has_operator_reason(self) -> None:
        lines = []
        svc = _service(lines, mode="operator")
        svc.core.ctx.state = SimpleNamespace(value="EDGE_SLIDE_SEARCH")
        svc.core.ctx.active_vision_mode = "TRACK_LOCAL"
        decision = MotionDecision(
            cmd=CmdVel(ts=now_ts(), mode="EDGE_SLIDE_SEARCH", vx_norm=0.0, vy_norm=0.0, wz_norm=0.0),
            control_summary={"state": "EDGE_SLIDE_SEARCH", "reason": "safety_hold"},
        )
        svc._emit_operator_control(decision)
        self.assertEqual(len(lines), 1)
        self.assertIn("[ORCH][SLIDE] mode=strong edge_valid=1", lines[0])
        self.assertIn("found=0 boxes=0", lines[0])
        self.assertIn("vx=+0.000 vy=+0.000 wz=+0.000", lines[0])
        self.assertIn("reason=waiting_first_target_obs", lines[0])

    def test_edge_lost_hold_reason_is_visible(self) -> None:
        lines = []
        svc = _service(lines, mode="operator")
        svc.core.ctx.state = SimpleNamespace(value="EDGE_SLIDE_SEARCH")
        decision = MotionDecision(
            cmd=CmdVel(ts=now_ts(), mode="EDGE_SLIDE_SEARCH", vx_norm=0.0, vy_norm=0.0, wz_norm=0.0),
            control_summary={"state": "EDGE_SLIDE_SEARCH", "reason": "safety_hold_no_edge", "edge_found": False},
        )
        svc._emit_operator_control(decision)
        self.assertIn("[ORCH][SLIDE] mode=strong edge_valid=0", lines[0])
        self.assertIn("found=0", lines[0])
        self.assertIn("reason=safety_hold_no_edge", lines[0])

    def test_state_transition_edge_slide_timeout_is_diagnostic(self) -> None:
        lines = []
        svc = _service(lines, mode="operator")
        svc._on_state_transition("EDGE_SLIDE_SEARCH", "LEAVE_EDGE", "当前桌边未找到目标，切换到边 right")
        self.assertIn("[ORCH] STATE EDGE_SLIDE_SEARCH -> LEAVE_EDGE reason=target_not_found timeout_s=10.0", lines)
        self.assertIn("[DEMO][TARGET] current edge timeout, relocating", lines)

    def test_state_transition_target_found_is_diagnostic(self) -> None:
        lines = []
        svc = _service(lines, mode="operator")
        svc._on_state_transition("EDGE_SLIDE_SEARCH", "TARGET_CONFIRM", "检测到目标候选，开始确认")
        self.assertIn("[ORCH] STATE EDGE_SLIDE_SEARCH -> TARGET_CONFIRM reason=target_found", lines)
        self.assertIn("[DEMO][TARGET] candidate found, confirming", lines)

    def test_state_trace_preserves_matched_target_reason(self) -> None:
        lines = []
        svc = _service(lines, mode="operator")
        svc.core.last_transition_snapshot = {
            "event": "state_transition",
            "previous_state": "EDGE_SLIDE_SEARCH",
            "next_state": "TARGET_CONFIRM",
            "reason": "",
        }
        svc._on_state_transition("EDGE_SLIDE_SEARCH", "TARGET_CONFIRM", "target_found matched_cls=apple matched_conf=0.780")
        self.assertEqual(svc._pending_state_traces[-1]["reason"], "target_found matched_cls=apple matched_conf=0.780")

    def test_search_table_entry_clears_stale_edge_and_locks(self) -> None:
        cfg = OrchestratorConfig()
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.CONTROLLED_APPROACH
        core.ctx.active_target = "apple"
        core.ctx.active_session_id = "sess_reset"
        core.ctx.last_table_obs = TableEdgeObs(ts=now_ts(), table_found=True, edge_found=True, confidence=0.9, yaw_err_rad=0.1, dist_err_m=0.02)
        core.ctx.locked_edge_conf = 0.9
        core.ctx.locked_yaw_err = 0.1
        core.ctx.slide_ref_ready = True
        core._transition(State.SEARCH_TABLE, "reset_for_new_search")
        self.assertIsNone(core.ctx.last_table_obs)
        self.assertIsNone(core.ctx.locked_edge_conf)
        self.assertFalse(core.ctx.slide_ref_ready)
        block = core.export_state_block()
        self.assertFalse(block["edge_valid"])
        self.assertIsNone(block["confidence"])
        self.assertIsNone(block["yaw_err_rad"])
        self.assertFalse(block["lock_ready"])
        self.assertTrue(any(t.get("reset_state") == "edge" for t in core._pending_reset_traces))

    def test_done_and_idle_console_are_explicit_and_warning_is_not_reason(self) -> None:
        lines = []
        svc = _service(lines, mode="operator")
        svc.cfg.serial.dry_run = True
        svc.core.ctx = SimpleNamespace(
            state=SimpleNamespace(value="DONE"),
            active_session_id="sess_done",
            active_target="apple",
            task_start_wall_ts=now_ts() - 3.0,
            dock_retry_count=1,
            edge_transition_count=2,
            task_slide_entries_count=4,
            task_target_confirm_count=2,
            task_target_locked_count=1,
            task_warning_history=["distance_too_far"],
            last_fail_reason="edge_follow_stale",
            last_target_obs=TargetObs(ts=now_ts(), found=True, target="apple", matched_cls="apple", matched_conf=0.82),
            last_table_obs=TableEdgeObs(ts=now_ts(), table_found=True, edge_found=True, confidence=0.77),
        )
        svc._on_state_transition("FREEZE_BASE", "DONE", "已在桌边锁定 apple")
        joined = "\n".join(lines)
        self.assertIn("[ORCH][TASK_DONE]", joined)
        self.assertIn("[DEMO][DRY_RUN] car commands are control intentions only; serial port is not opened.", joined)
        self.assertIn("[DEMO][SUCCESS] TASK DONE", joined)
        self.assertIn("next_state    : IDLE_HOT", joined)
        self.assertIn("preview       : kept alive", joined)
        self.assertIn("waiting       : next command", joined)
        self.assertIn("result=success", joined)
        self.assertIn("reason=已在桌边锁定 apple", joined)
        self.assertIn("warnings=['distance_too_far', 'edge_follow_stale']", joined)
        svc._on_state_transition("DONE", "IDLE", "任务完成，回到空闲")
        self.assertIn("[ORCH][IDLE] task finished, waiting for next command", "\n".join(lines))
        self.assertIn("[DEMO][IDLE_HOT] preview kept alive, waiting for next command", "\n".join(lines))

    def test_failed_demo_banner_is_explicit(self) -> None:
        lines = []
        svc = _service(lines, mode="operator")
        svc.core.ctx = SimpleNamespace(
            state=SimpleNamespace(value="ERROR_RECOVERY"),
            active_session_id="sess_failed",
            active_target="apple",
            task_start_wall_ts=now_ts() - 5.0,
            last_fail_reason="target_not_found",
            last_target_obs=None,
            last_table_obs=None,
        )
        svc._on_state_transition("EDGE_SLIDE_SEARCH", "ERROR_RECOVERY", "target_not_found timeout_s=10.0")
        joined = "\n".join(lines)
        self.assertIn("[DEMO][FAILED] TASK FAILED", joined)
        self.assertIn("final_state   : FAILED", joined)
        self.assertIn("reason        : target_not_found", joined)

    def test_target_search_req_summary_is_compact(self) -> None:
        lines = []
        svc = _service(lines, mode="operator")
        line = svc._vision_req_console_summary(
            {
                "type": "vision_req",
                "stage": "SEARCH",
                "mode_hint": "TRACK_LOCAL",
                "target": "apple",
                "req_id": "req_1",
                "payload": {"search_kind": "TARGET"},
            }
        )
        self.assertEqual(line, "[ORCH] REQ target_update stage=SEARCH kind=TARGET target=apple mode_hint=TRACK_LOCAL req=req_1")

    def test_target_obs_found_enters_confirm_stop_logic(self) -> None:
        cfg = OrchestratorConfig()
        cfg.control.target_found_frames_to_confirm = 1
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.EDGE_SLIDE_SEARCH
        core.ctx.active_target = "apple"
        core.ctx.task_start_wall_ts = 1.0
        core.handle_target_obs(TargetObs(ts=now_ts(), found=True, target="apple", confidence=0.81, cx_norm=0.54))
        decision = core.tick()
        self.assertEqual(core.ctx.state, State.TARGET_CONFIRM)
        self.assertEqual(decision.cmd.mode, "TARGET_CONFIRM")

    def test_edge_slide_accepts_matched_target_even_when_best_cls_differs(self) -> None:
        cfg = OrchestratorConfig()
        cfg.control.edge_slide_pause_s = 0.0
        cfg.control.target_confirm_conf_th = 0.65
        cfg.control.target_found_frames_to_confirm = 1
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.EDGE_SLIDE_SEARCH
        core.ctx.active_target = "apple"
        core.ctx.task_start_wall_ts = 1.0
        core.ctx.state_enter_mono = monotonic_ts() - 1.0
        _set_slide_ref(core)
        core.handle_table_obs(TableEdgeObs(ts=now_ts(), table_found=True, edge_found=True, confidence=0.9, yaw_err_rad=0.01, dist_err_m=0.02))
        core.handle_target_obs(
            TargetObs(
                ts=now_ts(),
                found=True,
                target_found=True,
                target="apple",
                matched_cls="apple",
                matched_conf=0.78,
                best_cls="keyboard",
                best_conf=0.92,
                confidence=0.78,
                cx_norm=0.54,
            )
        )
        decision = core.tick()
        self.assertEqual(core.ctx.state, State.TARGET_CONFIRM)
        self.assertEqual(decision.cmd.mode, "TARGET_CONFIRM")
        self.assertIn("matched_cls=apple", core.ctx.last_enter_reason)
        self.assertIn("matched_conf=0.780", core.ctx.last_enter_reason)

    def test_target_confirm_requires_configured_confidence(self) -> None:
        cfg = OrchestratorConfig()
        cfg.control.target_confirm_conf_th = 0.65
        cfg.control.target_found_frames_to_confirm = 1
        cfg.control.target_confirm_lost_frames = 1
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.TARGET_CONFIRM
        core.ctx.active_target = "apple"
        core.ctx.task_start_wall_ts = 1.0
        core.handle_target_obs(TargetObs(ts=now_ts(), found=True, target="apple", best_cls="apple", confidence=0.61, cx_norm=0.54))
        decision = core.tick()
        self.assertEqual(core.ctx.state, State.TARGET_CONFIRM)
        self.assertEqual(decision.cmd.mode, "TARGET_CONFIRM")
        self.assertIn("conf_low", core.ctx.target_last_lost_reason)

    def test_target_locked_accepts_matched_target_even_when_best_cls_differs(self) -> None:
        cfg = OrchestratorConfig()
        cfg.control.target_lock_conf_th = 0.70
        cfg.control.target_confirm_lost_frames = 1
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.TARGET_LOCKED
        core.ctx.active_target = "apple"
        core.ctx.task_start_wall_ts = 1.0
        core.handle_target_obs(
            TargetObs(
                ts=now_ts(),
                found=True,
                target_found=True,
                target="apple",
                matched_cls="apple",
                matched_conf=0.81,
                best_cls="mouse",
                best_conf=0.95,
                confidence=0.81,
                cx_norm=0.54,
            )
        )
        decision = core.tick()
        self.assertEqual(core.ctx.state, State.TARGET_LOCKED)
        self.assertEqual(decision.cmd.mode, "TARGET_LOCKED")

    def test_edge_slide_requires_consecutive_target_frames(self) -> None:
        cfg = OrchestratorConfig()
        cfg.control.edge_slide_pause_s = 0.0
        cfg.control.target_confirm_conf_th = 0.30
        cfg.control.target_found_frames_to_confirm = 3
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.EDGE_SLIDE_SEARCH
        core.ctx.active_target = "apple"
        core.ctx.task_start_wall_ts = 1.0
        core.ctx.state_enter_mono = monotonic_ts() - 1.0
        _set_slide_ref(core)
        core.handle_table_obs(TableEdgeObs(ts=now_ts(), table_found=True, edge_found=True, confidence=0.9, yaw_err_rad=0.01, dist_err_m=0.02))

        for expected_state in (State.EDGE_SLIDE_SEARCH, State.EDGE_SLIDE_SEARCH, State.TARGET_CONFIRM):
            core.handle_target_obs(TargetObs(ts=now_ts(), found=True, target="apple", matched_cls="apple", matched_conf=0.50, cx_norm=0.54))
            decision = core.tick()
            self.assertEqual(core.ctx.state, expected_state)

        self.assertEqual(decision.cmd.mode, "TARGET_CONFIRM")
        self.assertEqual(core.ctx.target_found_frames, 3)

    def test_target_confirm_respects_min_stable_before_lock(self) -> None:
        cfg = OrchestratorConfig()
        cfg.control.target_confirm_conf_th = 0.30
        cfg.control.target_lock_conf_th = 0.40
        cfg.control.target_confirm_min_s = 0.60
        cfg.control.target_lock_stable_s = 1.20
        cfg.control.target_lock_center_jitter_th = 0.08
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.TARGET_CONFIRM
        core.ctx.active_target = "apple"
        core.ctx.task_start_wall_ts = 1.0
        obs = TargetObs(ts=now_ts(), found=True, target="apple", matched_cls="apple", matched_conf=0.80, cx_norm=0.54, cy_norm=0.50)

        core.handle_target_obs(obs)
        decision = core.tick()
        self.assertEqual(core.ctx.state, State.TARGET_CONFIRM)
        self.assertEqual(decision.cmd.mode, "TARGET_CONFIRM")

        core.ctx.state_enter_mono = monotonic_ts() - 0.70
        core.ctx.target_stable_since_mono = monotonic_ts() - 1.30
        core.handle_target_obs(obs)
        decision = core.tick()
        self.assertEqual(core.ctx.state, State.TARGET_LOCKED)
        self.assertEqual(decision.cmd.mode, "TARGET_LOCKED")

    def test_target_locked_holds_short_loss(self) -> None:
        cfg = OrchestratorConfig()
        cfg.control.target_lock_conf_th = 0.40
        cfg.control.target_lock_lost_hold_s = 1.20
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.TARGET_LOCKED
        core.ctx.active_target = "apple"
        core.ctx.task_start_wall_ts = 1.0

        core.handle_target_obs(TargetObs(ts=now_ts(), found=False, target="apple", matched_cls="apple", matched_conf=0.0))
        decision = core.tick()
        self.assertEqual(core.ctx.state, State.TARGET_LOCKED)
        self.assertEqual(decision.cmd.mode, "TARGET_LOCKED")

        core.ctx.target_loss_since_mono = monotonic_ts() - 1.30
        decision = core.tick()
        self.assertEqual(core.ctx.state, State.EDGE_SLIDE_SEARCH)
        self.assertEqual(decision.cmd.mode, "EDGE_SLIDE_SEARCH")

    def test_target_locked_reaches_freeze_when_stable(self) -> None:
        cfg = OrchestratorConfig()
        cfg.control.target_lock_conf_th = 0.40
        cfg.control.target_lock_stable_s = 1.20
        cfg.control.target_lock_center_jitter_th = 0.08
        cfg.control.target_locked_freeze_after_s = 0.80
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.TARGET_LOCKED
        core.ctx.active_target = "apple"
        core.ctx.task_start_wall_ts = 1.0
        core.ctx.state_enter_mono = monotonic_ts() - 0.90
        core.ctx.target_stable_since_mono = monotonic_ts() - 1.50
        core.handle_target_obs(TargetObs(ts=now_ts(), found=True, target="apple", matched_cls="apple", matched_conf=0.80, cx_norm=0.54, cy_norm=0.50))

        decision = core.tick()
        self.assertEqual(core.ctx.state, State.FREEZE_BASE)
        self.assertEqual(decision.cmd.mode, "FREEZE_BASE")

    def test_edge_slide_config_nonzero_outputs_nonzero_vy(self) -> None:
        cfg = OrchestratorConfig()
        cfg.car.edge_slide_vy_norm = 0.04
        cfg.control.edge_slide_pause_s = 0.0
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.EDGE_SLIDE_SEARCH
        core.ctx.active_vision_mode = "TRACK_LOCAL"
        core.ctx.state_enter_mono = monotonic_ts() - 1.0
        _set_slide_ref(core)
        core.handle_table_obs(TableEdgeObs(ts=now_ts(), table_found=True, edge_found=True, confidence=0.9, yaw_err_rad=0.01, dist_err_m=0.02))
        decision = core.tick()
        self.assertEqual(decision.cmd.mode, "EDGE_SLIDE_SEARCH")
        self.assertAlmostEqual(decision.cmd.vy_norm, 0.04, places=6)
        self.assertNotEqual(decision.cmd.vx_norm, 0.0)

    def test_track_local_weak_edge_conf_continues_slow_slide(self) -> None:
        cfg = OrchestratorConfig()
        cfg.control.edge_slide_pause_s = 0.0
        cfg.car.edge_slide_vy_norm = 0.14
        cfg.car.edge_slide_weak_vy_norm = 0.05
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.EDGE_SLIDE_SEARCH
        core.ctx.active_vision_mode = "TRACK_LOCAL"
        core.ctx.locked_yaw_err = -0.03
        core.ctx.locked_dist_err = 0.02
        core.ctx.locked_edge_conf = 0.82
        _set_slide_ref(core, yaw=-0.035, dist=0.023, conf=0.22)
        core.ctx.state_enter_mono = monotonic_ts() - 1.0
        core.handle_table_obs(
            TableEdgeObs(
                ts=now_ts(),
                table_found=True,
                edge_found=True,
                edge_valid=True,
                confidence=0.22,
                yaw_err_rad=-0.035,
                dist_err_m=0.023,
                source_mode="TRACK_LOCAL",
            )
        )
        decision = core.tick()
        self.assertEqual(core.ctx.state, State.EDGE_SLIDE_SEARCH)
        self.assertEqual(decision.cmd.mode, "EDGE_SLIDE_SEARCH")
        self.assertAlmostEqual(decision.cmd.vy_norm, 0.05, places=6)
        self.assertEqual(decision.control_summary["reason"], "weak_edge_slide")
        self.assertEqual(core.ctx.last_edge_quality.get("mode"), "weak")
        self.assertTrue(core.ctx.last_edge_quality.get("edge_identity_ok"))

    def test_search_target_init_builds_slide_ref_from_track_local_light_edge(self) -> None:
        cfg = OrchestratorConfig()
        cfg.control.edge_slide_pause_s = 0.0
        cfg.control.edge_handoff_samples = 3
        cfg.control.edge_handoff_min_s = 0.5
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.SEARCH_TARGET_INIT
        core.ctx.active_vision_mode = "TRACK_LOCAL"
        core.ctx.locked_yaw_err = -0.035
        core.ctx.locked_dist_err = 0.016
        core.ctx.locked_edge_conf = 0.798
        core.ctx.handoff_state = "collecting"
        core.ctx.state_enter_mono = monotonic_ts() - 0.6

        for seq, yaw, dist, conf in (
            (1, -0.37, 0.036, 0.30),
            (2, -0.38, 0.038, 0.31),
            (3, -0.39, 0.040, 0.32),
        ):
            core.handle_table_obs(
                TableEdgeObs(
                    ts=now_ts(),
                    table_found=True,
                    edge_found=True,
                    edge_valid=True,
                    confidence=conf,
                    yaw_err_rad=yaw,
                    dist_err_m=dist,
                    frame_id=seq,
                    seq=seq,
                    source_mode="TRACK_LOCAL",
                )
            )
            decision = core.tick()

        self.assertEqual(core.ctx.state, State.EDGE_SLIDE_SEARCH)
        self.assertEqual(decision.cmd.mode, "SEARCH_TARGET_INIT")
        self.assertTrue(core.ctx.slide_ref_ready)
        self.assertAlmostEqual(core.ctx.slide_ref_yaw_err, -0.38, places=6)
        self.assertAlmostEqual(core.ctx.slide_ref_dist_err, 0.038, places=6)
        self.assertAlmostEqual(core._full_vs_light_yaw_offset(), -0.345, places=6)

        core.ctx.state_enter_mono = monotonic_ts() - 1.0
        core.handle_table_obs(
            TableEdgeObs(
                ts=now_ts(),
                table_found=True,
                edge_found=True,
                edge_valid=True,
                confidence=0.31,
                yaw_err_rad=-0.385,
                dist_err_m=0.039,
                frame_id=4,
                seq=4,
                source_mode="TRACK_LOCAL",
            )
        )
        decision = core.tick()
        self.assertEqual(core.ctx.state, State.EDGE_SLIDE_SEARCH)
        self.assertEqual(decision.control_summary["reason"], "weak_edge_slide")
        self.assertEqual(core.ctx.last_edge_quality.get("edge_identity_basis"), "slide_ref")
        self.assertTrue(core.ctx.last_edge_quality.get("edge_identity_ok"))
        self.assertLess(abs(float(core.ctx.last_edge_quality.get("yaw_delta_from_slide_ref"))), 0.01)
        self.assertGreater(abs(float(core.ctx.last_edge_quality.get("yaw_delta_from_locked"))), 0.30)

    def test_edge_slide_distance_out_of_tolerance_recovers_final_lock(self) -> None:
        cfg = OrchestratorConfig()
        cfg.control.edge_slide_pause_s = 0.0
        cfg.control.edge_slide_dist_tolerance_m = 0.05
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.EDGE_SLIDE_SEARCH
        core.ctx.active_vision_mode = "TRACK_LOCAL"
        core.ctx.state_enter_mono = monotonic_ts() - 1.0
        _set_slide_ref(core, yaw=0.01, dist=0.08, conf=0.9)
        core.handle_table_obs(TableEdgeObs(ts=now_ts(), table_found=True, edge_found=True, confidence=0.9, yaw_err_rad=0.01, dist_err_m=0.08))
        decision = core.tick()
        self.assertEqual(core.ctx.state, State.EDGE_SLIDE_SEARCH)
        self.assertEqual(decision.cmd.mode, "EDGE_SLIDE_SEARCH")
        core.ctx.table_loss_since_mono = monotonic_ts() - float(cfg.control.edge_slide_recover_timeout_s) - 0.1
        decision = core.tick()
        self.assertEqual(core.ctx.state, State.FINAL_LOCK)
        self.assertEqual(decision.cmd.mode, "FINAL_LOCK")
        self.assertEqual(core.ctx.edge_slide_relock_attempts, 1)

    def test_edge_slide_distance_out_of_tolerance_fails_after_retries(self) -> None:
        cfg = OrchestratorConfig()
        cfg.control.edge_slide_pause_s = 0.0
        cfg.control.edge_slide_dist_tolerance_m = 0.05
        cfg.control.edge_slide_dist_out_of_range_hold_s = 0.1
        cfg.control.edge_slide_max_relock_attempts = 1
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.EDGE_SLIDE_SEARCH
        core.ctx.active_target = "apple"
        core.ctx.active_vision_mode = "TRACK_LOCAL"
        _set_slide_ref(core, yaw=0.01, dist=0.08, conf=0.9)
        core.handle_table_obs(TableEdgeObs(ts=now_ts(), table_found=True, edge_found=True, confidence=0.9, yaw_err_rad=0.01, dist_err_m=0.08))
        core.ctx.table_loss_since_mono = monotonic_ts() - 0.2
        decision = core.tick()
        self.assertEqual(core.ctx.state, State.FINAL_LOCK)
        self.assertEqual(decision.cmd.mode, "FINAL_LOCK")
        core.ctx.state = State.EDGE_SLIDE_SEARCH
        core.ctx.state_enter_mono = monotonic_ts() - 1.0
        core.ctx.table_loss_since_mono = monotonic_ts() - 0.2
        core.handle_table_obs(TableEdgeObs(ts=now_ts(), table_found=True, edge_found=True, confidence=0.9, yaw_err_rad=0.01, dist_err_m=0.08))
        decision = core.tick()
        self.assertEqual(core.ctx.state, State.ERROR_RECOVERY)
        self.assertEqual(decision.cmd.mode, "ERROR_RECOVERY")
        self.assertEqual(core.ctx.last_fail_reason, "edge_distance_out_of_tolerance_after_retries")

    def test_edge_slide_identity_mismatch_is_reported_before_fallback(self) -> None:
        cfg = OrchestratorConfig()
        cfg.control.edge_slide_pause_s = 0.0
        cfg.control.edge_follow_low_conf_exit_s = 3.0
        cfg.control.edge_slide_recover_timeout_s = 5.0
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.EDGE_SLIDE_SEARCH
        core.ctx.active_vision_mode = "TRACK_LOCAL"
        core.ctx.locked_yaw_err = 0.0
        core.ctx.locked_dist_err = 0.02
        core.ctx.locked_edge_conf = 0.80
        _set_slide_ref(core, yaw=0.0, dist=0.02, conf=0.45)
        core.ctx.state_enter_mono = monotonic_ts() - 1.0
        core.handle_table_obs(
            TableEdgeObs(
                ts=now_ts(),
                table_found=True,
                edge_found=True,
                edge_valid=True,
                confidence=0.45,
                yaw_err_rad=0.20,
                dist_err_m=0.02,
                source_mode="TRACK_LOCAL",
            )
        )
        decision = core.tick()
        self.assertEqual(core.ctx.state, State.EDGE_SLIDE_SEARCH)
        self.assertEqual(decision.cmd.mode, "EDGE_SLIDE_SEARCH")
        self.assertIn("edge_pause", decision.control_summary["reason"])
        self.assertEqual(decision.control_summary["stop_reason"], "edge_identity_mismatch")
        self.assertEqual(core.ctx.last_edge_quality.get("mode"), "pause")
        self.assertEqual(core.ctx.last_edge_quality.get("raw_mode"), "identity_mismatch")
        self.assertFalse(core.ctx.last_edge_quality.get("edge_identity_ok"))
        core.ctx.table_loss_since_mono = monotonic_ts() - float(cfg.control.edge_follow_low_conf_exit_s) - 0.1
        decision = core.tick()
        self.assertEqual(core.ctx.state, State.EDGE_SLIDE_SEARCH)
        self.assertEqual(decision.cmd.mode, "EDGE_SLIDE_SEARCH")
        self.assertIn("edge_recover", decision.control_summary["reason"])
        core.ctx.table_loss_since_mono = monotonic_ts() - float(cfg.control.edge_slide_recover_timeout_s) - 0.1
        decision = core.tick()
        self.assertEqual(core.ctx.state, State.FINAL_LOCK)
        self.assertEqual(decision.cmd.mode, "FINAL_LOCK")

    def test_track_local_without_table_edge_holds_with_reason(self) -> None:
        cfg = OrchestratorConfig()
        cfg.control.edge_slide_pause_s = 0.0
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.EDGE_SLIDE_SEARCH
        core.ctx.active_vision_mode = "TRACK_LOCAL"
        core.ctx.state_enter_mono = monotonic_ts() - 1.0
        decision = core.tick()
        self.assertEqual(core.ctx.state, State.SEARCH_TARGET_INIT)
        self.assertEqual(decision.cmd.mode, "SEARCH_TARGET_INIT")
        self.assertEqual(decision.cmd.vy_norm, 0.0)
        self.assertIn("stop", decision.control_summary["reason"])

    def test_edge_slide_rejects_stale_table_edge_obs(self) -> None:
        cfg = OrchestratorConfig()
        cfg.control.edge_slide_pause_s = 0.0
        cfg.control.table_edge_obs_max_age_ms = 500
        cfg.control.edge_follow_stale_fallback_state = "FINAL_LOCK"
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.EDGE_SLIDE_SEARCH
        core.ctx.active_vision_mode = "TRACK_LOCAL"
        core.ctx.state_enter_mono = monotonic_ts() - 1.0
        _set_slide_ref(core)
        stale_ts = now_ts() - 0.8
        core.handle_table_obs(
            TableEdgeObs(
                ts=stale_ts,
                obs_ts=stale_ts,
                age_ms=800.0,
                table_found=True,
                edge_found=True,
                confidence=0.9,
                yaw_err_rad=0.01,
                dist_err_m=0.02,
                frame_id=7,
                seq=7,
                source_mode="TRACK_LOCAL",
            )
        )
        decision = core.tick()
        self.assertEqual(core.ctx.state, State.EDGE_SLIDE_SEARCH)
        self.assertEqual(decision.cmd.mode, "EDGE_SLIDE_SEARCH")
        self.assertEqual(decision.cmd.vy_norm, 0.0)
        self.assertIn("edge_follow_stale_hold", decision.control_summary["reason"])
        core.ctx.table_loss_since_mono = monotonic_ts() - float(cfg.control.table_loss_hold_s) - 0.1
        decision = core.tick()
        self.assertEqual(core.ctx.state, State.FINAL_LOCK)
        self.assertEqual(decision.cmd.mode, "FINAL_LOCK")
        self.assertEqual(decision.cmd.vx_norm, 0.0)


if __name__ == "__main__":
    unittest.main()
