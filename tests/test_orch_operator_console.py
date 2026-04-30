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
    svc._pending_state_traces = []
    svc._last_uart_tx_ts = 0.0
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


class OrchestratorOperatorConsoleTest(unittest.TestCase):
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
        self.assertEqual(lines, ["[ORCH] STATE SEARCH_TABLE -> COARSE_ALIGN reason=table_edge_seen"])

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
            "cx_norm": 0.54,
            "cy_norm": 0.47,
        })
        self.assertIn("[ORCH] OBS target=apple found=1 matched_cls=apple matched_conf=0.81 best_cls=keyboard best_conf=0.92 cx=0.54 cy=0.47", lines[-1])

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
        self.assertIn("[ORCH] SLIDE edge=1 target=0 boxes=0 status=searching", lines[0])
        self.assertIn("cmd vx=+0.000 vy=+0.000 wz=+0.000", lines[0])
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
        self.assertIn("[ORCH] SLIDE edge=0 target=0", lines[0])
        self.assertIn("reason=safety_hold_no_edge", lines[0])

    def test_state_transition_edge_slide_timeout_is_diagnostic(self) -> None:
        lines = []
        svc = _service(lines, mode="operator")
        svc._on_state_transition("EDGE_SLIDE_SEARCH", "LEAVE_EDGE", "当前桌边未找到目标，切换到边 right")
        self.assertEqual(lines, ["[ORCH] STATE EDGE_SLIDE_SEARCH -> LEAVE_EDGE reason=target_not_found timeout_s=10.0"])

    def test_state_transition_target_found_is_diagnostic(self) -> None:
        lines = []
        svc = _service(lines, mode="operator")
        svc._on_state_transition("EDGE_SLIDE_SEARCH", "TARGET_CONFIRM", "检测到目标候选，开始确认")
        self.assertEqual(lines, ["[ORCH] STATE EDGE_SLIDE_SEARCH -> TARGET_CONFIRM reason=target_found"])

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
        self.assertEqual(line, "[ORCH] REQ target_search stage=SEARCH kind=TARGET target=apple mode_hint=TRACK_LOCAL req=req_1")

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
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.EDGE_SLIDE_SEARCH
        core.ctx.active_target = "apple"
        core.ctx.task_start_wall_ts = 1.0
        core.ctx.state_enter_mono = monotonic_ts() - 1.0
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
        self.assertEqual(core.ctx.state, State.EDGE_SLIDE_SEARCH)
        self.assertEqual(decision.cmd.mode, "TARGET_CONFIRM")

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

    def test_edge_slide_config_nonzero_outputs_nonzero_vy(self) -> None:
        cfg = OrchestratorConfig()
        cfg.car.edge_slide_vy_norm = 0.04
        cfg.control.edge_slide_pause_s = 0.0
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.EDGE_SLIDE_SEARCH
        core.ctx.active_vision_mode = "TRACK_LOCAL"
        core.ctx.state_enter_mono = monotonic_ts() - 1.0
        core.handle_table_obs(TableEdgeObs(ts=now_ts(), table_found=True, edge_found=True, confidence=0.9, yaw_err_rad=0.01, dist_err_m=0.02))
        decision = core.tick()
        self.assertEqual(decision.cmd.mode, "EDGE_SLIDE_SEARCH")
        self.assertAlmostEqual(decision.cmd.vy_norm, 0.04, places=6)
        self.assertNotEqual(decision.cmd.vx_norm, 0.0)

    def test_edge_slide_distance_out_of_tolerance_recovers_approach(self) -> None:
        cfg = OrchestratorConfig()
        cfg.control.edge_slide_pause_s = 0.0
        cfg.control.edge_slide_dist_tolerance_m = 0.05
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.EDGE_SLIDE_SEARCH
        core.ctx.active_vision_mode = "TRACK_LOCAL"
        core.ctx.state_enter_mono = monotonic_ts() - 1.0
        core.handle_table_obs(TableEdgeObs(ts=now_ts(), table_found=True, edge_found=True, confidence=0.9, yaw_err_rad=0.01, dist_err_m=0.08))
        decision = core.tick()
        self.assertEqual(core.ctx.state, State.EDGE_SLIDE_SEARCH)
        self.assertEqual(decision.cmd.mode, "EDGE_SLIDE_SEARCH")
        core.ctx.table_loss_since_mono = monotonic_ts() - float(cfg.control.table_loss_hold_s) - 0.1
        decision = core.tick()
        self.assertEqual(core.ctx.state, State.CONTROLLED_APPROACH)
        self.assertEqual(decision.cmd.mode, "CONTROLLED_APPROACH")

    def test_track_local_without_table_edge_holds_with_reason(self) -> None:
        cfg = OrchestratorConfig()
        cfg.control.edge_slide_pause_s = 0.0
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.EDGE_SLIDE_SEARCH
        core.ctx.active_vision_mode = "TRACK_LOCAL"
        core.ctx.state_enter_mono = monotonic_ts() - 1.0
        decision = core.tick()
        self.assertEqual(decision.cmd.mode, "EDGE_SLIDE_SEARCH")
        self.assertEqual(decision.cmd.vy_norm, 0.0)
        self.assertIn("edge_obs_missing_hold", decision.control_summary["reason"])

    def test_edge_slide_rejects_stale_table_edge_obs(self) -> None:
        cfg = OrchestratorConfig()
        cfg.control.edge_slide_pause_s = 0.0
        cfg.control.table_edge_obs_max_age_ms = 500
        cfg.control.edge_follow_stale_fallback_state = "FINAL_LOCK"
        core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
        core.ctx.state = State.EDGE_SLIDE_SEARCH
        core.ctx.active_vision_mode = "TRACK_LOCAL"
        core.ctx.state_enter_mono = monotonic_ts() - 1.0
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
