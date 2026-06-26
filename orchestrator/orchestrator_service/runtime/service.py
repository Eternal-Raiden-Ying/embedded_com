#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import queue
import re
import signal
import time
import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from ..bridge.arm_protocol import encode_pose, parse_arm_response
from common.console_presenter import DemoConsolePresenter
from common.base_module import BaseModule
from common.runtime_logging import OperatorConsole
from common.system_metrics import SystemMetricsSampler
from ..bridge.simple_car_protocol import parse_car_state_line
from ..bridge.uart_bridge import UartBridge
from ..config.schema import OrchestratorConfig, SocketEndpoint
from ..control.motion_adapter import Stm32MotionAdapter
from ..control.motion.velocity_limits import SimpleCarMapper
from ..ipc.protocol import (
    HomeTagObs,
    TableEdgeObs,
    TargetObs,
    TaskCmd,
    CmdVel,
    VisionObsEnvelope,
    iter_vision_perception_payloads,
    make_task_ack,
)
from ..ipc.transport import AsyncJsonlClientSender, JsonlClientSender, JsonlInboundServer
from .common import RunLogger, ensure_dir, safe_dump
from .state_machine import OrchestratorCore


class OrchestratorService(BaseModule):
    def __init__(self, cfg: OrchestratorConfig):
        self.cfg = cfg
        super().__init__("orch", cfg.runtime.log_enabled, cfg.runtime.log_mode)
        ensure_dir(cfg.runtime.log_dir)
        ensure_dir(cfg.runtime.runs_dir)
        ensure_dir(cfg.runtime.pid_dir)
        self.run_logger = RunLogger("orch", cfg.runtime.runs_dir, cfg.runtime.stack_run_id)
        self.core = OrchestratorCore(cfg.control, cfg.car, cfg.docking, logger=self.log)
        self.core.transition_observer = self._on_state_transition
        self.operator_console = OperatorConsole(
            mode=os.getenv("ORCH_CONSOLE_MODE", "operator"),
            default_interval_s=self._env_float("ORCH_OPERATOR_SUMMARY_INTERVAL_S", 1.0),
        )
        self.demo_console = DemoConsolePresenter(self.operator_console, dry_run=cfg.serial.dry_run)
        self._ipc_console_enabled = self._env_bool("ORCH_IPC_CONSOLE", False)
        self._heartbeat_console_enabled = self._env_bool("ORCH_HEARTBEAT_CONSOLE", False)
        self._uart_console_mode = self._env_choice("ORCH_UART_CONSOLE", "operator", {"operator", "full", "silent"})
        self._uart_full_log = self._env_bool("ORCH_UART_FULL_LOG", False)
        self._vision_full_obs_log = self._env_bool("VISION_LOG_FULL_OBS", False) or self._env_bool("VISION_DEBUG_FULL_LOG", False)
        self._state_blocks_full_log = self._env_bool("ORCH_STATE_BLOCKS_FULL_LOG", False) or self._env_bool("VISION_DEBUG_FULL_LOG", False)
        self._system_metrics = SystemMetricsSampler("orchestrator", interval_s=self._env_float("ORCH_SYSTEM_METRICS_INTERVAL_S", 1.0))
        self._mobile_status_console_mode = self._env_choice("ORCH_MOBILE_STATUS_CONSOLE", "change", {"change", "full", "silent"})
        self._last_obs_flags = {"table_edge": False, "target": False}
        self._last_target_obs_console_payload: Dict[str, Any] = {}
        self._last_target_search_req_console_key = ""
        self._demo_start_pending_target = ""
        self._demo_deferred_phases: List[tuple] = []
        self._pending_state_traces: List[Dict[str, Any]] = []
        self._last_edge_slide_trace_ts = 0.0
        self._last_edge_slide_obs_key = None
        self._last_edge_slide_obs_ts = None
        self._last_edge_slide_obs_period_ms = None
        self.mapper = SimpleCarMapper(cfg.car)
        self._async_result_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._boot_ts = time.time()
        self._last_state_block_ts = 0.0
        self._last_state_block_key = ""
        self._last_heartbeat_ts = 0.0
        self._last_task_cmd_recv_ts = 0.0
        self._last_vision_obs_recv_ts = 0.0
        self._last_diagnostic_obs_metrics: Dict[str, Any] = {}
        self._last_home_obs_recv_ts = 0.0
        self._last_uart_tx_ts = 0.0
        self._edge_obs_rate_ts = deque(maxlen=128)
        self._target_obs_rate_ts = deque(maxlen=128)
        self._obs_freq_last_emit_ts = 0.0
        self._obs_trace_samples: Dict[str, deque] = {
            "table_edge_obs_recv_interval_ms": deque(maxlen=512),
            "state_machine_tick_interval_ms": deque(maxlen=512),
            "state_machine_consume_interval_ms": deque(maxlen=512),
            "obs_age_at_consume_ms": deque(maxlen=512),
            "vision_publish_to_orch_recv_ms": deque(maxlen=512),
            "orch_recv_to_state_consume_ms": deque(maxlen=512),
        }
        self._last_table_edge_recv_ts = 0.0
        self._last_state_machine_tick_ts = 0.0
        self._last_state_machine_consume_ts = 0.0
        self._last_consumed_table_obs_key = None
        self._last_consumed_obs_seq: Optional[int] = None
        self._same_obs_reuse_count = 0
        self._last_tx_summary: Dict[str, float] = {
            "task_ack_out": 0.0,
            "vision_req_out": 0.0,
            "tts_event_out": 0.0,
        }
        self._uart_console_key = ""
        self._uart_console_last_emit_ts = 0.0
        self._uart_console_repeat_count = 0
        self._uart_console_last_payload: Optional[Dict[str, Any]] = None
        self._uart_lowfreq_key = ""
        self._uart_lowfreq_last_emit_ts = 0.0
        self._uart_lowfreq_repeat_count = 0
        self._uart_lowfreq_last_payload: Optional[Dict[str, Any]] = None
        self._uart_keepalive_last_tx_ts = 0.0
        self._uart_keepalive_last_summary_ts = self._boot_ts
        self._uart_keepalive_tx_ts = deque(maxlen=512)
        self._uart_keepalive_interval_samples = deque(maxlen=512)
        self._uart_keepalive_last_key = ""
        self._uart_keepalive_same_cmd_repeat_count = 0
        self._uart_keepalive_tx_count = 0
        self._uart_keepalive_last_interval_ms: Optional[float] = None
        self._uart_keepalive_last_cmd_type = ""
        self._last_motion_log_ts = 0.0
        self._last_motion_log_signature = None
        self._last_motion_adapter_log_key = ""
        self._last_motion_adapter_log_emit_ts = 0.0
        self._last_motion_tx_context: Dict[str, Any] = {}
        self._last_valid_motion_cmd: Optional[Dict[str, Any]] = None
        self._last_valid_motion_ts = 0.0
        self._last_edge_slide_trace_signature = None
        self._motion_seq = 0
        self._last_stm32_status_poll_ts = 0.0
        self.motion_status: Dict[str, Any] = {
            "last_seq": None,
            "last_ack_seq": None,
            "last_done_seq": None,
            "jog_running": False,
            "stm32_timeout_seen": False,
            "last_rx_time": None,
        }
        self._stopped = False
        self.uart = UartBridge(
            cfg.serial.port,
            cfg.serial.baudrate,
            cfg.serial.timeout_s,
            dry_run=cfg.serial.dry_run,
            readback_enabled=cfg.serial.readback_enabled,
            dry_run_echo_stdout=cfg.serial.dry_run_echo_stdout,
            tx_callback=self._on_uart_tx,
            logger=self.log,
        )
        self.motion_adapter = Stm32MotionAdapter(
            self.uart,
            logger=self._motion_operator_emit,
            tx_meta_factory=self._build_stm32_motion_tx_meta,
            wheel_speed_limit=cfg.car.stm32_wheel_speed_limit,
            max_vx_mps=cfg.car.max_vx_mps,
            max_vy_mps=cfg.car.max_vy_mps,
            max_wz_radps=cfg.car.max_wz_radps,
            jog_forward_speed=cfg.car.jog_forward_speed,
            jog_turn_speed=cfg.car.jog_turn_speed,
            jog_duration_ms=cfg.car.jog_duration_ms,
        )
        self.task_server = self._build_server(cfg.task_cmd_in, "task_cmd_in")
        self.vision_server = self._build_server(cfg.vision_obs_in, "vision_obs_in")
        self.task_ack_sender = self._build_sender(cfg.task_ack_out, "task_ack_out", async_allowed=False)
        self.vision_req_sender = self._build_sender(cfg.vision_req_out, "vision_req_out", async_allowed=True)
        self.tts_sender = self._build_sender(cfg.tts_event_out, "tts_event_out", async_allowed=True)
        self._running = False

    @staticmethod
    def _lite_table_edge_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
        roi = obs.get("edge_roi") or obs.get("table_edge_roi") or obs.get("depth_edge_roi") or obs.get("plane_roi")
        keys = (
            "ts",
            "obs_seq",
            "camera_frame_seq",
            "camera_frame_ts_ms",
            "vision_process_start_ts_ms",
            "vision_process_end_ts_ms",
            "vision_publish_ts_ms",
            "obs_out_send_ts_ms",
            "orchestrator_recv_ts_ms",
            "state_machine_consume_ts_ms",
            "cmd_publish_ts_ms",
            "seq",
            "frame_id",
            "edge_found",
            "edge_geometry_valid",
            "edge_trusted",
            "yaw_err_rad",
            "dist_err_m",
            "edge_conf",
            "obs_total_age_ms",
            "vision_process_ms",
            "camera_frame_interval_ms",
            "camera_frame_hz",
            "vision_process_interval_ms",
            "vision_publish_interval_ms",
            "obs_out_send_interval_ms",
            "obs_out_send_hz",
            "table_edge_obs_recv_interval_ms",
            "orchestrator_recv_interval_ms",
            "table_edge_obs_recv_hz",
            "state_machine_tick_interval_ms",
            "state_machine_consume_interval_ms",
            "same_obs_reuse_count",
            "obs_seq_gap",
            "obs_age_at_consume_ms",
            "vision_publish_to_orch_recv_ms",
            "orch_recv_to_state_consume_ms",
            "edge_update_interval_ms",
            "depth_shape",
            "calib_source",
            "reject_reason",
            "source",
            "detector_mode",
        )
        out = {key: obs.get(key) for key in keys if obs.get(key) is not None}
        if roi is not None:
            out["roi"] = roi
        out["type"] = "table_edge_obs"
        return out

    @classmethod
    def _lite_vision_obs(cls, env: VisionObsEnvelope) -> Dict[str, Any]:
        perception = dict(env.perception or {})
        edge = perception.get("table_edge_obs") if isinstance(perception.get("table_edge_obs"), dict) else None
        target = perception.get("target_obs") if isinstance(perception.get("target_obs"), dict) else None
        lite_perception: Dict[str, Any] = {}
        if isinstance(edge, dict):
            lite_perception["table_edge_obs"] = cls._lite_table_edge_obs(edge)
        if isinstance(target, dict):
            lite_perception["target_obs"] = {
                key: target.get(key)
                for key in ("ts", "seq", "frame_id", "found", "target_found", "target", "confidence", "best_cls", "matched_cls")
                if target.get(key) is not None
            }
        return {
            "ts": env.ts,
            "stage": env.stage,
            "mode": env.mode,
            "status": env.status,
            "session_id": env.session_id,
            "req_id": env.req_id,
            "epoch": int(env.epoch),
            "obs_class": env.obs_class,
            "source": env.source,
            "perception": lite_perception,
            "type": "vision_obs",
        }

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return bool(default)
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)) or default)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _env_choice(name: str, default: str, choices: set) -> str:
        value = str(os.getenv(name, default) or default).strip().lower()
        return value if value in choices else default

    @staticmethod
    def _short_err(value: Any, limit: int = 96) -> str:
        text = " ".join(str(value or "").strip().split())
        return text[:limit] if len(text) <= limit else text[: max(0, limit - 3)] + "..."

    @staticmethod
    def _fmt_float(value: Any, digits: int = 3, signed: bool = True) -> str:
        if value is None:
            return "n/a"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "n/a"
        sign = "+" if signed else ""
        return f"{number:{sign}.{digits}f}"

    def _target_obs_console_line(self, payload: Dict[str, Any]) -> str:
        target = str(payload.get("target") or self.core.ctx.active_target or "").strip() or "target"
        found = bool(payload.get("target_found", payload.get("found", False)))
        matched_cls = str(payload.get("matched_cls") or payload.get("target_cls") or "").strip() or "n/a"
        matched_conf = payload.get("matched_conf", payload.get("confidence", payload.get("score")))
        best_cls = str(payload.get("best_cls") or payload.get("best_class") or "").strip() or "n/a"
        best_conf = payload.get("best_conf", payload.get("best_confidence"))
        if found:
            full_center = payload.get("matched_center_full_norm")
            if not isinstance(full_center, dict):
                full_center = payload.get("matched_center") if isinstance(payload.get("matched_center"), dict) else {}
            return (
                f"[ORCH] OBS target={target} found=1 "
                f"matched_cls={matched_cls} matched_conf={self._fmt_float(matched_conf, 2, signed=False)} "
                f"best_cls={best_cls} best_conf={self._fmt_float(best_conf, 2, signed=False)} "
                f"cx={self._fmt_float(full_center.get('cx', full_center.get('x_norm', payload.get('x_norm', payload.get('cx_norm', payload.get('cx'))))), 2, signed=False)} "
                f"cy={self._fmt_float(full_center.get('cy', full_center.get('y_norm', payload.get('y_norm', payload.get('cy_norm', payload.get('cy'))))), 2, signed=False)}"
            )
        boxes = payload.get("boxes_count", payload.get("box_count", payload.get("boxes", 0)))
        if isinstance(boxes, list):
            boxes = len(boxes)
        mode = str(payload.get("vision_mode") or payload.get("mode") or self.core.ctx.confirmed_vision_mode or "").strip() or "n/a"
        return f"[ORCH] OBS target={target} found=0 boxes={int(boxes or 0)} mode={mode}"

    def _vision_req_console_summary(self, payload: Dict[str, Any]) -> Optional[str]:
        if str(payload.get("type") or "").strip() != "vision_req":
            return None
        req_payload = dict(payload.get("payload") or {})
        kind = str(req_payload.get("search_kind") or payload.get("kind") or "").strip().upper()
        stage = str(payload.get("stage") or "").strip().upper()
        mode_hint = str(payload.get("mode_hint") or "").strip().upper()
        req_type = str(payload.get("req_type") or req_payload.get("req_type") or "target_update").strip().lower()
        if stage != "SEARCH" or kind != "TARGET":
            return None
        target = str(payload.get("target") or self.core.ctx.active_target or "target").strip() or "target"
        req_id = str(payload.get("req_id") or "").strip()
        return (
            f"[ORCH] REQ {req_type} stage={stage} kind={kind} "
            f"target={target} mode_hint={mode_hint or 'n/a'} req={req_id}"
        )

    def _emit_target_obs_console(self, payload: Dict[str, Any]) -> None:
        state = str(getattr(self.core.ctx.state, "value", self.core.ctx.state) or "")
        if state not in {"SEARCH_TARGET_INIT", "EDGE_SLIDE_SEARCH"}:
            return
        self._last_target_obs_console_payload = dict(payload or {})
        line = self._target_obs_console_line(payload)
        self.operator_console.emit_rate_limited("target_obs", line, self.operator_console.default_interval_s)

    def _operator_emit(self, line: str) -> bool:
        return self.operator_console.emit(line)

    def _motion_operator_emit(self, line: str) -> bool:
        text = str(line or "")
        if not text.startswith("[MOTION]"):
            return self._operator_emit(text)
        key = re.sub(r"\bseq=\d+\b", "seq=*", text)
        now = time.time()
        if key == self._last_motion_adapter_log_key and (now - self._last_motion_adapter_log_emit_ts) < 5.0:
            return False
        self._last_motion_adapter_log_key = key
        self._last_motion_adapter_log_emit_ts = now
        return self._operator_emit(text)

    def _emit_demo_dry_run_notice(self, key: str) -> None:
        self.demo_console.dry_run = bool(getattr(self.cfg.serial, "dry_run", False))
        self.demo_console.dry_run_notice()

    def _demo_next_state(self) -> str:
        return str(os.getenv("ORCH_DEMO_NEXT_STATE", "IDLE_HOT") or "IDLE_HOT").strip().upper()

    def _demo_preview_text(self) -> str:
        return str(os.getenv("ORCH_DEMO_PREVIEW_TEXT", "kept alive") or "kept alive").strip()

    def _demo_reason(self, reason: str, *, failed: bool = False) -> str:
        text = " ".join(str(reason or "").strip().split())
        lowered = text.lower()
        if "edge_distance_out_of_tolerance_after_retries" in lowered:
            return "edge_distance_out_of_tolerance_after_retries"
        if "已在桌边锁定" in text:
            return "target_locked"
        if "已返回起点" in text:
            return "returned_home"
        if "当前桌边未找到目标" in text or "target_not_found" in lowered:
            return "target_not_found"
        if "搜索桌边超时" in text or "table_not_found" in lowered:
            return "table_not_found"
        if "timeout" in lowered or "超时" in text:
            return "timeout"
        if not text:
            return "failed" if failed else "task_done"
        return text

    def _emit_demo_task_finished(self, *, success: bool, reason: str) -> None:
        self._emit_demo_dry_run_notice("task_end")
        target = getattr(self.core.ctx, "active_target", "") or "n/a"
        if success:
            self.demo_console.success_banner(
                target=target,
                next_state=self._demo_next_state(),
                preview=self._demo_preview_text(),
            )
        else:
            self.demo_console.failed_banner(
                target=target,
                reason=self._demo_reason(reason, failed=True),
                next_state=self._demo_next_state(),
            )

    def _emit_demo_idle_hot(self) -> None:
        self.demo_console.idle_hot(next_state=self._demo_next_state(), preview=self._demo_preview_text())

    def _endpoint_for_channel(self, channel: str) -> Optional[SocketEndpoint]:
        return {
            "task_cmd_in": self.cfg.task_cmd_in,
            "vision_obs_in": self.cfg.vision_obs_in,
            "task_ack_out": self.cfg.task_ack_out,
            "vision_req_out": self.cfg.vision_req_out,
            "tts_event_out": self.cfg.tts_event_out,
        }.get(str(channel or ""))

    def _endpoint_parts(self, channel: str) -> List[str]:
        ep = self._endpoint_for_channel(channel)
        if ep is None:
            return []
        if ep.transport == "tcp":
            return [f"host={ep.host}", f"port={ep.port}"]
        if ep.transport == "uds":
            return [f"path={ep.ipc_socket_path}"]
        return [f"transport={ep.transport}"]

    def _operator_ipc_line(self, channel: str, event: str, details: Dict[str, Any]) -> str:
        level = "ERROR" if event in {"send_failed", "invalid_json"} else ("WARN" if ("failed" in event or "closed" in event) and event != "peer_closed_empty" else "IPC")
        parts = [f"[ORCH] {level} {channel} {event}"]
        parts.extend(self._endpoint_parts(channel))
        if details.get("peer"):
            parts.append(f"peer={details.get('peer')}")
        if details.get("error"):
            parts.append(f"err={self._short_err(details.get('error'))}")
        if details.get("fail_count") is not None:
            parts.append(f"retry={details.get('fail_count')}")
        if details.get("owner") is not None:
            parts.append(f"owner={details.get('owner')}")
        if details.get("perm") is not None:
            parts.append(f"perm={details.get('perm')}")
        return " ".join(parts)

    def _operator_ipc_event(self, channel: str, event: str, details: Dict[str, Any]) -> None:
        success_events = {"send_ok", "send_attempt", "async_enqueue", "enqueue_ok", "received", "envelope_received", "ack_sent"}
        connectivity_events = {
            "connected",
            "listening",
            "peer_connected",
            "peer_closed",
            "connect_failed",
            "send_failed",
            "ack_send_failed",
            "enqueue_failed",
            "async_queue_full_drop_new",
            "async_queue_full_drop_oldest",
            "async_queue_full_retry_failed",
            "invalid_json",
            "bad_payload",
        }
        if self.operator_console.full or self._ipc_console_enabled:
            if event in success_events or event in connectivity_events:
                self.operator_console.emit_rate_limited(f"ipc:{channel}:{event}", self._operator_ipc_line(channel, event, details), 0.2)
            return
        if event not in connectivity_events:
            return
        line = self._operator_ipc_line(channel, event, details)
        if event in {"connect_failed", "send_failed", "ack_send_failed", "invalid_json", "bad_payload"}:
            self.operator_console.emit_error(f"ipc:{channel}:{event}:{details.get('error', '')}", line)
        else:
            self.operator_console.emit_change(f"ipc:{channel}:{event}", line)

    def _on_state_transition(self, old_state: str, new_state: str, reason: str) -> None:
        reason = str(reason or "state_transition").strip() or "state_transition"
        if old_state == "EDGE_SLIDE_SEARCH" and new_state in {"LEAVE_EDGE", "NEXT_TABLE"} and "未找到目标" in reason:
            reason = f"target_not_found timeout_s={float(self.cfg.control.target_search_timeout_s):.1f}"
        if old_state == "EDGE_SLIDE_SEARCH" and new_state == "TARGET_CONFIRM" and "matched_cls=" not in reason:
            reason = "target_found"
        trace = dict(getattr(self.core, "last_transition_snapshot", {}) or {})
        if trace:
            trace["reason"] = reason
            trace.setdefault("previous_state", old_state)
            trace.setdefault("next_state", new_state)
            trace.setdefault("planned_cmd", {"vx": None, "vy": None, "wz": None})
            self._pending_state_traces.append(trace)
        self.operator_console.emit_change(
            "state",
            f"[ORCH] STATE {old_state} -> {new_state} reason={reason}",
        )
        if self._demo_start_pending_target and new_state in {"SEARCH_TABLE", "RETURN_HOME"}:
            self._demo_deferred_phases.append((old_state, new_state, reason))
        else:
            self._emit_demo_phase(old_state, new_state, reason)
        if new_state == "DONE":
            self.operator_console.emit_change("task_done", self._task_done_summary_line(reason))
            self._emit_demo_task_finished(success=True, reason=reason)
        if new_state == "ERROR_RECOVERY":
            self.log("warn", "service", f"Immediate transition to ERROR_RECOVERY due to: {reason}, triggering emergency stop")
            self.uart.send_emergency_stop()
            self.motion_adapter.cancel_active_jogs()
            self._emit_demo_task_finished(success=False, reason=reason)
        if old_state == "DONE" and new_state == "IDLE":
            self.operator_console.emit_change("idle_after_done", "[ORCH][IDLE] task finished, waiting for next command")
            self._emit_demo_idle_hot()
        if old_state == "ERROR_RECOVERY" and new_state == "IDLE":
            self._emit_demo_idle_hot()

    def _emit_demo_phase(self, old_state: str, new_state: str, reason: str) -> None:
        target = getattr(self.core.ctx, "active_target", "") or "target"
        if new_state == "SEARCH_TABLE":
            self.demo_console.table_phase("searching")
        elif new_state == "EDGE_ADJUST":
            phase = getattr(self.core.ctx, "table_dock_phase", "aligning")
            if phase == "aligning":
                self.demo_console.table_phase("aligning")
            else:
                self.demo_console.table_phase("approaching")
        elif new_state == "FINAL_SLOW_STOP":
            self.demo_console.table_phase("final_locking")
            if "edge_distance_out_of_tolerance" in str(reason or ""):
                self.demo_console.recover("edge distance out of range, re-locking table edge")
        elif new_state in {"AT_TABLE_EDGE", "SEARCH_TARGET_INIT"}:
            self.demo_console.table_phase("locked")
        elif new_state == "EDGE_SLIDE_SEARCH":
            self.demo_console.target_phase("searching", target)
        elif new_state == "TARGET_CONFIRM":
            self.demo_console.target_phase("candidate", target)
        elif new_state in {"TARGET_LOCKED", "FREEZE_BASE"}:
            self.demo_console.target_phase("locked", target)
        elif old_state == "EDGE_SLIDE_SEARCH" and new_state in {"LEAVE_EDGE", "NEXT_TABLE"}:
            self.demo_console.target_phase("relocating", target)

    def _flush_demo_deferred_phases(self) -> None:
        pending = list(getattr(self, "_demo_deferred_phases", []) or [])
        self._demo_deferred_phases = []
        for old_state, new_state, reason in pending:
            self._emit_demo_phase(old_state, new_state, reason)

    def _task_done_summary_line(self, reason: str) -> str:
        ctx = self.core.ctx
        target_obs = getattr(ctx, "last_target_obs", None)
        edge_obs = getattr(ctx, "last_table_obs", None)
        warnings = list(getattr(ctx, "task_warning_history", []) or [])
        if getattr(ctx, "last_fail_reason", ""):
            last_fail = str(ctx.last_fail_reason).strip()
            if last_fail and last_fail not in warnings:
                warnings.append(last_fail)
        try:
            state_value = ctx.state.value
        except Exception:
            state_value = "DONE"
        result = "success" if state_value == "DONE" else "failed"
        total_time_s = 0.0
        if getattr(ctx, "task_start_wall_ts", 0.0):
            total_time_s = max(0.0, time.time() - float(ctx.task_start_wall_ts))
        matched_cls = getattr(target_obs, "matched_cls", None) or getattr(target_obs, "target", None)
        matched_conf = getattr(target_obs, "matched_conf", None)
        edge_conf = getattr(edge_obs, "confidence", None) if edge_obs is not None else None
        return (
            "[ORCH][TASK_DONE] "
            f"session_id={getattr(ctx, 'active_session_id', '') or 'n/a'} "
            f"target={getattr(ctx, 'active_target', '') or 'n/a'} "
            f"result={result} "
            f"final_state={state_value} "
            f"reason={reason or 'task_done'} "
            f"total_time_s={total_time_s:.1f} "
            f"edge_retries={int(getattr(ctx, 'dock_retry_count', 0) + getattr(ctx, 'edge_transition_count', 0))} "
            f"slide_entries={int(getattr(ctx, 'task_slide_entries_count', 0))} "
            f"target_confirm_count={int(getattr(ctx, 'task_target_confirm_count', 0))} "
            f"target_locked_count={int(getattr(ctx, 'task_target_locked_count', 0))} "
            f"last_matched_cls={matched_cls or 'n/a'} "
            f"last_matched_conf={self._fmt_float(matched_conf, 2, signed=False)} "
            f"last_edge_conf={self._fmt_float(edge_conf, 2, signed=False)} "
            f"warnings={warnings}"
        )

    @staticmethod
    def _motion_float_sig(value: Any) -> float:
        try:
            return round(float(value), 4)
        except (TypeError, ValueError):
            return 0.0

    def _motion_log_period_s(self) -> float:
        try:
            period_ms = float(os.getenv("ORCH_MOTION_LOG_PERIOD_MS", "500") or 500.0)
        except (TypeError, ValueError):
            period_ms = 500.0
        return max(0.0, period_ms / 1000.0)

    def _motion_log_signature(self, cmd, car_cmd, reason: str) -> tuple:
        cmd_payload = cmd.to_dict() if hasattr(cmd, "to_dict") else {}
        return (
            str(getattr(car_cmd, "mode", cmd_payload.get("mode", "")) or ""),
            str(getattr(car_cmd, "kind", cmd_payload.get("kind", "")) or ""),
            self._motion_float_sig(getattr(cmd, "vx_mps", cmd_payload.get("vx", cmd_payload.get("vx_mps", 0.0)))),
            self._motion_float_sig(getattr(cmd, "vy_mps", cmd_payload.get("vy", cmd_payload.get("vy_mps", 0.0)))),
            self._motion_float_sig(getattr(cmd, "wz_radps", cmd_payload.get("wz", cmd_payload.get("wz_radps", 0.0)))),
            bool(getattr(cmd, "stop", cmd_payload.get("stop", False))),
            bool(getattr(car_cmd, "brake", False)),
            str(reason or ""),
        )

    def _should_log_motion(self, signature: tuple, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else float(now)
        changed = signature != getattr(self, "_last_motion_log_signature", None)
        period_s = self._motion_log_period_s()
        periodic = (now - float(getattr(self, "_last_motion_log_ts", 0.0) or 0.0)) >= period_s
        if not changed and not periodic:
            return False
        self._last_motion_log_signature = signature
        self._last_motion_log_ts = now
        return True

    def _verbose_ipc_success_enabled(self) -> bool:
        return (
            self.operator_console.full
            or str(getattr(self.cfg.runtime, "log_mode", "") or "").strip().lower() == "full"
            or bool(getattr(self.cfg.runtime, "debug", False))
            or self._env_bool("ORCH_IPC_VERBOSE_SUCCESS", False)
        )

    def _log_json(self, payload):
        level = payload.get("level", "info")
        event = payload.get("event", "")
        name = payload.get("name", payload.get("src", "module"))
        extra = {k: v for k, v in payload.items() if k not in {"level", "msg"}}
        message = event or payload.get("msg", "")
        if payload.get("src") == "ipc":
            success_events = {"send_ok", "send_attempt", "async_enqueue", "enqueue_ok", "received", "envelope_received", "ack_sent"}
            if str(event or "").strip().lower() in success_events and not self._verbose_ipc_success_enabled():
                return
        log_level = level
        if payload.get("src") == "ipc" and not self.operator_console.full:
            log_level = "info"
        self.log(log_level, "ipc", f"{name} {message}".strip(), extra or None)
        if payload.get("src") == "ipc":
            ipc_fields = {k: v for k, v in payload.items() if k not in {"level", "src", "name", "event", "msg"}}
            self.run_logger.write_ipc(name, event or "log", direction=self._ipc_direction_for(name), **ipc_fields)
            self._operator_ipc_event(str(name), str(event or "log"), ipc_fields)

    def _ipc_direction_for(self, channel: str) -> str:
        return "RX" if str(channel).endswith("_in") else "TX"

    def _build_server(self, ep: SocketEndpoint, name: str):
        return JsonlInboundServer(mode=ep.transport, tcp_host=ep.host, tcp_port=ep.port, uds_path=ep.uds_path, name=name, logger=self._log_json)

    def _make_result_callback(self, name: str):
        def _cb(result: Dict[str, Any]):
            result = dict(result)
            result["channel"] = name
            self._async_result_queue.put(result)
        return _cb

    def _build_sender(self, ep: SocketEndpoint, name: str, async_allowed: bool = True):
        inner = JsonlClientSender(
            mode=ep.transport,
            tcp_host=ep.host,
            tcp_port=ep.port,
            uds_path=ep.uds_path,
            name=name,
            logger=self._log_json,
            send_mode=ep.send_mode,
        )
        if async_allowed and getattr(ep, "async_enabled", False) and ep.transport != "disabled":
            sender = AsyncJsonlClientSender(
                inner,
                queue_size=getattr(ep, "async_queue_size", 64),
                drop_oldest=getattr(ep, "async_drop_oldest", True),
                logger=self._log_json,
                result_callback=self._make_result_callback(name),
            )
            sender.start()
            return sender
        return inner

    def _sender_snapshot(self, sender: Any) -> Dict[str, Any]:
        try:
            return sender.snapshot()
        except Exception:
            return {"name": getattr(sender, "name", "sender"), "link_state": "UNKNOWN"}

    def _config_dump(self) -> Dict[str, Any]:
        return {
            "stack_run_id": self.run_logger.stack_run_id,
            "log_dir": self.cfg.runtime.log_dir,
            "log_file": self.cfg.runtime.log_file,
            "tick_hz": self.cfg.runtime.tick_hz,
            "state_machine_tick_hz": self.cfg.runtime.tick_hz,
            "orchestrator_table_edge_receiver_poll_interval_ms": int(round((1.0 / max(1.0, float(self.cfg.runtime.tick_hz))) * 1000.0)),
            "status_publish_hz": 1.0 / max(1e-6, float(self.cfg.runtime.state_block_period_s or 1.0)),
            "heartbeat_period_s": self.cfg.runtime.heartbeat_period_s,
            "console_mode": self.operator_console.mode,
            "operator_summary_interval_s": self.operator_console.default_interval_s,
            "ipc_console": self._ipc_console_enabled,
            "heartbeat_console": self._heartbeat_console_enabled,
            "uart_console": self._uart_console_mode,
            "mobile_status_console": self._mobile_status_console_mode,
            "runs_dir": self.cfg.runtime.runs_dir,
            "pid_dir": self.cfg.runtime.pid_dir,
            "pid_file": self.cfg.runtime.pid_file,
            "config_files": {
                "stage_params": self.cfg.runtime.stage_params_file,
                "car_cmd_params": self.cfg.runtime.car_cmd_params_file,
                "loaded": list(self.cfg.runtime.loaded_config_files),
            },
            "serial": {
                "port": self.cfg.serial.port,
                "baudrate": self.cfg.serial.baudrate,
                "dry_run": self.cfg.serial.dry_run,
                "readback_enabled": self.cfg.serial.readback_enabled,
                "dry_run_echo_stdout": self.cfg.serial.dry_run_echo_stdout,
                "dry_run_echo_on_change_only": self.cfg.serial.dry_run_echo_on_change_only,
                "dry_run_echo_summary_period_s": self.cfg.serial.dry_run_echo_summary_period_s,
                "dry_run_quiet_idle_stop": self.cfg.serial.dry_run_quiet_idle_stop,
                "uart_lowfreq_period_s": self.cfg.serial.uart_lowfreq_period_s,
                "stm32_status_enabled": self.cfg.serial.stm32_status_enabled,
                "stm32_status_period_s": self.cfg.serial.stm32_status_period_s,
            },
            "control": {
                "search_table_timeout_s": self.cfg.control.search_table_timeout_s,
                "approach_timeout_s": self.cfg.control.approach_timeout_s,
                "target_search_timeout_s": self.cfg.control.target_search_timeout_s,
                "final_lock_yaw_tol_rad": self.cfg.control.final_lock_yaw_tol_rad,
                "final_lock_dist_tol_m": self.cfg.control.final_lock_dist_tol_m,
                "final_lock_frames_to_arrive": self.cfg.control.final_lock_frames_to_arrive,
                "enable_final_lock": self.cfg.control.enable_final_lock,
                "enable_micro_adjust": self.cfg.control.enable_micro_adjust,
                "final_lock_enter_dist_th_m": self.cfg.control.final_lock_enter_dist_th_m,
                "final_lock_enter_yaw_th_rad": self.cfg.control.final_lock_enter_yaw_th_rad,
                "edge_slide_dist_tolerance_m": self.cfg.control.edge_slide_dist_tolerance_m,
                "table_edge_obs_max_age_ms": self.cfg.control.table_edge_obs_max_age_ms,
                "edge_follow_min_edge_conf": self.cfg.control.edge_follow_min_edge_conf,
                "edge_follow_log_period_ms": self.cfg.control.edge_follow_log_period_ms,
                "edge_follow_stale_hold_s": self.cfg.control.edge_follow_stale_hold_s,
                "edge_follow_track_local_edge_update_hz": self.cfg.control.edge_follow_track_local_edge_update_hz,
                "target_confirm_conf_th": self.cfg.control.target_confirm_conf_th,
                "target_found_frames_to_confirm": self.cfg.control.target_found_frames_to_confirm,
                "target_lock_conf_th": self.cfg.control.target_lock_conf_th,
                "target_lock_settle_s": self.cfg.control.target_lock_settle_s,
                "edge_relocate_enabled": self.cfg.control.edge_relocate_enabled,
                "max_edge_transitions_per_task": self.cfg.control.max_edge_transitions_per_task,
            },
            "car_cmd": {
                "send_period_ms": self.cfg.car.send_period_ms,
                "uart_keepalive_hz": self.cfg.car.uart_keepalive_hz,
                "min_uart_keepalive_hz": self.cfg.car.min_uart_keepalive_hz,
                "hold_ms": self.cfg.car.cmd_hold_ms,
                "max_vx_mps": self.cfg.car.max_vx_mps,
                "max_vy_mps": self.cfg.car.max_vy_mps,
                "max_wz_radps": self.cfg.car.max_wz_radps,
                "stm32_wheel_speed_limit": self.cfg.car.stm32_wheel_speed_limit,
                "stm32_vx_scale": self.cfg.car.stm32_vx_scale,
                "stm32_vy_scale": self.cfg.car.stm32_vy_scale,
                "stm32_wz_scale": self.cfg.car.stm32_wz_scale,
                "jog_forward_speed": self.cfg.car.jog_forward_speed,
                "jog_turn_speed": self.cfg.car.jog_turn_speed,
                "jog_duration_ms": self.cfg.car.jog_duration_ms,
                "stop_on_state_enter": self.cfg.car.stop_on_state_enter,
            },
            "table_docking": {
                "align_to_approach_yaw_rad": self.cfg.control.align_to_approach_yaw_rad,
                "approach_to_align_yaw_rad": self.cfg.control.approach_to_align_yaw_rad,
                "align_to_approach_stable_obs": self.cfg.control.align_to_approach_stable_obs,
                "approach_to_align_stable_obs": self.cfg.control.approach_to_align_stable_obs,
                "coarse_align_min_dwell_s": self.cfg.control.coarse_align_min_dwell_s,
                "controlled_approach_min_dwell_s": self.cfg.control.controlled_approach_min_dwell_s,
            },
            "table_docking_motion": {
                "approach_safe_vx_mps": self.cfg.car.table_approach_safe_vx_mps,
                "approach_max_vx_mps": self.cfg.car.table_approach_max_vx_mps,
                "approach_yaw_deadband_rad": self.cfg.car.table_approach_yaw_deadband_rad,
                "approach_yaw_realign_rad": self.cfg.car.table_approach_yaw_realign_rad,
                "approach_allow_wz": self.cfg.car.table_approach_allow_wz,
                "approach_allow_vy": self.cfg.car.table_approach_allow_vy,
                "pose_missing_safe_vx_mps": self.cfg.car.table_pose_missing_safe_vx_mps,
                "pose_missing_max_hold_s": self.cfg.car.table_pose_missing_max_hold_s,
                "coarse_align": {
                    "vx_max_mps": self.cfg.car.table_coarse_align_vx_max_mps,
                    "vy_min_mps": self.cfg.car.table_coarse_align_vy_min_mps,
                    "vy_max_mps": self.cfg.car.table_coarse_align_vy_max_mps,
                    "wz_min_radps": self.cfg.car.table_coarse_align_wz_min_radps,
                    "wz_max_radps": self.cfg.car.table_coarse_align_wz_max_radps,
                },
                "controlled_approach": {
                    "vx_mps": self.cfg.car.table_approach_safe_vx_mps,
                    "vx_min_mps": self.cfg.car.table_controlled_vx_min_mps,
                    "vx_max_mps": self.cfg.car.table_controlled_vx_max_mps,
                    "vy_min_mps": self.cfg.car.table_controlled_vy_min_mps,
                    "vy_max_mps": self.cfg.car.table_controlled_vy_max_mps,
                    "wz_min_radps": self.cfg.car.table_controlled_wz_min_radps,
                    "wz_max_radps": self.cfg.car.table_controlled_wz_max_radps,
                    "allow_vy": self.cfg.car.table_approach_allow_vy,
                    "allow_wz": self.cfg.car.table_approach_allow_wz,
                },
                "final_lock": {
                    "vx_min_mps": self.cfg.car.table_final_lock_vx_min_mps,
                    "vx_max_mps": self.cfg.car.table_final_lock_vx_max_mps,
                    "vy_min_mps": self.cfg.car.table_final_lock_vy_min_mps,
                    "vy_max_mps": self.cfg.car.table_final_lock_vy_max_mps,
                    "wz_min_radps": self.cfg.car.table_final_lock_wz_min_radps,
                    "wz_max_radps": self.cfg.car.table_final_lock_wz_max_radps,
                },
                "deadband": {
                    "vx_mps": self.cfg.car.table_vx_deadband_mps,
                    "vy_mps": self.cfg.car.table_vy_deadband_mps,
                    "wz_radps": self.cfg.car.table_wz_deadband_radps,
                },
            },
            "docking": {
                "min_confidence": self.cfg.docking.min_confidence,
                "enable_lateral_control": self.cfg.docking.enable_lateral_control,
                "approach_max_vx_mps": self.cfg.docking.approach_max_vx_mps,
                "approach_max_vy_mps": self.cfg.docking.approach_max_vy_mps,
                "approach_max_wz_radps": self.cfg.docking.approach_max_wz_radps,
            },
            "task_cmd_in": {
                "transport": self.cfg.task_cmd_in.transport,
                "host": self.cfg.task_cmd_in.host,
                "port": self.cfg.task_cmd_in.port,
                "uds_path": self.cfg.task_cmd_in.uds_path,
            },
            "task_ack_out": {
                "transport": self.cfg.task_ack_out.transport,
                "host": self.cfg.task_ack_out.host,
                "port": self.cfg.task_ack_out.port,
                "send_mode": self.cfg.task_ack_out.send_mode,
            },
            "vision_obs_in": {
                "transport": self.cfg.vision_obs_in.transport,
                "host": self.cfg.vision_obs_in.host,
                "port": self.cfg.vision_obs_in.port,
            },
            "vision_req_out": {
                "transport": self.cfg.vision_req_out.transport,
                "host": self.cfg.vision_req_out.host,
                "port": self.cfg.vision_req_out.port,
                "async_enabled": self.cfg.vision_req_out.async_enabled,
                "async_queue_size": self.cfg.vision_req_out.async_queue_size,
            },
            "tts_event_out": {
                "transport": self.cfg.tts_event_out.transport,
                "host": self.cfg.tts_event_out.host,
                "port": self.cfg.tts_event_out.port,
                "async_enabled": self.cfg.tts_event_out.async_enabled,
                "async_queue_size": self.cfg.tts_event_out.async_queue_size,
            },
        }

    def _classify_stop(self, reason: str) -> Dict[str, Any]:
        now = time.time()
        uptime_s = max(0.0, now - self._boot_ts)
        recent_cmd_age = self._age_or_none(self._last_task_cmd_recv_ts)
        last_cmd = self.core.ctx.last_task_cmd
        reason = str(reason or "").strip()
        if last_cmd is not None and last_cmd.intent == "STOP" and recent_cmd_age is not None and recent_cmd_age <= 3.0:
            src = (last_cmd.source or "").strip().lower()
            if src == "voice":
                stop_class = "voice_stop"
            elif src == "manual":
                stop_class = "manual_stop"
            elif src:
                stop_class = f"{src}_stop"
            else:
                stop_class = "command_stop"
            return {"stop_class": stop_class, "stop_reason": reason or "收到 STOP 命令"}
        failsafe_tokens = ["vision_req_out", "视觉链路异常", "底盘急停", "底盘故障", "底盘超时", "搜索超时", "自动搜索超时"]
        if any(token in reason for token in failsafe_tokens):
            return {"stop_class": "failsafe_stop", "stop_reason": reason or "failsafe"}
        task_done_tokens = ["已到达目标附近", "已返回起点"]
        if any(token in reason for token in task_done_tokens):
            return {"stop_class": "task_complete_stop", "stop_reason": reason}
        if reason in {"service_stop", "service_shutdown"}:
            return {"stop_class": "shutdown_stop", "stop_reason": reason}
        if uptime_s < 5.0 and self._last_task_cmd_recv_ts <= 0.0:
            return {"stop_class": "boot_init_stop", "stop_reason": reason or "启动后空闲停止"}
        return {"stop_class": "idle_stop", "stop_reason": reason or "空闲保持停止"}

    def _build_uart_tx_meta(self, car_cmd) -> Dict[str, Any]:
        reason = (self.core.ctx.last_enter_reason or self.core.ctx.last_fail_reason or "").strip()
        meta: Dict[str, Any] = {
            "mode": car_cmd.mode,
            "kind": car_cmd.kind,
            "state": self.core.ctx.state.value,
            "task_intent": self.core.ctx.task_intent or "",
            "active_target": self.core.ctx.active_target or "",
            "session_id": self.core.ctx.active_session_id or "",
            "epoch": self.core.ctx.active_epoch,
            "req_id": self.core.ctx.active_req_id or "",
            "reason": reason,
        }
        if str(car_cmd.mode).upper() == "STOP":
            meta.update(self._classify_stop(reason))
        else:
            meta["stop_class"] = ""
            meta["stop_reason"] = ""
        return meta

    def _build_stm32_motion_tx_meta(self, kind: str, seq: int, reason: str) -> Dict[str, Any]:
        meta = {
            "mode": str(self.core.ctx.state.value),
            "state": self.core.ctx.state.value,
            "task_intent": self.core.ctx.task_intent or "",
            "active_target": self.core.ctx.active_target or "",
            "session_id": self.core.ctx.active_session_id or "",
            "epoch": self.core.ctx.active_epoch,
            "req_id": self.core.ctx.active_req_id or "",
            "stm32_kind": str(kind or ""),
            "seq": int(seq),
            "reason": str(reason or ""),
        }
        context = dict(getattr(self, "_last_motion_tx_context", {}) or {})
        for key in (
            "speed_profile",
            "speed_limit_reason",
            "forward_block_reason",
            "vx_mps",
            "vy_mps",
            "wz_radps",
            "vx_mps",
            "vy_mps",
            "wz_radps",
            "table_approach_phase",
            "final_lock_enabled",
            "micro_adjust_enabled",
            "stop_ready_ignored_for_stage_transition",
            "micro_adjust_skipped",
            "pose_missing_duration_s",
            "pose_missing_safe_vx_active",
            "normalized_control_level",
            "yaw_err",
            "dist_err_m",
            "final_lock_enter_allowed",
            "final_lock_enter_block_reason",
            "stable_lock_count",
            "required_lock_count",
            "lock_count_inc_reason",
            "lock_count_hold_reason",
            "lock_count_reset_reason",
            "uart_emit_reason",
            "last_valid_motion_cmd",
            "last_valid_motion_age_ms",
            "soft_stale_hold_active",
            "zero_cmd_reason",
            "original_cmd",
            "effective_cmd",
        ):
            if context.get(key) is not None:
                meta[key] = context.get(key)
        return meta

    @staticmethod
    def _cmd_has_motion(cmd: Any) -> bool:
        try:
            return any(
                abs(float(getattr(cmd, key, 0.0) or 0.0)) > 1e-6
                for key in ("vx_mps", "vy_mps", "wz_radps")
            )
        except Exception:
            return False

    @staticmethod
    def _cmd_dict(cmd: Any) -> Dict[str, Any]:
        if hasattr(cmd, "to_dict"):
            try:
                return dict(cmd.to_dict())
            except Exception:
                pass
        return {
            "ts": float(getattr(cmd, "ts", time.time()) or time.time()),
            "mode": str(getattr(cmd, "mode", "") or ""),
            "vx_mps": float(getattr(cmd, "vx_mps", 0.0) or 0.0),
            "vy_mps": float(getattr(cmd, "vy_mps", 0.0) or 0.0),
            "wz_radps": float(getattr(cmd, "wz_radps", 0.0) or 0.0),
            "hold_ms": int(getattr(cmd, "hold_ms", 0) or 0),
            "brake": bool(getattr(cmd, "brake", False)),
        }

    def _last_valid_motion_age_ms(self, now: Optional[float] = None) -> Optional[float]:
        if not self._last_valid_motion_cmd or self._last_valid_motion_ts <= 0.0:
            return None
        now = time.time() if now is None else float(now)
        return max(0.0, (now - float(self._last_valid_motion_ts)) * 1000.0)

    def _make_stop_cmd(self, now: float, hold_ms: int = 0) -> CmdVel:
        return CmdVel(ts=float(now), mode="STOP", vx_mps=0.0, vy_mps=0.0, wz_radps=0.0, hold_ms=int(hold_ms), brake=True)

    def _repeat_last_valid_motion_cmd(self, now: float) -> Optional[CmdVel]:
        last = dict(self._last_valid_motion_cmd or {})
        if not last:
            return None
        return CmdVel(
            ts=float(now),
            mode=str(last.get("mode") or "SEARCH"),
            vx_mps=float(last.get("vx_mps", last.get("vx", 0.0)) or 0.0),
            vy_mps=float(last.get("vy_mps", last.get("vy", 0.0)) or 0.0),
            wz_radps=float(last.get("wz_radps", last.get("wz", 0.0)) or 0.0),
            hold_ms=int(last.get("hold_ms", getattr(self.cfg.car, "cmd_hold_ms", 150)) or getattr(self.cfg.car, "cmd_hold_ms", 150)),
            brake=False,
        )

    def _arbitrate_uart_motion_cmd(self, cmd: CmdVel, summary: Dict[str, Any]) -> Tuple[CmdVel, Dict[str, Any]]:
        now = time.time()
        state = str(getattr(self.core.ctx.state, "value", self.core.ctx.state) or "").strip().upper()
        mode = str(getattr(cmd, "mode", "") or "").strip().upper()
        control_source = str(summary.get("control_source") or "").strip().lower()
        stale_level = str(summary.get("stale_level") or "").strip().lower()
        stale_reason = str(summary.get("stale_guard_reason") or summary.get("reason") or "").strip().lower()
        last_age_ms = self._last_valid_motion_age_ms(now)
        hold_ms = max(0, int(getattr(self.cfg.car, "motion_hold_ms", getattr(self.cfg.car, "cmd_hold_ms", 150)) or 0))
        hard_stale_stop_ms = max(0, int(getattr(self.cfg.car, "hard_stale_stop_ms", 800) or 800))
        explicit_stop = bool(getattr(cmd, "brake", False) or mode in {"STOP", "IDLE", "DONE", "ERROR", "ERROR_RECOVERY"} or state in {"IDLE", "STOP"})
        soft_stale_timed_out = bool(stale_level == "soft_stale" and last_age_ms is not None and last_age_ms >= float(hard_stale_stop_ms))
        perception_dead = bool(
            "perception_dead" in stale_reason
            or "camera_dead" in stale_reason
            or "vision_dead" in stale_reason
            or "system_perception_dead" in stale_reason
        )
        yolo_allows_edge_stale = bool(
            control_source in {"yolo_forward", "yolo_track_forward", "edge_guided_forward"}
            and bool(summary.get("yolo_table_control_valid", False))
            and not perception_dead
        )
        search_allows_edge_stale = bool(
            (state == "SEARCH_TABLE" or mode == "SEARCH_TABLE" or control_source == "local_rotate_search")
            and not perception_dead
            and not bool(summary.get("table_lost_search_timeout", False))
        )
        hard_stale_raw = bool(
            stale_level in {"hard_stale", "dead"}
            or "hard_stale" in stale_reason
            or "dead" in stale_reason
            or "perception_dead" in stale_reason
            or soft_stale_timed_out
        )
        hard_stale = bool(hard_stale_raw and not search_allows_edge_stale and not yolo_allows_edge_stale)
        if perception_dead:
            stale_source = "vision"
        elif stale_level or stale_reason:
            stale_source = "edge" if bool(summary.get("yolo_table_control_valid", False)) else "table"
        else:
            stale_source = ""
        has_new_valid_motion = bool(self._cmd_has_motion(cmd) and not explicit_stop and not hard_stale)
        last_within_hold = bool(last_age_ms is not None and last_age_ms <= float(hold_ms))
        soft_stale_within_hard_timeout = bool(
            stale_level == "soft_stale"
            and bool(getattr(self.cfg.car, "soft_stale_hold_enable", True))
            and last_age_ms is not None
            and last_age_ms <= float(hard_stale_stop_ms)
        )
        soft_stale_hold_active = bool(
            stale_level == "soft_stale"
            and bool(getattr(self.cfg.car, "soft_stale_hold_enable", True))
            and (last_within_hold or soft_stale_within_hard_timeout)
            and not has_new_valid_motion
            and not explicit_stop
        )
        zero_cmd_reason = ""
        emit_reason = "controller_update"
        effective = cmd

        if explicit_stop:
            emit_reason = "explicit_stop"
            zero_cmd_reason = "explicit_stop"
            effective = self._make_stop_cmd(now, hold_ms=int(getattr(cmd, "hold_ms", self.cfg.car.cmd_hold_ms) or self.cfg.car.cmd_hold_ms))
            self._last_valid_motion_cmd = None
            self._last_valid_motion_ts = 0.0
            last_age_ms = None
        elif hard_stale:
            emit_reason = "hard_stale_stop"
            zero_cmd_reason = "soft_stale_timeout" if soft_stale_timed_out else (stale_level or "hard_stale")
            effective = self._make_stop_cmd(now, hold_ms=int(getattr(cmd, "hold_ms", self.cfg.car.cmd_hold_ms) or self.cfg.car.cmd_hold_ms))
            self._last_valid_motion_cmd = None
            self._last_valid_motion_ts = 0.0
            last_age_ms = None
        elif has_new_valid_motion:
            self._last_valid_motion_cmd = self._cmd_dict(cmd)
            self._last_valid_motion_ts = now
            last_age_ms = 0.0
        elif last_within_hold or soft_stale_within_hard_timeout:
            repeat = self._repeat_last_valid_motion_cmd(now)
            if repeat is not None:
                effective = repeat
                emit_reason = "keepalive_repeat_last_valid"
                soft_stale_hold_active = bool(stale_level == "soft_stale" and bool(getattr(self.cfg.car, "soft_stale_hold_enable", True)))
            else:
                emit_reason = "no_valid_cmd_stop"
                zero_cmd_reason = "last_valid_unavailable"
                effective = self._make_stop_cmd(now, hold_ms=int(getattr(cmd, "hold_ms", self.cfg.car.cmd_hold_ms) or self.cfg.car.cmd_hold_ms))
        else:
            emit_reason = "no_valid_cmd_stop"
            zero_cmd_reason = "last_valid_expired" if self._last_valid_motion_cmd else "no_last_valid_motion"
            effective = self._make_stop_cmd(now, hold_ms=int(getattr(cmd, "hold_ms", self.cfg.car.cmd_hold_ms) or self.cfg.car.cmd_hold_ms))
            self._last_valid_motion_cmd = None
            self._last_valid_motion_ts = 0.0
            last_age_ms = None

        return effective, {
            "uart_emit_reason": emit_reason,
            "last_valid_motion_cmd": dict(self._last_valid_motion_cmd or {}),
            "last_valid_motion_age_ms": last_age_ms,
            "soft_stale_hold_active": bool(soft_stale_hold_active),
            "zero_cmd_reason": zero_cmd_reason,
            "original_cmd": self._cmd_dict(cmd),
            "effective_cmd": self._cmd_dict(effective),
            "motion_hold_ms": int(hold_ms),
            "hard_stale_stop_ms": int(hard_stale_stop_ms),
            "soft_stale_hold_enable": bool(getattr(self.cfg.car, "soft_stale_hold_enable", True)),
            "edge_stale_dead": bool(hard_stale_raw and not perception_dead),
            "table_bbox_stale": bool(stale_level in {"hard_stale", "dead"} and not summary.get("yolo_table_control_valid", False)),
            "perception_dead": bool(perception_dead),
            "stale_source": stale_source,
            "yolo_allows_edge_stale": bool(yolo_allows_edge_stale),
            "search_allows_edge_stale": bool(search_allows_edge_stale),
            "search_table_stale_gate_bypass": bool(search_allows_edge_stale and hard_stale_raw),
            "stale_gate_stop_source_state": state if hard_stale else "",
        }

    def _render_uart_line(self, payload: Dict[str, Any]) -> str:
        raw = str(payload.get("raw", "")).strip("\n")
        parts = [part.strip() for part in raw.splitlines() if part.strip()]
        return " ; ".join(parts) if parts else (str(payload.get("mode", "")).strip() or "<empty>")

    def _uart_event_key(self, payload: Dict[str, Any]) -> str:
        key_fields = {
            "raw": payload.get("raw", ""),
            "uart_kind": payload.get("uart_kind", ""),
            "rendered_line": payload.get("rendered_line", ""),
            "mode": payload.get("mode", ""),
            "kind": payload.get("kind", ""),
            "stop_class": payload.get("stop_class", ""),
            "state": payload.get("state", ""),
            "task_intent": payload.get("task_intent", ""),
            "target": payload.get("active_target", ""),
        }
        return safe_dump(key_fields)

    def _emit_uart_lowfreq_record(self, payload: Dict[str, Any], emit_reason: str, repeat_count: int):
        record = dict(payload)
        record["emit_reason"] = emit_reason
        record["repeat_count"] = int(max(1, repeat_count))
        record["rendered"] = self._render_uart_line(payload)
        self.run_logger.write_jsonl("uart_tx_lowfreq", record)

    def _update_uart_lowfreq(self, payload: Dict[str, Any]):
        period_s = max(0.5, float(self.cfg.serial.uart_lowfreq_period_s))
        key = payload.get("summary_key", "")
        now = float(payload.get("ts", time.time()))
        if not self._uart_lowfreq_key:
            self._uart_lowfreq_key = key
            self._uart_lowfreq_last_payload = dict(payload)
            self._uart_lowfreq_repeat_count = 1
            self._uart_lowfreq_last_emit_ts = now
            self._emit_uart_lowfreq_record(payload, "initial", 1)
            return
        if key != self._uart_lowfreq_key:
            if self._uart_lowfreq_last_payload is not None and self._uart_lowfreq_repeat_count > 1:
                self._emit_uart_lowfreq_record(self._uart_lowfreq_last_payload, "before_change", self._uart_lowfreq_repeat_count)
            self._uart_lowfreq_key = key
            self._uart_lowfreq_last_payload = dict(payload)
            self._uart_lowfreq_repeat_count = 1
            self._uart_lowfreq_last_emit_ts = now
            self._emit_uart_lowfreq_record(payload, "change", 1)
            return
        self._uart_lowfreq_repeat_count += 1
        self._uart_lowfreq_last_payload = dict(payload)
        if (now - self._uart_lowfreq_last_emit_ts) >= period_s:
            self._emit_uart_lowfreq_record(payload, "periodic", self._uart_lowfreq_repeat_count)
            self._uart_lowfreq_last_emit_ts = now
            self._uart_lowfreq_repeat_count = 0

    def _should_quiet_console(self, payload: Dict[str, Any]) -> bool:
        stop_class = str(payload.get("stop_class", "") or "")
        return bool(self.cfg.serial.dry_run_quiet_idle_stop) and stop_class in {"idle_stop", "boot_init_stop"}

    def _console_message(self, payload: Dict[str, Any], emit_reason: str, repeat_count: int) -> str:
        del emit_reason, repeat_count
        uart_kind = str(payload.get("uart_kind") or "").strip()
        if uart_kind == "mode":
            return f"[ORCH] CAR_MODE mode={payload.get('car_mode') or payload.get('mode') or 'UNKNOWN'}"
        if uart_kind == "vel":
            base = (
                f"[ORCH] CAR_VEL "
                f"vx={self._fmt_float(payload.get('actual_vx_mps', payload.get('vx_mps', 0.0)))} "
                f"vy={self._fmt_float(payload.get('actual_vy_mps', payload.get('vy_mps', 0.0)))} "
                f"wz={self._fmt_float(payload.get('actual_wz_radps', payload.get('wz_radps', 0.0)))}"
            )
            if "actual_hold_ms" in payload or "hold_ms" in payload:
                base += f" hold={int(payload.get('actual_hold_ms', payload.get('hold_ms', 0)) or 0)}ms"
            return base
        if uart_kind == "stop":
            return f"[ORCH] CAR_STOP mode={payload.get('mode') or 'STOP'}"
        if uart_kind == "brake":
            return f"[ORCH] CAR_BRAKE mode={payload.get('mode') or 'BRAKE'}"
        if uart_kind == "stm32_vel":
            profile = str(payload.get("speed_profile") or "n/a")
            limit_reason = str(payload.get("speed_limit_reason") or "n/a")
            block_reason = str(payload.get("forward_block_reason") or "")
            return (
                f"[ORCH] STM32_V "
                f"state={payload.get('state', payload.get('mode', 'SEARCH'))} "
                f"phase={payload.get('table_approach_phase', 'n/a')} "
                f"profile={profile} "
                f"vx_mps={self._fmt_float(payload.get('vx_mps', 0.0))} "
                f"vy_mps={self._fmt_float(payload.get('vy_mps', 0.0))} "
                f"wz_radps={self._fmt_float(payload.get('wz_radps', 0.0))} "
                f"vx_mps={self._fmt_float(payload.get('vx_mps', payload.get('actual_vx_mps', 0.0)))} "
                f"vy_mps={self._fmt_float(payload.get('vy_mps', payload.get('actual_vy_mps', 0.0)))} "
                f"wz_radps={self._fmt_float(payload.get('wz_radps', payload.get('actual_wz_radps', 0.0)))} "
                f"limit={limit_reason} "
                f"block={block_reason or 'none'} "
                f"final_lock={str(payload.get('final_lock_enabled', 'n/a')).lower()} "
                f"micro_adjust={str(payload.get('micro_adjust_enabled', 'n/a')).lower()} "
                f"pose_missing_s={self._fmt_float(payload.get('pose_missing_duration_s', 0.0))} "
                f"pose_safe_vx={str(payload.get('pose_missing_safe_vx_active', False)).lower()} "
                f"seq={payload.get('seq', 'n/a')}"
            )
        if uart_kind == "stm32_jog":
            return (
                f"[ORCH] STM32_JOG "
                f"vx={self._fmt_float(payload.get('vx_mps', 0.0))} "
                f"vy={self._fmt_float(payload.get('vy_mps', 0.0))} "
                f"wz={self._fmt_float(payload.get('wz_radps', 0.0))} "
                f"duration={int(payload.get('duration_ms', 0) or 0)}ms "
                f"seq={payload.get('seq', 'n/a')}"
            )
        if uart_kind == "stm32_status":
            return "[ORCH] STM32_STATUS"
        if uart_kind == "stm32_stop":
            return (
                f"[ORCH] STM32_STOP state={payload.get('state', payload.get('mode', 'STOP'))} "
                f"profile={payload.get('speed_profile', 'stop')} block={payload.get('forward_block_reason', 'stop')} "
                f"seq={payload.get('seq', 'n/a')}"
            )
        mode = str(payload.get("mode") or payload.get("state") or "").strip() or "UNKNOWN"
        kind = str(payload.get("kind") or "").strip()
        if kind == "stop" or str(payload.get("raw", "")).strip().upper().endswith("STOP"):
            return f"[ORCH] CAR_STOP mode={mode}"
        return (
            f"[ORCH] CAR_VEL "
            f"vx={self._fmt_float(payload.get('vx_mps', 0.0))} "
            f"vy={self._fmt_float(payload.get('vy_mps', 0.0))} "
            f"wz={self._fmt_float(payload.get('wz_radps', 0.0))} "
            f"hold={int(payload.get('hold_ms', 0) or 0)}ms"
        )

    def _emit_uart_console(self, payload: Dict[str, Any], emit_reason: str, repeat_count: int):
        if self._uart_console_mode == "silent" or self._should_quiet_console(payload):
            return
        line = self._console_message(payload, emit_reason, repeat_count)
        if self._uart_console_mode == "full" or self.operator_console.full:
            self.operator_console.emit(line)
        elif emit_reason in {"initial", "change"}:
            self.operator_console.emit_change("car", line)
        else:
            self.operator_console.emit_rate_limited("car", line, self.operator_console.default_interval_s)

    def _uart_changed_for_operator(self, payload: Dict[str, Any]) -> bool:
        last = self._uart_console_last_payload
        if last is None:
            return True
        if str(payload.get("mode", "")) != str(last.get("mode", "")):
            return True
        if str(payload.get("uart_kind", "")) != str(last.get("uart_kind", "")):
            return True
        if str(payload.get("kind", "")) != str(last.get("kind", "")):
            return True
        for key in ("vx_mps", "vy_mps", "wz_radps", "actual_vx_mps", "actual_vy_mps", "actual_wz_radps"):
            try:
                if abs(float(payload.get(key, 0.0) or 0.0) - float(last.get(key, 0.0) or 0.0)) > 0.005:
                    return True
            except (TypeError, ValueError):
                return True
        return False

    def _update_uart_console(self, payload: Dict[str, Any]):
        if self._uart_console_mode == "silent":
            return
        if not (payload.get("dry_run") and self.cfg.serial.dry_run_echo_stdout):
            return
        key = payload.get("summary_key", "")
        now = float(payload.get("ts", time.time()))
        if not self._uart_console_key:
            self._uart_console_key = key
            self._uart_console_last_payload = dict(payload)
            self._uart_console_repeat_count = 1
            self._uart_console_last_emit_ts = now
            self._emit_uart_console(payload, "initial", 1)
            return
        if key != self._uart_console_key and (self._uart_console_mode == "full" or self._uart_changed_for_operator(payload)):
            self._uart_console_key = key
            self._uart_console_last_payload = dict(payload)
            self._uart_console_repeat_count = 1
            self._uart_console_last_emit_ts = now
            self._emit_uart_console(payload, "change", 1)
            return
        self._uart_console_repeat_count += 1
        self._uart_console_last_payload = dict(payload)
        if not bool(self.cfg.serial.dry_run_echo_on_change_only):
            self._uart_console_last_emit_ts = now
            self._emit_uart_console(payload, "periodic", self._uart_console_repeat_count)
            self._uart_console_repeat_count = 0
            return
        summary_period_s = max(1.0, float(self.cfg.serial.dry_run_echo_summary_period_s))
        if (now - self._uart_console_last_emit_ts) >= summary_period_s:
            self._uart_console_last_emit_ts = now
            self._emit_uart_console(payload, "periodic", self._uart_console_repeat_count)
            self._uart_console_repeat_count = 0

    def _actual_uart_payloads(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for line in str(payload.get("raw", "") or "").splitlines():
            text = line.strip()
            if not text:
                continue
            parts = text.split()
            item = dict(payload)
            item["rendered_line"] = text
            upper = parts[0].upper() if parts else ""
            if upper == "MODE" and len(parts) >= 2:
                item["uart_kind"] = "mode"
                item["car_mode"] = parts[1].upper()
            elif upper == "V" and len(parts) >= 4:
                item["uart_kind"] = "vel"
                try:
                    item["actual_vx_mps"] = float(parts[1])
                    item["actual_vy_mps"] = float(parts[2])
                    item["actual_wz_radps"] = float(parts[3])
                    item["vx_mps"] = float(parts[1])
                    item["vy_mps"] = float(parts[2])
                    item["wz_radps"] = float(parts[3])
                except Exception:
                    pass
            elif upper == "VEL" and len(parts) >= 5:
                item["uart_kind"] = "vel"
                try:
                    item["actual_vx_mps"] = float(parts[1])
                    item["actual_vy_mps"] = float(parts[2])
                    item["actual_wz_radps"] = float(parts[3])
                    item["actual_hold_ms"] = int(float(parts[4]))
                except Exception:
                    pass
            elif upper == "STATUS":
                item["uart_kind"] = "stm32_status"
            elif upper == "STOP":
                item["uart_kind"] = "stm32_stop" if len(parts) >= 2 else "stop"
                if len(parts) >= 2:
                    try:
                        item["seq"] = int(float(parts[1]))
                    except Exception:
                        pass
            elif upper == "BRAKE":
                item["uart_kind"] = "brake"
            else:
                item["uart_kind"] = "raw"
            item["summary_key"] = self._uart_event_key(item)
            item["rendered"] = self._render_uart_line(item)
            out.append(item)
        return out

    def _emit_no_vel_if_needed(self, payload: Dict[str, Any], actual_payloads: List[Dict[str, Any]]) -> None:
        expected_vx = float(payload.get("vx_mps", 0.0) or 0.0)
        expected_vy = float(payload.get("vy_mps", 0.0) or 0.0)
        expected_wz = float(payload.get("wz_radps", 0.0) or 0.0)
        expects_vel = any(abs(v) > 1e-9 for v in (expected_vx, expected_vy, expected_wz)) or str(payload.get("kind")) == "cmd_vel"
        has_vel = any(str(item.get("uart_kind")) == "vel" for item in actual_payloads)
        if expects_vel and not has_vel:
            state = str(payload.get("state") or self.core.ctx.state.value)
            reason = str(payload.get("reason") or "no_vel_line").strip() or "no_vel_line"
            self.operator_console.emit_error(
                f"no_vel_sent:{state}:{reason}",
                f"[ORCH] WARN no_vel_sent state={state} reason={reason}",
            )

    def _update_uart_keepalive_stats(self, payload: Dict[str, Any], actual_payloads: List[Dict[str, Any]]) -> None:
        now = float(payload.get("ts", time.time()))
        key = str(payload.get("summary_key") or self._uart_event_key(payload))
        if self._uart_keepalive_last_key and key == self._uart_keepalive_last_key:
            self._uart_keepalive_same_cmd_repeat_count += 1
        else:
            self._uart_keepalive_last_key = key
            self._uart_keepalive_same_cmd_repeat_count = 0
        if self._uart_keepalive_last_tx_ts > 0.0:
            interval_ms = max(0.0, (now - self._uart_keepalive_last_tx_ts) * 1000.0)
            self._uart_keepalive_last_interval_ms = interval_ms
            self._uart_keepalive_interval_samples.append((now, interval_ms))
        self._uart_keepalive_last_tx_ts = now
        self._uart_keepalive_tx_ts.append(now)
        self._uart_keepalive_tx_count += 1
        if actual_payloads:
            self._uart_keepalive_last_cmd_type = str(actual_payloads[-1].get("uart_kind") or "")
        elif payload.get("uart_kind"):
            self._uart_keepalive_last_cmd_type = str(payload.get("uart_kind") or "")
        self._emit_uart_keepalive_summary_if_needed(now)

    def _emit_uart_keepalive_summary_if_needed(self, now: Optional[float] = None, force: bool = False) -> None:
        now = time.time() if now is None else float(now)
        period_s = 5.0
        if not force and (now - float(self._uart_keepalive_last_summary_ts or 0.0)) < period_s:
            return
        self._uart_keepalive_last_summary_ts = now
        window_start = now - period_s
        recent_ts = [float(ts) for ts in self._uart_keepalive_tx_ts if float(ts) >= window_start]
        recent_intervals = [
            float(interval_ms)
            for ts, interval_ms in self._uart_keepalive_interval_samples
            if float(ts) >= window_start
        ]
        if len(recent_ts) >= 2:
            elapsed = max(1e-6, recent_ts[-1] - recent_ts[0])
            actual_hz = float(len(recent_ts) - 1) / elapsed
        elif recent_intervals:
            avg_interval_s = max(1e-6, (sum(recent_intervals) / float(len(recent_intervals))) / 1000.0)
            actual_hz = 1.0 / avg_interval_s
        else:
            actual_hz = 0.0
        p50 = self._percentile(recent_intervals, 0.50)
        p90 = self._percentile(recent_intervals, 0.90)
        record = {
            "ts": now,
            "uart_dry_run": bool(self.cfg.serial.dry_run),
            "uart_keepalive_target_hz": float(self.cfg.car.uart_keepalive_hz),
            "uart_min_keepalive_hz": float(self.cfg.car.min_uart_keepalive_hz),
            "uart_tx_interval_ms": self._uart_keepalive_last_interval_ms,
            "uart_tx_hz": actual_hz,
            "uart_tx_interval_p50": p50,
            "uart_tx_interval_p90": p90,
            "uart_tx_count": int(self._uart_keepalive_tx_count),
            "last_cmd_type": self._uart_keepalive_last_cmd_type,
            "same_cmd_repeat_count": int(self._uart_keepalive_same_cmd_repeat_count),
        }
        self.run_logger.write_jsonl("uart_keepalive_summary", record)
        self._operator_emit(
            "[UART_KEEPALIVE] "
            f"dry_run={str(record['uart_dry_run']).lower()} "
            f"target={record['uart_keepalive_target_hz']:.1f}Hz "
            f"actual={actual_hz:.1f}Hz "
            f"interval_p50={(p50 if p50 is not None else 0.0):.1f}ms "
            f"interval_p90={(p90 if p90 is not None else 0.0):.1f}ms "
            f"last_cmd={self._uart_keepalive_last_cmd_type or 'n/a'}"
        )

    def _on_uart_tx(self, raw_line: str, dry_run: bool, tx_meta: Optional[Dict[str, Any]] = None):
        self._last_uart_tx_ts = time.time()
        payload = {
            "ts": self._last_uart_tx_ts,
            "dry_run": bool(dry_run),
            "raw": str(raw_line).rstrip("\n"),
        }
        if tx_meta:
            payload.update({k: v for k, v in tx_meta.items() if v not in (None, "")})
        payload["summary_key"] = self._uart_event_key(payload)
        payload["rendered"] = self._render_uart_line(payload)
        if self._uart_full_log or not bool(dry_run):
            self.run_logger.write_jsonl("uart_tx", payload)
        self._update_uart_lowfreq(payload)
        actual_payloads = self._actual_uart_payloads(payload)
        self._emit_no_vel_if_needed(payload, actual_payloads)
        self._update_uart_keepalive_stats(payload, actual_payloads)
        for item in actual_payloads or [payload]:
            self._update_uart_console(item)

    def start(self):
        cfg_dump = self._config_dump()
        self._operator_emit(f"[ORCH] STARTING run={self.run_logger.stack_run_id}")
        self.run_logger.write_meta({
            "service": "orch",
            "run_dir": str(self.run_logger.run_dir),
            "project_root": self.cfg.runtime.project_root,
            "log_file": self.cfg.runtime.log_file,
            "pid_file": self.cfg.runtime.pid_file,
        })
        self.run_logger.write_service_event("SERVICE_STARTING", run_dir=str(self.run_logger.run_dir))
        self.run_logger.write_jsonl("config", cfg_dump)
        self.run_logger.write_timeline("BOOT", run_dir=str(self.run_logger.run_dir), config=cfg_dump)
        loaded_files = ",".join(self.cfg.runtime.loaded_config_files) or "<defaults>"
        self._operator_emit(
            "[ORCH] CONFIG "
            f"stage={self.cfg.runtime.stage_params_file} car_cmd={self.cfg.runtime.car_cmd_params_file} "
            f"loaded={loaded_files}"
        )
        self._operator_emit(
            "[ORCH] PARAMS "
            f"tick_hz={float(self.cfg.runtime.tick_hz):.2f} send_period_ms={int(self.cfg.car.send_period_ms)} "
            f"uart_keepalive={float(self.cfg.car.uart_keepalive_hz):.1f}Hz "
            f"protocol=MODE/V/STOP "
            f"final_lock yaw={self.cfg.control.final_lock_yaw_tol_rad:.3f} "
            f"dist={self.cfg.control.final_lock_dist_tol_m:.3f} "
            f"stable_frames={int(self.cfg.control.final_lock_frames_to_arrive)} "
            f"edge_conf={self.cfg.docking.min_confidence:.2f} "
            f"slide_vy={self.cfg.car.edge_slide_vy_mps:.2f}"
        )
        
        # Effective config dump to standard output
        from common.config.effective_dump import print_effective_config
        from common.config_loader import get_config
        effective_dry = bool(getattr(self.uart, "dry_run", self.cfg.serial.dry_run))
        print_effective_config(get_config(), effective_dry_run=effective_dry)
        
        self.uart.start()
        self.task_server.start()
        self.vision_server.start()
        self._running = True
        self.run_logger.write_service_event("SERVICE_READY", run_dir=str(self.run_logger.run_dir))
        effective_dry_run = bool(getattr(self.uart, "dry_run", self.cfg.serial.dry_run))
        uart_mode = "fake" if effective_dry_run else str(self.cfg.serial.port)
        self._operator_emit(f"[ORCH] READY state={self.core.ctx.state.value} dry_run={int(effective_dry_run)} uart={uart_mode}")
        self.log_info("runtime", "SERVICE_READY", {"run_dir": str(self.run_logger.run_dir)})
        self._emit_heartbeat_if_needed(force=True)
        self._emit_system_metrics_if_needed(force=True)

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        self._running = False
        self.run_logger.write_service_event("SERVICE_STOPPING", state=self.core.ctx.state.value)
        try:
            self.uart.send_stop(tx_meta={
                "mode": "STOP",
                "kind": "stop",
                "state": self.core.ctx.state.value,
                "task_intent": self.core.ctx.task_intent or "",
                "stop_class": "shutdown_stop",
                "stop_reason": "service_stop",
            })
        except Exception:
            pass
        self.uart.close()
        self.task_server.close()
        self.vision_server.close()
        self.task_ack_sender.close()
        self.vision_req_sender.close()
        self.tts_sender.close()
        self._emit_system_metrics_if_needed(force=True)
        self.run_logger.write_service_event("SERVICE_STOPPED")
        self.run_logger.write_timeline("STOP")
        self._operator_emit(" ".join([f"[ORCH] STOPPED", "reason=service_stop"]))
        self.run_logger.close()

    def run_forever(self):
        self.start()
        period_s = 1.0 / max(1.0, float(self.cfg.runtime.tick_hz))
        try:
            while self._running:
                loop_start = time.time()
                self._drain_async_tx_results()
                self._drain_uart_feedback()
                self._drain_task_cmds()
                self._drain_vision_msgs()
                self._mark_state_machine_consume(loop_start)
                decision = self.core.tick()
                self._mark_cmd_publish(decision)
                self._flush_pending_msgs()
                self._emit_motion(decision)
                self._poll_stm32_status_if_needed()
                self._emit_state_block_if_needed()
                self._emit_heartbeat_if_needed()
                self._emit_system_metrics_if_needed()
                elapsed = time.time() - loop_start
                time.sleep(max(0.0, period_s - elapsed))
        finally:
            self.stop()

    def _drain_async_tx_results(self):
        while True:
            try:
                item = self._async_result_queue.get_nowait()
            except queue.Empty:
                break
            channel = str(item.get("channel", item.get("name", "async_out")))
            payload = item.get("payload") or {}
            ok = bool(item.get("ok", False))
            snap = item.get("snapshot") or {}
            self._last_tx_summary[channel] = float(item.get("done_ts", time.time()))
            self.run_logger.write_jsonl(f"{channel}_tx_result", {
                "ts": item.get("done_ts", time.time()),
                "channel": channel,
                "ok": ok,
                "seq": item.get("seq"),
                "payload": payload,
                "snapshot": snap,
            })
            if channel == "vision_req_out":
                error = "" if ok else f"link_state={snap.get('link_state')} fail_count={snap.get('fail_count')}"
                self.core.handle_vision_req_send_result(ok, payload, error=error)
                self.run_logger.write_ipc(
                    channel,
                    "send_ok" if ok else "send_failed",
                    direction="TX",
                    req_id=payload.get("req_id"),
                    session_id=payload.get("session_id"),
                    epoch=payload.get("epoch"),
                    ok=ok,
                    op=payload.get("op"),
                    stage=payload.get("stage"),
                    mode_hint=payload.get("mode_hint"),
                    error=error,
                    link_state=snap.get("link_state"),
                )
                self.run_logger.write_timeline(
                    "VISION_REQ_SEND",
                    req_id=payload.get("req_id"),
                    op=payload.get("op"),
                    stage=payload.get("stage"),
                    mode_hint=payload.get("mode_hint"),
                    sent=ok,
                    session_id=payload.get("session_id"),
                    epoch=payload.get("epoch"),
                )
            elif channel == "tts_event_out":
                self.run_logger.write_ipc(
                    channel,
                    "send_ok" if ok else "send_failed",
                    direction="TX",
                    ok=ok,
                    text=payload.get("text"),
                    interrupt=payload.get("interrupt", False),
                    link_state=snap.get("link_state"),
                )
                self.run_logger.write_timeline("TTS_EVENT_SEND", text=payload.get("text"), sent=ok, interrupt=payload.get("interrupt", False))

    def _drain_uart_feedback(self):
        for raw in self.uart.drain_rx_lines():
            state = parse_car_state_line(raw)
            if state is not None:
                self._update_stm32_motion_status(state)
                if state.estop and self.cfg.car_estop_to_stop:
                    self.log("warn", "service", f"Immediate user ESTOP detected: {state.message or state.state}, triggering emergency stop")
                    self.uart.send_emergency_stop()
                    self.motion_adapter.cancel_active_jogs()
                self.core.handle_car_state(state)
                self.run_logger.write_jsonl("car_state", state.to_dict())
                self.run_logger.write_jsonl("motion_status", dict(self.motion_status))
                self.run_logger.write_timeline("CAR_STATE", state=state.state, message=state.message, estop=state.estop, timeout=state.timeout, fault=state.fault)
                continue
            arm_resp = parse_arm_response(raw)
            if arm_resp is not None:
                self.core.handle_arm_response(arm_resp)
                self.run_logger.write_timeline("ARM_RESPONSE", ok=arm_resp.ok, message=arm_resp.message)
                continue

    def _parse_motion_seq(self, text: Any) -> Optional[int]:
        match = re.search(r"\bseq\s*=\s*(-?\d+)\b", str(text or ""), re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None

    def _next_motion_seq(self) -> int:
        self._motion_seq = (int(self._motion_seq) % 999999) + 1
        return self._motion_seq

    def _record_motion_tx(self, seq: Optional[int], kind: str) -> None:
        if seq is not None:
            self.motion_status["last_seq"] = int(seq)
        self.run_logger.write_jsonl("motion_tx", {
            "ts": time.time(),
            "kind": kind,
            "seq": seq,
            "motion_status": dict(self.motion_status),
        })

    def send_stm32_vel(self, vx_mps, vy_mps, wz_radps, _unused=None, seq: Optional[int] = None) -> bool:
        if seq is None:
            adapter_seq = self.motion_adapter.set_velocity(vx_mps, vy_mps, wz_radps, mode="SEARCH", reason="service_api")
            self.motion_status["last_seq"] = adapter_seq
            return True
        tx_seq = self._next_motion_seq() if seq is None else int(seq)
        self._record_motion_tx(tx_seq, "vel")
        return self.uart.send_stm32_vel(vx_mps, vy_mps, wz_radps, _unused, tx_seq, tx_meta={
            "kind": "stm32_vel",
            "motion_protocol": "stm32",
            "seq": tx_seq,
        })

    def send_stm32_stop(self, seq: Optional[int] = None) -> bool:
        if seq is None:
            adapter_seq = self.motion_adapter.stop(reason="service_api")
            self.motion_status["last_seq"] = adapter_seq
            self.motion_status["jog_running"] = False
            return True
        tx_seq = self._next_motion_seq() if seq is None else int(seq)
        self.motion_status["jog_running"] = False
        self._record_motion_tx(tx_seq, "stop")
        return self.uart.send_stm32_stop(tx_seq, tx_meta={
            "kind": "stm32_stop",
            "motion_protocol": "stm32",
            "seq": tx_seq,
        })

    def send_stm32_jog(self, vx_mps, vy_mps, wz_radps, _unused, duration_ms, seq: Optional[int] = None) -> bool:
        if seq is None:
            old_duration = self.motion_adapter.jog_duration_ms
            self.motion_adapter.jog_duration_ms = self.motion_adapter._clamp_int(duration_ms, 60, 500)
            try:
                adapter_seq = self.motion_adapter.jog_velocity(
                    float(vx_mps or 0.0),
                    float(vy_mps or 0.0),
                    float(wz_radps or 0.0),
                    reason="service_api",
                )
            finally:
                self.motion_adapter.jog_duration_ms = old_duration
            self.motion_status["last_seq"] = adapter_seq
            self.motion_status["jog_running"] = True
            return True
        tx_seq = self._next_motion_seq() if seq is None else int(seq)
        self.motion_status["jog_running"] = True
        self._record_motion_tx(tx_seq, "jog")
        return self.uart.send_stm32_jog(vx_mps, vy_mps, wz_radps, _unused, duration_ms, tx_seq, tx_meta={
            "kind": "stm32_jog",
            "motion_protocol": "stm32",
            "seq": tx_seq,
        })

    def send_stm32_status(self) -> bool:
        self.motion_adapter.query_status()
        return True

    def _poll_stm32_status_if_needed(self) -> None:
        if not bool(getattr(self.cfg.serial, "stm32_status_enabled", False)):
            return
        now = time.time()
        period_s = max(0.1, float(getattr(self.cfg.serial, "stm32_status_period_s", 1.0) or 1.0))
        if (now - self._last_stm32_status_poll_ts) < period_s:
            return
        self._last_stm32_status_poll_ts = now
        self.send_stm32_status()

    def _update_stm32_motion_status(self, state) -> None:
        raw = str(getattr(state, "raw", "") or "")
        message = str(getattr(state, "message", "") or "")
        seq = self._parse_motion_seq(raw) if raw else None
        if seq is None:
            seq = self._parse_motion_seq(message)
        now = time.time()
        state_name = str(getattr(state, "state", "") or "").upper()
        raw_upper = raw.upper()

        self.motion_status["last_rx_time"] = now
        if state_name == "ACK_START" or "[JOG_START]" in raw_upper:
            if seq is not None:
                self.motion_status["last_ack_seq"] = seq
            if "[JOG_START]" in raw_upper:
                self.motion_status["jog_running"] = True
            self.motion_status["stm32_timeout_seen"] = False
        elif state_name == "DONE" or "[JOG_DONE]" in raw_upper:
            if seq is not None:
                self.motion_status["last_done_seq"] = seq
            self.motion_status["jog_running"] = False
            self.motion_status["stm32_timeout_seen"] = False
        elif state_name == "BUSY" or "[JOG_BUSY]" in raw_upper:
            if "[JOG_BUSY]" in raw_upper:
                self.motion_status["jog_running"] = True
        elif state_name == "TIMEOUT" or "[TIMEOUT]" in raw_upper:
            self.motion_status["jog_running"] = False
            self.motion_status["stm32_timeout_seen"] = True

    def _send_task_ack(self, cmd: TaskCmd, accepted: bool, reason: str):
        ack = make_task_ack(cmd, accepted=accepted, reason=reason, state=self.core.ctx.state.value)
        disabled = str(getattr(self.cfg.task_ack_out, "transport", "") or "").lower() == "disabled"
        sent = True if disabled else self.task_ack_sender.send(ack)
        self._last_tx_summary["task_ack_out"] = time.time()
        self.run_logger.write_jsonl("task_ack", ack)
        self.run_logger.write_ipc(
            "task_ack_out",
            "disabled" if disabled else ("ack_sent" if sent else "ack_send_failed"),
            direction="TX",
            cmd_id=cmd.cmd_id,
            session_id=cmd.session_id,
            epoch=cmd.epoch,
            accepted=accepted,
            ok=sent,
            reason=reason,
        )
        self._operator_ipc_event(
            "task_ack_out",
            "disabled" if disabled else ("ack_sent" if sent else "ack_send_failed"),
            {"cmd_id": cmd.cmd_id, "accepted": accepted, "error": "" if sent else reason},
        )
        self.log_ipc("TX", "task_ack", "disabled" if disabled else ("sent" if sent else "failed"), {"cmd_id": cmd.cmd_id, "accepted": accepted})

    def _drain_task_cmds(self):
        for item in self.task_server.drain():
            payload = item["payload"]
            self._last_task_cmd_recv_ts = float(item.get("recv_ts", time.time()))
            self.log_ipc("RX", "task_cmd", "received", {"cmd_id": payload.get("cmd_id"), "intent": payload.get("intent")})
            try:
                cmd = TaskCmd.from_dict(payload, set(self.cfg.frozen_targets.keys()))
            except Exception as exc:
                self.log_warn("task_cmd", f"bad task_cmd err={self._short_err(exc)}")
                self.operator_console.emit_error(
                    f"task_cmd_bad:{exc}",
                    f"[ORCH] ERROR task_cmd invalid err={self._short_err(exc)}",
                )
                self.run_logger.write_event(f"bad task_cmd: {payload} ({exc})")
                self.run_logger.write_ipc(
                    "task_cmd_in",
                    "bad_payload",
                    direction="RX",
                    ok=False,
                    error=str(exc),
                    payload=safe_dump(payload),
                )
                self.run_logger.write_timeline("TASK_CMD_BAD", payload=safe_dump(payload), error=str(exc))
                continue
            if cmd.intent in {"FIND", "RETURN"}:
                self._demo_start_pending_target = cmd.target or ("return_home" if cmd.intent == "RETURN" else "n/a")
            if cmd.intent == "STOP":
                self.log("warn", "service", "Immediate task_cmd STOP received: triggering emergency stop")
                self.uart.send_emergency_stop()
                self.motion_adapter.cancel_active_jogs()
            accepted, reason = self.core.handle_task_cmd(cmd)
            if accepted and cmd.intent in {"FIND", "RETURN"}:
                self.demo_console.dry_run = bool(getattr(self.cfg.serial, "dry_run", False))
                self.demo_console.task_start(self._demo_start_pending_target)
                self._flush_demo_deferred_phases()
            else:
                self._demo_deferred_phases = []
            self._demo_start_pending_target = ""
            self.operator_console.emit_change(
                f"task:{cmd.session_id}:{cmd.epoch}:{cmd.cmd_id}",
                f"[ORCH] TASK cmd={cmd.intent.lower()} target={cmd.target or ''} session={cmd.session_id or ''} epoch={cmd.epoch}",
            )
            self.run_logger.write_jsonl("task_cmd", cmd.to_dict())
            self.run_logger.write_ipc(
                "task_cmd_in",
                "received",
                direction="RX",
                cmd_id=cmd.cmd_id,
                session_id=cmd.session_id,
                epoch=cmd.epoch,
                ok=True,
                intent=cmd.intent,
                target=cmd.target,
            )
            self.run_logger.write_timeline("TASK_CMD_RECV", cmd_id=cmd.cmd_id, intent=cmd.intent, target=cmd.target, accepted=accepted, reason=reason, session_id=cmd.session_id, epoch=cmd.epoch)
            self._send_task_ack(cmd, accepted=accepted, reason=reason)

    def _flatten_vision_payloads(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        msg_type = str(payload.get("type", "") or "").strip().lower()
        if msg_type == "vision_obs":
            env = VisionObsEnvelope.from_dict(payload)
            self.run_logger.write_jsonl("vision_obs", env.to_dict() if self._vision_full_obs_log else self._lite_vision_obs(env))
            if env.obs_class == "diagnostic":
                metrics = payload.get("metrics")
                if not isinstance(metrics, dict):
                    perception = env.perception if isinstance(env.perception, dict) else {}
                    metrics = perception.get("metrics")
                self._last_diagnostic_obs_metrics = dict(metrics) if isinstance(metrics, dict) else {}
                self.log_ipc("RX", "vision_obs", "diagnostic_metrics", {
                    "req_id": env.req_id,
                    "stage": env.stage,
                    "mode": env.mode,
                    "status": env.status,
                    "session_id": env.session_id,
                    "epoch": env.epoch,
                    "obs_class": env.obs_class,
                    "metrics_keys": sorted(self._last_diagnostic_obs_metrics.keys()),
                })
                self.run_logger.write_ipc(
                    "vision_obs_in",
                    "diagnostic_metrics",
                    direction="RX",
                    req_id=env.req_id,
                    session_id=env.session_id,
                    epoch=env.epoch,
                    ok=True,
                    stage=env.stage,
                    mode=env.mode,
                    status=env.status,
                    obs_class=env.obs_class,
                    metrics=safe_dump(self._last_diagnostic_obs_metrics),
                )
                return []
            self.core.confirm_vision_state(env.stage, env.mode, source="vision_obs")
            perception = dict(env.perception or {})
            has_table_edge = isinstance(perception.get("table_edge_obs"), dict)
            has_target = isinstance(perception.get("target_obs"), dict)
            if has_table_edge != self._last_obs_flags.get("table_edge") or has_target != self._last_obs_flags.get("target"):
                self._last_obs_flags["table_edge"] = has_table_edge
                self._last_obs_flags["target"] = has_target
                line = f"[ORCH] OBS table_edge={int(has_table_edge)} target={int(has_target)} mode={env.mode} req={env.req_id or ''}"
                self.operator_console.emit_change("obs:flags", line)
            self.log_ipc("RX", "vision_obs", "received", {
                "req_id": env.req_id,
                "stage": env.stage,
                "mode": env.mode,
                "status": env.status,
                "session_id": env.session_id,
                "epoch": env.epoch,
                "obs_class": env.obs_class,
                "has_table_edge_obs": has_table_edge,
                "has_target_obs": has_target,
            })
            self.run_logger.write_ipc(
                "vision_obs_in",
                "envelope_received",
                direction="RX",
                req_id=env.req_id,
                session_id=env.session_id,
                epoch=env.epoch,
                ok=True,
                stage=env.stage,
                mode=env.mode,
                status=env.status,
                obs_class=env.obs_class,
                has_table_edge_obs=has_table_edge,
                has_target_obs=has_target,
            )
        out = list(iter_vision_perception_payloads(payload))
        if msg_type == "vision_obs":
            env = VisionObsEnvelope.from_dict(payload)
            if env.stage == "GRASP" and isinstance(env.result, dict) and env.result:
                grasp_obs = dict(env.result)
                grasp_obs["status"] = str(env.status or "")
                grasp_obs["type"] = "grasp_obs"
                grasp_obs["ts"] = float(env.ts)
                grasp_obs["req_id"] = env.req_id
                grasp_obs["session_id"] = env.session_id
                grasp_obs["epoch"] = int(env.epoch)
                out.append(grasp_obs)
        return out

    def _drain_vision_msgs(self):
        latest_table: Optional[TableEdgeObs] = None
        latest_target: Optional[TargetObs] = None
        latest_home: Optional[HomeTagObs] = None
        latest_grasp: Optional[Dict] = None
        latest_table_priority = -1
        latest_target_priority = -1
        latest_home_priority = -1
        latest_grasp_priority = -1
        raw_items: List[dict] = self.vision_server.drain()
        for item in raw_items:
            recv_ts = float(item.get("recv_ts", time.time()))
            for payload in self._flatten_vision_payloads(item["payload"]):
                msg_type = str(payload.get("type", "")).strip().lower()
                priority = int(payload.get("_perception_priority", 0) or 0)
                from_envelope = bool(payload.get("_from_vision_obs_envelope", False))
                self.log_ipc("RX", msg_type, "received", {"req_id": payload.get("req_id"), "found": payload.get("found", payload.get("table_found"))})
                try:
                    if msg_type == "table_edge_obs":
                        parsed = TableEdgeObs.from_dict(payload)
                        parsed.obs_recv_ts = recv_ts
                        parsed.orchestrator_recv_ts_ms = self._epoch_ms(recv_ts)
                        recv_interval_ms = (
                            max(0.0, (recv_ts - float(self._last_table_edge_recv_ts)) * 1000.0)
                            if float(self._last_table_edge_recv_ts or 0.0) > 0.0
                            else None
                        )
                        self._last_table_edge_recv_ts = recv_ts
                        recv_hz = 1000.0 / float(recv_interval_ms) if recv_interval_ms and recv_interval_ms > 0.0 else 0.0
                        parsed.table_edge_obs_recv_interval_ms = recv_interval_ms
                        parsed.orchestrator_recv_interval_ms = recv_interval_ms
                        parsed.table_edge_obs_recv_hz = float(recv_hz)
                        if parsed.vision_publish_ts_ms is not None:
                            try:
                                parsed.vision_publish_to_orch_recv_ms = max(0.0, float(parsed.orchestrator_recv_ts_ms) - float(parsed.vision_publish_ts_ms))
                            except Exception:
                                pass
                        if parsed.obs_publish_ts is not None:
                            try:
                                parsed.publish_delay_ms = max(0.0, (recv_ts - float(parsed.obs_publish_ts)) * 1000.0)
                            except Exception:
                                pass
                        self._edge_obs_rate_ts.append(recv_ts)
                        self._observe_trace_sample("table_edge_obs_recv_interval_ms", recv_interval_ms)
                        self._observe_trace_sample("vision_publish_to_orch_recv_ms", getattr(parsed, "vision_publish_to_orch_recv_ms", None))
                        if priority >= latest_table_priority:
                            latest_table = parsed
                            latest_table_priority = priority
                        self._last_vision_obs_recv_ts = recv_ts
                        reason = str(payload.get("reason") or parsed.reason or "").strip()
                        verdict = "table_edge_obs_accepted" if parsed.table_found else "table_edge_obs_rejected"
                        self.log_ipc("RX", "table_edge_obs", verdict, {
                            "req_id": parsed.req_id,
                            "found": parsed.table_found,
                            "edge_found": parsed.edge_found,
                            "depth_valid": parsed.depth_valid,
                            "obs_ts": parsed.obs_ts,
                            "age_ms": parsed.age_ms,
                            "obs_total_age_ms": parsed.obs_total_age_ms,
                            "vision_process_ms": parsed.vision_process_ms,
                            "edge_update_interval_ms": parsed.edge_update_interval_ms,
                            "frame_id": parsed.frame_id,
                            "seq": parsed.seq,
                            "obs_seq": parsed.obs_seq,
                            "camera_frame_seq": parsed.camera_frame_seq,
                            "orchestrator_recv_ts_ms": parsed.orchestrator_recv_ts_ms,
                            "table_edge_obs_recv_interval_ms": parsed.table_edge_obs_recv_interval_ms,
                            "orchestrator_recv_interval_ms": parsed.orchestrator_recv_interval_ms,
                            "table_edge_obs_recv_hz": parsed.table_edge_obs_recv_hz,
                            "vision_publish_to_orch_recv_ms": parsed.vision_publish_to_orch_recv_ms,
                            "source_mode": parsed.source_mode,
                            "is_stale": parsed.is_stale,
                            "point_count": parsed.point_count,
                            "table_point_count": parsed.table_point_count,
                            "reason": reason or None,
                        })
                        self.run_logger.write_ipc(
                            "vision_obs_in",
                            verdict,
                            direction="RX",
                            req_id=parsed.req_id,
                            session_id=parsed.session_id,
                            epoch=parsed.epoch,
                            ok=True,
                            found=parsed.table_found,
                            edge_found=parsed.edge_found,
                            depth_valid=parsed.depth_valid,
                            obs_ts=parsed.obs_ts,
                            age_ms=parsed.age_ms,
                            obs_total_age_ms=parsed.obs_total_age_ms,
                            vision_process_ms=parsed.vision_process_ms,
                            edge_update_interval_ms=parsed.edge_update_interval_ms,
                            frame_id=parsed.frame_id,
                            seq=parsed.seq,
                            obs_seq=parsed.obs_seq,
                            camera_frame_seq=parsed.camera_frame_seq,
                            orchestrator_recv_ts_ms=parsed.orchestrator_recv_ts_ms,
                            table_edge_obs_recv_interval_ms=parsed.table_edge_obs_recv_interval_ms,
                            orchestrator_recv_interval_ms=parsed.orchestrator_recv_interval_ms,
                            table_edge_obs_recv_hz=parsed.table_edge_obs_recv_hz,
                            vision_publish_to_orch_recv_ms=parsed.vision_publish_to_orch_recv_ms,
                            source_mode=parsed.source_mode,
                            is_stale=parsed.is_stale,
                            point_count=parsed.point_count,
                            table_point_count=parsed.table_point_count,
                            reason=reason,
                            msg_type=msg_type,
                            from_envelope=from_envelope,
                            priority=priority,
                        )
                    elif msg_type == "home_tag_obs":
                        parsed = HomeTagObs.from_dict(payload)
                        if priority >= latest_home_priority:
                            latest_home = parsed
                            latest_home_priority = priority
                        self._last_home_obs_recv_ts = recv_ts
                        self.run_logger.write_ipc(
                            "vision_obs_in",
                            "received",
                            direction="RX",
                            req_id=parsed.req_id,
                            session_id=parsed.session_id,
                            epoch=parsed.epoch,
                            ok=True,
                            found=parsed.found,
                            msg_type=msg_type,
                            from_envelope=from_envelope,
                            priority=priority,
                        )
                    elif msg_type == "target_obs":
                        parsed = TargetObs.from_dict(payload)
                        self._target_obs_rate_ts.append(recv_ts)
                        if priority >= latest_target_priority:
                            latest_target = parsed
                            latest_target_priority = priority
                        self._last_vision_obs_recv_ts = recv_ts
                        self._emit_target_obs_console(payload)
                        self.run_logger.write_ipc(
                            "vision_obs_in",
                            "received",
                            direction="RX",
                            req_id=parsed.req_id,
                            session_id=parsed.session_id,
                            epoch=parsed.epoch,
                            ok=True,
                            found=parsed.found,
                            msg_type=msg_type,
                            from_envelope=from_envelope,
                            priority=priority,
                        )
                    elif msg_type == "grasp_obs":
                        if priority >= latest_grasp_priority:
                            latest_grasp = dict(payload)
                            latest_grasp_priority = priority
                        self._last_vision_obs_recv_ts = recv_ts
                        self.run_logger.write_ipc(
                            "vision_obs_in",
                            "grasp_received",
                            direction="RX",
                            req_id=payload.get("req_id"),
                            session_id=payload.get("session_id"),
                            epoch=payload.get("epoch"),
                            ok=True,
                            grasp_status=payload.get("status"),
                            from_envelope=from_envelope,
                            msg_type=msg_type,
                        )
                    else:
                        self.run_logger.write_ipc(
                            "vision_obs_in",
                            "ignored_payload",
                            direction="RX",
                            ok=False,
                            payload=safe_dump(payload),
                            msg_type=msg_type,
                        )
                except Exception as exc:
                    self.log_warn("vision", f"bad vision msg type={msg_type} err={self._short_err(exc)}")
                    self.operator_console.emit_error(
                        f"vision_bad:{msg_type}:{exc}",
                        f"[ORCH] ERROR vision_obs invalid_json peer={item.get('peer', '')} err={self._short_err(exc)}",
                    )
                    self.run_logger.write_event(f"bad vision msg: {payload} ({exc})")
                    self.run_logger.write_ipc(
                        "vision_obs_in",
                        "bad_payload",
                        direction="RX",
                        ok=False,
                        error=str(exc),
                        payload=safe_dump(payload),
                        msg_type=msg_type,
                    )
                    self.run_logger.write_timeline("VISION_BAD", payload=safe_dump(payload), error=str(exc))
        if latest_table is not None:
            self.core.handle_table_obs(latest_table)
            self.run_logger.write_jsonl("table_edge_obs", latest_table.to_dict())
        if latest_target is not None:
            self.core.handle_target_obs(latest_target)
            self.run_logger.write_jsonl("target_obs", latest_target.to_dict())
        if latest_home is not None:
            self.core.handle_home_obs(latest_home)
            self.run_logger.write_jsonl("home_tag_obs", latest_home.to_dict())
        if latest_grasp is not None:
            self.core.handle_grasp_obs(latest_grasp)

    def _enqueue_async_or_fail(self, sender: Any, channel: str, msg: Dict[str, Any]) -> bool:
        ok = sender.send(msg)
        if ok:
            self.run_logger.write_ipc(
                channel,
                "enqueue_ok",
                direction="TX",
                req_id=msg.get("req_id"),
                session_id=msg.get("session_id"),
                epoch=msg.get("epoch"),
                ok=True,
                op=msg.get("op"),
                stage=msg.get("stage"),
                mode_hint=msg.get("mode_hint"),
                msg_type=msg.get("type"),
            )
            return True
        error = f"enqueue_failed link_state={self._sender_snapshot(sender).get('link_state')}"
        self.run_logger.write_ipc(
            channel,
            "enqueue_failed",
            direction="TX",
            req_id=msg.get("req_id"),
            session_id=msg.get("session_id"),
            epoch=msg.get("epoch"),
            ok=False,
            op=msg.get("op"),
            stage=msg.get("stage"),
            mode_hint=msg.get("mode_hint"),
            error=error,
        )
        self.run_logger.write_timeline(
            f"{channel.upper()}_ENQUEUE_FAIL",
            req_id=msg.get("req_id"),
            op=msg.get("op"),
            stage=msg.get("stage"),
            mode_hint=msg.get("mode_hint"),
            error=error,
        )
        return False

    def _flush_pending_msgs(self):
        for msg in self.core.drain_vision_msgs():
            self.run_logger.write_jsonl(msg.get("type", "vision_req"), msg)
            summary_line = self._vision_req_console_summary(msg)
            if summary_line is not None:
                req_payload = dict(msg.get("payload") or {})
                key = "|".join(
                    [
                        str(msg.get("req_type") or req_payload.get("req_type") or ""),
                        str(msg.get("session_id") or ""),
                        str(msg.get("target") or ""),
                        str(msg.get("stage") or ""),
                        str(msg.get("mode_hint") or ""),
                        str(req_payload.get("search_kind") or ""),
                    ]
                )
                if key != self._last_target_search_req_console_key:
                    self._last_target_search_req_console_key = key
                    self.operator_console.emit(summary_line)
            if str(getattr(self.cfg.vision_req_out, "transport", "") or "").lower() == "disabled":
                self._last_tx_summary["vision_req_out"] = time.time()
                self.core.handle_vision_req_send_result(True, msg)
                self.run_logger.write_ipc(
                    "vision_req_out",
                    "disabled",
                    direction="TX",
                    req_id=msg.get("req_id"),
                    session_id=msg.get("session_id"),
                    epoch=msg.get("epoch"),
                    ok=True,
                    op=msg.get("op"),
                    stage=msg.get("stage"),
                    mode_hint=msg.get("mode_hint"),
                )
                self.run_logger.write_timeline(
                    "VISION_REQ_DISABLED",
                    req_id=msg.get("req_id"),
                    op=msg.get("op"),
                    stage=msg.get("stage"),
                    mode_hint=msg.get("mode_hint"),
                    session_id=msg.get("session_id"),
                )
                continue
            if isinstance(self.vision_req_sender, AsyncJsonlClientSender):
                queued = self._enqueue_async_or_fail(self.vision_req_sender, "vision_req_out", msg)
                if not queued:
                    self.core.handle_vision_req_send_result(False, msg, error="async enqueue failed")
                continue
            sent = self.vision_req_sender.send(msg)
            self._last_tx_summary["vision_req_out"] = time.time()
            error = "" if sent else f"link_state={self.vision_req_sender.snapshot().get('link_state')}"
            self.core.handle_vision_req_send_result(sent, msg, error=error)
            self.run_logger.write_ipc(
                "vision_req_out",
                "send_ok" if sent else "send_failed",
                direction="TX",
                req_id=msg.get("req_id"),
                session_id=msg.get("session_id"),
                epoch=msg.get("epoch"),
                ok=sent,
                op=msg.get("op"),
                stage=msg.get("stage"),
                mode_hint=msg.get("mode_hint"),
                error=error,
            )
            self.run_logger.write_timeline(
                "VISION_REQ_SEND",
                req_id=msg.get("req_id"),
                op=msg.get("op"),
                stage=msg.get("stage"),
                mode_hint=msg.get("mode_hint"),
                sent=sent,
                session_id=msg.get("session_id"),
                epoch=msg.get("epoch"),
            )
            self.log_ipc(
                "TX",
                "vision_req",
                "sent" if sent else "failed",
                {
                    "req_id": msg.get("req_id"),
                    "op": msg.get("op"),
                    "stage": msg.get("stage"),
                    "mode_hint": msg.get("mode_hint"),
                },
            )
        for msg in self.core.drain_tts_msgs():
            self.run_logger.write_jsonl("tts_event", msg)
            if self.cfg.tts_event_out.transport == "disabled":
                self.run_logger.write_ipc(
                    "tts_event_out",
                    "disabled",
                    direction="TX",
                    ok=False,
                    text=msg.get("text"),
                    interrupt=msg.get("interrupt", False),
                )
                continue
            if isinstance(self.tts_sender, AsyncJsonlClientSender):
                self._enqueue_async_or_fail(self.tts_sender, "tts_event_out", msg)
                continue
            sent = self.tts_sender.send(msg)
            self._last_tx_summary["tts_event_out"] = time.time()
            self.run_logger.write_ipc(
                "tts_event_out",
                "send_ok" if sent else "send_failed",
                direction="TX",
                ok=sent,
                text=msg.get("text"),
                interrupt=msg.get("interrupt", False),
            )
            self.log_ipc("TX", "tts_event", "sent" if sent else "failed", {"text": msg.get("text"), "interrupt": msg.get("interrupt")})

    def _control_summary_with_context(self, decision) -> Dict[str, Any]:
        summary = dict(getattr(decision, "control_summary", None) or {})
        cmd = decision.cmd
        summary.setdefault("state", self.core.ctx.state.value)
        summary.setdefault("cmd", {"vx": cmd.vx_mps, "vy": cmd.vy_mps, "wz": cmd.wz_radps, "hold_ms": cmd.hold_ms})
        block = self.core.export_state_block()
        for key in (
            "edge_found",
            "edge_valid",
            "confidence",
            "edge_conf",
            "yaw_err_rad",
            "dist_err_m",
            "target_dist_m",
            "table_edge_obs_age_ms",
            "obs_total_age_ms",
            "vision_process_ms",
            "control_loop_age_ms",
            "obs_seq",
            "camera_frame_seq",
            "camera_frame_ts_ms",
            "vision_process_start_ts_ms",
            "vision_process_end_ts_ms",
            "vision_publish_ts_ms",
            "obs_out_send_ts_ms",
            "orchestrator_recv_ts_ms",
            "state_machine_consume_ts_ms",
            "cmd_publish_ts_ms",
            "camera_frame_interval_ms",
            "camera_frame_hz",
            "vision_process_interval_ms",
            "vision_publish_interval_ms",
            "obs_out_send_interval_ms",
            "obs_out_send_hz",
            "table_edge_obs_recv_interval_ms",
            "orchestrator_recv_interval_ms",
            "table_edge_obs_recv_hz",
            "state_machine_tick_interval_ms",
            "state_machine_consume_interval_ms",
            "same_obs_reuse_count",
            "obs_seq_gap",
            "obs_age_at_consume_ms",
            "vision_publish_to_orch_recv_ms",
            "orch_recv_to_state_consume_ms",
            "edge_update_interval_ms",
            "stale_level",
            "stale_guard_active",
            "stale_guard_reason",
            "edge_obs_is_stale",
            "edge_follow_stale",
            "lock_ready",
            "lock_reason",
        ):
            if summary.get(key) is None:
                summary[key] = block.get(key)
        if block.get("lock_reason"):
            summary["lock_reason"] = block.get("lock_reason")
        if summary.get("measured_distance_m") is None and summary.get("target_dist_m") is not None and summary.get("dist_err_m") is not None:
            summary["measured_distance_m"] = float(summary["target_dist_m"]) + float(summary["dist_err_m"])
        summary["current_edge_id"] = self.core.ctx.current_edge_id
        if str(summary.get("state")) == "EDGE_SLIDE_SEARCH":
            target_obs = self.core.ctx.last_target_obs
            table_obs = self.core.ctx.last_table_obs
            summary["target_found"] = bool(getattr(target_obs, "found", False)) if target_obs is not None else False
            summary["target_conf"] = (
                getattr(target_obs, "matched_conf", None)
                if target_obs is not None and getattr(target_obs, "matched_conf", None) is not None
                else (getattr(target_obs, "confidence", None) if target_obs is not None else None)
            )
            summary["matched_cls"] = (
                getattr(target_obs, "matched_cls", None) or getattr(target_obs, "target", None)
                if target_obs is not None
                else None
            )
            summary["matched_conf"] = (
                getattr(target_obs, "matched_conf", None)
                if target_obs is not None
                else None
            )
            summary["best_cls"] = getattr(target_obs, "best_cls", None) if target_obs is not None else None
            summary["best_conf"] = getattr(target_obs, "best_conf", None) if target_obs is not None else None
            summary["edge_obs_age_ms"] = block.get("table_edge_obs_age_ms")
            summary["edge_obs_is_stale"] = bool(block.get("edge_obs_is_stale", False))
            summary["edge_obs_frame_id"] = block.get("table_edge_obs_frame_id")
            summary["edge_obs_seq"] = block.get("table_edge_obs_seq")
            summary["edge_update_interval_ms"] = block.get("edge_update_interval_ms")
            summary["edge_process_ms"] = block.get("edge_process_ms")
            summary["edge_obs_source_mode"] = block.get("table_edge_obs_source_mode") or getattr(table_obs, "source_mode", None)
            summary["edge_obs_unavailable"] = bool(getattr(table_obs, "depth_valid", None) is False) if table_obs is not None else True
            edge_quality = dict(block.get("edge_quality") or {})
            summary["edge_quality_mode"] = summary.get("edge_quality_mode") or edge_quality.get("mode")
            summary["edge_conf_threshold_used"] = edge_quality.get("edge_conf_threshold_used")
            summary["locked_edge_conf"] = edge_quality.get("locked_edge_conf")
            summary["locked_yaw_err"] = edge_quality.get("locked_yaw_err")
            summary["locked_dist_err"] = edge_quality.get("locked_dist_err")
            summary["yaw_delta_from_locked"] = edge_quality.get("yaw_delta_from_locked")
            summary["dist_delta_from_locked"] = edge_quality.get("dist_delta_from_locked")
            summary["yaw_delta_from_slide_ref"] = edge_quality.get("yaw_delta_from_slide_ref")
            summary["dist_delta_from_slide_ref"] = edge_quality.get("dist_delta_from_slide_ref")
            summary["edge_identity_basis"] = edge_quality.get("edge_identity_basis")
            summary["handoff_state"] = edge_quality.get("handoff_state") or block.get("handoff_state")
            summary["handoff_samples_count"] = edge_quality.get("handoff_samples_count", block.get("handoff_samples_count"))
            summary["handoff_valid_samples_count"] = edge_quality.get("handoff_valid_samples_count", block.get("handoff_valid_samples_count"))
            summary["slide_ref_ready"] = edge_quality.get("slide_ref_ready", block.get("slide_ref_ready"))
            summary["slide_ref_yaw_err"] = edge_quality.get("slide_ref_yaw_err", block.get("slide_ref_yaw_err"))
            summary["slide_ref_dist_err"] = edge_quality.get("slide_ref_dist_err", block.get("slide_ref_dist_err"))
            summary["slide_ref_edge_conf"] = edge_quality.get("slide_ref_edge_conf", block.get("slide_ref_edge_conf"))
            summary["slide_ref_roi"] = block.get("slide_ref_roi")
            summary["slide_ref_seq"] = block.get("slide_ref_seq")
            summary["full_locked_yaw_err"] = edge_quality.get("full_locked_yaw_err", block.get("full_locked_yaw_err"))
            summary["full_locked_dist_err"] = edge_quality.get("full_locked_dist_err", block.get("full_locked_dist_err"))
            summary["full_vs_light_yaw_offset"] = edge_quality.get("full_vs_light_yaw_offset", block.get("full_vs_light_yaw_offset"))
            summary["full_vs_light_dist_offset"] = edge_quality.get("full_vs_light_dist_offset", block.get("full_vs_light_dist_offset"))
            summary["edge_identity_ok"] = edge_quality.get("edge_identity_ok")
            summary["weak_slide_vy"] = edge_quality.get("weak_slide_vy")
            summary["slide_vy_mps"] = summary.get("slide_vy_mps", edge_quality.get("slide_vy_mps"))
            summary["weak_slide_vy_mps"] = summary.get("weak_slide_vy_mps", edge_quality.get("weak_slide_vy_mps", edge_quality.get("weak_slide_vy")))
            summary["pause_elapsed_ms"] = summary.get("pause_elapsed_ms", edge_quality.get("pause_elapsed_ms"))
            summary["recover_elapsed_ms"] = summary.get("recover_elapsed_ms", edge_quality.get("recover_elapsed_ms"))
            summary["fallback_candidate_state"] = summary.get("fallback_candidate_state") or edge_quality.get("fallback_candidate_state")
            summary["fallback_decision"] = summary.get("fallback_decision", edge_quality.get("fallback_decision"))
            summary["fallback_suppressed_reason"] = edge_quality.get("fallback_suppressed_reason")
            summary["fallback_reason"] = summary.get("reason") or self.core.ctx.last_fail_reason or ""
            summary["control_reason"] = summary.get("reason") or ""
            summary["edge_loss_elapsed_s"] = self.core._loss_elapsed(self.core.ctx.table_loss_since_mono)
        if block.get("stop_reason") and str(summary.get("state")) in {"EDGE_ADJUST", "FINAL_SLOW_STOP"}:
            summary["reason"] = block.get("stop_reason")
        else:
            summary.setdefault("reason", summary.get("lock_reason") or self.core.ctx.last_enter_reason or "")
        return summary

    def _operator_control_line(self, summary: Dict[str, Any]) -> str:
        cmd = dict(summary.get("cmd") or {})
        state = str(summary.get("state") or self.core.ctx.state.value)
        reason = str(summary.get("reason") or summary.get("lock_reason") or "").strip() or "n/a"
        speed_profile = str(summary.get("speed_profile") or "n/a")
        speed_limit = str(summary.get("speed_limit_reason") or "n/a")
        forward_block = str(summary.get("forward_block_reason") or "none")
        vx_mps = summary.get("vx_mps")
        vy_mps = summary.get("vy_mps")
        wz_radps = summary.get("wz_radps")
        if vx_mps is None:
            vx_mps = float(cmd.get("vx", 0.0) or 0.0)
        if vy_mps is None:
            vy_mps = float(cmd.get("vy", 0.0) or 0.0)
        if wz_radps is None:
            wz_radps = float(cmd.get("wz", 0.0) or 0.0)
        base = (
            f"state={state} edge={int(bool(summary.get('edge_found')))} "
            f"conf={self._fmt_float(summary.get('confidence'), 2, signed=False)} "
            f"yaw={self._fmt_float(summary.get('yaw_err_rad'))} "
            f"dist={self._fmt_float(summary.get('dist_err_m'))} "
            f"age={self._fmt_float(summary.get('obs_total_age_ms', summary.get('table_edge_obs_age_ms')), 0, signed=False)}ms "
            f"proc={self._fmt_float(summary.get('vision_process_ms'), 0, signed=False)}ms "
            f"dt={self._fmt_float(summary.get('edge_update_interval_ms'), 0, signed=False)}ms "
            f"stale={summary.get('stale_level') or 'n/a'} "
            f"lock={int(bool(summary.get('lock_ready')))} reason={reason} "
            f"profile={speed_profile} limit={speed_limit} block={forward_block} "
            f"cmd vx_mps={self._fmt_float(cmd.get('vx'))} vy_mps={self._fmt_float(cmd.get('vy'))} wz_radps={self._fmt_float(cmd.get('wz'))} "
            f"vx_mps={self._fmt_float(vx_mps)} vy_mps={self._fmt_float(vy_mps)} wz_radps={self._fmt_float(wz_radps)}"
        )
        if state == "FINAL_SLOW_STOP":
            stable = int(self.core.ctx.table_lock_frames)
            needed = int(self.cfg.control.final_lock_frames_to_arrive)
            return (
                f"[ORCH] LOCK edge={self.core.ctx.current_edge_id} "
                f"conf={self._fmt_float(summary.get('confidence'), 2, signed=False)} "
                f"yaw={self._fmt_float(summary.get('yaw_err_rad'))} "
                f"dist={self._fmt_float(summary.get('dist_err_m'))} stable={stable}/{needed} "
                f"age={self._fmt_float(summary.get('obs_total_age_ms', summary.get('table_edge_obs_age_ms')), 0, signed=False)}ms "
                f"proc={self._fmt_float(summary.get('vision_process_ms'), 0, signed=False)}ms "
                f"dt={self._fmt_float(summary.get('edge_update_interval_ms'), 0, signed=False)}ms "
                f"stale={summary.get('stale_level') or 'n/a'} "
                f"ready={int(bool(summary.get('lock_ready')))} reason={reason} "
                f"profile={speed_profile} limit={speed_limit} block={forward_block} "
                f"cmd vx_mps={self._fmt_float(cmd.get('vx'))} vy_mps={self._fmt_float(cmd.get('vy'))} wz_radps={self._fmt_float(cmd.get('wz'))} "
                f"vx_mps={self._fmt_float(vx_mps)} vy_mps={self._fmt_float(vy_mps)} wz_radps={self._fmt_float(wz_radps)}"
            )
        if state == "EDGE_SLIDE_SEARCH":
            zero_cmd = all(abs(float(cmd.get(key, 0.0) or 0.0)) <= 1e-9 for key in ("vx", "vy", "wz"))
            if zero_cmd:
                reason = self._edge_slide_zero_reason(summary, reason)
            obs = dict(getattr(self, "_last_target_obs_console_payload", {}) or {})
            boxes = obs.get("boxes_count", obs.get("box_count", 0))
            if isinstance(boxes, list):
                boxes = len(boxes)
            found = bool(obs.get("target_found", obs.get("found", False)))
            target_conf = summary.get("target_conf", obs.get("confidence", obs.get("best_conf")))
            matched_cls = summary.get("matched_cls", obs.get("matched_cls", "n/a"))
            matched_conf = summary.get("matched_conf", obs.get("matched_conf", target_conf))
            best_cls = summary.get("best_cls", obs.get("best_cls", "n/a"))
            best_conf = summary.get("best_conf", obs.get("best_conf"))
            edge_valid = summary.get("edge_valid")
            if edge_valid is None:
                edge_valid = summary.get("edge_found", False)
            quality = str(summary.get("edge_quality_mode") or ("stale" if summary.get("edge_obs_is_stale") else "strong"))
            threshold = summary.get("edge_conf_threshold_used")
            return (
                f"[ORCH][SLIDE] "
                f"mode={quality} "
                f"edge_valid={int(bool(edge_valid))} "
                f"conf={self._fmt_float(summary.get('edge_conf', summary.get('confidence')), 2, signed=False)} "
                f"th={self._fmt_float(threshold, 2, signed=False)} "
                f"age={self._fmt_float(summary.get('edge_obs_age_ms'), 0, signed=False)}ms "
                f"yaw={self._fmt_float(summary.get('yaw_err_rad'))} "
                f"dist={self._fmt_float(summary.get('dist_err_m'))} "
                f"target={self.core.ctx.active_target or 'target'} found={int(found)} boxes={int(boxes or 0)} "
                f"matched_cls={matched_cls or 'n/a'} matched_conf={self._fmt_float(matched_conf, 2, signed=False)} "
                f"best_cls={best_cls or 'n/a'} best_conf={self._fmt_float(best_conf, 2, signed=False)} "
                f"vx={self._fmt_float(cmd.get('vx'))} vy={self._fmt_float(cmd.get('vy'))} wz={self._fmt_float(cmd.get('wz'))} "
                f"stale={int(bool(summary.get('edge_obs_is_stale')))} "
                f"reason={reason}"
            )
        return f"[ORCH] CTRL {base}"

    def _edge_slide_zero_reason(self, summary: Dict[str, Any], reason: str) -> str:
        raw = str(reason or summary.get("reason") or "").strip()
        if raw.startswith(("edge_pause", "edge_recover", "edge_follow_stale", "edge_distance_out_of_tolerance")):
            return raw
        if raw in {
            "safety_hold_no_edge",
            "waiting_first_target_obs",
            "edge_slide_vy_zero",
            "target_search_hold",
            "config_disabled",
            "no_table_edge_obs_in_track_local",
        }:
            return raw
        if raw in {"config_zero_vy", "edge_slide_vy_zero"}:
            return "edge_slide_vy_zero"
        if raw in {"target_confirming", "target_track", "stop"}:
            return "target_search_hold"
        if raw in {"edge_missing", "safety_hold_no_edge"}:
            return "safety_hold_no_edge"
        if raw in {"safety_hold", "waiting_target_obs"}:
            if self.core.ctx.last_target_obs is None:
                return "waiting_first_target_obs"
            return "target_search_hold"
        if str(self.core.ctx.confirmed_vision_mode or "").upper() == "FIND_OBJECT" and self.core.ctx.last_table_obs is None:
            return "no_table_edge_obs_in_track_local"
        if not bool(getattr(self.cfg.control, "edge_relocate_enabled", True)):
            return "config_disabled"
        return "target_search_hold"

    def _emit_target_obs_missing_warning(self) -> None:
        if str(getattr(self.core.ctx.state, "value", self.core.ctx.state) or "") != "EDGE_SLIDE_SEARCH":
            return
        try:
            elapsed = self.core._state_elapsed()
        except Exception:
            elapsed = 0.0
        if elapsed < 1.0:
            return
        try:
            fresh = self.core._fresh_target_obs()
        except Exception:
            fresh = self.core.ctx.last_target_obs
        if fresh is not None:
            return
        last_obs = getattr(self.core.ctx, "last_target_obs", None)
        if last_obs is not None and getattr(last_obs, "ts", 0.0):
            age = max(0.0, time.time() - float(getattr(last_obs, "ts", 0.0) or 0.0))
        else:
            age = max(0.0, float(elapsed))
        desired = str(getattr(self.core.ctx, "desired_vision_mode", "") or "n/a").strip() or "n/a"
        confirmed = str(getattr(self.core.ctx, "confirmed_vision_mode", "") or "n/a").strip() or "n/a"
        self.operator_console.emit_rate_limited(
            "target_obs_missing",
            f"[ORCH] WARN target_obs_missing desired_mode={desired} confirmed_mode={confirmed} age={age:.1f}s",
            self.operator_console.default_interval_s,
        )

    def _emit_operator_control(self, decision) -> None:
        summary = self._control_summary_with_context(decision)
        self.run_logger.write_jsonl("control_summary", summary)
        self._emit_motion_gate_trace(summary)
        state = str(summary.get("state") or self.core.ctx.state.value)
        self._emit_demo_health(summary)
        if state in {"IDLE", "DONE", "ERROR_RECOVERY"} and not self.operator_console.full:
            return
        line = self._operator_control_line(summary)
        key = "lock" if state == "FINAL_SLOW_STOP" else ("slide" if state == "EDGE_SLIDE_SEARCH" else "ctrl")
        period_s = 5.0
        self.operator_console.emit_rate_limited(key, line, period_s)
        self._emit_target_obs_missing_warning()

    @staticmethod
    def _motion_num(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _emit_motion_gate_trace(self, summary: Dict[str, Any]) -> None:
        cmd = dict(summary.get("cmd") or {})
        vx = self._motion_num(summary.get("vx_mps", cmd.get("vx", 0.0)))
        wz = self._motion_num(summary.get("wz_radps", cmd.get("wz", 0.0)))
        vy = self._motion_num(summary.get("vy_mps", cmd.get("vy", 0.0)))
        cx_norm = summary.get("bbox_cx_norm")
        if cx_norm is None:
            cx_norm = summary.get("yolo_bbox_center_x_norm")
        if cx_norm is None:
            raw_cx = summary.get("table_cx_norm")
            cx_norm = (self._motion_num(raw_cx) + 1.0) * 0.5 if raw_cx is not None else None
        center_error = summary.get("center_error")
        if center_error is None and cx_norm is not None:
            center_error = self._motion_num(cx_norm) - 0.5
        yaw_err = self._motion_num(summary.get("yaw_err_rad", summary.get("yaw_err", 0.0)))
        dist_err = self._motion_num(summary.get("dist_err_m", 0.0))
        target_dist = self._motion_num(summary.get("target_dist_m", getattr(self.cfg.control, "table_target_dist_m", 0.5)), 0.5)
        hard_yaw = abs(self._motion_num(summary.get("hard_rotate_only_yaw_rad", getattr(self.cfg.car, "table_edge_hard_rotate_only_yaw_rad", 0.45)), 0.45))
        tolerance = max(0.0, self._motion_num(getattr(self.cfg.control, "final_lock_dist_tol_m", 0.02), 0.02))
        yolo_visible = bool(summary.get("yolo_table_visible") or summary.get("table_bbox_found"))
        bbox_valid = bool(summary.get("table_bbox_control_valid") or summary.get("yolo_table_control_valid") or summary.get("yolo_reliable"))
        edge_trusted = bool(summary.get("edge_trusted") or summary.get("valid_for_control"))
        
        yaw_source = summary.get("yaw_source") or ""
        forward_source = summary.get("forward_source") or ""
        stop_source = summary.get("stop_source") or ""
        block_reason = summary.get("block_reason") or summary.get("forward_block_reason") or ""

        trace = {
            "ts": time.time(),
            "state": str(summary.get("state") or self.core.ctx.state.value),
            "control_source": summary.get("control_source") or "",
            "control_phase": summary.get("control_phase") or "",
            "phase_reason": summary.get("phase_reason") or "",
            "yaw_source": yaw_source,
            "forward_source": forward_source,
            "stop_source": stop_source,
            "yolo_table_visible": yolo_visible,
            "table_bbox_control_valid": bbox_valid,
            "edge_found": bool(summary.get("edge_found")),
            "edge_valid": bool(summary.get("edge_valid")),
            "edge_trusted": edge_trusted,
            "edge_control_allowed": bool(summary.get("edge_control_allowed") or summary.get("usable_for_approach")),
            "edge_stable_count": summary.get("yolo_table_edge_stable_count"),
            "cx_norm": cx_norm,
            "center_error": center_error,
            "yaw_err_rad": yaw_err,
            "dist_err_m": dist_err,
            "target_dist_m": target_dist,
            "stale_level": summary.get("stale_level") or "",
            "allow_forward": bool(summary.get("allow_forward", summary.get("forward_allowed", False))),
            "allow_rotate": bool(summary.get("allow_rotate", abs(wz) > 1e-9)),
            "vx_mps": vx,
            "vy_mps": vy,
            "wz_radps": wz,
            "forward_block_reason": summary.get("forward_block_reason") or "",
            "rotate_block_reason": summary.get("rotate_block_reason") or "",
            "block_reason": block_reason,
            "table_roi_depth_valid": bool(summary.get("table_roi_depth_valid", False)),
            "table_roi_depth_p10": summary.get("table_roi_depth_p10", summary.get("roi_depth_stat")),
            "table_roi_depth_median": summary.get("table_roi_depth_median"),
            "table_roi_depth_valid_ratio": summary.get("table_roi_depth_valid_ratio", summary.get("roi_depth_valid_ratio")),
            "table_roi_depth_sample_count": summary.get("table_roi_depth_sample_count", summary.get("roi_depth_sample_count")),
            "table_roi_depth_coord_space": summary.get("table_roi_depth_coord_space", ""),
            "roi_depth_window_ready": bool(summary.get("roi_depth_window_ready", False)),
            "transition_reason": summary.get("transition_reason") or summary.get("reason") or "",
            "bbox_wz_sign": summary.get("bbox_wz_sign", 0),
            "edge_wz_sign": summary.get("edge_wz_sign", 0),
            "yaw_conflict": bool(summary.get("yaw_conflict", False)),
            "search_wz_sign_latched": summary.get("search_wz_sign_latched", 0),
            "handoff_complete": bool(summary.get("edge_handoff_complete", False)),
            "handoff_timeout": bool(summary.get("handoff_timeout", False)),
            "phase_dwell_ms": summary.get("phase_dwell_ms", 0.0),
            "bbox_cx_norm_control": summary.get("bbox_cx_norm_control"),
            "bbox_center_error_control": summary.get("bbox_center_error_control"),
            "bbox_center_source": summary.get("bbox_center_source", ""),
            "bbox_xyxy_for_control": summary.get("bbox_xyxy_for_control"),
            "bbox_yaw_cmd": summary.get("bbox_yaw_cmd", 0.0),
            "bbox_lost_hold_active": bool(summary.get("bbox_lost_hold_active", False)),
            "bbox_lost_hold_age_ms": summary.get("bbox_lost_hold_age_ms", 0.0),
            "phase_reset_reason": summary.get("phase_reset_reason", ""),
            "search_latch_age_ms": summary.get("search_latch_age_ms", 0.0),
            "search_latch_reason": summary.get("search_latch_reason", ""),
            "wz_sign_final": summary.get("wz_sign_final", 0),
        }
        self.run_logger.write_jsonl("motion_gate_trace", trace)
        sig = (
            trace["state"],
            trace["control_source"],
            round(vx, 3),
            round(vy, 3),
            round(wz, 3),
            trace["block_reason"],
        )
        now = time.time()
        last_sig = getattr(self, "_last_motion_gate_trace_sig", None)
        last_ts = float(getattr(self, "_last_motion_gate_trace_ts", 0.0) or 0.0)
        if sig != last_sig or now - last_ts >= 1.0:
            self._last_motion_gate_trace_sig = sig
            self._last_motion_gate_trace_ts = now
            self.operator_console.emit_rate_limited(
                "motion_gate_trace",
                "[MOTION_GATE_TRACE] "
                f"state={trace['state']} control_source={trace['control_source']} "
                f"control_phase={trace['control_phase']} phase_reason={trace['phase_reason']} "
                f"search_latch={trace['search_wz_sign_latched']} latch_age_ms={trace['search_latch_age_ms']:.0f} wz_sign_final={trace['wz_sign_final']} "
                f"yaw_source={yaw_source} forward_source={forward_source} stop_source={stop_source} "
                f"allow_forward={int(trace['allow_forward'])} allow_rotate={int(trace['allow_rotate'])} "
                f"vx_mps={vx:.3f} vy_mps={vy:.3f} wz_radps={wz:.3f} block_reason={block_reason} "
                f"forward_block_reason={trace['forward_block_reason']} rotate_block_reason={trace['rotate_block_reason']} "
                f"roi_p10={trace['table_roi_depth_p10']} roi_ratio={trace['table_roi_depth_valid_ratio']} roi_samples={trace['table_roi_depth_sample_count']}",
                interval_s=0.5,
            )
        should_forward = bool(
            (yolo_visible or bbox_valid or edge_trusted)
            and dist_err > target_dist + tolerance
            and abs(yaw_err) < hard_yaw
            and abs(vx) <= 1e-9
            and str(trace["forward_block_reason"]).lower() not in {"soft_stale", "hard_stale", "dead", "vision_stale"}
            and str(trace["stale_level"]).lower() not in {"soft_stale", "hard_stale", "dead"}
        )
        if should_forward:
            self.run_logger.write_jsonl("motion_forward_block_bug", trace)
            self.operator_console.emit_rate_limited(
                "motion_forward_block_bug",
                "[MOTION_FORWARD_BLOCK_BUG] "
                f"reason={trace['forward_block_reason']} state={trace['state']} control_source={trace['control_source']} "
                f"dist_err_m={dist_err:.3f} target_dist_m={target_dist:.3f} yaw_err_rad={yaw_err:.3f} hard_rotate_only_yaw_rad={hard_yaw:.3f}",
                interval_s=0.5,
            )

    def _rate_hz(self, samples, window_s: float = 5.0) -> float:
        now = time.time()
        cutoff = now - max(1.0, float(window_s))
        recent = [float(ts) for ts in list(samples) if float(ts) >= cutoff]
        return float(len(recent)) / max(1.0, float(window_s))

    @staticmethod
    def _epoch_ms(ts: Any) -> int:
        try:
            return int(round(float(ts) * 1000.0))
        except Exception:
            return int(round(time.time() * 1000.0))

    @staticmethod
    def _obs_key(obs: Optional[TableEdgeObs]):
        if obs is None:
            return None
        return (
            getattr(obs, "obs_seq", None),
            getattr(obs, "camera_frame_seq", None),
            getattr(obs, "seq", None),
            getattr(obs, "frame_id", None),
            getattr(obs, "obs_ts", None),
        )

    @staticmethod
    def _percentile(values, p: float) -> Optional[float]:
        nums = sorted(float(v) for v in values if v is not None)
        if not nums:
            return None
        idx = int(round((len(nums) - 1) * max(0.0, min(1.0, float(p)))))
        return nums[idx]

    def _observe_trace_sample(self, key: str, value: Any) -> None:
        try:
            number = float(value)
        except Exception:
            return
        samples = self._obs_trace_samples.get(key)
        if samples is not None:
            samples.append(number)

    def _trace_stats(self, key: str) -> Dict[str, Optional[float]]:
        values = list(self._obs_trace_samples.get(key) or [])
        return {
            "p50": self._percentile(values, 0.50),
            "p90": self._percentile(values, 0.90),
            "max": max(values) if values else None,
        }

    def _mark_state_machine_consume(self, loop_start: float) -> None:
        now = time.time()
        tick_interval_ms = (
            max(0.0, (now - float(self._last_state_machine_tick_ts)) * 1000.0)
            if float(self._last_state_machine_tick_ts or 0.0) > 0.0
            else None
        )
        self._last_state_machine_tick_ts = now
        if tick_interval_ms is not None:
            self._observe_trace_sample("state_machine_tick_interval_ms", tick_interval_ms)
        obs = self.core.ctx.last_table_obs
        if obs is None:
            return
        consume_interval_ms = (
            max(0.0, (now - float(self._last_state_machine_consume_ts)) * 1000.0)
            if float(self._last_state_machine_consume_ts or 0.0) > 0.0
            else None
        )
        self._last_state_machine_consume_ts = now
        key = self._obs_key(obs)
        is_new_obs = key != self._last_consumed_table_obs_key
        if not is_new_obs:
            self._same_obs_reuse_count += 1
        else:
            self._last_consumed_table_obs_key = key
            self._same_obs_reuse_count = 0
        obs_seq = getattr(obs, "obs_seq", None)
        seq_gap = None
        try:
            if obs_seq is not None and self._last_consumed_obs_seq is not None:
                seq_gap = int(obs_seq) - int(self._last_consumed_obs_seq)
            if obs_seq is not None and is_new_obs:
                self._last_consumed_obs_seq = int(obs_seq)
        except Exception:
            seq_gap = None
        if obs_seq is not None and self._last_consumed_obs_seq is None:
            try:
                self._last_consumed_obs_seq = int(obs_seq)
            except Exception:
                pass
        consume_ms = self._epoch_ms(now)
        obs.state_machine_consume_ts_ms = consume_ms
        obs.state_machine_tick_interval_ms = tick_interval_ms
        obs.state_machine_consume_interval_ms = consume_interval_ms
        obs.same_obs_reuse_count = int(self._same_obs_reuse_count)
        obs.obs_seq_gap = seq_gap
        if getattr(obs, "frame_capture_ts", None) is not None:
            try:
                obs.obs_age_at_consume_ms = max(0.0, (now - float(obs.frame_capture_ts)) * 1000.0)
            except Exception:
                pass
        if getattr(obs, "obs_recv_ts", None) is not None:
            try:
                obs.orch_recv_to_state_consume_ms = max(0.0, (now - float(obs.obs_recv_ts)) * 1000.0)
            except Exception:
                pass
        if consume_interval_ms is not None:
            self._observe_trace_sample("state_machine_consume_interval_ms", consume_interval_ms)
        self._observe_trace_sample("obs_age_at_consume_ms", getattr(obs, "obs_age_at_consume_ms", None))
        self._observe_trace_sample("orch_recv_to_state_consume_ms", getattr(obs, "orch_recv_to_state_consume_ms", None))
        self._emit_obs_frequency_summary_if_needed()

    def _mark_cmd_publish(self, decision) -> None:
        now_ms = self._epoch_ms(time.time())
        obs = self.core.ctx.last_table_obs
        if obs is not None:
            obs.cmd_publish_ts_ms = now_ms
        summary = getattr(decision, "control_summary", None)
        if isinstance(summary, dict):
            summary["cmd_publish_ts_ms"] = now_ms

    def _emit_obs_frequency_summary_if_needed(self, force: bool = False) -> None:
        now = time.time()
        period_s = 5.0
        if not force and (now - float(self._obs_freq_last_emit_ts or 0.0)) < period_s:
            return
        self._obs_freq_last_emit_ts = now
        recv_hz = self._rate_hz(self._edge_obs_rate_ts, window_s=period_s)
        target_hz = self._rate_hz(self._target_obs_rate_ts, window_s=period_s)
        obs = self.core.ctx.last_table_obs
        record = {
            "ts": now,
            "table_edge_obs_recv_hz": float(recv_hz),
            "target_obs_recv_hz": float(target_hz),
            "same_obs_reuse_count": int(self._same_obs_reuse_count),
            "obs_seq": getattr(obs, "obs_seq", None) if obs is not None else None,
            "camera_frame_seq": getattr(obs, "camera_frame_seq", None) if obs is not None else None,
            "seq": getattr(obs, "seq", None) if obs is not None else None,
            "state": self.core.ctx.state.value,
            "tick_hz_config": float(self.cfg.runtime.tick_hz),
            "receiver_poll_interval_ms_config": int(round((1.0 / max(1.0, float(self.cfg.runtime.tick_hz))) * 1000.0)),
            "status_publish_hz_config": 1.0 / max(1e-6, float(self.cfg.runtime.state_block_period_s or 1.0)),
        }
        for key in self._obs_trace_samples:
            record[key] = self._trace_stats(key)
        self.run_logger.write_jsonl("obs_frequency_summary", record)
        self.operator_console.emit_rate_limited(
            "obs_frequency_summary",
            "[ORCH][OBS_FREQ] "
            f"edge_recv_hz={self._fmt_float(recv_hz, 1, signed=False)} "
            f"target_recv_hz={self._fmt_float(target_hz, 1, signed=False)} "
            f"tick_ms_p50={self._fmt_float(record['state_machine_tick_interval_ms'].get('p50'), 0, signed=False)} "
            f"consume_age_p50={self._fmt_float(record['obs_age_at_consume_ms'].get('p50'), 0, signed=False)} "
            f"same_reuse={int(self._same_obs_reuse_count)}",
            interval_s=period_s,
        )

    def _emit_demo_health(self, summary: Dict[str, Any]) -> None:
        if str(getattr(self.demo_console, "level", "normal")) == "demo":
            return
        state = str(summary.get("state") or self.core.ctx.state.value)
        if state == "IDLE":
            self.demo_console.emit_change("idle_waiting", "[DEMO][IDLE] waiting for command")
            self.demo_console.health(
                {
                    "state": "IDLE",
                    "target": "n/a",
                    "edge": "OFF",
                    "yolo": "OFF",
                    "preview": os.getenv("ORCH_DEMO_PREVIEW_STATUS", "n/a") or "n/a",
                    "dry_run": bool(self.cfg.serial.dry_run),
                    "interval_s": 10.0,
                }
            )
            return
        edge_ok = bool(summary.get("edge_found") or summary.get("edge_valid"))
        if bool(summary.get("edge_obs_is_stale")):
            edge = "STALE"
        else:
            edge = "OK" if edge_ok else "MISS"
        self.demo_console.health(
            {
                "state": state,
                "target": self.core.ctx.active_target or "n/a",
                "edge": edge,
                "edge_hz": self._rate_hz(self._edge_obs_rate_ts),
                "yolo_hz": self._rate_hz(self._target_obs_rate_ts),
                "preview": os.getenv("ORCH_DEMO_PREVIEW_STATUS", "n/a") or "n/a",
                "dry_run": bool(self.cfg.serial.dry_run),
            }
        )
        if state == "EDGE_SLIDE_SEARCH":
            cmd = dict(summary.get("cmd") or {})
            if summary.get("stop_reason") == "edge_distance_out_of_tolerance":
                dist = summary.get("dist_err_m")
                tol = summary.get("dist_tolerance_m", self.cfg.control.edge_slide_dist_tolerance_m)
                try:
                    dist_text = f"{float(dist):+.3f}m"
                except Exception:
                    dist_text = "n/a"
                try:
                    tol_text = f"{float(tol):.3f}m"
                except Exception:
                    tol_text = "n/a"
                self.demo_console.warning(f"edge distance out of tolerance: dist={dist_text} tol={tol_text}, re-locking edge")
            if any(abs(float(cmd.get(k, 0.0) or 0.0)) > 1e-6 for k in ("vx", "vy", "wz")):
                self.demo_console.slide_intent(cmd.get("vx"), cmd.get("vy"), cmd.get("wz"))

    def _emit_motion(self, decision):
        if getattr(decision, "arm_cmd", None) is not None:
            arm = decision.arm_cmd
            arm_line = encode_pose(
                arm.x_cm, arm.y_cm, arm.z_cm,
                arm.pitch_deg, arm.roll_deg,
                arm.claw_deg, arm.time_ms,
            )
            self.motion_adapter.cancel_active_jogs()
            self.uart.send_arm_command(arm_line)
            self.run_logger.write_jsonl("arm_cmd", arm.to_dict())
            return
        jog_action = str(getattr(decision, "jog_action", "") or "").strip().lower()
        if jog_action:
            self._emit_jog_motion(decision, jog_action)
            return
        cmd = decision.cmd
        self._emit_operator_control(decision)
        self._flush_state_traces(decision)
        summary = dict(getattr(decision, "control_summary", None) or {})
        effective_cmd, uart_arbitration = self._arbitrate_uart_motion_cmd(cmd, summary)
        car_cmd = self.mapper.from_cmd_vel(effective_cmd, cx_norm_abs=decision.cx_norm_abs, distance_ratio=decision.distance_ratio)
        tx_meta = self._build_uart_tx_meta(car_cmd)
        reason = str(tx_meta.get("reason") or car_cmd.kind or "").strip()
        velocity = self.motion_adapter.cmd_vel_to_velocity(effective_cmd)
        speed_profile = str(summary.get("speed_profile") or ("stop" if car_cmd.kind in {"stop", "brake"} else "search"))
        self._last_motion_tx_context = {
            "speed_profile": speed_profile,
            "speed_limit_reason": summary.get("speed_limit_reason") or "",
            "forward_block_reason": summary.get("forward_block_reason") or "",
            "table_approach_phase": summary.get("table_approach_phase") or "",
            "vx_mps": float(getattr(effective_cmd, "vx_mps", 0.0) or 0.0),
            "vy_mps": float(getattr(effective_cmd, "vy_mps", 0.0) or 0.0),
            "wz_radps": float(getattr(effective_cmd, "wz_radps", 0.0) or 0.0),
            "vx_mps": float(velocity[0]),
            "vy_mps": float(velocity[1]),
            "wz_radps": float(velocity[2]),
        }
        self._last_motion_tx_context.update(uart_arbitration)
        tx_meta.update(uart_arbitration)
        seq = self.motion_adapter.send_cmd_vel(effective_cmd, reason=reason)
        self.motion_status["last_seq"] = seq
        self.motion_status["jog_running"] = False
        car_record = {
            "ts": time.time(),
            "mode": car_cmd.mode,
            "kind": car_cmd.kind,
            "vx_mps": car_cmd.vx_mps,
            "vy_mps": car_cmd.vy_mps,
            "wz_radps": car_cmd.wz_radps,
            "hold_ms": car_cmd.hold_ms,
            "brake": car_cmd.brake,
            "stm32_seq": seq,
            "stm32_velocity": {
                "vx_mps": velocity[0],
                "vy_mps": velocity[1],
                "wz_radps": velocity[2],
            },
            "raw": "STOP" if car_cmd.kind in {"stop", "brake"} else f"V {velocity[0]:.3f} {velocity[1]:.3f} {velocity[2]:.3f}",
            "legacy_raw": car_cmd.raw_line.rstrip("\r\n"),
            "speed_profile": speed_profile,
            "speed_limit_reason": self._last_motion_tx_context["speed_limit_reason"],
            "forward_block_reason": self._last_motion_tx_context["forward_block_reason"],
            "uart_emit_reason": uart_arbitration.get("uart_emit_reason"),
            "last_valid_motion_cmd": uart_arbitration.get("last_valid_motion_cmd"),
            "last_valid_motion_age_ms": uart_arbitration.get("last_valid_motion_age_ms"),
            "soft_stale_hold_active": uart_arbitration.get("soft_stale_hold_active"),
            "zero_cmd_reason": uart_arbitration.get("zero_cmd_reason"),
        }
        car_record.update({k: v for k, v in tx_meta.items() if v not in (None, "")})
        log_now = time.time()
        motion_signature = self._motion_log_signature(effective_cmd, car_cmd, reason)
        if self._should_log_motion(motion_signature, now=log_now):
            cmd_record = effective_cmd.to_dict()
            cmd_record.update(self._last_motion_tx_context)
            self.run_logger.write_jsonl("cmd_vel", cmd_record)
            self.run_logger.write_jsonl("car_cmd", car_record)
        self._emit_edge_slide_trace(decision)

    def _emit_jog_motion(self, decision, jog_action: str) -> None:
        cmd = decision.cmd
        self.run_logger.write_jsonl("cmd_vel", cmd.to_dict())
        summary = dict(getattr(decision, "control_summary", None) or {})
        self.run_logger.write_jsonl("control_summary", summary)
        reason = str(getattr(decision, "jog_reason", "") or summary.get("reason") or jog_action)
        if jog_action == "forward":
            seq = self.motion_adapter.jog_forward_small(reason=reason)
        elif jog_action == "backward":
            seq = self.motion_adapter.jog_backward_small(reason=reason)
        elif jog_action == "turn_left":
            seq = self.motion_adapter.jog_turn_left_small(reason=reason)
        elif jog_action == "turn_right":
            seq = self.motion_adapter.jog_turn_right_small(reason=reason)
        else:
            self.log("warn", "runtime.service", f"unknown jog_action={jog_action}")
            return
        self.motion_status["last_seq"] = seq
        self.motion_status["jog_running"] = True
        self.run_logger.write_jsonl("car_cmd", {
            "ts": time.time(),
            "mode": str(getattr(cmd, "mode", "FINAL_SLOW_STOP") or "FINAL_SLOW_STOP"),
            "kind": "stm32_jog",
            "jog_action": jog_action,
            "stm32_seq": seq,
            "duration_ms": int(self.cfg.car.jog_duration_ms),
            "reason": reason,
        })

    def _emit_edge_slide_trace(self, decision) -> None:
        if str(getattr(self.core.ctx.state, "value", self.core.ctx.state) or "") != "EDGE_SLIDE_SEARCH":
            self._last_edge_slide_trace_signature = None
            return
        now = time.time()
        edge_obs = self.core.ctx.last_table_obs
        target_obs = self.core.ctx.last_target_obs
        cmd = decision.cmd
        decision_summary = dict(getattr(decision, "control_summary", None) or {})
        reason = str(decision_summary.get("reason") or self.core.ctx.last_fail_reason or "")
        edge_age_ms = self.core._table_obs_age_ms(edge_obs)
        edge_is_stale = self.core._edge_obs_is_stale(edge_obs)
        edge_key = (
            getattr(edge_obs, "source_mode", None),
            getattr(edge_obs, "frame_id", None),
            getattr(edge_obs, "seq", None),
        ) if edge_obs is not None else None
        edge_obs_ts = getattr(edge_obs, "obs_ts", None) if edge_obs is not None else None
        if edge_key is not None and edge_key != getattr(self, "_last_edge_slide_obs_key", None):
            previous_ts = getattr(self, "_last_edge_slide_obs_ts", None)
            try:
                if previous_ts is not None and edge_obs_ts is not None:
                    self._last_edge_slide_obs_period_ms = max(0.0, (float(edge_obs_ts) - float(previous_ts)) * 1000.0)
            except Exception:
                pass
            self._last_edge_slide_obs_key = edge_key
            self._last_edge_slide_obs_ts = edge_obs_ts
        edge_obs_period_ms = getattr(edge_obs, "edge_update_interval_ms", None) if edge_obs is not None else None
        if edge_obs_period_ms is None:
            edge_obs_period_ms = getattr(self, "_last_edge_slide_obs_period_ms", None)
        edge_valid = bool(getattr(edge_obs, "edge_valid", getattr(edge_obs, "edge_found", False))) if edge_obs is not None else False
        edge_conf = getattr(edge_obs, "edge_conf", None) if edge_obs is not None else None
        if edge_conf is None and edge_obs is not None:
            edge_conf = getattr(edge_obs, "confidence", None)
        matched_cls = (
            getattr(target_obs, "matched_cls", None) or getattr(target_obs, "target", None)
            if target_obs is not None
            else None
        )
        record = {
            "ts": now,
            "state": "EDGE_SLIDE_SEARCH",
            "target": self.core.ctx.active_target,
            "edge_valid": edge_valid,
            "edge_found": bool(getattr(edge_obs, "edge_found", False)) if edge_obs is not None else False,
            "edge_conf": edge_conf,
            "dist_err": getattr(edge_obs, "dist_err_m", None) if edge_obs is not None else None,
            "yaw_err": getattr(edge_obs, "yaw_err_rad", None) if edge_obs is not None else None,
            "edge_obs_ts": getattr(edge_obs, "obs_ts", None) if edge_obs is not None else None,
            "edge_obs_age_ms": edge_age_ms,
            "edge_obs_is_stale": bool(edge_is_stale),
            "edge_follow_stale": bool(edge_is_stale),
            "edge_obs_unavailable": bool(edge_obs is None or getattr(edge_obs, "depth_valid", None) is False),
            "edge_obs_frame_id": getattr(edge_obs, "frame_id", None) if edge_obs is not None else None,
            "edge_obs_seq": getattr(edge_obs, "seq", None) if edge_obs is not None else None,
            "edge_update_interval_ms": edge_obs_period_ms,
            "edge_obs_period_ms": edge_obs_period_ms,
            "edge_process_ms": getattr(edge_obs, "edge_process_ms", None) if edge_obs is not None else None,
            "edge_obs_source_mode": getattr(edge_obs, "source_mode", None) if edge_obs is not None else None,
            "edge_quality_mode": (self.core.ctx.last_edge_quality or {}).get("mode"),
            "edge_quality_raw_mode": (self.core.ctx.last_edge_quality or {}).get("raw_mode"),
            "edge_conf_threshold_used": (self.core.ctx.last_edge_quality or {}).get("edge_conf_threshold_used"),
            "locked_edge_conf": (self.core.ctx.last_edge_quality or {}).get("locked_edge_conf"),
            "locked_yaw_err": (self.core.ctx.last_edge_quality or {}).get("locked_yaw_err"),
            "locked_dist_err": (self.core.ctx.last_edge_quality or {}).get("locked_dist_err"),
            "yaw_delta_from_locked": (self.core.ctx.last_edge_quality or {}).get("yaw_delta_from_locked"),
            "dist_delta_from_locked": (self.core.ctx.last_edge_quality or {}).get("dist_delta_from_locked"),
            "handoff_state": (self.core.ctx.last_edge_quality or {}).get("handoff_state", getattr(self.core.ctx, "handoff_state", "")),
            "handoff_samples_count": (self.core.ctx.last_edge_quality or {}).get("handoff_samples_count", len(getattr(self.core.ctx, "slide_ref_samples", []) or [])),
            "handoff_valid_samples_count": (self.core.ctx.last_edge_quality or {}).get("handoff_valid_samples_count", len(getattr(self.core.ctx, "slide_ref_samples", []) or [])),
            "slide_ref_ready": (self.core.ctx.last_edge_quality or {}).get("slide_ref_ready", getattr(self.core.ctx, "slide_ref_ready", False)),
            "slide_ref_yaw_err": (self.core.ctx.last_edge_quality or {}).get("slide_ref_yaw_err", getattr(self.core.ctx, "slide_ref_yaw_err", None)),
            "slide_ref_dist_err": (self.core.ctx.last_edge_quality or {}).get("slide_ref_dist_err", getattr(self.core.ctx, "slide_ref_dist_err", None)),
            "slide_ref_edge_conf": (self.core.ctx.last_edge_quality or {}).get("slide_ref_edge_conf", getattr(self.core.ctx, "slide_ref_edge_conf", None)),
            "slide_ref_roi": getattr(self.core.ctx, "slide_ref_roi", None),
            "slide_ref_seq": getattr(self.core.ctx, "slide_ref_seq", None),
            "full_locked_yaw_err": (self.core.ctx.last_edge_quality or {}).get("full_locked_yaw_err", getattr(self.core.ctx, "locked_yaw_err", None)),
            "full_locked_dist_err": (self.core.ctx.last_edge_quality or {}).get("full_locked_dist_err", getattr(self.core.ctx, "locked_dist_err", None)),
            "full_vs_light_yaw_offset": (self.core.ctx.last_edge_quality or {}).get("full_vs_light_yaw_offset"),
            "full_vs_light_dist_offset": (self.core.ctx.last_edge_quality or {}).get("full_vs_light_dist_offset"),
            "yaw_delta_from_slide_ref": (self.core.ctx.last_edge_quality or {}).get("yaw_delta_from_slide_ref"),
            "dist_delta_from_slide_ref": (self.core.ctx.last_edge_quality or {}).get("dist_delta_from_slide_ref"),
            "edge_identity_basis": (self.core.ctx.last_edge_quality or {}).get("edge_identity_basis"),
            "edge_identity_ok": (self.core.ctx.last_edge_quality or {}).get("edge_identity_ok"),
            "weak_slide_vy": (self.core.ctx.last_edge_quality or {}).get("weak_slide_vy"),
            "control_reason": reason,
            "stop_reason": decision_summary.get("stop_reason") or (self.core.ctx.last_edge_quality or {}).get("reason") or "",
            "slide_vy_mps": decision_summary.get("slide_vy_mps", (self.core.ctx.last_edge_quality or {}).get("slide_vy_mps")),
            "weak_slide_vy_mps": decision_summary.get("weak_slide_vy_mps", (self.core.ctx.last_edge_quality or {}).get("weak_slide_vy_mps")),
            "vx_from_dist": decision_summary.get("vx_from_dist"),
            "wz_from_yaw": decision_summary.get("wz_from_yaw"),
            "final_vx": decision_summary.get("final_vx", float(cmd.vx_mps)),
            "final_vy": decision_summary.get("final_vy", float(cmd.vy_mps)),
            "final_wz": decision_summary.get("final_wz", float(cmd.wz_radps)),
            "pause_elapsed_ms": decision_summary.get("pause_elapsed_ms", (self.core.ctx.last_edge_quality or {}).get("pause_elapsed_ms")),
            "recover_elapsed_ms": decision_summary.get("recover_elapsed_ms", (self.core.ctx.last_edge_quality or {}).get("recover_elapsed_ms")),
            "fallback_candidate_state": decision_summary.get("fallback_candidate_state", (self.core.ctx.last_edge_quality or {}).get("fallback_candidate_state")),
            "fallback_decision": decision_summary.get("fallback_decision", (self.core.ctx.last_edge_quality or {}).get("fallback_decision")),
            "fallback_suppressed_reason": (self.core.ctx.last_edge_quality or {}).get("fallback_suppressed_reason"),
            "target_found": bool(getattr(target_obs, "found", False)) if target_obs is not None else False,
            "target_conf": (
                getattr(target_obs, "matched_conf", None)
                if target_obs is not None and getattr(target_obs, "matched_conf", None) is not None
                else (getattr(target_obs, "confidence", None) if target_obs is not None and getattr(target_obs, "confidence", None) is not None else None)
            ),
            "matched_conf": (
                getattr(target_obs, "matched_conf", None)
                if target_obs is not None
                else None
            ),
            "matched_cls": matched_cls,
            "target_cls": matched_cls,
            "best_cls": getattr(target_obs, "best_cls", None) if target_obs is not None else None,
            "best_conf": getattr(target_obs, "best_conf", None) if target_obs is not None else None,
            "vx": float(cmd.vx_mps),
            "vy": float(cmd.vy_mps),
            "wz": float(cmd.wz_radps),
            "fallback_reason": reason,
            "edge_loss_elapsed_s": self.core._loss_elapsed(self.core.ctx.table_loss_since_mono),
            "keep_dist_tolerance_m": float(self.cfg.control.edge_slide_dist_tolerance_m),
            "edge_lost_hold_s": float(self.cfg.control.table_loss_hold_s),
            "stale_hold_s": float(getattr(self.cfg.control, "edge_follow_stale_hold_s", self.cfg.control.table_loss_hold_s)),
            "fallback_state": str(getattr(self.cfg.control, "edge_slide_fallback_state", "CONTROLLED_APPROACH") or "CONTROLLED_APPROACH"),
        }
        trace_signature = (
            reason,
            self._motion_float_sig(cmd.vx_mps),
            self._motion_float_sig(cmd.vy_mps),
            self._motion_float_sig(cmd.wz_radps),
            bool(edge_is_stale),
            bool(edge_obs is None or not edge_valid),
            record.get("fallback_decision"),
            record.get("fallback_suppressed_reason"),
            record.get("stop_reason"),
        )
        try:
            period_ms = float(getattr(self.cfg.control, "edge_follow_log_period_ms", 500.0) or 500.0)
        except (TypeError, ValueError):
            period_ms = 500.0
        period_s = max(0.0, period_ms / 1000.0)
        previous_signature = getattr(self, "_last_edge_slide_trace_signature", None)
        changed = trace_signature != previous_signature
        periodic = (now - float(getattr(self, "_last_edge_slide_trace_ts", 0.0) or 0.0)) >= period_s
        if not changed and not periodic:
            return
        self._last_edge_slide_trace_signature = trace_signature
        self._last_edge_slide_trace_ts = now
        self.run_logger.write_jsonl("edge_slide_search", record)

    def _flush_state_traces(self, decision) -> None:
        reset_traces = list(getattr(self.core, "_pending_reset_traces", []) or [])
        if reset_traces:
            self.core._pending_reset_traces.clear()
        if not self._pending_state_traces and not reset_traces:
            return
        cmd = decision.cmd
        planned_cmd = {
            "vx": float(cmd.vx_mps),
            "vy": float(cmd.vy_mps),
            "wz": float(cmd.wz_radps),
        }
        for trace in reset_traces:
            trace["planned_cmd"] = planned_cmd
            trace["planned_cmd_mode"] = cmd.mode
            self.run_logger.write_jsonl("state_trace", self._normalize_state_trace(trace))
        while self._pending_state_traces:
            trace = self._pending_state_traces.pop(0)
            trace["planned_cmd"] = planned_cmd
            trace["planned_cmd_mode"] = cmd.mode
            self.run_logger.write_jsonl("state_trace", self._normalize_state_trace(trace))

    def _normalize_state_trace(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        fixed_fields = (
            "event",
            "previous_state",
            "next_state",
            "reason",
            "target",
            "session_id",
            "epoch",
            "req_id",
            "edge_id",
            "edge_conf",
            "yaw_err",
            "dist_err",
            "edge_obs_age_ms",
            "edge_obs_is_stale",
            "handoff_state",
            "handoff_samples_count",
            "handoff_valid_samples_count",
            "slide_ref_ready",
            "slide_ref_yaw_err",
            "slide_ref_dist_err",
            "slide_ref_edge_conf",
            "slide_ref_roi",
            "slide_ref_seq",
            "full_locked_yaw_err",
            "full_locked_dist_err",
            "full_vs_light_yaw_offset",
            "full_vs_light_dist_offset",
            "yaw_delta_from_slide_ref",
            "dist_delta_from_slide_ref",
            "edge_identity_basis",
            "stable_ms",
            "lost_ms",
            "target_found",
            "target_conf",
            "target_cls",
            "matched_cls",
            "matched_conf",
            "best_cls",
            "best_conf",
            "target_center",
            "matched_center",
            "matched_center_full_norm",
            "matched_center_offset_norm",
            "bbox_valid",
            "bbox_invalid_reason",
            "target_window_found_ratio",
            "target_conf_median",
            "target_conf_max",
            "bbox_valid_ratio",
            "target_window_latest_matched_cls",
            "target_window_latest_matched_conf",
            "found_frames",
            "lost_frames",
            "confirm_elapsed_ms",
            "lock_elapsed_ms",
            "lost_hold_ms",
            "lock_decision_reason",
            "unlock_reason",
            "target_stable_ms",
            "center_jitter",
            "lost_reason",
            "transition_reason",
            "car_mode",
            "planned_cmd",
            "planned_cmd_mode",
            "condition",
            "reset_edge_tracking",
            "reset_target_tracking",
            "reset_slide_reference",
            "reset_reason",
            "reset_state",
            "cleared_fields",
        )
        out = {key: trace.get(key) for key in fixed_fields}
        out["event"] = out.get("event") or "state_transition"
        if out["event"] == "reset_state":
            reset_state = str(trace.get("reset_state") or "")
            out["reset_edge_tracking"] = reset_state == "edge"
            out["reset_target_tracking"] = reset_state == "target"
            out["reset_slide_reference"] = reset_state == "slide_ref"
        out["planned_cmd"] = out.get("planned_cmd") or {"vx": None, "vy": None, "wz": None}
        out["condition"] = out.get("condition") or {}
        return out

    def _emit_state_block_if_needed(self):
        block = self.core.export_state_block()
        now = time.time()
        key = self._state_block_dedup_key(block)
        if (now - self._last_state_block_ts) < float(self.cfg.runtime.state_block_period_s) and key == self._last_state_block_key:
            return
        self._last_state_block_ts = now
        self._last_state_block_key = key
        if self._state_blocks_full_log:
            self.run_logger.write_state_block(block)
        else:
            self.run_logger.write_jsonl("state_blocks_lite", self._state_block_lite(block))

    @staticmethod
    def _state_block_lite(block: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: block.get(key)
            for key in (
                "ts",
                "state",
                "control_source",
                "control_intent",
                "table_bbox_current_found",
                "table_bbox_control_valid",
                "yolo_table_visible",
                "yolo_table_fresh",
                "yolo_table_age_ms",
                "edge_geometry_valid",
                "edge_valid",
                "edge_trusted",
                "edge_age_ms",
                "selected_timeout_reason",
                "fallback_action",
                "no_table_bbox_timeout",
                "edge_geometry_timeout",
                "bbox_visible_but_edge_invalid",
                "table_lost_search_timeout",
                "allow_forward",
                "allow_rotate",
                "allow_lateral",
                "forward_block_reason",
                "rotate_block_reason",
                "lateral_block_reason",
                "stop_reason",
            )
            if block.get(key) is not None
        }

    def _state_block_dedup_key(self, block: Dict[str, Any]) -> str:
        key_fields = {
            "state",
            "prev_state",
            "resume_state",
            "task_intent",
            "active_target",
            "session_id",
            "epoch",
            "req_id",
            "vision_stage",
            "vision_mode",
            "current_edge_id",
            "edge_visit_index",
            "edge_transition_count",
            "table_cycle_count",
            "last_enter_reason",
            "last_fail_reason",
            "last_safety_reason",
            "vision_req_fail_streak",
            "has_table_edge_obs",
            "has_target_obs",
            "lock_ready",
            "lock_reason",
            "lock_reset_reason",
            "vision_stale_reason",
            "final_lock_yaw_ok",
            "final_lock_dist_ok",
            "final_lock_age_ok",
            "final_lock_confidence_ok",
            "stable_lock_count",
            "required_lock_count",
            "lock_count_inc_reason",
            "lock_count_hold_reason",
            "lock_count_reset_reason",
            "table_found",
            "edge_found",
            "edge_valid",
            "control_level",
            "reject_reason",
            "control_reject_reason",
            "usable_for_approach",
            "usable_for_alignment",
            "usable_for_stop",
            "pose_found",
            "table_confirmed_by_yolo",
            "yolo_gate_open",
            "table_approach_phase",
            "stale_level",
            "stale_guard_active",
            "stale_guard_reason",
            "edge_obs_is_stale",
            "edge_follow_stale",
            "handoff_state",
            "slide_ref_ready",
            "task_result",
            "edge_retries",
            "slide_entries",
            "target_confirm_count",
            "target_locked_count",
            "last_matched_cls",
            "lost_reason",
            "warnings",
        }
        return safe_dump({name: block.get(name) for name in sorted(key_fields)})

    def _age_or_none(self, ts_value: float) -> Optional[float]:
        if not ts_value:
            return None
        return max(0.0, time.time() - float(ts_value))

    def _emit_heartbeat_if_needed(self, force: bool = False):
        now = time.time()
        if not force and (now - self._last_heartbeat_ts) < float(self.cfg.runtime.heartbeat_period_s):
            return
        self._last_heartbeat_ts = now
        task_snap = self.task_server.snapshot()
        vision_snap = self.vision_server.snapshot()
        ack_snap = self._sender_snapshot(self.task_ack_sender)
        vision_req_snap = self._sender_snapshot(self.vision_req_sender)
        tts_snap = self._sender_snapshot(self.tts_sender)
        uart_snap = self.uart.snapshot()
        summary = {
            "ts": now,
            "uptime_s": max(0.0, now - self._boot_ts),
            "state": self.core.ctx.state.value,
            "serial_dry_run": self.cfg.serial.dry_run,
            "last_rx": {
                "task_cmd_age_s": self._age_or_none(self._last_task_cmd_recv_ts),
                "vision_obs_age_s": self._age_or_none(self._last_vision_obs_recv_ts),
                "home_obs_age_s": self._age_or_none(self._last_home_obs_recv_ts),
                "uart_rx_age_s": self._age_or_none(self.uart.last_rx_ts),
            },
            "last_tx": {
                "task_ack_age_s": self._age_or_none(self._last_tx_summary.get("task_ack_out", 0.0)),
                "vision_req_age_s": self._age_or_none(self._last_tx_summary.get("vision_req_out", 0.0)),
                "tts_event_age_s": self._age_or_none(self._last_tx_summary.get("tts_event_out", 0.0)),
                "uart_tx_age_s": self._age_or_none(self._last_uart_tx_ts),
            },
            "servers": {
                "task_cmd_in": task_snap,
                "vision_obs_in": vision_snap,
            },
            "senders": {
                "task_ack_out": ack_snap,
                "vision_req_out": vision_req_snap,
                "tts_event_out": tts_snap,
            },
            "uart": uart_snap,
            "motion_status": dict(self.motion_status),
            "ready": {
                "task_cmd_in_listening": bool(task_snap.get("listening")),
                "vision_obs_in_listening": bool(vision_snap.get("listening")),
                "uart_ready": bool(self.cfg.serial.dry_run or uart_snap.get("serial_open")),
                "vision_req_async": bool(vision_req_snap.get("async", False)),
                "tts_async": bool(tts_snap.get("async", False)),
            },
        }
        self.run_logger.write_jsonl("heartbeat", summary)
        self.run_logger.write_event(
            f"HB state={summary['state']} dry_run={summary['serial_dry_run']} "
            f"task_rx_age={summary['last_rx']['task_cmd_age_s']} vision_rx_age={summary['last_rx']['vision_obs_age_s']} "
            f"vision_req_link={vision_req_snap.get('link_state')} tts_link={tts_snap.get('link_state')}"
        )
        if self.operator_console.full or self._heartbeat_console_enabled:
            self.operator_console.emit_rate_limited(
                "heartbeat",
                f"[ORCH] HEARTBEAT state={summary['state']} vision_req={vision_req_snap.get('link_state')} task_cmd={int(bool(task_snap.get('listening')))}",
                1.0,
            )

    def _emit_system_metrics_if_needed(self, force: bool = False) -> None:
        sample = self._system_metrics.sample_if_due(force=force)
        if sample is not None:
            self.run_logger.write_jsonl("system_metrics", sample)


def run_orchestrator_service(cfg: OrchestratorConfig):
    service = OrchestratorService(cfg)

    def _handle_sig(signum, frame):
        service.log_info("runtime", f"signal {signum} received; shutting down")
        service._running = False

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)
    service.run_forever()
