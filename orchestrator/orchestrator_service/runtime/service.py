#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import queue
import signal
import time
import os
from typing import Any, Dict, List, Optional

from ..bridge.simple_car_protocol import SimpleCarMapper, parse_car_state_line
from ..bridge.uart_bridge import UartBridge
from ..config.schema import OrchestratorConfig, SocketEndpoint
from ..ipc.protocol import (
    HomeTagObs,
    TableEdgeObs,
    TargetObs,
    TaskCmd,
    VisionObsEnvelope,
    iter_vision_perception_payloads,
    make_task_ack,
)
from ..ipc.transport import AsyncJsonlClientSender, JsonlClientSender, JsonlInboundServer
from .common import RunLogger, ensure_dir, safe_dump
from .state_machine import OrchestratorCore
from common.base_module import BaseModule
from common.runtime_logging import OperatorConsole


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
        self._ipc_console_enabled = self._env_bool("ORCH_IPC_CONSOLE", False)
        self._heartbeat_console_enabled = self._env_bool("ORCH_HEARTBEAT_CONSOLE", False)
        self._uart_console_mode = self._env_choice("ORCH_UART_CONSOLE", "operator", {"operator", "full", "silent"})
        self._mobile_status_console_mode = self._env_choice("ORCH_MOBILE_STATUS_CONSOLE", "change", {"change", "full", "silent"})
        self._last_obs_flags = {"table_edge": False, "target": False}
        self._last_target_obs_console_payload: Dict[str, Any] = {}
        self._last_target_search_req_console_key = ""
        self._pending_state_traces: List[Dict[str, Any]] = []
        self._last_edge_slide_trace_ts = 0.0
        self.mapper = SimpleCarMapper(cfg.car)
        self._async_result_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._boot_ts = time.time()
        self._last_state_block_ts = 0.0
        self._last_state_block_key = ""
        self._last_heartbeat_ts = 0.0
        self._last_task_cmd_recv_ts = 0.0
        self._last_vision_obs_recv_ts = 0.0
        self._last_home_obs_recv_ts = 0.0
        self._last_uart_tx_ts = 0.0
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
        self.task_server = self._build_server(cfg.task_cmd_in, "task_cmd_in")
        self.vision_server = self._build_server(cfg.vision_obs_in, "vision_obs_in")
        self.task_ack_sender = self._build_sender(cfg.task_ack_out, "task_ack_out", async_allowed=False)
        self.vision_req_sender = self._build_sender(cfg.vision_req_out, "vision_req_out", async_allowed=True)
        self.tts_sender = self._build_sender(cfg.tts_event_out, "tts_event_out", async_allowed=True)
        self._running = False

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
            return (
                f"[ORCH] OBS target={target} found=1 "
                f"matched_cls={matched_cls} matched_conf={self._fmt_float(matched_conf, 2, signed=False)} "
                f"best_cls={best_cls} best_conf={self._fmt_float(best_conf, 2, signed=False)} "
                f"cx={self._fmt_float(payload.get('cx_norm', payload.get('cx')), 2, signed=False)} "
                f"cy={self._fmt_float(payload.get('cy_norm', payload.get('cy')), 2, signed=False)}"
            )
        boxes = payload.get("boxes_count", payload.get("box_count", payload.get("boxes", 0)))
        if isinstance(boxes, list):
            boxes = len(boxes)
        mode = str(payload.get("vision_mode") or payload.get("mode") or self.core.ctx.active_vision_mode or "").strip() or "n/a"
        return f"[ORCH] OBS target={target} found=0 boxes={int(boxes or 0)} mode={mode}"

    def _vision_req_console_summary(self, payload: Dict[str, Any]) -> Optional[str]:
        if str(payload.get("type") or "").strip() != "vision_req":
            return None
        req_payload = dict(payload.get("payload") or {})
        kind = str(req_payload.get("search_kind") or payload.get("kind") or "").strip().upper()
        stage = str(payload.get("stage") or "").strip().upper()
        mode_hint = str(payload.get("mode_hint") or "").strip().upper()
        if stage != "SEARCH" or kind != "TARGET":
            return None
        target = str(payload.get("target") or self.core.ctx.active_target or "target").strip() or "target"
        req_id = str(payload.get("req_id") or "").strip()
        return f"[ORCH] REQ target_search stage={stage} kind={kind} target={target} mode_hint={mode_hint or 'n/a'} req={req_id}"

    def _emit_target_obs_console(self, payload: Dict[str, Any]) -> None:
        state = str(getattr(self.core.ctx.state, "value", self.core.ctx.state) or "")
        if state not in {"SEARCH_TARGET_INIT", "EDGE_SLIDE_SEARCH"}:
            return
        self._last_target_obs_console_payload = dict(payload or {})
        line = self._target_obs_console_line(payload)
        self.operator_console.emit_rate_limited("target_obs", line, self.operator_console.default_interval_s)

    def _operator_emit(self, line: str) -> bool:
        return self.operator_console.emit(line)

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
            return [f"path={ep.uds_path}"]
        return [f"transport={ep.transport}"]

    def _operator_ipc_line(self, channel: str, event: str, details: Dict[str, Any]) -> str:
        level = "ERROR" if event in {"send_failed", "invalid_json"} else ("WARN" if "failed" in event or "closed" in event else "IPC")
        parts = [f"[ORCH] {level} {channel} {event}"]
        parts.extend(self._endpoint_parts(channel))
        if details.get("peer"):
            parts.append(f"peer={details.get('peer')}")
        if details.get("error"):
            parts.append(f"err={self._short_err(details.get('error'))}")
        if details.get("fail_count") is not None:
            parts.append(f"retry={details.get('fail_count')}")
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

    def _log_json(self, payload):
        level = payload.get("level", "info")
        event = payload.get("event", "")
        name = payload.get("name", payload.get("src", "module"))
        extra = {k: v for k, v in payload.items() if k not in {"level", "msg"}}
        message = event or payload.get("msg", "")
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
            },
            "control": {
                "search_table_timeout_s": self.cfg.control.search_table_timeout_s,
                "approach_timeout_s": self.cfg.control.approach_timeout_s,
                "target_search_timeout_s": self.cfg.control.target_search_timeout_s,
                "final_lock_yaw_tol_rad": self.cfg.control.final_lock_yaw_tol_rad,
                "final_lock_dist_tol_m": self.cfg.control.final_lock_dist_tol_m,
                "final_lock_frames_to_arrive": self.cfg.control.final_lock_frames_to_arrive,
                "edge_slide_dist_tolerance_m": self.cfg.control.edge_slide_dist_tolerance_m,
                "target_confirm_conf_th": self.cfg.control.target_confirm_conf_th,
                "target_found_frames_to_confirm": self.cfg.control.target_found_frames_to_confirm,
                "target_lock_conf_th": self.cfg.control.target_lock_conf_th,
                "target_lock_settle_s": self.cfg.control.target_lock_settle_s,
                "edge_relocate_enabled": self.cfg.control.edge_relocate_enabled,
                "max_edge_transitions_per_task": self.cfg.control.max_edge_transitions_per_task,
            },
            "car_cmd": {
                "send_period_ms": self.cfg.car.send_period_ms,
                "hold_ms": self.cfg.car.cmd_hold_ms,
                "max_vx_norm": self.cfg.car.max_vx_norm,
                "max_vy_norm": self.cfg.car.max_vy_norm,
                "max_wz_norm": self.cfg.car.max_wz_norm,
                "stop_on_state_enter": self.cfg.car.stop_on_state_enter,
            },
            "docking": {
                "min_confidence": self.cfg.docking.min_confidence,
                "enable_lateral_control": self.cfg.docking.enable_lateral_control,
                "approach_max_vx_norm": self.cfg.docking.approach_max_vx_norm,
                "approach_max_vy_norm": self.cfg.docking.approach_max_vy_norm,
                "approach_max_wz_norm": self.cfg.docking.approach_max_wz_norm,
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
            return (
                f"[ORCH] CAR_VEL "
                f"vx={self._fmt_float(payload.get('actual_vx_norm', payload.get('vx_norm', 0.0)))} "
                f"vy={self._fmt_float(payload.get('actual_vy_norm', payload.get('vy_norm', 0.0)))} "
                f"wz={self._fmt_float(payload.get('actual_wz_norm', payload.get('wz_norm', 0.0)))} "
                f"hold={int(payload.get('actual_hold_ms', payload.get('hold_ms', 0)) or 0)}ms"
            )
        if uart_kind == "stop":
            return f"[ORCH] CAR_STOP mode={payload.get('mode') or 'STOP'}"
        if uart_kind == "brake":
            return f"[ORCH] CAR_BRAKE mode={payload.get('mode') or 'BRAKE'}"
        mode = str(payload.get("mode") or payload.get("state") or "").strip() or "UNKNOWN"
        kind = str(payload.get("kind") or "").strip()
        if kind == "stop" or str(payload.get("raw", "")).strip().upper().endswith("STOP"):
            return f"[ORCH] CAR_STOP mode={mode}"
        return (
            f"[ORCH] CAR_VEL "
            f"vx={self._fmt_float(payload.get('vx_norm', 0.0))} "
            f"vy={self._fmt_float(payload.get('vy_norm', 0.0))} "
            f"wz={self._fmt_float(payload.get('wz_norm', 0.0))} "
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
        for key in ("vx_norm", "vy_norm", "wz_norm", "actual_vx_norm", "actual_vy_norm", "actual_wz_norm"):
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
            elif upper == "VEL" and len(parts) >= 5:
                item["uart_kind"] = "vel"
                try:
                    item["actual_vx_norm"] = float(parts[1])
                    item["actual_vy_norm"] = float(parts[2])
                    item["actual_wz_norm"] = float(parts[3])
                    item["actual_hold_ms"] = int(float(parts[4]))
                except Exception:
                    pass
            elif upper == "STOP":
                item["uart_kind"] = "stop"
            elif upper == "BRAKE":
                item["uart_kind"] = "brake"
            else:
                item["uart_kind"] = "raw"
            item["summary_key"] = self._uart_event_key(item)
            item["rendered"] = self._render_uart_line(item)
            out.append(item)
        return out

    def _emit_no_vel_if_needed(self, payload: Dict[str, Any], actual_payloads: List[Dict[str, Any]]) -> None:
        expected_vx = float(payload.get("vx_norm", 0.0) or 0.0)
        expected_vy = float(payload.get("vy_norm", 0.0) or 0.0)
        expected_wz = float(payload.get("wz_norm", 0.0) or 0.0)
        expects_vel = any(abs(v) > 1e-9 for v in (expected_vx, expected_vy, expected_wz)) or str(payload.get("kind")) == "cmd_vel"
        has_vel = any(str(item.get("uart_kind")) == "vel" for item in actual_payloads)
        if expects_vel and not has_vel:
            state = str(payload.get("state") or self.core.ctx.state.value)
            reason = str(payload.get("reason") or "no_vel_line").strip() or "no_vel_line"
            self.operator_console.emit_error(
                f"no_vel_sent:{state}:{reason}",
                f"[ORCH] WARN no_vel_sent state={state} reason={reason}",
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
        self.run_logger.write_jsonl("uart_tx", payload)
        self._update_uart_lowfreq(payload)
        actual_payloads = self._actual_uart_payloads(payload)
        self._emit_no_vel_if_needed(payload, actual_payloads)
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
            f"final_lock yaw={self.cfg.control.final_lock_yaw_tol_rad:.3f} "
            f"dist={self.cfg.control.final_lock_dist_tol_m:.3f} "
            f"stable_frames={int(self.cfg.control.final_lock_frames_to_arrive)} "
            f"edge_conf={self.cfg.docking.min_confidence:.2f} "
            f"slide_vy={self.cfg.car.edge_slide_vy_norm:.2f}"
        )
        self.uart.start()
        self.task_server.start()
        self.vision_server.start()
        self._running = True
        self.run_logger.write_service_event("SERVICE_READY", run_dir=str(self.run_logger.run_dir))
        uart_mode = "fake" if self.cfg.serial.dry_run else str(self.cfg.serial.port)
        self._operator_emit(f"[ORCH] READY state={self.core.ctx.state.value} dry_run={int(bool(self.cfg.serial.dry_run))} uart={uart_mode}")
        self.log_info("runtime", "SERVICE_READY", {"run_dir": str(self.run_logger.run_dir)})
        self._emit_heartbeat_if_needed(force=True)

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
                decision = self.core.tick()
                self._flush_pending_msgs()
                self._emit_motion(decision)
                self._emit_state_block_if_needed()
                self._emit_heartbeat_if_needed()
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
            if state is None:
                continue
            self.core.handle_car_state(state)
            self.run_logger.write_jsonl("car_state", state.to_dict())
            self.run_logger.write_timeline("CAR_STATE", state=state.state, message=state.message, estop=state.estop, timeout=state.timeout, fault=state.fault)

    def _send_task_ack(self, cmd: TaskCmd, accepted: bool, reason: str):
        ack = make_task_ack(cmd, accepted=accepted, reason=reason, state=self.core.ctx.state.value)
        sent = self.task_ack_sender.send(ack)
        self._last_tx_summary["task_ack_out"] = time.time()
        self.run_logger.write_jsonl("task_ack", ack)
        self.run_logger.write_ipc(
            "task_ack_out",
            "ack_sent" if sent else "ack_send_failed",
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
            "ack_sent" if sent else "ack_send_failed",
            {"cmd_id": cmd.cmd_id, "accepted": accepted, "error": "" if sent else reason},
        )
        self.log_ipc("TX", "task_ack", "sent" if sent else "failed", {"cmd_id": cmd.cmd_id, "accepted": accepted})

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
            accepted, reason = self.core.handle_task_cmd(cmd)
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
            self.run_logger.write_jsonl("vision_obs", env.to_dict())
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
                has_table_edge_obs=has_table_edge,
                has_target_obs=has_target,
            )
        return iter_vision_perception_payloads(payload)

    def _drain_vision_msgs(self):
        latest_table: Optional[TableEdgeObs] = None
        latest_target: Optional[TargetObs] = None
        latest_home: Optional[HomeTagObs] = None
        latest_table_priority = -1
        latest_target_priority = -1
        latest_home_priority = -1
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
                            "frame_id": parsed.frame_id,
                            "seq": parsed.seq,
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
                            frame_id=parsed.frame_id,
                            seq=parsed.seq,
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
                key = str(msg.get("req_id") or summary_line)
                if key != self._last_target_search_req_console_key:
                    self._last_target_search_req_console_key = key
                    self.operator_console.emit(summary_line)
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
        summary.setdefault("cmd", {"vx": cmd.vx_norm, "vy": cmd.vy_norm, "wz": cmd.wz_norm, "hold_ms": cmd.hold_ms})
        block = self.core.export_state_block()
        for key in (
            "edge_found",
            "confidence",
            "yaw_err_rad",
            "dist_err_m",
            "target_dist_m",
            "table_edge_obs_age_ms",
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
            summary["edge_obs_source_mode"] = block.get("table_edge_obs_source_mode") or getattr(table_obs, "source_mode", None)
            summary["fallback_reason"] = summary.get("reason") or self.core.ctx.last_fail_reason or ""
            summary["edge_loss_elapsed_s"] = self.core._loss_elapsed(self.core.ctx.table_loss_since_mono)
        if block.get("lock_reason") and str(summary.get("state")) in {"CONTROLLED_APPROACH", "FINAL_LOCK", "COARSE_ALIGN"}:
            summary["reason"] = block.get("lock_reason")
        else:
            summary.setdefault("reason", summary.get("lock_reason") or self.core.ctx.last_enter_reason or "")
        return summary

    def _operator_control_line(self, summary: Dict[str, Any]) -> str:
        cmd = dict(summary.get("cmd") or {})
        state = str(summary.get("state") or self.core.ctx.state.value)
        reason = str(summary.get("reason") or summary.get("lock_reason") or "").strip() or "n/a"
        base = (
            f"state={state} edge={int(bool(summary.get('edge_found')))} "
            f"conf={self._fmt_float(summary.get('confidence'), 2, signed=False)} "
            f"yaw={self._fmt_float(summary.get('yaw_err_rad'))} "
            f"dist={self._fmt_float(summary.get('dist_err_m'))} "
            f"lock={int(bool(summary.get('lock_ready')))} reason={reason} "
            f"cmd vx={self._fmt_float(cmd.get('vx'))} vy={self._fmt_float(cmd.get('vy'))} wz={self._fmt_float(cmd.get('wz'))}"
        )
        if state == "FINAL_LOCK":
            stable = int(self.core.ctx.table_lock_frames)
            needed = int(self.cfg.control.final_lock_frames_to_arrive)
            return (
                f"[ORCH] LOCK edge={self.core.ctx.current_edge_id} "
                f"conf={self._fmt_float(summary.get('confidence'), 2, signed=False)} "
                f"yaw={self._fmt_float(summary.get('yaw_err_rad'))} "
                f"dist={self._fmt_float(summary.get('dist_err_m'))} stable={stable}/{needed} "
                f"ready={int(bool(summary.get('lock_ready')))} reason={reason} "
                f"cmd vx={self._fmt_float(cmd.get('vx'))} vy={self._fmt_float(cmd.get('vy'))} wz={self._fmt_float(cmd.get('wz'))}"
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
            search_note = "target_locked" if found else "searching"
            edge_visible = bool(summary.get("edge_found", False))
            return (
                f"[ORCH] SLIDE edge={int(edge_visible)} target={int(found)} boxes={int(boxes or 0)} status={search_note} "
                f"conf={self._fmt_float(summary.get('confidence'), 2, signed=False)} "
                f"yaw={self._fmt_float(summary.get('yaw_err_rad'))} "
                f"dist={self._fmt_float(summary.get('dist_err_m'))} "
                f"age={self._fmt_float(summary.get('edge_obs_age_ms'), 0, signed=False)}ms "
                f"stale={int(bool(summary.get('edge_obs_is_stale')))} "
                f"matched_cls={matched_cls or 'n/a'} matched_conf={self._fmt_float(matched_conf, 2, signed=False)} "
                f"best_cls={best_cls or 'n/a'} best_conf={self._fmt_float(best_conf, 2, signed=False)} "
                f"cmd vx={self._fmt_float(cmd.get('vx'))} vy={self._fmt_float(cmd.get('vy'))} wz={self._fmt_float(cmd.get('wz'))} "
                f"reason={reason}"
            )
        return f"[ORCH] CTRL {base}"

    def _edge_slide_zero_reason(self, summary: Dict[str, Any], reason: str) -> str:
        raw = str(reason or summary.get("reason") or "").strip()
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
        if str(self.core.ctx.active_vision_mode or "").upper() == "TRACK_LOCAL" and self.core.ctx.last_table_obs is None:
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
        mode = str(getattr(self.core.ctx, "active_vision_mode", "") or "n/a").strip() or "n/a"
        self.operator_console.emit_rate_limited(
            "target_obs_missing",
            f"[ORCH] WARN target_obs_missing mode={mode} age={age:.1f}s",
            self.operator_console.default_interval_s,
        )

    def _emit_operator_control(self, decision) -> None:
        summary = self._control_summary_with_context(decision)
        state = str(summary.get("state") or self.core.ctx.state.value)
        if state in {"IDLE", "DONE", "ERROR_RECOVERY"} and not self.operator_console.full:
            return
        line = self._operator_control_line(summary)
        key = "lock" if state == "FINAL_LOCK" else ("slide" if state == "EDGE_SLIDE_SEARCH" else "ctrl")
        self.operator_console.emit_rate_limited(key, line, self.operator_console.default_interval_s)
        self._emit_target_obs_missing_warning()

    def _emit_motion(self, decision):
        cmd = decision.cmd
        self.run_logger.write_jsonl("cmd_vel", cmd.to_dict())
        self._emit_edge_slide_trace(decision)
        self._emit_operator_control(decision)
        self._flush_state_traces(decision)
        car_cmd = self.mapper.from_cmd_vel(cmd, cx_norm_abs=decision.cx_norm_abs, distance_ratio=decision.distance_ratio)
        tx_meta = self._build_uart_tx_meta(car_cmd)
        car_record = {
            "ts": time.time(),
            "mode": car_cmd.mode,
            "kind": car_cmd.kind,
            "vx_norm": car_cmd.vx_norm,
            "vy_norm": car_cmd.vy_norm,
            "wz_norm": car_cmd.wz_norm,
            "hold_ms": car_cmd.hold_ms,
            "brake": car_cmd.brake,
            "raw": car_cmd.raw_line.rstrip("\n"),
        }
        car_record.update({k: v for k, v in tx_meta.items() if v not in (None, "")})
        self.run_logger.write_jsonl("car_cmd", car_record)
        self.uart.send_car_command(car_cmd, tx_meta=tx_meta)

    def _emit_edge_slide_trace(self, decision) -> None:
        if str(getattr(self.core.ctx.state, "value", self.core.ctx.state) or "") != "EDGE_SLIDE_SEARCH":
            return
        now = time.time()
        period_ms = max(0, int(getattr(self.cfg.control, "edge_follow_log_period_ms", 500) or 500))
        if period_ms > 0 and (now - float(getattr(self, "_last_edge_slide_trace_ts", 0.0) or 0.0)) < (period_ms / 1000.0):
            return
        self._last_edge_slide_trace_ts = now
        edge_obs = self.core.ctx.last_table_obs
        target_obs = self.core.ctx.last_target_obs
        cmd = decision.cmd
        reason = str((getattr(decision, "control_summary", None) or {}).get("reason") or self.core.ctx.last_fail_reason or "")
        edge_age_ms = self.core._table_obs_age_ms(edge_obs)
        edge_is_stale = self.core._edge_obs_is_stale(edge_obs)
        matched_cls = (
            getattr(target_obs, "matched_cls", None) or getattr(target_obs, "target", None)
            if target_obs is not None
            else None
        )
        record = {
            "ts": now,
            "state": "EDGE_SLIDE_SEARCH",
            "target": self.core.ctx.active_target,
            "edge_found": bool(getattr(edge_obs, "edge_found", False)) if edge_obs is not None else False,
            "edge_conf": getattr(edge_obs, "confidence", None) if edge_obs is not None else None,
            "dist_err": getattr(edge_obs, "dist_err_m", None) if edge_obs is not None else None,
            "yaw_err": getattr(edge_obs, "yaw_err_rad", None) if edge_obs is not None else None,
            "edge_obs_ts": getattr(edge_obs, "obs_ts", None) if edge_obs is not None else None,
            "edge_obs_age_ms": edge_age_ms,
            "edge_obs_is_stale": bool(edge_is_stale),
            "edge_follow_stale": bool(edge_is_stale),
            "edge_obs_frame_id": getattr(edge_obs, "frame_id", None) if edge_obs is not None else None,
            "edge_obs_seq": getattr(edge_obs, "seq", None) if edge_obs is not None else None,
            "edge_obs_source_mode": getattr(edge_obs, "source_mode", None) if edge_obs is not None else None,
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
            "vx": float(cmd.vx_norm),
            "vy": float(cmd.vy_norm),
            "wz": float(cmd.wz_norm),
            "fallback_reason": reason,
            "edge_loss_elapsed_s": self.core._loss_elapsed(self.core.ctx.table_loss_since_mono),
            "keep_dist_tolerance_m": float(self.cfg.control.edge_slide_dist_tolerance_m),
            "edge_lost_hold_s": float(self.cfg.control.table_loss_hold_s),
            "fallback_state": str(getattr(self.cfg.control, "edge_slide_fallback_state", "CONTROLLED_APPROACH") or "CONTROLLED_APPROACH"),
        }
        self.run_logger.write_jsonl("edge_slide_search", record)

    def _flush_state_traces(self, decision) -> None:
        if not self._pending_state_traces:
            return
        cmd = decision.cmd
        planned_cmd = {
            "vx": float(cmd.vx_norm),
            "vy": float(cmd.vy_norm),
            "wz": float(cmd.wz_norm),
        }
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
            "found_frames",
            "lost_frames",
            "confirm_elapsed_ms",
            "lock_elapsed_ms",
            "target_stable_ms",
            "center_jitter",
            "lost_reason",
            "transition_reason",
            "car_mode",
            "planned_cmd",
            "planned_cmd_mode",
            "condition",
        )
        out = {key: trace.get(key) for key in fixed_fields}
        out["event"] = out.get("event") or "state_transition"
        out["planned_cmd"] = out.get("planned_cmd") or {"vx": None, "vy": None, "wz": None}
        out["condition"] = out.get("condition") or {}
        return out

    def _emit_state_block_if_needed(self):
        block = self.core.export_state_block()
        now = time.time()
        key = safe_dump(block)
        if (now - self._last_state_block_ts) < float(self.cfg.runtime.state_block_period_s) and key == self._last_state_block_key:
            return
        self._last_state_block_ts = now
        self._last_state_block_key = key
        self.run_logger.write_state_block(block)

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


def run_orchestrator_service(cfg: OrchestratorConfig):
    service = OrchestratorService(cfg)

    def _handle_sig(signum, frame):
        service.log_info("runtime", f"signal {signum} received; shutting down")
        service._running = False

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)
    service.run_forever()
