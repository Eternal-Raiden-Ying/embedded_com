#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import queue
import signal
import time
from typing import Any, Dict, List, Optional

from ..bridge.simple_car_protocol import SimpleCarMapper, parse_car_state_line
from ..bridge.uart_bridge import UartBridge
from ..config.schema import OrchestratorConfig, SocketEndpoint
from ..ipc.protocol import HomeTagObs, TargetObs, TaskCmd, make_task_ack
from ..ipc.transport import AsyncJsonlClientSender, JsonlClientSender, JsonlInboundServer
from .common import RunLogger, configure_logging, ensure_dir, safe_dump
from .state_machine import OrchestratorCore


class OrchestratorService:
    def __init__(self, cfg: OrchestratorConfig):
        self.cfg = cfg
        configure_logging("full" if (cfg.runtime.debug or cfg.runtime.log_mode == "full") else "concise")
        self.log = logging.getLogger("OrchestratorService")
        ensure_dir(cfg.runtime.runs_dir)
        self.run_logger = RunLogger(cfg.runtime.runs_dir)
        self.core = OrchestratorCore(cfg.control, cfg.car)
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
        self.uart = UartBridge(
            cfg.serial.port,
            cfg.serial.baudrate,
            cfg.serial.timeout_s,
            dry_run=cfg.serial.dry_run,
            readback_enabled=cfg.serial.readback_enabled,
            dry_run_echo_stdout=cfg.serial.dry_run_echo_stdout,
            tx_callback=self._on_uart_tx,
        )
        self.task_server = self._build_server(cfg.task_cmd_in, "task_cmd_in")
        self.vision_server = self._build_server(cfg.vision_obs_in, "vision_obs_in")
        self.task_ack_sender = self._build_sender(cfg.task_ack_out, "task_ack_out", async_allowed=False)
        self.vision_req_sender = self._build_sender(cfg.vision_req_out, "vision_req_out", async_allowed=True)
        self.tts_sender = self._build_sender(cfg.tts_event_out, "tts_event_out", async_allowed=True)
        self._running = False

    def _log_json(self, payload):
        level = payload.get("level", "info")
        event = payload.get("event", "")
        name = payload.get("name", payload.get("src", "module"))
        extra = {k: v for k, v in payload.items() if k not in {"level", "msg"}}
        getattr(self.log, level, self.log.info)("%s | %s | extra=%s", name, event or payload.get("msg", ""), extra)
        if payload.get("src") == "ipc":
            self.run_logger.write_ipc(name, event or "log", **{k: v for k, v in payload.items() if k not in {"level", "src", "name", "event"}})

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
            "tick_hz": self.cfg.runtime.tick_hz,
            "heartbeat_period_s": self.cfg.runtime.heartbeat_period_s,
            "runs_dir": self.cfg.runtime.runs_dir,
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
        rendered = self._render_uart_line(payload)
        stop_class = str(payload.get("stop_class", "") or "")
        stop_reason = str(payload.get("stop_reason", "") or "")
        extras: List[str] = []
        if stop_class:
            extras.append(f"stop_class={stop_class}")
        if stop_reason and stop_class not in {"idle_stop", "boot_init_stop"}:
            extras.append(f"reason={stop_reason}")
        if emit_reason == "periodic":
            prefix = f"[fake-car summary] repeat={int(max(1, repeat_count))}"
        else:
            prefix = "[fake-car]"
        body = f"{prefix} {rendered}".strip()
        if extras:
            body += " | " + " | ".join(extras)
        return body

    def _emit_uart_console(self, payload: Dict[str, Any], emit_reason: str, repeat_count: int):
        if self._should_quiet_console(payload):
            return
        self.log.info(self._console_message(payload, emit_reason, repeat_count))

    def _update_uart_console(self, payload: Dict[str, Any]):
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
        if key != self._uart_console_key:
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
        self._update_uart_console(payload)

    def start(self):
        cfg_dump = self._config_dump()
        self.run_logger.write_event(f"run_dir={self.run_logger.run_dir}")
        self.run_logger.write_jsonl("config", cfg_dump)
        self.run_logger.write_timeline("BOOT", run_dir=str(self.run_logger.run_dir), config=cfg_dump)
        self.uart.start()
        self.task_server.start()
        self.vision_server.start()
        self._running = True
        self.run_logger.write_event("orchestrator started")
        self.log.info("orchestrator 已启动，日志目录: %s", self.run_logger.run_dir)
        self._emit_heartbeat_if_needed(force=True)

    def stop(self):
        self._running = False
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
        self.run_logger.write_timeline("STOP")
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
                self.run_logger.write_ipc(channel, "send_ok" if ok else "send_failed", req_id=payload.get("req_id"), mode=payload.get("mode"), error=error)
                self.run_logger.write_timeline(
                    "VISION_REQ_SEND",
                    req_id=payload.get("req_id"),
                    mode=payload.get("mode"),
                    sent=ok,
                    session_id=payload.get("session_id"),
                    epoch=payload.get("epoch"),
                )
            elif channel == "tts_event_out":
                self.run_logger.write_ipc(channel, "send_ok" if ok else "send_failed", text=payload.get("text"), interrupt=payload.get("interrupt", False), link_state=snap.get("link_state"))
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
        self.run_logger.write_ipc("task_ack_out", "ack_sent" if sent else "ack_send_failed", cmd_id=cmd.cmd_id, accepted=accepted, reason=reason)

    def _drain_task_cmds(self):
        for item in self.task_server.drain():
            payload = item["payload"]
            self._last_task_cmd_recv_ts = float(item.get("recv_ts", time.time()))
            try:
                cmd = TaskCmd.from_dict(payload, set(self.cfg.frozen_targets.keys()))
            except Exception as exc:
                self.log.warning("收到非法 task_cmd: %s (%s)", payload, exc)
                self.run_logger.write_event(f"bad task_cmd: {payload} ({exc})")
                self.run_logger.write_timeline("TASK_CMD_BAD", payload=safe_dump(payload), error=str(exc))
                continue
            accepted, reason = self.core.handle_task_cmd(cmd)
            self.run_logger.write_jsonl("task_cmd", cmd.to_dict())
            self.run_logger.write_timeline("TASK_CMD_RECV", cmd_id=cmd.cmd_id, intent=cmd.intent, target=cmd.target, accepted=accepted, reason=reason, session_id=cmd.session_id, epoch=cmd.epoch)
            self._send_task_ack(cmd, accepted=accepted, reason=reason)

    def _drain_vision_msgs(self):
        latest_target: Optional[TargetObs] = None
        latest_home: Optional[HomeTagObs] = None
        raw_items: List[dict] = self.vision_server.drain()
        for item in raw_items:
            payload = item["payload"]
            recv_ts = float(item.get("recv_ts", time.time()))
            msg_type = str(payload.get("type", "")).strip().lower()
            try:
                if msg_type == "home_tag_obs":
                    latest_home = HomeTagObs.from_dict(payload)
                    self._last_home_obs_recv_ts = recv_ts
                else:
                    latest_target = TargetObs.from_dict(payload)
                    self._last_vision_obs_recv_ts = recv_ts
            except Exception as exc:
                self.log.warning("收到非法视觉消息: %s (%s)", payload, exc)
                self.run_logger.write_event(f"bad vision msg: {payload} ({exc})")
                self.run_logger.write_timeline("VISION_BAD", payload=safe_dump(payload), error=str(exc))
        if latest_target is not None:
            self.core.handle_target_obs(latest_target)
            self.run_logger.write_jsonl("target_obs", latest_target.to_dict())
        if latest_home is not None:
            self.core.handle_home_obs(latest_home)
            self.run_logger.write_jsonl("home_tag_obs", latest_home.to_dict())

    def _enqueue_async_or_fail(self, sender: Any, channel: str, msg: Dict[str, Any]) -> bool:
        ok = sender.send(msg)
        if ok:
            return True
        error = f"enqueue_failed link_state={self._sender_snapshot(sender).get('link_state')}"
        self.run_logger.write_ipc(channel, "enqueue_failed", req_id=msg.get("req_id"), mode=msg.get("mode"), error=error)
        self.run_logger.write_timeline(f"{channel.upper()}_ENQUEUE_FAIL", req_id=msg.get("req_id"), mode=msg.get("mode"), error=error)
        return False

    def _flush_pending_msgs(self):
        for msg in self.core.drain_vision_msgs():
            self.run_logger.write_jsonl(msg.get("type", "vision_req"), msg)
            if isinstance(self.vision_req_sender, AsyncJsonlClientSender):
                queued = self._enqueue_async_or_fail(self.vision_req_sender, "vision_req_out", msg)
                if not queued:
                    self.core.handle_vision_req_send_result(False, msg, error="async enqueue failed")
                continue
            sent = self.vision_req_sender.send(msg)
            self._last_tx_summary["vision_req_out"] = time.time()
            error = "" if sent else f"link_state={self.vision_req_sender.snapshot().get('link_state')}"
            self.core.handle_vision_req_send_result(sent, msg, error=error)
            self.run_logger.write_ipc("vision_req_out", "send_ok" if sent else "send_failed", req_id=msg.get("req_id"), mode=msg.get("mode"), error=error)
            self.run_logger.write_timeline("VISION_REQ_SEND", req_id=msg.get("req_id"), mode=msg.get("mode"), sent=sent, session_id=msg.get("session_id"), epoch=msg.get("epoch"))
        for msg in self.core.drain_tts_msgs():
            self.run_logger.write_jsonl("tts_event", msg)
            if self.cfg.tts_event_out.transport == "disabled":
                self.run_logger.write_ipc("tts_event_out", "disabled", text=msg.get("text"), interrupt=msg.get("interrupt", False))
                continue
            if isinstance(self.tts_sender, AsyncJsonlClientSender):
                self._enqueue_async_or_fail(self.tts_sender, "tts_event_out", msg)
                continue
            sent = self.tts_sender.send(msg)
            self._last_tx_summary["tts_event_out"] = time.time()
            self.run_logger.write_ipc("tts_event_out", "send_ok" if sent else "send_failed", text=msg.get("text"), interrupt=msg.get("interrupt", False))

    def _emit_motion(self, decision):
        cmd = decision.cmd
        self.run_logger.write_jsonl("cmd_vel", cmd.to_dict())
        car_cmd = self.mapper.from_cmd_vel(cmd, cx_norm_abs=decision.cx_norm_abs, distance_ratio=decision.distance_ratio)
        tx_meta = self._build_uart_tx_meta(car_cmd)
        car_record = {
            "ts": time.time(),
            "mode": car_cmd.mode,
            "kind": car_cmd.kind,
            "vx_norm": car_cmd.vx_norm,
            "wz_norm": car_cmd.wz_norm,
            "raw": car_cmd.raw_line.rstrip("\n"),
        }
        car_record.update({k: v for k, v in tx_meta.items() if v not in (None, "")})
        self.run_logger.write_jsonl("car_cmd", car_record)
        self.uart.send_car_command(car_cmd, tx_meta=tx_meta)

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
            "ready": {
                "task_cmd_in_listening": bool(task_snap.get("listening")),
                "vision_obs_in_listening": bool(vision_snap.get("listening")),
                "uart_ready": bool(self.cfg.serial.dry_run or getattr(self.uart, "_ser", None) is not None),
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


def run_orchestrator_service(cfg: OrchestratorConfig):
    service = OrchestratorService(cfg)

    def _handle_sig(signum, frame):
        service.log.info("收到信号 %s，准备退出", signum)
        service._running = False

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)
    service.run_forever()
