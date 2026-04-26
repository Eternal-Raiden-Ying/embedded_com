#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import queue
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from common.base_module import BaseModule
from common.runtime_logging import RunLogger, ensure_dir, safe_dump
from orchestrator_service.ipc.transport import JsonlClientSender, JsonlInboundServer

from ..adapters.mqtt_adapter import MqttAdapter
from ..config.schema import GatewayEndpoint, MobileGatewayConfig
from ..protocol import (
    ERROR_CODES,
    MobileCommand,
    MobileProtocolError,
    MobileStatus,
    SUPPORTED_TARGETS,
    make_error_status,
    new_id,
    now_ts,
)


ORCHESTRATOR_STATE_MAP: Dict[str, Tuple[str, int]] = {
    "IDLE": ("idle", 0),
    "SEARCH_TABLE": ("searching", 15),
    "COARSE_ALIGN": ("searching", 30),
    "CONTROLLED_APPROACH": ("approaching", 45),
    "FINAL_LOCK": ("approaching", 60),
    "AT_TABLE_EDGE": ("approaching", 68),
    "SEARCH_TARGET_INIT": ("searching", 75),
    "EDGE_SLIDE_SEARCH": ("searching", 82),
    "TARGET_CONFIRM": ("searching", 88),
    "TARGET_LOCKED": ("approaching", 92),
    "FREEZE_BASE": ("approaching", 96),
    "LEAVE_EDGE": ("approaching", 72),
    "RELOCATE_TO_EDGE": ("searching", 70),
    "REACQUIRE_EDGE": ("searching", 72),
    "NEXT_TABLE": ("searching", 76),
    "AVOID_OBSTACLE": ("searching", 65),
    "RETURN_HOME": ("returning", 75),
    "ERROR_RECOVERY": ("error", 0),
    "DONE": ("completed", 100),
}


@dataclass
class TaskTemplate:
    command: str
    target: Optional[str]
    session_id: str
    text: Optional[str] = None


ACK_REQUIRED_MODES = {"orchestrator_tcp"}
TCP_BACKEND_ALIASES = {"orchestrator_bridge": "orchestrator_tcp", "dry_orchestrator_tcp": "tcp_no_ack"}


class TcpTaskCmdBackend:
    def __init__(self, endpoint: GatewayEndpoint, name: str = "mobile_gateway_task_cmd_out"):
        self.sender = JsonlClientSender(
            mode=endpoint.transport,
            tcp_host=endpoint.host,
            tcp_port=endpoint.port,
            uds_path=endpoint.uds_path,
            name=name,
            send_mode=endpoint.send_mode,
        )

    def start(self) -> None:
        return None

    def stop(self) -> None:
        self.sender.close()

    def submit(self, payload: Dict[str, Any]) -> Tuple[bool, str]:
        ok = self.sender.send(payload)
        if ok:
            return True, "task_cmd forwarded"
        snap = self.sender.snapshot()
        return False, f"task_cmd forward failed: {snap.get('link_state')}"


class MockOrchestratorBackend:
    def __init__(self, step_interval_s: float = 0.20):
        self.step_interval_s = max(0.05, float(step_interval_s))
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._stop_requested = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self._active_task: Optional[Dict[str, Any]] = None

    def start(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        self._callback = callback
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="mobile_gateway_mock_backend")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def submit(self, payload: Dict[str, Any]) -> Tuple[bool, str]:
        intent = str(payload.get("intent", "")).strip().upper()
        if intent == "STOP":
            session_id = str(payload.get("session_id") or (self._active_task or {}).get("session_id") or "")
            epoch = int(payload.get("epoch", 0) or 0)
            self._stop_requested.set()
            self._emit("task_ack", {
                "ts": now_ts(),
                "type": "task_ack",
                "cmd_id": payload.get("cmd_id"),
                "session_id": session_id,
                "epoch": epoch,
                "accepted": True,
                "state": "IDLE",
                "reason": "STOP accepted",
                "source": "mock_orchestrator",
            })
            self._emit("state_block", {
                "ts": now_ts(),
                "state": "IDLE",
                "prev_state": ((self._active_task or {}).get("state") or "SEARCH_TABLE"),
                "task_intent": "",
                "active_target": "",
                "session_id": session_id,
                "epoch": epoch,
                "last_enter_reason": "收到 STOP 命令",
                "last_fail_reason": "",
            })
            return True, "STOP accepted"
        self._queue.put(dict(payload))
        self._emit("task_ack", {
            "ts": now_ts(),
            "type": "task_ack",
            "cmd_id": payload.get("cmd_id"),
            "session_id": payload.get("session_id"),
            "epoch": int(payload.get("epoch", 0) or 0),
            "accepted": True,
            "state": "SEARCH_TABLE" if intent == "FIND" else "RETURN_HOME",
            "reason": f"{intent} accepted",
            "source": "mock_orchestrator",
        })
        return True, f"{intent} accepted"

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self._callback is not None:
            self._callback(event_type, dict(payload))

    def _emit_state(self, task: Dict[str, Any], state: str, reason: str, prev_state: str = "") -> None:
        self._emit("state_block", {
            "ts": now_ts(),
            "state": state,
            "prev_state": prev_state,
            "task_intent": task.get("intent", ""),
            "active_target": task.get("target", ""),
            "session_id": task.get("session_id", ""),
            "epoch": int(task.get("epoch", 0) or 0),
            "last_enter_reason": reason,
            "last_fail_reason": "",
        })

    def _sleep_or_stop(self, task: Dict[str, Any], prev_state: str) -> bool:
        deadline = time.time() + self.step_interval_s
        while time.time() < deadline:
            if self._stop_event.is_set():
                return False
            if self._stop_requested.is_set():
                self._stop_requested.clear()
                self._emit_state(task, "IDLE", "收到 STOP 命令", prev_state=prev_state)
                self._active_task = None
                return False
            time.sleep(0.02)
        return True

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._active_task = dict(task)
            intent = str(task.get("intent", "")).strip().upper()
            if intent == "FIND":
                seq = [
                    ("SEARCH_TABLE", "正在搜索桌边"),
                    ("COARSE_ALIGN", "正在对齐桌边"),
                    ("SEARCH_TARGET_INIT", f"正在搜索 {task.get('target') or '目标'}"),
                    ("TARGET_LOCKED", f"已经锁定 {task.get('target') or '目标'}"),
                    ("DONE", f"已完成 {task.get('target') or '目标'} 取物流程"),
                    ("IDLE", "任务完成，回到空闲"),
                ]
            else:
                seq = [
                    ("RETURN_HOME", "正在返回起点"),
                    ("DONE", "已返回起点"),
                    ("IDLE", "返航完成，回到空闲"),
                ]
            prev_state = ""
            for state, reason in seq:
                self._active_task["state"] = state
                self._emit_state(task, state, reason, prev_state=prev_state)
                if not self._sleep_or_stop(task, prev_state=state):
                    break
                prev_state = state
            self._active_task = None


class OrchestratorStateObserver:
    def __init__(self, runs_dir: str = "", state_blocks_path: str = ""):
        self.runs_dir = Path(runs_dir)
        self.state_blocks_path = Path(state_blocks_path) if str(state_blocks_path or "").strip() else None
        self._current_path: Optional[Path] = None
        self._offset = 0

    def poll(self) -> List[Dict[str, Any]]:
        latest = self._latest_state_file()
        if latest is None:
            return []
        if latest != self._current_path:
            self._current_path = latest
            try:
                self._offset = latest.stat().st_size
            except FileNotFoundError:
                self._offset = 0
        try:
            with latest.open("r", encoding="utf-8") as fp:
                fp.seek(self._offset)
                lines = fp.readlines()
                self._offset = fp.tell()
        except FileNotFoundError:
            return []
        out: List[Dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out

    def _latest_state_file(self) -> Optional[Path]:
        if self.state_blocks_path is not None:
            return self.state_blocks_path if self.state_blocks_path.exists() else None
        if not self.runs_dir.exists():
            return None
        candidates = sorted(self.runs_dir.glob("run_*/state_blocks.jsonl"))
        if not candidates:
            return None
        return candidates[-1]


class MobileGatewayService(BaseModule):
    def __init__(self, cfg: MobileGatewayConfig):
        self.cfg = cfg
        super().__init__("mobile_gateway", cfg.runtime.log_enabled, cfg.runtime.log_mode)
        ensure_dir(cfg.runtime.log_dir)
        ensure_dir(cfg.runtime.runs_dir)
        ensure_dir(cfg.runtime.pid_dir)
        self.run_logger = RunLogger("mobile_gateway", cfg.runtime.runs_dir, cfg.runtime.stack_run_id)
        self.command_server = JsonlInboundServer(
            mode=cfg.command_in.transport,
            tcp_host=cfg.command_in.host,
            tcp_port=cfg.command_in.port,
            uds_path=cfg.command_in.uds_path,
            name="mobile_command_in",
        )
        self.status_sender = JsonlClientSender(
            mode=cfg.status_out.transport,
            tcp_host=cfg.status_out.host,
            tcp_port=cfg.status_out.port,
            uds_path=cfg.status_out.uds_path,
            name="mobile_status_out",
            send_mode=cfg.status_out.send_mode,
        )
        requested_mode = str(cfg.backend.mode or "mock").strip().lower()
        self.backend_mode = TCP_BACKEND_ALIASES.get(requested_mode, requested_mode)
        self.ack_server = None
        if self.backend_mode in ACK_REQUIRED_MODES and str(cfg.orchestrator_task_ack_in.transport).strip().lower() != "disabled":
            self.ack_server = self._build_optional_server(cfg.orchestrator_task_ack_in, "orchestrator_task_ack_in")
        if self.backend_mode in {"orchestrator_tcp", "tcp_no_ack"}:
            self.backend: Any = TcpTaskCmdBackend(cfg.orchestrator_task_cmd_out)
        else:
            self.backend = MockOrchestratorBackend(cfg.backend.mock_step_interval_s)
        self.observer = None
        if self.backend_mode in {"orchestrator_tcp", "tcp_no_ack"} and cfg.backend.observer_enabled:
            self.observer = OrchestratorStateObserver(
                runs_dir=cfg.backend.orchestrator_runs_dir,
                state_blocks_path=cfg.backend.state_blocks_path,
            )
        self.mqtt_adapter: Optional[MqttAdapter] = None
        if cfg.mqtt.enabled:
            self.mqtt_adapter = MqttAdapter(cfg.mqtt, self.handle_mobile_command_payload, logger=self._log_mqtt_event)
        self._running = False
        self._stopped = False
        self._stdin_thread: Optional[threading.Thread] = None
        self._stdin_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._session_epochs: Dict[str, int] = {}
        self._status_seq: Deque[str] = deque(maxlen=8)
        self._last_status_key = ""
        self._last_status_ts = 0.0
        self._last_heartbeat_emit_ts = 0.0
        self._last_observer_poll_ts = 0.0
        self._last_req_ts = 0.0
        self._last_stop_ts = 0.0
        self._active_template: Optional[TaskTemplate] = None
        self._paused_template: Optional[TaskTemplate] = None
        self._last_fetch_template: Optional[TaskTemplate] = None
        self._last_stop_session_id = ""
        self._snapshot: Dict[str, Any] = MobileStatus(
            robot_id=self.cfg.backend.default_robot_id,
            session_id="",
            state="idle",
            ts=now_ts(),
            progress=0,
            message="gateway ready",
        ).to_dict()

    def _build_optional_server(self, ep: GatewayEndpoint, name: str) -> Optional[JsonlInboundServer]:
        if str(ep.transport).strip().lower() == "disabled":
            return None
        return JsonlInboundServer(
            mode=ep.transport,
            tcp_host=ep.host,
            tcp_port=ep.port,
            uds_path=ep.uds_path,
            name=name,
        )

    def _log_mqtt_event(self, level: str, event: str, data: Dict[str, Any]) -> None:
        self.log(level, "mqtt", event, data or None)

    def start(self) -> None:
        self.run_logger.write_meta({
            "service": "mobile_gateway",
            "backend_mode": self.backend_mode,
            "run_dir": str(self.run_logger.run_dir),
            "project_root": self.cfg.runtime.project_root,
            "repo_root": self.cfg.runtime.repo_root,
        })
        self.run_logger.write_service_event("SERVICE_STARTING", backend=self.backend_mode)
        self.run_logger.write_jsonl("config", {
            "backend_mode": self.backend_mode,
            "command_in": {
                "transport": self.cfg.command_in.transport,
                "host": self.cfg.command_in.host,
                "port": self.cfg.command_in.port,
            },
            "status_out": {
                "transport": self.cfg.status_out.transport,
                "host": self.cfg.status_out.host,
                "port": self.cfg.status_out.port,
            },
            "orchestrator_task_cmd_out": {
                "transport": self.cfg.orchestrator_task_cmd_out.transport,
                "host": self.cfg.orchestrator_task_cmd_out.host,
                "port": self.cfg.orchestrator_task_cmd_out.port,
            },
            "orchestrator_task_ack_in": {
                "transport": self.cfg.orchestrator_task_ack_in.transport,
                "host": self.cfg.orchestrator_task_ack_in.host,
                "port": self.cfg.orchestrator_task_ack_in.port,
            },
        })
        self.command_server.start()
        if self.ack_server is not None:
            self.ack_server.start()
        if self.backend_mode == "mock":
            self.backend.start(self._handle_backend_event)
        else:
            self.backend.start()
        if self.mqtt_adapter is not None:
            self.mqtt_adapter.start()
        if self.cfg.runtime.stdin_enabled:
            self._stdin_thread = threading.Thread(target=self._stdin_loop, daemon=True, name="mobile_gateway_stdin")
            self._stdin_thread.start()
        self._running = True
        self.run_logger.write_service_event("SERVICE_READY", backend=self.backend_mode)
        self._publish_status(dict(self._snapshot), force=True)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._running = False
        try:
            self.command_server.close()
        except Exception:
            pass
        if self.ack_server is not None:
            try:
                self.ack_server.close()
            except Exception:
                pass
        try:
            self.status_sender.close()
        except Exception:
            pass
        if self.mqtt_adapter is not None:
            try:
                self.mqtt_adapter.stop()
            except Exception:
                pass
        try:
            self.backend.stop()
        except Exception:
            pass
        self.run_logger.write_service_event("SERVICE_STOPPED")
        self.run_logger.close()

    def run_forever(self) -> None:
        self.start()
        period_s = 1.0 / max(1.0, float(self.cfg.runtime.tick_hz))
        try:
            while self._running:
                loop_start = time.time()
                self._drain_inbound_commands()
                self._drain_ack_messages()
                self._drain_orchestrator_observer()
                self._emit_heartbeat_if_needed()
                elapsed = time.time() - loop_start
                time.sleep(max(0.0, period_s - elapsed))
        finally:
            self.stop()

    def _stdin_loop(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception as exc:
                self._publish_status(make_error_status(
                    self.cfg.backend.default_robot_id,
                    "",
                    f"stdin JSON decode failed: {exc}",
                    ERROR_CODES["invalid_json"],
                ))
                continue
            self._stdin_queue.put({"payload": payload, "recv_ts": now_ts(), "source": "stdin"})

    def _drain_inbound_commands(self) -> None:
        items = list(self.command_server.drain())
        while True:
            try:
                items.append(self._stdin_queue.get_nowait())
            except queue.Empty:
                break
        for item in items:
            payload = dict(item.get("payload") or {})
            recv_ts = float(item.get("recv_ts", now_ts()))
            self.run_logger.write_ipc_record(
                "RX",
                "mobile_command_in",
                "received",
                msg_type="mobile_command",
                session_id=payload.get("session_id"),
                data={"payload": payload},
            )
            self._last_req_ts = recv_ts
            self._handle_command_payload(payload)

    def _drain_ack_messages(self) -> None:
        if self.ack_server is None:
            return
        for item in self.ack_server.drain():
            payload = dict(item.get("payload") or {})
            self._handle_task_ack(payload)

    def _drain_orchestrator_observer(self) -> None:
        if self.observer is None:
            return
        now = time.time()
        if (now - self._last_observer_poll_ts) < float(self.cfg.backend.observer_poll_interval_s):
            return
        self._last_observer_poll_ts = now
        for payload in self.observer.poll():
            self._handle_state_block(payload)

    def handle_mobile_command_payload(self, payload: Dict[str, Any]) -> None:
        try:
            command = MobileCommand.from_dict(
                payload,
                default_robot_id=self.cfg.backend.default_robot_id,
                supported_targets=SUPPORTED_TARGETS,
            )
        except MobileProtocolError as exc:
            fallback_session = str(payload.get("session_id") or "")
            self._publish_status(make_error_status(
                str(payload.get("robot_id") or self.cfg.backend.default_robot_id),
                fallback_session,
                str(exc),
                exc.error_code,
                target=payload.get("target"),
                command=payload.get("cmd"),
                epoch=int(payload.get("epoch", 0) or 0),
            ))
            return
        if command.cmd != "stop" and self._in_stop_cooldown():
            self._publish_status(make_error_status(
                command.robot_id,
                command.session_id,
                "stop cooldown active; retry after a short delay",
                ERROR_CODES["busy"],
                target=command.target,
                command=command.cmd,
                epoch=command.epoch,
            ))
            return
        if command.cmd == "query_status":
            snapshot = dict(self._snapshot)
            snapshot["robot_id"] = command.robot_id
            snapshot["session_id"] = snapshot.get("session_id") or command.session_id
            snapshot["command"] = "query_status"
            snapshot["ts"] = now_ts()
            self._publish_status(snapshot, force=True)
            return
        if command.cmd == "resume":
            self._handle_resume(command)
            return
        if command.cmd == "retry_search":
            self._handle_retry_search(command)
            return
        if command.cmd == "stop":
            self._handle_stop(command)
            return
        if self.cfg.backend.enforce_single_flight and self._is_busy():
            self._publish_status(make_error_status(
                command.robot_id,
                command.session_id,
                "gateway busy; send stop before starting another task",
                ERROR_CODES["busy"],
                target=command.target,
                command=command.cmd,
                epoch=command.epoch,
            ))
            return
        if command.cmd == "fetch_object":
            template = TaskTemplate(command="fetch_object", target=command.target, session_id=command.session_id, text=command.text)
            self._last_fetch_template = template
            self._active_template = template
            self._submit_orchestrator_task(command, intent="FIND", target=command.target, session_id=command.session_id)
            return
        if command.cmd == "go_home":
            template = TaskTemplate(command="go_home", target=None, session_id=command.session_id, text=command.text)
            self._active_template = template
            self._submit_orchestrator_task(command, intent="RETURN", session_id=command.session_id)
            return

    def _handle_command_payload(self, payload: Dict[str, Any]) -> None:
        self.handle_mobile_command_payload(payload)

    def _handle_resume(self, command: MobileCommand) -> None:
        if self._paused_template is None:
            self._publish_status(make_error_status(
                command.robot_id,
                command.session_id,
                "no paused task to resume",
                ERROR_CODES["resume_unavailable"],
                command=command.cmd,
                epoch=command.epoch,
            ))
            return
        template = self._paused_template
        self._active_template = template
        self._paused_template = None
        if template.command == "fetch_object":
            self._submit_orchestrator_task(command, intent="FIND", target=template.target, session_id=template.session_id, high_level_command="resume")
        else:
            self._submit_orchestrator_task(command, intent="RETURN", session_id=template.session_id, high_level_command="resume")

    def _handle_retry_search(self, command: MobileCommand) -> None:
        if self._last_fetch_template is None:
            self._publish_status(make_error_status(
                command.robot_id,
                command.session_id,
                "no fetch_object task to retry",
                ERROR_CODES["resume_unavailable"],
                command=command.cmd,
                epoch=command.epoch,
            ))
            return
        template = self._last_fetch_template
        self._active_template = template
        self._paused_template = None
        self._submit_orchestrator_task(command, intent="FIND", target=template.target, session_id=template.session_id, high_level_command="retry_search")

    def _handle_stop(self, command: MobileCommand) -> None:
        session_id = self._active_template.session_id if self._active_template is not None else command.session_id
        if self._active_template is not None and self._active_template.command in {"fetch_object", "go_home"}:
            self._paused_template = self._active_template
        self._last_stop_session_id = session_id
        self._last_stop_ts = now_ts()
        self._submit_orchestrator_task(command, intent="STOP", session_id=session_id, high_level_command="stop", force=True)

    def _submit_orchestrator_task(
        self,
        command: MobileCommand,
        *,
        intent: str,
        target: Optional[str] = None,
        session_id: str,
        high_level_command: Optional[str] = None,
        force: bool = False,
    ) -> None:
        epoch = self._next_epoch(session_id, explicit_epoch=command.epoch, force_new=not force)
        task_cmd = {
            "ts": now_ts(),
            "type": "task_cmd",
            "intent": intent,
            "confidence": float(self.cfg.backend.default_confidence),
            "cmd_id": new_id("cmd"),
            "session_id": session_id,
            "epoch": epoch,
            "source": "mobile_gateway",
            "text": command.text or self._default_text(high_level_command or command.cmd, target),
        }
        if target:
            task_cmd["target"] = target
        if intent == "STOP":
            task_cmd["high_priority"] = True
        ok, reason = self.backend.submit(task_cmd)
        self.run_logger.write_jsonl("mobile_command", command.to_dict())
        self.run_logger.write_jsonl("task_cmd_forward", task_cmd)
        self.run_logger.write_ipc_record(
            "TX",
            "orchestrator_task_cmd_out",
            "forwarded" if ok else "forward_failed",
            msg_type="task_cmd",
            session_id=session_id,
            epoch=epoch,
            ok=ok,
            data={"payload": task_cmd, "reason": reason},
        )
        if not ok:
            self._publish_status(make_error_status(
                command.robot_id,
                session_id,
                reason,
                ERROR_CODES["backend_unavailable"],
                target=target,
                command=high_level_command or command.cmd,
                epoch=epoch,
            ))
            return
        state = "stopping" if intent == "STOP" else "submitted"
        progress = 0 if intent == "STOP" else 5
        message = {
            "STOP": "已提交停止命令",
            "RETURN": "已提交返航命令",
            "FIND": f"已提交取物命令，目标 {target}",
        }.get(intent, reason)
        self._publish_status(MobileStatus(
            robot_id=command.robot_id,
            session_id=session_id,
            state=state,
            target=target,
            message=message,
            progress=progress,
            command=high_level_command or command.cmd,
            epoch=epoch,
            ts=now_ts(),
        ).to_dict())

    def _handle_backend_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if event_type == "task_ack":
            self._handle_task_ack(payload)
        elif event_type == "state_block":
            self._handle_state_block(payload)

    def _handle_task_ack(self, payload: Dict[str, Any]) -> None:
        session_id = str(payload.get("session_id") or "")
        accepted = bool(payload.get("accepted", False))
        state = "accepted" if accepted else "error"
        message = str(payload.get("reason") or ("command accepted" if accepted else "command rejected"))
        status = MobileStatus(
            robot_id=self.cfg.backend.default_robot_id,
            session_id=session_id,
            state=state,
            target=self._snapshot.get("target"),
            message=message,
            progress=8 if accepted else 0,
            error_code=None if accepted else ERROR_CODES["task_rejected"],
            command=self._snapshot.get("command"),
            backend_state=str(payload.get("state") or ""),
            epoch=int(payload.get("epoch", 0) or 0),
            ts=now_ts(),
        ).to_dict()
        self.run_logger.write_jsonl("task_ack", payload)
        self.run_logger.write_ipc_record(
            "RX",
            "orchestrator_task_ack_in",
            "received",
            msg_type="task_ack",
            session_id=session_id,
            epoch=int(payload.get("epoch", 0) or 0),
            ok=accepted,
            data={"payload": payload},
        )
        self._publish_status(status)

    def _handle_state_block(self, payload: Dict[str, Any]) -> None:
        raw_state = str(payload.get("state", "IDLE") or "IDLE").strip().upper()
        unified_state, progress = ORCHESTRATOR_STATE_MAP.get(raw_state, ("unknown", 0))
        message = str(payload.get("last_fail_reason") or payload.get("last_enter_reason") or raw_state).strip()
        target = str(payload.get("active_target") or self._snapshot.get("target") or "").strip() or None
        session_id = str(payload.get("session_id") or self._snapshot.get("session_id") or "")
        epoch = int(payload.get("epoch", 0) or 0)
        if raw_state == "IDLE" and self._last_stop_session_id and session_id == self._last_stop_session_id:
            unified_state = "stopped"
            progress = 0
            self._active_template = None
        elif raw_state == "DONE":
            unified_state = "completed"
            progress = 100
            self._active_template = None
            self._paused_template = None
        elif raw_state == "ERROR_RECOVERY":
            unified_state = "error"
            progress = 0
        status = MobileStatus(
            robot_id=self.cfg.backend.default_robot_id,
            session_id=session_id,
            state=unified_state,
            target=target,
            message=message or unified_state,
            progress=progress,
            command=self._snapshot.get("command"),
            backend_state=raw_state,
            epoch=epoch,
            error_code=ERROR_CODES["backend_unavailable"] if unified_state == "error" else None,
            ts=now_ts(),
        ).to_dict()
        self.run_logger.write_jsonl("orchestrator_state_block", payload)
        self._publish_status(status)

    def _publish_status(self, payload: Dict[str, Any], force: bool = False) -> None:
        payload = dict(payload)
        payload.setdefault("type", "mobile_status")
        payload.setdefault("robot_id", self.cfg.backend.default_robot_id)
        payload.setdefault("ts", now_ts())
        dedup_payload = dict(payload)
        dedup_payload.pop("ts", None)
        key = safe_dump(dedup_payload)
        if not force and key == self._last_status_key:
            return
        self._last_status_key = key
        self._last_status_ts = float(payload.get("ts", now_ts()))
        self._snapshot = dict(payload)
        self._status_seq.append(str(payload.get("state", "")))
        self.run_logger.write_jsonl("mobile_status", payload)
        self.run_logger.write_ipc_record(
            "TX",
            "mobile_status_out",
            "published",
            msg_type="mobile_status",
            session_id=payload.get("session_id"),
            epoch=int(payload.get("epoch", 0) or 0),
            ok=True,
            data={"payload": payload},
        )
        if self.cfg.runtime.status_stdout:
            print(json.dumps(payload, ensure_ascii=False))
        if self.cfg.status_out.transport != "disabled":
            self.status_sender.send(payload)
        if self.mqtt_adapter is not None:
            self.mqtt_adapter.publish_status(payload)
            if str(payload.get("state", "")).strip().lower() in {"submitted", "accepted", "error", "stopping", "stopped", "completed", "idle"}:
                self.mqtt_adapter.publish_ack(payload)

    def _emit_heartbeat_if_needed(self) -> None:
        now = now_ts()
        if (now - self._last_heartbeat_emit_ts) < float(self.cfg.runtime.heartbeat_period_s):
            return
        last_status_age = (now - self._last_status_ts) if self._last_status_ts else None
        self._last_heartbeat_emit_ts = now
        heartbeat_payload = {
            "type": "mobile_gateway_heartbeat",
            "robot_id": self.cfg.backend.default_robot_id,
            "ts": now,
            "backend_mode": self.backend_mode,
            "state": self._snapshot.get("state", "idle"),
            "session_id": self._snapshot.get("session_id", ""),
            "epoch": int(self._snapshot.get("epoch", 0) or 0),
            "status_age_s": last_status_age,
            "recent_states": list(self._status_seq),
        }
        self.run_logger.write_heartbeat_record(
            stage=str(self._snapshot.get("state", "idle")),
            mode=self.backend_mode,
            session_id=self._snapshot.get("session_id"),
            epoch=int(self._snapshot.get("epoch", 0) or 0),
            last_req_age_s=(now - self._last_req_ts) if self._last_req_ts else None,
            last_obs_send_age_s=last_status_age,
            ready={
                "command_in_listening": bool(self.command_server.snapshot().get("listening")),
                "ack_in_enabled": self.ack_server is not None,
                "observer_enabled": self.observer is not None,
                "mqtt_enabled": self.mqtt_adapter is not None,
            },
            data={
                "last_state": self._snapshot.get("state"),
                "recent_states": list(self._status_seq),
            },
        )
        if self.mqtt_adapter is not None:
            self.mqtt_adapter.publish_heartbeat(heartbeat_payload)

    def _is_busy(self) -> bool:
        return str(self._snapshot.get("state", "")).strip().lower() in {
            "submitted",
            "accepted",
            "searching",
            "approaching",
            "returning",
            "stopping",
        }

    def _next_epoch(self, session_id: str, explicit_epoch: int = 0, force_new: bool = True) -> int:
        if explicit_epoch > 0:
            self._session_epochs[session_id] = explicit_epoch
            return explicit_epoch
        next_epoch = int(self._session_epochs.get(session_id, 0) or 0)
        next_epoch = next_epoch + 1 if force_new else max(1, next_epoch)
        self._session_epochs[session_id] = next_epoch
        return next_epoch

    def _default_text(self, command: str, target: Optional[str]) -> str:
        if command in {"fetch_object", "retry_search"} and target:
            return f"fetch {target}"
        if command == "go_home":
            return "go home"
        if command == "resume":
            return "resume task"
        if command == "stop":
            return "stop"
        return command

    def _in_stop_cooldown(self) -> bool:
        if self._last_stop_ts <= 0:
            return False
        return (now_ts() - self._last_stop_ts) < float(self.cfg.backend.stop_cooldown_s)


def run_mobile_gateway_service(cfg: MobileGatewayConfig) -> None:
    service = MobileGatewayService(cfg)

    def _handle_sig(signum, frame) -> None:
        service.log_info("runtime", f"signal {signum} received; stopping gateway")
        service._running = False

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)
    service.run_forever()
