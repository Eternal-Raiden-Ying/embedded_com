#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import queue
import signal
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from common.base_module import BaseModule
from common.console_presenter import DemoConsolePresenter
from common.runtime_logging import OperatorConsole, RunLogger, ensure_dir, safe_dump
from orchestrator_service.ipc.transport import JsonlClientSender, JsonlInboundServer

from ..adapters.mqtt_adapter import MqttAdapter
from ..config.board_config import build_config
from ..config.schema import GatewayEndpoint, MobileGatewayConfig
from ..protocol import (
    ERROR_CODES,
    MobileCommand,
    MobileProtocolError,
    MobileStatus,
    ROBOT_ID,
    SUPPORTED_TARGETS,
    make_error_status,
    new_id,
    now_ts,
)


ORCHESTRATOR_STATE_MAP: Dict[str, Tuple[str, int]] = {
    "IDLE": ("idle", 0),
    "SEARCH_TABLE": ("searching_table", 20),
    "SEARCH_OBJECT": ("searching", 70),
    "COARSE_ALIGN": ("aligning_table", 35),
    "CONTROLLED_APPROACH": ("approaching_table", 50),
    "FINAL_LOCK": ("locking_table", 65),
    "EDGE_FOLLOW_OBJECT_SEARCH": ("searching_object", 82),
    "DETECTOR_UNAVAILABLE": ("detector_unavailable", 0),
    "DOCK_RETRY": ("docking_retry", 45),
    "AT_TABLE_EDGE": ("running", 68),
    "SEARCH_TARGET_INIT": ("searching", 75),
    "EDGE_SLIDE_SEARCH": ("searching", 82),
    "TARGET_CONFIRM": ("searching", 88),
    "TARGET_LOCKED": ("running", 92),
    "FREEZE_BASE": ("running", 96),
    "LEAVE_EDGE": ("running", 72),
    "RELOCATE_TO_EDGE": ("searching", 70),
    "REACQUIRE_EDGE": ("searching", 72),
    "NEXT_TABLE": ("searching", 76),
    "AVOID_OBSTACLE": ("searching", 65),
    "RETURN_HOME": ("running", 75),
    "GRASP": ("running", 90),
    "ERROR_RECOVERY": ("error", 0),
    "STOP": ("stopped", 0),
    "STOPPED": ("stopped", 0),
    "DONE": ("idle", 100),
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
            tcp_host=getattr(endpoint, "host", "127.0.0.1"),
            tcp_port=getattr(endpoint, "port", 0),
            uds_path=getattr(endpoint, "uds_path", "") or getattr(endpoint, "ipc_socket_path", ""),
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


class MobileHttpCommandServer:
    def __init__(
        self,
        endpoint: GatewayEndpoint,
        task_handler: Callable[[Dict[str, Any]], Dict[str, Any]],
        health_handler: Callable[[], Dict[str, Any]],
        logger: Callable[[str, str, Dict[str, Any]], None],
    ):
        self.host = str(getattr(endpoint, "host", "0.0.0.0") or getattr(endpoint, "tcp_host", "0.0.0.0") or "0.0.0.0")
        self.port = int(getattr(endpoint, "port", 0) or getattr(endpoint, "tcp_port", 0) or 0)
        self.task_handler = task_handler
        self.health_handler = health_handler
        self.logger = logger
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.listening = False
        self.total_recv_count = 0
        self.last_recv_ts = 0.0

    def start(self) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "MobileGatewayHTTP/1.0"

            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path.split("?", 1)[0] != "/health":
                    outer.logger("warn", "http request not found", {"method": "GET", "path": self.path, "client": self.client_address[0]})
                    self._send_json(404, {"ok": False, "error": "not found"})
                    return
                outer.logger("info", "http health request", {"method": "GET", "path": self.path, "client": self.client_address[0]})
                self._send_json(200, outer.health_handler())

            def do_POST(self) -> None:
                if self.path.split("?", 1)[0] != "/task":
                    outer.logger("warn", "http request not found", {"method": "POST", "path": self.path, "client": self.client_address[0]})
                    self._send_json(404, {"ok": False, "error": "not found"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0") or 0)
                    if length <= 0 or length > 1024 * 1024:
                        raise ValueError("invalid Content-Length")
                    raw = self.rfile.read(length)
                    payload = json.loads(raw.decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("task payload must be a JSON object")
                except Exception as exc:
                    outer.logger("warn", "http bad request", {"method": "POST", "path": self.path, "client": self.client_address[0], "error": str(exc)})
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                outer.total_recv_count += 1
                outer.last_recv_ts = now_ts()
                outer.logger("info", "http task request", {"method": "POST", "path": self.path, "client": self.client_address[0]})
                result = outer.task_handler(payload)
                self._send_json(202 if result.get("ok") else 502, result)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self.host, self.port = self._server.server_address[:2]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="mobile_gateway_http")
        self._thread.start()
        self.listening = True
        self.logger("info", "http listening", {"host": self.host, "port": self.port, "paths": ["/health", "/task"]})

    def close(self) -> None:
        self.listening = False
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "mode": "http",
            "host": self.host,
            "port": self.port,
            "listening": self.listening,
            "total_recv_count": self.total_recv_count,
            "last_recv_ts": self.last_recv_ts,
        }


class MobileJsonTcpCommandServer:
    def __init__(self, host: str, port: int, logger: Callable[[str, str, Dict[str, Any]], None]):
        self.host = str(host or "0.0.0.0")
        self.port = int(port or 0)
        self.logger = logger
        self._server_sock: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._client_threads: List[threading.Thread] = []
        self._stop = threading.Event()
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.listening = False
        self.start_ts = 0.0
        self.last_recv_ts = 0.0
        self.total_recv_count = 0
        self.invalid_json_count = 0
        self.client_count = 0

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        self.host, self.port = sock.getsockname()[:2]
        sock.listen(4)
        sock.settimeout(1.0)
        self._server_sock = sock
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True, name="mobile_gateway_json_tcp")
        self._accept_thread.start()
        self.listening = True
        self.start_ts = now_ts()
        self.logger("info", "json tcp listening", {"host": self.host, "port": self.port})

    def close(self) -> None:
        self._stop.set()
        self.listening = False
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except Exception:
                pass
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=1.5)
            self._accept_thread = None
        for thread in self._client_threads:
            thread.join(timeout=0.5)

    def drain(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return items

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": "mobile_command_in_legacy_json_tcp",
            "mode": "json_tcp",
            "host": self.host,
            "port": self.port,
            "listening": self.listening,
            "start_ts": self.start_ts,
            "last_recv_ts": self.last_recv_ts,
            "invalid_json_count": self.invalid_json_count,
            "queue_depth": self._queue.qsize(),
            "total_recv_count": self.total_recv_count,
            "client_count": self.client_count,
        }

    def _accept_loop(self) -> None:
        assert self._server_sock is not None
        while not self._stop.is_set():
            try:
                conn, addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            peer = str(addr)
            self.client_count += 1
            self.logger("info", "json tcp peer connected", {"peer": peer, "client_count": self.client_count})
            thread = threading.Thread(target=self._client_loop, args=(conn, peer), daemon=True, name="mobile_gateway_json_tcp_client")
            self._client_threads.append(thread)
            thread.start()

    def _client_loop(self, conn: socket.socket, peer: str) -> None:
        with conn:
            conn.settimeout(1.0)
            buffer = b""
            while not self._stop.is_set():
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                except Exception as exc:
                    self.logger("warn", "json tcp recv failed", {"peer": peer, "error": repr(exc)})
                    break
                if not chunk:
                    if buffer.strip():
                        self._handle_line(buffer.strip(), peer)
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    self._handle_line(line.strip(), peer)

    def _handle_line(self, raw: bytes, peer: str) -> None:
        if not raw:
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("payload must be a JSON object")
        except Exception as exc:
            self.invalid_json_count += 1
            self.logger("warn", "json tcp bad payload", {"peer": peer, "error": str(exc), "raw": raw[:200].decode("utf-8", errors="replace")})
            return
        self.last_recv_ts = now_ts()
        self.total_recv_count += 1
        self.logger("info", "json tcp command received", {"peer": peer, "cmd": payload.get("cmd"), "cmd_id": payload.get("cmd_id")})
        self._queue.put({"peer": peer, "payload": payload, "recv_ts": self.last_recv_ts})


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


class _NullRunLogger:
    def __init__(self, module_name: str, stack_run_id: str = ""):
        self.module_name = module_name
        self.stack_run_id = str(stack_run_id or "")
        self.run_dir = Path("/dev/null")

    def write_meta(self, payload: Dict[str, Any]) -> None:
        pass

    def write_service_event(self, event: str, **fields: Any) -> None:
        pass

    def write_jsonl(self, name: str, payload: Dict[str, Any]) -> None:
        pass

    def write_ipc_record(self, *args: Any, **kwargs: Any) -> None:
        pass

    def write_heartbeat_record(self, *args: Any, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass


class MobileGatewayService(BaseModule):
    def __init__(self, cfg: MobileGatewayConfig):
        self.cfg = cfg
        super().__init__("mobile_gateway", cfg.runtime.log_enabled, cfg.runtime.log_mode)
        if cfg.runtime.log_enabled:
            ensure_dir(cfg.runtime.log_dir)
        ensure_dir(cfg.runtime.runs_dir)
        ensure_dir(cfg.runtime.pid_dir)
        self.run_logger = RunLogger("mobile_gateway", cfg.runtime.runs_dir, cfg.runtime.stack_run_id) if cfg.runtime.log_enabled else _NullRunLogger("mobile_gateway", cfg.runtime.stack_run_id)
        command_transport = str(cfg.command_in.transport or "disabled").strip().lower()
        legacy_tcp_enabled = (
            command_transport == "http"
            and str(os.getenv("MOBILE_GATEWAY_LEGACY_TCP_ENABLED", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}
        )
        if legacy_tcp_enabled:
            self.command_server = MobileJsonTcpCommandServer(
                os.getenv("MOBILE_GATEWAY_LEGACY_TCP_HOST", "0.0.0.0"),
                int(os.getenv("MOBILE_GATEWAY_LEGACY_TCP_PORT", "9101") or 9101),
                lambda level, msg, data: self.log(level, "legacy_tcp", msg, data),
            )
        else:
            self.command_server = JsonlInboundServer(
                mode="disabled" if command_transport == "http" else command_transport,
                tcp_host=getattr(cfg.command_in, "host", "127.0.0.1"),
                tcp_port=getattr(cfg.command_in, "port", 0),
                uds_path=getattr(cfg.command_in, "uds_path", "") or getattr(cfg.command_in, "ipc_socket_path", ""),
                name="mobile_command_in",
            )
        self.legacy_tcp_enabled = legacy_tcp_enabled
        self.http_server: Optional[MobileHttpCommandServer] = None
        if command_transport == "http":
            self.http_server = MobileHttpCommandServer(
                cfg.command_in,
                self._handle_http_task_payload,
                self._http_health_payload,
                lambda level, msg, data: self.log(level, "http", msg, data),
            )
        self.status_sender = JsonlClientSender(
            mode=cfg.status_out.transport,
            tcp_host=getattr(cfg.status_out, "host", "127.0.0.1"),
            tcp_port=getattr(cfg.status_out, "port", 0),
            uds_path=getattr(cfg.status_out, "uds_path", "") or getattr(cfg.status_out, "ipc_socket_path", ""),
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
            self.mqtt_adapter = MqttAdapter(
                cfg.mqtt,
                self.handle_mobile_command_payload,
                logger=self._log_mqtt_event,
                enable_raw_debug=cfg.runtime.enable_raw_mqtt_debug,
                suppress_heartbeat_success_log=cfg.runtime.suppress_heartbeat_success_log,
            )
        self._running = False
        self._stopped = False
        self._stdin_thread: Optional[threading.Thread] = None
        self._stdin_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._session_epochs: Dict[str, int] = {}
        self._status_seq: Deque[str] = deque(maxlen=8)
        self._last_status_key = ""
        self._last_operator_status_key = ""
        self._mobile_status_console_mode = str(os.getenv("ORCH_MOBILE_STATUS_CONSOLE", "change") or "change").strip().lower()
        if self._mobile_status_console_mode not in {"change", "full", "silent"}:
            self._mobile_status_console_mode = "change"
        self.operator_console = OperatorConsole(
            mode=os.getenv("ORCH_CONSOLE_MODE", "operator"),
            default_interval_s=float(os.getenv("ORCH_OPERATOR_SUMMARY_INTERVAL_S", "1.0") or 1.0),
        )
        self.demo_console = DemoConsolePresenter(
            self.operator_console,
            dry_run=str(os.getenv("ORCH_SERIAL_DRY_RUN", "1")).strip().lower() in {"1", "true", "yes", "on"},
        )
        self._last_status_ts = 0.0
        self._last_heartbeat_emit_ts = 0.0
        self._last_heartbeat_log_ts = 0.0
        self._heartbeat_count = 0
        self._last_observer_poll_ts = 0.0
        self._last_state_block_log_ts = 0.0
        self._last_state_block_log_key = ""
        self._last_req_ts = 0.0
        self._last_stop_ts = 0.0
        self._active_template: Optional[TaskTemplate] = None
        self._paused_template: Optional[TaskTemplate] = None
        self._last_fetch_template: Optional[TaskTemplate] = None
        self._last_stop_session_id = ""
        self._recent_cmd_ids: Deque[str] = deque(maxlen=max(1, int(cfg.runtime.cmd_dedup_cache_size or 64)))
        self._recent_cmd_id_set: Set[str] = set()
        self._snapshot: Dict[str, Any] = MobileStatus(
            robot_id=ROBOT_ID,
            session_id="",
            state="idle",
            ts=now_ts(),
            progress=0,
            message="gateway ready",
        ).to_dict()

    def _is_debug_mode(self) -> bool:
        return str(self.cfg.runtime.mode or "production").strip().lower() == "debug"

    def _build_optional_server(self, ep: GatewayEndpoint, name: str) -> Optional[JsonlInboundServer]:
        if str(ep.transport).strip().lower() == "disabled":
            return None
        return JsonlInboundServer(
            mode=ep.transport,
            tcp_host=getattr(ep, "host", "127.0.0.1"),
            tcp_port=getattr(ep, "port", 0),
            uds_path=getattr(ep, "uds_path", "") or getattr(ep, "ipc_socket_path", ""),
            name=name,
        )

    def _log_mqtt_event(self, level: str, event: str, data: Dict[str, Any]) -> None:
        data = dict(data or {})
        if event == "mqtt_publish" and not self._is_debug_mode():
            return
        if event == "mqtt_message" and not self.cfg.runtime.enable_raw_mqtt_debug and not self._is_debug_mode():
            self.log("info", "mqtt", "cmd received via mqtt", {"topic": data.get("topic")})
            return
        msg = {
            "mqtt_starting": "mqtt starting",
            "mqtt_connected": "mqtt connected",
            "mqtt_disconnected": "mqtt disconnected",
            "mqtt_message": "mqtt message",
            "mqtt_bad_json": "mqtt bad json",
            "mqtt_publish": "mqtt publish",
            "mqtt_disabled": "mqtt disabled",
            "mqtt_stopped": "mqtt stopped",
        }.get(event, event)
        self.log(level, "mqtt", msg, data or None)

    def _endpoint_log_payload(self, endpoint: GatewayEndpoint) -> Dict[str, Any]:
        return {
            "transport": endpoint.transport,
            "host": getattr(endpoint, "host", ""),
            "port": getattr(endpoint, "port", 0),
            "ipc_socket_path": getattr(endpoint, "ipc_socket_path", "") or getattr(endpoint, "uds_path", ""),
        }

    def _command_in_snapshot(self) -> Dict[str, Any]:
        if self.http_server is not None:
            return self.http_server.snapshot()
        return self.command_server.snapshot()

    def _http_health_payload(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "service": "mobile_gateway",
            "backend_mode": self.backend_mode,
            "runtime_mode": self.cfg.runtime.mode,
            "command_in": self._command_in_snapshot(),
            "mqtt_enabled": self.mqtt_adapter is not None,
            "mqtt": {
                "enabled": bool(self.cfg.mqtt.enabled),
                "broker_host": self.cfg.mqtt.broker_host,
                "broker_port": self.cfg.mqtt.broker_port,
                "topic_cmd": self.cfg.mqtt.topics.cmd,
            },
            "legacy_tcp": self.command_server.snapshot() if self.legacy_tcp_enabled else {"enabled": False},
            "orchestrator_task_cmd_out": self._endpoint_log_payload(self.cfg.orchestrator_task_cmd_out),
            "status_out_enabled": str(self.cfg.status_out.transport).strip().lower() != "disabled",
            "ack_in_enabled": self.ack_server is not None,
        }

    def _coerce_http_task_cmd(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raw_action = str(payload.get("action") or "").strip().lower()
        raw_intent = str(payload.get("intent") or "").strip().upper()
        if not raw_intent:
            if raw_action in {"stop", "halt", "cancel"}:
                raw_intent = "STOP"
            elif raw_action in {"find", "fetch", "fetch_object", "pick"}:
                raw_intent = "FIND"
            else:
                raw_intent = "RETURN"

        params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
        target = payload.get("target") or params.get("target") or params.get("object")
        task_cmd = {
            "ts": float(payload.get("ts", now_ts()) or now_ts()),
            "type": "task_cmd",
            "intent": raw_intent,
            "confidence": float(payload.get("confidence", self.cfg.backend.default_confidence) or self.cfg.backend.default_confidence),
            "cmd_id": str(payload.get("cmd_id") or payload.get("task_id") or new_id("cmd")),
            "session_id": str(payload.get("session_id") or payload.get("task_id") or new_id("sess")),
            "epoch": int(payload.get("epoch", 0) or 0),
            "source": str(payload.get("source") or "mobile_http"),
            "text": str(payload.get("text") or raw_action or "").strip(),
        }
        if raw_intent == "FIND" and target:
            task_cmd["target"] = str(target).strip().lower()
        return {k: v for k, v in task_cmd.items() if v not in (None, "")}

    def _handle_http_task_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._last_req_ts = now_ts()
        self.log_info("protocol", "received mobile command", {"transport": "http", "payload": payload})
        if "cmd" in payload or str(payload.get("type", "")).strip().upper() == "FIND_AND_PICK":
            self.handle_mobile_command_payload(payload)
            return {"ok": True, "accepted": True, "mode": "mobile_command"}

        task_cmd = self._coerce_http_task_cmd(payload)
        ok, reason = self.backend.submit(task_cmd)
        self.run_logger.write_jsonl("mobile_command", payload)
        self.run_logger.write_jsonl("task_cmd_forward", task_cmd)
        self.run_logger.write_ipc_record(
            "TX",
            "orchestrator_task_cmd_out",
            "forwarded" if ok else "forward_failed",
            msg_type="task_cmd",
            session_id=task_cmd.get("session_id"),
            epoch=task_cmd.get("epoch"),
            ok=ok,
            data={"payload": task_cmd, "reason": reason},
        )
        if ok:
            self.log_info("backend", "forwarded task_cmd", {
                "cmd_id": task_cmd.get("cmd_id"),
                "intent": task_cmd.get("intent"),
                "session_id": task_cmd.get("session_id"),
                "reason": reason,
            })
        else:
            self.log_error("backend", "forward task_cmd failed", {
                "cmd_id": task_cmd.get("cmd_id"),
                "intent": task_cmd.get("intent"),
                "reason": reason,
            })
        return {"ok": ok, "accepted": ok, "reason": reason, "task_cmd": task_cmd}

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
                "ipc_socket_path": getattr(self.cfg.command_in, "ipc_socket_path", "") or getattr(self.cfg.command_in, "uds_path", ""),
                "host": getattr(self.cfg.command_in, "host", ""),
                "port": getattr(self.cfg.command_in, "port", 0),
            },
            "status_out": {
                "transport": self.cfg.status_out.transport,
                "ipc_socket_path": getattr(self.cfg.status_out, "ipc_socket_path", "") or getattr(self.cfg.status_out, "uds_path", ""),
                "host": getattr(self.cfg.status_out, "host", ""),
                "port": getattr(self.cfg.status_out, "port", 0),
            },
            "orchestrator_task_cmd_out": {
                "transport": self.cfg.orchestrator_task_cmd_out.transport,
                "ipc_socket_path": getattr(self.cfg.orchestrator_task_cmd_out, "ipc_socket_path", "") or getattr(self.cfg.orchestrator_task_cmd_out, "uds_path", ""),
                "host": getattr(self.cfg.orchestrator_task_cmd_out, "host", ""),
                "port": getattr(self.cfg.orchestrator_task_cmd_out, "port", 0),
            },
            "orchestrator_task_ack_in": {
                "transport": self.cfg.orchestrator_task_ack_in.transport,
                "ipc_socket_path": getattr(self.cfg.orchestrator_task_ack_in, "ipc_socket_path", "") or getattr(self.cfg.orchestrator_task_ack_in, "uds_path", ""),
                "host": getattr(self.cfg.orchestrator_task_ack_in, "host", ""),
                "port": getattr(self.cfg.orchestrator_task_ack_in, "port", 0),
            },
            "mqtt": {
                "enabled": bool(self.cfg.mqtt.enabled),
                "broker_host": self.cfg.mqtt.broker_host,
                "broker_port": self.cfg.mqtt.broker_port,
                "topic_cmd": self.cfg.mqtt.topics.cmd,
            },
        })
        self.command_server.start()
        if self.http_server is not None:
            self.http_server.start()
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
        self.log_info("runtime", "gateway online", {
            "backend_mode": self.backend_mode,
            "runtime_mode": self.cfg.runtime.mode,
            "mqtt_enabled": self.mqtt_adapter is not None,
            "command_in": self._endpoint_log_payload(self.cfg.command_in),
            "legacy_tcp": self.command_server.snapshot() if self.legacy_tcp_enabled else {"enabled": False},
            "orchestrator_task_cmd_out": self._endpoint_log_payload(self.cfg.orchestrator_task_cmd_out),
            "status_out_enabled": str(self.cfg.status_out.transport).strip().lower() != "disabled",
            "ack_in_enabled": self.ack_server is not None,
        })
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
        if self.http_server is not None:
            try:
                self.http_server.close()
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
                default_robot_id=ROBOT_ID,
                supported_targets=SUPPORTED_TARGETS,
                allow_legacy_command_compat=self.cfg.runtime.enable_legacy_command_compat,
            )
        except MobileProtocolError as exc:
            fallback_session = str(payload.get("session_id") or "")
            fallback_cmd_id = str(payload.get("cmd_id") or new_id("cmd"))
            self._publish_gateway_ack(
                {
                    "cmd_id": fallback_cmd_id,
                    "session_id": fallback_session,
                    "epoch": int(payload.get("epoch", 0) or 0),
                    "cmd": str(payload.get("cmd") or payload.get("type") or ""),
                    "target": payload.get("target"),
                    "source": str(payload.get("source") or "wechat_miniprogram"),
                },
                accepted=False,
                message=str(exc),
                error_code=exc.error_code,
            )
            self._publish_status(make_error_status(
                ROBOT_ID,
                fallback_session,
                str(exc),
                exc.error_code,
                target=payload.get("target"),
                command=payload.get("cmd"),
                epoch=int(payload.get("epoch", 0) or 0),
            ))
            return
        if self._is_duplicate_cmd(command.cmd_id):
            self._publish_gateway_ack(command.to_dict(), accepted=True, message="gateway command accepted")
            if self._is_debug_mode():
                self.log_info("protocol", "duplicate cmd ignored", {
                    "cmd_id": command.cmd_id,
                    "cmd": command.cmd,
                    "session_id": command.session_id,
                })
            return
        self._remember_cmd_id(command.cmd_id)
        self.log_info("protocol", "received mobile command", {
            "cmd_id": command.cmd_id,
            "cmd": command.cmd,
            "session_id": command.session_id,
            "target": command.target,
            "source": command.source,
        })
        if command.cmd not in {"query_status", "stop"}:
            self.demo_console.phone_command(command.target or "n/a")
        self._publish_gateway_ack(command.to_dict(), accepted=True, message="gateway command accepted")
        if command.cmd != "stop" and self._in_stop_cooldown():
            self._publish_status(make_error_status(
                ROBOT_ID,
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
            snapshot["robot_id"] = ROBOT_ID
            snapshot["session_id"] = snapshot.get("session_id") or command.session_id
            snapshot["command"] = "query_status"
            snapshot["ts"] = now_ts()
            snapshot["kind"] = "status"
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
                ROBOT_ID,
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
                ROBOT_ID,
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
                ROBOT_ID,
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
            "cmd_id": command.cmd_id,
            "session_id": session_id,
            "epoch": epoch,
            "source": command.source,
        }
        if target:
            task_cmd["target"] = target
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
            self.log_error("backend", "task_cmd forward failed", {
                "cmd_id": command.cmd_id,
                "intent": intent,
                "reason": reason,
            })
            self._publish_status(make_error_status(
                ROBOT_ID,
                session_id,
                reason,
                ERROR_CODES["backend_unavailable"],
                target=target,
                command=high_level_command or command.cmd,
                epoch=epoch,
            ))
            return
        self.log_info("backend", "forwarded task_cmd", {
            "cmd_id": command.cmd_id,
            "intent": intent,
            "session_id": session_id,
            "epoch": epoch,
        })
        state = "submitted"
        progress = 0 if intent == "STOP" else 5
        message = {
            "STOP": "已提交停止命令",
            "RETURN": "已提交返航命令",
            "FIND": f"已提交取物命令，目标 {target}",
        }.get(intent, reason)
        self._publish_status(MobileStatus(
            robot_id=ROBOT_ID,
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
        target = str(payload.get("target") or self._snapshot.get("target") or "").strip() or None
        backend_state = str(payload.get("state") or "").strip().upper()
        error_info = self._backend_error_info(payload)
        message = self._task_ack_message(payload, target=target, backend_state=backend_state, error_info=error_info)
        state = "accepted" if accepted and error_info is None else "error"
        self._publish_task_ack(
            payload,
            accepted=accepted,
            message=message,
            error_code=None if accepted and error_info is None else self._status_error_code(accepted, error_info),
        )
        status = MobileStatus(
            robot_id=ROBOT_ID,
            session_id=session_id,
            state=state,
            target=target,
            message=message,
            progress=8 if accepted else 0,
            error_code=None if accepted and error_info is None else self._status_error_code(accepted, error_info),
            command=self._snapshot.get("command"),
            epoch=int(payload.get("epoch", 0) or 0),
            ts=now_ts(),
        ).to_dict()
        if self._is_debug_mode():
            status["backend_state"] = backend_state
            if error_info is not None:
                status["raw_error"] = error_info["raw_error"]
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
        self.log_info("backend", "task_ack forwarded", {
            "cmd_id": payload.get("cmd_id"),
            "accepted": accepted,
            "session_id": session_id,
            "backend_state": backend_state or None,
        })
        self._publish_status(status)

    def _handle_state_block(self, payload: Dict[str, Any]) -> None:
        raw_state = str(payload.get("state", "IDLE") or "IDLE").strip().upper()
        unified_state, progress = ORCHESTRATOR_STATE_MAP.get(raw_state, ("unknown", 0))
        error_info = self._backend_error_info(payload)
        target = str(payload.get("active_target") or self._snapshot.get("target") or "").strip() or None
        message = self._status_message_for_state(
            raw_state,
            target=target,
            fallback=str(payload.get("last_fail_reason") or payload.get("last_enter_reason") or raw_state).strip(),
            error_info=error_info,
        )
        recoverable_message = self._recoverable_backend_warning(payload)
        if recoverable_message and error_info is None:
            message = recoverable_message
        target = str(payload.get("active_target") or self._snapshot.get("target") or "").strip() or None
        payload_session_id = str(payload.get("session_id") or "")
        session_id = str(payload_session_id or self._snapshot.get("session_id") or "")
        epoch = int(payload.get("epoch", 0) or 0)
        current_epoch = int(self._snapshot.get("epoch", 0) or 0)
        current_state = str(self._snapshot.get("state") or "").strip().lower()
        if (
            raw_state == "IDLE"
            and epoch < current_epoch
            and current_state in {"submitted", "accepted", "searching", "running"}
        ):
            self.run_logger.write_ipc_record(
                "RX",
                "orchestrator_state_block",
                "ignored_stale_epoch",
                msg_type="state_block",
                session_id=payload_session_id,
                epoch=epoch,
                ok=False,
                data={
                    "payload": payload,
                    "current_epoch": current_epoch,
                    "current_state": current_state,
                },
            )
            self.log_info("backend", "stale state_block ignored", {
                "state": raw_state,
                "epoch": epoch,
                "current_epoch": current_epoch,
                "current_state": current_state,
            })
            return
        if raw_state == "IDLE" and self._last_stop_session_id and session_id == self._last_stop_session_id:
            unified_state = "stopped"
            progress = 0
            message = "任务已停止"
            self._active_template = None
            self._last_stop_session_id = ""
        elif raw_state == "DONE":
            unified_state = "idle"
            progress = 100
            message = f"任务完成，已锁定目标 {target}" if target else "target locked, task done"
            self._active_template = None
            self._paused_template = None
        elif raw_state == "IDLE" and str(payload.get("prev_state") or "").strip().upper() == "DONE":
            message = "任务完成，回到空闲"
        elif raw_state == "ERROR_RECOVERY":
            unified_state = "error"
            progress = 0
        if error_info is not None:
            unified_state = "error"
            progress = 0
        status = MobileStatus(
            robot_id=ROBOT_ID,
            session_id=session_id,
            state=unified_state,
            target=target,
            message=message or unified_state,
            progress=progress,
            command=self._snapshot.get("command"),
            epoch=epoch,
            error_code=error_info["error_code"] if error_info is not None else (
                ERROR_CODES["backend_unavailable"] if unified_state == "error" else None
            ),
            ts=now_ts(),
        ).to_dict()
        status["backend_state"] = raw_state
        status["raw_backend_state"] = raw_state
        status["stage"] = payload.get("vision_stage") or payload.get("stage") or ""
        status["mode"] = payload.get("vision_mode") or payload.get("mode") or ""
        status["has_table_edge_obs"] = bool(payload.get("has_table_edge_obs", False))
        status["has_target_obs"] = bool(payload.get("has_target_obs", False))
        status["lock_ready"] = bool(payload.get("lock_ready", False))
        status["lock_reason"] = str(payload.get("lock_reason") or "")
        if isinstance(payload.get("warnings"), list):
            status["warnings"] = list(payload.get("warnings") or [])
        if error_info is not None:
            status["raw_error"] = error_info["raw_error"]
        self._log_orchestrator_state_block(payload)
        self._publish_status(status)

    def _log_orchestrator_state_block(self, payload: Dict[str, Any]) -> None:
        mode = str(getattr(self.cfg.backend, "state_block_log_mode", "summary") or "summary").strip().lower()
        if mode in {"0", "false", "off", "none", "disabled"}:
            return
        record = dict(payload) if mode in {"full", "raw", "debug"} else self._state_block_summary(payload)
        key = safe_dump(self._state_block_log_key(record))
        now = now_ts()
        period_s = max(0.0, float(getattr(self.cfg.backend, "state_block_log_period_s", 1.0) or 0.0))
        if (now - self._last_state_block_log_ts) < period_s and key == self._last_state_block_log_key:
            return
        self._last_state_block_log_ts = now
        self._last_state_block_log_key = key
        self.run_logger.write_jsonl("orchestrator_state_block", record)

    def _state_block_summary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        fields = (
            "ts",
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
            "has_table_edge_obs",
            "has_target_obs",
            "lock_ready",
            "lock_reason",
            "table_found",
            "edge_found",
            "edge_valid",
            "control_level",
            "table_approach_phase",
            "stale_level",
            "stale_guard_active",
            "stale_guard_reason",
            "task_result",
            "edge_retries",
            "slide_entries",
            "target_confirm_count",
            "target_locked_count",
            "last_matched_cls",
            "last_matched_conf",
            "last_edge_conf",
            "lost_reason",
            "warnings",
        )
        return {name: payload.get(name) for name in fields if name in payload}

    def _state_block_log_key(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        key = dict(payload)
        for name in (
            "ts",
            "table_edge_obs_ts",
            "table_edge_obs_age_ms",
            "obs_total_age_ms",
            "control_loop_age_ms",
            "warmup_elapsed_s",
            "table_loss_elapsed_s",
            "target_loss_elapsed_s",
            "tag_loss_elapsed_s",
            "task_total_time_s",
        ):
            key.pop(name, None)
        return key

    def _publish_status(self, payload: Dict[str, Any], force: bool = False) -> None:
        payload = dict(payload)
        payload.setdefault("type", "mobile_status")
        payload.setdefault("robot_id", ROBOT_ID)
        payload["robot_id"] = ROBOT_ID
        payload["kind"] = "status"
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
        self._log_status_event(payload, force=force)
        self._emit_mobile_status_console(payload)
        if self.cfg.runtime.status_stdout and self._mobile_status_console_mode == "full":
            print(json.dumps(payload, ensure_ascii=False))
        if self.cfg.status_out.transport != "disabled":
            self.status_sender.send(payload)
        if self.mqtt_adapter is not None:
            self.mqtt_adapter.publish_status(payload)

    def _emit_heartbeat_if_needed(self) -> None:
        now = now_ts()
        if (now - self._last_heartbeat_emit_ts) < float(self.cfg.runtime.heartbeat_period_s):
            return
        last_status_age = (now - self._last_status_ts) if self._last_status_ts else None
        self._last_heartbeat_emit_ts = now
        self._heartbeat_count += 1
        heartbeat_payload = {
            "type": "mobile_gateway_heartbeat",
            "robot_id": ROBOT_ID,
            "kind": "heartbeat",
            "ts": now,
            "online": True,
            "backend_mode": self.backend_mode,
            "state": self._snapshot.get("state", "idle"),
            "session_id": self._snapshot.get("session_id", ""),
            "epoch": int(self._snapshot.get("epoch", 0) or 0),
        }
        if self._is_debug_mode():
            heartbeat_payload["status_age_s"] = last_status_age
            heartbeat_payload["recent_states"] = list(self._status_seq)
        self.run_logger.write_heartbeat_record(
            stage=str(self._snapshot.get("state", "idle")),
            mode=self.backend_mode,
            session_id=self._snapshot.get("session_id"),
            epoch=int(self._snapshot.get("epoch", 0) or 0),
            last_req_age_s=(now - self._last_req_ts) if self._last_req_ts else None,
            last_obs_send_age_s=last_status_age,
            ready={
                "command_in_listening": bool(self._command_in_snapshot().get("listening")),
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
        self._log_heartbeat_summary(now)

    def _is_busy(self) -> bool:
        return str(self._snapshot.get("state", "")).strip().lower() in {
            "submitted",
            "accepted",
            "searching",
            "running",
        }

    def _next_epoch(self, session_id: str, explicit_epoch: int = 0, force_new: bool = True) -> int:
        if explicit_epoch > 0:
            self._session_epochs[session_id] = explicit_epoch
            return explicit_epoch
        next_epoch = int(self._session_epochs.get(session_id, 0) or 0)
        next_epoch = next_epoch + 1 if force_new else max(1, next_epoch)
        self._session_epochs[session_id] = next_epoch
        return next_epoch

    def _publish_gateway_ack(self, payload: Dict[str, Any], *, accepted: bool, message: str, error_code: Optional[int] = None) -> None:
        ack = {
            "type": "mobile_ack",
            "kind": "gateway_ack",
            "robot_id": ROBOT_ID,
            "cmd_id": str(payload.get("cmd_id") or new_id("cmd")),
            "session_id": str(payload.get("session_id") or ""),
            "epoch": int(payload.get("epoch", 0) or 0),
            "cmd": str(payload.get("cmd") or ""),
            "target": payload.get("target"),
            "message": message,
            "accepted": bool(accepted),
            "error_code": error_code,
            "source": "mobile_gateway",
            "ts": now_ts(),
        }
        ack = {k: v for k, v in ack.items() if v not in (None, "", [], {})}
        self.run_logger.write_jsonl("mobile_ack", ack)
        self.log_info("protocol", "gateway_ack sent", {
            "cmd_id": ack.get("cmd_id"),
            "accepted": ack.get("accepted"),
            "cmd": ack.get("cmd"),
            "session_id": ack.get("session_id"),
        })
        if bool(accepted) and str(ack.get("cmd") or "").strip() not in {"query_status", "stop"}:
            self.demo_console.phone_accepted()
        if self.cfg.runtime.status_stdout:
            print(json.dumps(ack, ensure_ascii=False))
        if self.mqtt_adapter is not None:
            self.mqtt_adapter.publish_ack(ack)

    def _publish_task_ack(
        self,
        payload: Dict[str, Any],
        *,
        accepted: bool,
        message: str,
        error_code: Optional[int] = None,
    ) -> None:
        ack = {
            "type": "mobile_ack",
            "kind": "task_ack",
            "robot_id": ROBOT_ID,
            "cmd_id": str(payload.get("cmd_id") or ""),
            "session_id": str(payload.get("session_id") or ""),
            "epoch": int(payload.get("epoch", 0) or 0),
            "message": message,
            "accepted": bool(accepted),
            "error_code": error_code,
            "source": "mobile_gateway",
            "ts": now_ts(),
        }
        if self._is_debug_mode() and str(payload.get("state") or "").strip():
            ack["backend_state"] = str(payload.get("state") or "").strip()
        ack = {k: v for k, v in ack.items() if v not in (None, "", [], {})}
        self.run_logger.write_jsonl("mobile_ack", ack)
        if self.cfg.runtime.status_stdout:
            print(json.dumps(ack, ensure_ascii=False))
        if self.mqtt_adapter is not None:
            self.mqtt_adapter.publish_ack(ack)

    def _in_stop_cooldown(self) -> bool:
        if self._last_stop_ts <= 0:
            return False
        return (now_ts() - self._last_stop_ts) < float(self.cfg.backend.stop_cooldown_s)

    def _is_duplicate_cmd(self, cmd_id: str) -> bool:
        cmd_id = str(cmd_id or "").strip()
        return bool(cmd_id) and cmd_id in self._recent_cmd_id_set

    def _remember_cmd_id(self, cmd_id: str) -> None:
        cmd_id = str(cmd_id or "").strip()
        if not cmd_id or cmd_id in self._recent_cmd_id_set:
            return
        if len(self._recent_cmd_ids) >= self._recent_cmd_ids.maxlen:
            expired = self._recent_cmd_ids.popleft()
            self._recent_cmd_id_set.discard(expired)
        self._recent_cmd_ids.append(cmd_id)
        self._recent_cmd_id_set.add(cmd_id)

    def _status_error_code(self, accepted: bool, error_info: Optional[Dict[str, Any]]) -> int:
        if error_info is not None:
            return int(error_info["error_code"])
        return ERROR_CODES["task_rejected"] if not accepted else ERROR_CODES["backend_unavailable"]

    def _backend_error_info(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raw_tokens: List[str] = []
        for key in ("reason", "message", "error", "last_fail_reason", "last_enter_reason", "link_state", "channel"):
            value = payload.get(key)
            if value not in (None, ""):
                raw_tokens.append(str(value))
        serialized = " | ".join(raw_tokens)
        lowered = serialized.lower()
        if (
            "vision_req_out connect_failed" in lowered
            or "connection refused" in lowered
            or "link_state=degraded" in lowered
            or str(payload.get("link_state") or "").strip().upper() == "DEGRADED"
        ):
            return {
                "error_code": ERROR_CODES["backend_unavailable"],
                "message": "视觉模块未连接，任务暂时无法继续",
                "raw_error": serialized or "vision_req_out connect_failed",
            }
        return None

    def _recoverable_backend_warning(self, payload: Dict[str, Any]) -> str:
        tokens: List[str] = []
        for key in ("last_enter_reason", "last_fail_reason", "reason", "message"):
            value = payload.get(key)
            if value not in (None, ""):
                tokens.append(str(value))
        edge_quality = payload.get("edge_quality")
        if isinstance(edge_quality, dict):
            for key in ("reason", "stop_reason"):
                value = edge_quality.get(key)
                if value not in (None, ""):
                    tokens.append(str(value))
        lowered = " | ".join(tokens).lower()
        if "edge_distance_out_of_tolerance_after_retries" in lowered:
            return ""
        recoverable = {
            "edge_distance_out_of_tolerance": "桌边距离偏离，正在重新调整",
            "edge_identity_mismatch": "桌边识别不稳定，正在重新调整",
            "edge_follow_stale": "桌边观测延迟，正在重新调整",
            "conf_low": "识别置信度偏低，继续观察",
            "edge_conf_low": "桌边置信度偏低，继续观察",
            "target_lost": "目标暂时丢失，继续搜索",
        }
        for token, message in recoverable.items():
            if token in lowered:
                return message
        return ""

    def _task_ack_message(
        self,
        payload: Dict[str, Any],
        *,
        target: Optional[str],
        backend_state: str,
        error_info: Optional[Dict[str, Any]],
    ) -> str:
        if error_info is not None:
            return str(error_info["message"])
        reason = str(payload.get("reason") or "").strip()
        if reason:
            if reason.upper() == "FIND ACCEPTED" and target:
                return f"FIND accepted: {target}"
            return reason
        if bool(payload.get("accepted", False)):
            if target:
                return f"FIND accepted: {target}"
            return f"{backend_state or 'task'} accepted"
        return "task rejected"

    def _status_message_for_state(
        self,
        raw_state: str,
        *,
        target: Optional[str],
        fallback: str,
        error_info: Optional[Dict[str, Any]],
    ) -> str:
        if error_info is not None:
            return str(error_info["message"])
        warning_message = self._recoverable_backend_warning({"state": raw_state, "last_enter_reason": fallback})
        if warning_message:
            return warning_message
        if raw_state == "SEARCH_TABLE":
            return "正在寻找桌边"
        if raw_state == "COARSE_ALIGN":
            return "已发现桌边，正在调整方向"
        if raw_state == "CONTROLLED_APPROACH":
            return "正在靠近桌边"
        if raw_state == "FINAL_LOCK":
            return "正在锁定桌边位置"
        if raw_state in {"EDGE_FOLLOW_OBJECT_SEARCH", "SEARCH_OBJECT", "SEARCH_TARGET_INIT", "EDGE_SLIDE_SEARCH", "TARGET_CONFIRM"}:
            return f"正在搜索目标 {target}" if target else "正在搜索目标"
        if raw_state == "DETECTOR_UNAVAILABLE":
            return "目标检测暂不可用"
        if raw_state == "DOCK_RETRY":
            return "锁边未完成，正在重试"
        if raw_state == "ERROR_RECOVERY":
            if "edge_distance_out_of_tolerance_after_retries" in str(fallback or "").lower():
                return "任务失败，无法稳定锁定桌边。"
            return fallback or "任务失败"
        if raw_state in {"TARGET_LOCKED", "GRASP", "CONTROLLED_APPROACH", "FINAL_LOCK", "AT_TABLE_EDGE"}:
            return f"正在执行取物任务，目标 {target}" if target else "正在执行取物任务"
        if raw_state in {"STOP", "STOPPED"}:
            return "任务已停止"
        if raw_state == "DONE":
            return f"任务完成，目标 {target}" if target else "任务完成，回到空闲"
        if raw_state == "IDLE":
            return "当前空闲"
        return fallback or raw_state

    def _log_status_event(self, payload: Dict[str, Any], force: bool = False) -> None:
        state = str(payload.get("state") or "").strip()
        if not state:
            return
        log_data = {
            "state": state,
            "session_id": payload.get("session_id"),
            "epoch": payload.get("epoch"),
            "message": payload.get("message"),
        }
        if payload.get("target") not in (None, ""):
            log_data["target"] = payload.get("target")
        self.log_info("status", "status changed", log_data)
        if state == "error":
            self.log_error("status", "error summary", {
                "session_id": payload.get("session_id"),
                "error_code": payload.get("error_code"),
                "message": payload.get("message"),
            })

    def _emit_mobile_status_console(self, payload: Dict[str, Any]) -> None:
        if self._mobile_status_console_mode == "silent":
            return
        state = str(payload.get("state") or "").strip()
        backend = str(payload.get("backend_state") or payload.get("raw_backend_state") or "").strip()
        progress = payload.get("progress")
        target = str(payload.get("target") or "").strip()
        reason = str(payload.get("lock_reason") or payload.get("reason") or "").strip()
        key = safe_dump({
            "state": state,
            "backend": backend,
            "progress": progress,
            "target": target,
            "reason": reason,
        })
        line = f"[ORCH] MOBILE state={state} backend={backend} target={target} progress={progress}"
        if reason:
            line += f" reason={reason}"
        if self._mobile_status_console_mode == "full":
            self.operator_console.emit(line)
            return
        if key == self._last_operator_status_key:
            return
        self._last_operator_status_key = key
        self.operator_console.emit_change("mobile_status", line)

    def _log_heartbeat_summary(self, now: float) -> None:
        interval = max(1.0, float(self.cfg.runtime.heartbeat_log_interval_s or 30.0))
        if (now - self._last_heartbeat_log_ts) < interval:
            return
        self._last_heartbeat_log_ts = now
        self.log_info("runtime", "heartbeat running", {
            "count": self._heartbeat_count,
            "last_state": self._snapshot.get("state", "idle"),
            "backend_mode": self.backend_mode,
        })


def run_mobile_gateway_service(cfg: MobileGatewayConfig) -> None:
    service = MobileGatewayService(cfg)

    def _handle_sig(signum, frame) -> None:
        service.log_info("runtime", f"signal {signum} received; stopping gateway")
        service._running = False

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)
    service.run_forever()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SC171 mobile gateway.")
    parser.add_argument(
        "--config",
        default="",
        help="Optional YAML/JSON config file path, for example configs/mobile_gateway.mqtt.yaml",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    cfg = build_config(config_file=args.config)
    run_mobile_gateway_service(cfg)


if __name__ == "__main__":
    main()
