#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import socket
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

Logger = Optional[Callable[[Dict[str, Any]], None]]


class JsonlClientSender:
    def __init__(self, mode: str = "tcp", tcp_host: str = "127.0.0.1", tcp_port: int = 9001,
                 uds_path: str = "/tmp/robot_stack/task_cmd.sock", reconnect_interval: float = 1.0,
                 send_timeout: float = 1.0, name: str = "jsonl_sender", logger: Logger = None, send_mode: str = "persistent"):
        self.mode = mode
        self.tcp_host = tcp_host
        self.tcp_port = int(tcp_port)
        self.uds_path = uds_path
        self.reconnect_interval = max(0.05, float(reconnect_interval))
        self.send_timeout = max(0.05, float(send_timeout))
        self.name = name
        self.logger = logger
        self.send_mode = str(send_mode or "persistent").strip().lower()
        self._lock = threading.Lock()
        self._sock: Optional[socket.socket] = None
        self._last_warn_ts = 0.0
        self._stdout_mirror = "+stdout" in mode
        self._transport = mode.replace("+stdout", "")
        self.link_state = "DISCONNECTED"
        self.fail_count = 0
        self.last_send_ok_ts = 0.0
        self.last_send_fail_ts = 0.0
        if self._transport not in {"stdout", "tcp", "uds"}:
            raise ValueError(f"unsupported sender mode: {mode}")
        if self.send_mode not in {"persistent", "oneshot"}:
            raise ValueError(f"unsupported send_mode: {self.send_mode}")

    def _log(self, payload: Dict[str, Any]):
        if self.logger is not None:
            self.logger(payload)

    def _emit(self, level: str, event: str, **kwargs):
        payload = {"level": level, "src": "ipc", "name": self.name, "event": event}
        payload.update(kwargs)
        self._log(payload)

    def _maybe_warn(self, event: str, err: Optional[Exception] = None):
        now = time.time()
        if now - self._last_warn_ts < 1.5:
            return
        payload = {"level": "warn", "src": "ipc", "name": self.name, "event": event}
        if err is not None:
            payload["error"] = str(err)
        self._log(payload)
        self._last_warn_ts = now

    def _make_socket(self) -> socket.socket:
        if self._transport == "tcp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(self.send_timeout)
            sock.connect((self.tcp_host, self.tcp_port))
            return sock
        if self._transport == "uds":
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.send_timeout)
            sock.connect(self.uds_path)
            return sock
        raise RuntimeError("stdout mode does not use socket")

    def _ensure_connected(self) -> bool:
        if self._transport == "stdout":
            self.link_state = "CONNECTED"
            return True
        if self._sock is not None:
            return True
        try:
            self.link_state = "CONNECTING"
            self._sock = self._make_socket()
            self.link_state = "CONNECTED"
            self._emit("info", "connected", transport=self._transport, recovered=self.fail_count > 0)
            return True
        except OSError as exc:
            self.link_state = "DISCONNECTED"
            self.last_send_fail_ts = time.time()
            self._maybe_warn("connect_failed", exc)
            self._sock = None
            return False

    def _close_socket(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def send(self, payload: Dict[str, Any]) -> bool:
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        if self._transport == "stdout" or self._stdout_mirror:
            print(line, end="", flush=True)
        if self._transport == "stdout":
            self.last_send_ok_ts = time.time()
            return True
        with self._lock:
            self._emit("info", "send_attempt", size=len(line), transport=self._transport)
            for attempt in range(2):
                if not self._ensure_connected():
                    time.sleep(self.reconnect_interval)
                    continue
                try:
                    assert self._sock is not None
                    self._sock.sendall(line.encode("utf-8", errors="ignore"))
                    self.last_send_ok_ts = time.time()
                    self.fail_count = 0
                    self.link_state = "CONNECTED"
                    self._emit("info", "send_ok", attempt=attempt + 1)
                    if self.send_mode == "oneshot":
                        self._close_socket()
                    return True
                except OSError as exc:
                    self.fail_count += 1
                    self.last_send_fail_ts = time.time()
                    self.link_state = "DEGRADED"
                    self._emit("warn", "send_fail", attempt=attempt + 1, error=str(exc), fail_count=self.fail_count)
                    self._close_socket()
                    time.sleep(self.reconnect_interval)
        self.link_state = "ACK_TIMEOUT" if self.fail_count > 0 else "DEGRADED"
        return False

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "transport": self._transport,
            "link_state": self.link_state,
            "fail_count": self.fail_count,
            "last_send_ok_ts": self.last_send_ok_ts,
            "last_send_fail_ts": self.last_send_fail_ts,
        }

    def close(self):
        with self._lock:
            self._close_socket()


class JsonlInboundListener:
    def __init__(self, mode: str = "tcp", tcp_host: str = "127.0.0.1", tcp_port: int = 9011,
                 uds_path: str = "/tmp/robot_stack/tts_event.sock", on_message: Optional[Callable[[Dict[str, Any]], None]] = None,
                 name: str = "jsonl_listener", logger: Logger = None):
        if mode not in {"tcp", "uds"}:
            raise ValueError(f"unsupported listener mode: {mode}")
        self.mode = mode
        self.tcp_host = tcp_host
        self.tcp_port = int(tcp_port)
        self.uds_path = uds_path
        self.on_message = on_message
        self.name = name
        self.logger = logger
        self._stop = threading.Event()
        self._server_thread: Optional[threading.Thread] = None
        self._server_sock: Optional[socket.socket] = None
        self._client_threads = []

    def _log(self, payload: Dict[str, Any]):
        if self.logger is not None:
            self.logger(payload)

    def _emit(self, level: str, event: str, **kwargs):
        payload = {"level": level, "src": "ipc", "name": self.name, "event": event}
        payload.update(kwargs)
        self._log(payload)

    def _bind_socket(self) -> socket.socket:
        if self.mode == "tcp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.tcp_host, self.tcp_port))
        else:
            path = Path(self.uds_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                path.unlink()
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(str(path))
        sock.listen(4)
        sock.settimeout(1.0)
        return sock

    def start(self):
        if self._server_thread is not None:
            return
        self._server_sock = self._bind_socket()
        self._server_thread = threading.Thread(target=self._serve, daemon=True, name=f"{self.name}_accept")
        self._server_thread.start()
        bind_desc = self.uds_path if self.mode == "uds" else f"{self.tcp_host}:{self.tcp_port}"
        self._emit("info", "listening", transport=self.mode, bind=bind_desc)

    def _serve(self):
        assert self._server_sock is not None
        while not self._stop.is_set():
            try:
                conn, addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self._emit("info", "peer_connected", peer=str(addr))
            t = threading.Thread(target=self._handle_client, args=(conn,), daemon=True)
            self._client_threads.append(t)
            t.start()

    def _handle_client(self, conn: socket.socket):
        with conn:
            conn.settimeout(1.0)
            buf = ""
            while not self._stop.is_set():
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    self._emit("warn", "peer_closed")
                    break
                buf += chunk.decode("utf-8", errors="ignore")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                        if self.on_message is not None:
                            self.on_message(payload)
                    except Exception as exc:
                        self._emit("warn", "invalid_inbound_json", error=str(exc), line=line[:200])

    def close(self):
        self._stop.set()
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except Exception:
                pass
            self._server_sock = None
        if self.mode == "uds":
            try:
                path = Path(self.uds_path)
                if path.exists():
                    path.unlink()
            except Exception:
                pass
        if self._server_thread is not None:
            self._server_thread.join(timeout=1.5)
            self._server_thread = None


class JsonlAckInbox:
    def __init__(self, logger: Logger = None):
        self.logger = logger
        self._cv = threading.Condition()
        self._acks: Dict[str, Dict[str, Any]] = {}

    def _log(self, payload: Dict[str, Any]):
        if self.logger is not None:
            self.logger(payload)

    def handle_message(self, payload: Dict[str, Any]):
        cmd_id = str(payload.get("cmd_id", "")).strip()
        if not cmd_id:
            self._log({"level": "warn", "src": "ipc", "name": "task_ack_in", "event": "ack_missing_cmd_id", "payload": payload})
            return
        with self._cv:
            self._acks[cmd_id] = dict(payload)
            self._cv.notify_all()
        self._log({"level": "info", "src": "ipc", "name": "task_ack_in", "event": "ack_received", "cmd_id": cmd_id, "accepted": bool(payload.get("accepted", False))})

    def wait_ack(self, cmd_id: str, timeout: float) -> Optional[Dict[str, Any]]:
        deadline = time.time() + max(0.0, float(timeout))
        with self._cv:
            while cmd_id not in self._acks:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._cv.wait(timeout=min(remaining, 0.10))
            return self._acks.pop(cmd_id, None)
