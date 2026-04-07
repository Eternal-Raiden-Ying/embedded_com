#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import queue
import socket
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

Logger = Optional[Callable[[Dict[str, Any]], None]]


class JsonlClientSender:
    def __init__(self, mode: str = "disabled", tcp_host: str = "127.0.0.1", tcp_port: int = 0,
                 uds_path: str = "", reconnect_interval: float = 1.0, send_timeout: float = 1.0,
                 name: str = "jsonl_sender", logger: Logger = None):
        self.mode = mode
        self.tcp_host = tcp_host
        self.tcp_port = int(tcp_port)
        self.uds_path = uds_path
        self.reconnect_interval = max(0.05, float(reconnect_interval))
        self.send_timeout = max(0.05, float(send_timeout))
        self.name = name
        self.logger = logger
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._last_warn_ts = 0.0
        self.link_state = "DISCONNECTED"
        self.fail_count = 0
        self.last_send_ok_ts = 0.0
        self.last_send_fail_ts = 0.0
        self.last_enqueue_ts = 0.0
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=5)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        if self.mode not in {"disabled", "tcp", "uds"}:
            raise ValueError(f"unsupported sender mode: {mode}")
        if self.mode != "disabled":
            self._worker_thread = threading.Thread(target=self._send_loop, daemon=True, name=f"{self.name}_worker")
            self._worker_thread.start()

    def _log(self, level: str, event: str, **kwargs):
        if self.logger is not None:
            payload = {"level": level, "src": "ipc", "name": self.name, "event": event}
            payload.update(kwargs)
            self.logger(payload)

    def _close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _make_socket(self) -> socket.socket:
        if self.mode == "tcp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.send_timeout)
            sock.connect((self.tcp_host, self.tcp_port))
            return sock
        if self.mode == "uds":
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.send_timeout)
            sock.connect(self.uds_path)
            return sock
        raise RuntimeError("disabled mode does not create socket")

    def _ensure_connected(self) -> bool:
        if self.mode == "disabled":
            self.link_state = "DISABLED"
            return False
        if self._sock is not None:
            return True
        try:
            self.link_state = "CONNECTING"
            self._sock = self._make_socket()
            self.link_state = "CONNECTED"
            self._log("info", "connected", transport=self.mode, recovered=self.fail_count > 0)
            return True
        except OSError as exc:
            now = time.time()
            self.link_state = "DISCONNECTED"
            self.last_send_fail_ts = now
            if now - self._last_warn_ts > 1.5:
                self._log("warn", "connect_failed", error=str(exc), transport=self.mode)
                self._last_warn_ts = now
            self._close()
            return False

    def send(self, payload: Dict[str, Any]) -> bool:
        if self.mode == "disabled":
            self.link_state = "DISABLED"
            return False
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._log("warn", "queue_drop_oldest", queue_depth=self._queue.qsize())
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(payload)
            self.last_enqueue_ts = time.time()
            self._log("info", "enqueue_ok", queue_depth=self._queue.qsize())
            return True
        except queue.Full:
            self._log("warn", "queue_drop_new", queue_depth=self._queue.qsize())
            return False

    def _send_loop(self):
        while not self._stop_event.is_set():
            try:
                payload = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            data = line.encode("utf-8", errors="ignore")
            self._log("info", "send_attempt", size=len(line), transport=self.mode)

            for _ in range(2):
                if self._stop_event.is_set():
                    break
                with self._lock:
                    connected = self._ensure_connected()
                if not connected:
                    time.sleep(self.reconnect_interval)
                    continue
                with self._lock:
                    try:
                        assert self._sock is not None
                        self._sock.sendall(data)
                        self.fail_count = 0
                        self.last_send_ok_ts = time.time()
                        self.link_state = "CONNECTED"
                        self._log("info", "send_ok")
                        break
                    except OSError as exc:
                        self.fail_count += 1
                        self.last_send_fail_ts = time.time()
                        self.link_state = "DEGRADED"
                        self._log("warn", "send_failed", error=str(exc), fail_count=self.fail_count)
                        self._close()
                time.sleep(self.reconnect_interval)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "link_state": self.link_state,
            "fail_count": self.fail_count,
            "last_send_ok_ts": self.last_send_ok_ts,
            "last_send_fail_ts": self.last_send_fail_ts,
            "last_enqueue_ts": self.last_enqueue_ts,
            "queue_depth": self._queue.qsize(),
            "queue_size": self._queue.maxsize,
        }

    def close(self):
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=1.0)
        with self._lock:
            self._close()


class JsonlInboundServer:
    def __init__(self, mode: str = "tcp", tcp_host: str = "127.0.0.1", tcp_port: int = 0,
                 uds_path: str = "", name: str = "jsonl_server", logger: Logger = None):
        if mode not in {"tcp", "uds"}:
            raise ValueError(f"unsupported server mode: {mode}")
        self.mode = mode
        self.tcp_host = tcp_host
        self.tcp_port = int(tcp_port)
        self.uds_path = uds_path
        self.name = name
        self.logger = logger
        self._stop = threading.Event()
        self._accept_thread: Optional[threading.Thread] = None
        self._server_sock: Optional[socket.socket] = None
        self._client_threads: List[threading.Thread] = []
        self._client_socks: List[socket.socket] = []
        self._client_lock = threading.Lock()
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.listening = False
        self.last_recv_ts = 0.0

    def _log(self, level: str, event: str, **kwargs):
        if self.logger is not None:
            payload = {"level": level, "src": "ipc", "name": self.name, "event": event}
            payload.update(kwargs)
            self.logger(payload)

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
        if self._accept_thread is not None:
            return
        self._server_sock = self._bind_socket()
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True, name=f"{self.name}_accept")
        self._accept_thread.start()
        self.listening = True
        desc = self.uds_path if self.mode == "uds" else f"{self.tcp_host}:{self.tcp_port}"
        self._log("info", "listening", transport=self.mode, bind=desc)

    def close(self):
        self._stop.set()
        self.listening = False
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except Exception:
                pass
        with self._client_lock:
            for conn in list(self._client_socks):
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
            self._client_socks.clear()
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=1.5)
        for th in self._client_threads:
            th.join(timeout=0.5)
        if self.mode == "uds" and self.uds_path:
            path = Path(self.uds_path)
            if path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass

    def drain(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return items

    def _accept_loop(self):
        assert self._server_sock is not None
        while not self._stop.is_set():
            try:
                conn, addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            peer = str(addr)
            with self._client_lock:
                self._client_socks.append(conn)
            th = threading.Thread(target=self._client_loop, args=(conn, peer), daemon=True, name=f"{self.name}_client")
            self._client_threads.append(th)
            th.start()

    def _client_loop(self, conn: socket.socket, peer: str):
        buffer = b""
        try:
            conn.settimeout(1.0)
            while not self._stop.is_set():
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    raw, buffer = buffer.split(b"\n", 1)
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:
                        self._log("warn", "invalid_json", peer=peer, error=str(exc), raw=line[:200])
                        continue
                    self.last_recv_ts = time.time()
                    self._log("info", "recv_ok", peer=peer, msg_type=payload.get("type"))
                    self._queue.put({"peer": peer, "payload": payload, "recv_ts": self.last_recv_ts})
        finally:
            with self._client_lock:
                try:
                    self._client_socks.remove(conn)
                except ValueError:
                    pass
            try:
                conn.close()
            except Exception:
                pass

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "listening": self.listening,
            "queue_depth": self._queue.qsize(),
            "last_recv_ts": self.last_recv_ts,
            "clients": len(self._client_socks),
        }
