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
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=5)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        if self.mode not in {"disabled", "tcp", "uds"}:
            raise ValueError(f"unsupported sender mode: {mode}")
        if self.mode != "disabled":
            self._worker_thread = threading.Thread(target=self._send_loop, daemon=True, name=f"{self.name}_worker")
            self._worker_thread.start()

    def _log(self, level: str, msg: str, **kwargs):
        if self.logger is not None:
            payload = {"level": level, "src": "ipc", "msg": f"{self.name}: {msg}"}
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
            return False
        if self._sock is not None:
            return True
        try:
            self._sock = self._make_socket()
            self._log("info", "connected", transport=self.mode)
            return True
        except OSError as exc:
            now = time.time()
            if now - self._last_warn_ts > 1.5:
                self._log("warn", "connect failed", error=str(exc), transport=self.mode)
                self._last_warn_ts = now
            self._close()
            return False

    def send(self, payload: Dict[str, Any]) -> bool:
        if self.mode == "disabled":
            return False
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(payload)
            return True
        except queue.Full:
            return False

    def _send_loop(self):
        while not self._stop_event.is_set():
            try:
                payload = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            data = line.encode("utf-8", errors="ignore")

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
                        break
                    except OSError as exc:
                        self._log("warn", "send failed", error=str(exc))
                        self._close()
                time.sleep(self.reconnect_interval)

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

    def _log(self, level: str, msg: str, **kwargs):
        if self.logger is not None:
            payload = {"level": level, "src": "ipc", "msg": f"{self.name}: {msg}"}
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
        desc = self.uds_path if self.mode == "uds" else f"{self.tcp_host}:{self.tcp_port}"
        self._log("info", "listening", transport=self.mode, bind=desc)

    def close(self):
        self._stop.set()
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
                        self._log("warn", "invalid json", peer=peer, error=str(exc), raw=line[:200])
                        continue
                    self._queue.put({"peer": peer, "payload": payload, "recv_ts": time.time()})
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
