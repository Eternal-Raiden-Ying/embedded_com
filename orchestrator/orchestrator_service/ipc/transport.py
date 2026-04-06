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
ResultCallback = Optional[Callable[[Dict[str, Any]], None]]


class JsonlClientSender:
    def __init__(self, mode: str = "disabled", tcp_host: str = "127.0.0.1", tcp_port: int = 0,
                 uds_path: str = "", reconnect_interval: float = 1.0, send_timeout: float = 1.0,
                 name: str = "jsonl_sender", logger: Logger = None, send_mode: str = "persistent"):
        self.mode = mode
        self.tcp_host = tcp_host
        self.tcp_port = int(tcp_port)
        self.uds_path = uds_path
        self.reconnect_interval = max(0.05, float(reconnect_interval))
        self.send_timeout = max(0.05, float(send_timeout))
        self.name = name
        self.logger = logger
        self.send_mode = str(send_mode or "persistent").strip().lower()
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._last_warn_ts = 0.0
        self.link_state = "DISCONNECTED"
        self.fail_count = 0
        self.last_send_ok_ts = 0.0
        self.last_send_fail_ts = 0.0
        if self.mode not in {"disabled", "tcp", "uds"}:
            raise ValueError(f"unsupported sender mode: {mode}")
        if self.send_mode not in {"persistent", "oneshot"}:
            raise ValueError(f"unsupported send_mode: {self.send_mode}")

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
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
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
            prev_state = self.link_state
            self.link_state = "CONNECTED"
            self._log("info", "connected", transport=self.mode, recovered=self.fail_count > 0, prev_state=prev_state)
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
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            self._log("info", "send_attempt", size=len(line), transport=self.mode)
            for _ in range(2):
                if not self._ensure_connected():
                    time.sleep(self.reconnect_interval)
                    continue
                try:
                    assert self._sock is not None
                    self._sock.sendall(line.encode("utf-8", errors="ignore"))
                    self.fail_count = 0
                    self.last_send_ok_ts = time.time()
                    self.link_state = "CONNECTED"
                    self._log("info", "send_ok")
                    if self.send_mode == "oneshot":
                        self._close()
                    return True
                except OSError as exc:
                    self.fail_count += 1
                    self.last_send_fail_ts = time.time()
                    self.link_state = "DEGRADED"
                    self._log("warn", "send_failed", error=str(exc), fail_count=self.fail_count)
                    self._close()
                    time.sleep(self.reconnect_interval)
            self.link_state = "ACK_TIMEOUT" if self.fail_count > 0 else "DEGRADED"
        return False

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "link_state": self.link_state,
            "fail_count": self.fail_count,
            "last_send_ok_ts": self.last_send_ok_ts,
            "last_send_fail_ts": self.last_send_fail_ts,
        }

    def close(self):
        with self._lock:
            self._close()


class AsyncJsonlClientSender:
    def __init__(self, inner: JsonlClientSender, queue_size: int = 64, drop_oldest: bool = True,
                 logger: Logger = None, result_callback: ResultCallback = None):
        self.inner = inner
        self.name = inner.name
        self.logger = logger
        self.result_callback = result_callback
        self.queue_size = max(1, int(queue_size))
        self.drop_oldest = bool(drop_oldest)
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=self.queue_size)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._seq = 0
        self.enqueued_count = 0
        self.sent_count = 0
        self.send_fail_count = 0
        self.dropped_count = 0
        self.last_enqueue_ts = 0.0
        self.last_dequeue_ts = 0.0

    def _log(self, level: str, event: str, **kwargs):
        if self.logger is not None:
            payload = {"level": level, "src": "ipc", "name": self.name, "event": event}
            payload.update(kwargs)
            self.logger(payload)

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name=f"{self.name}_async")
        self._thread.start()
        self._log("info", "async_started", queue_size=self.queue_size, drop_oldest=self.drop_oldest)

    def send(self, payload: Dict[str, Any]) -> bool:
        if self.inner.mode == "disabled":
            return False
        item = {
            "seq": self._seq,
            "payload": payload,
            "enqueue_ts": time.time(),
        }
        self._seq += 1
        try:
            self._queue.put_nowait(item)
            self.enqueued_count += 1
            self.last_enqueue_ts = item["enqueue_ts"]
            self._log("info", "async_enqueue", seq=item["seq"], queue_depth=self._queue.qsize())
            return True
        except queue.Full:
            if not self.drop_oldest:
                self.dropped_count += 1
                self._log("warn", "async_queue_full_drop_new", queue_depth=self._queue.qsize())
                return False
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
            self.dropped_count += 1
            try:
                self._queue.put_nowait(item)
                self.enqueued_count += 1
                self.last_enqueue_ts = item["enqueue_ts"]
                self._log("warn", "async_queue_full_drop_oldest", seq=item["seq"], queue_depth=self._queue.qsize())
                return True
            except queue.Full:
                self.dropped_count += 1
                self._log("warn", "async_queue_full_retry_failed", queue_depth=self._queue.qsize())
                return False

    def _emit_result(self, result: Dict[str, Any]):
        if self.result_callback is not None:
            try:
                self.result_callback(result)
            except Exception:
                pass

    def _worker_loop(self):
        while not self._stop.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self.last_dequeue_ts = time.time()
            payload = item["payload"]
            ok = self.inner.send(payload)
            snap = self.inner.snapshot()
            if ok:
                self.sent_count += 1
            else:
                self.send_fail_count += 1
            result = {
                "name": self.name,
                "ok": bool(ok),
                "payload": payload,
                "seq": item["seq"],
                "enqueue_ts": item["enqueue_ts"],
                "dequeue_ts": self.last_dequeue_ts,
                "done_ts": time.time(),
                "snapshot": snap,
            }
            self._emit_result(result)

    def snapshot(self) -> Dict[str, Any]:
        inner_snap = self.inner.snapshot()
        inner_snap.update({
            "async": True,
            "queue_depth": self._queue.qsize(),
            "queue_size": self.queue_size,
            "drop_oldest": self.drop_oldest,
            "enqueued_count": self.enqueued_count,
            "sent_count": self.sent_count,
            "send_fail_count": self.send_fail_count,
            "dropped_count": self.dropped_count,
            "last_enqueue_ts": self.last_enqueue_ts,
            "last_dequeue_ts": self.last_dequeue_ts,
        })
        return inner_snap

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.inner.close()


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
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.last_recv_ts = 0.0
        self.invalid_json_count = 0
        self.listening = False
        self.start_ts = 0.0
        self.total_recv_count = 0
        self.client_count = 0

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
        self.start_ts = time.time()
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

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "listening": self.listening,
            "start_ts": self.start_ts,
            "last_recv_ts": self.last_recv_ts,
            "invalid_json_count": self.invalid_json_count,
            "queue_depth": self._queue.qsize(),
            "total_recv_count": self.total_recv_count,
            "client_count": self.client_count,
        }

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
            self.client_count += 1
            self._log("info", "peer_connected", peer=peer, client_count=self.client_count)
            th = threading.Thread(target=self._client_loop, args=(conn, peer), daemon=True, name=f"{self.name}_client")
            self._client_threads.append(th)
            th.start()

    def _client_loop(self, conn: socket.socket, peer: str):
        with conn:
            conn.settimeout(1.0)
            file_obj = conn.makefile("r", encoding="utf-8", newline="\n")
            while not self._stop.is_set():
                try:
                    line = file_obj.readline()
                except socket.timeout:
                    continue
                except Exception:
                    break
                if not line:
                    self._log("warn", "peer_closed", peer=peer)
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    self.invalid_json_count += 1
                    self._log("warn", "invalid_json", peer=peer, error=str(exc), raw=line[:200], invalid_json_count=self.invalid_json_count)
                    continue
                self.last_recv_ts = time.time()
                self.total_recv_count += 1
                self._queue.put({"peer": peer, "payload": payload, "recv_ts": self.last_recv_ts})
