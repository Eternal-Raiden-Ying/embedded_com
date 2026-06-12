#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import queue
import socket
import threading
import time

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .protocol import pack_msg, unpack_msg

Logger = Optional[Callable[[Dict[str, Any]], None]]


class JsonlClientSender:
    def __init__(self, mode: str = "disabled", tcp_host: str = "127.0.0.1", tcp_port: int = 0,
                 uds_path: str = "", reconnect_interval: float = 1.0, send_timeout: float = 1.0,
                 name: str = "jsonl_sender", logger: Logger = None, queue_size: int = 5,
                 latest_only: bool = False):
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
        self._last_connect_ms = 0.0
        self.link_state = "DISCONNECTED"
        self.fail_count = 0
        self.last_send_ok_ts = 0.0
        self.last_send_fail_ts = 0.0
        self.last_enqueue_ts = 0.0
        self.latest_only = bool(latest_only)
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self.obs_replace_count = 0
        self.queue_drop_oldest_count = 0
        self.queue_drop_new_count = 0
        self._last_queue_delay_ms = None
        self._last_send_total_ms = None
        self._last_summary_ts = 0.0

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
        start = time.perf_counter()
        if self.mode == "disabled":
            raise RuntimeError("disabled mode does not create socket")
        if self.mode == "tcp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.send_timeout)
            sock.connect((self.tcp_host, self.tcp_port))
            self._last_connect_ms = max(0.0, (time.perf_counter() - start) * 1000.0)
            return sock
        if os.name == "nt":
            raise RuntimeError("uds transport is not supported on Windows")
        if not hasattr(socket, "AF_UNIX"):
            raise RuntimeError("uds transport requested but socket.AF_UNIX is unavailable")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.send_timeout)
        path_str = self.uds_path
        if not path_str and self.tcp_port:
            path_str = f"/tmp/robot_ipc_{self.tcp_port}.sock"
        if not path_str:
            raise ValueError("uds transport requires uds_path or tcp_port-derived path")
        sock.connect(path_str)
        self._last_connect_ms = max(0.0, (time.perf_counter() - start) * 1000.0)
        return sock

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
            self._log(
                "info",
                "connected",
                transport=self.mode,
                recovered=self.fail_count > 0,
                connect_ms=float(getattr(self, "_last_connect_ms", 0.0) or 0.0),
            )
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
        if self.latest_only:
            replaced = 0
            while True:
                try:
                    self._queue.get_nowait()
                    replaced += 1
                except queue.Empty:
                    break
            if replaced:
                self.obs_replace_count += replaced
                self.queue_drop_oldest_count += replaced
                self._emit_summary_if_needed()
        elif self._queue.full():
            try:
                self._queue.get_nowait()
                self.queue_drop_oldest_count += 1
                self._log("warn", "queue_drop_oldest", queue_depth=self._queue.qsize())
            except queue.Empty:
                pass
        try:
            self.last_enqueue_ts = time.time()
            payload["_ipc_enqueue_ts"] = self.last_enqueue_ts
            self._queue.put_nowait(payload)
            self._log("info", "enqueue_ok", queue_depth=self._queue.qsize())
            return True
        except queue.Full:
            self.queue_drop_new_count += 1
            self._log("warn", "queue_drop_new", queue_depth=self._queue.qsize())
            return False

    def _emit_summary_if_needed(self, force: bool = False):
        now = time.time()
        if not force and (now - float(self._last_summary_ts or 0.0)) < 1.0:
            return
        self._last_summary_ts = now
        self._log(
            "info",
            "sender_summary",
            queue_depth=self._queue.qsize(),
            queue_size=self._queue.maxsize,
            latest_only=bool(self.latest_only),
            obs_replace_count=int(self.obs_replace_count),
            queue_drop_oldest_count=int(self.queue_drop_oldest_count),
            queue_drop_new_count=int(self.queue_drop_new_count),
            queue_delay_ms=self._last_queue_delay_ms,
            send_total_ms=self._last_send_total_ms,
        )

    def _send_loop(self):
        while not self._stop_event.is_set():
            try:
                payload = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            send_total_start = time.perf_counter()
            enqueue_ts = 0.0
            if isinstance(payload, dict):
                try:
                    enqueue_ts = float(payload.pop("_ipc_enqueue_ts", 0.0) or 0.0)
                except Exception:
                    enqueue_ts = 0.0
            encode_start = time.perf_counter()
            packed = pack_msg(payload)
            data = len(packed).to_bytes(4, byteorder="big") + packed
            encode_ms = max(0.0, (time.perf_counter() - encode_start) * 1000.0)
            queue_delay_ms = max(0.0, (time.time() - enqueue_ts) * 1000.0) if enqueue_ts > 0.0 else None
            self._last_queue_delay_ms = queue_delay_ms
            self._log(
                "info",
                "send_attempt",
                size=len(packed),
                bytes=len(data),
                transport=self.mode,
                json_encode_ms=encode_ms,
                queue_delay_ms=queue_delay_ms,
            )

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
                        write_start = time.perf_counter()
                        self._sock.sendall(data)
                        send_ms = max(0.0, (time.perf_counter() - write_start) * 1000.0)
                        total_ms = max(0.0, (time.perf_counter() - send_total_start) * 1000.0)
                        self._last_send_total_ms = total_ms
                        self.fail_count = 0
                        self.last_send_ok_ts = time.time()
                        self.link_state = "CONNECTED"
                        self._log(
                            "info",
                            "send_ok",
                            bytes=len(data),
                            send_ms=send_ms,
                            send_total_ms=total_ms,
                            queue_delay_ms=queue_delay_ms,
                            json_encode_ms=encode_ms,
                            connect_ms=float(getattr(self, "_last_connect_ms", 0.0) or 0.0),
                        )
                        self._emit_summary_if_needed()
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
            "latest_only": bool(self.latest_only),
            "obs_replace_count": int(self.obs_replace_count),
            "queue_drop_oldest_count": int(self.queue_drop_oldest_count),
            "queue_drop_new_count": int(self.queue_drop_new_count),
            "queue_delay_ms": self._last_queue_delay_ms,
            "send_total_ms": self._last_send_total_ms,
        }

    def close(self):
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=1.0)
        self._emit_summary_if_needed(force=True)
        with self._lock:
            self._close()


class JsonlInboundServer:
    def __init__(self, mode: str = "tcp", tcp_host: str = "127.0.0.1", tcp_port: int = 0,
                 uds_path: str = "", name: str = "jsonl_server", logger: Logger = None):
        if mode not in {"disabled", "tcp", "uds"}:
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

    def _fix_uds_socket_permissions(self, path: str) -> None:
        if os.name == "nt":
            return
        try:
            import pwd
            
            sudo_uid = os.environ.get("SUDO_UID")
            sudo_gid = os.environ.get("SUDO_GID")
            
            target_uid = None
            target_gid = None
            
            if sudo_uid and sudo_gid:
                target_uid = int(sudo_uid)
                target_gid = int(sudo_gid)
            else:
                try:
                    user_info = pwd.getpwnam("aidlux")
                    target_uid = user_info.pw_uid
                    target_gid = user_info.pw_gid
                except KeyError:
                    pass
            
            if target_uid is not None and target_gid is not None:
                os.chown(path, target_uid, target_gid)
                os.chmod(path, 0o660)
            else:
                os.chmod(path, 0o666)
                self._log("warn", "ownership_not_resolved_fallback_666", path=path)
        except Exception as exc:
            self._log("warn", "failed_to_fix_uds_socket_permission", path=path, error=repr(exc))
            try:
                os.chmod(path, 0o666)
            except Exception:
                pass

    def _bind_socket(self) -> socket.socket:
        if self.mode == "disabled":
            raise RuntimeError("disabled mode does not create socket")
        if self.mode == "tcp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.tcp_host, self.tcp_port))
            self.tcp_host, self.tcp_port = sock.getsockname()[:2]
            sock.listen(4)
            sock.settimeout(1.0)
            return sock
        if os.name == "nt":
            raise RuntimeError("uds transport is not supported on Windows")
        if not hasattr(socket, "AF_UNIX"):
            raise RuntimeError("uds transport requested but socket.AF_UNIX is unavailable")
        path_str = self.uds_path
        if not path_str and self.tcp_port:
            path_str = f"/tmp/robot_ipc_{self.tcp_port}.sock"
        if not path_str:
            raise ValueError("uds transport requires uds_path or tcp_port-derived path")
        path = Path(path_str)
        parent_existed = path.parent.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not parent_existed and os.name != "nt":
            try:
                os.chmod(str(path.parent), 0o1777)
            except Exception:
                try:
                    os.chmod(str(path.parent), 0o777)
                except Exception:
                    pass
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(path))
        self._fix_uds_socket_permissions(str(path))
        sock.listen(4)
        sock.settimeout(1.0)
        return sock

    def start(self):
        if self._accept_thread is not None:
            return
        if self.mode == "disabled":
            self.listening = False
            self._log("info", "disabled", transport=self.mode)
            return
        self._server_sock = self._bind_socket()
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True, name=f"{self.name}_accept")
        self._accept_thread.start()
        self.listening = True
        desc = self.uds_path if self.mode == "uds" else f"{self.tcp_host}:{self.tcp_port}"
        if self.mode == "uds":
            owner = "unknown"
            perm = "unknown"
            try:
                st = os.stat(self.uds_path)
                owner = f"{st.st_uid}:{st.st_gid}"
                perm = oct(st.st_mode & 0o777)
                print(f"[IPC] server listening mode=uds path={self.uds_path} owner={owner} perm={perm}", flush=True)
            except Exception:
                pass
            self._log("info", "listening", transport=self.mode, bind=desc, owner=owner, perm=perm)
        else:
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
        path_str = ""
        if self.mode == "uds":
            path_str = self.uds_path
            if not path_str and self.tcp_port:
                path_str = f"/tmp/robot_ipc_{self.tcp_port}.sock"
        if path_str:
            path = Path(path_str)
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
                    recv_start = time.perf_counter()
                    chunk = conn.recv(4096)
                    recv_block_ms = max(0.0, (time.perf_counter() - recv_start) * 1000.0)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                chunk_recv_ts = time.time()
                buffer += chunk
                while len(buffer) >= 4:
                    msg_len = int.from_bytes(buffer[:4], byteorder="big")
                    if len(buffer) >= 4 + msg_len:
                        raw = buffer[4:4+msg_len]
                        buffer = buffer[4+msg_len:]
                        parse_start = time.perf_counter()
                        try:
                            payload = unpack_msg(raw)
                        except Exception as exc:
                            self._log("warn", "invalid_msgpack", peer=peer, error=str(exc), raw=raw[:200])
                            continue
                        self.last_recv_ts = time.time()
                        parse_ms = max(0.0, (time.perf_counter() - parse_start) * 1000.0)
                        recv_to_queue_ms = max(0.0, (self.last_recv_ts - chunk_recv_ts) * 1000.0)
                        self._log(
                            "info",
                            "recv_ok",
                            peer=peer,
                            msg_type=payload.get("type"),
                            bytes=len(raw),
                            recv_block_ms=recv_block_ms,
                            json_parse_ms=parse_ms,
                            recv_to_queue_ms=recv_to_queue_ms,
                        )
                        self._queue.put({"peer": peer, "payload": payload, "recv_ts": self.last_recv_ts})
                    else:
                        break
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
