#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Set

# Attempt to load from orchestrator_service package
try:
    from orchestrator_service.ipc.transport import JsonlClientSender, JsonlInboundServer
    from orchestrator_service.ipc.protocol import pack_msg, unpack_msg
except ImportError:
    # Fallback to direct path import or custom thin wrapper if we are running unit tests
    JsonlClientSender = None
    JsonlInboundServer = None

class InboundPollerThread(threading.Thread):
    """Periodically drains a JsonlInboundServer and routes messages to a callback."""
    def __init__(self, server: Any, handler: Callable[[Dict[str, Any]], None], name: str = "poller"):
        super().__init__(daemon=True, name=name)
        self.server = server
        self.handler = handler
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                # JsonlInboundServer exposes drain() which returns List[Dict[str, Any]]
                # Each item contains: {"peer": peer, "payload": payload, "recv_ts": ts}
                items = self.server.drain()
                for item in items:
                    payload = item.get("payload")
                    if payload is not None:
                        self.handler(payload)
            except Exception:
                pass
            time.sleep(0.02)  # 20ms poll interval

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=1.0)


class JsonlAckInbox:
    """Manages pending cmd_ids and collects task acknowledgments."""
    def __init__(self, logger: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.logger = logger
        self._cv = threading.Condition()
        self._acks: Dict[str, Dict[str, Any]] = {}
        self._pending_cmd_ids: Set[str] = set()

    def register_pending(self, cmd_id: str) -> None:
        with self._cv:
            self._pending_cmd_ids.add(cmd_id)

    def handle_message(self, payload: Dict[str, Any]) -> None:
        cmd_id = str(payload.get("cmd_id", "")).strip()
        if not cmd_id:
            return
        with self._cv:
            if cmd_id not in self._pending_cmd_ids:
                # Rule: Unknown cmd_id values must not trigger speech, session transitions or command retries
                if self.logger:
                    self.logger({"level": "debug", "src": "ipc", "name": "task_ack_in", "event": "ack_ignored_unknown_cmd_id", "cmd_id": cmd_id})
                return
            self._acks[cmd_id] = dict(payload)
            self._pending_cmd_ids.discard(cmd_id)
            self._cv.notify_all()
        if self.logger:
            self.logger({"level": "info", "src": "ipc", "name": "task_ack_in", "event": "ack_received", "cmd_id": cmd_id, "accepted": bool(payload.get("accepted", False)), "reason": payload.get("reason", "")})

    def wait_ack(self, cmd_id: str, timeout: float) -> Optional[Dict[str, Any]]:
        deadline = time.time() + max(0.0, float(timeout))
        with self._cv:
            while cmd_id not in self._acks:
                remaining = deadline - time.time()
                if remaining <= 0:
                    self._pending_cmd_ids.discard(cmd_id)
                    return None
                self._cv.wait(timeout=min(remaining, 0.10))
            return self._acks.pop(cmd_id, None)


def build_msgpack_client_sender(mode: str, host: str, port: int, uds_path: str, name: str, logger: Any, send_mode: str) -> JsonlClientSender:
    """Wrapper that resolves backend and initializes client sender."""
    if JsonlClientSender is None:
        raise RuntimeError("orchestrator_service.ipc.transport.JsonlClientSender is not available.")
    return JsonlClientSender(
        mode=mode,
        tcp_host=host,
        tcp_port=port,
        uds_path=uds_path,
        name=name,
        logger=logger,
        send_mode=send_mode,
    )


def build_msgpack_inbound_server(mode: str, host: str, port: int, uds_path: str, name: str, logger: Any) -> JsonlInboundServer:
    """Wrapper that resolves backend and initializes inbound server."""
    if JsonlInboundServer is None:
        raise RuntimeError("orchestrator_service.ipc.transport.JsonlInboundServer is not available.")
    return JsonlInboundServer(
        mode=mode,
        tcp_host=host,
        tcp_port=port,
        uds_path=uds_path,
        name=name,
        logger=logger,
    )
