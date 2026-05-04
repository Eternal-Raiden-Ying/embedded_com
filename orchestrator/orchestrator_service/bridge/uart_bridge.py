#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from .simple_car_protocol import SimpleCarCommand

try:
    import serial  # type: ignore
except Exception:  # pragma: no cover
    serial = None


class UartBridge:
    def __init__(
        self,
        port: str,
        baudrate: int,
        timeout_s: float,
        dry_run: bool = False,
        readback_enabled: bool = True,
        dry_run_echo_stdout: bool = True,
        tx_callback: Optional[Callable[[str, bool, Optional[Dict[str, Any]]], None]] = None,
        logger: Optional[Callable] = None,
    ):
        self.port = port
        self.baudrate = int(baudrate)
        self.timeout_s = float(timeout_s)
        self.dry_run = bool(dry_run) or serial is None or os.environ.get("ENV", "").lower() == "mock"
        self.readback_enabled = bool(readback_enabled)
        self.dry_run_echo_stdout = bool(dry_run_echo_stdout)
        self.tx_callback = tx_callback
        self._logger = logger
        self._ser = None
        self._last_line: Optional[str] = None
        self._stop = threading.Event()
        self._rx_thread: Optional[threading.Thread] = None
        self._tx_thread: Optional[threading.Thread] = None
        self._tx_event = threading.Event()
        self._pending_lock = threading.Lock()
        self._pending_tx: Optional[Dict[str, Any]] = None
        self._rx_queue: "queue.Queue[str]" = queue.Queue()
        self.last_tx_ts = 0.0
        self.last_rx_ts = 0.0
        self.last_tx_error = ""
        self.published_count = 0
        self.sent_count = 0
        self.send_fail_count = 0
        self.replaced_pending_count = 0

    def _log(self, level: str, msg: str, *args):
        if self._logger:
            self._logger(level, "uart", msg, {"args": [str(a) for a in args]} if args else None)

    def start(self):
        self._stop.clear()
        if self.dry_run:
            self._log("warn", "UART running in dry-run mode; serial port will not be opened")
        else:
            self._ser = serial.Serial(
                self.port,
                self.baudrate,
                timeout=self.timeout_s,
                write_timeout=self.timeout_s,
            )
            self._log("info", f"UART opened: {self.port} @ {self.baudrate}")
        self._tx_thread = threading.Thread(target=self._writer_loop, daemon=True, name="uart_writer")
        self._tx_thread.start()
        self._log("info", "UART writer thread started with latest-command override")
        if self.readback_enabled and not self.dry_run:
            self._rx_thread = threading.Thread(target=self._reader_loop, daemon=True, name="uart_reader")
            self._rx_thread.start()
            self._log("info", "UART reader thread started")

    def close(self):
        self._stop.set()
        self._tx_event.set()
        if self._tx_thread is not None:
            self._tx_thread.join(timeout=1.0)
            self._tx_thread = None
        if self._rx_thread is not None:
            self._rx_thread.join(timeout=1.0)
            self._rx_thread = None
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def send_car_command(self, cmd: SimpleCarCommand, tx_meta: Optional[Dict[str, Any]] = None) -> bool:
        if not str(cmd.raw_line or "").strip():
            return False
        return self._publish_latest(cmd.raw_line, tx_meta=tx_meta)

    def send_stop(self, tx_meta: Optional[Dict[str, Any]] = None) -> bool:
        return self._publish_latest("MODE STOP\nSTOP\n", tx_meta=tx_meta)

    def send_arm_command(self, command_line: str, tx_meta: Optional[Dict[str, Any]] = None) -> bool:
        """Send an arm command directly, bypassing the latest-command-override.

        Arm POSE commands must not be dropped — each one is a discrete
        trajectory that the arm MCU must execute exactly once.
        """
        if not str(command_line or "").strip():
            return False
        with self._pending_lock:
            self._write_line(command_line, tx_meta=tx_meta)
        return True

    def drain_rx_lines(self) -> List[str]:
        items: List[str] = []
        while True:
            try:
                items.append(self._rx_queue.get_nowait())
            except queue.Empty:
                break
        return items

    def snapshot(self) -> Dict[str, Any]:
        return {
            "port": self.port,
            "baudrate": self.baudrate,
            "dry_run": self.dry_run,
            "serial_open": bool(self._ser is not None) if not self.dry_run else True,
            "tx_worker_alive": bool(self._tx_thread is not None and self._tx_thread.is_alive()),
            "rx_worker_alive": bool(self._rx_thread is not None and self._rx_thread.is_alive()),
            "pending_tx": bool(self._pending_tx is not None),
            "published_count": self.published_count,
            "sent_count": self.sent_count,
            "send_fail_count": self.send_fail_count,
            "replaced_pending_count": self.replaced_pending_count,
            "last_tx_ts": self.last_tx_ts,
            "last_rx_ts": self.last_rx_ts,
            "last_tx_error": self.last_tx_error,
            "link_state": self._link_state(),
        }

    def _link_state(self) -> str:
        if self.dry_run:
            return "DRY_RUN"
        if self._ser is None:
            return "CLOSED"
        if self.last_tx_error:
            return "ERROR"
        return "OPEN"

    def _publish_latest(self, line: str, tx_meta: Optional[Dict[str, Any]] = None) -> bool:
        if not line:
            return False
        item = {
            "line": str(line),
            "tx_meta": dict(tx_meta or {}),
            "publish_ts": time.time(),
        }
        with self._pending_lock:
            if self._pending_tx is not None:
                self.replaced_pending_count += 1
            self._pending_tx = item
            self.published_count += 1
        self._tx_event.set()
        return True

    def _reader_loop(self):
        while not self._stop.is_set():
            if self._ser is None:
                break
            try:
                raw = self._ser.readline()
            except Exception as exc:
                self._log("warn", f"UART read failed: {exc}")
                continue
            if not raw:
                continue
            try:
                line = raw.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue
            if line:
                self.last_rx_ts = time.time()
                self._rx_queue.put(line)

    def _writer_loop(self):
        while not self._stop.is_set() or self._has_pending_tx():
            self._tx_event.wait(timeout=0.2)
            item = self._pop_pending_tx()
            if item is None:
                self._tx_event.clear()
                continue
            self._write_line(item["line"], tx_meta=item.get("tx_meta"))
            if not self._has_pending_tx():
                self._tx_event.clear()

    def _has_pending_tx(self) -> bool:
        with self._pending_lock:
            return self._pending_tx is not None

    def _pop_pending_tx(self) -> Optional[Dict[str, Any]]:
        with self._pending_lock:
            item = self._pending_tx
            self._pending_tx = None
            return item

    def _emit_tx_callback(self, line: str, dry_run: bool, tx_meta: Optional[Dict[str, Any]] = None):
        if self.tx_callback is not None:
            try:
                self.tx_callback(line, dry_run, tx_meta)
            except Exception:
                pass

    def _write_line(self, line: str, tx_meta: Optional[Dict[str, Any]] = None):
        if not line:
            return
        self._last_line = line.rstrip("\n")
        self.last_tx_ts = time.time()
        meta = dict(tx_meta or {})
        ok = False
        error = ""
        if self.dry_run:
            ok = True
        else:
            if self._ser is None:
                error = "UART not started"
            else:
                try:
                    self._ser.write(line.encode("utf-8"))
                    ok = True
                except Exception as exc:
                    error = str(exc)
                    self._log("warn", f"UART send failed: {exc}")
        if ok:
            self.sent_count += 1
            self.last_tx_error = ""
        else:
            self.send_fail_count += 1
            self.last_tx_error = error or "unknown error"
        meta["uart_tx_ok"] = ok
        if error:
            meta["uart_tx_error"] = error
        self._emit_tx_callback(line, self.dry_run, meta)
