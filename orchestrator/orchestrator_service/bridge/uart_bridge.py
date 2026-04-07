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
        self._rx_queue: "queue.Queue[str]" = queue.Queue()
        self.last_tx_ts = 0.0
        self.last_rx_ts = 0.0

    def _log(self, level: str, msg: str, *args):
        if self._logger:
            self._logger(level, "uart", msg, {"args": [str(a) for a in args]} if args else None)

    def start(self):
        if self.dry_run:
            self._log("warn", "UART 运行在 dry-run 模式，不会真正打开串口")
            return
        self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout_s)
        self._log("info", f"UART 已打开: {self.port} @ {self.baudrate}")
        if self.readback_enabled:
            self._stop.clear()
            self._rx_thread = threading.Thread(target=self._reader_loop, daemon=True, name="uart_reader")
            self._rx_thread.start()
            self._log("info", "UART 回传读取线程已启动")

    def close(self):
        self._stop.set()
        if self._rx_thread is not None:
            self._rx_thread.join(timeout=1.0)
            self._rx_thread = None
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def send_car_command(self, cmd: SimpleCarCommand, tx_meta: Optional[Dict[str, Any]] = None):
        if not str(cmd.raw_line or "").strip():
            return
        self._write_line(cmd.raw_line, tx_meta=tx_meta)

    def send_stop(self, tx_meta: Optional[Dict[str, Any]] = None):
        self._write_line("MODE STOP\nSTOP\n", tx_meta=tx_meta)

    def drain_rx_lines(self) -> List[str]:
        items: List[str] = []
        while True:
            try:
                items.append(self._rx_queue.get_nowait())
            except queue.Empty:
                break
        return items

    def _reader_loop(self):
        while not self._stop.is_set():
            if self._ser is None:
                break
            try:
                raw = self._ser.readline()
            except Exception as exc:
                self._log("warn", f"UART 读取失败: {exc}")
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
        self._emit_tx_callback(line, self.dry_run, tx_meta)
        if self.dry_run:
            return
        if self._ser is None:
            raise RuntimeError("UART 尚未启动")
        self._ser.write(line.encode("utf-8"))
        self._ser.flush()
