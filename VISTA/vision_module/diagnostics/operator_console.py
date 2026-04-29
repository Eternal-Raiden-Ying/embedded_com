#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Lightweight operator console reporter."""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, Optional


def _console_mode(default: str = "operator") -> str:
    mode = str(os.getenv("VISION_CONSOLE_MODE", default)).strip().lower()
    if mode == "concise":
        mode = "operator"
    return mode if mode in {"operator", "full", "silent"} else default


def _summary_interval_s(default: float = 1.0) -> float:
    try:
        return max(0.0, float(os.getenv("VISION_OPERATOR_SUMMARY_INTERVAL_S", str(default)) or default))
    except (TypeError, ValueError):
        return float(default)


class OperatorConsole:
    """Console-only reporter with de-dup and rate-limit helpers."""

    def __init__(
        self,
        mode: Optional[str] = None,
        default_interval_s: Optional[float] = None,
        sink: Optional[Callable[[str], None]] = None,
    ):
        self.mode = _console_mode() if mode is None else str(mode or "operator").strip().lower()
        if self.mode == "concise":
            self.mode = "operator"
        if self.mode not in {"operator", "full", "silent"}:
            self.mode = "operator"
        self.default_interval_s = (
            _summary_interval_s()
            if default_interval_s is None
            else max(0.0, float(default_interval_s))
        )
        self._sink = sink or print
        self._last_by_key: Dict[str, str] = {}
        self._last_ts_by_key: Dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return self.mode != "silent"

    @property
    def full(self) -> bool:
        return self.mode == "full"

    def emit(self, line: str) -> bool:
        if not self.enabled:
            return False
        text = str(line or "").strip()
        if not text:
            return False
        self._sink(text)
        return True

    def emit_change(self, key: str, line: str) -> bool:
        key = str(key or "default").strip() or "default"
        text = str(line or "").strip()
        if self._last_by_key.get(key) == text:
            return False
        self._last_by_key[key] = text
        return self.emit(text)

    def emit_rate_limited(self, key: str, line: str, interval_s: Optional[float] = None) -> bool:
        key = str(key or "default").strip() or "default"
        interval = self.default_interval_s if interval_s is None else max(0.0, float(interval_s))
        now = time.time()
        if now - self._last_ts_by_key.get(key, 0.0) < interval:
            return False
        self._last_ts_by_key[key] = now
        return self.emit(line)

    def emit_error(self, key: str, line: str, interval_s: Optional[float] = None) -> bool:
        return self.emit_rate_limited(f"error:{key}", line, interval_s)


ConsoleReporter = OperatorConsole

__all__ = ["ConsoleReporter", "OperatorConsole"]
