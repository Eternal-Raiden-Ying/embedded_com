#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Lightweight operator console reporter."""

from __future__ import annotations

import os
import time
import sys
from typing import Any, Callable, Dict, Optional

try:
    from common.runtime_logging import colorize_operator_line, should_use_console_color
except Exception:  # pragma: no cover
    def colorize_operator_line(line: str, enabled: bool = True) -> str:
        return str(line or "")

    def should_use_console_color(color_mode=None, stream=None, *, sink_provided: bool = False) -> bool:
        return False


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
        color_mode: Optional[str] = None,
        stream: Optional[Any] = None,
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
        self.color_mode = str(color_mode or os.getenv("ROBOT_CONSOLE_COLOR", "auto") or "auto").strip().lower()
        if self.color_mode not in {"auto", "always", "never"}:
            self.color_mode = "auto"
        self._color_enabled = should_use_console_color(
            self.color_mode,
            stream=stream if stream is not None else (None if sink is not None else sys.stdout),
            sink_provided=sink is not None,
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
        self._sink(colorize_operator_line(text, self._color_enabled))
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
