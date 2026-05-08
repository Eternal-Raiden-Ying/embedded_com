#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from typing import Any, Dict, Iterable, Optional

from .runtime_logging import OperatorConsole, should_use_console_emoji


def console_level(default: str = "normal") -> str:
    value = str(os.getenv("ROBOT_CONSOLE_LEVEL", default) or default).strip().lower()
    return value if value in {"demo", "normal", "debug"} else default


class DemoConsolePresenter:
    """Operator-facing demo console presenter with phase de-dupe and throttling."""

    def __init__(
        self,
        console: OperatorConsole,
        *,
        dry_run: bool = False,
        emoji_enabled: Optional[bool] = None,
        health_interval_s: Optional[float] = None,
    ):
        self.console = console
        self.dry_run = bool(dry_run)
        self.level = console_level()
        self.emoji_enabled = should_use_console_emoji() if emoji_enabled is None else bool(emoji_enabled)
        self.health_interval_s = (
            self._env_float("ROBOT_CONSOLE_HEALTH_INTERVAL_S", 4.0)
            if health_interval_s is None
            else max(1.0, float(health_interval_s))
        )
        self._last_phase_by_key: Dict[str, str] = {}
        self._last_emit_ts_by_key: Dict[str, float] = {}

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)) or default)
        except (TypeError, ValueError):
            return float(default)

    def _allowed(self, kind: str) -> bool:
        kind = str(kind or "demo").strip().lower()
        if self.level == "debug":
            return True
        if self.level == "normal":
            return kind in {"demo", "health", "warn", "error", "banner"}
        return kind in {"demo", "warn", "error", "banner"}

    def emit(self, line: str, *, kind: str = "demo") -> bool:
        if not self._allowed(kind):
            return False
        return self.console.emit(line)

    def emit_change(self, key: str, line: str, *, kind: str = "demo") -> bool:
        if not self._allowed(kind):
            return False
        key = str(key or "demo").strip() or "demo"
        text = str(line or "").strip()
        if self._last_phase_by_key.get(key) == text:
            return False
        self._last_phase_by_key[key] = text
        return self.console.emit(text)

    def emit_rate_limited(self, key: str, line: str, interval_s: Optional[float] = None, *, kind: str = "demo") -> bool:
        if not self._allowed(kind):
            return False
        key = str(key or "demo").strip() or "demo"
        interval = self.health_interval_s if interval_s is None else max(0.0, float(interval_s))
        now = time.time()
        if (now - self._last_emit_ts_by_key.get(key, 0.0)) < interval:
            return False
        self._last_emit_ts_by_key[key] = now
        return self.console.emit(line)

    def emit_block(self, lines: Iterable[str], *, level: str = "success") -> bool:
        if not self._allowed("banner"):
            return False
        return self.console.emit_demo_block(lines, level=level)

    def dry_run_notice(self) -> None:
        if self.dry_run:
            self.emit("[DEMO][DRY_RUN] car commands are control intentions only; serial port is not opened.")

    def task_start(self, target: str) -> None:
        mode = "DRY_RUN" if self.dry_run else "REAL_SERIAL"
        self.emit_change("task_start", f"[DEMO][START] target={target or 'n/a'} mode={mode}")
        self.dry_run_notice()

    def phone_command(self, target: str) -> None:
        self.emit_change("phone_command", f"[DEMO][PHONE] command received target={target or 'n/a'}")

    def phone_accepted(self) -> None:
        self.emit_change("phone_accepted", "[DEMO][PHONE] gateway accepted")

    def table_phase(self, phase: str) -> None:
        messages = {
            "searching": "[DEMO][TABLE] searching edge",
            "aligning": "[DEMO][TABLE] edge found, aligning",
            "approaching": "[DEMO][TABLE] approaching edge",
            "final_locking": "[DEMO][TABLE] final locking",
            "locked": "[DEMO][TABLE] locked, ready for target search",
        }
        line = messages.get(str(phase or "").strip())
        if line:
            self.emit_change("table_phase", line)

    def target_phase(self, phase: str, target: str = "") -> None:
        target = target or "target"
        messages = {
            "searching": f"[DEMO][TARGET] searching {target} along edge",
            "candidate": "[DEMO][TARGET] candidate found, confirming",
            "locked": f"[DEMO][TARGET] locked {target}",
            "relocating": "[DEMO][TARGET] current edge timeout, relocating",
        }
        line = messages.get(str(phase or "").strip())
        if line:
            self.emit_change("target_phase", line)

    def recover(self, message: str) -> None:
        self.emit_change("recover", f"[DEMO][RECOVER] {message}")

    def warning(self, message: str) -> None:
        self.emit_rate_limited("warn", f"[DEMO][WARN] {message}", 3.0)

    def preview_unavailable(self) -> None:
        self.emit_change(
            "preview_unavailable",
            "[DEMO][PREVIEW] unavailable: no UI backend, continuing without preview",
        )

    def success_banner(self, *, target: str, next_state: str = "IDLE_HOT", preview: str = "kept alive") -> None:
        title = "✅  DEMO SUCCESS / TASK DONE" if self.emoji_enabled else "[DEMO][SUCCESS] TASK DONE"
        border = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        self.emit_block(
            [
                border,
                title,
                f"target        : {target or 'n/a'}",
                "result        : success",
                "final_state   : DONE",
                f"next_state    : {next_state}",
                f"preview       : {preview}",
                "waiting       : next command",
                border,
            ],
            level="success",
        )

    def failed_banner(
        self,
        *,
        target: str,
        reason: str,
        next_state: str = "IDLE_HOT",
        recoverable: bool = False,
    ) -> None:
        title = "❌  DEMO FAILED" if self.emoji_enabled else "[DEMO][FAILED] TASK FAILED"
        border = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        self.emit_block(
            [
                border,
                title,
                f"target        : {target or 'n/a'}",
                f"reason        : {reason or 'failed'}",
                "final_state   : FAILED",
                f"recoverable   : {str(bool(recoverable)).lower()}",
                f"next_state    : {next_state}",
                border,
            ],
            level="failed",
        )

    def idle_hot(self, *, next_state: str = "IDLE_HOT", preview: str = "kept alive") -> None:
        prefix = "🟦 " if self.emoji_enabled else ""
        self.emit_change(
            "idle_hot",
            f"{prefix}[DEMO][{next_state}] preview {preview}, waiting for next command",
        )

    def health(self, data: Dict[str, Any]) -> None:
        if not self._allowed("health"):
            return
        state = str(data.get("state") or "n/a")
        target = str(data.get("target") or "n/a")
        edge = str(data.get("edge") or "UNKNOWN")
        yolo = data.get("yolo")
        edge_hz = data.get("edge_hz")
        yolo_hz = data.get("yolo_hz")
        preview = data.get("preview")
        dry_run = int(bool(data.get("dry_run", self.dry_run)))
        parts = [f"[DEMO][HEALTH] state={state}", f"target={target}", f"edge={edge}"]
        if yolo is not None:
            parts.append(f"yolo={str(yolo or 'n/a')}")
        if edge_hz is not None:
            parts.append(f"edge_hz={self._fmt_rate(edge_hz)}")
        if yolo_hz is not None:
            parts.append(f"yolo_hz={self._fmt_rate(yolo_hz)}")
        parts.append(f"preview={self._fmt_preview(preview)}")
        parts.append(f"dry_run={dry_run}")
        interval = max(1.0, float(data.get("interval_s", self.health_interval_s) or self.health_interval_s))
        self.emit_rate_limited(f"health:{state}", " ".join(parts), interval, kind="health")

    def slide_intent(self, vx: Any, vy: Any, wz: Any) -> None:
        if not self.dry_run:
            return
        line = (
            f"[DEMO][SLIDE_INTENT] vx={self._fmt_signed(vx)} "
            f"vy={self._fmt_signed(vy)} wz={self._fmt_signed(wz)} dry_run=1"
        )
        self.emit_rate_limited("slide_intent", line, self.health_interval_s, kind="health")

    @staticmethod
    def _fmt_signed(value: Any) -> str:
        try:
            return f"{float(value):+.2f}"
        except (TypeError, ValueError):
            return "+0.00"

    @staticmethod
    def _fmt_rate(value: Any) -> str:
        try:
            return f"{float(value):.1f}"
        except (TypeError, ValueError):
            return "n/a"

    @staticmethod
    def _fmt_preview(value: Any) -> str:
        text = str(value or "").strip()
        if text and text.lower() in {"unavailable", "off", "n/a"}:
            return text
        try:
            return f"{float(value):.1f}FPS"
        except (TypeError, ValueError):
            return text if text else "n/a"


__all__ = ["DemoConsolePresenter", "console_level"]
