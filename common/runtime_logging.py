#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional


TEXT_LOG_FORMAT = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"

EVENT_FIELD_ORDER = (
    "ts",
    "level",
    "module",
    "stack_run_id",
    "event",
    "stage",
    "mode",
    "trigger",
    "session_id",
    "req_id",
    "epoch",
    "interaction_id",
    "data",
)

IPC_FIELD_ORDER = (
    "ts",
    "level",
    "module",
    "stack_run_id",
    "direction",
    "channel",
    "event",
    "msg_type",
    "session_id",
    "req_id",
    "epoch",
    "ok",
    "peer",
    "error",
    "data",
)

HEARTBEAT_FIELD_ORDER = (
    "ts",
    "module",
    "stack_run_id",
    "stage",
    "mode",
    "session_id",
    "req_id",
    "epoch",
    "last_req_age_s",
    "last_obs_send_age_s",
    "ready",
    "data",
)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def monotonic_ts() -> float:
    return time.monotonic()


def safe_dump(obj: Optional[Dict[str, Any]]) -> str:
    if obj is None:
        return "{}"
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _is_empty_field(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _copy_mapping(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(value or {})


def logging_level_for_mode(mode: str) -> int:
    return logging.DEBUG if str(mode).strip().lower() == "full" else logging.INFO


def env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def operator_console_mode(default: str = "operator", env_name: str = "VISION_CONSOLE_MODE") -> str:
    mode = str(os.getenv(env_name, default)).strip().lower()
    if mode == "concise":
        mode = "operator"
    if mode not in {"operator", "full", "silent"}:
        mode = default
    return mode


def operator_summary_interval_s(default: float = 1.0, env_name: str = "VISION_OPERATOR_SUMMARY_INTERVAL_S") -> float:
    try:
        return max(0.0, float(os.getenv(env_name, str(default)) or default))
    except (TypeError, ValueError):
        return float(default)


ANSI_RESET = "\033[0m"
ANSI_BOLD_CYAN = "\033[1;36m"
ANSI_BLUE = "\033[34m"
ANSI_GREEN = "\033[32m"
ANSI_BRIGHT_GREEN = "\033[92m"
ANSI_YELLOW = "\033[33m"
ANSI_CYAN = "\033[36m"
ANSI_MAGENTA = "\033[35m"
ANSI_RED = "\033[31m"
ANSI_GRAY = "\033[90m"


def _color_mode(default: str = "auto") -> str:
    value = str(os.getenv("ROBOT_CONSOLE_COLOR", default) or default).strip().lower()
    return value if value in {"auto", "always", "never"} else default


def _env_truthy(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def should_use_console_color(
    color_mode: Optional[str] = None,
    stream: Optional[Any] = None,
    *,
    sink_provided: bool = False,
) -> bool:
    """Resolve ANSI color policy for operator console output only."""
    if _env_truthy("NO_COLOR"):
        return False
    if _env_truthy("FORCE_COLOR"):
        return True
    mode = _color_mode() if color_mode is None else str(color_mode or "auto").strip().lower()
    if mode == "never":
        return False
    if mode == "always":
        return True
    if sink_provided and stream is None:
        return False
    stream = stream if stream is not None else sys.stdout
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _ansi(text: str, color: str) -> str:
    return f"{color}{text}{ANSI_RESET}"


def colorize_operator_line(line: str, enabled: bool = True) -> str:
    """Apply small ANSI highlights to operator-facing status lines."""
    text = str(line or "")
    if not enabled or not text:
        return text

    replacements = (
        ("[VISTA]", _ansi("[VISTA]", ANSI_BLUE)),
        ("[ORCH]", _ansi("[ORCH]", ANSI_GREEN)),
        ("[GATEWAY]", _ansi("[GATEWAY]", ANSI_YELLOW)),
        ("[phone-gateway]", _ansi("[phone-gateway]", ANSI_GRAY)),
        (" FATAL ", f" {_ansi('FATAL', ANSI_RED)} "),
        (" ERROR ", f" {_ansi('ERROR', ANSI_RED)} "),
        (" WARN ", f" {_ansi('WARN', ANSI_YELLOW)} "),
        (" STATE ", f" {_ansi('STATE', ANSI_BOLD_CYAN)} "),
        (" MODE ", f" {_ansi('MODE', ANSI_BLUE)} "),
        (" CTRL ", f" {_ansi('CTRL', ANSI_GREEN)} "),
        (" CAR ", f" {_ansi('CAR', ANSI_MAGENTA)} "),
        (" EDGE ", f" {_ansi('EDGE', ANSI_CYAN)} "),
        (" TARGET ", f" {_ansi('TARGET', ANSI_YELLOW)} "),
    )
    padded = f" {text} "
    for needle, repl in replacements:
        padded = padded.replace(needle, repl)
    return padded[1:-1]


class OperatorConsole:
    """Small console-only reporter for operator-facing status lines."""

    def __init__(
        self,
        mode: Optional[str] = None,
        default_interval_s: Optional[float] = None,
        sink: Optional[Callable[[str], None]] = None,
        color_mode: Optional[str] = None,
        stream: Optional[Any] = None,
    ):
        self.mode = operator_console_mode() if mode is None else str(mode or "operator").strip().lower()
        if self.mode == "concise":
            self.mode = "operator"
        self.default_interval_s = (
            operator_summary_interval_s()
            if default_interval_s is None
            else max(0.0, float(default_interval_s))
        )
        self.color_mode = _color_mode() if color_mode is None else str(color_mode or "auto").strip().lower()
        if self.color_mode not in {"auto", "always", "never"}:
            self.color_mode = "auto"
        self._color_enabled = should_use_console_color(
            self.color_mode,
            stream=stream,
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


def _close_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def _resolve_log_file_for_logger(name: str, log_file: Optional[str] = None) -> str:
    explicit = str(log_file or "").strip()
    if explicit:
        return explicit
    prefix = str(name or "").split(".", 1)[0].strip().upper()
    if not prefix:
        return ""
    env_log_file = str(os.getenv(f"{prefix}_LOG_FILE", "")).strip()
    if env_log_file:
        return env_log_file
    env_log_dir = str(os.getenv(f"{prefix}_LOG_DIR", "")).strip()
    if env_log_dir:
        return str(Path(env_log_dir) / f"{prefix.lower()}.log")
    return ""


def make_stack_run_id(prefix: str = "run") -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:6]}"


@dataclass
class RuntimePaths:
    module_name: str
    project_root: str
    log_dir: str
    log_file: str
    runs_dir: str
    pid_dir: str
    pid_file: str
    stack_run_id: str

    @property
    def run_dir(self) -> str:
        return str(Path(self.runs_dir) / self.stack_run_id)


def resolve_runtime_paths(
    module_name: str,
    project_root: str,
    env_prefix: str,
    default_log_stem: str,
    default_pid_name: str,
    runs_dir: Optional[str] = None,
    stack_run_id: str = "",
) -> RuntimePaths:
    env_prefix = str(env_prefix or "").strip().upper()
    root = Path(project_root)
    log_dir = os.getenv(f"{env_prefix}_LOG_DIR", str(root / "logs")).strip()
    runs_root = os.getenv(f"{env_prefix}_RUNS_DIR", runs_dir or str(root / "runs")).strip()
    pid_dir = os.getenv(f"{env_prefix}_PID_DIR", str(root / "pids")).strip()
    resolved_stack_run_id = (
        str(stack_run_id).strip()
        or os.getenv("STACK_RUN_ID", "").strip()
        or make_stack_run_id()
    )
    log_file = os.getenv(f"{env_prefix}_LOG_FILE", str(Path(log_dir) / f"{default_log_stem}.log")).strip()
    pid_file = os.getenv(f"{env_prefix}_PID_FILE", str(Path(pid_dir) / default_pid_name)).strip()
    return RuntimePaths(
        module_name=module_name,
        project_root=str(root),
        log_dir=log_dir,
        log_file=log_file,
        runs_dir=runs_root,
        pid_dir=pid_dir,
        pid_file=pid_file,
        stack_run_id=resolved_stack_run_id,
    )


def configure_stream_logger(
    name: str,
    mode: str = "concise",
    enabled: bool = True,
    log_file: Optional[str] = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    level = logging_level_for_mode(mode)
    logger.setLevel(level)
    logger.propagate = False

    if not enabled:
        _close_handlers(logger)
        logger.disabled = True
        return logger

    logger.disabled = False
    formatter = logging.Formatter(TEXT_LOG_FORMAT)
    stream_handler = None
    file_handler = None
    resolved_log_file = _resolve_log_file_for_logger(name, log_file=log_file)

    for handler in list(logger.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            stream_handler = handler
        elif isinstance(handler, logging.FileHandler):
            existing_path = ""
            try:
                existing_path = str(Path(handler.baseFilename))
            except Exception:
                existing_path = ""
            target_path = str(Path(resolved_log_file)) if resolved_log_file else ""
            if target_path and existing_path == target_path:
                file_handler = handler
                continue
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    logger_prefix = str(name or "").split(".", 1)[0].strip().upper()
    console_default = "operator" if logger_prefix == "ORCH" else ""
    console_mode = str(os.getenv(f"{logger_prefix}_CONSOLE_MODE", console_default)).strip().lower()
    if console_mode == "concise":
        console_mode = "operator"
    stream_level = level
    if console_mode in {"operator", "silent"}:
        stream_level = logging.WARNING

    if stream_handler is None:
        stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(stream_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if resolved_log_file:
        ensure_dir(str(Path(resolved_log_file).parent))
        if file_handler is None:
            file_handler = logging.FileHandler(resolved_log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


class RunLogger:
    def __init__(self, module_name: str, runs_root: str, stack_run_id: str = "", enable_text_events: bool = True):
        self.module_name = str(module_name).strip()
        self.stack_run_id = str(stack_run_id).strip() or make_stack_run_id()
        self.run_dir = Path(runs_root) / self.stack_run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._event_fp = None
        if enable_text_events:
            self._event_fp = open(self.run_dir / "events.log", "a", encoding="utf-8")

    def _with_common_fields(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(payload)
        out.setdefault("module", self.module_name)
        out.setdefault("stack_run_id", self.stack_run_id)
        out.setdefault("ts", time.time())
        return out

    def _path_for(self, name: str) -> Path:
        filename = str(name).strip()
        if not filename:
            raise ValueError("log name must not be empty")
        return self.run_dir / filename

    def _write_json_line(self, path: Path, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with open(path, "a", encoding="utf-8") as fp:
            fp.write(line + "\n")

    def _ordered_payload(
        self,
        payload: Dict[str, Any],
        field_order: Iterable[str],
        extras_into_data: bool = False,
    ) -> Dict[str, Any]:
        ordered: Dict[str, Any] = {}
        field_names = tuple(field_order)
        for key in field_names:
            if key not in payload:
                continue
            value = payload.get(key)
            if _is_empty_field(value):
                continue
            if key == "data":
                value = _copy_mapping(value if isinstance(value, dict) else {"value": value})
                if not value:
                    continue
            ordered[key] = value

        extras: Dict[str, Any] = {}
        for key, value in payload.items():
            if key in field_names or _is_empty_field(value):
                continue
            extras[key] = value

        if extras_into_data and extras:
            data = _copy_mapping(ordered.get("data") if isinstance(ordered.get("data"), dict) else {})
            data.update(extras)
            ordered["data"] = data
        elif extras:
            for key, value in extras.items():
                ordered[key] = value

        return ordered

    def structured_paths(self, heartbeat_enabled: bool = False) -> Dict[str, str]:
        paths = {
            "run_dir": str(self.run_dir),
            "meta": str(self._path_for("meta.json")),
            "event": str(self._path_for("event.jsonl")),
            "ipc": str(self._path_for("ipc.jsonl")),
        }
        if heartbeat_enabled:
            paths["heartbeat"] = str(self._path_for("heartbeat.jsonl"))
        return paths

    def write_meta(self, payload: Dict[str, Any]) -> None:
        meta_payload = self._with_common_fields(payload)
        with open(self._path_for("meta.json"), "w", encoding="utf-8") as fp:
            json.dump(meta_payload, fp, ensure_ascii=False, indent=2)
            fp.write("\n")

    def write_jsonl(self, name: str, payload: Dict[str, Any]) -> None:
        path = self._path_for(f"{name}.jsonl")
        self._write_json_line(path, self._with_common_fields(payload))

    def write_event_record(
        self,
        event: str,
        level: str = "info",
        stage: Optional[str] = None,
        mode: Optional[str] = None,
        trigger: Optional[str] = None,
        session_id: Optional[str] = None,
        req_id: Optional[str] = None,
        epoch: Optional[int] = None,
        interaction_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> None:
        payload = self._with_common_fields(
            {
                "level": str(level or "info").strip().lower(),
                "event": str(event or "EVENT").strip().upper(),
                "stage": stage,
                "mode": mode,
                "trigger": trigger,
                "session_id": session_id,
                "req_id": req_id,
                "epoch": epoch,
                "interaction_id": interaction_id,
                "data": _copy_mapping(data),
                **extra,
            }
        )
        ordered = self._ordered_payload(payload, EVENT_FIELD_ORDER, extras_into_data=True)
        self._write_json_line(self._path_for("event.jsonl"), ordered)

    def write_ipc_record(
        self,
        direction: str,
        channel: str,
        event: str,
        level: str = "info",
        msg_type: Optional[str] = None,
        session_id: Optional[str] = None,
        req_id: Optional[str] = None,
        epoch: Optional[int] = None,
        ok: Optional[bool] = None,
        peer: Optional[str] = None,
        error: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> None:
        payload = self._with_common_fields(
            {
                "level": str(level or "info").strip().lower(),
                "direction": str(direction or "").strip().upper(),
                "channel": channel,
                "event": event,
                "msg_type": msg_type,
                "session_id": session_id,
                "req_id": req_id,
                "epoch": epoch,
                "ok": ok,
                "peer": peer,
                "error": error,
                "data": _copy_mapping(data),
                **extra,
            }
        )
        ordered = self._ordered_payload(payload, IPC_FIELD_ORDER, extras_into_data=True)
        self._write_json_line(self._path_for("ipc.jsonl"), ordered)

    def write_heartbeat_record(
        self,
        stage: str,
        mode: str,
        session_id: Optional[str] = None,
        req_id: Optional[str] = None,
        epoch: Optional[int] = None,
        last_req_age_s: Optional[float] = None,
        last_obs_send_age_s: Optional[float] = None,
        ready: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = self._with_common_fields(
            {
                "stage": stage,
                "mode": mode,
                "session_id": session_id,
                "req_id": req_id,
                "epoch": epoch,
                "last_req_age_s": last_req_age_s,
                "last_obs_send_age_s": last_obs_send_age_s,
                "ready": _copy_mapping(ready),
                "data": _copy_mapping(data),
            }
        )
        ordered = self._ordered_payload(payload, HEARTBEAT_FIELD_ORDER, extras_into_data=True)
        self._write_json_line(self._path_for("heartbeat.jsonl"), ordered)

    def write_event(self, text: str) -> None:
        if self._event_fp is None:
            return
        self._event_fp.write(f"[{time.strftime('%H:%M:%S')}] {text}\n")
        self._event_fp.flush()

    def write_timeline(self, event: str, **fields: Any) -> None:
        payload = {"event": str(event)}
        payload.update(fields)
        self.write_jsonl("timeline", payload)

    def write_service_event(self, event: str, **fields: Any) -> None:
        event_name = str(event).strip().upper()
        self.write_timeline(event_name, **fields)
        details = ", ".join(f"{k}={v}" for k, v in fields.items() if v not in (None, "", [], {}))
        self.write_event(f"{event_name}{' ' + details if details else ''}".rstrip())

    def write_ipc(self, channel: str, event: str, direction: str = "", **fields: Any) -> None:
        resolved_direction = str(direction or "").strip().upper()
        if not resolved_direction:
            resolved_direction = "RX" if str(channel).endswith("_in") else "TX"
        payload = {
            "direction": resolved_direction,
            "channel": channel,
            "event": event,
        }
        payload.update(fields)
        self.write_jsonl("ipc", payload)
        text = fields.get("msg") or fields.get("error") or ""
        self.write_event(f"IPC {resolved_direction} {channel} {event} {text}".rstrip())

    def write_state_block(self, block: Dict[str, Any]) -> None:
        self.write_jsonl("state_blocks", block)
        one_line = (
            f"state={block.get('state')} target={block.get('active_target')} "
            f"session={block.get('session_id')} epoch={block.get('epoch')} "
            f"reason={block.get('last_enter_reason')} fail={block.get('last_fail_reason')}"
        )
        self.write_event(one_line)

    def close(self) -> None:
        try:
            self._event_fp.close()
        except Exception:
            pass
