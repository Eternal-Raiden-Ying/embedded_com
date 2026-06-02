#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
import queue
import sys
import threading
import time
import uuid
from collections import Counter
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
ANSI_BOLD_GREEN = "\033[1;32m"
ANSI_BOLD_RED = "\033[1;31m"
ANSI_BOLD_BLUE = "\033[1;34m"


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
    mode = _color_mode() if color_mode is None else str(color_mode or "auto").strip().lower()
    if mode == "never":
        return False
    if sink_provided and stream is None:
        return False
    stream = stream if stream is not None else sys.stdout
    try:
        is_tty = bool(stream.isatty())
    except Exception:
        return False
    if not is_tty:
        return False
    if _env_truthy("FORCE_COLOR"):
        return True
    if mode == "always":
        return True
    return True


def should_use_console_emoji(
    stream: Optional[Any] = None,
    *,
    sink_provided: bool = False,
) -> bool:
    mode = str(os.getenv("ROBOT_CONSOLE_EMOJI", "auto") or "auto").strip().lower()
    if mode == "never":
        return False
    if sink_provided and stream is None:
        stream = sys.stdout
    stream = stream if stream is not None else sys.stdout
    encoding = str(getattr(stream, "encoding", "") or "").lower()
    can_encode = "utf" in encoding or mode == "always"
    if not can_encode:
        return False
    if mode == "always":
        return True
    term = str(os.getenv("TERM", "") or "").strip().lower()
    if term == "dumb":
        return False
    return True


def _ansi(text: str, color: str) -> str:
    return f"{color}{text}{ANSI_RESET}"


def colorize_operator_line(line: str, enabled: bool = True) -> str:
    """Apply small ANSI highlights to operator-facing status lines."""
    text = str(line or "")
    if not enabled or not text:
        return text

    if "[DEMO][SUCCESS]" in text or "DEMO SUCCESS" in text:
        return _ansi(text, ANSI_BOLD_GREEN)
    if "[DEMO][FAILED]" in text or "DEMO FAILED" in text:
        return _ansi(text, ANSI_BOLD_RED)
    if "[DEMO][IDLE" in text:
        return _ansi(text, ANSI_BOLD_BLUE)
    if "[DEMO][DRY_RUN]" in text:
        return _ansi(text, ANSI_YELLOW)
    if "[DEMO][PHONE]" in text or "[DEMO][START]" in text:
        return _ansi(text, ANSI_MAGENTA)
    if "[DEMO][HEALTH]" in text:
        return _ansi(text, ANSI_CYAN)
    if "[DEMO][PREVIEW]" in text or "[DEMO][WARN]" in text or "[DEMO][RECOVER]" in text:
        return _ansi(text, ANSI_YELLOW)
    if "[DEMO]" in text:
        return _ansi(text, ANSI_BOLD_CYAN)

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

    def emit_demo_block(self, lines: Iterable[str], level: str = "info") -> bool:
        if not self.enabled:
            return False
        color = {
            "success": ANSI_BOLD_GREEN,
            "failed": ANSI_BOLD_RED,
            "idle": ANSI_BOLD_BLUE,
            "warning": ANSI_YELLOW,
        }.get(str(level or "").strip().lower(), "")
        emitted = False
        for line in lines:
            text = str(line or "").rstrip()
            if not text:
                continue
            if self._color_enabled and color:
                self._sink(_ansi(text, color))
            else:
                self._sink(text)
            emitted = True
        return emitted

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
        self.run_dir = self._resolve_run_dir(runs_root)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.max_jsonl_bytes = self._env_int("ROBOT_JSONL_MAX_BYTES", 64 * 1024 * 1024)
        self.rotate_backups = self._env_int("ROBOT_JSONL_ROTATE_BACKUPS", 3)
        self._event_fp = None
        if enable_text_events:
            self._event_fp = open(self.run_dir / "events.log", "a", encoding="utf-8")
        self._async_enabled = env_flag("ASYNC_LOG_ENABLED", "1")
        self._async_queue_size = self._env_int("ASYNC_LOG_QUEUE_SIZE", 2048)
        self._async_flush_interval_s = self._env_float("ASYNC_LOG_FLUSH_INTERVAL_S", 1.0)
        self._async_drop_low_priority = env_flag("ASYNC_LOG_DROP_LOW_PRIORITY", "1")
        self._log_queue: Optional["queue.Queue[Dict[str, Any]]"] = None
        self._log_stop = threading.Event()
        self._log_thread: Optional[threading.Thread] = None
        self._log_writer_files: Dict[Path, Any] = {}
        self._log_writer_last_flush_ts = 0.0
        self._drop_lock = threading.Lock()
        self._drop_counts: Dict[str, int] = {}
        self._last_drop_summary_ts = 0.0
        self._summary_start_wall = time.time()
        self._summary_stats: Dict[str, Any] = {
            "table_edge_count": 0,
            "table_edge_found": 0,
            "table_edge_valid": 0,
            "vision_process_ms": [],
            "obs_total_age_ms": [],
            "camera_frame_interval_ms": [],
            "vision_process_interval_ms": [],
            "vision_publish_interval_ms": [],
            "obs_out_send_interval_ms": [],
            "orchestrator_recv_interval_ms": [],
            "state_machine_tick_interval_ms": [],
            "vision_publish_to_orch_recv_ms": [],
            "orch_recv_to_state_consume_ms": [],
            "same_obs_reuse_count": [],
            "table_edge_ts": [],
            "reject_reasons": Counter(),
            "detector_mode": Counter(),
            "depth_shape": Counter(),
            "calib_source": Counter(),
            "calib": {},
            "preview": {},
            "cpu": {
                "process_cpu_percent": [],
                "system_cpu_percent": [],
                "process_rss_mb": [],
                "system_mem_percent": [],
            },
            "ipc": Counter(),
        }
        if self._async_enabled:
            self._log_queue = queue.Queue(maxsize=max(1, int(self._async_queue_size or 2048)))
            self._log_thread = threading.Thread(target=self._log_writer_loop, daemon=True, name=f"{self.module_name}_log_writer")
            self._log_thread.start()

    def _env_int(self, name: str, default: int) -> int:
        try:
            return max(0, int(os.getenv(name, str(default)) or default))
        except (TypeError, ValueError):
            return int(default)

    def _env_float(self, name: str, default: float) -> float:
        try:
            return max(0.0, float(os.getenv(name, str(default)) or default))
        except (TypeError, ValueError):
            return float(default)

    def _module_dir_name(self) -> str:
        if self.module_name == "orch":
            return "orchestrator"
        return self.module_name or "module"

    def _resolve_run_dir(self, runs_root: str) -> Path:
        root = Path(runs_root)
        if env_flag("ROBOT_RUN_MODULE_SUBDIRS", "0"):
            return root / self.stack_run_id / self._module_dir_name()
        return root / self.stack_run_id

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
        self._rotate_jsonl_if_needed(path, len(line.encode("utf-8")) + 1)
        with open(path, "a", encoding="utf-8") as fp:
            fp.write(line + "\n")

    def _write_json_line_async_worker(self, path: Path, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        pending_bytes = len(line.encode("utf-8")) + 1
        fp = self._log_writer_files.get(path)
        if fp is not None:
            try:
                fp.flush()
            except Exception:
                pass
        if self.max_jsonl_bytes > 0:
            try:
                needs_rotate = path.exists() and path.stat().st_size + max(0, int(pending_bytes)) > self.max_jsonl_bytes
            except OSError:
                needs_rotate = False
            if needs_rotate and fp is not None:
                try:
                    fp.close()
                except Exception:
                    pass
                self._log_writer_files.pop(path, None)
                fp = None
        self._rotate_jsonl_if_needed(path, pending_bytes)
        try:
            if fp is not None and getattr(fp, "closed", False):
                fp = None
            if fp is None:
                fp = open(path, "a", encoding="utf-8")
                self._log_writer_files[path] = fp
            fp.write(line + "\n")
        except Exception:
            try:
                if fp is not None:
                    fp.close()
            except Exception:
                pass
            self._log_writer_files.pop(path, None)

    def _flush_log_writer_files(self) -> None:
        for fp in list(self._log_writer_files.values()):
            try:
                fp.flush()
            except Exception:
                pass
        self._log_writer_last_flush_ts = time.time()

    def _close_log_writer_files(self) -> None:
        for fp in list(self._log_writer_files.values()):
            try:
                fp.flush()
                fp.close()
            except Exception:
                pass
        self._log_writer_files.clear()

    def _log_priority_for(self, name: str, payload: Dict[str, Any]) -> str:
        log_name = str(name or "").strip()
        event = str(payload.get("event") or "").strip().lower()
        level = str(payload.get("level") or "").strip().lower()
        low_names = {
            "frame_timing",
            "edge_profile",
            "log_timing",
            "rate",
            "cmd_vel",
            "car_cmd",
            "edge_slide_search",
        }
        ipc_success = event in {"recv_ok", "send_attempt", "send_ok", "enqueue_ok", "async_enqueue", "received", "envelope_received", "ack_sent"}
        if log_name in low_names or (log_name == "ipc" and ipc_success and level not in {"warn", "warning", "error", "fatal"}):
            return "low"
        critical_events = {
            "service_starting",
            "service_ready",
            "service_stopping",
            "service_stopped",
            "state_transition",
            "task_cmd",
            "task_ack",
            "send_failed",
            "connect_failed",
            "queue_drop",
            "queue_drop_oldest",
            "queue_drop_new",
            "queue_drop_failed",
            "stale_obs",
            "emergency_stop",
            "arm_cmd",
            "grasp",
        }
        if level in {"warn", "warning", "error", "fatal"} or any(key in event for key in critical_events):
            return "high"
        return "normal"

    def _record_log_drop(self, name: str, priority: str) -> None:
        with self._drop_lock:
            key = f"{name}:{priority}"
            self._drop_counts[key] = int(self._drop_counts.get(key, 0)) + 1

    def _pop_log_drop_summary(self) -> Dict[str, int]:
        with self._drop_lock:
            counts = dict(self._drop_counts)
            self._drop_counts.clear()
            return counts

    def _emit_log_writer_summary_if_needed(self, force: bool = False) -> None:
        now = time.time()
        interval_s = min(5.0, max(1.0, float(self._async_flush_interval_s or 1.0)))
        if not force and (now - float(self._last_drop_summary_ts or 0.0)) < interval_s:
            return
        counts = self._pop_log_drop_summary()
        if not counts:
            return
        self._last_drop_summary_ts = now
        payload = self._with_common_fields(
            {
                "channel": "log_writer_summary",
                "priority": "normal",
                "dropped": counts,
                "queue_size": self._async_queue_size,
                "queue_depth": self._log_queue.qsize() if self._log_queue is not None else 0,
            }
        )
        self._write_json_line_async_worker(self._path_for("log_writer_summary.jsonl"), payload)

    def _enqueue_json_line(self, name: str, payload: Dict[str, Any]) -> None:
        if not self._async_enabled or self._log_queue is None:
            self._write_json_line(self._path_for(f"{name}.jsonl"), payload)
            return
        priority = self._log_priority_for(name, payload)
        record = dict(payload)
        record.setdefault("channel", name)
        record.setdefault("priority", priority)
        item = {"name": name, "path": self._path_for(f"{name}.jsonl"), "payload": record, "priority": priority}
        try:
            self._log_queue.put_nowait(item)
            return
        except queue.Full:
            if priority == "low" and self._async_drop_low_priority:
                self._record_log_drop(name, priority)
                return
            self._write_json_line(self._path_for(f"{name}.jsonl"), record)

    def _log_writer_loop(self) -> None:
        self._log_writer_last_flush_ts = time.time()
        while not self._log_stop.is_set() or (self._log_queue is not None and not self._log_queue.empty()):
            try:
                item = self._log_queue.get(timeout=0.1) if self._log_queue is not None else None
            except queue.Empty:
                item = None
            if item is not None:
                self._write_json_line_async_worker(item["path"], item["payload"])
            now = time.time()
            if (now - float(self._log_writer_last_flush_ts or 0.0)) >= max(0.1, float(self._async_flush_interval_s or 1.0)):
                self._emit_log_writer_summary_if_needed()
                self._flush_log_writer_files()
        self._emit_log_writer_summary_if_needed(force=True)
        self._flush_log_writer_files()
        self._close_log_writer_files()

    def _rotate_jsonl_if_needed(self, path: Path, pending_bytes: int) -> None:
        if self.max_jsonl_bytes <= 0:
            return
        try:
            current_size = path.stat().st_size
        except FileNotFoundError:
            return
        except OSError:
            return
        if current_size + max(0, int(pending_bytes)) <= self.max_jsonl_bytes:
            return
        backups = max(1, int(self.rotate_backups or 1))
        for index in range(backups, 0, -1):
            src = path.with_name(f"{path.name}.{index}")
            dst = path.with_name(f"{path.name}.{index + 1}")
            try:
                if index == backups and src.exists():
                    src.unlink()
                elif src.exists():
                    src.replace(dst)
            except OSError:
                return
        try:
            path.replace(path.with_name(f"{path.name}.1"))
        except OSError:
            return

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
        log_name = str(name).strip()
        record = self._with_common_fields(payload)
        self._observe_summary(log_name, record)
        self._enqueue_json_line(log_name, record)

    @staticmethod
    def _finite_float(value: Any) -> Optional[float]:
        try:
            out = float(value)
        except Exception:
            return None
        if out != out or out in (float("inf"), float("-inf")):
            return None
        return out

    @classmethod
    def _summary_percentiles(cls, values: Iterable[Any]) -> Dict[str, Optional[float]]:
        vals = sorted(v for v in (cls._finite_float(item) for item in values) if v is not None)
        if not vals:
            return {"avg": None, "p50": None, "p90": None, "p95": None, "max": None}
        avg = sum(vals) / float(len(vals))
        def pick(p: float) -> float:
            idx = min(len(vals) - 1, int(round(float(p) * float(len(vals) - 1))))
            return float(vals[idx])
        return {"avg": float(avg), "p50": pick(0.50), "p90": pick(0.90), "p95": pick(0.95), "max": float(vals[-1])}

    def _observe_table_edge_timing_summary(self, payload: Dict[str, Any]) -> None:
        stats = self._summary_stats
        for key in (
            "vision_process_ms",
            "obs_total_age_ms",
            "camera_frame_interval_ms",
            "vision_process_interval_ms",
            "vision_publish_interval_ms",
            "obs_out_send_interval_ms",
            "table_edge_obs_recv_interval_ms",
            "orchestrator_recv_interval_ms",
            "state_machine_tick_interval_ms",
            "vision_publish_to_orch_recv_ms",
            "orch_recv_to_state_consume_ms",
            "same_obs_reuse_count",
        ):
            value = self._finite_float(payload.get(key))
            if value is not None:
                stat_key = "orchestrator_recv_interval_ms" if key == "table_edge_obs_recv_interval_ms" else key
                if stat_key in stats:
                    stats[stat_key].append(value)

    def _observe_table_edge_summary(self, payload: Dict[str, Any]) -> None:
        stats = self._summary_stats
        stats["table_edge_count"] += 1
        if bool(payload.get("edge_found")):
            stats["table_edge_found"] += 1
        if bool(payload.get("valid_for_control") or payload.get("edge_valid")):
            stats["table_edge_valid"] += 1
        self._observe_table_edge_timing_summary(payload)
        ts = self._finite_float(payload.get("obs_ts", payload.get("ts")))
        if ts is not None:
            stats["table_edge_ts"].append(ts)
        reason = str(payload.get("reject_reason") or payload.get("reason") or "none").strip() or "none"
        stats["reject_reasons"][reason] += 1
        mode = str(payload.get("detector_mode") or "unknown").strip() or "unknown"
        stats["detector_mode"][mode] += 1
        depth_shape = payload.get("depth_shape")
        if isinstance(depth_shape, (list, tuple)) and len(depth_shape) >= 2:
            stats["depth_shape"][f"{int(depth_shape[1])}x{int(depth_shape[0])}"] += 1
        source = str(payload.get("calib_source") or "unknown").strip() or "unknown"
        stats["calib_source"][source] += 1
        if payload.get("fx") is not None:
            stats["calib"] = {
                "depth_shape": payload.get("depth_shape"),
                "calib_source": payload.get("calib_source"),
                "fx": payload.get("fx"),
                "fy": payload.get("fy"),
                "cx": payload.get("cx"),
                "cy": payload.get("cy"),
                "depth_scale": payload.get("depth_scale"),
            }

    def _observe_summary(self, name: str, payload: Dict[str, Any]) -> None:
        stats = self._summary_stats
        if name in {"table_edge_obs", "table_edge_obs_lite"}:
            self._observe_table_edge_summary(payload)
        elif name == "control_summary":
            self._observe_table_edge_timing_summary(payload)
        elif name == "vision_obs":
            perception = payload.get("perception") if isinstance(payload.get("perception"), dict) else {}
            edge = perception.get("table_edge_obs") if isinstance(perception, dict) else None
            if isinstance(edge, dict):
                self._observe_table_edge_summary(edge)
        elif name == "rate":
            preview = stats["preview"]
            for key in ("preview_fps", "preview_total_ms_avg", "preview_total_ms_p95", "preview_total_ms_max"):
                if payload.get(key) is not None:
                    preview[key] = payload.get(key)
        elif name == "preview_timing":
            stats["preview"] = dict(payload)
        elif name == "system_metrics":
            cpu = stats["cpu"]
            for key in tuple(cpu.keys()):
                value = self._finite_float(payload.get(key))
                if value is not None:
                    cpu[key].append(value)
        elif name == "ipc":
            channel = str(payload.get("channel") or "unknown")
            event = str(payload.get("event") or "event")
            stats["ipc"][f"{channel}:{event}"] += 1

    def write_run_summary(self) -> None:
        stats = self._summary_stats
        end_ts = time.time()
        count = int(stats["table_edge_count"])
        edge_ts = sorted(float(v) for v in stats["table_edge_ts"])
        hz = None
        if len(edge_ts) >= 2 and edge_ts[-1] > edge_ts[0]:
            hz = float(len(edge_ts) - 1) / max(1e-6, edge_ts[-1] - edge_ts[0])
        summary = {
            "run_start_time": self._summary_start_wall,
            "run_end_time": end_ts,
            "duration_s": max(0.0, end_ts - float(self._summary_start_wall)),
            "module": self.module_name,
            "stack_run_id": self.stack_run_id,
            "detector_mode": dict(stats["detector_mode"].most_common()),
            "depth_shape": dict(stats["depth_shape"].most_common()),
            "calib_source": dict(stats["calib_source"].most_common()),
            "calib": dict(stats["calib"]),
            "table_edge_obs": {
                "count": count,
                "found_rate": (float(stats["table_edge_found"]) / float(count)) if count else None,
                "valid_for_control_rate": (float(stats["table_edge_valid"]) / float(count)) if count else None,
                "avg_hz": hz,
            },
            "vision_process_ms": self._summary_percentiles(stats["vision_process_ms"]),
            "camera_frame_interval_ms": self._summary_percentiles(stats["camera_frame_interval_ms"]),
            "vision_process_interval_ms": self._summary_percentiles(stats["vision_process_interval_ms"]),
            "vision_publish_interval_ms": self._summary_percentiles(stats["vision_publish_interval_ms"]),
            "obs_out_send_interval_ms": self._summary_percentiles(stats["obs_out_send_interval_ms"]),
            "orchestrator_recv_interval_ms": self._summary_percentiles(stats["orchestrator_recv_interval_ms"]),
            "state_machine_tick_interval_ms": self._summary_percentiles(stats["state_machine_tick_interval_ms"]),
            "obs_total_age_ms": self._summary_percentiles(stats["obs_total_age_ms"]),
            "vision_publish_to_orch_recv_ms": self._summary_percentiles(stats["vision_publish_to_orch_recv_ms"]),
            "orch_recv_to_state_consume_ms": self._summary_percentiles(stats["orch_recv_to_state_consume_ms"]),
            "same_obs_reuse_count": self._summary_percentiles(stats["same_obs_reuse_count"]),
            "ipc_summary": dict(stats["ipc"].most_common(32)),
            "preview_summary": dict(stats["preview"]),
            "cpu_summary": {key: self._summary_percentiles(values) for key, values in stats["cpu"].items()},
            "main_warnings_reject_reasons": dict(stats["reject_reasons"].most_common(32)),
        }
        with open(self._path_for("run_summary.json"), "w", encoding="utf-8") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2, sort_keys=True)
            fp.write("\n")
        with open(self._path_for("metrics_summary.json"), "w", encoding="utf-8") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2, sort_keys=True)
            fp.write("\n")

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
        self._enqueue_json_line("event", ordered)

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
        self._enqueue_json_line("ipc", ordered)

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
        self._enqueue_json_line("heartbeat", ordered)

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
            self.write_run_summary()
        except Exception:
            pass
        if self._async_enabled and self._log_thread is not None:
            self._log_stop.set()
            self._log_thread.join(timeout=2.0)
        try:
            self._event_fp.close()
        except Exception:
            pass
