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
from typing import Any, Dict, Optional


TEXT_LOG_FORMAT = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def monotonic_ts() -> float:
    return time.monotonic()


def safe_dump(obj: Optional[Dict[str, Any]]) -> str:
    if obj is None:
        return "{}"
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def logging_level_for_mode(mode: str) -> int:
    return logging.DEBUG if str(mode).strip().lower() == "full" else logging.INFO


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


def configure_stream_logger(name: str, mode: str = "concise", enabled: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging_level_for_mode(mode))
    logger.propagate = False

    if not enabled:
        logger.handlers.clear()
        logger.disabled = True
        return logger

    logger.disabled = False
    formatter = logging.Formatter(TEXT_LOG_FORMAT)
    for handler in logger.handlers:
        handler.setLevel(logging_level_for_mode(mode))
        handler.setFormatter(formatter)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging_level_for_mode(mode))
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class RunLogger:
    def __init__(self, module_name: str, runs_root: str, stack_run_id: str = "", enable_text_events: bool = True):
        self.module_name = str(module_name).strip()
        self.stack_run_id = str(stack_run_id).strip() or make_stack_run_id()
        self.run_dir = Path(runs_root) / self.stack_run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._enable_text_events = bool(enable_text_events)
        self._event_fp = open(self.run_dir / "events.log", "a", encoding="utf-8")

    def structured_paths(self, heartbeat_enabled: bool = True) -> Dict[str, str]:
        paths = {
            "run_dir": str(self.run_dir),
            "meta": str(self.run_dir / "meta.json"),
            "event": str(self.run_dir / "event.jsonl"),
            "ipc": str(self.run_dir / "ipc.jsonl"),
        }
        if heartbeat_enabled:
            paths["heartbeat"] = str(self.run_dir / "heartbeat.jsonl")
        return paths

    def _with_common_fields(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(payload)
        out.setdefault("module", self.module_name)
        out.setdefault("stack_run_id", self.stack_run_id)
        out.setdefault("ts", time.time())
        return out

    def write_meta(self, payload: Dict[str, Any]) -> None:
        meta_payload = self._with_common_fields(payload)
        with open(self.run_dir / "meta.json", "w", encoding="utf-8") as fp:
            json.dump(meta_payload, fp, ensure_ascii=False, indent=2)
            fp.write("\n")

    def write_jsonl(self, name: str, payload: Dict[str, Any]) -> None:
        path = self.run_dir / f"{name}.jsonl"
        line = json.dumps(self._with_common_fields(payload), ensure_ascii=False, separators=(",", ":"))
        with open(path, "a", encoding="utf-8") as fp:
            fp.write(line + "\n")

    def write_event(self, text: str) -> None:
        if not self._enable_text_events:
            return
        self._event_fp.write(f"[{time.strftime('%H:%M:%S')}] {text}\n")
        self._event_fp.flush()

    def write_event_record(self, event: str, level: str = "info", trigger: str = "", data=None, **fields: Any) -> None:
        payload = {
            "event": str(event).strip().upper(),
            "level": str(level or "info").strip().lower(),
            "trigger": str(trigger or "").strip(),
            "data": dict(data or {}),
        }
        payload.update(fields)
        self.write_jsonl("event", payload)
        details = ", ".join(f"{k}={v}" for k, v in fields.items() if v not in (None, "", [], {}))
        self.write_event(f"{payload['event']}{' ' + details if details else ''}".rstrip())

    def write_ipc_record(self, direction: str, channel: str, event: str, level: str = "info", data=None, **fields: Any) -> None:
        payload = {
            "direction": str(direction or "").strip().upper(),
            "channel": str(channel or "").strip(),
            "event": str(event or "").strip(),
            "level": str(level or "info").strip().lower(),
            "data": dict(data or {}),
        }
        payload.update(fields)
        self.write_jsonl("ipc", payload)

    def write_heartbeat_record(self, **payload: Any) -> None:
        self.write_jsonl("heartbeat", payload)

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
