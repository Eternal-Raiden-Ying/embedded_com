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
from typing import Any, Dict, Iterable, Optional


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

    if stream_handler is None:
        stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
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
