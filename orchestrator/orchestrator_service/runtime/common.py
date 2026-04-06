#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional


def configure_logging(mode: str = "concise"):
    level = logging.DEBUG if mode == "full" else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


class RunLogger:
    def __init__(self, runs_root: str):
        ts = time.strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = Path(runs_root) / ts
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._event_fp = open(self.run_dir / "events.log", "a", encoding="utf-8")

    def write_jsonl(self, name: str, payload: Dict[str, Any]):
        path = self.run_dir / f"{name}.jsonl"
        with open(path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

    def write_event(self, text: str):
        self._event_fp.write(f"[{time.strftime('%H:%M:%S')}] {text}\n")
        self._event_fp.flush()

    def write_timeline(self, event: str, **fields):
        payload = {"ts": time.time(), "event": event}
        payload.update(fields)
        self.write_jsonl("timeline", payload)

    def write_ipc(self, channel: str, event: str, **fields):
        payload = {"ts": time.time(), "channel": channel, "event": event}
        payload.update(fields)
        self.write_jsonl("ipc", payload)
        msg = fields.get("msg") or fields.get("error") or ""
        self.write_event(f"IPC {channel} {event} {msg}".rstrip())

    def write_state_block(self, block: Dict[str, Any]):
        self.write_jsonl("state_blocks", block)
        one_line = (
            f"state={block.get('state')} target={block.get('active_target')} "
            f"session={block.get('session_id')} epoch={block.get('epoch')} "
            f"reason={block.get('last_enter_reason')} fail={block.get('last_fail_reason')}"
        )
        self.write_event(one_line)

    def close(self):
        try:
            self._event_fp.close()
        except Exception:
            pass


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def monotonic_ts() -> float:
    return time.monotonic()


def safe_dump(obj: Optional[Dict[str, Any]]) -> str:
    if obj is None:
        return "{}"
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
