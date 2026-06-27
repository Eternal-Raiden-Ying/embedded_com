#!/usr/bin/env python3
"""Static audit for table docking control-path ownership.

This is intentionally lightweight.  It highlights places that still mention
velocity, stale, STOP, or FOV/commit gates so reviewers can quickly spot new
scattered control ownership after the central motion arbiter refactor.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[2]

SCAN_FILES = (
    "orchestrator/orchestrator_service/runtime/docking_model.py",
    "orchestrator/orchestrator_service/runtime/motion_arbiter.py",
    "orchestrator/orchestrator_service/runtime/states/table_docking.py",
    "orchestrator/orchestrator_service/runtime/service.py",
    "orchestrator/orchestrator_service/bridge/uart_bridge.py",
    "orchestrator/orchestrator_service/runtime/control_authority.py",
    "orchestrator/orchestrator_service/runtime/context.py",
    "orchestrator/orchestrator_service/control/motion_controller.py",
)

KEYWORDS = (
    "vx_mps =",
    "wz_radps =",
    "forward_block_reason",
    "rotate_block_reason",
    "allow_forward",
    "allow_rotate",
    "bbox_fov_guard",
    "stale_level",
    "hard_stale",
    "dead",
    "send_emergency_stop",
    "send_stm32_stop",
    "_clear_motion_queues_for_estop",
    "_last_estop_mono",
    "last_valid_expired",
    "approach_commit",
    "zero_cmd_reason",
    "service_override",
)


def _matches(line: str) -> List[str]:
    return [kw for kw in KEYWORDS if kw in line]


def _category(rel: str, line: str) -> str:
    text = line.strip()
    if "docking_model.py" in rel:
        return "allowed central arbiter paths"
    if "motion_arbiter.py" in rel:
        return "allowed central arbiter paths"
    if "table_docking.py" in rel and ("decision.cmd.vx_mps" in text or "decision.cmd.wz_radps" in text):
        return "allowed central arbiter paths"
    if "uart_bridge.py" in rel and (
        "send_emergency_stop" in text
        or "send_stm32_stop" in text
        or "_clear_motion_queues_for_estop" in text
        or "_last_estop_mono" in text
        or "writer_discard_reason" in text
    ):
        return "allowed hard safety paths"
    if "service.py" in rel and (
        "send_emergency_stop" in text
        or "send_stm32_stop" in text
        or "estop_cooldown" in text
        or "service_override" in text
    ):
        return "allowed hard safety paths"
    if "control_authority.py" in rel:
        return "diagnostic-only paths"
    if "context.py" in rel:
        return "diagnostic-only paths"
    if "motion_controller.py" in rel:
        return "diagnostic-only paths"
    if "service.py" in rel:
        return "diagnostic-only paths"
    if "table_docking.py" in rel:
        return "diagnostic-only paths"
    return "suspicious scattered control paths"


def scan() -> Dict[str, List[Tuple[str, int, str, str]]]:
    buckets: Dict[str, List[Tuple[str, int, str, str]]] = defaultdict(list)
    for rel in SCAN_FILES:
        path = ROOT / rel
        if not path.exists():
            buckets["suspicious scattered control paths"].append((rel, 0, "missing_file", ""))
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            matched = _matches(line)
            if not matched:
                continue
            buckets[_category(rel, line)].append((rel, lineno, ",".join(matched), line.strip()))
    return buckets


def _print_bucket(name: str, rows: Iterable[Tuple[str, int, str, str]], *, limit: int = 24) -> None:
    rows = list(rows)
    print(f"\n{name}: {len(rows)}")
    for rel, lineno, matched, text in rows[:limit]:
        print(f"  {rel}:{lineno}: [{matched}] {text}")
    if len(rows) > limit:
        print(f"  ... {len(rows) - limit} more")


def main() -> None:
    buckets = scan()
    for name in (
        "allowed central arbiter paths",
        "allowed hard safety paths",
        "diagnostic-only paths",
        "suspicious scattered control paths",
    ):
        _print_bucket(name, buckets.get(name, []))
    suspicious = buckets.get("suspicious scattered control paths", [])
    if suspicious:
        print("\nAudit note: suspicious entries require review before real-robot testing.")
    else:
        print("\ndocking control path audit: PASS")


if __name__ == "__main__":
    main()
