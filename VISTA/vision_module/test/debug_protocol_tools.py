#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


STATE_PATH = Path(__file__).with_name(".vista_debug_state.json")


def load_debug_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_debug_state(state: Dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def remember_interaction(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if str(payload.get("type", "")).strip().lower() != "vision_obs":
        return None
    interaction = payload.get("interaction") or {}
    interaction_id = interaction.get("interaction_id")
    if not interaction_id:
        return None
    state = {
        "last_interaction_id": str(interaction_id),
        "last_stage": payload.get("stage"),
        "last_mode": payload.get("mode"),
        "last_status": payload.get("status"),
        "last_session_id": payload.get("session_id"),
        "last_req_id": payload.get("req_id"),
        "updated_at": time.time(),
    }
    save_debug_state(state)
    return state


def summarize_obs(payload: Dict[str, Any]) -> str:
    msg_type = str(payload.get("type", "unknown")).strip().upper()
    if msg_type != "VISION_OBS":
        return msg_type

    stage = payload.get("stage") or "?"
    mode = payload.get("mode") or "?"
    status = payload.get("status") or "?"
    parts = [f"stage={stage}", f"mode={mode}", f"status={status}"]

    interaction = payload.get("interaction") or {}
    if interaction.get("interaction_id"):
        parts.append(f"interaction_id={interaction['interaction_id']}")

    perception = payload.get("perception") or {}
    for key in ("target_obs", "home_tag_obs"):
        item = perception.get(key)
        if isinstance(item, dict) and "found" in item:
            parts.append(f"{key}.found={item.get('found')}")
            break

    return " | ".join(parts)
