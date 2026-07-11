#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from typing import Any, Dict, Optional, Set

ALLOWED_INTENTS: Set[str] = {"FIND", "RETURN", "STOP"}

def new_id(prefix: str) -> str:
    import uuid
    return f"{prefix}_{uuid.uuid4().hex[:10]}"

def build_task_cmd(payload: Dict[str, Any]) -> Dict[str, Any]:
    intent = str(payload.get("intent", "")).upper().strip()
    if intent not in ALLOWED_INTENTS:
        raise ValueError(f"unsupported intent: {intent!r}")
        
    out: Dict[str, Any] = {
        "ts": float(payload.get("ts", time.time())),
        "type": "task_cmd",
        "intent": intent,
        "confidence": float(payload.get("confidence", 0.0)),
        "cmd_id": str(payload.get("cmd_id") or new_id("cmd")),
        "session_id": str(payload.get("session_id") or new_id("sess")),
        "epoch": int(payload.get("epoch", 0) or 0),
        "source": str(payload.get("source", "voice_gateway") or "voice_gateway"),
    }
    
    if intent == "FIND":
        target = str(payload.get("target", "")).strip()
        if not target:
            raise ValueError("FIND requires non-empty target")
        out["target"] = target
    
    for key in ("text", "raw_text", "high_priority", "wake_score"):
        if key in payload and payload[key] not in (None, ""):
            out[key] = payload[key]
            
    if intent == "STOP":
        out["high_priority"] = True
            
    return out

def normalize_task_ack(payload: Dict[str, Any]) -> Dict[str, Any]:
    if str(payload.get("type", "task_ack")) != "task_ack":
        raise ValueError("not a task_ack payload")
    cmd_id = str(payload.get("cmd_id", "")).strip()
    if not cmd_id:
        raise ValueError("task_ack requires cmd_id")
    return {
        "ts": float(payload.get("ts", time.time())),
        "type": "task_ack",
        "cmd_id": cmd_id,
        "session_id": str(payload.get("session_id", "") or ""),
        "epoch": int(payload.get("epoch", 0) or 0),
        "accepted": bool(payload.get("accepted", False)),
        "state": str(payload.get("state", "") or ""),
        "reason": str(payload.get("reason", "") or ""),
        "source": str(payload.get("source", "orchestrator") or "orchestrator"),
    }

def normalize_tts_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = str(payload.get("text", "")).strip()
    if not text:
        raise ValueError("tts_event requires non-empty text")
    return {
        "ts": float(payload.get("ts", time.time())),
        "type": str(payload.get("type", "tts_event")) or "tts_event",
        "text": text,
        "source": str(payload.get("source", "orchestrator")),
        "interrupt": bool(payload.get("interrupt", False)),
    }
