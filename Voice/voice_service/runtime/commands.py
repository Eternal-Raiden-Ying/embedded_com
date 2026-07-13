#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Voice intent parsing backed by the shared trained-target catalog."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from common.target_catalog import resolve_target_in_text, target_display_name
from .common import normalize_text

RESIDUAL_TEXTS = {"啊", "哦", "嗯", "请问", "谢谢", "你好", "您好", "这个", "那个"}
DEFAULT_COMMAND_RULES = {
    "stop": ["小车停止", "小车停下", "停止", "停下", "别动", "取消", "危险", "紧急停止", "马上停下", "stop"],
    "return": ["回来", "返回", "回去", "return"],
}

class CommandInterpreter:
    def __init__(self, rules: Optional[Dict[str, Any]] = None):
        supplied = rules or {}
        self.rules = {
            "stop": list(supplied.get("stop", DEFAULT_COMMAND_RULES["stop"])),
            "return": list(supplied.get("return", DEFAULT_COMMAND_RULES["return"])),
        }

    @classmethod
    def from_json(cls, json_path: str) -> "CommandInterpreter":
        if not json_path or not Path(json_path).exists():
            return cls()
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        # Legacy find mappings are deliberately ignored: catalog aliases are authoritative.
        return cls({"stop": data.get("stop", DEFAULT_COMMAND_RULES["stop"]), "return": data.get("return", DEFAULT_COMMAND_RULES["return"])})

    def is_stop_text(self, text: str) -> bool:
        raw = normalize_text(text)
        lower = raw.lower()
        return bool(raw) and any(keyword in lower or keyword in raw for keyword in self.rules["stop"])

    def is_return_text(self, text: str) -> bool:
        raw = normalize_text(text)
        lower = raw.lower()
        return bool(raw) and any(keyword in lower or keyword in raw for keyword in self.rules["return"])

    def is_residual_text(self, text: str) -> bool:
        raw = normalize_text(text)
        if not raw or raw in RESIDUAL_TEXTS:
            return True
        if len(raw) == 1:
            return resolve_target_in_text(raw) is None and not self.is_stop_text(raw) and not self.is_return_text(raw)
        return False

    def target_display_name(self, target: Optional[str]) -> str:
        return target_display_name(target) if target else "目标"

    def infer_intent_and_target(self, text: str) -> Tuple[str, Optional[str], float]:
        raw = normalize_text(text)
        if not raw:
            return "REJECT", None, 0.0
        if self.is_stop_text(raw):
            return "STOP", None, 0.92
        if self.is_return_text(raw):
            return "RETURN", None, 0.86
        spec = resolve_target_in_text(raw, selectable_only=True)
        if spec is not None:
            return "FIND", spec.canonical_name, 0.78
        return "REJECT", None, 0.0
