#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .common import normalize_text

RESIDUAL_TEXTS = {
    "嗯", "啊", "呃", "额", "诶", "欸", "哎", "唉", "哦", "噢", "喔",
    "车", "小车", "你好", "您好", "这个", "那个"
}


DEFAULT_COMMAND_RULES = {
    "stop": ["小车停止", "小车停下", "停止", "停下", "别动", "取消", "危险", "紧急停止", "马上停下", "stop", "停"],
    "return": ["回来", "返回", "回去", "return"],
    "find": {
        "cup": ["水杯", "杯子", "杯", "马克杯", "玻璃杯"],
        "bottle": ["瓶子", "水瓶", "饮料瓶"],
        "phone": ["手机", "电话"],
        "remote": ["遥控器", "遥控"],
        "medicine_box": ["药盒", "药箱", "药"],
        "keys": ["钥匙", "钥匙串"],
        "apple": ["苹果"],
        "banana": ["香蕉"],
        "book": ["书", "书本"],
        "wallet": ["钱包"],
    },
}


class CommandInterpreter:
    def __init__(self, rules: Optional[Dict[str, Any]] = None):
        self.rules = rules or json.loads(json.dumps(DEFAULT_COMMAND_RULES, ensure_ascii=False))

    @classmethod
    def from_json(cls, json_path: str) -> "CommandInterpreter":
        if not json_path:
            return cls()
        path = Path(json_path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        rules = {
            "stop": list(data.get("stop", DEFAULT_COMMAND_RULES["stop"])),
            "return": list(data.get("return", DEFAULT_COMMAND_RULES["return"])),
            "find": dict(data.get("find", DEFAULT_COMMAND_RULES["find"])),
        }
        return cls(rules)

    def is_stop_text(self, text: str) -> bool:
        t_raw = normalize_text(text)
        t = t_raw.lower()
        stop_kw = self.rules.get("stop", [])
        return bool(t_raw) and any(k in t or k in t_raw for k in stop_kw)

    def is_return_text(self, text: str) -> bool:
        t_raw = normalize_text(text)
        t = t_raw.lower()
        return_kw = self.rules.get("return", [])
        return bool(t_raw) and any(k in t or k in t_raw for k in return_kw)

    def is_residual_text(self, text: str) -> bool:
        t = normalize_text(text)
        if not t:
            return True
        if t in RESIDUAL_TEXTS:
            return True
        if len(t) == 1:
            for kws in self.rules.get("find", {}).values():
                if any(t == normalize_text(k) for k in kws):
                    return False
            return not self.is_stop_text(t) and not self.is_return_text(t)
        return False

    def target_display_name(self, target: Optional[str]) -> str:
        if not target:
            return "目标"
        kws = self.rules.get("find", {}).get(target, [])
        if kws:
            return str(kws[0])
        return str(target)

    def infer_intent_and_target(self, text: str) -> Tuple[str, Optional[str], float]:
        t_raw = normalize_text(text)
        if not t_raw:
            return "REJECT", None, 0.0

        if self.is_stop_text(t_raw):
            return "STOP", None, 0.92

        if self.is_return_text(t_raw):
            return "RETURN", None, 0.86

        target_map = self.rules.get("find", {})
        for target, kws in target_map.items():
            if any(k in t_raw for k in kws):
                return "FIND", target, 0.78

        return "REJECT", None, 0.0
