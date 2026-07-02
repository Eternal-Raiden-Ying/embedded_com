#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TargetSpec:
    canonical_target: str
    class_name: str
    class_id: int
    aliases: tuple


OBJECT_REGISTRY: Dict[str, TargetSpec] = {
    "apple": TargetSpec("apple", "apple", 1, ("apple", "苹果")),
    "banana": TargetSpec("banana", "banana", 2, ("banana", "香蕉")),
    "basket": TargetSpec("basket", "basket", 3, ("basket", "篮子")),
    "bottle": TargetSpec("bottle", "bottle", 4, ("bottle", "瓶子", "水瓶", "矿泉水", "饮料瓶")),
    "grape": TargetSpec("grape", "grape", 5, ("grape", "葡萄")),
    "key": TargetSpec("key", "key", 6, ("key", "keys", "钥匙", "钥匙串")),
    "kiwi_fruit": TargetSpec("kiwi_fruit", "kiwi fruit", 7, ("kiwi", "kiwi fruit", "猕猴桃", "奇异果")),
    "lemon": TargetSpec("lemon", "lemon", 8, ("lemon", "柠檬")),
    "mango": TargetSpec("mango", "mango", 9, ("mango", "芒果")),
    "mouse": TargetSpec("mouse", "mouse", 10, ("mouse", "鼠标")),
    "orange": TargetSpec("orange", "orange", 11, ("orange", "橙子")),
    "peach": TargetSpec("peach", "peach", 12, ("peach", "桃子")),
    "star_fruit": TargetSpec("star_fruit", "star fruit", 13, ("star fruit", "starfruit", "杨桃")),
    "strawberry": TargetSpec("strawberry", "strawberry", 14, ("strawberry", "草莓")),
}


def _norm(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


TARGET_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for _canonical, _spec in OBJECT_REGISTRY.items():
    TARGET_ALIAS_TO_CANONICAL[_norm(_canonical)] = _canonical
    TARGET_ALIAS_TO_CANONICAL[_norm(_spec.class_name)] = _canonical
    for _alias in _spec.aliases:
        TARGET_ALIAS_TO_CANONICAL[_norm(_alias)] = _canonical

TARGET_NAME_TO_CLASS_ID = {
    alias: OBJECT_REGISTRY[canonical].class_id
    for alias, canonical in TARGET_ALIAS_TO_CANONICAL.items()
}


def resolve_target(target: str) -> Optional[TargetSpec]:
    canonical = TARGET_ALIAS_TO_CANONICAL.get(_norm(target))
    if not canonical:
        return None
    return OBJECT_REGISTRY.get(canonical)


def supported_targets() -> List[str]:
    return sorted(OBJECT_REGISTRY.keys())


def target_to_canonical(target: str) -> str:
    spec = resolve_target(target)
    if spec is None:
        raise KeyError(f"Unknown target '{target}'. Known targets: {supported_targets()}")
    return spec.canonical_target


def target_to_class_name(target: str) -> str:
    spec = resolve_target(target)
    if spec is None:
        raise KeyError(f"Unknown target '{target}'. Known targets: {supported_targets()}")
    return spec.class_name


def target_to_class_id(target: str) -> int:
    """Look up finetune yolo26s class_id for a target name.

    Args:
        target: Target name, case-insensitive. Must be in TARGET_NAME_TO_CLASS_ID.

    Returns:
        Finetune yolo26s class_id integer.

    Raises:
        KeyError: target is not in the mapping table.
    """
    spec = resolve_target(target)
    if spec is None:
        raise KeyError(f"Unknown target '{target}'. Known targets: {supported_targets()}")
    return int(spec.class_id)
