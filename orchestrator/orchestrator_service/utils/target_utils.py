#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Mapping from target name to the finetune yolo26s classes.txt 0-based class_id.
# This is the single point of change when switching to a different YOLO class set.

TARGET_NAME_TO_CLASS_ID = {
    "apple": 1,
    "banana": 2,
    "basket": 3,
    "bottle": 4,
    "grape": 5,
    "key": 6,
    "keys": 6,
    "kiwi": 7,
    "kiwi fruit": 7,
    "lemon": 8,
    "mango": 9,
    "mouse": 10,
    "orange": 11,
    "peach": 12,
    "star fruit": 13,
    "starfruit": 13,
    "strawberry": 14,
}


def target_to_class_id(target: str) -> int:
    """Look up finetune yolo26s class_id for a target name.

    Args:
        target: Target name, case-insensitive. Must be in TARGET_NAME_TO_CLASS_ID.

    Returns:
        Finetune yolo26s class_id integer.

    Raises:
        KeyError: target is not in the mapping table.
    """
    key = target.strip().lower()
    if key not in TARGET_NAME_TO_CLASS_ID:
        raise KeyError(
            f"Unknown target '{target}'. "
            f"Known targets: {sorted(TARGET_NAME_TO_CLASS_ID.keys())}"
        )
    return TARGET_NAME_TO_CLASS_ID[key]
