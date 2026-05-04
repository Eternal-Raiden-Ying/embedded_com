#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Mapping from target name to COCO80 class_id.
# This is the single point of change when switching to a different YOLO class set.

TARGET_NAME_TO_CLASS_ID = {
    "apple": 47,
    "banana": 46,
    "bottle": 39,
    "cup": 41,
}


def target_to_class_id(target: str) -> int:
    """Look up COCO80 class_id for a target name.

    Args:
        target: Target name, case-insensitive. Must be in TARGET_NAME_TO_CLASS_ID.

    Returns:
        COCO80 class_id integer.

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
