#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Iterable, Tuple


def normalize_class_name(name: object) -> str:
    return " ".join(str(name or "").strip().lower().split())


def normalize_class_names(names: Iterable[object] | None) -> Tuple[str, ...]:
    if not names:
        return ()
    return tuple(normalize_class_name(name) for name in tuple(names))


def normalize_vocab_map(vocab_map: Dict[object, Iterable[object]]) -> Dict[str, set]:
    normalized: Dict[str, set] = {}
    for key, values in dict(vocab_map or {}).items():
        normalized_key = normalize_class_name(key)
        if not normalized_key:
            continue
        normalized_values = {
            normalize_class_name(value)
            for value in tuple(values or ())
            if normalize_class_name(value)
        }
        normalized[normalized_key] = normalized_values
    return normalized


#########################################
#              yolo target              #
#########################################

coco80 = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


grasping_coco20 = (
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "orange",
    "broccoli",
    "carrot",
    "mouse",
    "remote",
    "cell phone",
    "book",
    "clock",
    "scissors",
    "teddy bear",
    "toothbrush",
)

finetune_yolo26s_bgr15 = (
    "table1",
    "apple",
    "banana",
    "basket",
    "bottle",
    "grape",
    "key",
    "kiwi fruit",
    "lemon",
    "mango",
    "mouse",
    "orange",
    "peach",
    "star fruit",
    "strawberry",
)

COCO80_CLASSES = normalize_class_names(coco80)
GRASPING_COCO20_CLASSES = normalize_class_names(grasping_coco20)
FINETUNE_YOLO26S_BGR15_CLASSES = normalize_class_names(finetune_yolo26s_bgr15)


#########################################
#            asr vocabulary             #
#########################################

asr_class_map = {
    "cup": set(),
    "bottle": {"bottle"},
    "phone": set(),
    "remote": set(),
    "apple": {"apple"},
    "banana": {"banana"},
    "basket": {"basket"},
    "grape": {"grape"},
    "key": {"key"},
    "keys": {"key"},
    "kiwi": {"kiwi fruit"},
    "kiwi fruit": {"kiwi fruit"},
    "lemon": {"lemon"},
    "mango": {"mango"},
    "mouse": {"mouse"},
    "orange": {"orange"},
    "peach": {"peach"},
    "star fruit": {"star fruit"},
    "starfruit": {"star fruit"},
    "strawberry": {"strawberry"},
    "book": set(),
    # The following targets do not have a reliable class in the current model.
    "medicine_box": set(),
    "wallet": set(),
}

ASR_VOCAB_MAP = normalize_vocab_map(asr_class_map)
