import json
import logging
import os


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_int_list_csv(raw_value):
    if raw_value is None:
        return []
    values = []
    for item in str(raw_value).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def log_kv_block(logger: logging.Logger, title: str, items: dict):
    logger.info("%s:", title)
    for key, value in items.items():
        logger.info(" - %s: %s", key, value)
