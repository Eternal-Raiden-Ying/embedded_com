#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local board config delegating to unified loader."""

from common.config_loader import get_config, load_global_config, load_yaml_file
from .schema import MobileGatewayConfig

CONFIG = get_config().gateway


def _set_attrs(obj, values) -> None:
    for key, value in dict(values or {}).items():
        if value in (None, ""):
            continue
        if hasattr(obj, key):
            setattr(obj, key, value)


def _apply_legacy_gateway_config(cfg: MobileGatewayConfig, config_file: str) -> MobileGatewayConfig:
    data = load_yaml_file(config_file)
    if not isinstance(data, dict) or "gateway" in data:
        return cfg

    runtime = data.get("runtime")
    if isinstance(runtime, dict):
        _set_attrs(cfg.runtime, runtime)

    mqtt = data.get("mqtt")
    if isinstance(mqtt, dict):
        topics = mqtt.get("topics")
        _set_attrs(cfg.mqtt, {k: v for k, v in mqtt.items() if k != "topics"})
        if isinstance(topics, dict):
            _set_attrs(cfg.mqtt.topics, topics)

    # Older mobile_gateway.mqtt.yaml files used top-level sections. Keep the
    # current unified profile's southbound transport unless explicitly
    # overridden by environment variables; the legacy file mainly restores the
    # cloud MQTT phone entry.
    return cfg


def build_config(config_file: str = "") -> MobileGatewayConfig:
    if config_file:
        data = load_yaml_file(config_file)
        if isinstance(data, dict) and "gateway" in data:
            return load_global_config(config_path=config_file).gateway
        cfg = load_global_config().gateway
        return _apply_legacy_gateway_config(cfg, config_file)
    return get_config().gateway
