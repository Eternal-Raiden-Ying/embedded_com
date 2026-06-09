#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local board config delegating to unified loader."""

from common.config_loader import get_config, load_global_config
from .schema import MobileGatewayConfig

CONFIG = get_config().gateway


def build_config(config_file: str = "") -> MobileGatewayConfig:
    if config_file:
        return load_global_config(config_path=config_file).gateway
    return get_config().gateway
