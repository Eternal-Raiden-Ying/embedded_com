#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local board config delegating to unified loader."""

from common.config_loader import get_config

try:
    from .schema import OnlineEdgeConfig
except ImportError:
    from schema import OnlineEdgeConfig

CONFIG = get_config().online_edge
