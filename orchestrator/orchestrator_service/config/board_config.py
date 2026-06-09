#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local board config delegating to unified loader."""

from common.config_loader import get_config
from .schema import OrchestratorConfig

CONFIG = get_config().orchestrator
