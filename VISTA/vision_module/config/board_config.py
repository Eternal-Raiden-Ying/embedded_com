#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local board config delegating to unified loader."""

from common.config_loader import get_config
from .schema import VisionServiceConfig, SingleModelConfig
from .data import coco80, finetune_yolo26s_bgr15, grasping_coco20

CONFIG = get_config().vision
