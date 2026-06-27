#!/usr/bin/env python3
# -*- coding: utf-8 -*-

CONTROL_PERCEPTION_KEYS = ("table_edge_obs", "target_obs", "home_tag_obs")
DIAGNOSTIC_EXCLUDED_PERCEPTION_KEYS = set(CONTROL_PERCEPTION_KEYS)

URGENT_STATUSES = {"START", "STOP", "FAILED", "FATAL"}
URGENT_REQUEST_OPS = {"START", "STOP"}
