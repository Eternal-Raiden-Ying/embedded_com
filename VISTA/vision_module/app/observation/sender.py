#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict, Optional


class ObservationSender:
    """Small adapter boundary around the app-level sender callback."""

    def __init__(self, send_func):
        self._send_func = send_func

    def send(self, payload: Dict[str, Any], *, sender: Optional[Any] = None, obs_class: str = "control") -> bool:
        return bool(self._send_func(payload, sender=sender, obs_class=obs_class))
