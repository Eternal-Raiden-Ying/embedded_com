#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class RemoteModeProfile:
    enabled: bool = False
    base_url: str = ""


@dataclass(frozen=True)
class PreviewModeProfile:
    enabled: bool = False
    sink_name: str = "null"
    window_name: str = "VISTA Preview"


@dataclass(frozen=True)
class ModeProfile:
    name: str
    enabled_cameras: Tuple[str, ...] = ()
    camera_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    predictor_enabled: bool = False
    predictor_model: str = ""
    remote: RemoteModeProfile = field(default_factory=RemoteModeProfile)
    preview: PreviewModeProfile = field(default_factory=PreviewModeProfile)
    loop_hz: float = 8.0
    send_hz: float = 5.0
    release_cooldown_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "name", str(self.name or "IDLE").strip().upper() or "IDLE")
