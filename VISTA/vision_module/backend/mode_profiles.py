#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class RemoteProfile:
    """Capability requirements for remote inference or grasp cooperation."""

    enabled: bool = False
    base_url: Optional[str] = None
    command: str = "predict"
    require_depth: bool = False
    require_segmentation: bool = False
    timeout_s: float = 10.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreviewProfile:
    """Preview sink settings associated with a mode profile."""

    enabled: bool = False
    sink_name: str = "null"
    overlay_enabled: bool = True
    window_name: str = "VISTA App Dashboard"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModeProfile:
    """Resource-only definition for one runtime mode."""

    name: str
    enabled_cameras: Tuple[str, ...] = ()
    camera_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    predictor_enabled: bool = False
    predictor_model: Optional[str] = None
    remote: RemoteProfile = field(default_factory=RemoteProfile)
    preview: PreviewProfile = field(default_factory=PreviewProfile)
    loop_hz: Optional[float] = None
    send_hz: Optional[float] = None
    release_cooldown_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def camera_enabled(self, name: str) -> bool:
        """Check whether a camera should be active in this mode."""
        return str(name) in set(self.enabled_cameras)
