#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Dict

from ..backend.mode_profiles import ModeProfile, PreviewProfile, RemoteProfile


def build_default_stage_entry_modes() -> Dict[str, str]:
    """Return the recommended initial mode for each business stage."""
    return {
        "IDLE": "IDLE",
        "SEARCH": "TRACK_LOCAL",
        "GRASP": "MICRO_ADJUST",
        "RETURN": "TRACK_LOCAL",
    }


def build_default_mode_profiles(active_model: str) -> Dict[str, ModeProfile]:
    """Build the initial mode profile set for VISTA."""
    return {
        "IDLE": ModeProfile(
            name="IDLE",
            enabled_cameras=(),
            predictor_enabled=False,
            predictor_model=None,
            preview=PreviewProfile(enabled=False, sink_name="null"),
            release_cooldown_s=0.0,
            metadata={"contract": {"stage": "IDLE"}},
        ),
        "TRACK_LOCAL": ModeProfile(
            name="TRACK_LOCAL",
            enabled_cameras=("rgb",),
            predictor_enabled=True,
            predictor_model=active_model,
            preview=PreviewProfile(enabled=True, sink_name="opencv"),
            release_cooldown_s=2.0,
            metadata={
                "contract": {
                    "cameras": ["rgb"],
                    "predictor": "required",
                    "remote": "disabled",
                    "perception": "target_obs",
                }
            },
        ),
        "DEPTH_PERCEPTION": ModeProfile(
            name="DEPTH_PERCEPTION",
            enabled_cameras=("depth",),
            predictor_enabled=False,
            predictor_model=None,
            preview=PreviewProfile(enabled=True, sink_name="opencv"),
            release_cooldown_s=2.0,
            metadata={
                "contract": {
                    "cameras": ["depth"],
                    "predictor": "disabled",
                    "remote": "disabled",
                    "perception": "table_edge_obs",
                }
            },
        ),
        "MICRO_ADJUST": ModeProfile(
            name="MICRO_ADJUST",
            enabled_cameras=("rgb", "depth"),
            predictor_enabled=True,
            predictor_model=active_model,
            preview=PreviewProfile(enabled=True, sink_name="opencv"),
            release_cooldown_s=2.0,
            metadata={"contract": {"interaction": "MOVE_HINT"}},
        ),
        "GRASP_REMOTE": ModeProfile(
            name="GRASP_REMOTE",
            enabled_cameras=("rgb", "depth"),
            predictor_enabled=False,
            predictor_model=None,
            remote=RemoteProfile(
                enabled=True,
                base_url=str(os.getenv("VISION_REMOTE_BASE_URL", "")).strip() or None,
                require_depth=True,
                require_segmentation=False,
            ),
            preview=PreviewProfile(enabled=True, sink_name="opencv"),
            release_cooldown_s=3.0,
            metadata={
                "contract": {
                    "cameras": ["rgb", "depth"],
                    "predictor": "disabled",
                    "remote": "required",
                    "result": "remote_result",
                }
            },
        ),
        "IDLE_HOT": ModeProfile(
            name="IDLE_HOT",
            enabled_cameras=("rgb",),
            predictor_enabled=False,
            predictor_model=None,
            preview=PreviewProfile(enabled=True, sink_name="opencv"),
            release_cooldown_s=5.0,
            metadata={"contract": {"stage": "IDLE_HOT"}},
        ),
    }
