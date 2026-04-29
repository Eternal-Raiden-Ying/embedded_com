#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Dict

from ..backend.mode_profiles import ModeProfile, PreviewProfile, RemoteProfile


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


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
    table_bbox_enabled = _env_bool("VISTA_TABLE_BBOX_ENABLE", False)
    depth_cameras = ("rgb", "depth") if table_bbox_enabled else ("depth",)
    table_model = str(os.getenv("VISTA_TABLE_MODEL", "yolov7_detect") or "yolov7_detect").strip()
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
            enabled_cameras=depth_cameras,
            predictor_enabled=table_bbox_enabled,
            predictor_model=table_model if table_bbox_enabled else None,
            preview=PreviewProfile(enabled=True, sink_name="opencv"),
            release_cooldown_s=2.0,
            metadata={
                "contract": {
                    "cameras": list(depth_cameras),
                    "predictor": "optional_table_bbox" if table_bbox_enabled else "disabled",
                    "remote": "disabled",
                    "perception": "table_edge_obs",
                }
            },
        ),
        "TABLE_EDGE_PERCEPTION": ModeProfile(
            name="TABLE_EDGE_PERCEPTION",
            enabled_cameras=("rgb", "depth"),
            predictor_enabled=True,
            predictor_model=active_model,
            preview=PreviewProfile(enabled=True, sink_name="opencv"),
            release_cooldown_s=2.0,
            metadata={
                "contract": {
                    "cameras": ["rgb", "depth"],
                    "predictor": "required",
                    "table_edge": "required",
                    "remote": "disabled",
                    "perception": ["local_perception", "table_edge_obs"],
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
