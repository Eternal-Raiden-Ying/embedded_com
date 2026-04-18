#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict

from ..backend.mode_profiles import ModeProfile, PreviewModeProfile, RemoteModeProfile


def build_default_mode_profiles(active_model_name: str) -> Dict[str, ModeProfile]:
    model_name = str(active_model_name or "").strip()
    preview = PreviewModeProfile(enabled=False, sink_name="null", window_name="VISTA Preview")
    return {
        "IDLE": ModeProfile(
            name="IDLE",
            enabled_cameras=(),
            predictor_enabled=False,
            predictor_model=model_name,
            preview=preview,
            metadata={"contract": {"stage": "IDLE"}},
        ),
        "IDLE_HOT": ModeProfile(
            name="IDLE_HOT",
            enabled_cameras=("rgb",),
            predictor_enabled=False,
            predictor_model=model_name,
            preview=preview,
            release_cooldown_s=5.0,
            metadata={"contract": {"stage": "IDLE_HOT"}},
        ),
        "TRACK_LOCAL": ModeProfile(
            name="TRACK_LOCAL",
            enabled_cameras=("rgb",),
            predictor_enabled=True,
            predictor_model=model_name,
            preview=preview,
            release_cooldown_s=2.0,
            metadata={"contract": {"perception": "target_obs"}},
        ),
        "DEPTH_PERCEPTION": ModeProfile(
            name="DEPTH_PERCEPTION",
            enabled_cameras=("depth",),
            predictor_enabled=False,
            predictor_model=model_name,
            preview=preview,
            release_cooldown_s=2.0,
            metadata={"contract": {"perception": "table_edge_obs"}},
        ),
        "MICRO_ADJUST": ModeProfile(
            name="MICRO_ADJUST",
            enabled_cameras=("rgb", "depth"),
            predictor_enabled=True,
            predictor_model=model_name,
            preview=preview,
            release_cooldown_s=2.0,
            metadata={"contract": {"interaction": "MOVE_HINT"}},
        ),
        "GRASP_REMOTE": ModeProfile(
            name="GRASP_REMOTE",
            enabled_cameras=("rgb", "depth"),
            predictor_enabled=False,
            predictor_model=model_name,
            remote=RemoteModeProfile(enabled=True),
            preview=preview,
            release_cooldown_s=2.0,
            metadata={"contract": {"result": "remote_result"}},
        ),
    }
