#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Any, Dict, Optional

from ..backend.mode_profiles import ModeProfile, PreviewProfile, RemoteProfile


def build_default_stage_entry_modes() -> Dict[str, str]:
    """Return the recommended initial mode for each business stage."""
    return {
        "IDLE": "IDLE",
        "SEARCH": "TRACK_LOCAL",
        "GRASP": "MICRO_ADJUST",
        "RETURN": "TRACK_LOCAL",
    }


def _camera_override_from_config(cfg, camera_name: str) -> Dict[str, Any]:
    camera_cfg = getattr(getattr(cfg, "camera", None), "streams", {}).get(camera_name)
    if camera_cfg is None:
        return {}
    keys_by_camera = {
        "rgb": (
            "source",
            "in_w",
            "in_h",
            "out_w",
            "out_h",
            "in_format",
            "format",
            "fps",
            "crop_x",
            "crop_y",
            "crop_w",
            "crop_h",
            "auto_exposure",
            "exposure",
            "brightness",
        ),
        "depth": ("source", "width", "height", "fps"),
        "grey": (
            "source",
            "in_w",
            "in_h",
            "out_w",
            "out_h",
            "in_format",
            "format",
            "fps",
            "crop_x",
            "crop_y",
            "crop_w",
            "crop_h",
        ),
    }
    fields = {}
    for key in keys_by_camera.get(camera_name, ()):
        value = getattr(camera_cfg, key, None)
        if value is not None:
            fields[key] = value
    return fields


def _camera_overrides_for(cfg, camera_names) -> Dict[str, Dict[str, Any]]:
    overrides: Dict[str, Dict[str, Any]] = {}
    for camera_name in tuple(camera_names or ()):
        payload = _camera_override_from_config(cfg, str(camera_name))
        if payload:
            overrides[str(camera_name)] = payload
    return overrides


def _camera_override_with_updates(cfg, camera_name: str, **updates: Any) -> Dict[str, Any]:
    payload = _camera_override_from_config(cfg, camera_name)
    for key, value in dict(updates or {}).items():
        if value is not None:
            payload[str(key)] = value
    return payload


def _default_remote_profile(*, enabled: bool, require_depth: bool = False) -> RemoteProfile:
    return RemoteProfile(
        enabled=bool(enabled),
        base_url=str(os.getenv("VISION_REMOTE_BASE_URL", "")).strip() or None,
        require_depth=bool(require_depth),
        rgb_encoding=str(os.getenv("VISION_REMOTE_RGB_ENCODING", "jpeg")).strip().lower() or "jpeg",
        depth_encoding=str(os.getenv("VISION_REMOTE_DEPTH_ENCODING", "png")).strip().lower() or "png",
        rgb_quality=int(os.getenv("VISION_REMOTE_RGB_QUALITY", "90") or 90),
        depth_compression=int(os.getenv("VISION_REMOTE_DEPTH_COMPRESSION", "3") or 3),
    )


def build_default_mode_profiles(active_model: str, cfg: Optional[Any] = None) -> Dict[str, ModeProfile]:
    """Build the initial mode profile set for VISTA.

    These defaults are intentionally simple and document the intended mode
    inventory before the mode controller is fully wired into the engine.
    """
    track_local_rgb = _camera_override_with_updates(
        cfg,
        "rgb",
        format="BGR",
        in_w=1280,
        in_h=720,
        out_w=640,
        out_h=640,
        fps=24,
        crop_x=280,
        crop_y=0,
        crop_w=720,
        crop_h=720,
    )
    micro_adjust_rgb = _camera_override_with_updates(
        cfg,
        "rgb",
        format="BGR",
        in_w=1280,
        in_h=720,
        out_w=640,
        out_h=640,
        fps=30,
        crop_x=320,
        crop_y=40,
        crop_w=640,
        crop_h=640,
    )
    grasp_remote_rgb = _camera_override_with_updates(
        cfg,
        "rgb",
        format="BGR",
        in_w=1280,
        in_h=720,
        out_w=640,
        out_h=640,
        fps=15,
        crop_x=280,
        crop_y=0,
        crop_w=720,
        crop_h=720,
    )
    idle_hot_rgb = _camera_override_with_updates(
        cfg,
        "rgb",
        format="BGR",
        in_w=1280,
        in_h=720,
        out_w=640,
        out_h=640,
        fps=10,
        crop_x=280,
        crop_y=0,
        crop_w=720,
        crop_h=720,
    )
    grasp_remote_cameras = {
        "rgb": grasp_remote_rgb,
        **_camera_overrides_for(cfg, ("depth",)),
    }
    return {
        "IDLE": ModeProfile(
            name="IDLE",
            enabled_cameras=(),
            predictor_enabled=False,
            predictor_model=None,
            remote=_default_remote_profile(enabled=False),
            preview=PreviewProfile(enabled=False, sink_name="null"),
            release_cooldown_s=0.0,
        ),
        "TRACK_LOCAL": ModeProfile(
            name="TRACK_LOCAL",
            enabled_cameras=("rgb",),
            camera_overrides={"rgb": track_local_rgb},
            predictor_enabled=True,
            predictor_model=active_model,
            remote=_default_remote_profile(enabled=False),
            preview=PreviewProfile(enabled=True, sink_name="opencv"),
            release_cooldown_s=2.0,
            metadata={
                "contract": {
                    "cameras": ["rgb"],
                    "predictor": "required",
                    "remote": "disabled",
                }
            },
        ),
        "MICRO_ADJUST": ModeProfile(
            name="MICRO_ADJUST",
            enabled_cameras=("rgb",),
            camera_overrides={"rgb": micro_adjust_rgb},
            predictor_enabled=True,
            predictor_model=active_model,
            remote=_default_remote_profile(enabled=False),
            preview=PreviewProfile(enabled=True, sink_name="opencv"),
            release_cooldown_s=2.0,
        ),
        "GRASP_REMOTE": ModeProfile(
            name="GRASP_REMOTE",
            enabled_cameras=("rgb", "depth"),
            camera_overrides=grasp_remote_cameras,
            predictor_enabled=False,
            predictor_model=None,
            remote=_default_remote_profile(enabled=True, require_depth=True),
            preview=PreviewProfile(enabled=True, sink_name="opencv"),
            release_cooldown_s=3.0,
            metadata={
                "contract": {
                    "cameras": ["rgb", "depth"],
                    "predictor": "disabled",
                    "remote": "required",
                }
            },
        ),
        "IDLE_HOT": ModeProfile(
            name="IDLE_HOT",
            enabled_cameras=("rgb",),
            camera_overrides={"rgb": idle_hot_rgb},
            predictor_enabled=False,
            predictor_model=None,
            remote=_default_remote_profile(enabled=False),
            preview=PreviewProfile(enabled=True, sink_name="opencv"),
            release_cooldown_s=5.0,
        ),
    }
