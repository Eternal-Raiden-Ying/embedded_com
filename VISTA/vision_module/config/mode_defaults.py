#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Any, Dict, Optional

from ..backend.mode_profiles import ModeProfile, PreviewProfile, RemoteProfile, TableEdgeProfile


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _as_camera_tuple(value: Any, default) -> tuple:
    if value is None:
        return tuple(default or ())
    if isinstance(value, str):
        return tuple(item.strip() for item in value.replace(";", ",").split(",") if item.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return tuple(default or ())



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


def _default_remote_profile(*, enabled: bool, require_depth: bool = False,
                             kind: str = "loop", action: str = "",
                             max_retries: int = 1) -> RemoteProfile:
    return RemoteProfile(
        enabled=bool(enabled),
        kind=str(kind or "loop").strip().lower() or "loop",
        action=str(action or "").strip().lower(),
        max_retries=int(max_retries),
        base_url=str(os.getenv("VISION_REMOTE_BASE_URL", "")).strip() or None,
        require_depth=bool(require_depth),
        rgb_encoding=str(os.getenv("VISION_REMOTE_RGB_ENCODING", "jpeg")).strip().lower() or "jpeg",
        depth_encoding=str(os.getenv("VISION_REMOTE_DEPTH_ENCODING", "png")).strip().lower() or "png",
        rgb_quality=int(os.getenv("VISION_REMOTE_RGB_QUALITY", "90") or 90),
        depth_compression=int(os.getenv("VISION_REMOTE_DEPTH_COMPRESSION", "3") or 3),
    )


def build_default_mode_profiles(active_model: str, cfg: Optional[Any] = None) -> Dict[str, ModeProfile]:
    """Build the initial mode profile set for VISTA."""
    mode_cfg = dict(getattr(cfg, "mode_profiles", {}) or {})

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
    depth_overrides = _camera_overrides_for(cfg, ("depth",))
    grasp_remote_cameras = {
        "rgb": grasp_remote_rgb,
        **depth_overrides,
    }
    preview_layout_defaults = {
        "INIT": "rgb_minimal",
        "SILENT": "rgb_minimal",
        "IDLE": "rgb_minimal",
        "FIND_OBJECT": "rgb_yolo_edge_overlay",
        "FIND_EDGE": "rgb_depth_edge",
        "FIND_TABLE": "rgb_yolo_overlay",
        "MICRO_ADJUST": "rgb_minimal",
        "GRASP_REMOTE_INIT": "rgb_minimal",
        "GRASP_REMOTE": "rgb_depth_edge",
        "IDLE_HOT": "rgb_hot_preview",
    }
    preview_layouts = {
        str(k).strip().upper(): str(v).strip()
        for k, v in dict(getattr(getattr(cfg, "preview", None), "mode_layouts", preview_layout_defaults) or preview_layout_defaults).items()
    }

    def preview_profile(mode: str, *, enabled: bool, sink_name: str = "opencv") -> PreviewProfile:
        mode_name = str(mode or "IDLE").strip().upper() or "IDLE"
        preview_cfg = getattr(cfg, "preview", None)
        return PreviewProfile(
            enabled=bool(enabled),
            sink_name=sink_name,
            metadata={
                "layout": preview_layouts.get(mode_name, preview_layouts.get("IDLE", "rgb_minimal")),
                "mode_layouts": dict(preview_layouts),
                "debug_four_panel_in_track_local": bool(getattr(preview_cfg, "debug_four_panel_in_track_local", False)),
                "show_edge_overlay_in_track_local": bool(getattr(preview_cfg, "show_edge_overlay_in_track_local", True)),
                "show_age_ms": bool(getattr(preview_cfg, "show_age_ms", True)),
                "clear_overlay_on_mode_switch": bool(getattr(preview_cfg, "clear_overlay_on_mode_switch", True)),
            },
        )

    profiles = {
        "INIT": ModeProfile(
            name="INIT",
            enabled_cameras=(),
            camera_overrides={},
            predictor_enabled=False,
            predictor_model=None,
            remote=_default_remote_profile(enabled=True, require_depth=False,
                                             kind="task", action="init", max_retries=3),
            preview=preview_profile("INIT", enabled=False, sink_name="null"),
            release_cooldown_s=0.0,
            metadata={"contract": {"stage": "INIT", "remote": "required"}},
        ),
        "SILENT": ModeProfile(
            name="SILENT",
            enabled_cameras=(),
            camera_overrides={},
            predictor_enabled=False,
            predictor_model=None,
            remote=_default_remote_profile(enabled=False),
            preview=preview_profile("SILENT", enabled=False, sink_name="null"),
            release_cooldown_s=0.0,
            metadata={"contract": {"stage": "SILENT"}},
        ),
        "FIND_OBJECT": ModeProfile(
            name="FIND_OBJECT",
            enabled_cameras=("rgb", "depth"),
            camera_overrides={"rgb": track_local_rgb},
            predictor_enabled=True,
            predictor_model=active_model,
            remote=_default_remote_profile(enabled=False),
            preview=preview_profile("FIND_OBJECT", enabled=True),
            table_edge=TableEdgeProfile(
                enabled=True,
                detector_mode="lightweight",
                update_hz=5.0,
            ),
            release_cooldown_s=2.0,
            metadata={
                "contract": {
                    "cameras": ["rgb", "depth"],
                    "predictor": "required",
                    "table_edge": "required",
                    "remote": "disabled",
                    "perception": ["target_obs", "table_edge_obs"],
                },
            },
        ),
        "FIND_EDGE": ModeProfile(
            name="FIND_EDGE",
            enabled_cameras=("rgb", "depth"),
            camera_overrides={"rgb": track_local_rgb},
            predictor_enabled=True,
            predictor_model=active_model,
            remote=_default_remote_profile(enabled=False),
            preview=preview_profile("FIND_EDGE", enabled=True),
            table_edge=TableEdgeProfile(
                enabled=True,
                detector_mode="fast_plane_only",
                update_hz=10.0,
            ),
            release_cooldown_s=2.0,
            metadata={
                "contract": {
                    "cameras": ["rgb", "depth"],
                    "predictor": "required",
                    "table_edge": "required",
                    "remote": "disabled",
                    "perception": ["local_perception", "table_edge_obs"],
                },
            },
        ),
        "FIND_TABLE": ModeProfile(
            name="FIND_TABLE",
            enabled_cameras=("rgb",),
            camera_overrides={"rgb": track_local_rgb},
            predictor_enabled=True,
            predictor_model=active_model,
            remote=_default_remote_profile(enabled=False),
            preview=preview_profile("FIND_TABLE", enabled=True),
            table_edge=TableEdgeProfile(
                enabled=False,
            ),
            release_cooldown_s=2.0,
            metadata={
                "contract": {
                    "cameras": ["rgb"],
                    "predictor": "required",
                    "table_edge": "disabled",
                    "remote": "disabled",
                    "perception": ["target_obs"],
                },
            },
        ),
        "MICRO_ADJUST": ModeProfile(
            name="MICRO_ADJUST",
            enabled_cameras=(),
            camera_overrides={},
            predictor_enabled=False,
            predictor_model=None,
            remote=_default_remote_profile(enabled=False),
            preview=preview_profile("MICRO_ADJUST", enabled=False, sink_name="null"),
            release_cooldown_s=0.0,
            metadata={"contract": {"interaction": "MOVE_HINT"}},
        ),
        "GRASP_REMOTE_INIT": ModeProfile(
            name="GRASP_REMOTE_INIT",
            enabled_cameras=(),
            camera_overrides={},
            predictor_enabled=False,
            predictor_model=None,
            remote=_default_remote_profile(enabled=True, require_depth=False,
                                             kind="task", action="init", max_retries=3),
            preview=preview_profile("GRASP_REMOTE_INIT", enabled=True),
            release_cooldown_s=3.0,
            metadata={
                "contract": {
                    "cameras": [],
                    "predictor": "disabled",
                    "remote": "required",
                    "result": "remote_result",
                }
            },
        ),
        "GRASP_REMOTE": ModeProfile(
            name="GRASP_REMOTE",
            enabled_cameras=("rgb", "depth"),
            camera_overrides=grasp_remote_cameras,
            predictor_enabled=False,
            predictor_model=None,
            remote=_default_remote_profile(enabled=True, require_depth=True,
                                             kind="task", action="predict", max_retries=1),
            preview=preview_profile("GRASP_REMOTE", enabled=True),
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
            camera_overrides={"rgb": idle_hot_rgb},
            predictor_enabled=False,
            predictor_model=None,
            remote=_default_remote_profile(enabled=False),
            preview=preview_profile("IDLE_HOT", enabled=True),
            release_cooldown_s=5.0,
            metadata={"contract": {"stage": "IDLE_HOT"}},
        ),
    }

    for mode_name, section_raw in dict(mode_cfg.get("modes") or {}).items():
        mode_key = str(mode_name or "").strip().upper()
        profile = profiles.get(mode_key)
        if profile is None or not isinstance(section_raw, dict):
            continue
        section = dict(section_raw)
        if "enabled_cameras" in section:
            profile.enabled_cameras = _as_camera_tuple(section.get("enabled_cameras"), profile.enabled_cameras)
        if "predictor_enabled" in section:
            profile.predictor_enabled = bool(section.get("predictor_enabled"))
        if "predictor_model" in section:
            predictor_model = section.get("predictor_model")
            profile.predictor_model = str(predictor_model).strip() if predictor_model else None
        if "release_cooldown_s" in section and section.get("release_cooldown_s") is not None:
            profile.release_cooldown_s = float(section.get("release_cooldown_s"))
        if "loop_hz" in section and section.get("loop_hz") is not None:
            profile.loop_hz = float(section.get("loop_hz"))
        if "send_hz" in section and section.get("send_hz") is not None:
            profile.send_hz = float(section.get("send_hz"))
        if "preview_enabled" in section:
            profile.preview.enabled = bool(section.get("preview_enabled"))
        if "preview_layout" in section and section.get("preview_layout"):
            profile.preview.metadata["layout"] = str(section.get("preview_layout")).strip()

        table_edge_section = section.get("table_edge")
        if isinstance(table_edge_section, dict):
            te = dict(table_edge_section)
            if "enabled" in te:
                profile.table_edge.enabled = bool(te.get("enabled"))
            if "detector_mode" in te:
                profile.table_edge.detector_mode = str(te.get("detector_mode"))
            if "update_hz" in te and te.get("update_hz") is not None:
                profile.table_edge.update_hz = float(te.get("update_hz"))
            if "light_stride" in te and te.get("light_stride") is not None:
                profile.table_edge.light_stride = int(te.get("light_stride"))
            if "fast_plane_stride" in te and te.get("fast_plane_stride") is not None:
                profile.table_edge.fast_plane_stride = int(te.get("fast_plane_stride"))
            if "require_yolo_confirm" in te:
                profile.table_edge.require_yolo_confirm = bool(te.get("require_yolo_confirm"))
            if "static_roi_enabled" in te:
                profile.table_edge.static_roi_enabled = bool(te.get("static_roi_enabled"))

        camera_overrides = dict(section.get("camera_overrides") or {})
        for camera_name in ("rgb", "depth", "grey"):
            camera_section = section.get(camera_name)
            if isinstance(camera_section, dict):
                camera_overrides.setdefault(camera_name, {}).update(camera_section)
        for camera_name, updates in camera_overrides.items():
            if not isinstance(updates, dict):
                continue
            current = dict(profile.camera_overrides.get(str(camera_name), {}))
            current.update({str(k): v for k, v in updates.items() if v is not None})
            profile.camera_overrides[str(camera_name)] = current

    return profiles
