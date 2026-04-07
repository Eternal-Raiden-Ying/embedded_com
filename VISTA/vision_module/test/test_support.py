#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import platform
from typing import Tuple


def resolve_backend() -> str:
    backend = str(os.environ.get("VISTA_TEST_BACKEND", os.environ.get("VISTA_BACKEND", "auto"))).strip().lower()
    if backend in {"mock", "real", "auto"}:
        return backend
    return "auto"


def apply_backend_env() -> str:
    backend = resolve_backend()
    os.environ["VISTA_BACKEND"] = backend
    if backend == "mock":
        os.environ["ENV"] = "mock"
    elif backend == "real":
        os.environ["ENV"] = "prod"
    else:
        os.environ.pop("ENV", None)
    return backend


def running_on_windows() -> bool:
    return platform.system().lower().startswith("win")


def describe_backend(backend: str, obj) -> str:
    if obj is None:
        return backend.upper()
    name = type(obj).__name__
    if "Mock" in name:
        return "MOCK"
    return "REAL"


def build_camera_instance(stream_name: str, cfg):
    from vision_module.backend.camera import HardwareCamera, RealSenseDepthCamera

    if stream_name == "depth":
        return RealSenseDepthCamera(
            width=getattr(cfg, "width", 424),
            height=getattr(cfg, "height", 240),
            fps=getattr(cfg, "fps", 15),
        )

    source = getattr(cfg, "source", "0")
    device = f"/dev/video{source}" if str(source).isdigit() else str(source)
    kwargs = {
        "device": device,
        "fps": getattr(cfg, "fps", 30),
    }
    if hasattr(cfg, "format"):
        kwargs["format"] = getattr(cfg, "format")
    if hasattr(cfg, "in_format"):
        kwargs["in_format"] = getattr(cfg, "in_format")
    if hasattr(cfg, "in_w"):
        kwargs["in_w"] = getattr(cfg, "in_w")
    if hasattr(cfg, "in_h"):
        kwargs["in_h"] = getattr(cfg, "in_h")
    if hasattr(cfg, "out_w"):
        kwargs["out_w"] = getattr(cfg, "out_w")
    if hasattr(cfg, "out_h"):
        kwargs["out_h"] = getattr(cfg, "out_h")
    if hasattr(cfg, "crop_x"):
        kwargs["crop_x"] = getattr(cfg, "crop_x")
    if hasattr(cfg, "crop_y"):
        kwargs["crop_y"] = getattr(cfg, "crop_y")
    if hasattr(cfg, "crop_w"):
        kwargs["crop_w"] = getattr(cfg, "crop_w")
    if hasattr(cfg, "crop_h"):
        kwargs["crop_h"] = getattr(cfg, "crop_h")
    return HardwareCamera(**kwargs)


def safe_release(obj):
    if obj is None:
        return
    try:
        obj.release()
    except Exception:
        pass
