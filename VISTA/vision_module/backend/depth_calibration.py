#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class DepthIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 0.001
    source: str = "calib_json_fallback"
    profile_info: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "depth_scale": float(self.depth_scale),
            "source": str(self.source or ""),
            "profile_info": str(self.profile_info or ""),
        }


def depth_intrinsics_from_dict(value: Any) -> Optional[DepthIntrinsics]:
    if isinstance(value, DepthIntrinsics):
        return value
    if not isinstance(value, dict):
        return None
    try:
        return DepthIntrinsics(
            width=int(value.get("width")),
            height=int(value.get("height")),
            fx=float(value.get("fx")),
            fy=float(value.get("fy")),
            cx=float(value.get("cx", value.get("ppx"))),
            cy=float(value.get("cy", value.get("ppy"))),
            depth_scale=float(value.get("depth_scale", 0.001)),
            source=str(value.get("source") or "runtime_profile"),
            profile_info=str(value.get("profile_info") or value.get("stream_name") or ""),
        )
    except Exception:
        return None


def depth_intrinsics_from_rs_profile(profile: Any, *, depth_scale: float = 0.001, source: str) -> Optional[DepthIntrinsics]:
    try:
        video_profile = profile.as_video_stream_profile()
        intr = video_profile.get_intrinsics()
        try:
            stream_name = str(profile.stream_type())
            fmt = str(profile.format())
            fps = int(video_profile.fps())
            profile_info = f"{stream_name} {fmt} {int(intr.width)}x{int(intr.height)}@{fps}"
        except Exception:
            profile_info = f"{int(intr.width)}x{int(intr.height)}"
        return DepthIntrinsics(
            width=int(intr.width),
            height=int(intr.height),
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.ppx),
            cy=float(intr.ppy),
            depth_scale=float(depth_scale or 0.001),
            source=str(source or "runtime_profile"),
            profile_info=profile_info,
        )
    except Exception:
        return None


def depth_intrinsics_payload(value: Any) -> Optional[Dict[str, Any]]:
    intr = depth_intrinsics_from_dict(value)
    return intr.to_dict() if intr is not None else None
