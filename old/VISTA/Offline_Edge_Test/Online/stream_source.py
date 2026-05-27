#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyrealsense2 as rs

try:
    from .schema import RealSenseConfig
except ImportError:
    from schema import RealSenseConfig


class RealSenseStreamSource:
    def __init__(self, cfg: RealSenseConfig, logger=None):
        self.cfg = cfg
        self.log = logger
        self.pipeline: Optional[rs.pipeline] = None
        self.align = None
        self.stream_profiles: List[Dict] = []
        self._is_bag = bool(str(cfg.bag_path or "").strip())

    def _log(self, msg: str):
        if self.log is not None:
            try:
                self.log.info(msg)
            except Exception:
                pass

    def start(self):
        pipeline = rs.pipeline()
        config = rs.config()
        if self._is_bag:
            bag_path = Path(self.cfg.bag_path).expanduser().resolve()
            rs.config.enable_device_from_file(config, str(bag_path), repeat_playback=False)
        else:
            if self.cfg.depth_enabled:
                config.enable_stream(rs.stream.depth, int(self.cfg.depth_width), int(self.cfg.depth_height), rs.format.z16, int(self.cfg.depth_fps))
            if self.cfg.color_enabled:
                config.enable_stream(rs.stream.color, int(self.cfg.color_width), int(self.cfg.color_height), rs.format.rgb8, int(self.cfg.color_fps))
        profile = pipeline.start(config)
        device = profile.get_device()
        if self._is_bag:
            device.as_playback().set_real_time(False)
        self.stream_profiles = []
        for p in profile.get_streams():
            vp = p.as_video_stream_profile()
            self.stream_profiles.append({
                "stream_name": str(p.stream_type()),
                "format": str(p.format()),
                "width": int(vp.width()),
                "height": int(vp.height()),
                "fps": int(vp.fps()),
            })
        self.pipeline = pipeline
        if self.cfg.align_to_color and any(item["stream_name"] == "stream.color" for item in self.stream_profiles):
            self.align = rs.align(rs.stream.color)
        else:
            self.align = None
        self._log(f"RealSense source started: {self.stream_profiles}")

    def stop(self):
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None

    def read(self, timeout_ms: int = 3000):
        if self.pipeline is None:
            return None
        frames = self.pipeline.wait_for_frames(int(timeout_ms))
        if self.align is not None:
            frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame and not color_frame:
            return None
        return {
            "ts_ms": float(frames.get_timestamp()),
            "depth": (np.asanyarray(depth_frame.get_data()) if depth_frame else None),
            "color": (np.asanyarray(color_frame.get_data()) if color_frame else None),
        }
