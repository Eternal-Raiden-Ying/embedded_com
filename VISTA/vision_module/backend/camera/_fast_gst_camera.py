#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import subprocess
import sys


logger = logging.getLogger("vision.camera")

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

try:
    import fast_cam
except ImportError as e:
    raise ImportError(
        f"Failed to import fast_cam from {_CURRENT_DIR}: {e}\n"
        "Build the camera backend in backend/camera/csrc and copy the generated fast_cam module here."
    )


class FastGstCameraBase:
    def __init__(
        self,
        *,
        device: str,
        in_w: int,
        in_h: int,
        out_w: int,
        out_h: int,
        fps: int,
        in_format: str,
        format: str,
        flip_h: bool = False,
        flip_v: bool = False,
        rotate: int = 0,
        crop_x: int = 0,
        crop_y: int = 0,
        crop_w: int = 0,
        crop_h: int = 0,
    ):
        self.device = device
        self.in_format = in_format
        self.format = format
        self._cam = fast_cam.Camera(
            device,
            in_w,
            in_h,
            out_w,
            out_h,
            fps,
            in_format,
            format,
            flip_h,
            flip_v,
            rotate,
            crop_x,
            crop_y,
            crop_w,
            crop_h,
        )

    def _v4l2_set_ctrl(self, ctrl_name: str, value: int):
        try:
            subprocess.run(
                ["v4l2-ctl", "-d", self.device, "-c", f"{ctrl_name}={value}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logger.warning("device %s does not support ctrl %s: %s", self.device, ctrl_name, e)
        except FileNotFoundError:
            logger.error("v4l2-ctl not found; install v4l2-utils")

    def read_frame(self):
        if self._cam is None:
            raise RuntimeError("camera released or not initialized")
        return self._cam.read_frame()

    def release(self):
        if self._cam is not None:
            del self._cam
            self._cam = None
            logger.info("camera released: %s", self.device)

    def __del__(self):
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
