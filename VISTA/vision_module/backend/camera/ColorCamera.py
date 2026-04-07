#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ._fast_gst_camera import FastGstCameraBase


class ColorCamera(FastGstCameraBase):
    def __init__(
        self,
        device: str = "/dev/video0",
        in_w: int = 1280,
        in_h: int = 720,
        out_w: int = 640,
        out_h: int = 640,
        fps: int = 30,
        format: str = "RGB",
        in_format: str = "YUY2",
        flip_h: bool = False,
        flip_v: bool = False,
        rotate: int = 0,
        crop_x: int = 0,
        crop_y: int = 0,
        crop_w: int = 0,
        crop_h: int = 0,
        auto_exposure: bool = None,
        exposure: int = None,
        brightness: int = None,
    ):
        super().__init__(
            device=device,
            in_w=in_w,
            in_h=in_h,
            out_w=out_w,
            out_h=out_h,
            fps=fps,
            in_format=in_format,
            format=format,
            flip_h=flip_h,
            flip_v=flip_v,
            rotate=rotate,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_w=crop_w,
            crop_h=crop_h,
        )
        if auto_exposure is not None:
            self.set_auto_exposure(auto_exposure)
        if exposure is not None:
            self.set_exposure(exposure)
        if brightness is not None:
            self.set_brightness(brightness)

    def set_auto_exposure(self, enable: bool):
        self._v4l2_set_ctrl("exposure_auto", 3 if enable else 1)

    def set_exposure(self, value: int):
        self._v4l2_set_ctrl("exposure_absolute", value)

    def set_brightness(self, value: int):
        self._v4l2_set_ctrl("brightness", value)
