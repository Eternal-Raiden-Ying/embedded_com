#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ._fast_gst_camera import FastGstCameraBase


class IRCamera(FastGstCameraBase):
    def __init__(
        self,
        device: str = "/dev/video4",
        in_w: int = 640,
        in_h: int = 480,
        out_w: int = 640,
        out_h: int = 480,
        fps: int = 30,
        format: str = "GRAY8",
        in_format: str = "GREY",
        flip_h: bool = False,
        flip_v: bool = False,
        rotate: int = 0,
        crop_x: int = 0,
        crop_y: int = 0,
        crop_w: int = 0,
        crop_h: int = 0,
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
