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
        format: str = "BGR",
        in_format: str = "GRAY8",
        flip_h: bool = False,
        flip_v: bool = False,
        rotate: int = 0,
        crop_x: int = 0,
        crop_y: int = 0,
        crop_w: int = 0,
        crop_h: int = 0,
    ):
        tried = []
        in_formats = []
        for value in (in_format, "GRAY8", "UYVY"):
            value = str(value).strip().upper()
            if value and value not in in_formats:
                in_formats.append(value)
        out_formats = []
        for value in (format, "BGR", "RGB"):
            value = str(value).strip().upper()
            if value and value not in out_formats:
                out_formats.append(value)

        last_error = None
        for candidate_in in in_formats:
            for candidate_out in out_formats:
                try:
                    super().__init__(
                        device=device,
                        in_w=in_w,
                        in_h=in_h,
                        out_w=out_w,
                        out_h=out_h,
                        fps=fps,
                        in_format=candidate_in,
                        format=candidate_out,
                        flip_h=flip_h,
                        flip_v=flip_v,
                        rotate=rotate,
                        crop_x=crop_x,
                        crop_y=crop_y,
                        crop_w=crop_w,
                        crop_h=crop_h,
                    )
                    self.in_format = candidate_in
                    self.format = candidate_out
                    return
                except Exception as exc:
                    last_error = exc
                    tried.append(f"{candidate_in}->{candidate_out}: {exc}")
        raise RuntimeError("IR camera init failed | " + " | ".join(tried)) from last_error
