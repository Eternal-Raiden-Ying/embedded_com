#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import threading
import time
from threading import RLock
from typing import Any, Callable, Dict, Optional

import numpy as np

try:
    import aidcv as cv2
except ImportError:
    import cv2

from .camera import ColorCamera, IRCamera, RealSenseDepthCamera, camera_backend_status
from .depth_calibration import depth_intrinsics_from_rs_profile
from ..config.schema import VisionServiceConfig


CapabilitySink = Optional[Callable[[str, str, Dict[str, Any]], None]]


@dataclass(frozen=True)
class CameraSpec:
    name: str
    params: tuple


class _SharedRgbdStreamProxy:
    def __init__(self, owner: "CameraManager", stream_name: str):
        self._owner = owner
        self._stream_name = str(stream_name)

    def read_frame(self):
        return self._owner._read_shared_rgbd_stream(self._stream_name)

    def release(self) -> None:
        pass

    def get_depth_intrinsics(self):
        if self._stream_name != "depth":
            return None
        session = self._owner._shared_rgbd
        getter = getattr(session, "get_depth_intrinsics", None)
        if callable(getter):
            return getter()
        return None


class _SharedRealSenseRgbdSession:
    """Read RealSense depth and color together so the color stream is not starved."""

    def __init__(self, depth_params: Dict[str, Any], rgb_params: Dict[str, Any], logger: logging.Logger):
        import pyrealsense2 as rs

        self.rs = rs
        self.log = logger
        self.depth_params = dict(depth_params or {})
        self.rgb_params = dict(rgb_params or {})
        self.depth_sensor = None
        self.color_sensor = None
        self.depth_queue = rs.frame_queue(1)
        self.color_queue = rs.frame_queue(1)
        self.depth_profile = None
        self.color_profile = None
        self.color_w = 0
        self.color_h = 0
        self.output_format = str(self.rgb_params.get("format") or "BGR").strip().upper()

        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) <= 0:
            raise RuntimeError("RealSense device not found")
        dev = devices[0]
        self.depth_sensor = dev.first_depth_sensor()
        for sensor in dev.query_sensors():
            try:
                if sensor.get_info(rs.camera_info.name) == "RGB Camera":
                    self.color_sensor = sensor
                    break
            except Exception:
                continue
        if self.color_sensor is None:
            raise RuntimeError("RealSense RGB Camera sensor not found")

        depth_w = int(self.depth_params.get("width") or 424)
        depth_h = int(self.depth_params.get("height") or 240)
        depth_fps = int(self.depth_params.get("fps") or 15)
        color_w = int(self.rgb_params.get("in_w") or 1280)
        color_h = int(self.rgb_params.get("in_h") or 720)
        requested_color_fps = int(self.rgb_params.get("fps") or depth_fps or 15)
        self.depth_profile = self._find_profile(self.depth_sensor, rs.stream.depth, rs.format.z16, depth_w, depth_h, depth_fps)
        self.color_profile = self._find_color_yuyv_profile(color_w, color_h, requested_color_fps, depth_fps)
        self.depth_scale = float(self.depth_sensor.get_depth_scale())
        self.depth_intrinsics = depth_intrinsics_from_rs_profile(
            self.depth_profile,
            depth_scale=self.depth_scale,
            source="realsense_profile",
        )
        vp = self.color_profile.as_video_stream_profile()
        self.color_w = int(vp.width())
        self.color_h = int(vp.height())
        self.color_intrinsics = self._intrinsics_dict(vp, source="realsense_color_profile")

        self.depth_sensor.open(self.depth_profile)
        self.color_sensor.open(self.color_profile)
        self._apply_color_controls()
        self.depth_sensor.start(self.depth_queue)
        self.color_sensor.start(self.color_queue)
        self.log.info(
            "shared realsense rgbd started: depth=%sx%s@%s color=%sx%s@%s yuyv calib_source=%s fx=%.3f fy=%.3f cx=%.3f cy=%.3f depth_scale=%.6f",
            depth_w,
            depth_h,
            depth_fps,
            self.color_w,
            self.color_h,
            int(vp.fps()),
            getattr(self.depth_intrinsics, "source", "unavailable"),
            float(getattr(self.depth_intrinsics, "fx", 0.0) or 0.0),
            float(getattr(self.depth_intrinsics, "fy", 0.0) or 0.0),
            float(getattr(self.depth_intrinsics, "cx", 0.0) or 0.0),
            float(getattr(self.depth_intrinsics, "cy", 0.0) or 0.0),
            self.depth_scale,
        )

    def _rs_option(self, name: str):
        return getattr(getattr(self.rs, "option", None), name, None)

    def _get_option_best_effort(self, sensor, option_name: str):
        option = self._rs_option(option_name)
        if sensor is None or option is None:
            return None
        try:
            if not sensor.supports(option):
                return None
        except Exception:
            return None
        try:
            return sensor.get_option(option)
        except Exception:
            return None

    def _set_option_best_effort(self, sensor, option_name: str, value: float) -> bool:
        option = self._rs_option(option_name)
        if sensor is None or option is None:
            return False
        try:
            if not sensor.supports(option):
                return False
        except Exception:
            return False
        try:
            sensor.set_option(option, float(value))
            return True
        except Exception as exc:
            self.log.warning("realsense color option set failed: option=%s value=%s error=%s", option_name, value, exc)
            return False

    def _apply_color_controls(self) -> None:
        try:
            ae_requested = bool(self.rgb_params.get("auto_exposure", True))
            ae_applied = self._set_option_best_effort(self.color_sensor, "enable_auto_exposure", 1.0 if ae_requested else 0.0)
            if not ae_requested:
                exposure = self.rgb_params.get("exposure")
                if exposure is not None:
                    self._set_option_best_effort(self.color_sensor, "exposure", float(exposure))

            # Auto White Balance
            awb_requested = bool(self.rgb_params.get("auto_white_balance", True))
            awb_applied = self._set_option_best_effort(self.color_sensor, "enable_auto_white_balance", 1.0 if awb_requested else 0.0)

            brightness = self.rgb_params.get("brightness")
            if brightness is not None:
                self._set_option_best_effort(self.color_sensor, "brightness", float(brightness))

            controls = {
                "auto_exposure_requested": ae_requested,
                "auto_exposure_applied": bool(ae_applied),
                "auto_exposure_enabled": self._get_option_best_effort(self.color_sensor, "enable_auto_exposure"),
                "auto_white_balance_requested": awb_requested,
                "auto_white_balance_applied": bool(awb_applied),
                "auto_white_balance_enabled": self._get_option_best_effort(self.color_sensor, "enable_auto_white_balance"),
                "exposure": self._get_option_best_effort(self.color_sensor, "exposure"),
                "gain": self._get_option_best_effort(self.color_sensor, "gain"),
                "brightness": self._get_option_best_effort(self.color_sensor, "brightness"),
            }
            self.log.info("realsense_color_controls_applied %s", controls)
        except Exception as exc:
            self.log.warning("failed to apply realsense color controls defensively: %s", exc)

    @staticmethod
    def _intrinsics_dict(video_profile, *, source: str) -> Optional[Dict[str, Any]]:
        try:
            intr = video_profile.get_intrinsics()
        except Exception:
            return None
        return {
            "width": int(getattr(intr, "width", 0) or 0),
            "height": int(getattr(intr, "height", 0) or 0),
            "fx": float(getattr(intr, "fx", 0.0) or 0.0),
            "fy": float(getattr(intr, "fy", 0.0) or 0.0),
            "cx": float(getattr(intr, "ppx", 0.0) or 0.0),
            "cy": float(getattr(intr, "ppy", 0.0) or 0.0),
            "model": str(getattr(intr, "model", "") or ""),
            "coeffs": [float(v) for v in list(getattr(intr, "coeffs", []) or [])],
            "source": str(source or ""),
        }

    def _find_profile(self, sensor, stream, fmt, width: int, height: int, fps: int):
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() == stream and profile.format() == fmt:
                vp = profile.as_video_stream_profile()
                if int(vp.width()) == int(width) and int(vp.height()) == int(height) and int(vp.fps()) == int(fps):
                    return profile
        raise RuntimeError(f"RealSense profile not found: stream={stream} format={fmt} {width}x{height}@{fps}")

    def _find_color_yuyv_profile(self, width: int, height: int, requested_fps: int, depth_fps: int):
        candidates = []
        for profile in self.color_sensor.get_stream_profiles():
            if profile.stream_type() == self.rs.stream.color and profile.format() == self.rs.format.yuyv:
                vp = profile.as_video_stream_profile()
                if int(vp.width()) == int(width) and int(vp.height()) == int(height):
                    candidates.append((int(vp.fps()), profile))
        if not candidates:
            raise RuntimeError(f"RealSense YUYV color profile not found: {width}x{height}")
        preferred = [int(requested_fps), int(depth_fps), 15, 30, 6]
        by_fps = {fps: profile for fps, profile in candidates}
        for fps in preferred:
            if fps in by_fps:
                return by_fps[fps]
        return sorted(candidates, key=lambda item: abs(item[0] - int(depth_fps or requested_fps or 15)))[0][1]

    def read_bundle(self) -> Dict[str, Any]:
        depth_frame = None
        color_frame = None
        try:
            depth_frame = self.depth_queue.wait_for_frame(1000)
        except Exception:
            depth_frame = None
        try:
            color_frame = self.color_queue.wait_for_frame(1000)
        except Exception:
            color_frame = None
        depth = np.asanyarray(depth_frame.get_data()).copy() if depth_frame is not None else np.array([])
        color = self._convert_color(color_frame) if color_frame is not None else np.array([])
        # Query auto controls defensively
        camera_auto_exposure = None
        camera_auto_white_balance = None
        try:
            ae = self._get_option_best_effort(self.color_sensor, "enable_auto_exposure")
            if ae is not None:
                camera_auto_exposure = bool(ae)
        except Exception:
            pass
        try:
            awb = self._get_option_best_effort(self.color_sensor, "enable_auto_white_balance")
            if awb is not None:
                camera_auto_white_balance = bool(awb)
        except Exception:
            pass

        return {
            "depth": depth,
            "rgb": color,
            "depth_intrinsics": self.get_depth_intrinsics(),
            "color_intrinsics": self.get_color_intrinsics(),
            "depth_scale": self.depth_scale,
            "depth_unit": "m",
            "depth_aligned_to_color": "best_effort_true",
            "camera_auto_exposure": camera_auto_exposure,
            "camera_auto_white_balance": camera_auto_white_balance,
        }

    def get_depth_intrinsics(self):
        return self.depth_intrinsics.to_dict() if self.depth_intrinsics is not None else None

    def get_color_intrinsics(self):
        return dict(self.color_intrinsics or {}) if self.color_intrinsics is not None else None

    def _convert_color(self, color_frame) -> np.ndarray:
        raw = np.asanyarray(color_frame.get_data())
        if raw.size <= 0:
            return np.array([])
        yuyv = raw.view(np.uint8).reshape((self.color_h, self.color_w, 2))
        bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
        crop_x = int(self.rgb_params.get("crop_x") or 0)
        crop_y = int(self.rgb_params.get("crop_y") or 0)
        crop_w = int(self.rgb_params.get("crop_w") or 0)
        crop_h = int(self.rgb_params.get("crop_h") or 0)
        if crop_w > 0 and crop_h > 0:
            x0 = max(0, min(bgr.shape[1] - 1, crop_x))
            y0 = max(0, min(bgr.shape[0] - 1, crop_y))
            x1 = max(x0 + 1, min(bgr.shape[1], x0 + crop_w))
            y1 = max(y0 + 1, min(bgr.shape[0], y0 + crop_h))
            bgr = bgr[y0:y1, x0:x1]
        out_w = int(self.rgb_params.get("out_w") or bgr.shape[1])
        out_h = int(self.rgb_params.get("out_h") or bgr.shape[0])
        if out_w > 0 and out_h > 0 and (bgr.shape[1], bgr.shape[0]) != (out_w, out_h):
            bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        if self.output_format == "RGB":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return bgr

    def release(self) -> None:
        for sensor in (self.color_sensor, self.depth_sensor):
            if sensor is None:
                continue
            try:
                sensor.stop()
            except Exception:
                pass
            try:
                sensor.close()
            except Exception:
                pass
        self.log.info("shared realsense rgbd closed")


def _freeze_params(params: Dict[str, Any]) -> tuple:
    return tuple(sorted((str(k), repr(v)) for k, v in params.items()))


class CameraManager:
    """Own camera lifecycle, reuse, and reconfiguration."""

    def __init__(
        self,
        cfg: VisionServiceConfig,
        logger: Optional[logging.Logger] = None,
        capability_sink: CapabilitySink = None,
    ):
        self.cfg = cfg
        self.log = logger or logging.getLogger("vision.camera_manager")
        self._capability_sink = capability_sink
        self._lock = RLock()
        self.cams: Dict[str, Any] = {}
        self._specs: Dict[str, CameraSpec] = {}
        self._scheduler = None
        self._generation_getter = lambda: 0
        self._runtime_running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_stop = threading.Event()
        max_fps = int(getattr(getattr(self.cfg, "camera", None), "max_fps", 30) or 30)
        self._worker_interval_s = 1.0 / max(1, max_fps)
        self._last_frame_seq = 0
        self._backend_status = camera_backend_status()
        self._active_implementation = str(self._backend_status.get("resolved_backend") or "mock")
        self._params: Dict[str, Dict[str, Any]] = {}
        self._shared_rgbd: Optional[_SharedRealSenseRgbdSession] = None
        self._shared_rgbd_signature = None
        self._shared_rgbd_cycle_bundle: Optional[Dict[str, Any]] = None
        self._last_shape_log_key = ""

    @staticmethod
    def _shared_rgbd_enabled() -> bool:
        raw = os.getenv("VISTA_SHARED_REALSENSE_RGBD", "1")
        return str(raw or "").strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _is_shared_proxy(camera: Any) -> bool:
        return type(camera).__name__ == "_SharedRgbdStreamProxy"

    def _emit(self, action: str, resource_name: str, **fields: Any) -> None:
        if self._capability_sink is None:
            return
        try:
            self._capability_sink(str(action or "updated").strip().lower(), str(resource_name or ""), dict(fields or {}))
        except Exception:
            pass

    def _resolve_params(self, name: str, override: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        cam_cfg = self.cfg.camera.streams.get(name)
        if cam_cfg is None and not override:
            return None
        override = dict(override or {})

        def _pick(key: str, default=None):
            if key in override:
                return override[key]
            if cam_cfg is None:
                return default
            return getattr(cam_cfg, key, default)

        if name == "depth":
            return {
                "width": _pick("width"),
                "height": _pick("height"),
                "fps": _pick("fps"),
            }
        if name in {"ir", "grey"}:
            source = _pick("source")
            device = f"/dev/video{source}" if str(source).isdigit() else source
            return {
                "device": device,
                "in_format": _pick("in_format", "GRAY8"),
                "format": _pick("format", "BGR"),
                "fps": _pick("fps"),
                "in_w": _pick("in_w"),
                "in_h": _pick("in_h"),
                "out_w": _pick("out_w"),
                "out_h": _pick("out_h"),
                "crop_x": _pick("crop_x", 0),
                "crop_y": _pick("crop_y", 0),
                "crop_w": _pick("crop_w", 0),
                "crop_h": _pick("crop_h", 0),
            }
        source = _pick("source")
        device = f"/dev/video{source}" if str(source).isdigit() else source
        return {
            "device": device,
            "in_format": _pick("in_format", "YUY2"),
            "format": _pick("format"),
            "fps": _pick("fps"),
            "in_w": _pick("in_w"),
            "in_h": _pick("in_h"),
            "out_w": _pick("out_w"),
            "out_h": _pick("out_h"),
            "crop_x": _pick("crop_x"),
            "crop_y": _pick("crop_y"),
            "crop_w": _pick("crop_w"),
            "crop_h": _pick("crop_h"),
            "auto_exposure": _pick("auto_exposure"),
            "exposure": _pick("exposure"),
            "brightness": _pick("brightness"),
        }

    def _build_camera(self, name: str, params: Dict[str, Any]) -> Any:
        if name == "depth":
            return RealSenseDepthCamera(**params)
        if name in {"ir", "grey"}:
            return IRCamera(**params)
        return ColorCamera(**params)

    @staticmethod
    def _shape_hw_from_frame(frame: Any) -> Optional[list[int]]:
        shape = getattr(frame, "shape", None)
        if not isinstance(shape, tuple) or len(shape) < 2:
            return None
        try:
            h = int(shape[0])
            w = int(shape[1])
        except Exception:
            return None
        if h <= 0 or w <= 0:
            return None
        return [h, w]

    @staticmethod
    def _rgb_config_meta(params: Dict[str, Any]) -> Dict[str, Any]:
        values = dict(params or {})
        try:
            in_w = int(values.get("in_w") or 0)
            in_h = int(values.get("in_h") or 0)
        except Exception:
            in_w, in_h = 0, 0
        try:
            out_w = int(values.get("out_w") or in_w or 0)
            out_h = int(values.get("out_h") or in_h or 0)
        except Exception:
            out_w, out_h = in_w, in_h
        try:
            crop_x = int(values.get("crop_x") or 0)
            crop_y = int(values.get("crop_y") or 0)
            crop_w = int(values.get("crop_w") or 0)
            crop_h = int(values.get("crop_h") or 0)
        except Exception:
            crop_x, crop_y, crop_w, crop_h = 0, 0, 0, 0
        if crop_w <= 0:
            crop_w = in_w
        if crop_h <= 0:
            crop_h = in_h
        crop_x = max(0, crop_x)
        crop_y = max(0, crop_y)
        return {
            "rgb_native_shape": [in_h, in_w] if in_w > 0 and in_h > 0 else None,
            "rgb_crop_rect": [crop_x, crop_y, crop_w, crop_h],
            "rgb_output_shape_config": [out_h, out_w] if out_w > 0 and out_h > 0 else None,
            "rgb_config_source": "vision_params.yaml+mode_profile" if values else "default",
            "camera_color_frame_format": str(values.get("format") or "BGR").upper(),
        }

    def bind_runtime(self, scheduler, generation_getter=None) -> None:
        self._scheduler = scheduler
        if callable(generation_getter):
            self._generation_getter = generation_getter

    def start_runtime(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._runtime_running = True
        self._worker_stop.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, name="camera_manager.loop", daemon=True)
        self._worker_thread.start()

    def stop_runtime(self) -> None:
        self._runtime_running = False
        self._worker_stop.set()
        thread = self._worker_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._worker_thread = None

    def _publish_result(self, route: str, payload: Any) -> None:
        scheduler = self._scheduler
        if scheduler is None:
            return
        try:
            generation = int(self._generation_getter())
        except Exception:
            generation = 0
        try:
            scheduler.publish_result(route, payload, generation=generation)
        except Exception:
            pass

    def _worker_loop(self) -> None:
        while self._runtime_running and not self._worker_stop.is_set():
            self._ensure_shared_rgbd_if_needed()
            self._shared_rgbd_cycle_bundle = None
            active = self.iter_cameras()
            if not active:
                self._publish_result("frame_meta", {"has_frames": False, "cameras": []})
                self._worker_stop.wait(timeout=0.1)
                continue
            frame_bundle: Dict[str, Any] = {}
            for name, cam in active:
                try:
                    frame = cam.read_frame()
                except Exception as exc:
                    self.log.debug("camera read failed | name=%s error=%s", name, exc)
                    frame = None
                if frame is None or getattr(frame, "size", 0) <= 0:
                    continue
                try:
                    frame_bundle[name] = frame.copy()
                except Exception:
                    frame_bundle[name] = frame
                if name == "depth":
                    getter = getattr(cam, "get_depth_intrinsics", None)
                    if callable(getter):
                        try:
                            depth_intrinsics = getter()
                        except Exception:
                            depth_intrinsics = None
                        if depth_intrinsics:
                            frame_bundle["depth_intrinsics"] = depth_intrinsics
            if not frame_bundle:
                self._publish_result("frame_meta", {"has_frames": False, "cameras": []})
                self._worker_stop.wait(timeout=self._worker_interval_s)
                continue
            self._last_frame_seq += 1
            camera_names = sorted(frame_bundle.keys())
            frame_capture_ts = time.time()
            frame_bundle["camera_frame_seq"] = int(self._last_frame_seq)
            frame_bundle["frame_capture_ts"] = float(frame_capture_ts)
            frame_bundle["camera_frame_ts_ms"] = int(round(frame_capture_ts * 1000.0))
            rgb_shape = None
            rgb = frame_bundle.get("rgb")
            if rgb is not None:
                try:
                    rgb_shape = tuple(int(v) for v in rgb.shape)
                except Exception:
                    rgb_shape = None
            depth_shape_actual = self._shape_hw_from_frame(frame_bundle.get("depth"))
            rgb_shape_actual = list(rgb_shape[:2]) if isinstance(rgb_shape, tuple) and len(rgb_shape) >= 2 else None
            rgb_meta = self._rgb_config_meta(self._params.get("rgb") or {})
            frame_bundle.update(rgb_meta)
            if self._shared_rgbd is not None and self._shared_rgbd_cycle_bundle:
                for key in ("camera_auto_exposure", "camera_auto_white_balance"):
                    if key in self._shared_rgbd_cycle_bundle:
                        frame_bundle[key] = self._shared_rgbd_cycle_bundle[key]
            frame_bundle.setdefault("camera_auto_exposure", None)
            frame_bundle.setdefault("camera_auto_white_balance", None)
            frame_bundle["rgb_output_shape_actual"] = rgb_shape_actual
            frame_bundle["depth_shape_actual"] = depth_shape_actual
            log_key = (
                f"{rgb_meta.get('rgb_native_shape')}|{rgb_meta.get('rgb_crop_rect')}|"
                f"{rgb_meta.get('rgb_output_shape_config')}|{rgb_shape_actual}|{depth_shape_actual}|"
                f"{rgb_meta.get('rgb_config_source')}"
            )
            if log_key != self._last_shape_log_key:
                self._last_shape_log_key = log_key
                self.log.info(
                    "camera shape config | rgb_native_shape=%s rgb_crop_rect=%s rgb_output_shape_config=%s rgb_output_shape_actual=%s depth_shape_actual=%s rgb_config_source=%s",
                    rgb_meta.get("rgb_native_shape"),
                    rgb_meta.get("rgb_crop_rect"),
                    rgb_meta.get("rgb_output_shape_config"),
                    rgb_shape_actual,
                    depth_shape_actual,
                    rgb_meta.get("rgb_config_source"),
                )
            self._publish_result("camera_frames", frame_bundle)
            self._publish_result(
                "frame_meta",
                {
                    "has_frames": True,
                    "cameras": camera_names,
                    "rgb_shape": rgb_shape,
                    "rgb_shape_actual": rgb_shape_actual,
                    "rgb_output_shape_actual": rgb_shape_actual,
                    "depth_shape_actual": depth_shape_actual,
                    **rgb_meta,
                    "frame_seq": int(self._last_frame_seq),
                    "camera_frame_seq": int(self._last_frame_seq),
                    "camera_frame_ts_ms": int(round(frame_capture_ts * 1000.0)),
                },
            )
            self._worker_stop.wait(timeout=self._worker_interval_s)
            self._shared_rgbd_cycle_bundle = None

    def active_names(self) -> set:
        with self._lock:
            return set(self.cams.keys())

    def iter_cameras(self):
        with self._lock:
            return list(self.cams.items())

    def get_camera(self, name: str):
        with self._lock:
            return self.cams.get(name)

    def ensure_camera(self, name: str, override: Optional[Dict[str, Any]] = None) -> bool:
        params = self._resolve_params(name, override)
        if params is None:
            self.log.error("camera config not found: %s", name)
            return False
        target_spec = CameraSpec(name=name, params=_freeze_params(params))
        if name in {"rgb", "depth"} and self._attach_existing_shared_rgbd_proxy(name, params, target_spec):
            return True

        with self._lock:
            current = self._specs.get(name)
            if name in self.cams and current == target_spec:
                return False

            old_cam = self.cams.pop(name, None)
            self._specs.pop(name, None)
            self._params.pop(name, None)

        if old_cam is not None:
            try:
                old_cam.release()
            except Exception as exc:
                self.log.warning("camera release failed: %s", exc)
            self._emit("reconfigured", name)

        try:
            camera = self._build_camera(name, params)
        except Exception as exc:
            self.log.error("camera enable failed: %s | %s", name, exc)
            self._emit("enable_failed", name, error=str(exc))
            return False

        implementation = "mock" if type(camera).__name__ == "MockCamera" else str(self._backend_status.get("resolved_backend") or "real")

        with self._lock:
            self.cams[name] = camera
            self._specs[name] = target_spec
            self._params[name] = dict(params)
            self._active_implementation = implementation
        self.log.info("camera enabled: %s", name)
        self._emit(
            "enabled",
            name,
            params=params,
            implementation=implementation,
        )
        return True

    def disable_camera(self, name: str) -> bool:
        with self._lock:
            camera = self.cams.pop(name, None)
            self._specs.pop(name, None)
            self._params.pop(name, None)
        if name in {"rgb", "depth"}:
            self._release_or_restore_shared_rgbd_after_disable()
        if camera is None:
            return False
        try:
            camera.release()
        except Exception as exc:
            self.log.warning("camera release failed: %s", exc)
        self.log.info("camera disabled: %s", name)
        self._emit("disabled", name)
        return True

    def _shared_signature_for_params(self, rgb_params: Dict[str, Any], depth_params: Dict[str, Any]):
        return (
            tuple(sorted(dict(rgb_params or {}).items())),
            tuple(sorted(dict(depth_params or {}).items())),
        )

    def _attach_existing_shared_rgbd_proxy(self, name: str, params: Dict[str, Any], target_spec: CameraSpec) -> bool:
        if self._shared_rgbd is None:
            return False
        with self._lock:
            rgb_params = dict(self._params.get("rgb") or {})
            depth_params = dict(self._params.get("depth") or {})
            if name == "rgb":
                rgb_params = dict(params or {})
            if name == "depth":
                depth_params = dict(params or {})
            if not rgb_params or not depth_params:
                return False
            signature = self._shared_signature_for_params(rgb_params, depth_params)
            if signature != self._shared_rgbd_signature:
                return False
            current = self._specs.get(name)
            if name in self.cams and current == target_spec:
                return False
            old_cam = self.cams.get(name)
            if old_cam is not None and not self._is_shared_proxy(old_cam):
                return False
            self.cams[name] = _SharedRgbdStreamProxy(self, name)
            self._specs[name] = target_spec
            self._params[name] = dict(params or {})
            self._active_implementation = "realsense_shared_rgbd"
        self.log.info("camera attached to shared rgbd: %s", name)
        self._emit("enabled", name, params=params, implementation="realsense_shared_rgbd")
        return True

    def release_all(self) -> None:
        self.stop_runtime()
        self._release_shared_rgbd()
        with self._lock:
            names = list(self.cams.keys())
        for name in names:
            self.disable_camera(name)

    def _ensure_shared_rgbd_if_needed(self) -> None:
        if not self._shared_rgbd_enabled():
            return
        with self._lock:
            has_rgb_depth = {"rgb", "depth"}.issubset(set(self.cams.keys()))
            rgb_cam = self.cams.get("rgb")
            depth_cam = self.cams.get("depth")
            rgb_params = dict(self._params.get("rgb") or {})
            depth_params = dict(self._params.get("depth") or {})
        if not has_rgb_depth:
            if self._shared_rgbd is not None and (
                self._is_shared_proxy(rgb_cam) or self._is_shared_proxy(depth_cam)
            ):
                return
            self._release_shared_rgbd()
            return
        if type(rgb_cam).__name__ == "MockCamera" or type(depth_cam).__name__ == "MockCamera":
            return
        signature = (
            tuple(sorted(rgb_params.items())),
            tuple(sorted(depth_params.items())),
        )
        if self._shared_rgbd is not None and self._shared_rgbd_signature == signature:
            return
        self._release_shared_rgbd()
        old_cams = []
        with self._lock:
            for stream_name in ("rgb", "depth"):
                old_cam = self.cams.get(stream_name)
                if old_cam is not None and type(old_cam).__name__ != "_SharedRgbdStreamProxy":
                    old_cams.append((stream_name, old_cam))
        for _stream_name, old_cam in old_cams:
            try:
                old_cam.release()
            except Exception as exc:
                self.log.warning("camera release failed before shared rgbd: %s", exc)
        try:
            session = _SharedRealSenseRgbdSession(depth_params=depth_params, rgb_params=rgb_params, logger=self.log)
        except Exception as exc:
            high_res_depth_requested = bool(
                int(depth_params.get("width") or 0) >= 1280 and int(depth_params.get("height") or 0) >= 720
            )
            if high_res_depth_requested:
                self.log.error(
                    "shared realsense rgbd unavailable for requested high-res depth profile: depth=%sx%s@%s color=%sx%s@%s error=%s",
                    depth_params.get("width"),
                    depth_params.get("height"),
                    depth_params.get("fps"),
                    rgb_params.get("in_w"),
                    rgb_params.get("in_h"),
                    rgb_params.get("fps"),
                    exc,
                )
            else:
                self.log.warning("shared realsense rgbd unavailable: %s", exc)
            restored = {}
            for stream_name, stream_params in (("depth", depth_params), ("rgb", rgb_params)):
                try:
                    restored[stream_name] = self._build_camera(stream_name, stream_params)
                except Exception as restore_exc:
                    self.log.warning("camera restore failed after shared rgbd fallback | name=%s error=%s", stream_name, restore_exc)
            with self._lock:
                for stream_name, camera in restored.items():
                    self.cams[stream_name] = camera
            return
        with self._lock:
            for stream_name in ("rgb", "depth"):
                self.cams[stream_name] = _SharedRgbdStreamProxy(self, stream_name)
            self._shared_rgbd = session
            self._shared_rgbd_signature = signature
            self._active_implementation = "realsense_shared_rgbd"

    def _release_or_restore_shared_rgbd_after_disable(self) -> None:
        if self._shared_rgbd is None:
            self._restore_shared_rgbd_proxies()
            return
        with self._lock:
            has_remaining_proxy = any(
                self._is_shared_proxy(self.cams.get(stream_name))
                for stream_name in ("rgb", "depth")
            )
        if has_remaining_proxy:
            return
        self._release_shared_rgbd()
        self._restore_shared_rgbd_proxies()

    def _release_shared_rgbd(self) -> None:
        session = self._shared_rgbd
        self._shared_rgbd = None
        self._shared_rgbd_signature = None
        self._shared_rgbd_cycle_bundle = None
        if session is not None:
            try:
                session.release()
            except Exception:
                pass

    def _restore_shared_rgbd_proxies(self) -> None:
        restore_items = []
        with self._lock:
            for stream_name in ("rgb", "depth"):
                camera = self.cams.get(stream_name)
                if type(camera).__name__ != "_SharedRgbdStreamProxy":
                    continue
                params = dict(self._params.get(stream_name) or {})
                if params:
                    restore_items.append((stream_name, params))
        for stream_name, params in restore_items:
            try:
                camera = self._build_camera(stream_name, params)
            except Exception as exc:
                self.log.warning("camera restore failed after shared rgbd release | name=%s error=%s", stream_name, exc)
                continue
            with self._lock:
                self.cams[stream_name] = camera
            self.log.info("camera restored after shared rgbd release: %s", stream_name)

    def _read_shared_rgbd_stream(self, stream_name: str):
        session = self._shared_rgbd
        if session is None:
            return np.array([])
        if self._shared_rgbd_cycle_bundle is None:
            self._shared_rgbd_cycle_bundle = session.read_bundle()
        frame = (self._shared_rgbd_cycle_bundle or {}).get(stream_name)
        return frame if frame is not None else np.array([])

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled_cameras": sorted(self._specs.keys()),
                "camera_specs": sorted(self._specs.keys()),
                "runtime_running": bool(self._runtime_running),
                "last_frame_seq": int(self._last_frame_seq),
                "implementation": str(self._active_implementation or "real"),
                "backend_status": dict(self._backend_status or {}),
                "shared_rgbd_running": bool(self._shared_rgbd is not None),
            }
