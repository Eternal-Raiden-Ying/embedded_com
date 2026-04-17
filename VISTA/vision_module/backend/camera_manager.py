#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from threading import RLock
from typing import Any, Callable, Dict, Optional

from .camera import ColorCamera, IRCamera, RealSenseDepthCamera
from .camera.mock import MockCamera
from ..config.schema import VisionServiceConfig


CapabilitySink = Optional[Callable[[str, str, Dict[str, Any]], None]]


@dataclass(frozen=True)
class CameraSpec:
    name: str
    params: tuple


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

    def _use_placeholder(self) -> bool:
        return bool(getattr(self.cfg.runtime, "capability_placeholder", False))

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
        if self._use_placeholder():
            return MockCamera(**params)
        if name == "depth":
            return RealSenseDepthCamera(**params)
        if name in {"ir", "grey"}:
            return IRCamera(**params)
        return ColorCamera(**params)

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
            if not frame_bundle:
                self._publish_result("frame_meta", {"has_frames": False, "cameras": []})
                self._worker_stop.wait(timeout=self._worker_interval_s)
                continue
            self._last_frame_seq += 1
            rgb_shape = None
            rgb = frame_bundle.get("rgb")
            if rgb is not None:
                try:
                    rgb_shape = tuple(int(v) for v in rgb.shape)
                except Exception:
                    rgb_shape = None
            self._publish_result("camera_frames", frame_bundle)
            self._publish_result(
                "frame_meta",
                {
                    "has_frames": True,
                    "cameras": sorted(frame_bundle.keys()),
                    "rgb_shape": rgb_shape,
                    "frame_seq": int(self._last_frame_seq),
                },
            )
            self._worker_stop.wait(timeout=self._worker_interval_s)

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

        with self._lock:
            current = self._specs.get(name)
            if name in self.cams and current == target_spec:
                return False

            old_cam = self.cams.pop(name, None)
            self._specs.pop(name, None)

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

        with self._lock:
            self.cams[name] = camera
            self._specs[name] = target_spec
        self.log.info("camera enabled: %s", name)
        self._emit(
            "enabled",
            name,
            params=params,
            implementation="mock" if self._use_placeholder() else "real",
        )
        return True

    def disable_camera(self, name: str) -> bool:
        with self._lock:
            camera = self.cams.pop(name, None)
            self._specs.pop(name, None)
        if camera is None:
            return False
        try:
            camera.release()
        except Exception as exc:
            self.log.warning("camera release failed: %s", exc)
        self.log.info("camera disabled: %s", name)
        self._emit("disabled", name)
        return True

    def release_all(self) -> None:
        self.stop_runtime()
        with self._lock:
            names = list(self.cams.keys())
        for name in names:
            self.disable_camera(name)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled_cameras": sorted(self.cams.keys()),
                "camera_specs": sorted(self._specs.keys()),
                "runtime_running": bool(self._runtime_running),
                "last_frame_seq": int(self._last_frame_seq),
            }
