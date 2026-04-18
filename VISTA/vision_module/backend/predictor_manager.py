#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import threading
import time
from threading import RLock
from typing import Any, Callable, Dict, Optional, Tuple

from .predictor import QNN_YOLO_Dectec_Predictor, QNN_YOLO_Segment_Predictor
from .predictor.base import IPredictor
from .predictor.mock import MockPredictor
from ..config.schema import VisionServiceConfig


CapabilitySink = Optional[Callable[[str, str, Dict[str, Any]], None]]


class PredictorManager:
    """Own predictor model lifecycle and readiness checks."""

    def __init__(
        self,
        cfg: VisionServiceConfig,
        logger: Optional[logging.Logger] = None,
        capability_sink: CapabilitySink = None,
    ):
        self.cfg = cfg
        self.log = logger or logging.getLogger("vision.predictor_manager")
        self._capability_sink = capability_sink
        self._lock = RLock()
        self.predictor: Optional[IPredictor] = None
        self.active_model_name: Optional[str] = None
        self.inference_enabled = False
        self._scheduler = None
        self._generation_getter = lambda: 0
        self._runtime_running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_stop = threading.Event()
        self._worker_interval_s = 0.02
        self._last_camera_seq = 0
        self._last_publish_ts = 0.0

    def _use_placeholder(self) -> bool:
        return bool(getattr(self.cfg.runtime, "capability_placeholder", False))

    def _emit(self, action: str, model_name: Optional[str], **fields: Any) -> None:
        if self._capability_sink is None:
            return
        try:
            payload = dict(fields or {})
            if model_name is not None:
                payload["model_name"] = model_name
            self._capability_sink(str(action or "updated").strip().lower(), "predictor_model", payload)
        except Exception:
            pass

    def bind_runtime(self, scheduler, generation_getter=None) -> None:
        self._scheduler = scheduler
        if callable(generation_getter):
            self._generation_getter = generation_getter

    @staticmethod
    def _predictor_type_for(profile) -> str:
        return str(getattr(profile, "predictor_type", "detect") or "detect").strip().lower()

    def _predictor_class_for_profile(self, profile):
        predictor_type = self._predictor_type_for(profile)
        if predictor_type == "segment":
            return QNN_YOLO_Segment_Predictor
        return QNN_YOLO_Dectec_Predictor

    def set_inference_enabled(self, enable: bool) -> None:
        self.inference_enabled = bool(enable)

    def start_runtime(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._runtime_running = True
        self._worker_stop.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, name="predictor_manager.loop", daemon=True)
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
            self._last_publish_ts = time.time()
        except Exception:
            pass

    def _worker_loop(self) -> None:
        while self._runtime_running and not self._worker_stop.is_set():
            scheduler = self._scheduler
            if scheduler is None:
                self._worker_stop.wait(timeout=self._worker_interval_s)
                continue
            frame_slot = scheduler.read_slot("camera_frames")
            if not isinstance(frame_slot, dict):
                self._publish_result(
                    "local_perception",
                    {
                        "has_infer": False,
                        "box_count": 0,
                        "infer_boxes": [],
                        "infer_masks": [],
                        "rgb_shape": None,
                    },
                )
                self._worker_stop.wait(timeout=self._worker_interval_s)
                continue

            seq = int(frame_slot.get("seq", 0) or 0)
            frames = frame_slot.get("payload")
            if seq <= self._last_camera_seq or not isinstance(frames, dict):
                self._worker_stop.wait(timeout=self._worker_interval_s)
                continue
            self._last_camera_seq = seq
            rgb = frames.get("rgb")
            rgb_shape = None
            if rgb is not None:
                try:
                    rgb_shape = tuple(int(v) for v in rgb.shape)
                except Exception:
                    rgb_shape = None

            boxes: list = []
            masks: list = []
            if self.inference_enabled and rgb is not None and self.is_ready():
                boxes, masks = self.predict_frame(rgb)

            self._publish_result(
                "local_perception",
                {
                    "has_infer": bool(self.inference_enabled and rgb is not None and self.is_ready()),
                    "predictor_type": str(type(self.predictor).__name__) if self.predictor is not None else None,
                    "box_count": int(len(boxes or [])),
                    "infer_boxes": list(boxes or []),
                    "infer_masks": list(masks or []),
                    "rgb_shape": rgb_shape,
                },
            )
            self._worker_stop.wait(timeout=self._worker_interval_s)

    def is_ready(self) -> bool:
        with self._lock:
            predictor = self.predictor
        return predictor is not None and predictor.is_ready()

    def ensure_model(self, model_name: str) -> bool:
        target = str(model_name or "").strip()
        if not target:
            self.log.error("empty model name")
            return False

        with self._lock:
            predictor = self.predictor
            active = self.active_model_name
            if predictor is not None and active == target and predictor.is_ready():
                return False
            old_predictor = predictor
            old_model = active
            self.predictor = None
            self.active_model_name = None

        if old_predictor is not None:
            try:
                old_predictor.release()
            except Exception as exc:
                self.log.warning("predictor release failed: %s", exc)
            self._emit("released", old_model)

        profile = self.cfg.model.profiles.get(target)
        if profile is None and not self._use_placeholder():
            self.log.error("model config not found: %s", target)
            self._emit("load_failed", target, error="missing_model_profile")
            return False

        try:
            predictor_cls = MockPredictor if self._use_placeholder() else self._predictor_class_for_profile(profile)
            predictor = predictor_cls(profile)
        except Exception as exc:
            self.log.error("model load failed: %s | %s", target, exc)
            self._emit("load_failed", target, error=str(exc))
            return False

        with self._lock:
            self.predictor = predictor
            self.active_model_name = target
        self.log.info("model loaded: %s", target)
        self._emit(
            "loaded",
            target,
            predictor_type=self._predictor_type_for(profile),
            implementation="mock" if self._use_placeholder() else "real",
        )
        return True

    def disable_model(self) -> bool:
        with self._lock:
            predictor = self.predictor
            model_name = self.active_model_name
            self.predictor = None
            self.active_model_name = None
        if predictor is None:
            return False
        try:
            predictor.release()
        except Exception as exc:
            self.log.warning("predictor release failed: %s", exc)
        self.log.info("model disabled: %s", model_name)
        self._emit("released", model_name)
        return True

    def predict_frame(self, frame) -> Tuple[list, list]:
        with self._lock:
            predictor = self.predictor
            if predictor is None or not predictor.is_ready():
                return [], []
            return predictor.predict_frame(frame)

    def release_all(self) -> None:
        self.stop_runtime()
        self.disable_model()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            predictor = self.predictor
            return {
                "active_model_name": self.active_model_name,
                "predictor_ready": bool(predictor is not None and predictor.is_ready()),
                "predictor_type": type(predictor).__name__ if predictor is not None else None,
                "inference_enabled": bool(self.inference_enabled),
                "runtime_running": bool(self._runtime_running),
                "last_camera_seq": int(self._last_camera_seq),
                "last_publish_ts": float(self._last_publish_ts),
            }
