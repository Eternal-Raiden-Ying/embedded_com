#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import queue
import threading
import traceback
import logging
from typing import Optional, Dict, Tuple, Any, Callable
import cv2

from .camera import ColorCamera, IRCamera, RealSenseDepthCamera
from .predictor import QNNPredictor
from ..config.schema import VisionServiceConfig


class VisionEngine:
    """视觉引擎能力层。"""
    def __init__(
        self,
        cfg: VisionServiceConfig,
        logger: Optional[logging.Logger] = None,
        event_sink: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.cfg = cfg
        self.log = logger or logging.getLogger("vision.engine")
        self._event_sink = event_sink
        self.cams: Dict[str, Any] = {}
        self.predictor: Optional[Any] = None
        self.active_model_name: Optional[str] = None
        self.running = False
        self.lock = threading.RLock()
        self.infer_queue = queue.Queue(maxsize=2)
        self._has_new_data = False
        self.latest_data: Tuple[Optional[Dict[str, cv2.Mat]], Optional[Dict[str, list]]] = (None, None)
        self._capture_thread: Optional[threading.Thread] = None
        self._infer_thread: Optional[threading.Thread] = None
        self.infer_enabled = True

    def _emit_event(self, name: str, **fields: Any):
        if self._event_sink is not None:
            try:
                self._event_sink(name, fields)
            except Exception:
                pass

    def init(self):
        self.log.info("engine init: system ready")
        self._emit_event("engine_init")

    def start(self):
        if self.running:
            return
        self.running = True
        self._capture_thread = threading.Thread(target=self._capture_worker, name="Engine_Capture", daemon=True)
        self._infer_thread = threading.Thread(target=self._infer_worker, name="Engine_Infer", daemon=True)
        self._capture_thread.start()
        self._infer_thread.start()
        self.log.info("vision engine pipeline started")
        self._emit_event("engine_started")

    def _clear_latest(self):
        self._has_new_data = False
        self.latest_data = (None, None)

    def _drain_infer_queue(self):
        while True:
            try:
                frames = self.infer_queue.get_nowait()
            except queue.Empty:
                break
            for v in frames.values():
                try:
                    del v
                except Exception:
                    pass

    def reset_runtime_state(self):
        with self.lock:
            self._clear_latest()
        self._drain_infer_queue()

    def set_inference_enabled(self, enable: bool):
        with self.lock:
            changed = (self.infer_enabled != bool(enable))
            self.infer_enabled = bool(enable)
            self._clear_latest()
        self._drain_infer_queue()
        if changed:
            self.log.info("inference %s", "enabled" if enable else "disabled")
            self._emit_event("inference_changed", enabled=bool(enable))

    def stop(self):
        self.running = False
        time.sleep(0.2)
        self._drain_infer_queue()
        with self.lock:
            self.cams.clear()
            predictor = self.predictor
            self.predictor = None
            self.active_model_name = None
            self.infer_enabled = False
            self._clear_latest()
        if predictor is not None:
            predictor.release()
        self.log.info("pipeline stopped")
        self._emit_event("engine_stopped")

    def set_camera(self, name: str, enable: bool, cfg: dict = None):
        with self.lock:
            if enable:
                if name not in self.cams:
                    # 获取默认配置对象
                    cam_cfg = self.cfg.camera.streams.get(name)
                    
                    # 容错：如果既没有默认配置，也没有传入自定义配置，则报错
                    if not cam_cfg and not cfg:
                        self.log.error("camera config not found: %s", name)
                        return
                    
                    # 初始化自定义配置为字典，方便后续统一处理
                    cfg = cfg or {}

                    # 内部辅助函数：优先从传入的 cfg 获取参数，没有则回退到 cam_cfg
                    def get_param(key, default=None):
                        if key in cfg:
                            return cfg[key]
                        if cam_cfg is None:
                            return default
                        return getattr(cam_cfg, key, default)

                    if name == 'depth':
                        self.cams[name] = RealSenseDepthCamera(
                            height=get_param('height'),
                            width=get_param('width'),
                            fps=get_param('fps')
                        )
                        log_target = "RealSense Depth"
                    elif name in {'ir', 'grey'}:
                        source = get_param('source')
                        video_node = f"/dev/video{source}" if str(source).isdigit() else source
                        self.cams[name] = IRCamera(
                            device=video_node,
                            in_format=get_param('in_format', 'GRAY8'),
                            format=get_param('format', 'BGR'),
                            fps=get_param('fps'),
                            in_w=get_param('in_w'),
                            in_h=get_param('in_h'),
                            out_w=get_param('out_w'),
                            out_h=get_param('out_h'),
                            crop_x=get_param('crop_x', 0),
                            crop_y=get_param('crop_y', 0),
                            crop_w=get_param('crop_w', 0),
                            crop_h=get_param('crop_h', 0),
                        )
                        log_target = video_node
                    else:
                        source = get_param('source')
                        video_node = f"/dev/video{source}" if str(source).isdigit() else source
                        self.cams[name] = ColorCamera(
                            device=video_node,
                            in_format=get_param('in_format', 'YUY2'),
                            format=get_param('format'),
                            fps=get_param('fps'),
                            in_w=get_param('in_w'),
                            in_h=get_param('in_h'),
                            out_w=get_param('out_w'),
                            out_h=get_param('out_h'),
                            crop_x=get_param('crop_x'),
                            crop_y=get_param('crop_y'),
                            crop_w=get_param('crop_w'),
                            crop_h=get_param('crop_h'),
                            auto_exposure=get_param('auto_exposure'),
                            exposure=get_param('exposure'),
                            brightness=get_param('brightness'),
                        )
                        log_target = video_node

                    self._clear_latest()
                    self.log.info("camera enabled: %s", log_target)
                    self._emit_event("camera_enabled", camera=name, target=log_target)
                return
            
            # --- 卸载相机的逻辑保持不变 ---
            if name in self.cams:
                del self.cams[name]
                self._clear_latest()
                self._drain_infer_queue()
                self.log.info("camera disabled: %s", name)
                self._emit_event("camera_disabled", camera=name)

    def set_model(self, name: str, enable: bool):
        old_predictor = None
        if enable:
            with self.lock:
                if self.predictor is not None and self.active_model_name == name and self.predictor.is_ready():
                    return
                old_predictor = self.predictor
                self.predictor = None
                self.active_model_name = None
                self._clear_latest()
            self._drain_infer_queue()
            if old_predictor is not None:
                old_predictor.release()
            model_profile = self.cfg.model.profiles.get(name)
            if not model_profile:
                self.log.error("model config not found: %s", name)
                return
            self.log.info("loading model: %s", name)
            predictor = QNNPredictor(model_profile)
            with self.lock:
                self.predictor = predictor
                self.active_model_name = name
                self._clear_latest()
            self.log.info("model loaded: %s", name)
            self._emit_event("model_loaded", model=name)
            return

        with self.lock:
            old_predictor = self.predictor
            self.predictor = None
            self.active_model_name = None
            self._clear_latest()
        self._drain_infer_queue()
        if old_predictor is not None:
            old_predictor.release()
            self.log.info("model disabled: %s", name)
            self._emit_event("model_disabled", model=name)

    def get_new_data(self) -> Tuple[Optional[Dict[str, cv2.Mat]], Optional[Dict[str, list]]]:
        with self.lock:
            if not self._has_new_data:
                return None, None
            self._has_new_data = False
            return self.latest_data

    def _capture_worker(self):
        while self.running:
            with self.lock:
                active_cams = list(self.cams.items())
            if not active_cams:
                time.sleep(0.01)
                continue

            hw_frames = {}
            for name, cam in active_cams:
                try:
                    frm = cam.read_frame()
                except Exception:
                    frm = None
                if frm is not None and getattr(frm, 'size', 0) > 0:
                    hw_frames[name] = frm
            if not hw_frames:
                continue

            if self.infer_queue.full():
                try:
                    old_frames = self.infer_queue.get_nowait()
                    for v in old_frames.values():
                        del v
                except queue.Empty:
                    pass
                self._emit_event("capture_queue_drop", queue_size=self.infer_queue.maxsize)
            self.infer_queue.put(hw_frames)

    def _infer_worker(self):
        while self.running:
            try:
                hw_frames = self.infer_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                infer_res = None
                with self.lock:
                    predictor = self.predictor
                    infer_enabled = self.infer_enabled
                rgb_frame = hw_frames.get("rgb")
                if infer_enabled and predictor is not None and rgb_frame is not None and predictor.is_ready():
                    out_boxes, masks = predictor.predict_frame(rgb_frame)
                    infer_res = {"boxes": out_boxes, "masks": masks}

                safe_frames = {}
                for name, frm in hw_frames.items():
                    safe_frames[name] = frm.copy()
                    del frm

                with self.lock:
                    self.latest_data = (safe_frames, infer_res)
                    self._has_new_data = True

            except Exception:
                self.log.error("pipeline exception: %s", traceback.format_exc())
                self._emit_event("pipeline_exception", error=traceback.format_exc())
                for v in hw_frames.values():
                    if v is not None:
                        del v
