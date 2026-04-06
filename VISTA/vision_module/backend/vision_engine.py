#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import queue
import threading
import traceback
from typing import Optional, Dict, Tuple, Any
import cv2

# HAL: 将项目根目录加入 sys.path，使 src.hal 可被 import
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.hal.factory import is_mock as _is_mock
_IS_MOCK = _is_mock()

from ..config.schema import VisionServiceConfig


class VisionEngine:
    """视觉引擎能力层。"""
    def __init__(self, cfg: VisionServiceConfig, logger):
        self.cfg = cfg
        self.log = logger
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

    def init(self):
        self.log.info("⚙️ 引擎初始化：系统准备就绪，等待 App 层分配硬件资源...")

    def start(self):
        if self.running:
            return
        self.running = True
        self._capture_thread = threading.Thread(target=self._capture_worker, name="Engine_Capture", daemon=True)
        self._infer_thread = threading.Thread(target=self._infer_worker, name="Engine_Infer", daemon=True)
        self._capture_thread.start()
        self._infer_thread.start()
        self.log.info("🚀 视觉底层引擎流水线已运转")

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
            self.log.info(f"🧭 推理状态切换 -> {'ON' if enable else 'OFF'}")

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
        self.log.info("✅ 底层流水线停止，硬件资源已全部安全释放")

    def set_camera(self, name: str, enable: bool, cfg: dict = None):
        with self.lock:
            if enable:
                if name not in self.cams:
                    # 获取默认配置对象
                    cam_cfg = self.cfg.camera.streams.get(name)
                    
                    # 容错：如果既没有默认配置，也没有传入自定义配置，则报错
                    if not cam_cfg and not cfg:
                        self.log.error(f"❌ 找不到相机 [{name}] 的配置参数")
                        return
                    
                    # 初始化自定义配置为字典，方便后续统一处理
                    cfg = cfg or {}

                    # 内部辅助函数：优先从传入的 cfg 获取参数，没有则回退到 cam_cfg
                    def get_param(key, default=None):
                        if key in cfg:
                            return cfg[key]
                        else:
                            return cam_cfg[key]

                    if name == 'depth':
                        if _IS_MOCK:
                            from src.hal.mock.camera import MockCamera
                            self.cams[name] = MockCamera(
                                out_w=get_param('width'), out_h=get_param('height')
                            )
                        else:
                            from .camera import RealSenseDepthCamera
                            self.cams[name] = RealSenseDepthCamera(
                                height=get_param('height'),
                                width=get_param('width'),
                                fps=get_param('fps')
                            )
                        log_target = "RealSense Depth (mock)" if _IS_MOCK else "RealSense Depth"
                    else:
                        source = get_param('source')
                        video_node = f"/dev/video{source}" if str(source).isdigit() else source
                        if _IS_MOCK:
                            from src.hal.mock.camera import MockCamera
                            self.cams[name] = MockCamera(
                                out_w=get_param('out_w'), out_h=get_param('out_h')
                            )
                        else:
                            from .camera import HardwareCamera
                            self.cams[name] = HardwareCamera(
                                device=video_node,
                                format=get_param('format'),
                                fps=get_param('fps'),
                                in_w=get_param('in_w'),
                                in_h=get_param('in_h'),
                                out_w=get_param('out_w'),
                                out_h=get_param('out_h'),
                                crop_x=get_param('crop_x'),
                                crop_y=get_param('crop_y'),
                                crop_w=get_param('crop_w'),
                                crop_h=get_param('crop_h')
                            )
                        log_target = f"{video_node} (mock)" if _IS_MOCK else video_node

                    self._clear_latest()
                    self.log.info(f"📸 挂载并启动相机 [{name.upper()}] -> {log_target}")
                return
            
            # --- 卸载相机的逻辑保持不变 ---
            if name in self.cams:
                del self.cams[name]
                self._clear_latest()
                self._drain_infer_queue()
                self.log.info(f"🛑 卸载并释放相机 [{name.upper()}]")

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
                self.log.error(f"❌ 找不到模型 [{name}] 的配置参数")
                return
            self.log.info(f"🧠 加载 NPU 模型 [{name}] 中...")
            if _IS_MOCK:
                from src.hal.mock.predictor import MockPredictor
                predictor = MockPredictor()
            else:
                from .predictor import QNN_YOLO_Segment_Predictor
                predictor = QNN_YOLO_Segment_Predictor(model_profile)
            with self.lock:
                self.predictor = predictor
                self.active_model_name = name
                self._clear_latest()
            self.log.info(f"🧠 加载 NPU 模型 [{name}] 完成")
            return

        with self.lock:
            old_predictor = self.predictor
            self.predictor = None
            self.active_model_name = None
            self._clear_latest()
        self._drain_infer_queue()
        if old_predictor is not None:
            old_predictor.release()
            self.log.info(f"🛑 卸载 NPU 模型 [{name}]，释放 DSP 内存")

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
                self.log.error("流水线处理异常\n%s", traceback.format_exc())
                for v in hw_frames.values():
                    if v is not None:
                        del v
