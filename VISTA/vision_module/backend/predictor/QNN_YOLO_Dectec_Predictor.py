#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import threading

import aidlite
import numpy as np

from .base import IPredictor
from .detect_utils import (
    default_yolov7_anchors,
    default_yolov7_strides,
    detect_postprocess,
    normalize_anchors,
    normalize_strides,
    preprocess_img,
)
from .yolov7_head import Detect


logger = logging.getLogger("vision.inference")


class QNN_YOLO_Dectec_Predictor(IPredictor):
    """QNN YOLO detector predictor adapted from the local tmp benchmark code."""

    def __init__(self, args) -> None:
        self._lock = threading.RLock()
        self.interpreter = None

        backend_name = str(getattr(args, "model_backend", "qnn") or "qnn").strip().lower()
        config = aidlite.Config.create_instance()
        config.implement_type = aidlite.ImplementType.TYPE_LOCAL
        if backend_name in {"snpe", "snpe2"}:
            config.framework_type = aidlite.FrameworkType.TYPE_SNPE2
        else:
            config.framework_type = aidlite.FrameworkType.TYPE_QNN
        config.accelerate_type = aidlite.AccelerateType.TYPE_DSP
        config.is_quantify_model = 1

        model = aidlite.Model.create_instance(args.target_model)

        self.conf = float(getattr(args, "conf_thres", 0.45))
        self.iou = float(getattr(args, "iou_thres", 0.45))
        self.width = int(getattr(args, "width", 640))
        self.height = int(getattr(args, "height", 640))
        self.class_num = int(getattr(args, "class_num", 80))
        self.anchors = normalize_anchors(getattr(args, "anchors", None))
        self.strides = normalize_strides(getattr(args, "strides", None))
        if not self.anchors:
            self.anchors = default_yolov7_anchors()
        if not self.strides:
            self.strides = default_yolov7_strides()
        self.yolo_head = Detect(self.class_num, self.anchors, self.strides, self.width)

        self.input_shape = [[1, self.height, self.width, 3]]
        self.output_shape = []
        for stride in self.strides:
            grid_h = int(self.height / int(stride))
            grid_w = int(self.width / int(stride))
            channels = int((self.class_num + 5) * (len(self.anchors[0]) // 2))
            self.output_shape.append([1, grid_h, grid_w, channels])

        model.set_model_properties(
            self.input_shape,
            aidlite.DataType.TYPE_FLOAT32,
            self.output_shape,
            aidlite.DataType.TYPE_FLOAT32,
        )
        interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(model, config)
        interpreter.init()
        interpreter.load_model()
        self.interpreter = interpreter
        logger.info(
            "qnn detect predictor loaded: %s | backend=%s",
            getattr(args, "target_model", ""),
            backend_name,
        )

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def is_ready(self) -> bool:
        with self._lock:
            return self.interpreter is not None

    def release(self) -> None:
        with self._lock:
            interpreter = getattr(self, "interpreter", None)
            if interpreter is None:
                return
            logger.info("releasing qnn detect predictor resources")
            try:
                interpreter.destory()
            except Exception as exc:
                logger.warning("detect predictor release failed: %s", exc)
            finally:
                self.interpreter = None
            logger.info("qnn detect predictor released")

    def predict_frame(self, orig_img_rgb: np.ndarray):
        with self._lock:
            interpreter = self.interpreter
            if interpreter is None:
                return [], []

            input_img = preprocess_img(orig_img_rgb, target_shape=(self.height, self.width))
            interpreter.set_input_tensor(0, input_img)
            interpreter.invoke()

            outputs = []
            for idx, shape in enumerate(self.output_shape):
                output = interpreter.get_output_tensor(idx).reshape(*shape).transpose(0, 3, 1, 2)
                outputs.append(output)

        pred = self.yolo_head(outputs)
        detections = detect_postprocess(
            pred,
            image_shape=orig_img_rgb.shape,
            input_shape=(self.height, self.width),
            conf_thres=self.conf,
            iou_thres=self.iou,
        )
        return detections, []


QNN_YOLO_Detect_Predictor = QNN_YOLO_Dectec_Predictor
QNNPredictor = QNN_YOLO_Dectec_Predictor
