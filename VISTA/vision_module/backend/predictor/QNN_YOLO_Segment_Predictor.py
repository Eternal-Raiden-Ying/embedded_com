#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import threading

import aidlite
import cv2
import numpy as np

from .base import IPredictor
from .utils import NMS_fast, process_mask_fast, xywh2xyxy


logger = logging.getLogger("vision.inference")


class QNN_YOLO_Segment_Predictor(IPredictor):
    def __init__(self, args) -> None:
        self._lock = threading.RLock()
        self.interpreter = None

        config = aidlite.Config.create_instance()
        config.implement_type = aidlite.ImplementType.TYPE_LOCAL
        config.framework_type = aidlite.FrameworkType.TYPE_QNN
        config.accelerate_type = aidlite.AccelerateType.TYPE_DSP
        config.is_quantify_model = 1

        model = aidlite.Model.create_instance(args.target_model)

        self.conf = args.conf_thres
        self.iou = args.iou_thres
        self.width = args.width
        self.height = args.height
        self.class_num = args.class_num
        self.input_shape = [[1, self.height, self.width, 3]]
        self.blocks = int(self.height * self.width * (1 / 64 + 1 / 256 + 1 / 1024))
        self.maskw = int(self.width / 4)
        self.maskh = int(self.height / 4)
        self.output_shape = [
            [1, 32, self.blocks],
            [1, 4, self.blocks],
            [1, self.class_num, self.blocks],
            [1, self.maskh, self.maskw, 32],
        ]

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
        logger.info("qnn predictor loaded: %s", getattr(args, "target_model", ""))

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
            logger.info("releasing qnn predictor resources")
            try:
                interpreter.destory()
            except Exception as exc:
                logger.warning("predictor release failed: %s", exc)
            finally:
                self.interpreter = None
            logger.info("qnn predictor released")

    def predict_frame(self, orig_img_rgb: np.ndarray):
        with self._lock:
            interpreter = self.interpreter
            if interpreter is None:
                return [], []

            input_img = orig_img_rgb.astype(np.float32) * 0.003921568627
            input_img = np.expand_dims(input_img, 0)
            input_img = np.ascontiguousarray(input_img)

            interpreter.set_input_tensor(0, input_img)
            interpreter.invoke()

            input0_data = interpreter.get_output_tensor("bboxes").reshape(1, 4, self.blocks)
            input1_data = interpreter.get_output_tensor("scores").reshape(1, self.class_num, self.blocks)
            input2_data = interpreter.get_output_tensor("mask_coefs").reshape(1, 32, self.blocks)
            protos = interpreter.get_output_tensor("mask_protos").reshape(1, self.maskh, self.maskw, 32).transpose(0, 3, 1, 2)

        boxes = np.concatenate([input0_data, input1_data, input2_data], axis=1)
        x = boxes.transpose(0, 2, 1)

        max_scores = np.amax(x[..., 4:-32], axis=-1)
        x = x[max_scores > self.conf]
        if len(x) < 1:
            return [], []

        cls_scores = np.amax(x[:, 4:-32], axis=-1)
        cls_ids = np.argmax(x[:, 4:-32], axis=-1)
        x = np.c_[x[:, :4], cls_scores, cls_ids, x[:, -32:]]
        x[:, :4] = xywh2xyxy(x[:, :4])

        index = NMS_fast(x[:, :4], x[:, 4], self.iou)
        out_boxes = x[index].astype(np.float16)
        masks = process_mask_fast(protos[0], out_boxes[:, -32:], out_boxes[:, :4], orig_img_rgb.shape)
        return out_boxes, masks


QNNPredictor = QNN_YOLO_Segment_Predictor
