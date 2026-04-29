#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import os
import threading
import time
from threading import RLock
from typing import Any, Callable, Dict, Optional, Tuple

from .predictor import (
    MockPredictor,
    QNN_YOLO_Dectec_Predictor,
    QNN_YOLO_Segment_Predictor,
    predictor_backend_status,
)
from .predictor.base import IPredictor
from ..config.data import COCO80_CLASSES, normalize_class_names
from ..config.schema import VisionServiceConfig
from ..utils.table_roi import build_table_roi, find_table_bbox


CapabilitySink = Optional[Callable[[str, str, Dict[str, Any]], None]]

LOCAL_PERCEPTION_CONTRACT = "local_perception.v1"
DETECT_BOX_FORMAT = "xyxy_score_class_id"
SEGMENT_BOX_FORMAT = "xyxy_score_class_id_mask_coeffs"


def _to_python_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        try:
            return _to_python_value(value.tolist())
        except Exception:
            pass
    if isinstance(value, tuple):
        return [_to_python_value(item) for item in value]
    if isinstance(value, list):
        return [_to_python_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_python_value(item) for key, item in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _normalize_box_rows(value: Any) -> list:
    normalized = _to_python_value(value)
    if normalized is None:
        return []
    if isinstance(normalized, list) and normalized and not isinstance(normalized[0], list):
        normalized = [normalized]
    if not isinstance(normalized, list):
        return []
    rows = []
    for row in normalized:
        if isinstance(row, list):
            rows.append([_to_python_value(item) for item in row])
    return rows


def _normalize_mask_payload(value: Any) -> list:
    normalized = _to_python_value(value)
    if normalized is None:
        return []
    if isinstance(normalized, list):
        return normalized
    return []


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
        self._active_predictor_type = "detect"
        self._active_class_names: Tuple[str, ...] = ()
        self._active_class_name_source = "profile"
        self._backend_status = predictor_backend_status()
        self._active_implementation = str(self._backend_status.get("resolved_backend") or "mock")
        self.inference_enabled = False
        self._scheduler = None
        self._generation_getter = lambda: 0
        self._runtime_running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_stop = threading.Event()
        self._worker_interval_s = 0.02
        self._last_camera_generation = 0
        self._last_camera_seq = 0
        self._last_publish_ts = 0.0
        self._last_summary_ts = 0.0
        self._last_contract_ok = True
        self._last_contract_error = ""
        self._last_contract_warning_count = 0

    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return bool(default)
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _plain_list(value: Any) -> list:
        if value is None:
            return []
        try:
            return value.tolist()
        except Exception:
            pass
        try:
            return list(value)
        except Exception:
            return []

    @classmethod
    def _mock_table_bbox(cls, rgb_shape: Any) -> Optional[list]:
        raw = os.getenv("VISTA_MOCK_TABLE_BBOX")
        parsed = find_table_bbox({"mock_table_bbox": raw}) if raw else None
        if parsed is not None:
            return parsed
        if not cls._env_bool("VISTA_MOCK_TABLE_BBOX", False):
            return None
        if not isinstance(rgb_shape, (list, tuple)) or len(rgb_shape) < 2:
            return None
        try:
            h = int(rgb_shape[0])
            w = int(rgb_shape[1])
        except Exception:
            return None
        if h <= 0 or w <= 0:
            return None
        return [w // 4, h // 2, (w * 3) // 4, (h * 9) // 10]

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

    def _infer_box_format(self) -> str:
        if self._active_predictor_type == "segment":
            return SEGMENT_BOX_FORMAT
        return DETECT_BOX_FORMAT

    def _effective_class_names(self) -> Tuple[Tuple[str, ...], str]:
        if self._active_class_names:
            return self._active_class_names, "profile"
        if self._active_predictor_type == "detect":
            return COCO80_CLASSES, "fallback_coco80"
        return (), "missing"

    @staticmethod
    def _is_finite_number(value: float) -> bool:
        try:
            return value == value and abs(value) != float("inf")
        except Exception:
            return False

    def _validate_detect_boxes(self, boxes: Any) -> Tuple[list, list, str]:
        normalized_boxes = _normalize_box_rows(boxes)
        valid_rows = []
        warnings = []
        errors = []
        for index, row in enumerate(normalized_boxes):
            if len(row) < 6:
                errors.append(f"row[{index}] expected >=6 values, got {len(row)}")
                continue
            try:
                x1 = float(row[0])
                y1 = float(row[1])
                x2 = float(row[2])
                y2 = float(row[3])
                score = float(row[4])
                class_id = int(float(row[5]))
            except Exception as exc:
                errors.append(f"row[{index}] parse_failed: {exc}")
                continue
            values = (x1, y1, x2, y2, score, float(class_id))
            if not all(self._is_finite_number(value) for value in values):
                errors.append(f"row[{index}] contains non_finite values")
                continue
            valid_rows.append([x1, y1, x2, y2, score, class_id])
            if len(row) > 6:
                warnings.append(f"row[{index}] truncated extra columns: {len(row) - 6}")
        return valid_rows, warnings, "; ".join(errors)

    def _normalize_local_perception(self, boxes: Any, masks: Any) -> Tuple[list, list, bool, str, list]:
        if self._active_predictor_type == "segment":
            return _normalize_box_rows(boxes), _normalize_mask_payload(masks), True, "", []
        valid_rows, warnings, error = self._validate_detect_boxes(boxes)
        return valid_rows, _normalize_mask_payload(masks), not bool(error), error, warnings

    def _build_local_perception_payload(self, rgb_shape, boxes, masks, has_infer: bool) -> Dict[str, Any]:
        class_names, class_name_source = self._effective_class_names()
        normalized_boxes, normalized_masks, contract_ok, contract_error, contract_warnings = self._normalize_local_perception(
            boxes,
            masks,
        )
        if class_name_source != "profile":
            contract_warnings = list(contract_warnings) + [f"class_names_source={class_name_source}"]
        self._last_contract_ok = bool(contract_ok)
        self._last_contract_error = str(contract_error or "")
        self._last_contract_warning_count = int(len(contract_warnings))
        self._active_class_name_source = class_name_source
        return {
            "contract": LOCAL_PERCEPTION_CONTRACT,
            "contract_ok": bool(contract_ok),
            "contract_error": str(contract_error or ""),
            "contract_warnings": list(contract_warnings),
            "has_infer": bool(has_infer),
            "implementation": str(self._active_implementation or "real"),
            "model_name": self.active_model_name,
            "predictor_type": str(self._active_predictor_type or "detect"),
            "class_names": list(class_names),
            "class_names_source": class_name_source,
            "infer_box_format": self._infer_box_format(),
            "box_count": int(len(normalized_boxes)),
            "infer_boxes": normalized_boxes,
            "infer_masks": normalized_masks,
            "rgb_shape": rgb_shape,
        }

    def set_inference_enabled(self, enable: bool) -> None:
        self.inference_enabled = bool(enable)

    def start_runtime(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._runtime_running = True
        self._worker_stop.clear()
        self._last_camera_generation = 0
        self._last_camera_seq = 0
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
                payload = self._build_local_perception_payload(None, [], [], False)
                payload.update(
                    {
                        "table_bbox": None,
                        "table_quadrant": None,
                        "rgb_search_roi": None,
                        "table_roi_source": "yolo_unavailable",
                    }
                )
                self._publish_result("local_perception", payload)
                self._worker_stop.wait(timeout=self._worker_interval_s)
                continue

            generation = int(frame_slot.get("generation", 0) or 0)
            seq = int(frame_slot.get("seq", 0) or 0)
            frames = frame_slot.get("payload")
            if generation != self._last_camera_generation:
                self._last_camera_generation = generation
                self._last_camera_seq = 0
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

            boxes: Any = []
            masks: Any = []
            has_infer = bool(self.inference_enabled and rgb is not None and self.is_ready())
            if has_infer:
                boxes, masks = self.predict_frame(rgb)

            payload = self._build_local_perception_payload(rgb_shape, boxes, masks, has_infer)
            contract_error = str(payload.get("contract_error") or "")
            if contract_error:
                self.log.warning(
                    "local_perception contract degraded | model=%s error=%s warnings=%s",
                    self.active_model_name,
                    contract_error,
                    payload.get("contract_warnings"),
                )

            boxes_list = list(payload.get("infer_boxes") or [])
            roi_input = {"infer_boxes": boxes_list, "rgb_shape": rgb_shape}
            table_bbox = find_table_bbox(roi_input)
            table_source = "yolo_table_bbox" if table_bbox is not None else "yolo_unavailable"
            if table_bbox is not None:
                roi_input["table_bbox"] = table_bbox
            else:
                mock_bbox = self._mock_table_bbox(rgb_shape)
                if mock_bbox is not None:
                    roi_input["mock_table_bbox"] = mock_bbox
                    table_bbox = find_table_bbox(roi_input)
                    table_source = "mock_table_bbox"
            roi_meta = build_table_roi(roi_input, rgb_shape, None)
            table_bbox_payload = roi_meta.get("table_bbox")
            self._log_local_summary(
                boxes_list=boxes_list,
                has_infer=bool(has_infer),
                rgb_shape=rgb_shape,
                table_bbox=table_bbox_payload,
                table_quadrant=roi_meta.get("table_quadrant"),
                rgb_search_roi=roi_meta.get("rgb_search_roi"),
                table_source=table_source,
            )
            payload.update(
                {
                    "table_bbox": table_bbox_payload,
                    "table_quadrant": roi_meta.get("table_quadrant"),
                    "rgb_search_roi": roi_meta.get("rgb_search_roi"),
                    "table_roi_source": table_source,
                }
            )
            self._publish_result("local_perception", payload)
            self._worker_stop.wait(timeout=self._worker_interval_s)

    def _log_local_summary(
        self,
        *,
        boxes_list: list,
        has_infer: bool,
        rgb_shape: Any,
        table_bbox: Any,
        table_quadrant: Any,
        rgb_search_roi: Any,
        table_source: str,
    ) -> None:
        now = time.time()
        if now - self._last_summary_ts < 1.0:
            return
        self._last_summary_ts = now
        class_ids = []
        for row in list(boxes_list or [])[:12]:
            try:
                if isinstance(row, dict):
                    cid = row.get("class_id", row.get("cls", row.get("class")))
                elif isinstance(row, (list, tuple)) and len(row) >= 6:
                    cid = row[5]
                else:
                    continue
                class_ids.append(int(float(cid)))
            except Exception:
                continue
        self.log.info(
            "local_perception summary | has_infer=%s boxes=%s class_ids=%s rgb_shape=%s table_bbox=%s table_quadrant=%s rgb_search_roi=%s table_roi_source=%s",
            bool(has_infer),
            int(len(boxes_list or [])),
            class_ids,
            rgb_shape,
            table_bbox,
            table_quadrant,
            rgb_search_roi,
            table_source,
        )

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
            self._active_class_names = ()

        if old_predictor is not None:
            try:
                old_predictor.release()
            except Exception as exc:
                self.log.warning("predictor release failed: %s", exc)
            self._emit("released", old_model)

        profile = self.cfg.model.profiles.get(target)
        if profile is None:
            self.log.error("model config not found: %s", target)
            self._emit("load_failed", target, error="missing_model_profile")
            return False

        predictor_type = self._predictor_type_for(profile)
        active_class_names = normalize_class_names(getattr(profile, "classes", None))
        try:
            predictor_cls = self._predictor_class_for_profile(profile)
            predictor = predictor_cls(profile)
        except Exception as exc:
            self.log.error("model load failed: %s | %s", target, exc)
            self._emit("load_failed", target, error=str(exc))
            return False

        implementation = "mock" if type(predictor).__name__ == "MockPredictor" else str(self._backend_status.get("resolved_backend") or "real")
        with self._lock:
            self.predictor = predictor
            self.active_model_name = target
            self._active_predictor_type = predictor_type
            self._active_class_names = active_class_names
            self._active_implementation = implementation
            self._active_class_name_source = "profile"
        self.log.info("model loaded: %s", target)
        if predictor_type == "detect" and not active_class_names:
            self.log.warning("detect profile missing classes | model=%s fallback=coco80", target)
        self._emit(
            "loaded",
            target,
            predictor_type=predictor_type,
            implementation=implementation,
            contract=LOCAL_PERCEPTION_CONTRACT,
            infer_box_format=self._infer_box_format(),
            class_name_count=int(len(active_class_names)),
            class_name_source="profile" if active_class_names else "fallback_coco80",
        )
        return True

    def disable_model(self) -> bool:
        with self._lock:
            predictor = self.predictor
            model_name = self.active_model_name
            self.predictor = None
            self.active_model_name = None
            self._active_class_names = ()
            self._active_predictor_type = "detect"
            self._active_class_name_source = "profile"
            self._active_implementation = str(self._backend_status.get("resolved_backend") or "real")
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
                "predictor_profile_type": str(self._active_predictor_type or "detect"),
                "class_name_count": int(len(self._active_class_names)),
                "class_name_source": str(self._active_class_name_source or "profile"),
                "local_perception_contract": LOCAL_PERCEPTION_CONTRACT,
                "infer_box_format": self._infer_box_format(),
                "implementation": str(self._active_implementation or "real"),
                "inference_enabled": bool(self.inference_enabled),
                "runtime_running": bool(self._runtime_running),
                "last_camera_generation": int(self._last_camera_generation),
                "last_camera_seq": int(self._last_camera_seq),
                "last_contract_ok": bool(self._last_contract_ok),
                "last_contract_error": str(self._last_contract_error or ""),
                "last_contract_warning_count": int(self._last_contract_warning_count),
                "backend_status": dict(self._backend_status or {}),
                "last_publish_ts": float(self._last_publish_ts),
            }
