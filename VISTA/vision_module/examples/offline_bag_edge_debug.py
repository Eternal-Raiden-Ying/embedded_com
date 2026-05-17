#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Optional


HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
VISTA_ROOT = HERE.parents[2]
for item in (REPO_ROOT, VISTA_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from VISTA.vision_module.backend.table_edge_roi import ROI_PRESETS, choose_depth_roi


cv2 = None
np = None
OnlineTableEdgeDetector = None
load_calib = None
DetectorConfig = None
PreviewFrame = None
PreviewOverlay = None
OpenCVPreviewSink = None


DEFAULT_BAG = VISTA_ROOT / "20260516_161436.bag"
DEFAULT_CALIB = VISTA_ROOT / "Offline_Edge_Test" / "calib.json"
DEFAULT_YOLO_MODEL = (
    VISTA_ROOT
    / "vision_module"
    / "model"
    / "qnn216"
    / "model_farm_yolov7_qcs6490_qnn2.16_int8_aidlite"
    / "models"
    / "cutoff_yolov7_w8a8.qnn216.ctx.bin"
)
DEFAULT_YOLO_ANCHORS = (
    (12, 16, 19, 36, 40, 28),
    (36, 75, 76, 55, 72, 146),
    (142, 110, 192, 243, 459, 401),
)
DEFAULT_YOLO_STRIDES = (8, 16, 32)


def _load_runtime_deps() -> None:
    global cv2, np, OnlineTableEdgeDetector, load_calib, DetectorConfig, PreviewFrame, PreviewOverlay, OpenCVPreviewSink
    if np is not None:
        return
    import numpy as _np

    try:
        import aidcv as _cv2
    except ImportError:
        import cv2 as _cv2

    from VISTA.Online_Edge_Detect.detector import OnlineTableEdgeDetector as _OnlineTableEdgeDetector
    from VISTA.Online_Edge_Detect.detector import load_calib as _load_calib
    from VISTA.Online_Edge_Detect.schema import DetectorConfig as _DetectorConfig
    from VISTA.vision_module.backend.preview.base import PreviewFrame as _PreviewFrame
    from VISTA.vision_module.backend.preview.base import PreviewOverlay as _PreviewOverlay
    from VISTA.vision_module.backend.preview.opencv_sink import OpenCVPreviewSink as _OpenCVPreviewSink

    cv2 = _cv2
    np = _np
    OnlineTableEdgeDetector = _OnlineTableEdgeDetector
    load_calib = _load_calib
    DetectorConfig = _DetectorConfig
    PreviewFrame = _PreviewFrame
    PreviewOverlay = _PreviewOverlay
    OpenCVPreviewSink = _OpenCVPreviewSink


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a RealSense RGB+Depth bag through the table-edge detector.")
    parser.add_argument("--bag", type=Path, default=DEFAULT_BAG, help="RealSense .bag file path.")
    parser.add_argument("--stride", type=int, default=10, help="Process one frame every N recorded frames.")
    parser.add_argument("--start-frame", type=int, default=0, help="Skip frames before this 0-based frame index.")
    parser.add_argument("--max-frames", type=int, default=500, help="Maximum recorded frames to read.")
    parser.add_argument(
        "--roi-preset",
        default="center_lower",
        choices=sorted(ROI_PRESETS.keys()),
        help="Depth ROI preset from vision_module.backend.table_edge_roi.",
    )
    parser.add_argument("--show", action="store_true", help="Show the reused OpenCV preview overlay.")
    parser.add_argument("--save-dir", type=Path, default=None, help="Optional directory for sampled preview PNGs.")
    parser.add_argument("--save-csv", type=Path, default=None, help="Optional CSV path for per-frame geometry diagnostics.")
    parser.add_argument("--calib-json", type=Path, default=DEFAULT_CALIB, help="Camera calibration JSON.")
    parser.add_argument("--yolo", action="store_true", help="Run YOLO on sampled RGB frames and draw detection boxes only.")
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO_MODEL, help="QNN YOLO detect model path for --yolo.")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold for --yolo.")
    parser.add_argument("--yolo-iou", type=float, default=0.45, help="YOLO NMS IoU threshold for --yolo.")
    return parser


def _profile_summary(profile: Any) -> Dict[str, Any]:
    stream = profile.as_video_stream_profile()
    return {
        "stream_name": str(profile.stream_type()),
        "format": str(profile.format()),
        "width": int(stream.width()),
        "height": int(stream.height()),
        "fps": int(stream.fps()),
    }


def iter_bag_frames(bag_path: Path, max_frames: int) -> Iterator[Dict[str, Any]]:
    _load_runtime_deps()
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 is required to read RealSense .bag files") from exc

    if not bag_path.exists():
        raise FileNotFoundError(f"bag file not found: {bag_path}")

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(bag_path.expanduser().resolve()), repeat_playback=False)
    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)
    streams = [_profile_summary(p) for p in profile.get_streams()]
    print("[BAG_EDGE] opened bag=", bag_path)
    for item in streams:
        print("[BAG_EDGE] stream=", item)

    align = rs.align(rs.stream.color)
    frame_count = 0
    try:
        while max_frames <= 0 or frame_count < max_frames:
            try:
                frames = pipeline.wait_for_frames(3000)
            except RuntimeError as exc:
                print(f"[BAG_EDGE] playback stopped: {exc}")
                break
            if not frames:
                break
            frame_count += 1
            try:
                aligned = align.process(frames)
            except Exception:
                aligned = frames
            depth_frame = aligned.get_depth_frame() or frames.get_depth_frame()
            color_frame = aligned.get_color_frame() or frames.get_color_frame()
            if not depth_frame:
                continue
            yield {
                "frame": frame_count - 1,
                "ts_ms": float(frames.get_timestamp()),
                "depth": np.asanyarray(depth_frame.get_data()),
                "rgb": np.asanyarray(color_frame.get_data()) if color_frame else None,
            }
    finally:
        pipeline.stop()


def _fallback_roi(cfg: DetectorConfig) -> list[int]:
    return [int(cfg.roi_x0), int(cfg.roi_y0), int(cfg.roi_x1), int(cfg.roi_y1)]


def _load_yolo_predictor(args: argparse.Namespace):
    from VISTA.vision_module.backend.predictor.QNN_YOLO_Detect_Predictor import QNN_YOLO_Detect_Predictor

    model_path = args.yolo_model.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    profile = SimpleNamespace(
        target_model=str(model_path),
        conf_thres=float(args.yolo_conf),
        iou_thres=float(args.yolo_iou),
        width=640,
        height=640,
        class_num=80,
        model_backend="qnn",
        anchors=DEFAULT_YOLO_ANCHORS,
        strides=DEFAULT_YOLO_STRIDES,
    )
    predictor = QNN_YOLO_Detect_Predictor(profile)
    if not predictor.is_ready():
        raise RuntimeError("YOLO predictor is not ready after initialization")
    print(f"[BAG_EDGE] yolo=enabled model={model_path}")
    return predictor


def _rgb_to_bgr(rgb: Any) -> Any:
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] < 3:
        return rgb
    return cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)


def _yolo_local_perception(predictor: Any, rgb: Any) -> Dict[str, Any]:
    from VISTA.vision_module.config.data import COCO80_CLASSES

    if predictor is None or not isinstance(rgb, np.ndarray) or rgb.size <= 0:
        return {"has_infer": False, "box_count": 0, "infer_boxes": [], "class_names": list(COCO80_CLASSES), "rgb_shape": getattr(rgb, "shape", None)}
    boxes, _masks = predictor.predict_frame(_rgb_to_bgr(rgb))
    rows = []
    box_rows = boxes.tolist() if hasattr(boxes, "tolist") else (list(boxes) if boxes is not None else [])
    for row in box_rows:
        try:
            values = row.tolist() if hasattr(row, "tolist") else list(row)
            if len(values) < 6:
                continue
            x1, y1, x2, y2 = [float(v) for v in values[:4]]
            score = float(values[4])
            class_id = int(float(values[5]))
            class_name = COCO80_CLASSES[class_id] if 0 <= class_id < len(COCO80_CLASSES) else str(class_id)
            rows.append([x1, y1, x2, y2, score, class_id, class_name])
        except Exception:
            continue
    return {
        "has_infer": True,
        "box_count": int(len(rows)),
        "infer_boxes": rows,
        "class_names": list(COCO80_CLASSES),
        "rgb_shape": getattr(rgb, "shape", None),
        "table_roi_source": "yolo_preview_only",
    }


def _table_edge_payload(
    result: Any,
    debug: Dict[str, Any],
    roi_meta: Dict[str, Any],
    frame_id: int,
    age_ms: float,
    target_dist_m: float,
    cfg: DetectorConfig,
) -> Dict[str, Any]:
    roi_box = debug.get("roi_box") if isinstance(debug, dict) else None
    roi = [int(v) for v in roi_box] if roi_box is not None else roi_meta.get("depth_edge_roi")
    edge_found = bool(getattr(result, "edge_found", False))
    valid_for_control = bool(getattr(result, "valid_for_control", False))
    yaw = float(getattr(result, "yaw_err_rad", 0.0)) if edge_found else None
    dist = float(getattr(result, "dist_err_m", 0.0)) if edge_found else None
    table_points = int(getattr(result, "table_point_count", 0) or 0)
    all_points = int(getattr(result, "point_count", 0) or 0)
    payload = {
        "table_found": bool(table_points > 0),
        "edge_found": edge_found,
        "edge_valid": valid_for_control,
        "valid": valid_for_control,
        "confidence": float(getattr(result, "edge_confidence", 0.0) or 0.0),
        "edge_conf": float(getattr(result, "edge_confidence", 0.0) or 0.0),
        "yaw_err_rad": yaw,
        "yaw_err": yaw,
        "dist_err_m": dist,
        "dist_err": dist,
        "edge_k": getattr(result, "line_k", None),
        "edge_b": getattr(result, "line_b", None),
        "image_line_k": getattr(result, "image_line_k", None),
        "image_line_b": getattr(result, "image_line_b", None),
        "depth_valid": True,
        "point_count": all_points,
        "valid_edge_points": all_points,
        "table_point_count": table_points,
        "edge_inlier_count": int(getattr(result, "inlier_count", 0) or 0),
        "selected_edge": edge_found,
        "near_edge": valid_for_control,
        "frame_id": int(frame_id),
        "frame_seq": int(frame_id),
        "source": "offline_bag_edge_debug",
        "reason": getattr(result, "reject_reason", "") or ("" if edge_found else ("roi_empty" if all_points <= 0 and table_points <= 0 else "no_valid_edge")),
        "reject_reason": getattr(result, "reject_reason", "") or "",
        "target_dist_m": float(target_dist_m),
        "age_ms": float(age_ms),
        "depth_z_min_m": float(cfg.z_min),
        "depth_z_max_m": float(cfg.z_max),
        "table_y_min_m": float(cfg.table_y_min),
        "table_y_max_m": float(cfg.table_y_max),
        "type": "table_edge_obs",
    }
    for key in (
        "raw_found",
        "pose_found",
        "valid_for_control",
        "pose_source",
        "plane_found",
        "line_found",
        "plane_confidence",
        "line_confidence",
        "plane_residual_mean",
        "line_residual_mean",
        "plane_x_span_m",
        "line_x_span_m",
        "candidate_count",
        "inlier_count",
        "stable_count",
        "front_face_area_ratio",
        "plane_yaw_err_rad",
        "plane_dist_err_m",
        "line_yaw_err_rad",
        "line_dist_err_m",
        "plane_k",
        "plane_b",
        "upper_line_found",
        "upper_line_confidence",
        "upper_line_candidate_count",
        "upper_line_inlier_count",
        "upper_line_residual_mean",
        "upper_line_x_span_m",
        "upper_line_y_norm_mean",
        "upper_line_k",
        "upper_line_b",
        "upper_line_yaw_err_rad",
        "upper_line_dist_err_m",
        "lower_line_found",
        "lower_line_confidence",
        "lower_line_candidate_count",
        "lower_line_inlier_count",
        "lower_line_residual_mean",
        "lower_line_x_span_m",
        "lower_line_y_norm_mean",
        "lower_line_k",
        "lower_line_b",
        "lower_line_yaw_err_rad",
        "lower_line_dist_err_m",
        "selected_line_type",
        "table_geometry_score",
        "front_plane_score",
        "line_score",
        "plane_line_consistency_score",
        "roi_boundary_score",
        "temporal_score",
        "geometry_reject_reason",
        "usable_for_approach",
        "usable_for_alignment",
        "usable_for_stop",
        "control_level",
        "control_reject_reason",
    ):
        payload[key] = getattr(result, key, None)
    if isinstance(debug, dict):
        payload["front_plane_candidate_pixels"] = debug.get("front_plane_candidate_pixels") or []
        payload["crease_candidate_pixels"] = debug.get("crease_candidate_pixels") or []
        payload["crease_inlier_pixels"] = debug.get("crease_inlier_pixels") or []
        payload["upper_line_candidate_pixels"] = debug.get("upper_line_candidate_pixels") or []
        payload["upper_line_inlier_pixels"] = debug.get("upper_line_inlier_pixels") or []
        payload["lower_line_candidate_pixels"] = debug.get("lower_line_candidate_pixels") or []
        payload["lower_line_inlier_pixels"] = debug.get("lower_line_inlier_pixels") or []
    payload.update(dict(roi_meta or {}))
    payload["roi_source"] = getattr(result, "roi_source", "") or payload.get("roi_source")
    payload.update({"depth_edge_roi": roi, "table_edge_roi": roi, "edge_roi": roi, "roi_format": "xyxy"})
    return payload


CSV_FIELDS = [
    "frame_id",
    "ts_ms",
    "roi_preset",
    "roi_box",
    "pose_source",
    "raw_found",
    "pose_found",
    "valid_for_control",
    "stable_count",
    "reject_reason",
    "yaw_err_rad",
    "dist_err_m",
    "edge_confidence",
    "plane_found",
    "line_found",
    "plane_confidence",
    "line_confidence",
    "plane_residual_mean",
    "line_residual_mean",
    "plane_x_span_m",
    "line_x_span_m",
    "front_face_area_ratio",
    "candidate_count",
    "inlier_count",
    "edge_k",
    "edge_b",
    "target_dist_m",
    "selected_line_type",
    "upper_line_found",
    "upper_line_confidence",
    "upper_line_candidate_count",
    "upper_line_inlier_count",
    "upper_line_residual_mean",
    "upper_line_x_span_m",
    "upper_line_y_norm_mean",
    "upper_line_k",
    "upper_line_b",
    "upper_line_yaw_err_rad",
    "upper_line_dist_err_m",
    "lower_line_found",
    "lower_line_confidence",
    "lower_line_candidate_count",
    "lower_line_inlier_count",
    "lower_line_residual_mean",
    "lower_line_x_span_m",
    "lower_line_y_norm_mean",
    "lower_line_k",
    "lower_line_b",
    "lower_line_yaw_err_rad",
    "lower_line_dist_err_m",
    "table_geometry_score",
    "front_plane_score",
    "line_score",
    "plane_line_consistency_score",
    "roi_boundary_score",
    "temporal_score",
    "geometry_reject_reason",
    "usable_for_approach",
    "usable_for_alignment",
    "usable_for_stop",
    "control_level",
    "control_reject_reason",
]


def _csv_row(table_edge: Dict[str, Any], frame_id: int, ts_ms: float, roi_preset: str, target_dist_m: float) -> Dict[str, Any]:
    roi = table_edge.get("depth_edge_roi") or table_edge.get("edge_roi") or ""
    return {
        "frame_id": int(frame_id),
        "ts_ms": float(ts_ms),
        "roi_preset": roi_preset,
        "roi_box": ",".join(str(int(v)) for v in roi) if isinstance(roi, (list, tuple)) else str(roi),
        "pose_source": table_edge.get("pose_source"),
        "raw_found": int(bool(table_edge.get("raw_found"))),
        "pose_found": int(bool(table_edge.get("pose_found"))),
        "valid_for_control": int(bool(table_edge.get("valid_for_control"))),
        "stable_count": int(table_edge.get("stable_count") or 0),
        "reject_reason": table_edge.get("reject_reason") or table_edge.get("reason") or "",
        "yaw_err_rad": table_edge.get("yaw_err_rad"),
        "dist_err_m": table_edge.get("dist_err_m"),
        "edge_confidence": table_edge.get("confidence"),
        "plane_found": int(bool(table_edge.get("plane_found"))),
        "line_found": int(bool(table_edge.get("line_found"))),
        "plane_confidence": table_edge.get("plane_confidence"),
        "line_confidence": table_edge.get("line_confidence"),
        "plane_residual_mean": table_edge.get("plane_residual_mean"),
        "line_residual_mean": table_edge.get("line_residual_mean"),
        "plane_x_span_m": table_edge.get("plane_x_span_m"),
        "line_x_span_m": table_edge.get("line_x_span_m"),
        "front_face_area_ratio": table_edge.get("front_face_area_ratio"),
        "candidate_count": table_edge.get("candidate_count"),
        "inlier_count": table_edge.get("inlier_count"),
        "edge_k": table_edge.get("edge_k"),
        "edge_b": table_edge.get("edge_b"),
        "target_dist_m": float(target_dist_m),
        "selected_line_type": table_edge.get("selected_line_type"),
        "upper_line_found": int(bool(table_edge.get("upper_line_found"))),
        "upper_line_confidence": table_edge.get("upper_line_confidence"),
        "upper_line_candidate_count": table_edge.get("upper_line_candidate_count"),
        "upper_line_inlier_count": table_edge.get("upper_line_inlier_count"),
        "upper_line_residual_mean": table_edge.get("upper_line_residual_mean"),
        "upper_line_x_span_m": table_edge.get("upper_line_x_span_m"),
        "upper_line_y_norm_mean": table_edge.get("upper_line_y_norm_mean"),
        "upper_line_k": table_edge.get("upper_line_k"),
        "upper_line_b": table_edge.get("upper_line_b"),
        "upper_line_yaw_err_rad": table_edge.get("upper_line_yaw_err_rad"),
        "upper_line_dist_err_m": table_edge.get("upper_line_dist_err_m"),
        "lower_line_found": int(bool(table_edge.get("lower_line_found"))),
        "lower_line_confidence": table_edge.get("lower_line_confidence"),
        "lower_line_candidate_count": table_edge.get("lower_line_candidate_count"),
        "lower_line_inlier_count": table_edge.get("lower_line_inlier_count"),
        "lower_line_residual_mean": table_edge.get("lower_line_residual_mean"),
        "lower_line_x_span_m": table_edge.get("lower_line_x_span_m"),
        "lower_line_y_norm_mean": table_edge.get("lower_line_y_norm_mean"),
        "lower_line_k": table_edge.get("lower_line_k"),
        "lower_line_b": table_edge.get("lower_line_b"),
        "lower_line_yaw_err_rad": table_edge.get("lower_line_yaw_err_rad"),
        "lower_line_dist_err_m": table_edge.get("lower_line_dist_err_m"),
        "table_geometry_score": table_edge.get("table_geometry_score"),
        "front_plane_score": table_edge.get("front_plane_score"),
        "line_score": table_edge.get("line_score"),
        "plane_line_consistency_score": table_edge.get("plane_line_consistency_score"),
        "roi_boundary_score": table_edge.get("roi_boundary_score"),
        "temporal_score": table_edge.get("temporal_score"),
        "geometry_reject_reason": table_edge.get("geometry_reject_reason"),
        "usable_for_approach": int(bool(table_edge.get("usable_for_approach"))),
        "usable_for_alignment": int(bool(table_edge.get("usable_for_alignment"))),
        "usable_for_stop": int(bool(table_edge.get("usable_for_stop"))),
        "control_level": table_edge.get("control_level"),
        "control_reject_reason": table_edge.get("control_reject_reason"),
    }


def _make_preview_canvas(
    sink: OpenCVPreviewSink,
    rgb: Any,
    depth: np.ndarray,
    table_edge: Dict[str, Any],
    frame_id: int,
    local_perception: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    panel_w = max(320, sink.canvas_w // 2)
    panel_h = max(220, sink.canvas_h // 2)
    panel_size = (panel_w, panel_h)
    metadata = {
        "preview_layout": "rgb_depth_edge",
        "runtime_status": {"stage": "OFFLINE_BAG", "mode": "TABLE_EDGE_PERCEPTION", "epoch": 0},
        "local_perception": local_perception or {"rgb_shape": getattr(rgb, "shape", None), "box_count": 0},
        "table_edge_obs": table_edge,
        "target_obs": {},
        "source_cameras": ["rgb", "depth"] if isinstance(rgb, np.ndarray) else ["depth"],
        "window_id": sink.window_id,
        "show_age_ms": True,
        "show_yolo_boxes": bool((local_perception or {}).get("has_infer", False)),
        "frame_age_s": 0.0,
    }
    frame = PreviewFrame(
        ts=time.time(),
        image={"rgb": rgb, "depth": depth},
        stage="OFFLINE_BAG",
        mode="TABLE_EDGE_PERCEPTION",
        overlay=PreviewOverlay(title="Offline Bag Edge Debug", metadata=metadata),
    )
    rgb_panel = sink._make_rgb_panel(rgb, metadata, panel_size)
    depth_panel = sink._make_depth_panel(depth, table_edge, panel_size)
    edge_panel = sink._make_edge_panel(depth, table_edge, panel_size)
    info_panel = sink._make_info_panel(frame, metadata, table_edge, {}, panel_size)
    canvas = np.vstack([np.hstack([rgb_panel, depth_panel]), np.hstack([edge_panel, info_panel])])
    cv2.putText(canvas, f"bag_frame={frame_id}", (18, canvas.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 235, 240), 1)
    if canvas.shape[:2] != (sink.canvas_h, sink.canvas_w):
        canvas = cv2.resize(canvas, (sink.canvas_w, sink.canvas_h), interpolation=cv2.INTER_AREA)
    return canvas


def _preview_frame(
    rgb: Any,
    depth: np.ndarray,
    table_edge: Dict[str, Any],
    frame_id: int,
    local_perception: Optional[Dict[str, Any]] = None,
) -> PreviewFrame:
    metadata = {
        "preview_layout": "rgb_depth_edge",
        "runtime_status": {"stage": "OFFLINE_BAG", "mode": "TABLE_EDGE_PERCEPTION", "epoch": 0},
        "local_perception": local_perception or {"rgb_shape": getattr(rgb, "shape", None), "box_count": 0},
        "table_edge_obs": table_edge,
        "target_obs": {},
        "source_cameras": ["rgb", "depth"] if isinstance(rgb, np.ndarray) else ["depth"],
        "show_age_ms": True,
        "show_yolo_boxes": bool((local_perception or {}).get("has_infer", False)),
        "frame_age_s": 0.0,
    }
    lines = [
        "stage=OFFLINE_BAG",
        "mode=TABLE_EDGE_PERCEPTION",
        f"frame={frame_id}",
        f"valid={int(bool(table_edge.get('edge_valid')))}",
        f"yaw={table_edge.get('yaw_err_rad')}",
        f"dist={table_edge.get('dist_err_m')}",
        f"age_ms={table_edge.get('age_ms')}",
    ]
    return PreviewFrame(time.time(), {"rgb": rgb, "depth": depth}, "OFFLINE_BAG", "TABLE_EDGE_PERCEPTION", PreviewOverlay("Offline Bag Edge Debug", lines, metadata=metadata))


def main() -> None:
    args = build_parser().parse_args()
    _load_runtime_deps()
    stride = max(1, int(args.stride))
    start_frame = max(0, int(args.start_frame))
    max_frames = int(args.max_frames)
    save_dir: Optional[Path] = args.save_dir
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    csv_file = None
    csv_writer = None
    if args.save_csv is not None:
        args.save_csv.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        csv_file = args.save_csv.expanduser().open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        csv_writer.writeheader()

    cfg = DetectorConfig()
    calib, target_dist = load_calib(args.calib_json.expanduser().resolve())
    if float(cfg.target_dist_m_override) > 0:
        target_dist = float(cfg.target_dist_m_override)
    detector = OnlineTableEdgeDetector(calib, cfg, target_dist)
    yolo_predictor = _load_yolo_predictor(args) if args.yolo else None
    sink = OpenCVPreviewSink("Offline Bag Edge Debug")
    if args.show:
        sink.open()

    processed = 0
    try:
        for pack in iter_bag_frames(args.bag, max_frames=max_frames):
            frame_id = int(pack["frame"])
            if frame_id < start_frame or ((frame_id - start_frame) % stride) != 0:
                continue
            depth = pack["depth"]
            rgb = pack.get("rgb")
            t0 = time.perf_counter()
            roi_meta = choose_depth_roi(
                {},
                getattr(rgb, "shape", None),
                getattr(depth, "shape", None),
                _fallback_roi(cfg),
                manual_static=False,
                roi_preset=args.roi_preset,
            )
            result, debug = detector.process_depth(depth, roi_override=roi_meta.get("depth_edge_roi"))
            local_perception = _yolo_local_perception(yolo_predictor, rgb) if args.yolo else {
                "has_infer": False,
                "box_count": 0,
                "infer_boxes": [],
                "rgb_shape": getattr(rgb, "shape", None),
            }
            age_ms = (time.perf_counter() - t0) * 1000.0
            table_edge = _table_edge_payload(result, debug, roi_meta, frame_id, age_ms, target_dist, cfg)
            if csv_writer is not None:
                csv_writer.writerow(_csv_row(table_edge, frame_id, float(pack.get("ts_ms", 0.0) or 0.0), args.roi_preset, target_dist))
            processed += 1

            print(
                "[BAG_EDGE] "
                f"frame={frame_id} valid={int(bool(table_edge.get('edge_valid')))} "
                f"dist={table_edge.get('dist_err_m')} yaw={table_edge.get('yaw_err_rad')} "
                f"geom={table_edge.get('table_geometry_score')} level={table_edge.get('control_level')} "
                f"line={table_edge.get('selected_line_type')} "
                f"roi={table_edge.get('depth_edge_roi')} preset={args.roi_preset} "
                f"yolo_boxes={int(local_perception.get('box_count', 0) or 0)} age_ms={age_ms:.1f}"
            )

            if save_dir is not None:
                canvas = _make_preview_canvas(sink, rgb, depth, table_edge, frame_id, local_perception)
                cv2.imwrite(str(save_dir / f"bag_edge_{frame_id:06d}_{args.roi_preset}.png"), canvas)

            if args.show:
                keep_running = sink.render(_preview_frame(rgb, depth, table_edge, frame_id, local_perception))
                key = cv2.waitKey(1) & 0xFF
                if not keep_running or key in (ord("q"), ord("Q")):
                    break
    finally:
        if csv_file is not None:
            csv_file.close()
        if yolo_predictor is not None:
            try:
                yolo_predictor.release()
            except Exception:
                pass
        if args.show:
            sink.close()
    print(f"[BAG_EDGE] processed={processed} stride={stride} start_frame={start_frame} max_frames={max_frames}")


if __name__ == "__main__":
    main()
