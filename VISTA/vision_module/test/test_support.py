#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


TEST_DIR = Path(__file__).resolve().parent
VISION_ROOT = TEST_DIR.parent
VISTA_ROOT = VISION_ROOT.parent
if str(VISTA_ROOT) not in sys.path:
    sys.path.insert(0, str(VISTA_ROOT))
if str(VISION_ROOT) not in sys.path:
    sys.path.insert(0, str(VISION_ROOT))

from vision_module.config.schema import (  # noqa: E402
    CameraConfig,
    ColorCameraConfig,
    DebugConfig,
    DepthCameraConfig,
    IRCameraConfig,
    ModelConfig,
    RuntimeConfig,
    SingleModelConfig,
    VisionServiceConfig,
)


EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_INTERRUPT = 130


DEFAULT_RGB_DEVICE = "/dev/video6"
DEFAULT_DEPTH_DEVICE = "/dev/video2"
DEFAULT_IR_DEVICE = "/dev/video4"
DEFAULT_RGB_IN_W = 1280
DEFAULT_RGB_IN_H = 720
DEFAULT_RGB_OUT_W = 640
DEFAULT_RGB_OUT_H = 640
DEFAULT_RGB_FPS = 30
DEFAULT_DEPTH_W = 424
DEFAULT_DEPTH_H = 240
DEFAULT_DEPTH_FPS = 15
DEFAULT_IR_IN_W = 640
DEFAULT_IR_IN_H = 480
DEFAULT_IR_OUT_W = 640
DEFAULT_IR_OUT_H = 480
DEFAULT_IR_FPS = 30
DEFAULT_MODEL_WIDTH = 640
DEFAULT_MODEL_HEIGHT = 640
DEFAULT_CONF_THRES = 0.25
DEFAULT_IOU_THRES = 0.15
DEFAULT_CLASS_NUM = 20
DEFAULT_ITERATIONS = 10


@dataclass
class AttemptResult:
    requested: str
    resolved: str
    ok: bool
    detail: str


class PrintLogger:
    def __init__(self, prefix: str):
        self.prefix = prefix

    def info(self, msg: str, *args):
        print(f"[INFO] [{self.prefix}] {msg % args if args else msg}")

    def warning(self, msg: str, *args):
        print(f"[WARN] [{self.prefix}] {msg % args if args else msg}")

    def error(self, msg: str, *args):
        print(f"[ERROR] [{self.prefix}] {msg % args if args else msg}")

    def debug(self, msg: str, *args):
        print(f"[DEBUG] [{self.prefix}] {msg % args if args else msg}")


def backend_order(requested: str) -> List[str]:
    requested = str(requested).strip().lower()
    if requested == "real":
        return ["real"]
    if requested == "mock":
        return ["mock"]
    return ["real", "mock"]


def add_common_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["auto", "mock", "real"], default="auto")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)


def add_camera_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rgb-device", default=DEFAULT_RGB_DEVICE)
    parser.add_argument("--depth-device", default=DEFAULT_DEPTH_DEVICE)
    parser.add_argument("--ir-device", default=DEFAULT_IR_DEVICE)
    parser.add_argument("--rgb-in-w", type=int, default=DEFAULT_RGB_IN_W)
    parser.add_argument("--rgb-in-h", type=int, default=DEFAULT_RGB_IN_H)
    parser.add_argument("--rgb-out-w", type=int, default=DEFAULT_RGB_OUT_W)
    parser.add_argument("--rgb-out-h", type=int, default=DEFAULT_RGB_OUT_H)
    parser.add_argument("--rgb-fps", type=int, default=DEFAULT_RGB_FPS)
    parser.add_argument("--depth-width", type=int, default=DEFAULT_DEPTH_W)
    parser.add_argument("--depth-height", type=int, default=DEFAULT_DEPTH_H)
    parser.add_argument("--depth-fps", type=int, default=DEFAULT_DEPTH_FPS)
    parser.add_argument("--ir-in-w", type=int, default=DEFAULT_IR_IN_W)
    parser.add_argument("--ir-in-h", type=int, default=DEFAULT_IR_IN_H)
    parser.add_argument("--ir-out-w", type=int, default=DEFAULT_IR_OUT_W)
    parser.add_argument("--ir-out-h", type=int, default=DEFAULT_IR_OUT_H)
    parser.add_argument("--ir-fps", type=int, default=DEFAULT_IR_FPS)


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", default="")
    parser.add_argument("--model-width", type=int, default=DEFAULT_MODEL_WIDTH)
    parser.add_argument("--model-height", type=int, default=DEFAULT_MODEL_HEIGHT)
    parser.add_argument("--conf-thres", type=float, default=DEFAULT_CONF_THRES)
    parser.add_argument("--iou-thres", type=float, default=DEFAULT_IOU_THRES)
    parser.add_argument("--class-num", type=int, default=DEFAULT_CLASS_NUM)


def print_header(title: str, args: argparse.Namespace) -> None:
    emit_line("=" * 72)
    emit_line(title)
    emit_line("=" * 72)
    emit_line(f"requested_backend={args.backend}")
    if hasattr(args, "iterations"):
        emit_line(f"iterations={args.iterations}")


def safe_text(text: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return str(text).encode(encoding, errors="replace").decode(encoding, errors="replace")


def emit_line(text: str = "") -> None:
    print(safe_text(text))


def print_step(name: str, status: str, detail: str = "") -> None:
    message = f"{status:8s} | {name}"
    if detail:
        message += f" | {detail}"
    emit_line(message)


def print_summary(requested: str, resolved: str, overall: str, details: List[Tuple[str, str]]) -> None:
    emit_line("-" * 72)
    emit_line(f"requested={requested} resolved={resolved} overall={overall}")
    for key, value in details:
        emit_line(f"{key}: {value}")
    emit_line("-" * 72)


def describe_frame(frame: Any) -> str:
    shape = getattr(frame, "shape", None)
    dtype = getattr(frame, "dtype", None)
    return f"shape={shape} dtype={dtype}"


def safe_release(obj: Any) -> None:
    if obj is None:
        return
    try:
        obj.release()
    except Exception:
        pass


def import_camera_classes(backend: str):
    if backend == "mock":
        module = importlib.import_module("vision_module.backend.camera.mock")
        return module.MockCamera, module.MockCamera, module.MockCamera
    color = importlib.import_module("vision_module.backend.camera.ColorCamera")
    ir = importlib.import_module("vision_module.backend.camera.IRCamera")
    depth = importlib.import_module("vision_module.backend.camera.RealSenseDepthCamera")
    return color.ColorCamera, ir.IRCamera, depth.RealSenseDepthCamera


def import_predictor_class(backend: str):
    if backend == "mock":
        module = importlib.import_module("vision_module.backend.predictor.mock")
        return module.MockPredictor
    module = importlib.import_module("vision_module.backend.predictor.QNNPredictor")
    return module.QNN_YOLO_Segment_Predictor


def make_rgb_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "device": args.rgb_device,
        "in_w": args.rgb_in_w,
        "in_h": args.rgb_in_h,
        "out_w": args.rgb_out_w,
        "out_h": args.rgb_out_h,
        "fps": args.rgb_fps,
        "format": "RGB",
        "in_format": "YUY2",
    }


def make_depth_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "width": args.depth_width,
        "height": args.depth_height,
        "fps": args.depth_fps,
    }


def make_ir_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "device": args.ir_device,
        "in_w": args.ir_in_w,
        "in_h": args.ir_in_h,
        "out_w": args.ir_out_w,
        "out_h": args.ir_out_h,
        "fps": args.ir_fps,
        "in_format": "GRAY8",
        "format": "BGR",
    }


def make_model_profile(args: argparse.Namespace) -> SingleModelConfig:
    return SingleModelConfig(
        target_model=args.model_path,
        width=args.model_width,
        height=args.model_height,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        class_num=args.class_num,
        classes=None,
    )


def build_test_config(args: argparse.Namespace) -> VisionServiceConfig:
    runtime = RuntimeConfig(
        project_root=str(VISTA_ROOT),
        log_enabled=False,
        debug=False,
        keep_preview_after_stop=False,
        keep_model_hot_in_standby=False,
        enable_infer_during_hot_standby=False,
    )
    camera = CameraConfig(
        streams={
            "rgb": ColorCameraConfig(
                source=args.rgb_device,
                in_w=args.rgb_in_w,
                in_h=args.rgb_in_h,
                out_w=args.rgb_out_w,
                out_h=args.rgb_out_h,
                fps=args.rgb_fps,
                enable=True,
            ),
            "depth": DepthCameraConfig(
                source=args.depth_device,
                width=args.depth_width,
                height=args.depth_height,
                fps=args.depth_fps,
                enable=False,
            ),
            "grey": IRCameraConfig(
                source=args.ir_device,
                in_w=args.ir_in_w,
                in_h=args.ir_in_h,
                out_w=args.ir_out_w,
                out_h=args.ir_out_h,
                in_format="GRAY8",
                format="BGR",
                fps=args.ir_fps,
                enable=False,
            ),
        }
    )
    model = ModelConfig(active_model="test_model", profiles={"test_model": make_model_profile(args)})
    debug = DebugConfig(preview=False, draw_boxes=False, draw_masks=False)
    return VisionServiceConfig(runtime=runtime, camera=camera, model=model, debug=debug)


def try_with_backends(
    requested: str,
    real_factory: Callable[[], Any],
    mock_factory: Callable[[], Any],
) -> Tuple[Any, AttemptResult]:
    last_error = ""
    for backend in backend_order(requested):
        try:
            instance = real_factory() if backend == "real" else mock_factory()
            detail = "ok"
            if requested == "auto" and backend == "mock" and last_error:
                detail = f"fallback_from_real: {last_error}"
            return instance, AttemptResult(requested=requested, resolved=backend, ok=True, detail=detail)
        except Exception as exc:
            last_error = str(exc)
            if backend == "real" and requested == "auto":
                print_step("backend fallback", "WARN", f"real unavailable: {exc}")
                continue
            return None, AttemptResult(requested=requested, resolved=backend, ok=False, detail=str(exc))
    return None, AttemptResult(requested=requested, resolved="mock", ok=False, detail=last_error or "unknown error")


def patch_engine_backends(engine_module: Any, camera_backend: str, predictor_backend: str) -> None:
    color_cls, ir_cls, depth_cls = import_camera_classes(camera_backend)
    predictor_cls = import_predictor_class(predictor_backend)
    engine_module.ColorCamera = color_cls
    engine_module.IRCamera = ir_cls
    engine_module.HardwareCamera = color_cls
    engine_module.RealSenseDepthCamera = depth_cls
    engine_module.QNN_YOLO_Segment_Predictor = predictor_cls
