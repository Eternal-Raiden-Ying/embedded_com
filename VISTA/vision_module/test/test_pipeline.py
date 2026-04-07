#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib
import sys
import time
from typing import Tuple

from test_support import (
    EXIT_FAIL,
    EXIT_INTERRUPT,
    EXIT_OK,
    EXIT_USAGE,
    PrintLogger,
    add_camera_args,
    add_common_backend_args,
    add_model_args,
    build_test_config,
    describe_frame,
    import_camera_classes,
    import_predictor_class,
    make_model_profile,
    patch_engine_backends,
    print_header,
    print_step,
    print_summary,
    safe_release,
    try_with_backends,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VISTA pipeline backend smoke test")
    add_common_backend_args(parser)
    add_camera_args(parser)
    add_model_args(parser)
    return parser


def build_engine_camera_cfg(args: argparse.Namespace) -> dict:
    return {
        "source": args.rgb_device,
        "in_w": args.rgb_in_w,
        "in_h": args.rgb_in_h,
        "out_w": args.rgb_out_w,
        "out_h": args.rgb_out_h,
        "fps": args.rgb_fps,
        "format": "RGB",
        "in_format": "YUY2",
        "crop_x": 0,
        "crop_y": 0,
        "crop_w": 0,
        "crop_h": 0,
    }


def build_direct_camera_kwargs(args: argparse.Namespace) -> dict:
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


def resolve_camera_backend(requested: str, args: argparse.Namespace):
    rgb_kwargs = build_direct_camera_kwargs(args)

    def real_factory():
        hardware_cls, _ = import_camera_classes("real")
        camera = hardware_cls(**rgb_kwargs)
        frame = camera.read_frame()
        if frame is None or getattr(frame, "size", 0) == 0:
            safe_release(camera)
            raise RuntimeError("camera opened but returned empty frame")
        safe_release(camera)
        return hardware_cls

    def mock_factory():
        hardware_cls, _ = import_camera_classes("mock")
        camera = hardware_cls(**rgb_kwargs)
        frame = camera.read_frame()
        if frame is None or getattr(frame, "size", 0) == 0:
            safe_release(camera)
            raise RuntimeError("mock camera returned empty frame")
        safe_release(camera)
        return hardware_cls

    return try_with_backends(requested, real_factory, mock_factory)


def resolve_predictor_backend(requested: str, args: argparse.Namespace):
    profile = make_model_profile(args)

    def real_factory():
        if not args.model_path:
            raise RuntimeError("--model-path is required for real predictor")
        predictor_cls = import_predictor_class("real")
        predictor = predictor_cls(profile)
        if not predictor.is_ready():
            safe_release(predictor)
            raise RuntimeError("predictor not ready")
        safe_release(predictor)
        return predictor_cls

    def mock_factory():
        predictor_cls = import_predictor_class("mock")
        predictor = predictor_cls(profile)
        if not predictor.is_ready():
            safe_release(predictor)
            raise RuntimeError("mock predictor not ready")
        safe_release(predictor)
        return predictor_cls

    return try_with_backends(requested, real_factory, mock_factory)


def load_engine_module():
    return importlib.import_module("vision_module.backend.vision_engine")


def phase_camera_only(engine_module, cfg, camera_kwargs: dict) -> Tuple[bool, str]:
    engine = engine_module.VisionEngine(cfg, logger=PrintLogger("pipeline"))
    try:
        engine.set_camera("rgb", True, cfg=camera_kwargs)
        if "rgb" not in engine.cams:
            return False, "engine.cams missing rgb"
        frame = engine.cams["rgb"].read_frame()
        if frame is None or getattr(frame, "size", 0) == 0:
            return False, "camera returned empty frame"
        return True, describe_frame(frame)
    finally:
        engine.stop()
        engine.stop()


def phase_predictor_only(engine_module, cfg) -> Tuple[bool, str]:
    engine = engine_module.VisionEngine(cfg, logger=PrintLogger("pipeline"))
    try:
        engine.set_model("test_model", True)
        predictor = engine.predictor
        if predictor is None:
            return False, "engine.predictor is None"
        if not predictor.is_ready():
            return False, "predictor not ready"
        return True, type(predictor).__name__
    finally:
        engine.stop()
        engine.stop()


def phase_combined(engine_module, cfg, camera_kwargs: dict, iterations: int) -> Tuple[bool, str]:
    engine = engine_module.VisionEngine(cfg, logger=PrintLogger("pipeline"))
    frames_seen = 0
    infer_seen = 0
    try:
        engine.set_camera("rgb", True, cfg=camera_kwargs)
        engine.set_model("test_model", True)
        engine.set_inference_enabled(True)
        engine.init()
        engine.start()
        for _ in range(iterations):
            frames, infer_res = engine.get_new_data()
            if frames:
                frames_seen += 1
            if infer_res is not None:
                infer_seen += 1
            time.sleep(0.05)
        if frames_seen <= 0:
            return False, "no frames observed"
        return True, f"frames_seen={frames_seen} infer_seen={infer_seen}"
    finally:
        engine.stop()
        engine.stop()


def main() -> int:
    args = build_parser().parse_args()
    print_header("VISTA Pipeline Backend Test", args)

    camera_backend_cls, camera_result = resolve_camera_backend(args.backend, args)
    predictor_backend_cls, predictor_result = resolve_predictor_backend(args.backend, args)

    if camera_backend_cls is None and predictor_backend_cls is None:
        print_step("camera_backend", "FAIL", camera_result.detail)
        print_step("predictor_backend", "FAIL", predictor_result.detail)
        print_summary(
            args.backend,
            "none",
            "FAIL",
            [("camera", camera_result.detail), ("predictor", predictor_result.detail)],
        )
        return EXIT_USAGE if "--model-path is required for real predictor" in predictor_result.detail else EXIT_FAIL

    print_step("camera_backend", "PASS" if camera_backend_cls else "FAIL", camera_result.detail)
    print_step("predictor_backend", "PASS" if predictor_backend_cls else "FAIL", predictor_result.detail)

    cfg = build_test_config(args)
    camera_kwargs = build_engine_camera_cfg(args)
    engine_module = load_engine_module()
    camera_backend = camera_result.resolved if camera_backend_cls else "mock"
    predictor_backend = predictor_result.resolved if predictor_backend_cls else "mock"
    patch_engine_backends(engine_module, camera_backend, predictor_backend)

    phase_results = []

    if camera_backend_cls is not None:
        ok, detail = phase_camera_only(engine_module, cfg, camera_kwargs)
        phase_results.append(("camera_only", ok, detail))
        print_step("camera_only", "PASS" if ok else "FAIL", detail)
    else:
        phase_results.append(("camera_only", False, "camera backend unavailable"))
        print_step("camera_only", "FAIL", "camera backend unavailable")

    if predictor_backend_cls is not None:
        ok, detail = phase_predictor_only(engine_module, cfg)
        phase_results.append(("predictor_only", ok, detail))
        print_step("predictor_only", "PASS" if ok else "FAIL", detail)
    else:
        phase_results.append(("predictor_only", False, "predictor backend unavailable"))
        print_step("predictor_only", "FAIL", "predictor backend unavailable")

    if camera_backend_cls is not None and predictor_backend_cls is not None:
        ok, detail = phase_combined(engine_module, cfg, camera_kwargs, args.iterations)
        phase_results.append(("combined", ok, detail))
        print_step("combined", "PASS" if ok else "FAIL", detail)
    else:
        phase_results.append(("combined", False, "combined phase skipped"))
        print_step("combined", "FAIL", "combined phase skipped")

    pass_count = sum(1 for _, ok, _ in phase_results if ok)
    if pass_count == len(phase_results):
        overall = "PASS"
        exit_code = EXIT_OK
    elif pass_count > 0:
        overall = "PARTIAL"
        exit_code = EXIT_FAIL
    else:
        overall = "FAIL"
        exit_code = EXIT_FAIL

    print_summary(
        args.backend,
        f"camera={camera_backend} predictor={predictor_backend}",
        overall,
        [(name, detail) for name, _, detail in phase_results],
    )
    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(EXIT_INTERRUPT)
