#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import time
from typing import Tuple

import numpy as np

from test_support import (
    EXIT_FAIL,
    EXIT_INTERRUPT,
    EXIT_OK,
    EXIT_USAGE,
    add_common_backend_args,
    add_model_args,
    import_predictor_class,
    make_model_profile,
    print_header,
    print_step,
    print_summary,
    safe_release,
    try_with_backends,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VISTA predictor backend smoke test")
    add_common_backend_args(parser)
    add_model_args(parser)
    return parser


def build_dummy_frame(args: argparse.Namespace) -> np.ndarray:
    return np.random.randint(0, 255, (args.model_height, args.model_width, 3), dtype=np.uint8)


def choose_predictor(requested_backend: str, args: argparse.Namespace):
    profile = make_model_profile(args)
    predictor_type = getattr(args, "predictor_type", "detect") or "detect"

    def real_factory():
        if not args.model_path:
            raise RuntimeError("--model-path is required for real predictor")
        cls = import_predictor_class("real", predictor_type=predictor_type)
        return cls(profile)

    def mock_factory():
        cls = import_predictor_class("mock")
        return cls(profile)

    return try_with_backends(requested_backend, real_factory, mock_factory)


def verify_output(boxes, masks) -> Tuple[bool, str]:
    if boxes is None or masks is None:
        return False, "boxes or masks is None"
    return True, f"boxes={len(boxes)} masks={len(masks)}"


def run_inference(predictor, args: argparse.Namespace) -> Tuple[bool, str]:
    frame = build_dummy_frame(args)
    ok = predictor.is_ready()
    if not ok:
        return False, "predictor not ready"

    boxes, masks = predictor.predict_frame(frame)
    output_ok, detail = verify_output(boxes, masks)
    if not output_ok:
        return False, detail

    times_ms = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        boxes, masks = predictor.predict_frame(frame)
        times_ms.append((time.perf_counter() - start) * 1000.0)
    output_ok, detail = verify_output(boxes, masks)
    if not output_ok:
        return False, detail

    return True, (
        f"{detail} avg_ms={sum(times_ms)/len(times_ms):.2f} "
        f"min_ms={min(times_ms):.2f} max_ms={max(times_ms):.2f}"
    )


def main() -> int:
    args = build_parser().parse_args()
    print_header("VISTA Predictor Backend Test", args)

    predictor, backend_result = choose_predictor(args.backend, args)
    if predictor is None:
        print_step("predictor_init", "FAIL", backend_result.detail)
        print_summary(args.backend, backend_result.resolved, "FAIL", [("init", backend_result.detail)])
        return EXIT_USAGE if "required for real predictor" in backend_result.detail else EXIT_FAIL

    resolved = backend_result.resolved
    print_step("predictor_init", "PASS", f"resolved={resolved} type={type(predictor).__name__}")
    try:
        ok, detail = run_inference(predictor, args)
        print_step("predictor_inference", "PASS" if ok else "FAIL", detail)
        release_ok = True
        release_detail = "release ok"
        try:
            predictor.release()
            predictor.release()
        except Exception as exc:
            release_ok = False
            release_detail = str(exc)
        print_step("predictor_release", "PASS" if release_ok else "FAIL", release_detail)

        overall = "PASS" if ok and release_ok else "FAIL"
        print_summary(
            args.backend,
            resolved,
            overall,
            [
                ("init", f"type={type(predictor).__name__}"),
                ("inference", detail),
                ("release", release_detail),
            ],
        )
        return EXIT_OK if overall == "PASS" else EXIT_FAIL
    finally:
        safe_release(predictor)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(EXIT_INTERRUPT)
