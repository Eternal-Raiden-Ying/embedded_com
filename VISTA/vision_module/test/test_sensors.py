#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from typing import Tuple

from test_support import (
    EXIT_FAIL,
    EXIT_INTERRUPT,
    EXIT_OK,
    add_camera_args,
    add_common_backend_args,
    describe_frame,
    import_camera_classes,
    make_depth_kwargs,
    make_ir_kwargs,
    make_rgb_kwargs,
    print_header,
    print_step,
    print_summary,
    safe_release,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VISTA sensor backend smoke test")
    add_common_backend_args(parser)
    add_camera_args(parser)
    return parser


def choose_camera_class(requested: str, stream_name: str):
    from test_support import try_with_backends

    def real_factory():
        color_cls, ir_cls, depth_cls = import_camera_classes("real")
        if stream_name == "depth":
            return depth_cls
        if stream_name == "ir":
            return ir_cls
        return color_cls

    def mock_factory():
        color_cls, ir_cls, depth_cls = import_camera_classes("mock")
        if stream_name == "depth":
            return depth_cls
        if stream_name == "ir":
            return ir_cls
        return color_cls

    return try_with_backends(requested, real_factory, mock_factory)


def read_twice(camera) -> Tuple[bool, str]:
    frame1 = camera.read_frame()
    frame2 = camera.read_frame()
    if frame1 is None or getattr(frame1, "size", 0) == 0:
        return False, "first read empty"
    if frame2 is None or getattr(frame2, "size", 0) == 0:
        return False, "second read empty"
    return True, f"frame1 {describe_frame(frame1)} | frame2 {describe_frame(frame2)}"


def optional_controls(camera) -> str:
    checks = []
    for name, value in [("set_exposure", 10), ("set_brightness", 0)]:
        if hasattr(camera, name):
            try:
                getattr(camera, name)(value)
                checks.append(f"{name}=ok")
            except Exception as exc:
                checks.append(f"{name}=fail({exc})")
        else:
            checks.append(f"{name}=n/a")
    return ", ".join(checks)


def test_stream(stream_name: str, kwargs: dict, requested_backend: str, rgb_controls: bool = False) -> dict:
    result = {
        "stream": stream_name,
        "requested": requested_backend,
        "resolved": "",
        "status": "FAIL",
        "detail": "",
    }
    cls, backend_result = choose_camera_class(requested_backend, stream_name)
    result["resolved"] = backend_result.resolved
    if not backend_result.ok or cls is None:
        result["detail"] = backend_result.detail
        print_step(stream_name, "FAIL", backend_result.detail)
        return result

    camera = None
    try:
        camera = cls(**kwargs)
        ok, detail = read_twice(camera)
        if not ok:
            result["detail"] = detail
            print_step(stream_name, "FAIL", detail)
            return result
        if rgb_controls:
            detail = f"{detail} | {optional_controls(camera)}"
        safe_release(camera)
        safe_release(camera)
        result["status"] = "REAL PASS" if backend_result.resolved == "real" else "MOCK PASS"
        result["detail"] = detail
        print_step(stream_name, result["status"], detail)
        return result
    except Exception as exc:
        result["detail"] = str(exc)
        print_step(stream_name, "FAIL", str(exc))
        return result
    finally:
        safe_release(camera)


def main() -> int:
    args = build_parser().parse_args()
    print_header("VISTA Sensor Backend Test", args)

    results = [
        test_stream("rgb", make_rgb_kwargs(args), args.backend, rgb_controls=True),
        test_stream("depth", make_depth_kwargs(args), args.backend),
        test_stream("ir", make_ir_kwargs(args), args.backend),
    ]

    statuses = [item["status"] for item in results]
    resolved_set = {item["resolved"] for item in results if item["resolved"]}
    resolved = resolved_set.pop() if len(resolved_set) == 1 else "mixed"
    overall = "PASS" if all(status.endswith("PASS") for status in statuses) else "FAIL"
    print_summary(
        args.backend,
        resolved,
        overall,
        [(item["stream"], f"{item['status']} | {item['detail']}") for item in results],
    )
    return EXIT_OK if overall == "PASS" else EXIT_FAIL


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(EXIT_INTERRUPT)
