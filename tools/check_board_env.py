#!/usr/bin/env python3
"""Check the board runtime expected by the robot stack."""

from __future__ import annotations

import ctypes.util
import importlib
import os
import sys
from typing import Iterable, Tuple


PY_MODULES: Tuple[Tuple[str, str, bool], ...] = (
    ("numpy", "numeric / vision geometry", True),
    ("cv2", "OpenCV image pipeline", True),
    ("yaml", "YAML config loader", True),
    ("requests", "HTTP helpers", True),
    ("serial", "STM32 UART full mode", True),
    ("paho.mqtt.client", "mobile gateway MQTT mode", True),
    ("psutil", "runtime process helpers", True),
    ("onnxruntime", "offline/model helper paths", True),
    ("aidlite", "AidLux QNN inference runtime", True),
    ("aidcv", "AidLux camera/CV runtime", True),
    ("scipy", "vision geometry helpers", True),
    ("shapely", "table edge geometry helpers", True),
    ("pyrealsense2", "RealSense depth camera / bag replay", False),
)

SHARED_LIBS: Tuple[Tuple[str, bool], ...] = (
    ("aidlite", True),
    ("aidlite_qnn236", True),
    ("aidrtcm", True),
    ("aidlms", True),
    ("GLdispatch", False),
)


def check_module(name: str) -> Tuple[bool, str]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - this is a diagnostics script.
        return False, f"{type(exc).__name__}: {exc}"
    version = getattr(module, "__version__", "")
    return True, str(version)


def check_lib(name: str) -> Tuple[bool, str]:
    found = ctypes.util.find_library(name)
    if found:
        return True, found
    local_candidates = (
        f"/usr/local/lib/lib{name}.so",
        f"/lib/aarch64-linux-gnu/lib{name}.so.0",
        f"/lib/aarch64-linux-gnu/lib{name}.so",
    )
    for path in local_candidates:
        if os.path.exists(path):
            return True, path
    return False, "not found by ldconfig/ctypes"


def print_section(title: str) -> None:
    print(f"\n== {title} ==")


def main() -> int:
    failures = 0
    print(f"python: {sys.executable} ({sys.version.split()[0]})")

    print_section("Python modules")
    for module, purpose, required in PY_MODULES:
        ok, detail = check_module(module)
        level = "OK" if ok else ("MISS" if required else "OPTIONAL")
        print(f"{level:8s} {module:20s} {purpose:36s} {detail}")
        if required and not ok:
            failures += 1

    print_section("Shared libraries")
    for lib, required in SHARED_LIBS:
        ok, detail = check_lib(lib)
        level = "OK" if ok else ("MISS" if required else "OPTIONAL")
        print(f"{level:8s} lib{lib}.so{'':13s} {detail}")
        if required and not ok:
            failures += 1

    print_section("Result")
    if failures:
        print(f"FAIL: {failures} required environment item(s) are missing.")
        print("Install pip deps with: /usr/bin/python3 -m pip install -r requirements-runtime.txt")
        print("Restore AidLux/QNN runtime from board image, apt packages, or board backup.")
        return 1

    print("OK: required runtime checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
