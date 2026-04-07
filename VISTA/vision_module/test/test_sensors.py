#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    try:
        import aidcv as cv2
    except ImportError:
        cv2 = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VISION_ROOT = os.path.dirname(CURRENT_DIR)
VISTA_ROOT = os.path.dirname(VISION_ROOT)
sys.path.insert(0, VISTA_ROOT)
sys.path.insert(0, VISION_ROOT)

from test_support import apply_backend_env, build_camera_instance, safe_release

TEST_BACKEND = apply_backend_env()

from vision_module.config.board_config import CONFIG


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("test_sensors")


class SensorTestResult:
    def __init__(self, stream_name: str):
        self.stream_name = stream_name
        self.hw_available = False
        self.mock_available = False
        self.frame_captured = False
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.error_msg = ""


def test_rgb_stream(logger) -> SensorTestResult:
    result = SensorTestResult("RGB")
    logger.info("=" * 50)
    logger.info("Testing RGB Stream...")

    rgb_config = CONFIG.camera.streams.get("rgb")
    if not rgb_config:
        result.error_msg = "No RGB config found"
        logger.error(result.error_msg)
        return result

    result.mock_available = TEST_BACKEND != "real"
    cam = None
    try:
        cam = build_camera_instance("rgb", rgb_config)
        frame = cam.read_frame()
        if frame is not None and getattr(frame, "size", 0) > 0:
            result.frame_captured = True
            result.width = frame.shape[1]
            result.height = frame.shape[0]
            result.hw_available = "Mock" not in type(cam).__name__
            result.mock_available = not result.hw_available
            logger.info(f"RGB frame: {result.width}x{result.height} via {type(cam).__name__}")
        else:
            result.error_msg = "No RGB frame received"
            logger.warning(result.error_msg)
    except Exception as e:
        result.error_msg = f"RGB test failed: {e}"
        logger.warning(result.error_msg)
    finally:
        safe_release(cam)

    return result


def test_depth_stream(logger) -> SensorTestResult:
    result = SensorTestResult("Depth")
    logger.info("=" * 50)
    logger.info("Testing Depth Stream...")

    depth_config = CONFIG.camera.streams.get("depth")
    if not depth_config:
        result.error_msg = "No Depth config found"
        logger.warning(result.error_msg)
        return result

    if not getattr(depth_config, 'enable', True):
        result.error_msg = "Depth camera disabled in config (expected)"
        logger.info(result.error_msg)
        result.mock_available = True
        result.frame_captured = True
        return result

    result.mock_available = TEST_BACKEND != "real"
    cam = None
    try:
        cam = build_camera_instance("depth", depth_config)
        frame = cam.read_frame()
        if frame is not None and getattr(frame, "size", 0) > 0:
            result.frame_captured = True
            result.width = frame.shape[1]
            result.height = frame.shape[0]
            result.hw_available = "Mock" not in type(cam).__name__
            result.mock_available = not result.hw_available
            logger.info(f"Depth frame: {result.width}x{result.height} via {type(cam).__name__}")
        else:
            result.error_msg = "No Depth frame received"
            logger.warning(result.error_msg)
    except Exception as e:
        result.error_msg = f"Depth test failed: {e}"
        logger.warning(result.error_msg)
    finally:
        safe_release(cam)

    return result


def test_ir_stream(logger) -> SensorTestResult:
    result = SensorTestResult("IR")
    logger.info("=" * 50)
    logger.info("Testing IR Stream...")

    ir_config = CONFIG.camera.streams.get("ir")
    if not ir_config:
        ir_config = CONFIG.camera.streams.get("grey")

    if not ir_config:
        result.error_msg = "No IR config found"
        logger.warning(result.error_msg)
        return result

    result.mock_available = TEST_BACKEND != "real"
    cam = None
    try:
        cam = build_camera_instance("grey", ir_config)
        frame = cam.read_frame()
        if frame is not None and getattr(frame, "size", 0) > 0:
            result.frame_captured = True
            result.width = frame.shape[1]
            result.height = frame.shape[0]
            result.hw_available = "Mock" not in type(cam).__name__
            result.mock_available = not result.hw_available
            logger.info(f"IR frame: {result.width}x{result.height} via {type(cam).__name__}")
        else:
            result.error_msg = "No IR frame received"
            logger.warning(result.error_msg)
    except Exception as e:
        result.error_msg = f"IR test failed: {e}"
        logger.warning(result.error_msg)
    finally:
        safe_release(cam)

    return result


def run_all_tests(logger):
    results = []

    rgb_result = test_rgb_stream(logger)
    results.append(rgb_result)

    depth_result = test_depth_stream(logger)
    results.append(depth_result)

    ir_result = test_ir_stream(logger)
    results.append(ir_result)

    logger.info("=" * 50)
    logger.info("Test Summary:")
    logger.info("=" * 50)

    all_passed = True
    for r in results:
        status = "PASS" if r.frame_captured else "FAIL"
        logger.info(f"{r.stream_name:8s}: {status} (hw={r.hw_available}, mock={r.mock_available}, err={r.error_msg})")
        if not r.frame_captured:
            all_passed = False

    return all_passed


if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Starting VISTA Sensor Tests")
    logger.info(f"Backend mode: {TEST_BACKEND}")

    try:
        success = run_all_tests(logger)
        if success:
            logger.info("All sensor tests PASSED")
            sys.exit(0)
        else:
            logger.warning("Some sensor tests FAILED (check hardware availability)")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        sys.exit(1)
