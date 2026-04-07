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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VISION_ROOT = os.path.dirname(CURRENT_DIR)
VISTA_ROOT = os.path.dirname(VISION_ROOT)
sys.path.insert(0, VISTA_ROOT)
sys.path.insert(0, VISION_ROOT)

from vision_module.config.board_config import CONFIG


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("test_predictor")


def test_predictor_init(logger):
    logger.info("=" * 50)
    logger.info("Testing Predictor Initialization...")

    predictor = None
    is_mock = False
    error_msg = ""

    try:
        from vision_module.backend.predictor import QNN_YOLO_Segment_Predictor
        predictor = QNN_YOLO_Segment_Predictor(CONFIG.model.active_model)
        logger.info(f"Predictor initialized: {type(predictor).__name__}")
        is_mock = "Mock" in type(predictor).__name__
    except ImportError as e:
        error_msg = f"Import failed: {e}"
        logger.error(error_msg)
    except Exception as e:
        error_msg = f"Init failed: {e}"
        logger.error(error_msg)

    return predictor, is_mock, error_msg


def test_predictor_inference(logger, predictor):
    logger.info("=" * 50)
    logger.info("Testing Predictor Inference...")

    if predictor is None:
        logger.error("No predictor available for inference test")
        return None

    test_results = {
        "warmup": {"time_ms": 0, "boxes": 0, "masks": 0},
        "single_frame": {"time_ms": 0, "boxes": 0, "masks": 0},
        "avg_time_ms": 0.0,
        "min_time_ms": 0.0,
        "max_time_ms": 0.0,
    }

    if not predictor.is_ready():
        logger.warning("Predictor not ready after init")
        return test_results

    try:
        h, w = CONFIG.camera.streams["rgb"].out_h, CONFIG.camera.streams["rgb"].out_w
        dummy_frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        logger.info("Running warmup inference...")
        boxes, masks = predictor.predict_frame(dummy_frame)
        test_results["warmup"]["boxes"] = len(boxes)
        test_results["warmup"]["masks"] = len(masks)
        logger.info(f"Warmup: {len(boxes)} boxes, {len(masks)} masks")

        num_runs = 10
        times = []

        logger.info(f"Running {num_runs} inference iterations...")
        for i in range(num_runs):
            start = time.perf_counter()
            boxes, masks = predictor.predict_frame(dummy_frame)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        test_results["single_frame"]["boxes"] = len(boxes)
        test_results["single_frame"]["masks"] = len(masks)

        test_results["avg_time_ms"] = sum(times) / len(times)
        test_results["min_time_ms"] = min(times)
        test_results["max_time_ms"] = max(times)

        logger.info(f"Avg inference time: {test_results['avg_time_ms']:.2f}ms")
        logger.info(f"Min/Max: {test_results['min_time_ms']:.2f}ms / {test_results['max_time_ms']:.2f}ms")
        logger.info(f"Last run: {len(boxes)} boxes, {len(masks)} masks")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return None

    return test_results


def test_predictor_release(logger, predictor):
    logger.info("=" * 50)
    logger.info("Testing Predictor Release...")

    if predictor is None:
        logger.warning("No predictor to release")
        return True

    try:
        predictor.release()
        logger.info("Predictor released successfully")
        return True
    except Exception as e:
        logger.error(f"Release failed: {e}")
        return False


def run_all_tests(logger):
    predictor, is_mock, error_msg = test_predictor_init(logger)

    if predictor is None:
        logger.error(f"Predictor init failed: {error_msg}")
        return False

    results = test_predictor_inference(logger, predictor)

    release_ok = test_predictor_release(logger, predictor)

    logger.info("=" * 50)
    logger.info("Test Summary:")
    logger.info("=" * 50)
    logger.info(f"Type:        {'MOCK' if is_mock else 'QNN (Hardware)'}")
    logger.info(f"Init:        {'PASS' if predictor else 'FAIL'}")
    if results:
        logger.info(f"Inference:   PASS")
        logger.info(f"Avg Time:    {results['avg_time_ms']:.2f}ms")
        logger.info(f"Min/Max:     {results['min_time_ms']:.2f}ms / {results['max_time_ms']:.2f}ms")
    else:
        logger.info(f"Inference:   FAIL")
    logger.info(f"Release:     {'PASS' if release_ok else 'FAIL'}")

    return predictor is not None and results is not None and release_ok


if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Starting VISTA Predictor Tests")

    try:
        success = run_all_tests(logger)
        if success:
            logger.info("All predictor tests PASSED")
            sys.exit(0)
        else:
            logger.warning("Some predictor tests FAILED")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        sys.exit(1)
