#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import threading

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

os.environ['PYTHONIOENCODING'] = 'utf-8'

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VISION_ROOT = os.path.dirname(CURRENT_DIR)
VISTA_ROOT = os.path.dirname(VISION_ROOT)
sys.path.insert(0, VISTA_ROOT)
sys.path.insert(0, VISION_ROOT)

from vision_module.config.board_config import CONFIG
from vision_module.backend.vision_engine import VisionEngine


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("test_pipeline")


class SimpleLogger:
    def info(self, msg, *args):
        print(f"INFO | {msg % args if args else msg}")

    def warning(self, msg, *args):
        print(f"WARN | {msg % args if args else msg}")

    def error(self, msg, *args):
        print(f"ERROR | {msg % args if args else msg}")

    def debug(self, msg, *args):
        print(f"DEBUG | {msg % args if args else msg}")


def test_pipeline_init(logger):
    logger.info("=" * 50)
    logger.info("Testing Pipeline Initialization...")

    engine = None
    error_msg = ""

    try:
        engine = VisionEngine(CONFIG, SimpleLogger())
        logger.info("VisionEngine created successfully")
    except Exception as e:
        error_msg = f"Init failed: {e}"
        logger.error(error_msg)

    return engine, error_msg


def test_pipeline_camera_setup(logger, engine):
    logger.info("=" * 50)
    logger.info("Testing Camera Setup...")

    if engine is None:
        logger.error("No engine available")
        return False

    try:
        from vision_module.backend.camera.mock import MockCamera
        logger.info("Using MockCamera for test")

        rgb_config = CONFIG.camera.streams.get("rgb")
        if rgb_config:
            try:
                rgb_cam = MockCamera(
                    out_w=rgb_config.out_w,
                    out_h=rgb_config.out_h,
                    width=rgb_config.in_w,
                    height=rgb_config.in_h
                )
                engine.cams["rgb"] = rgb_cam
                logger.info("RGB camera added to engine")
            except Exception as e:
                logger.warning(f"RGB camera setup error: {e}")

        depth_config = CONFIG.camera.streams.get("depth")
        if depth_config and getattr(depth_config, 'enable', False):
            try:
                depth_cam = MockCamera(
                    out_w=depth_config.out_w if hasattr(depth_config, 'out_w') else 640,
                    out_h=depth_config.out_h if hasattr(depth_config, 'out_h') else 640,
                    width=depth_config.width if hasattr(depth_config, 'width') else depth_config.in_w,
                    height=depth_config.height if hasattr(depth_config, 'height') else depth_config.in_h
                )
                engine.cams["depth"] = depth_cam
                logger.info("Depth camera added to engine")
            except Exception as e:
                logger.warning(f"Depth camera setup skipped: {e}")

        ir_config = CONFIG.camera.streams.get("ir") or CONFIG.camera.streams.get("grey")
        if ir_config and getattr(ir_config, 'enable', False):
            try:
                ir_cam = MockCamera(
                    out_w=ir_config.out_w,
                    out_h=ir_config.out_h,
                    width=ir_config.in_w,
                    height=ir_config.in_h
                )
                engine.cams["ir"] = ir_cam
                logger.info("IR camera added to engine")
            except Exception as e:
                logger.warning(f"IR camera setup skipped: {e}")

        return len(engine.cams) > 0

    except Exception as e:
        logger.error(f"Camera setup failed: {e}")
        return False


def test_pipeline_model_setup(logger, engine):
    logger.info("=" * 50)
    logger.info("Testing Model Setup...")

    if engine is None:
        logger.error("No engine available")
        return False

    try:
        model_name = CONFIG.model.active_model
        logger.info(f"Loading model: {model_name}")
        engine.set_model(model_name, enable=True)
        logger.info(f"Model {model_name} loaded")
        return True
    except Exception as e:
        logger.warning(f"Model setup failed (may be mock env): {e}")
        return False


def test_pipeline_inference_loop(logger, engine):
    logger.info("=" * 50)
    logger.info("Testing Inference Loop...")

    if engine is None:
        logger.error("No engine available")
        return None

    results = {
        "iterations": 0,
        "frames_received": 0,
        "inference_done": 0,
        "errors": 0,
    }

    try:
        engine.init()
        engine.start()
        logger.info("Engine started, running inference loop...")

        num_iterations = 20
        for i in range(num_iterations):
            results["iterations"] += 1
            try:
                frames, infer_res = engine.get_new_data()
                if frames:
                    results["frames_received"] += 1
                if infer_res is not None:
                    results["inference_done"] += 1
                time.sleep(0.05)
            except Exception as e:
                results["errors"] += 1
                logger.debug(f"Iteration {i} error: {e}")

        engine.stop()
        logger.info(f"Loop completed: {results['iterations']} iterations")

    except Exception as e:
        logger.error(f"Inference loop failed: {e}")
        return None

    return results


def test_pipeline_end_to_end(logger):
    logger.info("=" * 50)
    logger.info("Testing End-to-End Pipeline...")

    results = {
        "init_ok": False,
        "camera_ok": False,
        "model_ok": False,
        "inference_ok": False,
    }

    engine, error_msg = test_pipeline_init(logger)
    if engine is None:
        logger.error(f"Init failed: {error_msg}")
        return results

    results["init_ok"] = True

    camera_ok = test_pipeline_camera_setup(logger, engine)
    results["camera_ok"] = camera_ok

    model_ok = test_pipeline_model_setup(logger, engine)
    results["model_ok"] = model_ok

    inference_results = test_pipeline_inference_loop(logger, engine)
    if inference_results and inference_results["frames_received"] > 0:
        results["inference_ok"] = True
        logger.info(f"Inference results: {inference_results}")

    return results


def run_all_tests(logger):
    logger.info("Starting VISTA Pipeline Integration Tests")
    logger.info(f"Active model: {CONFIG.model.active_model}")
    logger.info(f"RGB stream: {CONFIG.camera.streams.get('rgb')}")

    results = test_pipeline_end_to_end(logger)

    logger.info("=" * 50)
    logger.info("Test Summary:")
    logger.info("=" * 50)
    logger.info(f"Init:        {'PASS' if results['init_ok'] else 'FAIL'}")
    logger.info(f"Camera:      {'PASS' if results['camera_ok'] else 'FAIL'}")
    logger.info(f"Model:       {'PASS' if results['model_ok'] else 'FAIL'}")
    logger.info(f"Inference:   {'PASS' if results['inference_ok'] else 'FAIL'}")

    all_passed = all(results.values())
    return all_passed


if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Starting VISTA Pipeline Integration Tests")

    try:
        success = run_all_tests(logger)
        if success:
            logger.info("All pipeline tests PASSED")
            sys.exit(0)
        else:
            logger.warning("Some pipeline tests FAILED")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
