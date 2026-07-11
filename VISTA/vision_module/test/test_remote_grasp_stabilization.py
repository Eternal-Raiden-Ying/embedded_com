#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
VISTA_ROOT = ROOT / "VISTA"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(VISTA_ROOT) not in sys.path:
    sys.path.insert(0, str(VISTA_ROOT))

from vision_module.backend.remote.manager import RemoteManager
from vision_module.config.mode_defaults import build_default_mode_profiles
from vision_module.config.schema import VisionServiceConfig
from vision_module.app.stages.grasp import canonicalize_remote_grasp_target
from vision_module.app.observation.router import ObservationRouter


def test_grasp_remote_depth_profile_is_high_res_and_find_edge_stays_low_res():
    cfg = VisionServiceConfig()
    cfg.camera.streams["depth"].width = 424
    cfg.camera.streams["depth"].height = 240
    cfg.camera.streams["depth"].fps = 15

    profiles = build_default_mode_profiles("yolo26s_detect", cfg)

    assert profiles["GRASP_REMOTE"].camera_overrides["depth"]["width"] == 1280
    assert profiles["GRASP_REMOTE"].camera_overrides["depth"]["height"] == 720
    assert profiles["GRASP_REMOTE"].camera_overrides["depth"]["fps"] == 15
    assert "depth" not in profiles["FIND_EDGE"].camera_overrides
    assert cfg.camera.streams["depth"].width == 424
    assert cfg.camera.streams["depth"].height == 240


def test_capture_metadata_contains_shapes_dtype_and_depth_stats():
    manager = RemoteManager()
    frames = {
        "rgb": np.zeros((720, 1280, 3), dtype=np.uint8),
        "depth": np.array([[0, 1000], [1200, 1300]], dtype=np.uint16),
        "camera_frame_seq": 7,
        "camera_frame_ts_ms": 12345,
        "depth_scale": 0.001,
        "depth_unit": "m",
        "depth_aligned_to_color": "best_effort_true",
        "color_intrinsics": {"width": 1280, "height": 720},
        "depth_intrinsics": {"width": 1280, "height": 720},
    }

    capture = manager._capture_metadata(frames, {"payload": frames})

    assert capture["rgb_shape"] == [720, 1280, 3]
    assert capture["depth_shape"] == [2, 2]
    assert capture["rgb_dtype"] == "uint8"
    assert capture["depth_dtype"] == "uint16"
    assert capture["depth_min"] == 1000.0
    assert capture["depth_max"] == 1300.0
    assert capture["depth_valid_count"] == 3
    assert capture["depth_scale"] == 0.001
    assert capture["depth_aligned_to_color"] == "best_effort_true"


def test_capture_shape_mismatch_is_hard_precheck_failure():
    manager = RemoteManager()
    manager.enabled = True
    frames = {
        "rgb": np.zeros((720, 1280, 3), dtype=np.uint8),
        "depth": np.zeros((240, 424), dtype=np.uint16),
        "camera_frame_seq": 9,
    }

    capture = manager._precheck_capture_shapes(
        frames=frames,
        frame_slot={"payload": frames},
        request_id="rr_test",
        require_depth=True,
    )

    assert capture is None
    result = manager.result_summary()
    assert result["last_ok"] is False
    assert result["last_error"] == "capture_shape_mismatch"
    assert result["result"]["actual_depth_shape"] == [240, 424]
    assert result["result"]["expected_depth_shape"] == [720, 1280]


def test_remote_grasp_target_canonicalization():
    raw = {"x": 24.41, "y": 3.16, "z": 9.35, "pitch": -2.57, "roll": 81.18, "width": 8.68, "depth": 3.0, "score": 0.4692, "dist": 1.56}

    canonical, error = canonicalize_remote_grasp_target(raw)

    assert error is None
    assert canonical["x_cm"] == 24.41
    assert canonical["y_cm"] == 3.16
    assert canonical["z_cm"] == 9.35
    assert canonical["pitch_deg"] == -2.57
    assert canonical["roll_deg"] == 81.18
    assert canonical["gripper_width_cm"] == 8.68
    assert canonical["approach_depth_cm"] == 3.0
    assert canonical["score"] == 0.4692
    assert canonical["dist_cm"] == 1.56


def test_remote_grasp_target_missing_required_field_fails():
    canonical, error = canonicalize_remote_grasp_target({"y": 3.16, "z": 9.35, "pitch": -2.57, "roll": 81.18})

    assert canonical is None
    assert error["reason"] == "grasp_pose_schema_invalid"
    assert "x_cm" in error["missing_fields"]
    assert "gripper_width_cm" in error["missing_fields"]


def test_observation_router_keeps_grasp_result_for_control_obs():
    router = ObservationRouter(control_send_interval_s=0.0)
    obs = {
        "type": "vision_obs",
        "ts": 1.0,
        "stage": "GRASP",
        "mode": "GRASP_REMOTE",
        "status": "RESULT_READY",
        "obs_class": "control",
        "perception": {"target_obs": {"found": True}},
        "result": {"grasp": {"x_cm": 24.41}},
    }

    routed = router.route(vision_obs=obs, frame_meta={"frame_seq": 1}, now=2.0, force_send=True)

    assert routed.control_obs is not None
    assert routed.control_obs["result"]["grasp"]["x_cm"] == 24.41
