#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Contract tests for RemoteGraspClient + RemoteManager with v1.2 schema validation.

Covers two layers:
- RemoteGraspClient  — real HTTP integration (requires a running grasp server)
- RemoteManager      — lifecycle + encoding contract (mocked transport, no server needed)

StagePlan-level behaviour belongs in architecture tests, not here.
"""

import os
import time
import unittest

import numpy as np

try:
    from .test_support import PrintLogger
except ImportError:
    from test_support import PrintLogger

from vision_module.backend.remote.client import RemoteGraspClient
from vision_module.backend.remote.manager import RemoteManager
from vision_module.backend.remote.protocol import (
    RemoteMetadata,
    RemotePredictRequest,
    RemotePredictResponse,
    build_predict_multipart,
)
from vision_module.backend.scheduler import Scheduler

# ---------------------------------------------------------------------------
# test constants
# ---------------------------------------------------------------------------

_DEFAULT_CLASS_ID = 1  # apple (finetune yolo26s bgr15)

_HERE = os.path.dirname(os.path.abspath(__file__))
_TEST_DATA_DIR = os.path.join(_HERE, "test_data")


def _load_test_image(filename):
    """Read a test image file as bytes."""
    path = os.path.join(_TEST_DATA_DIR, filename)
    with open(path, "rb") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# v1.2 schema helpers
# ---------------------------------------------------------------------------

_V1_2_TARGET_FIELDS = frozenset({
    "x_cm", "y_cm", "z_cm",
    "pitch_deg", "roll_deg",
    "gripper_width_cm", "approach_depth_cm",
    "confidence", "feasible_distance_cm",
    "position_frame", "angle_frame",
})

_V1_2_VALID_REASONS = frozenset({
    "no_grasp_detected", "no_feasible_grasp", "score_below_threshold",
})

_VALID_INIT_STATUSES = frozenset({"success", "already_loaded"})


def make_v12_target(**overrides):
    t = {
        "x_cm": 12.5, "y_cm": -3.2, "z_cm": 18.0,
        "pitch_deg": 15.3, "roll_deg": -2.1,
        "gripper_width_cm": 8.5, "approach_depth_cm": 5.0,
        "confidence": 0.87, "feasible_distance_cm": 4.2,
        "position_frame": "robot", "angle_frame": "robot",
    }
    t.update(overrides)
    return t


def make_v12_success_response(targets=None, **kw):
    if targets is None:
        targets = [make_v12_target()]
    return {
        "status": "success",
        "grasp_count": kw.get("grasp_count", len(targets) * 3),
        "feasible_count": kw.get("feasible_count", len(targets)),
        "output_count": kw.get("output_count", len(targets)),
        "targets": targets,
    }


def make_v12_reposition_response(reason="no_feasible_grasp", dx_cm=5.0, dy_cm=-3.0):
    return {
        "status": "reposition_required",
        "grasp_count": 0, "feasible_count": 0, "output_count": 0,
        "targets": [],
        "reason": reason,
        "reposition_proposal": {
            "dx_cm": dx_cm, "dy_cm": dy_cm,
            "reference_line_new_xy_cm": [10.0, -8.0],
        },
    }


def validate_target_schema(target):
    missing = _V1_2_TARGET_FIELDS - set(target.keys())
    assert not missing, f"Missing v1.2 fields: {missing}"
    assert isinstance(target["feasible_distance_cm"], (int, float)), \
        "feasible_distance_cm must be numeric (v1.2)"
    assert "feasible_angle_deg" not in target, \
        "feasible_angle_deg is DEPRECATED in v1.2"


def validate_grasp_response(payload):
    assert isinstance(payload, dict), "response payload must be dict"
    assert "status" in payload, "missing 'status'"
    status = str(payload["status"]).strip().lower()
    assert status in ("success", "reposition_required", "failure"), \
        f"unknown status: {status!r}"
    if status == "success":
        targets = payload.get("targets")
        assert isinstance(targets, list), "targets must be a list"
        for t in targets:
            validate_target_schema(t)
    elif status == "reposition_required":
        reason = str(payload.get("reason") or "")
        assert reason in _V1_2_VALID_REASONS, f"unknown reason: {reason!r}"
        if reason == "no_feasible_grasp":
            rp = payload.get("reposition_proposal")
            assert isinstance(rp, dict), "reposition_proposal required for no_feasible_grasp"
            for k in ("dx_cm", "dy_cm"):
                assert k in rp, f"reposition_proposal missing {k}"
    elif status == "failure":
        reason = str(payload.get("reason") or "")
        assert reason, "failure response must have a reason"


# ---------------------------------------------------------------------------
# mock transport for RemoteManager unit tests
# ---------------------------------------------------------------------------

def _install_mock_transport(client, responses=None):
    if responses is None:
        responses = {}

    def _mock_post_json(path, timeout_s, **kwargs):
        if path in responses:
            return responses[path]
        return RemotePredictResponse(ok=False, error=f"mock: unregistered path {path}")

    client._post_json = _mock_post_json
    client._session_open = True
    client._session = object()
    return responses


def _wait_until(predicate, timeout_s=1.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    return None


# ---------------------------------------------------------------------------
# RemoteGraspClient — real HTTP integration
# ---------------------------------------------------------------------------

_GRASP_SERVER_URL = os.environ.get("GRASP_SERVER_URL", "http://127.0.0.1:6006")


def _server_reachable(url, timeout_s=10.0):
    try:
        import requests as _r
        resp = _r.post(f"{url}/api/v1/init", timeout=timeout_s)
        return resp.status_code == 200
    except Exception:
        return False


class RemoteGraspClientIntegrationTest(unittest.TestCase):
    """Real HTTP tests — set GRASP_SERVER_URL env var to point at a running server."""

    @classmethod
    def setUpClass(cls):
        cls.server_available = _server_reachable(_GRASP_SERVER_URL)

    def setUp(self):
        if not self.server_available:
            self.skipTest(f"grasp server unreachable at {_GRASP_SERVER_URL}")
        self.client = RemoteGraspClient(base_url=_GRASP_SERVER_URL)
        self.client.open()

    def tearDown(self):
        try:
            self.client.close()
        except Exception:
            pass

    # --- basic HTTP ---

    def test_init(self):
        resp = self.client.init_server(timeout_s=10)
        self.assertTrue(resp.ok, f"init failed: {resp.error}")
        status = (resp.payload or {}).get("status", "")
        self.assertIn(status, _VALID_INIT_STATUSES,
                      f"init status {status!r} not in {_VALID_INIT_STATUSES}")

    def test_release(self):
        self.client.init_server(timeout_s=10)
        resp = self.client.release_server(timeout_s=5)
        self.assertTrue(resp.ok, f"release failed: {resp.error}")

    # --- v1.2 predict with real test images ---

    def test_predict_with_test_images(self):
        self.client.init_server(timeout_s=10)
        rgb_bytes = _load_test_image("color.png")
        depth_bytes = _load_test_image("depth.png")
        resp = self.client.predict(
            RemotePredictRequest(
                rgb_bytes=rgb_bytes, depth_bytes=depth_bytes,
                class_id=_DEFAULT_CLASS_ID, timeout_s=15.0,
            )
        )
        self.assertIsNotNone(resp)
        self.assertTrue(resp.ok, f"predict HTTP error: {resp.error}")
        payload = resp.payload or {}
        validate_grasp_response(payload)
        # NOTE: status depends on model + image content.
        # success / failure / reposition_required are all valid v1.2 shapes.
        self.assertIn(payload["status"], ("success", "failure", "reposition_required"))

    def test_predict_metadata_round_trip(self):
        self.client.init_server(timeout_s=10)
        rgb_bytes = _load_test_image("color.png")
        depth_bytes = _load_test_image("depth.png")
        resp = self.client.predict(
            RemotePredictRequest(
                rgb_bytes=rgb_bytes, depth_bytes=depth_bytes,
                class_id=_DEFAULT_CLASS_ID,
                metadata=RemoteMetadata(robot_id="test_contract", command="predict",
                                        class_id=_DEFAULT_CLASS_ID),
                timeout_s=15.0,
            )
        )
        self.assertIsNotNone(resp)
        validate_grasp_response(resp.payload or {})

    # --- error handling ---

    def test_bad_url_returns_error(self):
        bad = RemoteGraspClient(base_url="http://127.0.0.1:19999")
        bad.open()
        resp = bad.init_server(timeout_s=2)
        self.assertFalse(resp.ok)
        self.assertIn("onnection", str(resp.error).lower())


# ---------------------------------------------------------------------------
# RemoteManager — lifecycle + encoding (mocked transport)
# ---------------------------------------------------------------------------

class RemoteManagerContractTest(unittest.TestCase):
    def setUp(self):
        self.scheduler = Scheduler()
        self.scheduler.start_runtime()
        self.scheduler.configure(
            {
                "mode": "GRASP_REMOTE",
                "routes": {
                    "camera_frames":  {"policy": "slot",  "scope": "backend"},
                    "remote_result":  {"policy": "slot",  "scope": "stage"},
                },
            },
            generation=1,
        )
        self.client = RemoteGraspClient(base_url="http://127.0.0.1:6006")
        self.mock_rsp = _install_mock_transport(self.client)

        self.manager = RemoteManager(client=self.client, logger=PrintLogger("remote_contract"))
        self.manager.bind_runtime(self.scheduler, lambda: 1)
        self.manager.configure_runtime(
            {
                "enabled": True,
                "base_url": "http://127.0.0.1:6006",
                "command": "predict",
                "require_depth": True,
                "timeout_s": 12.0,
                "rgb_encoding": "png",
                "depth_encoding": "jpeg",
                "rgb_quality": 87,
                "depth_compression": 2,
                "metadata": {"profile": "grasp_remote"},
            }
        )
        self.encode_calls = []

        def _fake_encode(encoding, frame, *, quality=90, compression=3):
            self.encode_calls.append((encoding, int(quality), int(compression),
                                      tuple(int(v) for v in frame.shape)))
            return f"{encoding}:{quality}:{compression}".encode()

        self.manager._encode_frame = _fake_encode

        self.mock_rsp["/api/v1/init"] = RemotePredictResponse(
            ok=True, payload={"status": "success", "message": "ready"}, status_code=200,
        )
        self.manager.enable()
        self.manager.start_runtime()
        _wait_until(
            lambda: (
                self.scheduler.read_result("remote_result", default={})
                if int((self.scheduler.read_result("remote_result", default={}) or {}).get(
                    "service_init_attempts", 0) or 0) >= 1
                else None
            )
        )

    def tearDown(self):
        self.mock_rsp["/api/v1/release"] = RemotePredictResponse(
            ok=True, payload={"status": "success", "message": "freed"}, status_code=200,
        )
        self.manager.stop_runtime()
        self.manager.disable()
        self.scheduler.stop_runtime()

    # --- lifecycle ---

    def test_service_init_on_enable(self):
        payload = _wait_until(
            lambda: (
                self.scheduler.read_result("remote_result", default={})
                if bool((self.scheduler.read_result("remote_result", default={}) or {}).get(
                    "service_init_confirmed", False))
                else None
            )
        )
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["last_action"], "init")
        self.assertTrue(payload["last_ok"])
        self.assertEqual(payload["service_init_state"], "ready")

    @unittest.skip("effects channel removed — test needs rewrite for task-worker path")
    def test_predict_after_init(self):
        self.mock_rsp["/api/v1/predict"] = RemotePredictResponse(
            ok=True, payload=make_v12_success_response(), status_code=200,
        )
        self.scheduler.publish_result(
            "camera_frames",
            {"rgb": np.zeros((16, 16, 3), dtype=np.uint8),
             "depth": np.zeros((8, 8), dtype=np.uint8)},
            generation=1,
        )
        self.scheduler.publish_event(
            "remote_cmd",
            {"op": "PREDICT", "request_id": "rr_1", "target": "bottle", "class_id": 4, "need_depth": True},
            generation=1,
        )
        payload = _wait_until(
            lambda: (
                self.scheduler.read_result("remote_result", default={})
                if (self.scheduler.read_result("remote_result", default={}) or {}).get("last_action") == "predict"
                else None
            )
        )
        self.assertTrue(payload["last_ok"], f"predict failed: {payload.get('last_error')}")
        ack = self.scheduler.consume_event("remote_ack")
        self.assertEqual(ack["op"], "PREDICT")
        self.assertTrue(ack["ok"])

    def test_disable_releases(self):
        self.mock_rsp["/api/v1/release"] = RemotePredictResponse(
            ok=True, payload={"status": "success", "message": "freed"}, status_code=200,
        )
        self.manager.stop_runtime()
        self.manager.disable()
        self.assertFalse(self.manager.enabled)

    # --- encoding contract ---

    @unittest.skip("effects channel removed — test needs rewrite for task-worker path")
    def test_profile_drives_encoding(self):
        self.mock_rsp["/api/v1/predict"] = RemotePredictResponse(
            ok=True, payload=make_v12_success_response(), status_code=200,
        )
        self.scheduler.publish_result(
            "camera_frames",
            {"rgb": np.zeros((16, 16, 3), dtype=np.uint8),
             "depth": np.zeros((8, 8), dtype=np.uint8)},
            generation=1,
        )
        self.scheduler.publish_event(
            "remote_cmd",
            {"op": "PREDICT", "request_id": "rr_enc", "target": "banana", "class_id": 2},
            generation=1,
        )
        _wait_until(
            lambda: (
                self.scheduler.read_result("remote_result", default={})
                if (self.scheduler.read_result("remote_result", default={}) or {}).get("last_action") == "predict"
                else None
            )
        )
        self.assertGreaterEqual(len(self.encode_calls), 2)
        self.assertEqual(self.encode_calls[0][0], "png")
        self.assertEqual(self.encode_calls[1][0], "jpeg")

    # --- multipart ---

    def test_multipart_no_segmentation_surface(self):
        data, files = build_predict_multipart(
            RemotePredictRequest(rgb_bytes=b"rgb", depth_bytes=b"depth", class_id=2)
        )
        self.assertEqual(data["class_id"], "2")
        self.assertEqual(sorted(files.keys()), ["depth_file", "rgb_file"])

    # --- v1.2 schema validation ---

    @unittest.skip("effects channel removed — test needs rewrite for task-worker path")
    def test_v12_success_response_passthrough(self):
        targets = [make_v12_target(confidence=0.92, feasible_distance_cm=3.1)]
        self.mock_rsp["/api/v1/predict"] = RemotePredictResponse(
            ok=True, payload=make_v12_success_response(targets=targets), status_code=200,
        )
        self.scheduler.publish_result(
            "camera_frames",
            {"rgb": np.zeros((16, 16, 3), dtype=np.uint8),
             "depth": np.zeros((8, 8), dtype=np.uint8)},
            generation=1,
        )
        self.scheduler.publish_event(
            "remote_cmd",
            {"op": "PREDICT", "request_id": "rr_v12", "target": "apple", "class_id": _DEFAULT_CLASS_ID},
            generation=1,
        )
        payload = _wait_until(
            lambda: (
                self.scheduler.read_result("remote_result", default={})
                if (self.scheduler.read_result("remote_result", default={}) or {}).get("last_action") == "predict"
                else None
            )
        )
        self.assertTrue(payload["has_result"], f"no result: {payload.get('last_error')}")
        result = payload["result"]
        validate_grasp_response(result)
        self.assertAlmostEqual(result["targets"][0]["confidence"], 0.92)
        self.assertAlmostEqual(result["targets"][0]["feasible_distance_cm"], 3.1)

    @unittest.skip("effects channel removed — test needs rewrite for task-worker path")
    def test_v12_reposition_required_passthrough(self):
        self.mock_rsp["/api/v1/predict"] = RemotePredictResponse(
            ok=True,
            payload=make_v12_reposition_response("no_feasible_grasp", dx_cm=8.0, dy_cm=-4.0),
            status_code=200,
        )
        self.scheduler.publish_result(
            "camera_frames",
            {"rgb": np.zeros((16, 16, 3), dtype=np.uint8),
             "depth": np.zeros((8, 8), dtype=np.uint8)},
            generation=1,
        )
        self.scheduler.publish_event(
            "remote_cmd",
            {"op": "PREDICT", "request_id": "rr_repo", "target": "bottle", "class_id": 4},
            generation=1,
        )
        payload = _wait_until(
            lambda: (
                self.scheduler.read_result("remote_result", default={})
                if (self.scheduler.read_result("remote_result", default={}) or {}).get("last_action") == "predict"
                else None
            )
        )
        result = payload["result"]
        validate_grasp_response(result)
        self.assertEqual(result["reason"], "no_feasible_grasp")
        self.assertAlmostEqual(result["reposition_proposal"]["dx_cm"], 8.0)

    # --- schema validator unit tests ---

    def test_schema_rejects_missing_feasible_distance_cm(self):
        bad = make_v12_target()
        del bad["feasible_distance_cm"]
        with self.assertRaises(AssertionError):
            validate_target_schema(bad)

    def test_schema_rejects_deprecated_feasible_angle_deg(self):
        bad = make_v12_target()
        bad["feasible_angle_deg"] = 30.0
        with self.assertRaises(AssertionError):
            validate_target_schema(bad)

    def test_schema_rejects_unknown_reason(self):
        bad = make_v12_reposition_response("some_future_reason")
        with self.assertRaises(AssertionError):
            validate_grasp_response(bad)


if __name__ == "__main__":
    unittest.main()
