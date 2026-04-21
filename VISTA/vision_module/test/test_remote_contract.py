#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import unittest

import numpy as np

try:
    from .test_support import PrintLogger
except ImportError:
    from test_support import PrintLogger

from vision_module.app.stages.base import StageContext, StageTickInput
from vision_module.app.stages.grasp import GraspStagePlan
from vision_module.backend.remote.manager import RemoteManager
from vision_module.backend.remote.protocol import RemotePredictRequest, RemotePredictResponse, build_predict_multipart
from vision_module.backend.scheduler import Scheduler
from vision_module.ipc.protocol import VisionReq


class _FakeRemoteClient:
    def __init__(self):
        self.base_url = ""
        self.session_open = False
        self.calls = []
        self.last_predict_request = None

    def configure(self, base_url: str) -> None:
        self.base_url = str(base_url or "")
        self.calls.append(("configure", self.base_url))

    def open(self) -> None:
        self.session_open = True
        self.calls.append(("open", None))

    def close(self) -> None:
        self.session_open = False
        self.calls.append(("close", None))

    def init_server(self, timeout_s: float = 15.0):
        self.calls.append(("init", float(timeout_s)))
        return RemotePredictResponse(ok=True, payload={"ready": True}, status_code=200)

    def predict(self, request):
        self.last_predict_request = request
        self.calls.append(("predict", request))
        return RemotePredictResponse(ok=True, payload={"grasps": [{"x": 1.0}]}, status_code=200)

    def release_server(self, timeout_s: float = 5.0):
        self.calls.append(("release", float(timeout_s)))
        return RemotePredictResponse(ok=True, payload={"released": True}, status_code=200)

    def snapshot(self):
        return {
            "base_url": self.base_url,
            "session_open": self.session_open,
        }


def _wait_until(predicate, timeout_s: float = 1.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    return None


class RemoteManagerContractTest(unittest.TestCase):
    def setUp(self):
        self.scheduler = Scheduler()
        self.scheduler.start_runtime()
        self.scheduler.configure(
            {
                "mode": "GRASP_REMOTE",
                "routes": {
                    "camera_frames": {"policy": "slot", "scope": "backend"},
                    "remote_cmd": {"policy": "event", "scope": "backend"},
                    "remote_ack": {"policy": "event", "scope": "backend"},
                    "remote_result": {"policy": "slot", "scope": "stage"},
                },
            },
            generation=1,
        )
        self.client = _FakeRemoteClient()
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
            self.encode_calls.append((encoding, int(quality), int(compression), tuple(int(v) for v in frame.shape)))
            return f"{encoding}:{quality}:{compression}".encode()

        self.manager._encode_frame = _fake_encode
        self.manager.enable()
        self.manager.start_runtime()
        _wait_until(
            lambda: (
                self.scheduler.read_result("remote_result", default={})
                if int((self.scheduler.read_result("remote_result", default={}) or {}).get("service_init_attempts", 0) or 0) >= 1
                else None
            )
        )

    def tearDown(self):
        self.manager.stop_runtime()
        self.manager.disable()
        self.scheduler.stop_runtime()

    def test_service_start_attempts_best_effort_init(self):
        payload = _wait_until(
            lambda: (
                self.scheduler.read_result("remote_result", default={})
                if bool((self.scheduler.read_result("remote_result", default={}) or {}).get("service_init_confirmed", False))
                else None
            )
        )
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["last_action"], "init")
        self.assertTrue(payload["last_ok"])
        self.assertEqual(payload["service_init_state"], "ready")
        self.assertEqual(payload["service_init_attempts"], 1)
        self.assertIn(("init", 12.0), self.client.calls)

    def test_predict_uses_service_level_init_without_per_request_init(self):
        self.scheduler.publish_result(
            "camera_frames",
            {
                "rgb": np.zeros((16, 16, 3), dtype=np.uint8),
                "depth": np.zeros((8, 8), dtype=np.uint8),
            },
            generation=1,
        )
        self.scheduler.publish_event(
            "remote_cmd",
            {
                "op": "PREDICT",
                "request_id": "rr_pre_init",
                "target": "cup",
                "class_id": 41,
                "need_depth": True,
            },
            generation=1,
        )
        payload = _wait_until(
            lambda: (
                self.scheduler.read_result("remote_result", default={})
                if (self.scheduler.read_result("remote_result", default={}) or {}).get("last_action") == "predict"
                else None
            )
        )
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["last_action"], "predict")
        self.assertTrue(payload["last_ok"])
        ack = self.scheduler.consume_event("remote_ack")
        self.assertEqual(ack["op"], "PREDICT")
        self.assertTrue(ack["ok"])

    def test_profile_drives_encoding_and_explicit_class_id_contract(self):
        self.scheduler.publish_result(
            "camera_frames",
            {
                "rgb": np.zeros((16, 16, 3), dtype=np.uint8),
                "depth": np.zeros((8, 8), dtype=np.uint8),
            },
            generation=1,
        )
        self.scheduler.publish_event(
            "remote_cmd",
            {
                "op": "PREDICT",
                "request_id": "rr_full_flow",
                "target": "banana",
                "class_id": 46,
                "need_depth": True,
                "robot_id": "arm_ut",
                "base_url": "http://override.invalid",
                "metadata": {"request_scope": "test"},
            },
            generation=1,
        )
        predict_payload = _wait_until(
            lambda: (
                self.scheduler.read_result("remote_result", default={})
                if (self.scheduler.read_result("remote_result", default={}) or {}).get("last_action") == "predict"
                else None
            )
        )
        self.assertTrue(predict_payload["last_ok"])
        self.assertTrue(predict_payload["has_result"])
        request = self.client.last_predict_request
        self.assertIsNotNone(request)
        self.assertEqual(request.class_id, 46)
        self.assertEqual(request.timeout_s, 12.0)
        self.assertEqual(request.rgb_encoding, "png")
        self.assertEqual(request.depth_encoding, "jpeg")
        self.assertEqual(request.metadata.robot_id, "arm_ut")
        self.assertEqual(request.metadata.command, "predict")
        self.assertEqual(request.metadata.extras["profile"], "grasp_remote")
        self.assertEqual(request.metadata.extras["request_scope"], "test")
        self.assertEqual(request.metadata.extras["target"], "banana")
        self.assertEqual(request.metadata.extras["request_id"], "rr_full_flow")
        self.assertGreaterEqual(len(self.encode_calls), 2)
        self.assertEqual(self.encode_calls[0][0], "png")
        self.assertEqual(self.encode_calls[1][0], "jpeg")
        self.assertNotIn(("configure", "http://override.invalid"), self.client.calls)

    def test_disable_releases_ready_service_session(self):
        self.manager.stop_runtime()
        self.manager.disable()
        self.assertIn(("release", 12.0), self.client.calls)
        self.assertIn(("close", None), self.client.calls)

    def test_predict_multipart_no_longer_advertises_segmentation_surface(self):
        data, files = build_predict_multipart(
            RemotePredictRequest(
                rgb_bytes=b"rgb",
                depth_bytes=b"depth",
                class_id=46,
            )
        )
        self.assertEqual(data["class_id"], "46")
        self.assertEqual(sorted(files.keys()), ["depth_file", "rgb_file"])


class RemoteStageContractTest(unittest.TestCase):
    def test_accept_remote_grasp_requires_explicit_class_id(self):
        plan = GraspStagePlan()
        ctx = StageContext()
        start_req = VisionReq(
            ts=time.time(),
            op="START",
            stage="GRASP",
            target="cup",
            payload={"remote_grasp": True, "need_depth": True},
        )
        plan.on_enter(start_req, ctx)

        tick = plan.tick(
            StageTickInput(
                ts=time.time(),
                results={"local_perception": {"target_obs": {"found": True, "target": "cup"}}},
            ),
            ctx,
        )
        self.assertEqual(tick.vision_obs["status"], "WAITING_RESPONSE")

        respond = VisionReq(
            ts=time.time(),
            op="RESPOND",
            stage="GRASP",
            target="cup",
            interaction_id=ctx.interaction_id,
            response={"decision": "ACCEPT"},
        )
        output = plan.on_respond(respond, ctx)
        self.assertIsNotNone(output)
        self.assertEqual(output.vision_obs["status"], "FAILED")
        self.assertEqual(output.vision_obs["result"]["reason"], "missing_class_id")
        self.assertEqual(output.effects, [])


if __name__ == "__main__":
    unittest.main()
