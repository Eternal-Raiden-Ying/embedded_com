#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ORCH_ROOT = ROOT / "orchestrator"
VISTA_ROOT = ROOT / "VISTA" / "vision_module"
for path in (ROOT, ORCH_ROOT, VISTA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.observation.router import ObservationRouter
from orchestrator_service.ipc.protocol import VisionObsEnvelope
from orchestrator_service.runtime.context import RuntimeContext
from orchestrator_service.runtime.service import OrchestratorService
from orchestrator_service.runtime.task_runtime import TaskRuntimeMixin


CANONICAL_GRASP = {
    "x_cm": 24.488,
    "y_cm": 4.927,
    "z_cm": 9.345,
    "pitch_deg": -2.562,
    "roll_deg": 96.347,
    "gripper_width_cm": 9.227,
}


def _grasp_result_ready_payload():
    return {
        "type": "vision_obs",
        "ts": time.time(),
        "stage": "GRASP",
        "mode": "GRASP_REMOTE",
        "status": "RESULT_READY",
        "obs_class": "control",
        "session_id": "sess_test",
        "req_id": "req_test",
        "epoch": 7,
        "perception": {},
        "result": {
            "grasp": dict(CANONICAL_GRASP),
            "source": "remote_grasp_client",
            "request_id": "req_test",
            "server_status": "success",
        },
    }


class _RuntimeHarness(TaskRuntimeMixin):
    def __init__(self):
        self.ctx = RuntimeContext()
        self.logs = []

    def _log(self, level, message):
        self.logs.append((level, message))

    def _enter_error_recovery(self, reason, *args, **kwargs):
        self.ctx.last_fail_reason = reason


def test_vista_router_preserves_grasp_result_in_control_obs():
    router = ObservationRouter(control_send_interval_s=0.0)
    payload = _grasp_result_ready_payload()

    routed = router.route(vision_obs=payload, frame_meta={}, now=time.time(), force_send=True)

    assert routed.control_obs is not None
    assert routed.control_obs["result"]["grasp"]["x_cm"] == 24.488


def test_orchestrator_grasp_result_payload_flattens_to_grasp_obs():
    payload = _grasp_result_ready_payload()
    env = VisionObsEnvelope.from_dict(payload)

    grasp_obs = OrchestratorService._grasp_obs_from_vision_payload(payload, env)

    assert grasp_obs is not None
    assert grasp_obs["type"] == "grasp_obs"
    assert grasp_obs["status"] == "RESULT_READY"
    assert grasp_obs["grasp"]["x_cm"] == 24.488
    assert grasp_obs["result"]["grasp"]["gripper_width_cm"] == 9.227


def test_handle_grasp_obs_writes_ctx_grasp_result():
    payload = _grasp_result_ready_payload()
    env = VisionObsEnvelope.from_dict(payload)
    grasp_obs = OrchestratorService._grasp_obs_from_vision_payload(payload, env)
    runtime = _RuntimeHarness()

    runtime.handle_grasp_obs(grasp_obs)

    assert runtime.ctx.grasp_status == "RESULT_READY"
    assert runtime.ctx.grasp_result["x_cm"] == 24.488


def test_result_ready_without_grasp_sets_missing_reason():
    runtime = _RuntimeHarness()

    runtime.handle_grasp_obs({"type": "grasp_obs", "status": "RESULT_READY", "result": {}})

    assert runtime.ctx.grasp_status == "FAILED"
    assert runtime.ctx.grasp_reason == "grasp_result_missing"
