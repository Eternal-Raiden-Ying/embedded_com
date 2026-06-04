#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dry-run test for GRASP state without mini-program / VISTA / camera / hardware.

The full search pipeline (IDLE → ... → FREEZE_BASE) is covered by
orchestrator_nohw_demo.py. This test focuses on FREEZE_BASE → GRASP → DONE.

Usage:
  cd orchestrator
  python -m orchestrator_service.examples.test_grasp_dryrun
"""

import json
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = str(SCRIPT_DIR.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


HOST = "127.0.0.1"
VISION_OBS_PORT = 19002
VISION_REQ_PORT = 19003


def send_jsonl(host: str, port: int, payload: Dict[str, Any]) -> None:
    data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        with socket.create_connection((host, port), timeout=2.0) as sock:
            sock.sendall(data)
    except Exception as e:
        print(f"  [send failed] {e}")


def make_grasp_obs(
    session_id: str,
    epoch: int,
    status: str,
    *,
    grasp: Optional[Dict] = None,
    reposition_proposal: Optional[Dict] = None,
    reason: str = "",
    detection: Optional[Dict] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "source": "remote_grasp_client",
        "request_id": f"rr_{int(time.time() * 1000)}",
    }
    if grasp is not None:
        result["grasp"] = grasp
    if detection is not None:
        result["detection"] = detection
    if reposition_proposal is not None:
        result["reposition_proposal"] = reposition_proposal
    if reason:
        result["reason"] = reason
    mode = "MICRO_ADJUST" if status == "WAITING_RESPONSE" else "GRASP_REMOTE"
    return {
        "ts": time.time(),
        "type": "vision_obs",
        "stage": "GRASP",
        "mode": mode,
        "status": status,
        "session_id": session_id,
        "epoch": int(epoch),
        "perception": {
            "target_obs": {
                "found": True,
                "target": "bottle",
                "confidence": 0.88,
            }
        },
        "result": result,
    }


MOCK_GRASP = {
    "x_cm": 15.0, "y_cm": 0.0, "z_cm": 12.0,
    "pitch_deg": 0.0, "roll_deg": 0.0,
    "gripper_width_cm": 8.5, "approach_depth_cm": 5.0,
    "confidence": 0.87, "feasible_distance_cm": 2.5,
    "position_frame": "robot", "angle_frame": "robot",
}

MOCK_REPOSITION_PROPOSAL = {
    "dx_cm": 5.0,
    "dy_cm": -10.0,
    "reference_line_new_xy_cm": [5.0, -10.0],
    "distance_lg_cm": 55.9,
    "capped": False,
    "reference_grasp": {
        "score": 0.58,
        "x_cm": 50.0, "y_cm": 5.0, "z_cm": 7.0,
        "feasible_distance_cm": 17.9,
    },
}


def main():
    print("=== GRASP Dry-Run Test ===\n")

    from orchestrator_service.config.schema import OrchestratorConfig
    from orchestrator_service.runtime.service import OrchestratorService
    from orchestrator_service.runtime.context import State
    from orchestrator_service.ipc.protocol import ArmResponse, now_ts

    cfg = OrchestratorConfig()
    tmp_root = tempfile.mkdtemp(prefix="orch_grasp_test_")
    cfg.runtime.project_root = PROJECT_ROOT
    cfg.runtime.runs_dir = f"{tmp_root}/runs"
    cfg.runtime.tick_hz = 10.0
    cfg.serial.dry_run = True
    cfg.serial.port = "/dev/null"
    cfg.task_cmd_in.transport = "tcp"
    cfg.task_cmd_in.host = HOST
    cfg.task_cmd_in.port = 19001
    cfg.vision_obs_in.host = HOST
    cfg.vision_obs_in.port = VISION_OBS_PORT
    cfg.vision_req_out.host = HOST
    cfg.vision_req_out.port = VISION_REQ_PORT
    cfg.vision_req_out.send_mode = "persistent"
    cfg.task_ack_out.host = HOST
    cfg.task_ack_out.port = 19012
    cfg.tts_event_out.transport = "disabled"

    from orchestrator_service.bridge.uart_bridge import UartBridge
    _orig_write = UartBridge._write_line

    def _echo_write(self, line: str, tx_meta=None):
        line_show = line.rstrip("\n").replace("\n", " | ")
        print(f"  [UART] {line_show}")
        return _orig_write(self, line, tx_meta=tx_meta)

    UartBridge._write_line = _echo_write

    service = OrchestratorService(cfg)
    thread = threading.Thread(target=service.run_forever, daemon=True, name="orchestrator")
    thread.start()
    time.sleep(0.5)

    try:
        core = service.core
        ctx = core.ctx

        # Jump to FREEZE_BASE (simulating completed search pipeline)
        print("--- Step 1: Setup context + jump to FREEZE_BASE ---")
        ctx.task_intent = "FIND"
        ctx.active_target = "bottle"
        ctx.active_session_id = "grasp_test"
        ctx.active_epoch = 1
        core._transition(State.FREEZE_BASE, "test: simulate search complete")
        time.sleep(0.3)
        print(f"  state={ctx.state.value}\n")

        # FREEZE_BASE should settle and transition to GRASP
        print("--- Step 2: Wait for FREEZE_BASE settle → GRASP ---")
        for _ in range(30):
            if ctx.state == State.GRASP:
                break
            time.sleep(0.1)
        print(f"  state={ctx.state.value}  substate={ctx.grasp_substate}")
        assert ctx.state == State.GRASP, f"Expected GRASP, got {ctx.state.value}"

        # AWAITING_RESPOND → send WAITING_RESPONSE
        print("\n--- Step 3: AWAITING_RESPOND → send WAITING_RESPONSE ---")
        time.sleep(0.5)
        assert ctx.grasp_substate == "AWAITING_RESPOND"
        send_jsonl(HOST, VISION_OBS_PORT, make_grasp_obs("grasp_test", 1, "WAITING_RESPONSE"))
        time.sleep(0.5)
        print(f"  substate={ctx.grasp_substate}")
        assert ctx.grasp_substate == "AWAITING_RESULT", f"Expected AWAITING_RESULT, got {ctx.grasp_substate}"

        # AWAITING_RESULT → send RESULT_READY with grasp
        print("\n--- Step 4: AWAITING_RESULT → send RESULT_READY ---")
        time.sleep(0.3)
        send_jsonl(HOST, VISION_OBS_PORT, make_grasp_obs(
            "grasp_test", 1, "RESULT_READY",
            grasp=MOCK_GRASP,
            detection={"found": True, "confidence": 0.88, "similar_detection_result": False},
        ))
        time.sleep(0.5)
        print(f"  substate={ctx.grasp_substate}")
        assert ctx.grasp_substate == "AWAITING_ARM", f"Expected AWAITING_ARM, got {ctx.grasp_substate}"

        # AWAITING_ARM → inject OK POSE
        print("\n--- Step 5: AWAITING_ARM → inject OK POSE ---")
        core.handle_arm_response(ArmResponse(
            ok=True, message="OK POSE 15 0 12 0 0 40 500",
            raw_line="OK POSE 15 0 12 0 0 40 500\n", ts=now_ts(),
        ))
        time.sleep(0.5)
        print(f"  state={ctx.state.value}")

        # Check DONE → IDLE
        for _ in range(20):
            if ctx.state == State.IDLE:
                break
            time.sleep(0.1)
        assert ctx.state == State.IDLE, f"Expected IDLE, got {ctx.state.value}"

        print("\n=== GRASP test PASSED ===")
        print(f"Logs: {service.run_logger.run_dir}")

        # --- Reposition test ---
        print("\n--- Step 6: Reposition test ---")
        ctx.task_intent = "FIND"
        ctx.active_target = "bottle"
        ctx.active_session_id = "grasp_repo_test"
        ctx.active_epoch = 2
        core._transition(State.FREEZE_BASE, "test: reposition flow")
        time.sleep(0.3)

        for _ in range(30):
            if ctx.state == State.GRASP:
                break
            time.sleep(0.1)
        assert ctx.state == State.GRASP, f"Expected GRASP, got {ctx.state.value}"

        # AWAITING_RESPOND → WAITING_RESPONSE
        time.sleep(0.5)
        send_jsonl(HOST, VISION_OBS_PORT, make_grasp_obs("grasp_repo_test", 2, "WAITING_RESPONSE"))
        time.sleep(0.5)
        assert ctx.grasp_substate == "AWAITING_RESULT"

        # AWAITING_RESULT → send RUNNING with reposition_proposal
        send_jsonl(HOST, VISION_OBS_PORT, make_grasp_obs(
            "grasp_repo_test", 2, "RUNNING",
            reposition_proposal=MOCK_REPOSITION_PROPOSAL,
            reason="no_feasible_grasp",
        ))
        time.sleep(0.5)
        assert ctx.grasp_substate == "REPOSITIONING", f"Expected REPOSITIONING, got {ctx.grasp_substate}"
        print(f"  substate={ctx.grasp_substate} proposal={ctx.grasp_reposition_proposal}")

        # Wait for reposition movement to complete (dy=-10cm → distance=10 → ~1s)
        for _ in range(25):
            if ctx.grasp_substate != "REPOSITIONING":
                break
            time.sleep(0.1)
        assert ctx.grasp_substate == "AWAITING_RESPOND", f"Expected AWAITING_RESPOND after reposition, got {ctx.grasp_substate}"
        print(f"  reposition complete, back to {ctx.grasp_substate}")

        print("\n=== All tests PASSED ===")

    finally:
        service._running = False
        thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
