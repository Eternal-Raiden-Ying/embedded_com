#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib.util
import sys
import time
from pathlib import Path

PKG_NAME = "orchestrator_service"
ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
WORKSPACE = ROOT.parent.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

if PKG_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(PKG_NAME, ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(ROOT)]
    sys.modules[PKG_NAME] = module
    if spec and spec.loader:
        spec.loader.exec_module(module)

from orchestrator_service.config.schema import CarMotionConfig, ControlThresholds
from orchestrator_service.ipc.protocol import (
    TableEdgeObs,
    TargetObs,
    TaskCmd,
    iter_vision_perception_payloads,
)
from orchestrator_service.runtime.common import monotonic_ts
from orchestrator_service.runtime.context import State
from orchestrator_service.runtime.state_machine import OrchestratorCore


def make_find_cmd(target: str = "apple") -> TaskCmd:
    return TaskCmd(
        ts=time.time(),
        intent="FIND",
        confidence=0.99,
        target=target,
        cmd_id="cmd_test_find",
        session_id="sess_test_find",
        epoch=1,
        source="test",
    )


def make_table_obs(*, yaw=0.0, dist=0.20, edge_ready=False, table_cx=0.0, table_size=0.30, source_mode=None, session_id="sess_test_find") -> TableEdgeObs:
    return TableEdgeObs(
        ts=time.time(),
        table_found=True,
        edge_found=True,
        confidence=0.95,
        yaw_err_rad=yaw,
        dist_err_m=dist,
        lateral_err_m=0.0,
        table_cx_norm=table_cx,
        table_size_norm=table_size,
        edge_ready=edge_ready,
        source_mode=source_mode,
        session_id=session_id,
        epoch=1,
    )


def make_target_obs(*, found=False, target="apple", session_id="sess_test_find") -> TargetObs:
    return TargetObs(
        ts=time.time(),
        found=found,
        target=target if found else None,
        confidence=0.9 if found else None,
        cx_norm=0.02,
        size_norm=0.14,
        session_id=session_id,
        epoch=1,
    )


def assert_state(core: OrchestratorCore, expected: State, msg: str):
    actual = core.ctx.state
    if actual != expected:
        raise AssertionError(f"{msg}: expected={expected.value}, actual={actual.value}")


def drain_last_vision_req(core: OrchestratorCore) -> dict:
    msgs = core.drain_vision_msgs()
    if not msgs:
        raise AssertionError("expected pending vision_req, got none")
    return msgs[-1]


def assert_vision_req(req: dict, *, op: str, stage: str, mode_hint: str, search_kind: str, target: str = ""):
    if req.get("type") != "vision_req":
        raise AssertionError(f"unexpected req type: {req}")
    if req.get("op") != op:
        raise AssertionError(f"unexpected req op: {req}")
    if req.get("stage") != stage:
        raise AssertionError(f"unexpected req stage: {req}")
    if req.get("mode_hint") != mode_hint:
        raise AssertionError(f"unexpected req mode_hint: {req}")
    payload = req.get("payload") or {}
    if payload.get("search_kind") != search_kind:
        raise AssertionError(f"unexpected req payload.search_kind: {req}")
    actual_target = str(req.get("target") or "")
    if actual_target != target:
        raise AssertionError(f"unexpected req target: {req}")


def main():
    cfg = ControlThresholds()
    car_cfg = CarMotionConfig()
    cfg.search_table_timeout_s = 20.0
    cfg.approach_timeout_s = 8.0
    cfg.target_search_timeout_s = 4.0
    cfg.table_found_frames_to_approach = 2
    cfg.coarse_align_frames_to_advance = 2
    cfg.final_lock_frames_to_arrive = 2
    cfg.table_settle_s = 0.0
    cfg.table_stable_frames = 2
    cfg.edge_settle_s = 0.1
    cfg.search_target_init_hold_s = 0.05
    cfg.edge_handoff_min_s = 0.01
    cfg.edge_handoff_max_s = 0.01
    cfg.edge_handoff_samples = 1
    cfg.target_found_frames_to_confirm = 2
    cfg.target_confirm_min_s = 0.0
    cfg.target_confirm_timeout_s = 1.0
    cfg.target_lock_stable_s = 0.0
    cfg.target_lock_settle_s = 0.05
    cfg.target_locked_freeze_after_s = 0.05
    cfg.freeze_settle_s = 0.05
    cfg.done_hold_s = 0.05
    cfg.req_resend_period_s = 0.0

    core = OrchestratorCore(cfg, car_cfg)
    accepted, reason = core.handle_task_cmd(make_find_cmd())
    if not accepted:
        raise AssertionError(f"FIND should be accepted, got reason={reason}")
    assert_state(core, State.SEARCH_TABLE, "after FIND")
    assert_vision_req(
        drain_last_vision_req(core),
        op="START",
        stage="SEARCH",
        mode_hint="FIND_EDGE",
        search_kind="TABLE_EDGE",
        target="",
    )

    for _ in range(2):
        core.handle_table_obs(make_table_obs(yaw=0.20, dist=0.30, edge_ready=False, table_cx=0.16))
        core.tick()
    assert_state(core, State.COARSE_ALIGN, "stable table should enter COARSE_ALIGN")
    assert_vision_req(
        drain_last_vision_req(core),
        op="UPDATE",
        stage="SEARCH",
        mode_hint="FIND_EDGE",
        search_kind="TABLE_EDGE",
        target="",
    )

    for _ in range(2):
        core.handle_table_obs(make_table_obs(yaw=0.03, dist=0.22, edge_ready=False, table_cx=0.03))
        core.tick()
    assert_state(core, State.CONTROLLED_APPROACH, "coarse align should advance")

    core.handle_table_obs(make_table_obs(yaw=0.02, dist=0.06, edge_ready=True, table_cx=0.01, table_size=0.55))
    core.tick()
    assert_state(core, State.FINAL_LOCK, "edge ready should enter FINAL_LOCK")

    for _ in range(3):
        core.handle_table_obs(make_table_obs(yaw=0.01, dist=0.01, edge_ready=True, table_cx=0.0, table_size=0.60))
        core.tick()
    assert_state(core, State.AT_TABLE_EDGE, "final lock should reach AT_TABLE_EDGE")

    core.ctx.state_enter_mono = monotonic_ts() - 0.2
    core.handle_table_obs(make_table_obs(yaw=0.01, dist=0.01, edge_ready=True, table_cx=0.0, table_size=0.60, source_mode="FIND_OBJECT"))
    core.tick()
    assert_state(core, State.SEARCH_TARGET_INIT, "after settle should enter SEARCH_TARGET_INIT")
    assert_vision_req(
        drain_last_vision_req(core),
        op="UPDATE",
        stage="SEARCH",
        mode_hint="FIND_OBJECT",
        search_kind="TARGET",
        target="apple",
    )

    core.ctx.state_enter_mono = monotonic_ts() - 0.2
    core.tick()
    assert_state(core, State.EDGE_SLIDE_SEARCH, "search target init should enter EDGE_SLIDE_SEARCH")

    for _ in range(3):
        core.handle_target_obs(make_target_obs(found=True))
        core.tick()
    assert_state(core, State.TARGET_LOCKED, "stable target should enter TARGET_LOCKED")

    core.ctx.state_enter_mono = monotonic_ts() - 0.2
    core.tick()
    assert_state(core, State.FREEZE_BASE, "target lock should freeze base")

    core.ctx.state_enter_mono = monotonic_ts() - 0.2
    core.tick()
    assert_state(core, State.DONE, "freeze base should complete task")

    core.ctx.state_enter_mono = monotonic_ts() - 0.2
    core.tick()
    assert_state(core, State.IDLE, "done should return to idle")

    core.handle_task_cmd(make_find_cmd())
    drain_last_vision_req(core)
    core.ctx.state = State.EDGE_SLIDE_SEARCH
    core.ctx.state_enter_mono = monotonic_ts() - (cfg.target_search_timeout_s + 0.2)
    core.tick()
    assert_state(core, State.LEAVE_EDGE, "search timeout should trigger edge relocation")

    flattened = iter_vision_perception_payloads(
        {
            "type": "vision_obs",
            "ts": time.time(),
            "stage": "SEARCH",
            "mode": "FIND_OBJECT",
            "status": "RUNNING",
            "session_id": "sess_test_find",
            "req_id": "req_test",
            "epoch": 1,
            "perception": {
                "target_obs": {
                    "found": True,
                    "target": "apple",
                    "confidence": 0.95,
                    "cx_norm": 0.02,
                    "size_norm": 0.14,
                }
            },
        }
    )
    if len(flattened) != 1 or flattened[0].get("type") != "target_obs":
        raise AssertionError(f"vision_obs flatten failed: {flattened}")

    print("PASS: state_machine_regression_test")
    print("  - search/dock/target happy path reaches DONE")
    print("  - outbound vision_req uses START/UPDATE + SEARCH stage mapping")
    print("  - vision_obs envelope can be flattened into target_obs payloads")
    print("  - edge search timeout triggers LEAVE_EDGE")
    print("  - state machine returns to IDLE after completion")


if __name__ == "__main__":
    main()
