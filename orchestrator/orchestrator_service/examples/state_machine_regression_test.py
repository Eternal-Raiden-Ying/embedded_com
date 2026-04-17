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
from orchestrator_service.ipc.protocol import TableEdgeObs, TargetObs, TaskCmd
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


def make_table_obs(*, yaw=0.0, dist=0.20, edge_ready=False, table_cx=0.0, table_size=0.30, session_id="sess_test_find") -> TableEdgeObs:
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


def main():
    cfg = ControlThresholds()
    car_cfg = CarMotionConfig()
    cfg.search_table_timeout_s = 20.0
    cfg.approach_timeout_s = 8.0
    cfg.target_search_timeout_s = 4.0
    cfg.table_found_frames_to_approach = 2
    cfg.coarse_align_frames_to_advance = 2
    cfg.final_lock_frames_to_arrive = 2
    cfg.edge_settle_s = 0.1
    cfg.search_target_init_hold_s = 0.05
    cfg.target_found_frames_to_confirm = 2
    cfg.target_lock_settle_s = 0.05
    cfg.freeze_settle_s = 0.05
    cfg.done_hold_s = 0.05

    core = OrchestratorCore(cfg, car_cfg)
    accepted, reason = core.handle_task_cmd(make_find_cmd())
    if not accepted:
        raise AssertionError(f"FIND should be accepted, got reason={reason}")
    assert_state(core, State.SEARCH_TABLE, "after FIND")

    for _ in range(2):
        core.handle_table_obs(make_table_obs(yaw=0.20, dist=0.30, edge_ready=False, table_cx=0.16))
        core.tick()
    assert_state(core, State.COARSE_ALIGN, "stable table should enter COARSE_ALIGN")

    for _ in range(2):
        core.handle_table_obs(make_table_obs(yaw=0.03, dist=0.22, edge_ready=False, table_cx=0.03))
        core.tick()
    assert_state(core, State.CONTROLLED_APPROACH, "coarse align should advance")

    core.handle_table_obs(make_table_obs(yaw=0.02, dist=0.06, edge_ready=True, table_cx=0.01, table_size=0.55))
    core.tick()
    assert_state(core, State.FINAL_LOCK, "edge ready should enter FINAL_LOCK")

    for _ in range(2):
        core.handle_table_obs(make_table_obs(yaw=0.01, dist=0.01, edge_ready=True, table_cx=0.0, table_size=0.60))
        core.tick()
    assert_state(core, State.AT_TABLE_EDGE, "final lock should reach AT_TABLE_EDGE")

    core.ctx.state_enter_mono = monotonic_ts() - 0.2
    core.tick()
    assert_state(core, State.SEARCH_TARGET_INIT, "after settle should enter SEARCH_TARGET_INIT")

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
    core.ctx.state = State.EDGE_SLIDE_SEARCH
    core.ctx.state_enter_mono = monotonic_ts() - (cfg.target_search_timeout_s + 0.2)
    core.tick()
    assert_state(core, State.LEAVE_EDGE, "search timeout should trigger edge relocation")

    print("PASS: state_machine_regression_test")
    print("  - search/dock/target happy path reaches DONE")
    print("  - edge search timeout triggers LEAVE_EDGE")
    print("  - state machine returns to IDLE after completion")


if __name__ == "__main__":
    main()
