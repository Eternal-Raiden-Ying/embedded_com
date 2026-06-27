#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORCH_ROOT = ROOT / "orchestrator"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from orchestrator_service.config.schema import OrchestratorConfig
from orchestrator_service.ipc.protocol import TableEdgeObs, TargetObs, now_ts
from orchestrator_service.runtime.context import State
from orchestrator_service.runtime.state_machine import OrchestratorCore


def _core() -> OrchestratorCore:
    cfg = OrchestratorConfig()
    return OrchestratorCore(cfg.control, cfg.car, cfg.docking)


def test_unknown_table_status_enters_error_recovery() -> None:
    core = _core()
    obs = TableEdgeObs(
        ts=now_ts(),
        table_found=True,
        edge_found=True,
        vision_status="BROKEN_STATUS",
        source_mode="FIND_EDGE",
    )

    core.handle_table_obs(obs)

    assert core.ctx.state == State.ERROR_RECOVERY
    assert "unknown vision status: BROKEN_STATUS" in core.ctx.last_enter_reason
    assert core.ctx.last_table_obs is None


def test_unknown_target_status_enters_error_recovery() -> None:
    core = _core()
    obs = TargetObs(
        ts=now_ts(),
        found=True,
        target="cup",
        vision_status="BROKEN_STATUS",
    )

    core.handle_target_obs(obs)

    assert core.ctx.state == State.ERROR_RECOVERY
    assert "unknown vision status: BROKEN_STATUS" in core.ctx.last_enter_reason
    assert core.ctx.last_target_obs is None
