from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "orchestrator"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from orchestrator_service.ipc.protocol import TaskCmd, make_task_ack
from orchestrator_service.runtime.context import State
from orchestrator_service.runtime.service import OrchestratorService
from orchestrator_service.config.schema import OrchestratorConfig
from orchestrator_service.runtime.state_machine import OrchestratorCore


def _cmd(source="voice"):
    return TaskCmd.from_dict({
        "type": "task_cmd", "intent": "FIND", "target": "apple", "confidence": 1.0,
        "cmd_id": "ack-contract", "session_id": "session-1", "source": source,
    }, set())


class _RunLogger:
    def write_jsonl(self, *_args, **_kwargs):
        pass

    def write_ipc(self, *_args, **_kwargs):
        pass


class _Sender:
    def __init__(self, result=True, error=None):
        self.result = result
        self.error = error
        self.messages = []

    def send(self, message):
        if self.error:
            raise self.error
        self.messages.append(message)
        return self.result


def _service(extra, sender):
    service = object.__new__(OrchestratorService)
    service._module_name = "orch"
    service.core = SimpleNamespace(ctx=SimpleNamespace(
        state=State.SEARCH_TABLE, last_task_ack_extra=extra,
        canonical_target="apple", active_target="apple", class_name="apple", class_id=2,
        active_task_id="task-1",
    ))
    service.task_ack_sender = sender
    service.run_logger = _RunLogger()
    service._task_ack_skip_until_ts = 0.0
    service._task_ack_last_fail_log_ts = 0.0
    service._last_tx_summary = {"task_ack_out": 0.0}
    service._task_ack_out_unavailable_reason = lambda: ""
    service._operator_ipc_event = lambda *_args, **_kwargs: None
    service.log_warn = lambda *_args, **_kwargs: None
    service.log_ipc = lambda *_args, **_kwargs: None
    return service


def test_legacy_make_task_ack_call_is_compatible():
    ack = make_task_ack(_cmd(), accepted=True, state="SEARCH_TABLE", execution_status="ready")
    assert ack["accepted"] is True
    assert ack["execution_status"] == "ready"


@pytest.mark.parametrize("source", ["voice", "mobile_gateway"])
def test_find_action_policy_is_internal_and_ack_builds_for_all_sources(source):
    sender = _Sender()
    service = _service({
        "raw_target": "apple", "canonical_target": "apple", "class_name": "apple",
        "class_id": 2, "task_id": "task-1", "execution_status": "ready",
        "action_policy": "fixed_grasp", "unexpected": "must_not_escape",
    }, sender)

    assert service._send_task_ack(_cmd(source), accepted=True, reason="accepted") is True
    ack = sender.messages[-1]
    assert ack["accepted"] is True
    assert ack["canonical_target"] == "apple"
    assert "action_policy" not in ack
    assert "unexpected" not in ack


def test_find_apple_transitions_and_sends_ack_without_action_policy_type_error():
    cfg = OrchestratorConfig()
    core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
    cmd = _cmd()
    accepted, reason = core.handle_task_cmd(cmd)
    sender = _Sender()
    service = _service({}, sender)
    service.core = core

    assert accepted is True
    assert core.ctx.state == State.SEARCH_TABLE
    assert service._send_task_ack(cmd, accepted=accepted, reason=reason) is True
    assert sender.messages[-1]["accepted"] is True
    assert "action_policy" not in sender.messages[-1]


def test_manual_stop_ack_contract_is_unchanged():
    cmd = TaskCmd(ts=0.0, intent="MANUAL_STOP", confidence=1.0, cmd_id="stop-1", source="mobile_gateway", cmd="manual_stop")
    ack = make_task_ack(cmd, accepted=True, state="IDLE", reason="manual_stop accepted")
    assert ack["cmd"] == "manual_stop"
    assert ack["accepted"] is True


def test_ack_build_and_send_exceptions_return_failure(monkeypatch):
    service = _service({"action_policy": "fixed_grasp"}, _Sender(error=RuntimeError("send boom")))
    assert service._send_task_ack(_cmd(), accepted=True, reason="accepted") is False

    monkeypatch.setattr("orchestrator_service.runtime.service.make_task_ack", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("build boom")))
    assert service._send_task_ack(_cmd(), accepted=True, reason="accepted") is False
