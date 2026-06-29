from types import SimpleNamespace

from orchestrator.orchestrator_service.runtime.context import RuntimeContext, State
from orchestrator.orchestrator_service.runtime.depth_safety import apply_close_range_depth_safety_gate
from orchestrator.orchestrator_service.runtime.motion_arbiter import ArbitrationResult


def _cfg():
    return SimpleNamespace(
        roi_final_stop_p10_m=0.42,
        roi_final_slow_p10_m=0.52,
        roi_final_missing_hold_s=0.8,
        final_probe_vx_mps=0.008,
        final_missing_probe_vx_mps=0.004,
        close_range_probe_vx_mps=0.008,
        close_range_missing_probe_vx_mps=0.004,
        final_probe_timeout_s=8.0,
        final_probe_distance_budget_m=0.15,
    )


def _obs(valid=True, p10=None):
    return SimpleNamespace(
        table_roi_depth_valid=bool(valid),
        table_roi_depth_p10=p10,
        depth_p10=None,
    )


def _result(vx=0.006):
    return ArbitrationResult(
        final_vx=vx,
        final_vy=0.01,
        final_wz=0.15,
        motion_class="normal",
        stop_class="none",
        blocked_by="",
        reason="test_probe",
        allow_uart_send=True,
        service_may_override=False,
        summary={"close_range_latched": True, "docking_action": "CLOSE_RANGE_PROBE"},
    )


def test_close_range_current_hard_stop_holds_on_first_frame():
    ctx = RuntimeContext(state=State.FINAL_SLOW_STOP)
    out = apply_close_range_depth_safety_gate(ctx, _obs(valid=True, p10=0.39), _result(), _cfg(), now_mono=10.0)

    assert out.final_vx == 0.0
    assert out.final_vy == 0.0
    assert out.final_wz == 0.0
    assert out.summary["depth_safety_state"] == "hard_stop_confirming"
    assert out.summary["final_locked"] is False
    assert out.summary["docking_action"] == "DEPTH_SAFETY_HOLD"


def test_close_range_current_hard_stop_locks_on_second_frame():
    ctx = RuntimeContext(state=State.FINAL_SLOW_STOP)
    apply_close_range_depth_safety_gate(ctx, _obs(valid=True, p10=0.39), _result(), _cfg(), now_mono=10.0)
    out = apply_close_range_depth_safety_gate(ctx, _obs(valid=True, p10=0.39), _result(), _cfg(), now_mono=10.1)

    assert out.final_vx == 0.0
    assert out.summary["depth_safety_state"] == "hard_stop_locked"
    assert out.summary["final_locked"] is True
    assert out.summary["docking_action"] == "FINAL_LOCKED_STOP"


def test_close_range_slow_depth_caps_probe_speed_and_disables_yaw_lateral():
    ctx = RuntimeContext(state=State.FINAL_SLOW_STOP)
    out = apply_close_range_depth_safety_gate(ctx, _obs(valid=True, p10=0.50), _result(vx=0.02), _cfg(), now_mono=10.0)

    assert 0.0 < out.final_vx <= 0.008
    assert out.final_vy == 0.0
    assert out.final_wz == 0.0
    assert out.summary["depth_safety_state"] == "slow_cap"


def test_close_range_recent_missing_depth_caps_to_missing_probe_speed():
    ctx = RuntimeContext(state=State.FINAL_SLOW_STOP)
    ctx.last_valid_depth_p10_m = 0.50
    ctx.last_valid_depth_p10_source = "obs.table_roi_depth_p10"
    ctx.last_valid_depth_p10_mono = 9.7
    ctx.depth_missing_started_mono = 9.7

    out = apply_close_range_depth_safety_gate(ctx, _obs(valid=False, p10=None), _result(vx=0.02), _cfg(), now_mono=10.0)

    assert 0.0 < out.final_vx <= 0.004
    assert out.final_vy == 0.0
    assert out.final_wz == 0.0
    assert out.summary["depth_safety_state"] == "missing_short_probe"


def test_close_range_depth_missing_timeout_holds_without_locking():
    ctx = RuntimeContext(state=State.FINAL_SLOW_STOP)
    ctx.last_valid_depth_p10_m = 0.50
    ctx.last_valid_depth_p10_source = "obs.table_roi_depth_p10"
    ctx.last_valid_depth_p10_mono = 9.0
    ctx.depth_missing_started_mono = 9.1

    out = apply_close_range_depth_safety_gate(ctx, _obs(valid=False, p10=None), _result(vx=0.02), _cfg(), now_mono=10.0)

    assert out.final_vx == 0.0
    assert out.final_vy == 0.0
    assert out.final_wz == 0.0
    assert out.summary["depth_safety_state"] == "missing_hold"
    assert out.summary["final_locked"] is False
    assert out.summary["docking_action"] == "DEPTH_SAFETY_HOLD"
