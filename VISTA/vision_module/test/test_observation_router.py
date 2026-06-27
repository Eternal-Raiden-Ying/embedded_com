#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from VISTA.vision_module.app.observation.router import ObservationRouter


def _vision_obs(status: str = "RUNNING") -> dict:
    return {
        "type": "vision_obs",
        "ts": 100.0,
        "stage": "SEARCH",
        "mode": "FIND_OBJECT",
        "status": status,
        "session_id": "session-1",
        "req_id": "req-1",
        "epoch": 7,
        "interaction": {"id": "interaction-1"},
        "perception": {
            "table_edge_obs": {"seq": 42, "ts": 99.95, "distance_m": 0.3},
            "target_obs": {"seq": 42, "ts": 99.95, "cx": 0.1},
            "home_tag_obs": {"seq": 42, "ts": 99.95, "tag_id": 3},
            "local_perception": {"yolo_infer_ms": 15.0},
        },
        "proposal": {"candidate": "cup"},
        "result": {"ok": True},
    }


def test_control_obs_excludes_diagnostic_payloads() -> None:
    router = ObservationRouter(control_send_interval_s=0.10)

    result = router.route(
        vision_obs=_vision_obs(),
        frame_meta={"frame_seq": 42, "frame_capture_ts": 99.95},
        now=100.0,
    )

    assert result.control_obs is not None
    assert set(result.control_obs["perception"]) == {"table_edge_obs", "target_obs", "home_tag_obs"}
    assert "local_perception" not in result.control_obs["perception"]
    assert "proposal" not in result.control_obs
    assert "result" not in result.control_obs


def test_diagnostic_obs_excludes_control_observations() -> None:
    router = ObservationRouter(control_send_interval_s=0.10)

    result = router.route(
        vision_obs=_vision_obs(),
        frame_meta={"frame_seq": 42, "frame_capture_ts": 99.95},
        now=100.0,
    )

    assert result.diagnostic_obs is not None
    assert result.diagnostic_obs["obs_class"] == "diagnostic"
    assert "table_edge_obs" not in result.diagnostic_obs["perception"]
    assert "target_obs" not in result.diagnostic_obs["perception"]
    assert "home_tag_obs" not in result.diagnostic_obs["perception"]
    assert result.diagnostic_obs["perception"]["local_perception"]["yolo_infer_ms"] == 15.0
    assert result.diagnostic_obs["proposal"] == {"candidate": "cup"}
    assert result.diagnostic_obs["result"] == {"ok": True}


def test_diagnostic_is_limited_to_one_hz() -> None:
    router = ObservationRouter(control_send_interval_s=0.10, diagnostic_send_interval_s=1.0)

    first = router.route(vision_obs=_vision_obs(), frame_meta={"frame_seq": 1, "frame_capture_ts": 100.0}, now=100.0)
    assert first.diagnostic_obs is not None
    router.mark_control_sent(100.0)
    router.mark_diagnostic_sent(100.0)

    second = router.route(vision_obs=_vision_obs(), frame_meta={"frame_seq": 2, "frame_capture_ts": 100.2}, now=100.2)
    assert second.control_obs is not None
    assert second.diagnostic_obs is None
    router.mark_control_sent(100.2)

    third = router.route(vision_obs=_vision_obs(), frame_meta={"frame_seq": 3, "frame_capture_ts": 101.0}, now=101.0)
    assert third.control_obs is not None
    assert third.diagnostic_obs is not None


def test_control_can_route_at_ten_hz() -> None:
    router = ObservationRouter(control_send_interval_s=0.10)

    first = router.route(vision_obs=_vision_obs(), frame_meta={"frame_seq": 1, "frame_capture_ts": 100.0}, now=100.0)
    assert first.control_obs is not None
    router.mark_control_sent(100.0)

    skipped = router.route(vision_obs=_vision_obs(), frame_meta={"frame_seq": 2, "frame_capture_ts": 100.05}, now=100.05)
    assert skipped.control_obs is None
    assert skipped.skipped

    second = router.route(vision_obs=_vision_obs(), frame_meta={"frame_seq": 3, "frame_capture_ts": 100.1}, now=100.1)
    assert second.control_obs is not None


def test_latency_fields_are_injected() -> None:
    router = ObservationRouter(control_send_interval_s=0.10)

    result = router.route(
        vision_obs=_vision_obs(),
        frame_meta={"frame_seq": 42, "frame_capture_ts": 99.95},
        now=100.0,
    )

    assert result.control_obs is not None
    control = result.control_obs
    for field in (
        "frame_id",
        "capture_ts",
        "process_done_ts",
        "send_ts",
        "process_latency_ms",
        "send_latency_ms",
        "obs_total_age_ms",
    ):
        assert field in control
        assert field in control["perception"]["table_edge_obs"]
    assert control["frame_id"] == 42
    assert control["capture_ts"] == 99.95
    assert control["process_done_ts"] == 100.0
    assert control["process_latency_ms"] >= 49.9


def test_skip_and_drop_counts_update() -> None:
    router = ObservationRouter(control_send_interval_s=0.10)

    first = router.route(vision_obs=_vision_obs(), frame_meta={"frame_seq": 1, "frame_capture_ts": 100.0}, now=100.0)
    assert first.control_obs is not None
    router.mark_control_sent(100.0)

    skipped = router.route(vision_obs=_vision_obs(), frame_meta={"frame_seq": 1, "frame_capture_ts": 100.0}, now=100.01)
    assert skipped.skipped
    assert router.metrics.obs_skip_count == 1

    router.mark_drop()
    assert router.metrics.obs_drop_count == 1

    second = router.route(vision_obs=_vision_obs(), frame_meta={"frame_seq": 1, "frame_capture_ts": 100.0}, now=100.2)
    assert second.control_obs is not None
    assert second.control_obs["metrics"]["obs_skip_count"] == 1
    assert second.control_obs["metrics"]["obs_drop_count"] == 1
    assert second.control_obs["metrics"]["same_frame_reuse_count"] == 1
