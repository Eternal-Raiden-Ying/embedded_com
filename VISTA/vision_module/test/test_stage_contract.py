#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from vision_module.app.stages.base import StageContext, StageTickInput
from vision_module.app.stages.search import SearchStagePlan
from vision_module.app.stages.search.request_mapping import canonical_search_mode
from vision_module.ipc.protocol import VisionObs, VisionObsEnvelope, VisionReq


def _start_req(mode_hint="FIND_OBJECT", search_kind="TARGET", target="cup") -> VisionReq:
    return VisionReq(
        ts=123.0,
        op="START",
        stage="SEARCH",
        mode_hint=mode_hint,
        target=target,
        payload={"search_kind": search_kind},
    )


def test_track_local_maps_to_find_object() -> None:
    assert canonical_search_mode("TRACK_LOCAL") == "FIND_OBJECT"
    req = _start_req(mode_hint="TRACK_LOCAL", search_kind="TARGET")
    ctx = StageContext()

    output = SearchStagePlan().on_enter(req, ctx)

    assert output is None
    assert ctx.current_mode == "FIND_OBJECT"


def test_depth_perception_maps_to_find_edge() -> None:
    assert canonical_search_mode("DEPTH_PERCEPTION") == "FIND_EDGE"
    req = _start_req(mode_hint="DEPTH_PERCEPTION", search_kind="TABLE_EDGE")
    ctx = StageContext()

    output = SearchStagePlan().on_enter(req, ctx)

    assert output is None
    assert ctx.current_mode == "FIND_EDGE"


def test_invalid_search_kind_returns_failed() -> None:
    req = _start_req(mode_hint="FIND_OBJECT", search_kind="INVALID_KIND")
    ctx = StageContext()

    output = SearchStagePlan().on_enter(req, ctx)

    assert output is not None
    assert output.vision_obs is not None
    assert output.vision_obs["status"] == "FAILED"
    assert "invalid search_kind" in output.vision_obs["result"]["reason"]


def test_find_edge_outputs_table_edge_obs() -> None:
    plan = SearchStagePlan()
    ctx = StageContext()
    plan.on_enter(_start_req(mode_hint="FIND_EDGE", search_kind="TABLE_EDGE"), ctx)

    output = plan.tick(
        StageTickInput(
            ts=200.0,
            generation=1,
            results={"table_edge_obs": {"edge_found": True, "confidence": 0.8, "obs_ts": 199.99}},
        ),
        ctx,
    )

    perception = output.vision_obs["perception"]
    assert output.vision_obs["mode"] == "FIND_EDGE"
    assert "table_edge_obs" in perception
    assert "target_obs" not in perception
    assert perception["table_edge_obs"]["edge_found"] is True


def test_find_object_outputs_target_obs() -> None:
    plan = SearchStagePlan()
    ctx = StageContext()
    plan.on_enter(_start_req(mode_hint="FIND_OBJECT", search_kind="TARGET", target="cup"), ctx)

    output = plan.tick(
        StageTickInput(
            ts=200.0,
            generation=1,
            results={"local_perception": {"target_obs": {"found": True, "target": "cup"}}},
        ),
        ctx,
    )

    perception = output.vision_obs["perception"]
    assert output.vision_obs["mode"] == "FIND_OBJECT"
    assert "target_obs" in perception
    assert perception["target_obs"]["found"] is True


def test_vision_obs_envelope_uses_shared_obs_class_contract() -> None:
    env = VisionObsEnvelope.from_dict(
        {
            "ts": 1.0,
            "stage": "search",
            "mode": "find_object",
            "status": "running",
            "obs_class": "Diagnostic",
            "perception": {"local_perception": {"ok": True}},
        }
    )

    assert env.stage == "SEARCH"
    assert env.mode == "FIND_OBJECT"
    assert env.status == "RUNNING"
    assert env.obs_class == "diagnostic"
    assert env.to_dict()["obs_class"] == "diagnostic"
    assert VisionObs(ts=1.0, stage="SEARCH", mode="FIND_EDGE", status="RUNNING").to_dict()["obs_class"] == "control"
