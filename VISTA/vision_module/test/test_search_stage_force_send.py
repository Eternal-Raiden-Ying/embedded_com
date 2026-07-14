import sys
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
VISION_ROOT = TEST_DIR.parent
VISTA_ROOT = VISION_ROOT.parent
STACK_ROOT = VISTA_ROOT.parent
if str(STACK_ROOT) not in sys.path:
    sys.path.insert(0, str(STACK_ROOT))
if str(VISTA_ROOT) not in sys.path:
    sys.path.insert(0, str(VISTA_ROOT))

from vision_module.app.stages.base import StageContext, StageTickInput
from vision_module.app.stages.search.stage import SearchStagePlan


def _edge_result(obs_seq=7, frame_id=500, obs_ts=50.0):
    return {
        "obs_seq": obs_seq,
        "frame_id": frame_id,
        "obs_ts": obs_ts,
        "source_frame_id": frame_id,
        "sync_status": "unavailable",
        "edge_found": False,
        "edge_valid": False,
    }


def _process(plan, ctx, local):
    results = {"table_edge_obs": _edge_result(), "local_perception": local}
    return plan._process_table_edge_obs(
        results=results,
        ctx=ctx,
        tick_input=StageTickInput(ts=50.05, generation=1, results=results),
        local_perception=local,
    )


def _process_at(plan, ctx, local, *, tick_ts, edge=None):
    results = {"table_edge_obs": edge or _edge_result(), "local_perception": local}
    return plan._process_table_edge_obs(
        results=results,
        ctx=ctx,
        tick_input=StageTickInput(ts=tick_ts, generation=1, results=results),
        local_perception=local,
    )


def test_local_bbox_change_forces_send_with_unchanged_edge_identity():
    plan = SearchStagePlan()
    ctx = StageContext(current_mode="FIND_EDGE")
    empty = {
        "frame_id": 501,
        "obs_ts": 50.01,
        "trace_id": "rgb-empty",
        "table_bbox_current_found": False,
        "table_bbox": None,
        "rgb_shape": (480, 640, 3),
    }
    with_bbox = {
        "frame_id": 502,
        "obs_ts": 50.02,
        "trace_id": "rgb-table",
        "table_bbox_current_found": True,
        "table_bbox": [300, 80, 620, 460],
        "rgb_shape": (480, 640, 3),
    }

    _, _, first_force = _process(plan, ctx, empty)
    merged, _, bbox_force = _process(plan, ctx, with_bbox)
    _, _, duplicate_force = _process(plan, ctx, with_bbox)

    assert first_force is True
    assert bbox_force is True
    assert merged["table_bbox_current_found"] is True
    assert merged["force_send_reason"].startswith("local_perception_identity_changed")
    assert duplicate_force is False


def test_local_bbox_coordinates_and_freshness_participate_in_identity():
    plan = SearchStagePlan()
    ctx = StageContext(current_mode="FIND_EDGE")
    local = {
        "frame_id": 510,
        "obs_ts": 51.0,
        "trace_id": "rgb-510",
        "table_bbox_current_found": True,
        "yolo_table_fresh": True,
        "table_bbox": [100, 80, 400, 440],
        "rgb_shape": (480, 640, 3),
    }
    _process(plan, ctx, local)

    moved = dict(local, table_bbox=[120, 80, 420, 440])
    _, _, moved_force = _process(plan, ctx, moved)
    stale = dict(moved, yolo_table_fresh=False)
    _, _, freshness_force = _process(plan, ctx, stale)

    assert moved_force is True
    assert freshness_force is True


def _completed_positive(seq=1, obs_ts=50.0):
    return {
        "frame_id": 501,
        "obs_ts": obs_ts,
        "inference_seq": seq,
        "inference_executed": True,
        "inference_completed": True,
        "has_infer": True,
        "table_bbox_current_found": True,
        "table_bbox": [300, 80, 620, 460],
        "rgb_shape": (480, 640, 3),
    }


def test_no_inference_ticks_reuse_last_completed_bbox_without_negative():
    plan = SearchStagePlan()
    ctx = StageContext(current_mode="FIND_EDGE")
    _process_at(plan, ctx, _completed_positive(), tick_ts=50.0)

    for index in range(1, 6):
        held, _, force_send = _process_at(
            plan,
            ctx,
            {
                "frame_id": 501 + index,
                "obs_ts": 50.0 + index * 0.1,
                "has_infer": False,
                "inference_executed": False,
                "inference_completed": False,
            },
            tick_ts=50.0 + index * 0.1,
        )
        assert held["table_bbox_current_found"] is True
        assert held["table_bbox_control_valid"] is True
        assert held["yolo_table_fresh"] is True
        assert held["table_bbox_xyxy"] == [300.0, 80.0, 620.0, 460.0]
        assert held["has_new_inference"] is False
        assert held["explicit_negative_detection"] is False
        assert held["bbox_hold_reason"] == "no_new_inference"
        assert force_send is False


def test_pending_inference_reuses_last_completed_bbox():
    plan = SearchStagePlan()
    ctx = StageContext(current_mode="FIND_EDGE")
    _process_at(plan, ctx, _completed_positive(), tick_ts=50.0)
    held, _, _ = _process_at(
        plan,
        ctx,
        {
            "frame_id": 502,
            "obs_ts": 50.2,
            "has_infer": True,
            "inference_executed": True,
            "inference_completed": False,
        },
        tick_ts=50.2,
    )
    assert held["table_bbox_current_found"] is True
    assert held["bbox_hold_reason"] == "inference_pending"
    assert held["explicit_negative_detection"] is False


def test_completed_negative_is_countable_once_not_repeated_by_empty_ticks():
    plan = SearchStagePlan()
    ctx = StageContext(current_mode="FIND_EDGE")
    _process_at(plan, ctx, _completed_positive(), tick_ts=50.0)
    negative = {
        "frame_id": 510,
        "obs_ts": 50.2,
        "inference_seq": 2,
        "inference_executed": True,
        "inference_completed": True,
        "has_infer": True,
        "table_bbox_current_found": False,
        "table_bbox": None,
        "rgb_shape": (480, 640, 3),
    }
    cleared, _, negative_force = _process_at(plan, ctx, negative, tick_ts=50.2)
    assert cleared["table_bbox_current_found"] is False
    assert cleared["explicit_negative_detection"] is True
    assert cleared["inference_seq"] == 2
    assert negative_force is True

    repeated, _, repeated_force = _process_at(
        plan,
        ctx,
        {"frame_id": 511, "obs_ts": 50.3, "has_infer": False, "inference_completed": False},
        tick_ts=50.3,
    )
    assert repeated["table_bbox_current_found"] is False
    assert repeated["explicit_negative_detection"] is False
    assert repeated["inference_seq"] == 2
    assert repeated_force is False


def test_edge_update_does_not_clear_held_completed_bbox():
    plan = SearchStagePlan()
    ctx = StageContext(current_mode="FIND_EDGE")
    _process_at(plan, ctx, _completed_positive(), tick_ts=50.0)
    edge = _edge_result(obs_seq=8, frame_id=502, obs_ts=50.1)
    edge.update({"edge_found": True, "edge_valid": True, "point_count": 25, "yaw_err_rad": 0.06})
    merged, _, force_send = _process_at(
        plan,
        ctx,
        {"frame_id": 502, "obs_ts": 50.1, "has_infer": False, "inference_completed": False},
        tick_ts=50.1,
        edge=edge,
    )
    assert merged["edge_found"] is True
    assert merged["point_count"] == 25
    assert merged["table_bbox_current_found"] is True
    assert force_send is True


def test_completed_bbox_expires_after_absolute_ttl():
    plan = SearchStagePlan()
    ctx = StageContext(current_mode="FIND_EDGE")
    _process_at(plan, ctx, _completed_positive(obs_ts=50.0), tick_ts=50.0)
    expired, _, force_send = _process_at(
        plan,
        ctx,
        {"frame_id": 520, "obs_ts": 51.3, "has_infer": False, "inference_completed": False},
        tick_ts=51.3,
    )
    assert expired["table_bbox_current_found"] is False
    assert expired["table_bbox_control_valid"] is False
    assert expired["yolo_table_fresh"] is False
    assert expired["explicit_negative_detection"] is False
    assert expired["bbox_hold_reason"] == "completed_bbox_stale_timeout"
    assert force_send is True
