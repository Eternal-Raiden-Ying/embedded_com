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


def _edge_result():
    return {
        "obs_seq": 7,
        "frame_id": 500,
        "obs_ts": 50.0,
        "source_frame_id": 500,
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
