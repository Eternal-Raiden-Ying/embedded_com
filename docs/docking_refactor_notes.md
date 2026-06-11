# Docking Refactor Notes

This file is a legacy note for previous docking refactor work. It is not the source of truth for current state names or runtime ownership.

Current source-of-truth documents:

- `docs/architecture.md`
- `docs/config.md`
- `docs/system_runbook.md`

## Current State Names

Current table docking path:

```text
SEARCH_TABLE
YOLO_ACQUIRE_ALIGN
YOLO_APPROACH
EDGE_ADJUST
FINAL_SLOW_STOP
AT_TABLE_EDGE
```

Legacy names from older notes:

- legacy `COARSE_ALIGN` maps to current `YOLO_ACQUIRE_ALIGN` and `EDGE_ADJUST`.
- legacy `CONTROLLED_APPROACH` maps to current `YOLO_APPROACH`.
- legacy `FINAL_LOCK` maps to current `FINAL_SLOW_STOP`.

Do not introduce new code, tests, or documentation that treats the legacy names as active runtime states.

## Current Ownership

- Runtime state flow: `orchestrator/orchestrator_service/runtime/states/table_docking.py`
- Base motion safety: `orchestrator/orchestrator_service/runtime/safety/base_motion_safety.py`
- Velocity limits and STOP policy: `orchestrator/orchestrator_service/control/motion/`
- Tunable docking parameters: `orchestrator/configs/stage_params.yaml`
- Base command parameters: `orchestrator/configs/car_cmd_params.yaml`

## Notes For Future Refactors

Keep docking algorithm changes separate from directory or documentation cleanup. If a threshold, formula, or transition condition changes, add a focused test under `tests/orchestrator/` and record the runtime parameter affected.
