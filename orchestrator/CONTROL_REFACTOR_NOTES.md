# Control Layer Refactor Notes

This package contains a first-pass control-layer refactor for the table docking pipeline.

## Added modules

- `orchestrator_service/runtime/perception_semantics.py`
  - Normalizes raw `TableEdgeObs` into explicit semantic flags:
    - `table_bbox_found`
    - `edge_detected`
    - `edge_valid`
    - `edge_stable`
    - `edge_trusted`
  - YOLO table bbox confidence is logged but not used as a control gate.
  - `edge_trusted` requires table bbox + valid/stable edge + basic quality checks.

- `orchestrator_service/runtime/control_authority.py`
  - Defines `ControlAuthority` and the first-pass authority vocabulary:
    - `yolo_forward`
    - `edge_adjust`
    - `local_rotate_search`
    - `final_lock`
    - `search_failed_stop`

## Main behavior changes

- `yolo_forward` is now a forward-safe command: `vx > 0`, `wz = 0`.
- If a table bbox exists and edge is not trusted, `COARSE_ALIGN` is blocked and control falls back to `yolo_forward`.
- `SEARCH_TABLE` prioritizes table bbox / YOLO forward before old plane/docking alignment transitions.
- `CONTROLLED_APPROACH` does not switch back to `COARSE_ALIGN` unless edge is trusted.
- The old 0.40 bbox-area gate is deprecated. `bbox_area_ratio` may remain as diagnostic data but should not control authority.

## Validation performed

- `python3 -m compileall -q orchestrator_service` passed.

