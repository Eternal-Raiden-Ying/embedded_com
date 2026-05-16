# SC171 Overnight Codex Summary

Date: 2026-05-16
Branch: `sc171-table-docking-test-prep`
Base: `a22b551 Add STM32 motion protocol probe`

## Commits

- `814ba62 docs: record table docking test context`
- `4b7d849 test: verify STM32 motion probe`
- `71c2611 bridge: support STM32 motion protocol encoding and feedback`
- `84ba7c6 vision: add table edge ROI debug presets`
- `3774939 vision: add table detection debug and coarse direction hint`
- `6560840 orchestrator: add table edge only test mode`
- `6607ae7 orchestrator: add grasp verification gate`
- `be15f02 docs: add table docking field test plan`

## Modified Areas

- `orchestrator/stm32_motion_probe.py`: verified dry-run sequence path from previous STM32 protocol work.
- `orchestrator/orchestrator_service/bridge/simple_car_protocol.py`: STM32-compatible encoder aliases and feedback parsing compatibility.
- `VISTA/vision_module/utils/table_roi.py`: explicit ROI presets for table-edge test setup.
- `VISTA/vision_module/backend/table_edge_roi.py`: ROI debug and preset support.
- `VISTA/vision_module/backend/table_edge_manager.py`: edge debug output wiring.
- `VISTA/vision_module/backend/predictor_manager.py`: table detection debug and coarse direction hint.
- `orchestrator/orchestrator_service/runtime/state_machine.py`: table edge-only stop path and grasp verification gate.
- `orchestrator/orchestrator_service/config/schema.py`: SC171 table docking defaults and test switches.
- `orchestrator/orchestrator_service/config/board_config.py`: environment variable wiring.
- `tests/test_simple_car_protocol.py`, `tests/test_orch_operator_console.py`, `VISTA/vision_module/test/test_table_roi.py`: focused regression coverage.
- `docs/tomorrow_test_context.md`, `docs/tomorrow_field_test_plan.md`: test context and copyable field commands.

## New Configuration

- `ORCH_TABLE_EDGE_ONLY_TEST=0`: default off. When enabled, final table-edge lock stops and ends before target search/grasp.
- `ORCH_TABLE_TARGET_DIST_CM=30`: default target measured table distance.
- `ORCH_TABLE_DIST_TOL_CM=5`: default distance tolerance.
- `ORCH_TABLE_STOP_MARGIN_CM=5`: default stop margin before final settle.
- `ORCH_TABLE_SETTLE_MS=500`: default stop settle time.
- `ORCH_TABLE_STABLE_FRAMES=5`: default stable frames before dock done.
- `ORCH_TABLE_YAW_TOL_DEG=8`: default yaw tolerance.
- `ORCH_TABLE_MAX_MICRO_ADJUST=3`: default bounded micro-adjust count.
- `ORCH_ASSUME_GRASP_SUCCESS_FOR_TEST=0`: default off. When enabled, `GRASP_VERIFY` can enter `RETURN_HOME` without a real verification source.
- `VISTA_TABLE_EDGE_ROI_PRESET`: explicit ROI preset, expected values include `center_mid`, `center_lower`, and `full_width_lower`.
- `VISTA_EDGE_DBG=1`: emits low-rate edge debug.
- `ORCH_TABLE_DET_ENABLED=1`: enables table detection debug.
- `ORCH_TABLE_DET_MIN_CONF`, `ORCH_TABLE_DET_CENTER_TOL`: table detection debug thresholds.

## Offline Verification

Passed:

```bash
python3 -m unittest tests.test_simple_car_protocol tests.test_orch_operator_console
```

Result: 57 tests passed.

Passed:

```bash
python3 orchestrator/stm32_motion_probe.py --dry-run --cmd sequence
```

Observed dry-run TX lines:

```text
[PROBE][TX] STOP 1
[PROBE][TX] STATUS
[PROBE][TX] JOG 30 30 30 30 100 2
[PROBE][TX] STOP 3
[PROBE][TX] STATUS
```

Passed:

```bash
/usr/bin/python3 -m unittest VISTA.vision_module.test.test_table_roi VISTA.vision_module.test.test_preview_table_bbox
```

Result: 19 tests passed.

Passed:

```bash
python3 -m py_compile orchestrator/stm32_motion_probe.py orchestrator/orchestrator_service/bridge/simple_car_protocol.py orchestrator/orchestrator_service/config/schema.py orchestrator/orchestrator_service/config/board_config.py orchestrator/orchestrator_service/runtime/context.py orchestrator/orchestrator_service/runtime/state_machine.py VISTA/vision_module/utils/table_roi.py VISTA/vision_module/backend/preview/opencv_sink.py VISTA/vision_module/backend/table_edge_manager.py
```

Environment note:

- `python3 -m unittest VISTA.vision_module.test.test_table_roi VISTA.vision_module.test.test_preview_table_bbox` failed in the active conda Python because `numpy` is not installed there.
- The same VISTA tests passed under `/usr/bin/python3`, which has the needed dependencies.

## Tomorrow Priority Steps

1. Confirm `/dev/ttyHS1` or the actual STM32 serial device.
2. Run STM32 probe dry-run, then real serial sequence with the robot raised or wheels safe.
3. Confirm wheel direction mapping for forward, backward, turn, and stop.
4. Pick the ROI preset that visibly covers the physical table edge.
5. Run `ORCH_TABLE_DET_ENABLED=1` and confirm whether the model has COCO table class `60`.
6. Run table edge-only dry-run with the SC171 defaults.
7. Run table edge-only real serial only after dry-run STOP behavior is stable.
8. Measure the physical stop margin and adjust `ORCH_TABLE_STOP_MARGIN_CM` before enabling target search or grasp.
9. Keep `ORCH_ASSUME_GRASP_SUCCESS_FOR_TEST=0` unless explicitly testing return-home skeleton behavior.

## Manual Confirmation Points

- Actual serial device path and baudrate.
- Four-wheel direction and scale mapping.
- STM32 feedback text for ACK, BUSY, STATUS, TIMEOUT, and JOG events.
- Camera mount angle and table-edge ROI preset.
- Whether table detection debug reports real table class detections or `[TABLE_DET][NO_TABLE_CLASS]`.
- Safe final `ORCH_TABLE_TARGET_DIST_CM` and `ORCH_TABLE_STOP_MARGIN_CM` values.

## Working Tree Note

- `ROBOT_MOTION_CONTRACT.md` remains untracked and was intentionally not committed.
