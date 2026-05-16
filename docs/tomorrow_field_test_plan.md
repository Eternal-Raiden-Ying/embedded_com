# SC171 Table Docking Field Test Plan

Date: 2026-05-16
Branch: `sc171-table-docking-test-prep`

## Safety Defaults

- Keep `ORCH_SERIAL_DRY_RUN=1` until wheel direction, serial port, and stop behavior are confirmed.
- `ORCH_TABLE_EDGE_ONLY_TEST` defaults to `0`; enable it only for table edge docking prep.
- `ORCH_ASSUME_GRASP_SUCCESS_FOR_TEST` defaults to `0`; enable it only when intentionally testing the return-home skeleton without a real grasp verification source.
- Do not enable grasp until table edge-only STOP behavior is verified on the real chassis.

## 1. STM32 Motion Probe

Dry-run command encoding:

```bash
python3 orchestrator/stm32_motion_probe.py --dry-run --cmd sequence
```

Real serial smoke test, after confirming the port:

```bash
python3 orchestrator/stm32_motion_probe.py --port /dev/ttyHS1 --baudrate 115200 --cmd sequence
```

Pass criteria:

- `VEL`, `STOP`, `JOG_*`, and `STATUS` commands encode without exceptions.
- Real serial test shows expected wheel directions and STM32 ACK/status lines.
- Any unexpected wheel direction is fixed in mapping/config before continuing.

## 2. ROI Debug

Use explicit presets only; default ROI behavior remains unchanged.

```bash
ORCH_SERIAL_DRY_RUN=1 \
VISTA_TABLE_EDGE_ROI_PRESET=center_mid \
VISTA_EDGE_DBG=1 \
python3 -m VISTA.vision_module.app.app
```

Try `center_lower` and `full_width_lower` if the edge is outside the preview overlay:

```bash
VISTA_TABLE_EDGE_ROI_PRESET=center_lower VISTA_EDGE_DBG=1 python3 -m VISTA.vision_module.app.app
VISTA_TABLE_EDGE_ROI_PRESET=full_width_lower VISTA_EDGE_DBG=1 python3 -m VISTA.vision_module.app.app
```

Pass criteria:

- Preview ROI overlay covers the physical table edge.
- Low-rate logs show `[EDGE_DBG] valid=... dist=... yaw=... age_ms=... roi=...`.
- `age_ms` remains fresh during motion.

## 3. YOLO Table Detection Debug

Enable detection debug without faking table detections:

```bash
ORCH_TABLE_DET_ENABLED=1 \
ORCH_TABLE_DET_MIN_CONF=0.30 \
ORCH_TABLE_DET_CENTER_TOL=0.12 \
python3 -m VISTA.vision_module.app.app
```

Pass criteria:

- If the model exposes COCO table class `60`, logs show `[TABLE_DET]` with confidence, bbox center, and `left/center/right` hint.
- If the model has no table class, logs show `[TABLE_DET][NO_TABLE_CLASS]`.
- Do not treat non-table classes as table evidence.

## 4. Table Edge-Only Dry Run

Use the docking target values prepared for SC171:

```bash
ORCH_SERIAL_DRY_RUN=1 \
ORCH_TABLE_EDGE_ONLY_TEST=1 \
ORCH_TABLE_TARGET_DIST_CM=30 \
ORCH_TABLE_DIST_TOL_CM=5 \
ORCH_TABLE_STOP_MARGIN_CM=5 \
ORCH_TABLE_SETTLE_MS=500 \
ORCH_TABLE_STABLE_FRAMES=5 \
ORCH_TABLE_YAW_TOL_DEG=8 \
ORCH_TABLE_MAX_MICRO_ADJUST=3 \
python3 -m orchestrator_service.app.main
```

Pass criteria:

- State reaches `FINAL_LOCK`, sends STOP in the stop window, settles, and logs `[TABLE_DOCK][DONE]`.
- With edge-only enabled, it logs `[TABLE_EDGE_ONLY][DONE]` and stops before target search or grasp.
- Stale edge observations cause STOP/hold, not blind forward motion.

## 5. Table Edge-Only Real Serial

Run only after dry-run behavior is stable:

```bash
ORCH_SERIAL_DRY_RUN=0 \
ORCH_SERIAL_PORT=/dev/ttyHS1 \
ORCH_SERIAL_BAUDRATE=115200 \
ORCH_TABLE_EDGE_ONLY_TEST=1 \
ORCH_TABLE_TARGET_DIST_CM=30 \
ORCH_TABLE_DIST_TOL_CM=5 \
ORCH_TABLE_STOP_MARGIN_CM=5 \
ORCH_TABLE_SETTLE_MS=500 \
ORCH_TABLE_STABLE_FRAMES=5 \
ORCH_TABLE_YAW_TOL_DEG=8 \
ORCH_TABLE_MAX_MICRO_ADJUST=3 \
python3 -m orchestrator_service.app.main
```

Pass criteria:

- Chassis approaches slowly and stops outside collision distance.
- `JOG_FORWARD`, `JOG_BACKWARD`, or `JOG_TURN` appears only for bounded micro-adjusts.
- The robot remains stopped after `[TABLE_EDGE_ONLY][DONE]`.

## 6. Grasp Verification Entry

Default behavior after arm OK is conservative:

```bash
ORCH_ASSUME_GRASP_SUCCESS_FOR_TEST=0 python3 -m orchestrator_service.app.main
```

Expected default log when no real verification source exists:

```text
[GRASP][VERIFY_UNAVAILABLE] no real grasp verification source; not assuming success
```

To exercise the return-home skeleton only:

```bash
ORCH_ASSUME_GRASP_SUCCESS_FOR_TEST=1 python3 -m orchestrator_service.app.main
```

Pass criteria:

- Without a verification source and without assume-success, the system does not mark grasp success.
- With assume-success enabled, `GRASP_VERIFY` transitions to `RETURN_HOME`.
- Verification failure retries through existing grasp retry/error recovery.

## 7. Field Success Criteria

- STM32 probe commands are acknowledged and wheel directions are correct.
- ROI overlay tracks the actual edge in the camera mount used tomorrow.
- Table detection debug either reports real table class detections or explicitly reports no table class.
- Edge-only mode reaches `[TABLE_EDGE_ONLY][DONE]` and remains stopped.
- Stop margin is verified physically before enabling target search or grasp.

## Manual Confirmations

- Actual serial device path.
- Four-wheel direction mapping.
- STM32 feedback line format for ACK/BUSY/STATUS/TIMEOUT.
- Camera mounting angle and table-edge ROI preset.
- Actual safe `ORCH_TABLE_STOP_MARGIN_CM`.
- Whether the YOLO model contains COCO table class `60`.
