# Tomorrow Test Context

## Branch
- Work branch: `sc171-table-docking-test-prep`
- Base branch observed before this work: `sc171-stm32-motion-protocol`
- Protocol source of truth: `ROBOT_MOTION_CONTRACT.md`

## Current Motion Path
- Existing old chain: `state_machine -> MotionController -> CmdVel -> SimpleCarMapper -> UartBridge`
- Current preparation rule: do not let the state machine directly assemble STM32 UART strings.
- First test the SC171/STM32 protocol with the standalone probe, then enable state-machine integration only through config-gated test modes.

## STM32 Motion Protocol
SC171 sends:

```text
VEL <s006> <s007> <s008> <s009> <seq>
STOP <seq>
JOG <s006> <s007> <s008> <s009> <duration_ms> <seq>
STATUS
```

Limits:
- Wheel values `s006/s007/s008/s009`: `-100..100`.
- `seq`: increasing command number.
- `duration_ms`: recommended `20..1000`.

STM32 returns:

```text
[CAR][JOG_START] seq=...
[CAR][JOG_DONE] seq=...
[CAR][JOG_BUSY] seq=...
[CAR][TIMEOUT] auto stop
```

## Tomorrow Test Goals
- Confirm `/dev/ttyHS1 @ 115200` or record the actual serial port.
- Verify `STOP`, `STATUS`, and a short `JOG` from SC171 without starting orchestrator/VISTA/mobile gateway.
- Record four-wheel direction mapping and low-speed values: stable forward/reverse start P, `dz_pos`, `dz_neg`, `max_offset`, continuous slow speed, JOG speed/duration.
- Mount camera and check whether table edge is inside the debug ROI presets, especially `center_mid`, `center_lower`, and `full_width_lower`.
- Confirm whether YOLO/local perception provides a real table class/bbox; if not, use manual initial pose and table-edge ROI debug.
- In table-edge-only mode, verify approach, stop at about 30 cm, settle, stable frames, yaw tolerance, and final STOP behavior.
- Measure STOP overshoot and update `stop_margin_cm`.

## Confirmed Files
- `orchestrator/orchestrator_service/runtime/service.py`
- `orchestrator/orchestrator_service/runtime/state_machine.py`
- `orchestrator/orchestrator_service/runtime/controller.py`
- `orchestrator/orchestrator_service/control/docking_controller.py`
- `orchestrator/orchestrator_service/bridge/simple_car_protocol.py`
- `orchestrator/orchestrator_service/bridge/uart_bridge.py`
- `orchestrator/orchestrator_service/ipc/protocol.py`
- `orchestrator/orchestrator_service/config/schema.py`
- `orchestrator/orchestrator_service/config/board_config.py`
- `orchestrator/stm32_motion_probe.py`

## Unknowns To Resolve On Robot
- Actual STM32 serial device and permissions.
- Actual STM32 line endings and exact feedback spelling.
- Four-wheel sign mapping for `006/007/008/009`.
- Camera angle and whether depth/edge ROI covers the true table edge.
- Whether the active model has COCO table class `60` or table-like labels.
- Safe `target_dist_cm`, `stop_margin_cm`, `yaw_tolerance`, and stable-frame settings after measuring real latency.
