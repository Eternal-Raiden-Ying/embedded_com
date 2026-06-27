# Table Docking Motion Architecture

## Layer Ownership

Table docking motion is a one-way pipeline:

`Raw observation / VISTA signal -> DockingObservation -> DockingStage -> DockingAction / MotionIntent -> DockingMotionArbiter -> DockingMotionResult -> Service TxPolicy -> UartBridge`

- `DockingObservation` describes perception and safety facts only.
- `DockingStage` describes task phase only.
- `DockingMotionResult` is the single source of truth for final `vx/vy/wz`, `DockingAction`, `StopClass`, and log-compatible summary fields.
- Legacy summary fields are derived from `DockingMotionResult`; they must not drive final table-docking motion.
- Service only applies emergency/safety stops, explicit idle/shutdown outside active docking, and send/write metadata.
- UART only sends protocol lines and suppresses velocity during emergency/safety cooldown.

## DockingStage

- `SEARCH`: no valid table observation; search rotate only.
- `BBOX_ACQUIRE`: bbox is usable and owns reacquire yaw; forward is blocked.
- `EDGE_HANDOFF`: bbox is safe enough while edge stability is being confirmed.
- `EDGE_APPROACH`: edge owns yaw and forward approach.
- `NEAR_EDGE_APPROACH`: near latch is active; YOLO no longer owns yaw.
- `FINAL_DISTANCE_HOLD`: final depth is latched; `vx/vy` are blocked.
- `FINAL_YAW_ALIGN`: final depth is latched and yaw remains large; `wz` is allowed.
- `FINAL_LOCKED`: final depth and yaw are locked; all axes zero.
- `RECOVERY_ROTATE`: recoverable perception/control condition; rotate or hold, not safety STOP.
- `SAFETY_STOP`: hard safety condition.
- `EMERGENCY_STOP`: explicit emergency condition.

## DockingAction

- `SEARCH_ROTATE`: `vx=0`, `vy=0`, `wz=search_wz`.
- `BBOX_REACQUIRE_ROTATE`: `vx=0`, `vy=0`, `wz=bbox_yaw_cmd`.
- `EDGE_APPROACH_FORWARD`: low forward velocity plus edge yaw correction.
- `NEAR_EDGE_FORWARD`: near-stage low forward or near hold with edge/last-good yaw.
- `PERCEPTION_DROPOUT_HOLD`: short approach dropout hold, not STOP.
- `FINAL_YAW_ALIGN`: `vx=0`, `vy=0`, `wz=edge_yaw_cmd` or last-good edge yaw.
- `FINAL_LOCKED_STOP`: final locked or final distance hold; all axes zero.
- `CONTROL_RECOVERY_ROTATE`: recoverable control rotate.
- `SAFETY_STOP`: hard safety stop.
- `EMERGENCY_STOP`: emergency stop.

## STOP Classes

- `none`: normal motion, search rotate, bbox reacquire, final yaw align, final locked hold.
- `control_recovery`: non-safety control recovery without emergency cooldown.
- `stale_recovery`: stale recovery without emergency cooldown when a recovery command is available.
- `safety`: hard depth collision, hard safety, or hardware safety failure.
- `emergency`: explicit STOP, E-stop, obstacle emergency.

Only `safety` and `emergency` may trigger E-stop cooldown or suppress velocity in the UART writer.

## Recovery Is Not Safety

These conditions must never directly become safety STOP:

- `bbox_fov_guard_hard`
- `bbox_center_extreme`
- `side_touch_center_error_streak`
- recoverable bbox loss
- short perception dropout
- hard stale with last-good recovery available

They should map to `BBOX_REACQUIRE_ROTATE`, `CONTROL_RECOVERY_ROTATE`, `PERCEPTION_DROPOUT_HOLD`, or `SEARCH_ROTATE`.

## Final And Near Rules

- Final depth latch blocks `vx/vy`; it does not block final yaw align.
- `FINAL_DEPTH_LATCHED + yaw large` must emit nonzero `wz` when edge or last-good yaw is available.
- Near/final latch downgrades YOLO bbox to diagnostic/FOV guard.
- After near/final latch, bbox lost must not transition docking back to `SEARCH_TABLE`.
- Active docking must not remain long-term `vx=0,wz=0` unless final hold, final locked, safety, or emergency is active.

## Real-Robot Log Checklist

Watch these fields during the next table docking run:

- Observation: `docking_observation`, `table_bbox_control_valid`, `yolo_bbox_center_x_norm`, `bbox_center_error_control`, `table_bbox_touch_left`, `table_bbox_touch_right`, `edge_found`, `edge_trusted`, `yaw_err_rad`, `table_roi_depth_valid`, `table_roi_depth_p10`.
- Stage/action: `docking_stage`, `docking_action`, `motion_intent_type`, `yaw_owner`, `arbitration_reason`, `blocked_by`.
- Final command: `final_vx`, `final_vy`, `final_wz`, `vx_mps`, `vy_mps`, `wz_radps`.
- Recovery: `fov_guard_level`, `fov_guard_reason`, `bbox_fov_guard_level`, `bbox_fov_guard_reason`, `perception_dropout_hold_active`, `zero_escape_reason`.
- Latches: `near_table_latched`, `near_table_latch_reason`, `final_depth_latched`, `final_depth_latch_reason`, `final_yaw_align_active`, `final_locked`.
- Service/UART: `active_table_docking`, `service_override`, `service_override_reason`, `effective_cmd_before_service`, `effective_cmd_after_service`, `estop_cooldown_applied`, `writer_accept_cmd`, `writer_discard_reason`, `serial_write_ok`, `uart_tx_ok`.
