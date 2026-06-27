# Table Docking Motion Architecture

## Layer Ownership

Table docking motion is a one-way pipeline:

`Raw observation / VISTA signal -> DockingObservation -> DockingStage -> DockingAction / MotionIntent -> DockingMotionArbiter -> DockingMotionResult -> Service TxPolicy -> UartBridge`

- `DockingObservation` describes perception and safety facts only.
- `DockingStage` describes task phase only.
- `DockingMotionResult` is the single source of truth for final `vx/vy/wz`, `DockingAction`, `StopClass`, and log-compatible summary fields.
- `yaw_owner`, `forward_owner`, and `lateral_owner` name the subsystem that owns each axis for the emitted action. Use `none` when an axis is intentionally blocked.
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
- `BBOX_TRACK_FORWARD`: far/mid-range bbox-assisted forward motion; `yaw_owner=bbox`, `forward_owner=bbox_track`, `lateral_owner=none`.
- `EDGE_READINESS_HANDOFF`: pre-approach edge handoff stabilization; `yaw_owner=edge_candidate` or `bbox_hold`, `forward_owner=none` or slow, `lateral_owner=none`.
- `EDGE_APPROACH_FORWARD`: low forward velocity plus edge yaw correction.
- `NEAR_EDGE_FORWARD`: near-stage low forward or near hold with edge/last-good yaw.
- `NEAR_EDGE_LATERAL_ALIGN`: reserved future Y-axis action; defined for logs/config only and must not emit nonzero `vy` in this round.
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

- **Authoritative Status**: `DockingStage` and `DockingAction` are the sole authoritative sources of table docking status. The legacy runtime `State` enum is kept for internal state machine transitions only and is downgraded to compatibility-only (displayed as `legacy_state` under diagnostics).
- `edge_readiness_score` is a pre-final handoff signal only. It may move BBOX tracking into `EDGE_READINESS_HANDOFF` / `EDGE_APPROACH_FORWARD`, but it must never override final depth latch, final yaw align, final locked stop, emergency, obstacle, explicit stop, or hard safety.
- Readiness uses enter/exit hysteresis: enter at `edge_readiness_enter_score`, remain stable between enter/exit, and fall back only at or below `edge_readiness_exit_score`.
- Final depth latch blocks `vx/vy`; it does not block final yaw align.
- The Y-axis interface is present through `lateral_owner`, `lateral_err_norm`, `lateral_err_m`, `lateral_source`, `vy_enabled`, `vy_block_reason`, `vy_cmd_raw`, and `vy_cmd_limited`. In this round `lateral_enabled=false` and all automatic table docking actions must keep `final_vy=0.0`.
- Before enabling nonzero automatic `vy`, run real-car direction calibration with manual commands `V 0.000 +0.008 0.000` and `V 0.000 -0.008 0.000`, then document the positive/negative direction convention.
- `FINAL_DEPTH_LATCHED + yaw large` must emit nonzero `wz` when edge or last-good yaw is available.
- Near/final latch downgrades YOLO bbox to diagnostic/FOV guard.
- After near/final latch, bbox lost must not transition docking back to `SEARCH_TABLE`.
- Active docking must not remain long-term `vx=0,wz=0` unless final hold, final locked, safety, or emergency is active.

## Final Yaw Lock and Realignment Hysteresis

To enter `FINAL_LOCKED`:
1. `final_depth_latched` must be True.
2. The absolute edge yaw error `abs(yaw_err_rad)` must be `<= final_yaw_deadband_rad` (default 0.12).
3. The yaw error must remain within the deadband for consecutive stability of at least `final_yaw_stable_frames` (default 6).
4. The system must undergo yaw alignment for at least `final_yaw_align_min_duration_ms` (default 1000ms), unless yaw error was already within deadband when final depth was latched.

To trigger Realignment:
- If currently `FINAL_LOCKED`, but subsequent fresh edge yaw shows `abs(yaw_err_rad) >= final_yaw_realign_rad` (default 0.18) consecutively for at least 3 frames, the system will break lock and re-enter `FINAL_YAW_ALIGN` (`vx=0, vy=0, wz!=0`).
- If edge data goes completely stale (age > `final_yaw_last_good_hold_s`), the vehicle will hold position (`vx=0, vy=0, wz=0`) under `FINAL_DISTANCE_HOLD` stage with reason `edge_yaw_stale` rather than rotating blindly or returning to SEARCH.

## Logging and Terminal Output

- **Console Display**: Active table docking terminal print prefers the low-frequency `[DOCK]` summary line (printed immediately on stage/action changes, or throttled to 0.5s):
  `[DOCK] stage=... action=... vx=... vy=... wz=... yaw=... depth_p10=... near=... final_depth=... locked=... uart_ok=... legacy_state=...`
- **Differentiated Commands**: Differentiated commands are logged inside structured run logs (`control_summary` and `motion_gate_trace` JSONL):
  - `candidate_cmd`: initial velocity command before table arbitration.
  - `arbiter_final_cmd`: output velocity command after table docking arbiter calculations.
  - `service_effective_cmd`: velocity command after service-level limits/safety/dry-run overrides.
  - `uart_tx_cmd`: velocity command actually written to serial (suppressed to 0 if transmission not allowed).

## Real-Robot Log Checklist

Watch these fields during the next table docking run:

- Observation: `docking_observation`, `table_bbox_control_valid`, `yolo_bbox_center_x_norm`, `bbox_center_error_control`, `table_bbox_touch_left`, `table_bbox_touch_right`, `edge_found`, `edge_trusted`, `yaw_err_rad`, `table_roi_depth_valid`, `table_roi_depth_p10`.
- Stage/action: `docking_stage`, `docking_action`, `docking_reason`, `yaw_owner`, `forward_owner`, `lateral_owner`, `advance_condition`, `fallback_condition`, `arbitration_reason`, `blocked_by`.
- Commands: `candidate_cmd`, `arbiter_final_cmd`, `service_effective_cmd`, `uart_tx_cmd`, `final_vx`, `final_vy`, `final_wz`, `vx_mps`, `vy_mps`, `wz_radps`.
- Recovery & Diagnostics: `fov_guard_level`, `fov_guard_reason`, `bbox_fov_guard_level`, `bbox_fov_guard_reason`, `perception_dropout_hold_active`, `zero_escape_reason`, `final_yaw_deadband_rad`, `final_yaw_realign_rad`, `final_yaw_stable_count`, `final_yaw_align_elapsed_ms`, `final_yaw_lock_block_reason`, `final_realign_triggered`, `final_yaw_source`.
- Latches: `near_table_latched`, `near_table_latch_reason`, `final_depth_latched`, `final_depth_latch_reason`, `final_yaw_align_active`, `final_locked`.
- Service/UART: `active_table_docking`, `service_override`, `estop_cooldown_applied`, `uart_tx_ok`.
