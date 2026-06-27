# Table Docking Control Contract

## VISTA -> Orchestrator table_edge_obs

VISTA must publish current-frame table signals under `perception.table_edge_obs`
or the legacy `table_edge_obs` message. The Orchestrator parser preserves the
following contract fields and records missing contract keys in
`obs_parse_missing_fields` for diagnostics:

- YOLO/table bbox: `table_bbox_xyxy`, `rgb_shape`, `yolo_bbox_center_x_norm`,
  `table_bbox_control_valid`, `yolo_table_control_valid`,
  `yolo_table_visible`, `yolo_table_fresh`, `table_bbox_touch_left`,
  `table_bbox_touch_right`, `table_bbox_touch_bottom`.
- Edge: `edge_found`, `edge_valid`, `edge_trusted`, `edge_confidence`
  or `edge_conf`, `yaw_err_rad`, `dist_err_m`.
- Dynamic ROI depth: `table_roi_depth_valid`, `table_roi_depth_p10`,
  `table_roi_depth_median`, `table_roi_depth_mean`,
  `table_roi_depth_sample_count`, `table_roi_depth_valid_ratio`.

If `yolo_bbox_center_x_norm` is missing, Orchestrator may still compute bbox
center from `table_bbox_xyxy + rgb_shape`. A valid VISTA bbox must not become
invalid merely because one of the alias fields is absent.

## Control Ownership

Table docking state code owns state transitions and candidate intent. The final
`vx/vy/wz`, STOP class, and service override permission are decided by
`runtime/motion_arbiter.py`.

Layer responsibilities:

- State/controller: produce `MotionIntent` candidates and diagnostic summary.
- Motion arbiter: produce `final_vx`, `final_vy`, `final_wz`, `motion_class`,
  `stop_class`, `arbitration_reason`.
- Service: hard emergency/safety override, enqueue/write metadata, and
  `effective_cmd_before_service` / `effective_cmd_after_service` logging.
- UART bridge: latest-command enqueue, writer acceptance/discard reason, and
  actual `serial_write_ok` / `uart_tx_ok`.

## STOP Classes

- `emergency`: explicit STOP, E-stop, obstacle, hard emergency.
- `safety`: final depth stop, hard safety stop, final locked.
- `control_recovery`: local control recovery such as bbox lost hold expiry.
- `stale_recovery`: stale/dropout expiry recovery.
- `none`: normal motion, search rotate, final yaw align.

Only emergency and necessary safety stops should apply E-stop cooldown or writer
velocity suppression. Search rotate and normal recovery velocity commands must
remain writer-acceptable.

## Near/Final Latches

`near_table_latched` is refreshed on the main table docking tick before final
arbitration. It can be triggered by edge-guided approach with valid depth,
near dynamic ROI depth, final/depth-stop evidence, or large/touching bbox with
valid edge/depth.

`final_depth_latched` has higher priority than no-bbox search and forward coast.
After it latches, `vx` must remain zero. If edge yaw or last-good edge yaw is
available and above deadband, final yaw align may rotate in place; otherwise the
robot holds final position and must not return to `SEARCH_TABLE` because YOLO
bbox is lost.

## Anti-Stall Invariants

- Active table docking cannot emit long-lived `vx=0,wz=0` unless final stop,
  emergency, obstacle, or hard safety is active.
- Near/final latch suppresses bbox-lost fallback to `SEARCH_TABLE`.
- Final depth latch outranks forward coast.
- `SEARCH_TABLE` rotate velocity is writer-accepted outside emergency/safety
  cooldown.
- `EDGE_GUIDED_APPROACH + soft FOV + usable edge` keeps forward motion.
- Short table-edge dropout after healthy last-good observation holds motion and
  does not immediately STOP.

## Real-Robot Log Checklist

For tomorrow's table docking run, watch:

- Input contract: `obs_parse_missing_fields`, `table_bbox_control_valid`,
  `yolo_bbox_center_x_norm`, `edge_found`, `edge_trusted`, `yaw_err_rad`,
  `table_roi_depth_valid`, `table_roi_depth_p10`, `table_roi_depth_mean`.
- Arbiter: `motion_intent_type`, `yaw_owner`, `arbitration_reason`,
  `motion_class`, `stop_class`, `blocked_by`, `final_vx`, `final_wz`.
- Latches: `near_table_latched`, `near_table_latch_reason`,
  `final_depth_latched`, `final_depth_latch_reason`,
  `final_yaw_align_active`, `final_locked`.
- Service/UART: `service_override`, `service_override_reason`,
  `effective_cmd_before_service`, `effective_cmd_after_service`,
  `uart_enqueue_ok`, `writer_accept_cmd`, `writer_discard_reason`,
  `serial_write_attempted`, `serial_write_ok`, `uart_tx_ok`.
