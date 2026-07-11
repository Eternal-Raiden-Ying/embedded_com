# 2026-07-03 Final Handoff And Logging Fix

## Pre-change Version

- Branch: `feature/docking-control-v3`
- HEAD: `1f153ce`
- Last commit: `1f153ce bug fix`
- Git status summary:

```text
 M VISTA/vision_module/app/app.py
 M VISTA/vision_module/backend/camera_manager.py
 M VISTA/vision_module/backend/remote/manager.py
 M VISTA/vision_module/config/schema.py
 M VISTA/vision_module/test/demo_camera.py
 M common/config/schema.py
 M configs/system_config.yaml
 M orchestrator/orchestrator_service/config/schema.py
 M orchestrator/orchestrator_service/control/velocity_smoother.py
 M orchestrator/orchestrator_service/mobile_gateway/runtime/service.py
 M orchestrator/orchestrator_service/runtime/context.py
 M orchestrator/orchestrator_service/runtime/core.py
 M orchestrator/orchestrator_service/runtime/core_types.py
 M orchestrator/orchestrator_service/runtime/export_state.py
 M orchestrator/orchestrator_service/runtime/motion_arbiter.py
 M orchestrator/orchestrator_service/runtime/service.py
 M orchestrator/orchestrator_service/runtime/states/grasp_flow.py
 M orchestrator/orchestrator_service/runtime/states/table_docking.py
 M orchestrator/orchestrator_service/runtime/states/target_search.py
 M orchestrator/orchestrator_service/runtime/task_runtime.py
 M orchestrator/orchestrator_service/runtime/vision_sync.py
 M orchestrator/orchestrator_service/utils/target_utils.py
?? orchestrator/orchestrator_service/runtime/states/return_place.py
```

## Bug Definition

1. Vision can create log files that remain empty, making field diagnosis difficult.
2. When close-range YOLO table / bbox / depth labels are lost, the state machine may still return to `SEARCH_TABLE` / rotate search instead of handing off to Final.
3. In `FINAL_SLOW_STOP`, fixed ROI median can reach the final stop threshold with sufficient stable count, but the state machine does not immediately enter `AT_TABLE_EDGE`.
4. Normal safety / hold / `final_depth_latched` branches can have too much priority and swallow the fixed ROI arrival transition.
5. Logs contain old paths, legacy fields, stale RGB archive names, or misleading names that make diagnosis harder.

## Fix Goals

- Make Vision log startup/shutdown writes reliable and non-empty.
- Ensure close-range YOLO loss prefers handoff to `FINAL_SLOW_STOP` over rotating search.
- Make fixed ROI median arrival the canonical final stop path.
- Keep normal safety hold from blocking arrival transition; only emergency depth can block it.
- Keep final fixed ROI threshold, slow threshold, hard hold threshold, and emergency threshold clearly separated.

## Pre-change Key Config

- `final_enter_depth_threshold_m`: `0.45`
- `final_fixed_roi_stop_threshold_m`: `0.45`
- `depth_envelope_slow_p10_m`: `0.50`
- hard hold equivalent `depth_envelope_stop_p10_m`: `0.30`
- emergency equivalent `depth_emergency_stop_p10_m`: `0.20`
- final slow probe equivalent `final_probe_vx_mps`: `0.015`
- `final_slow_stop_timeout_s`: not found in initial grep output
- `remote_rgb_capture_warmup_frames`: `10`
- `remote_rgb_capture_wait_timeout_s`: `2.0`
- `remote_rgb_min_luma_mean`: `40.0`

## Post-change Key Config

- `final_enter_depth_threshold_m`: `0.58`
- `final_enter_stable_count_required`: `2`
- `final_handoff_on_yolo_lost_enable`: `true`
- `final_handoff_recent_obs_max_age_s`: `1.0`
- `final_handoff_min_recent_depth_m`: `0.65`
- `final_fixed_roi_stop_threshold_m`: `0.45`
- `final_fixed_roi_stop_stable_count_required`: `3`
- `depth_envelope_slow_p10_m`: `0.50`
- hard hold equivalent `depth_envelope_stop_p10_m`: `0.30`
- emergency equivalent `depth_emergency_stop_p10_m`: `0.20`
- final slow probe equivalent `final_probe_vx_mps`: `0.020`
- `final_missing_probe_grace_s`: `2.0`
- `final_slow_stop_timeout_s`: `12.0`
- `remote_rgb_capture_warmup_frames`: `10`
- `remote_rgb_capture_wait_timeout_s`: `2.0`
- `remote_rgb_min_luma_mean`: `40.0`

## Validation

- `python3 -m py_compile` passed for modified Python files in this round.
- Scoped `git diff --check` passed for files touched in this round.
- Full `git diff --check` is blocked by pre-existing whitespace changes in `VISTA/vision_module/test/demo_camera.py`.
- `rg` confirmed Vision lifecycle logs: `[VISION][LOG_START]`, `[VISION][LOG_STOP]`, and `log_health.json`.
- `rg` confirmed final handoff / arrival logs and fields: `[FINAL][HANDOFF_ON_YOLO_LOST]`, `[FINAL][VISION_REQ]`, `[FINAL][ARRIVAL_REACHED]`, `final_arrival_reached`, `final_arrival_latched`, and `safety_blocks_arrival_transition`.
- `rg` confirmed converged thresholds: `final_enter_depth_threshold_m: 0.58`, `depth_envelope_stop_p10_m: 0.30`, and `depth_emergency_stop_p10_m: 0.20`.
- `rg` confirmed `final_stop_threshold_m` is no longer used as the fixed ROI final stop field in the touched Final control path.
- `rg` confirmed remote metadata now reports `depth_aligned_to_color: not_aligned` and the remote manager does not write `rgb_raw.jpg` / `rgb.jpg` archive paths.
