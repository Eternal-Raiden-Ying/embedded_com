# Docking Field Cleanup Audit

Legacy override files were removed in V3 round 2. Current canonical sources are:

| Removed field/path | Replacement |
| --- | --- |
| `orchestrator/configs/stage_params.yaml` | `configs/system_config.yaml: orchestrator.control.*`, `orchestrator.car.*`, `orchestrator.docking.*` |
| `orchestrator/configs/car_cmd_params.yaml` | `configs/system_config.yaml: orchestrator.car.*` |
| VISTA `detector_mode=full` / `detector_mode=lightweight` | `detector_mode=fast_plane_only` only |
| `light_stride` | removed with lightweight detector |

Close-range depth/P10 behavior remains under the canonical `orchestrator.control.*` fields, including `roi_final_stop_p10_m`, `depth_envelope_stop_p10_m`, `close_range_probe_vx_mps`, `roi_final_probe_vx_mps`, `final_forward_vx_max_mps`, `close_range_missing_probe_vx_mps`, and `roi_final_missing_probe_vx_mps`.
