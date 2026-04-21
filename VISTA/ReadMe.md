# VISTA

**Vision Intelligent Search & Tracking Assistant**

VISTA is the vision service in this robot stack. It runs on the edge device, accepts `vision_req`, produces `vision_obs`, and switches runtime resources according to the current business stage.

This file is the current-state baseline for operators and developers.

For other viewpoints, use:

- `ARCHITECTURE.md`: current internal topology
- `INTERFACES.md`: external IPC contract
- `IMPLEMENTATION_STATUS.md`: master plan and completion state
- `NEXT_TODO.md`: current next-round action list

## Current Scope

VISTA currently focuses on three business stages:

- `SEARCH`: local target search and tracking
- `GRASP`: micro-adjust interaction plus remote grasp cooperation
- `RETURN`: return-target / home-tag observation

Current runtime design is stage-driven, not a single monolithic inference loop.

## Current Runtime Baseline

```text
Orchestrator --vision_req--> VistaApp
                              |
                              v
                        StageController
                         |           \
                         |            \
                         v             v
                    StagePlan      ModeController
                         |             |
                         +-------> VisionEngine.apply_mode_plan(...)
                                       |
                                       v
                               RuntimeSupervisor
                             /    |      |      \
                            v     v      v       v
                   CameraManager Predictor Remote Preview
                            \      |      /        |
                             +-----+-----+---------+
                                           |
                                           v
                                      Scheduler
                                           |
                                           v
                                 VistaApp --vision_obs--> Orchestrator
```

Current responsibility split:

- `VistaApp`: service lifecycle, IPC, main loop, logging, heartbeat
- `StageController`: request handling, stage state, stage output, mode requests
- `ModeController`: mode profile registration, switch state, runtime plan compilation
- `VisionEngine`: runtime assembly root and backend facade
- `RuntimeSupervisor`: capability reconcile for camera / predictor / remote / preview
- `Scheduler`: route bus between managers and stage logic
- managers: own worker loops and capability-specific state

## Current Stage And Mode Baseline

Current default stage-entry modes:

| Stage | Default Mode | Current role |
| --- | --- | --- |
| `IDLE` | `IDLE` | cold idle |
| `SEARCH` | `TRACK_LOCAL` | local RGB tracking and target observation |
| `GRASP` | `MICRO_ADJUST` | micro-adjust interaction before remote grasp |
| `RETURN` | `TRACK_LOCAL` | local return-target observation |

Current registered default modes:

| Mode | Current role |
| --- | --- |
| `IDLE` | no active runtime capability |
| `TRACK_LOCAL` | local RGB + local predictor |
| `MICRO_ADJUST` | local RGB + local predictor, used by `GRASP` workflow |
| `GRASP_REMOTE` | RGB + depth + remote grasp path |
| `IDLE_HOT` | hot standby after stop |

Important distinction:

- active baseline modes are the five above
- future mode ideas such as `DEPTH_PERCEPTION` are not current runtime truth

## Current Backend Baseline

Backend selection ownership:

- `VISTA_BACKEND=mock|real|auto` is now the runtime source of truth for camera and predictor backend selection
- `capability_placeholder` may still exist in config for test scaffolding, but it is no longer allowed to decide the main runtime real/mock path

### Camera

Current default board config:

- `rgb`: input `1280x720`, cropped and output as `640x640`, format `BGR`
- `depth`: `424x240 @ 15 fps`
- `grey`: available as a separate stream

These defaults currently live in `vision_module/config/board_config.py`.

### Predictor

Current default active model:

- `yolov7_detect`

Current built-in model profiles:

- `yolov7_detect`: default local detect baseline, `coco80`
- `yolov8s_seg`: optional segmentation profile
- `yolo26s_seg`: optional segmentation profile using `grasping_coco20`

Important note:

- current local baseline is detect-first, not segmentation-first
- current camera color baseline is BGR; detect follows that baseline directly, while the optional segment predictor adapts internally
- stage-side detect decoding now follows the active model profile `classes`, not one global hardcoded table
- `PredictorManager` publishes detect boxes as flattened `infer_boxes = [[x1, y1, x2, y2, score, class_id], ...]`
- `local_perception.class_names` is the current source of truth for stage-side detect class decoding
- if a detect profile omits `classes`, `PredictorManager` now publishes fallback `coco80` class names explicitly and marks the payload as weakened via `class_names_source=fallback_coco80`
- malformed detect rows are now surfaced at the manager boundary through `local_perception.contract_ok`, `contract_error`, and `contract_warnings`

### Remote

Current remote path is implemented through:

- `vision_module/backend/remote/client.py`
- `vision_module/backend/remote/manager.py`
- `vision_module/app/stages/grasp.py`
- `grasp_module/simulate_client_request.py` as the minimal reference script

Current design intent is remote grasp by `class_id`.

Current remote execution baseline:

- `RemoteManager` owns service-scoped `/init` and attempts one best-effort init when a usable `base_url` exists
- `GRASP` gates `PREDICT` on service init confirmation plus fresh `GRASP_REMOTE` frames, and may retry init up to 3 times
- `RemoteManager` rejects `PREDICT` if service init is not confirmed
- `class_id` is now treated as explicit request truth and is no longer synthesized from `target`
- `GRASP_REMOTE` camera parameters and upload encoding defaults now come from mode/profile data
- `/release` is no longer a default per-grasp action; it is used on shutdown, remote disable, or explicit reset

## Current IPC Baseline

Default transport:

- request inbound: `127.0.0.1:9003`
- observation outbound: `127.0.0.1:9002`

Current request contract:

- inbound message: `vision_req`
- supported `op`: `START`, `UPDATE`, `RESPOND`, `STOP`
- supported `stage`: `SEARCH`, `GRASP`, `RETURN`, `IDLE`

Current observation contract:

- outbound message: `vision_obs`
- stable current `status` values:
  - `RUNNING`
  - `WAITING_RESPONSE`
  - `RESULT_READY`
  - `FAILED`

`DONE` is not a stable current output state and should not be used as the contract baseline.

For full field definitions, use `INTERFACES.md`.

## Current Repo Layout

```text
VISTA/
├── ARCHITECTURE.md
├── INTERFACES.md
├── PRODUCT_REQUIREMENTS.md
├── IMPLEMENTATION_STATUS.md
├── NEXT_TODO.md
├── grasp_module/
│   └── simulate_client_request.py
└── vision_module/
    ├── app/
    │   └── app.py
    ├── backend/
    │   ├── vision_engine.py
    │   ├── runtime_supervisor.py
    │   ├── scheduler.py
    │   ├── camera_manager.py
    │   ├── predictor_manager.py
    │   ├── mode_controller.py
    │   ├── remote/
    │   └── preview/
    ├── config/
    │   ├── board_config.py
    │   └── mode_defaults.py
    ├── ipc/
    ├── model/
    └── test/
```

## Current Debug And Validation Tools

Current useful tools in `vision_module/test/` include:

- `debug_send_req.py`
- `debug_recv_obj.py`
- `debug_protocol_tools.py`
- `demo_camera.py`
- `test_sensors.py`
- `test_predictor.py`
- `test_pipeline.py`
- `test_runtime_architecture.py`
- `test_color_controls.py`

The old references to `new_engine.py` and `test_grasp_only.py` are obsolete and should not be used.

Current mode-level RGB capture baselines are no longer identical copies of board config:

- `TRACK_LOCAL`: BGR, `1280x720 -> 640x640`, `24 fps`, wide center crop
- `MICRO_ADJUST`: BGR, `1280x720 -> 640x640`, `30 fps`, tighter center crop
- `GRASP_REMOTE`: BGR, `1280x720 -> 640x640`, `15 fps`, remote-oriented stable capture profile

## Known Contract Gaps

This section is intentionally explicit. The current structure is real, but some contracts are not settled yet.

### Detect line

- `RETURN` is now backed by the default detect line; outward compatibility remains `perception.home_tag_obs`, but the payload may be detect-backed with `source=detect`
- `auto` backend may still resolve to `mock`, but the resolution is now explicit and no longer hidden inside the runtime manager path
- the default service color contract is already frozen to `BGR`; any later real-device validation is accuracy validation, not contract discovery

### Remote line

- segmentation-related remote surface has been deleted from the integrated path
- request-level `base_url` override is no longer part of the remote request contract; endpoint ownership now stays with mode/profile/runtime
- remote profile fields are intentionally trimmed to runtime capability defaults plus limited debug/test overrides

### Runtime policy

- `release_cooldown_s` exists in mode profiles, but delayed release is not yet a meaningful runtime behavior
- frame-consuming managers now gate on `(generation, seq)` rather than raw `seq` only, so mode switches no longer stall predictor or freeze preview after scheduler slot reset

## Run

Target device runtime:

```bash
cd /home/aidlux/2026/VISTA
/usr/bin/python3 -m vision_module.app.app
```

## Local Development Note

This workspace may be opened on Windows, but the real runtime target is AidLux / QCS6490.

Important implications:

- real camera and QNN model execution are target-device concerns
- missing model files on the Windows workspace is expected
- local Windows work is mainly for protocol, architecture, and mock-path validation unless explicitly configured otherwise

## Documentation Rule

When code and docs disagree, the current implementation plus `ARCHITECTURE.md` and `INTERFACES.md` should be treated as the nearer source of truth than historical planning notes.
