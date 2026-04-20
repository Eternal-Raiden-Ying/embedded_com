# VISTA

**Vision Intelligent Search & Tracking Assistant**

VISTA is the vision service in this robot stack. It runs on the edge device, accepts `vision_req`, produces `vision_obs`, and switches runtime resources according to the current business stage.

This file is the current-state baseline for operators and developers.

For other viewpoints, use:

- `ARCHITECTURE.md`: current internal topology
- `INTERFACES.md`: external IPC contract
- `AUDIT_TODO.md`: coding handoff backlog
- `MD_DRIFT_AUDIT.md`: markdown drift audit

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

### Camera

Current default board config:

- `rgb`: input `1280x720`, cropped and output as `640x640`, format `RGB`
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

### Remote

Current remote path is implemented through:

- `vision_module/backend/remote/client.py`
- `vision_module/backend/remote/manager.py`
- `vision_module/app/stages/grasp.py`
- `grasp_module/simulate_client_request.py` as the minimal reference script

Current design intent is remote grasp by `class_id`.

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
├── AUDIT_TODO.md
├── MD_DRIFT_AUDIT.md
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
- `vision_stream.py`

The old references to `new_engine.py` and `test_grasp_only.py` are obsolete and should not be used.

## Known Contract Gaps

This section is intentionally explicit. The current structure is real, but some contracts are not settled yet.

### Detect line

- default local model is `coco80 detect`, but stage-side class resolution is not yet cleanly aligned with that baseline
- real predictor output handling needs a safer contract at the predictor-manager boundary
- current backend import path can still hide real-path problems by falling back to `mock` in some cases

### Remote line

- `GRASP_REMOTE` still needs a proven fresh-frame barrier before `PREDICT`
- server-side `INIT` completion is not yet a clearly enforced gate before `PREDICT`
- remote upload encoding should become configurable, not fixed
- remote camera parameters should move into mode/profile ownership
- segmentation-related remote surface is still present and should be deleted after parity is confirmed

### Runtime policy

- `release_cooldown_s` exists in mode profiles, but delayed release is not yet a meaningful runtime behavior

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
