# VISTA Implementation Status

## Purpose

This is the master plan/status file for VISTA follow-up work.

Use it to answer, in each later conversation:

- what has already been completed
- what is only partially completed
- what is still pending

The live short-horizon action list now lives in `VISTA/NEXT_TODO.md`.

Status labels:

- `DONE`: landed in code and covered by the current architecture direction
- `PARTIAL`: direction is implemented, but contract cleanup or subtraction is still unfinished
- `PENDING`: not landed yet

Last reviewed: 2026-04-21

Verification note:

- This review is based on current code inspection.
- Mock-capable test modules exist.
- The current implementation test gate has been executed on the mock-capable path using `.venv\Scripts\python.exe`.

Consolidation note:

- older overlapping planning files have been folded into this file plus `VISTA/NEXT_TODO.md`

## Detect Line

### D1. Real predictor output contract at manager boundary

Status: `DONE`

Evidence:

- `vision_module/backend/predictor_manager.py`
  - normalizes NumPy outputs into Python payloads
  - validates detect rows
  - publishes explicit contract fields

Notes:

- detect rows are now validated against `xyxy_score_class_id`
- contract degradation is surfaced through `contract_ok`, `contract_error`, and `contract_warnings`

### D2. Default `coco80` detect vs stage-side class decoding

Status: `DONE`

Evidence:

- manager publishes `class_names`
- stage-side detect decoding prefers model/profile class names
- direct detect fallback now resolves to normalized `coco80`
- weakened fallback is surfaced via `class_names_source=fallback_coco80`

### D3. tmp detect parity vs backend detect contract

Status: `DONE`

What is done:

- backend detect output is explicitly frozen as `xyxy_score_class_id`
- manager publishes `infer_box_format`

What is still worth monitoring:

- color-order parity with `tmp/` is still a separate validation question, not a hidden contract anymore

### D4. Silent real-to-mock fallback

Status: `DONE`

Evidence:

- manager now reads backend resolution from package-level backend status
- runtime camera/predictor class selection comes from package-level selectors
- `capability_placeholder` is retained only as explicit test/mock scaffolding metadata and no longer chooses the main runtime path

### D5. `RETURN` backed by default detect line

Status: `DONE`

Notes:

- `RETURN` now consumes `local_perception.infer_boxes`, `class_names`, `rgb_shape`, `contract_ok`, `contract_error`, and `contract_warnings`
- outward compatibility remains `perception.home_tag_obs`, but the payload may now be detect-backed with `source=detect`
- `vision_req.target` is now the authoritative detect-backed return target

## Remote Line

### R1. `INIT -> wait -> PREDICT -> RELEASE` parity with minimal script

Status: `DONE`

What is done:

- remote `/init` is now service-scoped and owned by `RemoteManager`
- service startup performs a best-effort init attempt when a usable `base_url` exists
- `GraspStagePlan.tick()` gates `PREDICT` on service init confirmation plus fresh-frame readiness
- `RemoteManager` rejects `PREDICT` with `init_not_confirmed` if service init has not been confirmed
- tests exist in:
  - `vision_module/test/test_runtime_architecture.py`
  - `vision_module/test/test_remote_contract.py`

What is still open:

- subtractive field cleanup and minimal-script sync remain separate follow-up work

### R2. Fresh-frame barrier before remote predict

Status: `DONE`

Evidence:

- `vision_module/app/stages/grasp.py`
  - tracks `remote_init_confirmed`
  - waits for `frame_meta.has_frames`
  - requires the needed camera set before emitting `PREDICT`
- covered by `test_grasp_stage_waits_for_init_and_fresh_frames_before_predict`

### R3. Camera parameters in mode/profile ownership

Status: `DONE`

Evidence:

- `ModeProfile.camera_overrides` exists
- `mode_defaults.py` builds per-mode `camera_overrides`
- `RuntimeSupervisor._configure_camera()` passes overrides into `CameraManager.ensure_camera()`
- `app.py` now passes `CONFIG` into `build_default_mode_profiles(...)`
- `TRACK_LOCAL`, `MICRO_ADJUST`, `GRASP_REMOTE`, and `IDLE_HOT` now carry explicit RGB camera profiles with mode-specific format / fps / crop values

### R4. Configurable upload encoding

Status: `DONE`

Evidence:

- `RemoteProfile` now carries `rgb_encoding`, `depth_encoding`, `rgb_quality`, `depth_compression`
- `RemoteManager` uses these values when encoding outgoing payloads
- covered by `test_remote_contract.py`

### R5. Remove segmentation branch

Status: `DONE`

What is done:

- current stage path is now `class_id`-driven
- explicit `class_id` is required in the real `GRASP` accept path
- segmentation-related contract surface has been removed from:
  - `RemoteProfile`
  - `RemotePredictRequest`
  - remote multipart builder
  - mode compile / runtime profile payloads
  - the minimal remote script

### R6. Remote profile field reduction after parity audit

Status: `DONE`

What is done:

- remote profile now carries real operational fields for timeout and image encodings
- request-level `base_url` override has been removed to avoid split-brain with service-scoped init
- runtime/profile upload defaults remain owned by `RemoteProfile`
- `command` is still retained as a compatibility field, while `timeout_s` and `metadata` remain bounded debug/test overrides

### R7. `class_id` external-input ownership only

Status: `DONE`

Evidence:

- `GraspStagePlan.on_respond()` now fails with `missing_class_id` if explicit input is absent
- `RemoteManager` no longer infers `class_id` from `target`
- covered by `test_remote_contract.py`

## Runtime / Infra Follow-Ups

### B1. Backend selection owned by `VISTA_BACKEND`

Status: `DONE`

Evidence:

- package `camera/__init__.py` and `predictor/__init__.py` resolve real/mock through `VISTA_BACKEND`
- `CameraManager` and `PredictorManager` now use package-level resolved classes instead of letting `capability_placeholder` choose the runtime path
- `board_config.py` now documents `capability_placeholder` as scaffolding only, not as the main backend selector

### B2. Default detect fallback should be `coco80`

Status: `DONE`

Evidence:

- `compute_target_obs()` now falls back to `COCO80_CLASSES`
- manager exposes `class_names_source=fallback_coco80` when the payload is weakened

### B3. Camera color contract aligned to the chosen `tmp` parity direction

Status: `DONE`

Evidence:

- default color camera baseline is now `BGR`
- detect predictor contract is now explicitly `BGR`
- optional segment predictor performs `BGR -> RGB` internally
- mode profiles carry explicit BGR camera settings for `TRACK_LOCAL`, `MICRO_ADJUST`, `GRASP_REMOTE`, and `IDLE_HOT`

Note:

- real-device empirical validation is still a separate execution concern, but the architecture and code contract are now aligned to the chosen BGR baseline

### B4. Preview / debug alignment with the BGR baseline

Status: `DONE`

Evidence:

- unnecessary `RGB -> BGR` conversion paths have been removed from current debug tooling
- `demo_camera.py` and `test_color_controls.py` now operate on BGR camera output directly
- preview manager forwards the current frame as-is instead of assuming an RGB-only contract

### B5. Preview / predictor generation-reset handling

Status: `DONE`

Evidence:

- predictor and preview now reset sequence tracking when generation changes
- regression tests exist in `test_runtime_architecture.py`

### B6. Old predictor alias cleanup

Status: `DONE`

Evidence:

- `QNNPredictor` has been removed from the supported predictor export surface
- old alias exports in predictor modules have been removed
- `vision_module/test/vision_stream.py` has been deleted
- no executable `QNNPredictor` references remain in `vision_module/**/*.py`

### B7. Regression tests for the recent contract changes

Status: `DONE`

What is done:

- test coverage now exists for:
  - backend selection contract
  - `coco80` fallback behavior
  - malformed detect row contract degradation
  - generation-aware predictor/preview resume
  - remote init gating
  - remote explicit `class_id`
- segmentation-surface removal from remote multipart/protocol
- request-level `base_url` override removal
- mock pipeline regression

### B8. Documentation sync for the recent contract changes

Status: `DONE`

What is done:

- root architecture/interface/product docs have been updated to describe backend ownership, detect contract, generation-aware gating, camera profile ownership, and the BGR baseline
- current-state docs now describe service-scoped remote init, detect-backed `RETURN`, segmentation-surface removal, and the removal of request-level `base_url` override

Notes:

- keep watching for doc drift as the minimal remote script and any future real-device notes evolve

## Current Action File

See `VISTA/NEXT_TODO.md` for the current short-horizon task list.

## If Real-Device Evidence Is Needed Later

No extra log package is required for the current architectural next-step analysis.

If later validation is needed for remote init strategy or remote upload encoding, the most useful artifacts would be:

- VISTA `event.jsonl`
- VISTA `ipc.jsonl`
- remote server access/application logs with timestamps
- one full `request_id` trace across `/init`, `/predict`, and `/release`
- sample encoded RGB/depth payload sizes for both `png` and `jpeg`

## Update Rule

When code changes land, update this file in the same change series if the change affects:

- detect contract state
- remote flow state
- backend ownership
- camera/profile ownership
- old alias cleanup
