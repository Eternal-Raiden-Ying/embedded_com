# VISTA Markdown Drift Audit

## Purpose

This document records the current documentation alignment state inside `VISTA/`.

It is an architecture audit artifact, not a coding backlog.

## Current Status

The high-priority markdown drift identified in the earlier audit pass has now been addressed.

The following files were refreshed and are now the current baseline documents:

1. `VISTA/ReadMe.md`
2. `VISTA/VISION_ENGINE_TODO.md`
3. `VISTA/vision_module/backend/camera/README.md`
4. `VISTA/INTERFACES.md`
5. `VISTA/ARCHITECTURE.md`
6. `VISTA/PRODUCT_REQUIREMENTS.md`

No root contract-bearing markdown file is currently classified as clearly stale.

## Classification

### Current baseline docs, aligned with the present architecture direction

1. `VISTA/ReadMe.md`
2. `VISTA/INTERFACES.md`
3. `VISTA/ARCHITECTURE.md`
4. `VISTA/PRODUCT_REQUIREMENTS.md`
5. `VISTA/vision_module/backend/camera/README.md`

### Special-purpose docs, aligned with their intended role

1. `VISTA/VISION_ENGINE_TODO.md`
   Current role: historical note plus remaining engine-level backlog, not active architecture truth.
2. `VISTA/AUDIT_TODO.md`
   Current role: coding handoff backlog for detect/remote stabilization work.
3. `VISTA/MD_DRIFT_AUDIT.md`
   Current role: documentation alignment record.

### Largely aligned, low urgency

1. `VISTA/EVENT_DESCRIPTION.md`
2. `VISTA/LOGGING_RULES.md`
3. `VISTA/vision_module/backend/camera/camera_info.md`

## File-by-File Status

### 1. `VISTA/ReadMe.md`

Current state: aligned.

What was corrected:

- removed dead references such as `new_engine.py` and `test_grasp_only.py`
- switched the documented local baseline from segmentation-first wording to the actual detect-first baseline
- removed `DONE` from the stable current `vision_obs.status` baseline
- separated current runtime modes from future extension ideas
- added an explicit `Known Contract Gaps` section

Residual risk:

- if detect class-vocabulary ownership or remote init strategy changes in code, the README must be updated in the same change set

### 2. `VISTA/VISION_ENGINE_TODO.md`

Current state: aligned to its new role.

What was corrected:

- stopped presenting already-landed structural work as future work
- reclassified the file as historical context plus remaining engine-level backlog

Residual risk:

- if contributors start treating it as active architecture truth again, drift will reappear immediately

### 3. `VISTA/vision_module/backend/camera/README.md`

Current state: aligned.

What was corrected:

- replaced the obsolete `aidlux_cam/csrc` build path with the current repo path
- rewrote the file as a VISTA camera-backend note instead of a detached standalone component README

Residual risk:

- if camera profile ownership moves substantially into mode data, the note should be refreshed again

### 4. `VISTA/INTERFACES.md`

Current state: aligned with the present contract direction.

What was corrected:

- documented `GRASP.payload.class_id` as the current remote target field
- clarified that `class_id` should come from external input
- clarified that `target` is business semantics, not the remote `class_id` surrogate
- documented which remote payload fields are stable direction versus higher-order overrides

Residual risk:

- the implementation still contains internal fallback logic from `target -> class_id`
- if that fallback is removed or the request shape changes, `INTERFACES.md` must be updated together with code

### 5. `VISTA/ARCHITECTURE.md`

Current state: aligned.

What was corrected:

- preserved the real current topology
- added explicit `known contract gaps`
- clarified that `remote_ack` is not the primary business truth route
- clarified ownership boundaries between stage, mode/profile, predictor/model contract, scheduler, and remote manager

Residual risk:

- if `release_cooldown_s`, `remote_ack`, or remote gating behavior changes materially, this file must move with that change

### 6. `VISTA/PRODUCT_REQUIREMENTS.md`

Current state: aligned with still-valid product intent.

What was corrected:

- removed stale segment-only naming assumptions
- locked in `coco80 detect` as a supported baseline requirement
- documented remote `class_id` direction, configurable upload encoding, and service/init gating requirements
- preserved valid but not-yet-fully-landed requirements instead of deleting them as if they were obsolete

Residual risk:

- if remote capture contract, upload encoding ownership, or detect class semantics change, this file must be revised in lockstep

### 7. `VISTA/EVENT_DESCRIPTION.md`

Current state: largely aligned.

Residual risk:

- if `remote_ack` is removed or demoted further, event descriptions may need cleanup

### 8. `VISTA/LOGGING_RULES.md`

Current state: largely aligned.

Residual risk:

- if runtime event families change because of remote-flow simplification, logging guidance should be refreshed

### 9. `VISTA/vision_module/backend/camera/camera_info.md`

Current state: acceptable as a hardware reference note.

Residual risk:

- it is not an architecture baseline and should not be used as one

## Remaining Documentation Risks

The main documentation risk is no longer stale root docs. It is future contract work landing in code without synchronized doc updates.

The next high-probability drift triggers are:

1. detect real-output contract changes
2. detect class-vocabulary ownership changes
3. remote `INIT -> PREDICT` gating changes
4. remote fresh-frame barrier implementation
5. remote camera/profile ownership changes
6. upload encoding ownership changes
7. `release_cooldown_s` becoming real behavior or being deleted

## Rule Going Forward

For VISTA, the contract-bearing docs should now be treated as part of the architecture surface:

- `ReadMe.md`
- `INTERFACES.md`
- `ARCHITECTURE.md`
- `PRODUCT_REQUIREMENTS.md`

Any change to detect or remote contracts should update the affected markdown files in the same pull request or commit series.
