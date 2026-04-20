# VISTA Vision Engine Todo

## Status

This file is no longer the active architecture baseline.

The large structural refactor it originally described has already landed in the codebase:

- `VistaApp`
- `StageController`
- `ModeController`
- `VisionEngine`
- `RuntimeSupervisor`
- `Scheduler`
- manager-owned worker loops

Use these files as the active current-state references instead:

- `ReadMe.md`
- `ARCHITECTURE.md`
- `INTERFACES.md`

Use these files as the active follow-up backlog references instead:

- `AUDIT_TODO.md`
- `MD_DRIFT_AUDIT.md`

## What Has Already Been Achieved

The codebase has already moved past the following historical goals:

- external protocol is centered on `vision_req` / `vision_obs`
- business workflow is stage-driven
- runtime resource ownership is mode-driven
- camera / predictor / remote / preview are split into managers
- high-frequency work is off the app main loop
- stage logic consumes summarized scheduler outputs rather than raw frames

That means this file should not continue to describe those items as future work.

## Remaining Engine-Level Work

The remaining work is not the original refactor anymore. It is contract stabilization and entropy reduction.

### 1. Detect contract stabilization

- make the real detect predictor output contract safe at the manager boundary
- keep default `coco80 detect` viable in app flow
- remove hidden real-to-mock downgrade behavior for non-mock runs
- decide whether detect postprocess should match the old `tmp/` baseline or remain predictor-local with explicit documentation

### 2. Remote parity and gating

- verify integrated remote flow against `grasp_module/simulate_client_request.py`
- ensure `/init` completion is confirmed before `/predict`
- add a fresh-frame barrier after `GRASP_REMOTE` mode switch before remote predict is triggered
- decide whether remote `/init` should move to service startup

### 3. Profile ownership cleanup

- move remote-relevant camera parameters into mode/profile data
- make upload encoding configurable, at least `png/jpeg`
- keep only the remote profile fields that survive the parity audit

### 4. Runtime policy cleanup

- either implement meaningful delayed release semantics for `release_cooldown_s`
- or stop pretending cooldown is an active runtime policy

### 5. Entropy reduction

- remove remote segmentation branch after parity confidence is established
- remove dead or ceremonial remote profile fields
- reduce stale documentation branches that still point to removed experiments

## Historical Note

The older version of this file was useful during the refactor from a single engine-heavy design toward the current stage/mode/manager split.

At this point, its value is historical context only.

Any new work should be tracked as:

- current architecture statement in `ARCHITECTURE.md`
- concrete coding handoff in `AUDIT_TODO.md`
- documentation drift in `MD_DRIFT_AUDIT.md`
