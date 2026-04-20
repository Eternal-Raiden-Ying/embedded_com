# VISTA Audit Todo

## Purpose

This document is a handoff backlog for follow-up coding work.

It is not a rewrite spec and it is not a final architecture document.
Its purpose is to record:

- confirmed problems
- required direction changes
- open architectural questions
- a recommended execution order for a coding-focused agent

## Scope

Current focus:

- `Detect` line parity and app integration
- `REMOTE` line parity, control flow, and entropy reduction
- selective `.md` refresh after behavior and contracts are settled

Out of scope for now:

- `Voice`
- `orchestrator`
- broad refactors unrelated to the above two lines

## Locked Decisions

These points are already decided and should be treated as requirements unless changed explicitly later.

### Detect line

- `coco80 detect` must remain supported.
- Future migration to `grasping_coco20` is only a possibility, not the current baseline.
- Real predictor import failure should not silently fall back to `mock` in non-mock runs.
- Missing model files on the current Windows workspace are expected and are not themselves a bug.
- Postprocess behavior may differ per predictor implementation if needed; it does not have to be globally unified, but the chosen behavior must be explicit and stable.

### Remote line

- Upload encoding should be configurable, at least `png/jpeg`.
- Camera parameters such as resolution and output format should belong to mode/profile data so `GRASP_REMOTE` can define its own capture contract.
- `GRASP_REMOTE` predict should only be triggered after the new mode has produced fresh required frames.
- The segmentation branch can begin moving toward deletion, but only after parity against the minimal working remote script is confirmed.
- `class_id` may be stored in stage state, but its source of truth must be external input.
- Server-side `/init` completion must be confirmed before `/predict` is allowed.
- Remote server `/init` may be moved to service startup if that does not consume local runtime resources.

## Detect Line Audit

### D1. Real predictor output contract is not safely consumed

Observed behavior:

- `vision_module/backend/predictor/QNN_YOLO_Dectec_Predictor.py` returns `np.ndarray` detections.
- `vision_module/backend/predictor_manager.py` currently publishes `infer_boxes` and `infer_masks` using `boxes or []` and `masks or []` patterns.

Risk:

- For non-empty NumPy arrays, truth-value checks are ambiguous.
- The real predictor worker can fail exactly when detections appear.
- This would starve `SEARCH` and `GRASP` of valid `local_perception` updates.

Required direction:

- Define the real predictor output contract explicitly.
- Ensure `PredictorManager` handles real outputs without ambiguous truthiness checks.
- Confirm the published `local_perception` payload is stable for both `mock` and `real` backends.

Questions for coding follow-up:

- Should predictor outputs be normalized into pure Python lists before leaving the predictor layer?
- Or should downstream code formally accept NumPy arrays and serialize them at the manager boundary?

### D2. Default `coco80` detect does not align with stage-side class decoding

Observed behavior:

- `board_config.py` sets default active model to `yolov7_detect` with `classes=coco80`.
- `utils/detect.py` resolves class names through `TARGET_CLASSES`.
- `config/data.py` currently binds `TARGET_CLASSES` to `grasping_coco20`.

Risk:

- Default detect class IDs are interpreted against the wrong class table.
- Stage-side `target_obs` derivation can silently fail even when detect inference itself is correct.
- This weakens the core success path: `detect result -> target match -> target_obs -> vision_obs`.

Required direction:

- Remove the current effective dependency on `grasping_coco20` for the default detect path.
- Make class-name decoding depend on the active predictor/model contract, not one global hardcoded target table.
- Keep `coco80` viable as the baseline.

Questions for coding follow-up:

- Should class tables live on the model profile and flow downstream with `local_perception`?
- Or should stage-side target matching query a predictor/model registry instead of global constants?

### D3. `tmp` detect and backend detect are related but not logically identical

Observed behavior:

- The backend detect implementation is clearly adapted from the `tmp` benchmark path.
- Preprocess and YOLO head decode remain close.
- Postprocess and output shape differ materially.

Current differences to verify carefully:

- confidence scoring
- NMS behavior
- output structure
- whether class-aware behavior matches the original benchmark intent
- whether original benchmark assumptions were BGR or RGB

Required direction:

- Perform a parity audit between:
  - `tmp/run_test.py`
  - `tmp/utils.py`
  - `tmp/yolov7_head.py`
  - `vision_module/backend/predictor/QNN_YOLO_Dectec_Predictor.py`
  - `vision_module/backend/predictor/detect_utils.py`
  - `vision_module/backend/predictor/yolov7_head.py`
- Decide whether to:
  - restore original postprocess semantics inside the detect predictor, or
  - keep the new semantics but treat them as an explicit predictor-local contract

Architectural rule:

- Predictor-specific postprocess is acceptable.
- What is not acceptable is undocumented drift that breaks downstream assumptions.

### D4. Silent fallback to `mock` hides real-path bugs

Observed behavior:

- `vision_module/backend/predictor/__init__.py` attempts real import and falls back to `mock` on failure in auto mode.

Risk:

- On-device real-path failures can be masked.
- Integration problems may look like a healthy pipeline when the system has actually downgraded itself.

Required direction:

- Non-mock runs should surface real backend failure clearly.
- Fallback behavior, if retained at all, must be explicit and operator-visible.

Question for coding follow-up:

- Should `auto` still exist, but emit a hard warning and structured failure record when it downgrades?
- Or should `real` and `mock` be the only operationally safe modes?

### D5. `RETURN` is not yet backed by the default detect line

Observed behavior:

- `RETURN` consumes `home_tag_obs`.
- Default detect line currently emits generic `infer_boxes` and does not appear to produce a real `home_tag_obs` adapter path.

Required direction:

- Make the intended relationship explicit:
  - either `RETURN` is still mock/adapter-based for now, or
  - add a real conversion path from default local perception to `home_tag_obs`

This is lower priority than `SEARCH` / `GRASP` detect correctness.

## Remote Line Audit

### R1. Integrated remote flow is not yet logically equivalent to the minimal working script

Reference success path:

- `VISTA/grasp_module/simulate_client_request.py`

Expected logic:

1. `/init`
2. wait until init is actually complete
3. `/predict`
4. `/release`

Observed framework behavior:

- `GRASP_REMOTE` currently emits `INIT` and `PREDICT` back-to-back from stage logic.
- The remote worker processes commands asynchronously.

Risk:

- `PREDICT` can run without proven init completion.
- The framework may appear layered but actually weakens the original success path.

Required direction:

- Compare the integrated remote path against the minimal script step by step.
- Do not shrink remote profile surface until this parity audit is complete.
- Determine whether the integrated branch is actually functionally complete enough for subtractive cleanup.

Key question:

- After `/init` is sent, is there any confirmed gate today that prevents `/predict` until server init has actually succeeded?

Current audit answer:

- It does not appear so.

### R2. `GRASP_REMOTE` predict is triggered too early after mode switch

Observed behavior:

- Mode switch clears scheduler state.
- Stage logic then issues remote commands immediately.
- Remote request building reads `camera_frames` from the new generation.

Risk:

- `missing_camera_frames`
- `missing_depth_frame`
- race-dependent false failures

Required direction:

- Add a stage-visible readiness condition.
- `PREDICT` must be triggered only after fresh required frames are available in the new mode.
- This should be stage-driven, not hidden as a fragile timing assumption inside the worker.

Suggested control principle:

- `mode applied` is not the same as `data ready`.

### R3. Camera parameters belong in profiles/modes, not implicit globals

Observed behavior:

- `GRASP_REMOTE` needs its own capture contract.
- Current mode profile enables cameras but does not fully own per-mode capture parameters.

Risk:

- Remote capture may silently reuse local tracking settings.
- Resolution and output format become accidental global defaults rather than explicit mode requirements.

Required direction:

- Move remote-relevant camera parameters into mode/profile data.
- Allow `GRASP_REMOTE` to specify at least:
  - resolution
  - crop/output geometry
  - output format
  - any remote-specific capture profile needed for data quality

### R4. Upload encoding should be configurable, not hardcoded

Observed behavior:

- Remote manager currently encodes uploads in fixed formats.
- Real network bandwidth and server tolerance are still uncertain.

Required direction:

- Preserve both `png` and `jpeg` options for now.
- Put encoding choice under explicit config/profile control.
- Do not prematurely collapse to one format before measurement.

Question for coding follow-up:

- Should encoding live in remote profile, camera profile, or request-stage payload?

Recommended answer:

- Prefer mode/profile ownership for default operational behavior.
- Reserve request payload overrides for testing only if truly necessary.

### R5. Segmentation branch is now a deletion candidate

Observed behavior:

- Project direction is now `class_id`-driven remote requests.
- Segmentation-related fields still exist in the integrated contract surface.

Required direction:

- Start with a parity check against the minimal working script.
- If no required path still depends on segmentation, begin removing:
  - `seg_file`
  - `seg_bytes`
  - `require_segmentation`
  - any dead branching related to segmentation submission

Architectural rule:

- Delete only after parity confidence is established.
- Do not keep dead branches “just in case”.

### R6. Remote profile attributes need re-evaluation after parity audit

Observed behavior:

- Some remote attributes are defined in profiles.
- Some remote request semantics are owned by stage logic.
- The current ownership boundary is unclear.

Required direction:

- After the parity audit, list every current remote field and classify it as:
  - required runtime capability field
  - required request field
  - test-only override
  - dead field

Candidates to review carefully:

- `base_url`
- `command`
- `require_depth`
- `require_segmentation`
- `timeout_s`
- metadata extras
- upload encoding
- remote init strategy

Goal:

- keep only the minimum fields needed to preserve structural integrity

### R7. `class_id` source of truth must be external input only

Requirement:

- `class_id` may exist in `StagePlan` state as stored request context.
- It must not be synthesized from target-name heuristics during the real remote path.

Current concern:

- There are code paths that can infer `class_id` from target metadata.

Required direction:

- Trace all `class_id` write and fallback paths.
- Verify whether the current upstream protocol already supports explicit external `class_id` delivery.
- Remove or block implicit inference if the protocol can already provide the needed field.

Question for coding follow-up:

- Is `VisionReq.payload.class_id` already sufficient as the external contract, or is a top-level protocol field needed?

## Markdown Sync Status

The high-priority root doc refresh has already been completed.

Completed baseline refresh:

1. `VISTA/ReadMe.md`
2. `VISTA/VISION_ENGINE_TODO.md`
3. `VISTA/vision_module/backend/camera/README.md`
4. `VISTA/INTERFACES.md`
5. `VISTA/ARCHITECTURE.md`
6. `VISTA/PRODUCT_REQUIREMENTS.md`

Going forward, markdown work is no longer a standalone cleanup phase. It is part of contract maintenance.

### Doc-sync triggers for future coding work

- if the detect output contract changes, update `ARCHITECTURE.md`, `PRODUCT_REQUIREMENTS.md`, and any externally visible contract docs affected by the change
- if detect class-vocabulary ownership changes, update `ReadMe.md`, `ARCHITECTURE.md`, `INTERFACES.md` if request semantics change, and `PRODUCT_REQUIREMENTS.md`
- if remote `INIT -> PREDICT` sequencing changes, update `INTERFACES.md`, `ARCHITECTURE.md`, and `PRODUCT_REQUIREMENTS.md`
- if `class_id` ownership changes, update `INTERFACES.md`, `ARCHITECTURE.md`, and `PRODUCT_REQUIREMENTS.md`
- if remote camera parameters move into mode/profile ownership, update `ReadMe.md`, `ARCHITECTURE.md`, `PRODUCT_REQUIREMENTS.md`, and `vision_module/backend/camera/README.md`
- if upload encoding ownership changes, update `INTERFACES.md`, `ARCHITECTURE.md`, and `PRODUCT_REQUIREMENTS.md`
- if `release_cooldown_s` becomes real behavior or is removed, update `ReadMe.md`, `ARCHITECTURE.md`, and `PRODUCT_REQUIREMENTS.md`

## Recommended Execution Order

### Phase 1: stabilize Detect reality first

1. Fix the real detect output contract at the predictor-manager boundary.
2. Fix class-table ownership so default `coco80 detect` produces correct `target_obs`.
3. Decide whether to restore original tmp postprocess semantics or freeze the new detect-local semantics.
4. Remove hidden real-to-mock downgrade behavior for non-mock runs.

### Phase 2: prove remote parity before subtracting

1. Compare integrated remote flow against `simulate_client_request.py`.
2. Add confirmed init completion gating before predict.
3. Add stage-driven fresh-frame barrier before predict.
4. Move remote capture parameters into mode/profile data.
5. Add configurable upload encoding.
6. Audit `class_id` ownership and eliminate implicit generation if protocol already supports explicit input.

### Phase 3: reduce entropy

1. Remove segmentation-related remote dead branches.
2. Remove remote profile fields that are no longer justified.
3. Sync the affected VISTA markdown files in the same change series as the contract/code update.

## Deliverable Expectation For Coding Agent

Each coding step should report back with:

- the exact contract it changed
- whether the change preserves or intentionally alters current behavior
- what was deleted vs added
- what tests or static checks were used to validate it
- what `.md` files must be updated because of that change
