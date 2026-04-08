# VISTA Vision Engine Todo

## Goal

Refactor VISTA into a two-layer orchestration system:

- app / protocol layer manages `vision_req` / `vision_obs`, task session, stage transitions, and interaction rounds
- `vision_engine.py` manages internal mode switching and resource lifecycle

The key rule is:

- external protocol is session- and interaction-oriented
- internal engine is mode-driven
- `stage` and `mode` are different concepts
- current design assumes there is usually only one active mode at a time

## Step 0. Unify external protocol

Before engine refactor, freeze the new protocol shape.

### `vision_req`

Keep the original message name, but upgrade the payload to support:

- `op`: `START` / `UPDATE` / `RESPOND` / `STOP`
- `stage`: `SEARCH` / `GRASP` / `RETURN` / `IDLE`
- `target`
- `mode_hint` (optional)
- `interaction_id` (optional)
- `response` (optional)
- `payload` (optional)

This is required because VISTA must support not only one-shot search requests, but also multi-round grasp workflows such as:

1. enter `GRASP`
2. run a mode such as `MICRO_ADJUST`
3. output motion / distance / yaw adjustment suggestion
4. wait for outside confirmation or execution result
5. switch back to `GRASP_REMOTE`
6. run another remote grasp prediction

### `vision_obs`

Keep the original message name and make it the only outbound envelope.

The minimum fields should include:

- `stage`
- `mode`
- `status`
- `interaction`
- `perception`
- `proposal`
- `result`

### Legacy handling

- Legacy `FIND` / `RETURN` / `IDLE` request parsing may remain temporarily as an adapter
- Legacy outbound `target_obs` / `home_tag_obs` should not constrain the new protocol design
- Test tools should move to the new protocol first

## Step 1. Define stage plan and mode profiles

Define two separate concepts:

### `StagePlan`

Represents business workflow, for example:

- `SEARCH`
- `GRASP`
- `RETURN`

Responsibilities:

- decide which mode should run now
- react to `vision_req` updates and responses
- emit `vision_obs`
- decide when to request outside confirmation
- decide when to switch to another mode

### `ModeProfile`

Represents only resource requirements.

The minimum fields should include:

- `name`
- `enabled_cameras`
- `camera_profiles`
- `predictor_enabled`
- `predictor_model`
- `network_enabled`
- `preview_enabled`
- `loop_hz`
- `send_hz`
- `release_cooldown_s`

The new field `release_cooldown_s` is important:

- do not release an old mode's expensive resources immediately
- allow delayed release to avoid repeated camera / model load costs during rapid switching

Initial concrete modes:

- `TRACK_LOCAL`
- `GRASP_REMOTE`
- `MICRO_ADJUST`
- `DEPTH_PERCEPTION`

## Step 2. Refactor `vision_engine.py` into managers

Keep `vision_engine.py` as the main public entry, but split responsibilities internally.

Introduce:

- `ModeController`
- `CameraManager`
- `PredictorManager`
- optional delayed-release helper / resource cooldown manager

### `CameraManager`

- owns camera lifecycle
- opens `ColorCamera`, `IRCamera`, `RealSenseDepthCamera`
- supports diff-based camera updates
- avoids rebuilding unchanged cameras
- supports delayed release / cooldown policy

### `PredictorManager`

- owns predictor lifecycle
- loads and unloads `QNN_YOLO_Segment_Predictor`
- avoids reloading the same model
- supports delayed release / cooldown policy for expensive model teardown

### `ModeController`

- stores current mode
- applies `ModeProfile`
- switches only changed resources
- manages mode enter / exit timestamps
- handles mode cooldown windows

### `VisionEngine`

- remains the public facade
- delegates resource changes to managers
- keeps frame queues and latest data cache
- does not own business workflow logic

## Step 3. Introduce stage controller in app layer

Refactor `app.py` so that it no longer directly toggles cameras and models.

Introduce a stage-oriented control layer, for example:

- `StageController`
- per-stage handlers or plans

Responsibilities:

- parse `vision_req`
- maintain session / req / interaction context
- drive `VisionEngine.set_mode(...)`
- generate `vision_obs`
- process `RESPOND` messages

## Step 4. Implement `SEARCH`

Purpose:

- local target search / track / navigation support

Initial mode path:

- default mode: `TRACK_LOCAL`
- later may extend with `DEPTH_PERCEPTION`

Expected behavior:

- low latency RGB path
- local predictor enabled
- stable target observation output

Expected outbound payload:

- `vision_obs.stage = SEARCH`
- `vision_obs.mode = TRACK_LOCAL`
- `perception.target_obs`

## Step 5. Implement `GRASP`

Purpose:

- remote grasp prediction and visual micro-adjustment loop

Initial mode path:

- `GRASP_REMOTE`
- `MICRO_ADJUST`
- back to `GRASP_REMOTE` if another remote prediction is needed

This step must support multi-round interaction:

1. capture RGB + depth
2. run remote grasp or local adjustment analysis
3. emit `proposal` or `result`
4. optionally wait for external response
5. continue with next mode

Expected outbound payload examples:

- motion delta suggestion
- distance / yaw adjustment suggestion
- grasp pose result
- retry / failure signal

## Step 6. Implement `RETURN`

Purpose:

- detect and track home tag / return target

Initial mode path:

- start with `TRACK_LOCAL`
- later extend if depth or other sensing is required

Expected outbound payload:

- `vision_obs.stage = RETURN`
- `vision_obs.mode = TRACK_LOCAL`
- `perception.home_tag_obs`

## Step 7. Preview as a side channel

- do not bake preview into stage or mode core logic
- preview subscribes to latest cache only
- preview disabled by default
- preview must not change mode switching behavior

## Step 8. Validation and tools

### Protocol tools

Upgrade test tools first:

- `test/debug_send_req.py`
- `test/debug_recv_obj.py`

They should support:

- new `vision_req`
- new `vision_obs`
- manual multi-round `GRASP` interaction simulation

### Mock validation

- `SEARCH` with `TRACK_LOCAL`
- `RETURN` with `TRACK_LOCAL`
- `GRASP` with `GRASP_REMOTE <-> MICRO_ADJUST`
- mode switching with cooldown behavior

### Real validation

- local RGB + predictor path
- RGB + depth path
- remote grasp request path

Verify:

- cameras are not leaked on switch
- predictor is not reloaded unnecessarily
- stop / release remains idempotent
- cooldown logic reduces thrash during fast mode changes
- outbound payload shape is stable per stage / mode

## Notes for future extension

- future business stages should reuse the same protocol shape instead of adding ad-hoc message types
- future modes should reuse the same mode profile structure instead of adding more booleans
- likely future modes:
  - `PRE_GRASP_VERIFY`
  - `POST_GRASP_CHECK`
  - `NAV_ASSIST`
- `VisionEngine` should manage resources
- stage controller should manage workflow
