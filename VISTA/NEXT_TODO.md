# VISTA Next Todo

## Purpose

This is the only live short-horizon action file for VISTA.

Use `IMPLEMENTATION_STATUS.md` for the master plan and completion state.
Use this file for the current next-round work only, and update it as tasks are finished or re-scoped.

## Current Objective

Finish remote-line subtraction and cleanup now that:

- detect mainline has been stabilized
- service-scoped remote init has landed
- `RETURN` has been connected to the detect mainline

## Locked Baseline

Do not redesign these points unless real server behavior forces it:

- remote `/init` is service-scoped
- startup init failure must not block VISTA service startup
- `GRASP_REMOTE` may retry init at most 3 times before returning `remote_init_failed`
- `PREDICT` requires both service init confirmation and fresh required frames
- default detect / camera color baseline is `BGR`
- `RETURN` consumes the default detect line and still publishes `perception.home_tag_obs`
- remote `class_id` comes from explicit external input

## Active Tasks

### T1. Remove remote segmentation surface

Status: `DONE`

Required work:

- remove `RemoteProfile.require_segmentation`
- remove `seg_encoding`
- remove `RemotePredictRequest.seg_bytes`
- remove multipart `seg_file`
- remove dead segmentation branches in remote manager / protocol / mode compiler

Result:

- remote segmentation surface has been removed from profile, mode compile, protocol, manager, and the minimal remote script

Files likely affected:

- `vision_module/backend/mode_profiles.py`
- `vision_module/backend/mode_controller.py`
- `vision_module/backend/remote/manager.py`
- `vision_module/backend/remote/protocol.py`
- `vision_module/config/mode_defaults.py`

### T2. Shrink remote profile and request-field surface

Status: `DONE`

Required work:

- classify each remaining remote field as:
  - required runtime capability field
  - required request field
  - debug/test override
  - dead field
- remove dead fields after classification

Fields to review:

- `base_url`
- `command`
- `require_depth`
- `timeout_s`
- `metadata`
- `rgb_encoding`
- `depth_encoding`
- `rgb_quality`
- `depth_compression`

Result:

- runtime capability fields remain: `base_url`, `require_depth`, `rgb_encoding`, `depth_encoding`, `rgb_quality`, `depth_compression`
- request fields remain: `request_id`, `target`, `class_id`, `robot_id`, `need_depth`
- bounded debug/test overrides remain: `timeout_s`, `metadata`
- `command` is retained as a compatibility field but no longer expanded
- request-level `base_url` override has been removed

### T3. Sync the minimal remote script with the product direction

Status: `DONE`

Required work:

- update `grasp_module/simulate_client_request.py` so `class_id` is the normal path
- stop advertising `seg_file` as a normal production path
- keep the script aligned with the real integrated remote contract

Result:

- the minimal script now uses `class_id + rgb/depth + /init -> /predict -> /release` as the normal path
- `seg_file` and segmentation upload options are no longer advertised

### T4. Remove legacy detect/documentation residue

Status: `DONE`

Required work:

- remove doc wording that still treats BGR alignment as undecided if the contract is already frozen
- trim any `TARGET_CLASSES` historical mentions if they stop adding explanatory value
- keep root docs aligned with the service-scoped remote init lifecycle

Result:

- current-state docs now describe frozen `BGR`, service-scoped remote init, detect-backed `RETURN`, segmentation-surface removal, and the removal of request-level `base_url` override

### T5. Optional follow-up: validate `RETURN` product semantics

Status: `WATCH`

Question:

- is detect-backed `home_tag_obs` sufficient for the real return workflow, or will a dedicated tag detector path still be required later?

This is not a blocker for the current cleanup round.

## Update Rule

When a task here is completed:

1. update this file
2. update `IMPLEMENTATION_STATUS.md`
3. update any affected contract docs in the same change series
