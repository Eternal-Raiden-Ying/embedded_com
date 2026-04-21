# Environment Notes

## Scope

This document describes the local runtime environment required by `grasp_module`, with emphasis on the third-party components that are not installed from a clean upstream state.

The current runnable environment is not just a plain `pip install -r requirements.txt` setup. It depends on several locally compiled or locally adjusted components.

## Main Runtime

- Platform: Windows
- Python: project-local interpreter under `env/`
- Core runtime packages already installed into `env/` include:
  - `torch`
  - `MinkowskiEngine`
  - `open3d`
  - `pyrealsense2`
  - `ultralytics`
  - `pointnet2`
  - `graspnetAPI`

Recommended interpreter:

```powershell
E:\Documents_E\vscode\embedded_com\env\python.exe
```

## Third-Party Dependencies

### 1. `graspnetAPI`

- Local source directory: `graspnetAPI/`
- Purpose:
  - provides `GraspGroup`
  - used by the grasp post-processing / output path in `grasp_module/backend/engine.py`
- Current expectation:
  - installed into the local environment
  - not treated as part of the main business code repository

### 2. `MinkowskiEngineCuda13`

- Local source directory: `MinkowskiEngineCuda13/`
- Purpose:
  - provides `MinkowskiEngine`
  - required by the sparse convolution backbone and preprocessing path
- Important note:
  - this copy was modified so that `MinkowskiEngine` can be compiled on Windows
  - it should be treated as a locally maintained third-party dependency, not assumed to be identical to upstream

This is the most important non-upstream dependency in the environment. If another machine needs to reproduce the same build, it must use either:

- the same modified source tree, or
- a patch set / fork that contains the Windows build changes

### 3. `anygrasp_sdk/pointnet2`

- Source location: `anygrasp_sdk/pointnet2/`
- Purpose:
  - provides the `pointnet2` extension used by the grasp model
  - referenced by:
    - `grasp_module/backend/models/graspnet.py`
    - `grasp_module/backend/models/modules.py`
- Current expectation:
  - compiled and installed into `env/`

The main project uses the installed `pointnet2` package, not the source directory directly at runtime.

### 4. `graspness_unofficial` KNN

- Reference source directory: `graspness_unofficial/`
- Runtime KNN extension path in this project:
  - `grasp_module/backend/models/knn/`
- Purpose:
  - supports KNN-related operations used in label / geometry processing
- Important note:
  - the KNN implementation used here was adjusted to fix some bugs for the current environment

In practice, this means the KNN code should also be treated as a locally maintained dependency variant rather than assumed upstream-clean.

## Suggested Build / Install Order

For rebuilding the environment on another Windows machine, a reasonable order is:

1. Create and activate the local Python environment
2. Install PyTorch and matching CUDA runtime first
3. Install `graspnetAPI`
4. Build and install the modified `MinkowskiEngineCuda13`
5. Build and install `anygrasp_sdk/pointnet2`
6. Build and install the adjusted KNN extension under `grasp_module/backend/models/knn/`
7. Install the remaining Python runtime packages such as `open3d`, `pyrealsense2`, `ultralytics`
8. Verify imports using the project interpreter before running the server

## What Is Local-Only

The following should normally remain local and not be pushed as part of the main project repository:

- `env/`
- `output_dataset/`
- model weights under `grasp_module/weights/`
- service logs under `grasp_module/log/`
- generated debug outputs under `grasp_module/test/debug_res/`
- cloned third-party source mirrors:
  - `graspnetAPI/`
  - `MinkowskiEngineCuda13/`
  - `anygrasp_sdk/`
  - `graspness_unofficial/`
  - `yolo-source/`

## Remote Push Strategy

If the goal is to push the business project to a remote repository, the recommended baseline is:

- keep only the main project code tracked
- keep local environment, datasets, weights, logs, and third-party clone directories ignored
- track documentation that explains how the environment is assembled

That is the reason the root `.gitignore` is configured to ignore the large local-only directories above.

## How To Preserve Modified Third-Party Dependencies

Ignoring the third-party directories is good for a clean main repository, but it also means your local modifications inside those directories will not be preserved remotely.

For the modified dependencies in this project, especially:

- `MinkowskiEngineCuda13`
- `pointnet2`
- KNN code derived from `graspness_unofficial`

the better options are:

### Option A: Maintain forks

Use your own remote forks for the modified third-party repositories, then reference them in the environment documentation.

Best when:

- you want the full modified source history
- the dependency will continue evolving

### Option B: Keep the main repo clean and track patches only

Keep the third-party source trees ignored, but add tracked patch files and notes in the main repository, for example under a directory such as:

```text
third_party_patches/
```

Suggested contents:

- `MinkowskiEngineCuda13_windows.patch`
- `pointnet2_windows.patch`
- `knn_bugfix.patch`
- `README.md` describing patch order and target upstream commits

Best when:

- the main repo should stay small
- you only need reproducibility, not full vendor history

### Option C: Use submodules

If you want the remote project to reconstruct the environment more directly, convert the modified third-party dependencies into submodules pointing to your own forks.

Best when:

- team members need to clone everything consistently
- you want dependency revisions pinned explicitly

## Practical Recommendation For This Project

For this repository, the most pragmatic setup is:

1. Keep the root repo focused on `grasp_module` and project docs
2. Keep the cloned third-party directories ignored
3. Add environment documentation like this file
4. For modified third-party code, choose one of:
   - maintain forks, or
   - track patch files separately

If you do nothing except ignore the third-party directories, the project may still run locally, but another machine will not be able to reproduce the exact Windows build state.

## `.gitignore` Recommendation

For remote pushes of the main project, it is reasonable to ignore:

- local Python environment
- datasets
- weights
- logs
- debug artifacts
- cloned third-party source trees
- local extension build outputs

One important caveat:

If any compiled artifacts were already committed in the past, adding ignore rules is not enough by itself. Those files must also be removed from git tracking history or index separately.

Typical examples in this project include:

- `grasp_module/backend/models/knn/build/`
- `grasp_module/backend/models/knn/*.egg-info/`

If needed later, these can be removed from tracking with `git rm --cached`, while still keeping the files locally.
