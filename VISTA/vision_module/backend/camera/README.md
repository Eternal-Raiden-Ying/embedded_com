# VISTA Camera Backend

This directory contains the camera backend used by VISTA runtime managers.

It is not a standalone demo package. It is part of the current VISTA stage/mode architecture.

## Current Role

The camera backend provides the low-level camera implementations selected by `CameraManager`.

Current exported runtime classes:

- `ColorCamera`
- `IRCamera`
- `HardwareCamera`
- `RealSenseDepthCamera`

The import selector lives in `__init__.py` and currently supports:

- `VISTA_BACKEND=mock`
- `VISTA_BACKEND=real`
- `VISTA_BACKEND=auto`

## Current Directory Contents

- `ColorCamera.py`: color camera implementation
- `IRCamera.py`: IR camera implementation
- `HardwareCamera.py`: hardware-accelerated camera path
- `RealSenseDepthCamera.py`: depth camera implementation
- `base.py`: camera abstraction base
- `mock.py`: mock backend
- `_fast_gst_camera.py`: Python bridge for the fast camera path
- `cxx/`: native source/build directory for camera extension work
- `camera_info.md`: raw capability notes for the current hardware

The prebuilt `fast_cam.cpython-38-aarch64-linux-gnu.so` target-device binary is
stored outside this Python source directory at `VISTA/vision_module/libs/aarch64/`.

## Build Path

If the native camera extension needs to be rebuilt on the target device, use the current repo path:

```bash
cd VISTA/vision_module/backend/camera/cxx
mkdir -p build
cd build
cmake ..
make
```

Copy the generated `fast_cam.cpython-38-aarch64-linux-gnu.so` into
`VISTA/vision_module/libs/aarch64/`. The old `aidlux_cam/csrc` path is obsolete
for this repository.

## Current Architectural Notes

- Camera lifecycle is owned by `vision_module/backend/camera_manager.py`, not by the app layer.
- Camera instances are selected and reconfigured according to runtime mode plans.
- Current board defaults still live in `vision_module/config/board_config.py`.
- `GRASP_REMOTE` now consumes explicit `ModeProfile.camera_overrides` instead of implicitly reusing local tracking defaults.
- Board config still provides the source defaults, but runtime ownership now belongs to mode/profile data.
- The default color camera baseline is now `BGR`, and mode profiles own the final per-mode RGB-camera format / crop / fps contract.

## Current Limitations

- The real runtime target is AidLux / QCS6490, not Windows.
- The `fast_cam` native extension is a Python 3.8 aarch64 Linux binary. Windows
  host environments cannot import it; host-side tests that require it must skip
  unless both `platform.machine() == "aarch64"` and `fast_cam` is importable.
- On Windows or unsupported environments, the backend may resolve to `mock` depending on runtime settings.
- `camera_info.md` is a hardware note, not the authoritative architecture contract.

## Related Docs

- `VISTA/ReadMe.md`
- `VISTA/ARCHITECTURE.md`
- `VISTA/PRODUCT_REQUIREMENTS.md`
