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
- `fast_cam.cpython-38-aarch64-linux-gnu.so`: prebuilt target-device binary
- `cxx/`: native source/build directory for camera extension work
- `camera_info.md`: raw capability notes for the current hardware

## Build Path

If the native camera extension needs to be rebuilt on the target device, use the current repo path:

```bash
cd VISTA/vision_module/backend/camera/cxx
mkdir -p build
cd build
cmake ..
make
```

The old `aidlux_cam/csrc` path is obsolete for this repository.

## Current Architectural Notes

- Camera lifecycle is owned by `vision_module/backend/camera_manager.py`, not by the app layer.
- Camera instances are selected and reconfigured according to runtime mode plans.
- Current board defaults still live in `vision_module/config/board_config.py`.
- Future cleanup should move more camera behavior into explicit mode/profile ownership for paths such as `GRASP_REMOTE`.

## Current Limitations

- The real runtime target is AidLux / QCS6490, not Windows.
- On Windows or unsupported environments, the backend may resolve to `mock` depending on runtime settings.
- `camera_info.md` is a hardware note, not the authoritative architecture contract.

## Related Docs

- `VISTA/ReadMe.md`
- `VISTA/ARCHITECTURE.md`
- `VISTA/PRODUCT_REQUIREMENTS.md`
