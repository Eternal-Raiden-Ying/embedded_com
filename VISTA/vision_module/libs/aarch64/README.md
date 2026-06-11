# VISTA aarch64 Native Libraries

This directory stores board-only native binaries used by VISTA.

## fast_cam

- File: `fast_cam.cpython-38-aarch64-linux-gnu.so`
- Source: built from the camera extension sources under `VISTA/vision_module/backend/camera/cxx`
- Python ABI: CPython 3.8
- Platform: aarch64 Linux target board
- Host behavior: Windows hosts cannot import this module
- Test behavior: tests requiring `fast_cam` must skip unless running on aarch64 and the module is importable

Keep native target binaries here instead of mixing them into the Python camera source directory.
