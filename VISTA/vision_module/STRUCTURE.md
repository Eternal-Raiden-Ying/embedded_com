# vision_module structure

This pass keeps the current runtime layout intact and only adds clearer homes
for diagnostics, ROI helpers, and future manual tools.

## Current Responsibilities

- `app/`: service entrypoint, `VistaApp`, `StageController`, and stage plans.
- `backend/`: runtime managers plus hardware and algorithm adapters.
- `backend/camera/`: camera adapters for RGB, depth, mock, RealSense, and board hardware cameras.
- `backend/predictor/`: QNN and mock predictors plus YOLO post-processing.
- `backend/preview/`: OpenCV and null preview sinks.
- `backend/remote/`: remote grasp interface.
- `backend/edge_detect/`: table edge detector capability driver (`OnlineTableEdgeDetector`, board config, calibration).
- `config/`: `board_config`, schema definitions, and mode defaults.
- `ipc/`: `vision_req` / `vision_obs` protocol objects and JSONL transport.
- `diagnostics/`: operator console rate limiting, field summaries, and future debug dump helpers.
- `model/`: QNN models and example model resources.
- `test/`: automated tests plus historical manual debug scripts.
- `tools/`: future home for manual debugging scripts and operator-run helpers.
- `utils/`: shared utility functions.

The current debug scripts in `test/`, including `debug_send_req.py`,
`debug_recv_obj.py`, `debug_protocol_tools.py`, and `demo_camera.py`, are
temporarily retained there for compatibility. Move them to `tools/` later with
old-path shims.

## Compatibility Rules

- Do not move core runtime files in one broad pass.
- Keep old import paths working when a file is eventually moved.
- Do not change `vision_req` / `vision_obs` protocol fields during structure cleanup.
- Do not change stage, mode, or scheduler behavior as part of directory cleanup.
- `QNN_YOLO_Detect_Predictor.py` naming has been corrected (was `QNN_YOLO_Dectec_Predictor.py`).

## Long-Term Direction

The following shape is recommended, but it is not implemented in this cleanup:

- `runtime/`: `vision_engine`, `scheduler`, `mode_controller`, and `runtime_supervisor`.
- `perception/`: predictor, table-edge, and remote perception modules.
- `io/`: camera, IPC, and preview input/output modules.
- `diagnostics/`: operator console, summaries, trace dumps, and debug snapshots.
- `tools/`: manual debugging scripts.
- `tests/`: automated tests and regression tests.

Native camera extensions such as `backend/camera/fast_cam.cpython-38-aarch64-linux-gnu.so`
are board runtime artifacts. Keep them in place for now; a later cleanup can
move native camera code under `backend/camera/native/` once build and packaging
instructions are explicit.
